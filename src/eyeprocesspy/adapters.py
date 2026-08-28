from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import warnings
from typing import Callable
import numpy as np
import pandas as pd

from .dataset import EyeDataset, add_provenance, is_eye_dataset, new_eye_dataset, validate_eye_dataset
from .schema import canonical_table_names
from .importers import _read_delimited, read_eye_generic

_ADAPTERS: dict[str, dict] = {}


def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def register_eye_adapter(name, detect, read, validate=None, priority=0, overwrite=False):
    """Register a vendor/custom reader using the frozen R adapter contract."""
    if not isinstance(name,str) or not name: raise TypeError("`name` must be a single character value.")
    if not callable(detect): raise TypeError("`detect` must be a function.")
    if not callable(read): raise TypeError("`read` must be a function.")
    if validate is not None and not callable(validate): raise TypeError("`validate` must be NULL/None or a function.")
    if not overwrite and name in _ADAPTERS: raise ValueError(f"Adapter `{name}` is already registered.")
    _ADAPTERS[name]={"name":name,"detect":detect,"read":read,"validate":validate,"priority":float(priority),"registered_at":_now_utc()}
    return name


def unregister_eye_adapter(name):
    if not isinstance(name,str) or not name: raise TypeError("`name` must be a single character value.")
    _ADAPTERS.pop(name,None)
    return name


def supported_eye_formats():
    if not _ADAPTERS: return pd.DataFrame()
    out=pd.DataFrame([{"name":a["name"],"priority":a["priority"],"has_validator":a["validate"] is not None,"registered_at":a["registered_at"]} for a in _ADAPTERS.values()])
    return out.sort_values(["priority","name"],ascending=[False,True],kind="stable").reset_index(drop=True)


def detect_eye_format(path, inspect_rows=20, candidates=None):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Path does not exist: {path}")
    adapters=_ADAPTERS if candidates is None else {k:v for k,v in _ADAPTERS.items() if k in set(candidates)}
    if not adapters: raise ValueError("No eye-data adapters are registered.")
    rows=[]
    for a in adapters.values():
        try:
            score=a["detect"](path,inspect_rows=inspect_rows)
            if isinstance(score,(bool,np.bool_)): score=1.0 if score else 0.0
            score=float(np.ravel(score)[0]); score=score if np.isfinite(score) else 0.0
        except Exception:
            score=0.0
        rows.append({"format":a["name"],"confidence":max(0.0,min(1.0,score)),"priority":a["priority"]})
    return pd.DataFrame(rows).sort_values(["confidence","priority","format"],ascending=[False,False,True],kind="stable").reset_index(drop=True)


def read_eye_export(path,vendor="auto",confidence_threshold=0.55,**kwargs):
    detection=None
    if vendor=="auto":
        detection=detect_eye_format(path)
        if detection.empty or float(detection.iloc[0].confidence)<confidence_threshold:
            raise ValueError(f"No adapter reached the required confidence threshold ({confidence_threshold}). Use `read_eye_generic()` with an explicit mapping.")
        if len(detection)>1 and detection.iloc[0].confidence==detection.iloc[1].confidence:
            warnings.warn(f"Format detection is tied; selecting `{detection.iloc[0].format}` by adapter priority.",RuntimeWarning,stacklevel=2)
        vendor=str(detection.iloc[0].format)
    if vendor not in _ADAPTERS: raise ValueError(f"Unknown adapter `{vendor}`.")
    out=_ADAPTERS[vendor]["read"](path,**kwargs)
    if isinstance(out,EyeDataset): out.format_detection=detection
    return out


def read_eye_folder(path,recursive=False,vendor="auto",pattern=None,combine=True,**kwargs):
    p=Path(path)
    if not p.is_dir(): raise FileNotFoundError(f"Directory does not exist: {path}")
    files=[q for q in (p.rglob('*') if recursive else p.glob('*')) if q.is_file() and (pattern is None or re.search(pattern,q.name))]
    if not files: raise ValueError(f"No files found in: {path}")
    if not combine: return [read_eye_export(f,vendor=vendor,**kwargs) for f in files]
    objs=[]
    for f in files:
        try: obj=read_eye_export(f,vendor=vendor,**kwargs)
        except Exception: continue
        if is_eye_dataset(obj): objs.append(obj)
    if not objs: raise ValueError("No files in the folder could be imported.")
    return combine_eye_datasets(objs)


def remap_recording_ids(x,mapping):
    if not is_eye_dataset(x): raise TypeError("Expected an `eye_dataset` object.")
    if not isinstance(mapping,dict): mapping=dict(mapping)
    out=x.copy()
    for name in canonical_table_names():
        d=out[name].copy()
        if "recording_id" in d.columns and not d.empty:
            d["recording_id"]=d["recording_id"].map(lambda z:mapping.get(z,z))
            out[name]=d
    return add_provenance(out,"remap_recording_ids","dataset",";".join(f"{a}->{b}" for a,b in mapping.items()))


def combine_eye_datasets(*xs,resolve_ids=True):
    if len(xs)==1 and isinstance(xs[0],(list,tuple)) and not is_eye_dataset(xs[0]): xs=tuple(xs[0])
    if not xs or not all(is_eye_dataset(x) for x in xs): raise TypeError("All inputs must be `eye_dataset` objects.")
    xs=[x.copy() for x in xs]
    if resolve_ids:
        seen=set()
        for i,x in enumerate(xs,1):
            ids=[str(v) for v in x["recordings"].recording_id.dropna().unique()]
            duplicates=[v for v in ids if v in seen]
            if duplicates: xs[i-1]=remap_recording_ids(x,{v:f"{v}_set{i}" for v in duplicates})
            seen.update(str(v) for v in xs[i-1]["recordings"].recording_id.dropna().unique())
    tables={}
    for name in canonical_table_names():
        frames=[x[name] for x in xs]
        tables[name]=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()
    identity={"recordings":"recording_id","streams":"stream_id","coordinate_spaces":"coordinate_space_id","aoi_definitions":"aoi_id","calibrations":"calibration_id"}
    for name,key in identity.items():
        if not tables[name].empty and key in tables[name].columns:
            tables[name]=tables[name].drop_duplicates(subset=[key],keep="first").reset_index(drop=True)
    raw=[]; metadata={}
    for x in xs:
        if isinstance(x.raw,dict): raw.extend(x.raw.items())
        elif isinstance(x.raw,list): raw.extend(x.raw)
        if isinstance(x.vendor_metadata,dict): metadata.update(x.vendor_metadata)
    out=new_eye_dataset(**tables,raw=raw,vendor_metadata=metadata,validate=False)
    out=add_provenance(out,"combine_datasets","dataset",f"Combined {len(xs)} datasets.")
    out.validation=validate_eye_dataset(out)
    return out


def _detect_generic_delimited(path,inspect_rows=20):
    p=Path(path)
    if p.is_dir() or p.suffix.lower() not in {".csv",".tsv",".txt",".asc"}: return 0.0
    try: d=_read_delimited(path,nrows=min(2,int(inspect_rows)))
    except Exception: return 0.0
    return 0.1 if d.shape[1]>=2 else 0.0


def _register_builtin_adapters():
    # Only adapters whose corresponding reader is actually implemented are registered.
    # This deliberately avoids advertising unimplemented vendor parity.
    register_eye_adapter("generic",_detect_generic_delimited,read_eye_generic,priority=1,overwrite=True)

_register_builtin_adapters()
