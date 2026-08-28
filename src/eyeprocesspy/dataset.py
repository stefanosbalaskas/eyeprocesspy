from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from itertools import count
import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError, EyeProcessSchemaError
from .schema import SCHEMA_VERSION, canonical_table_names, empty_eye_table, standardize_eye_table, validate_eye_table

_id_counter=count(1)

def _next_id(prefix: str) -> str:
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S.%f")
    return f"{prefix}_{stamp}_{next(_id_counter):09d}"

def _now_utc() -> str:
    return datetime.now(timezone.utc).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def _file_md5(path) -> str | None:
    p=Path(path)
    if not p.is_file(): return None
    h=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _missing_scalar(v) -> bool:
    if pd.isna(v): return True
    return not bool(str(v).strip())

class EyeDataset(dict):
    """Dict-like Python counterpart of the R `eye_dataset` list class."""
    def __init__(self, *args, raw=None, vendor_metadata=None, schema_version=SCHEMA_VERSION, validation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw=[] if raw is None else raw
        self.vendor_metadata={} if vendor_metadata is None else vendor_metadata
        self.schema_version=schema_version
        self.validation=validation
        self.empty_components=[]
    def copy(self):
        return copy.deepcopy(self)
    def __repr__(self):
        rec=len(self.get('recordings',[])); streams=len(self.get('streams',[])); gaze=len(self.get('gaze_samples',[]))
        return f"<eye_dataset schema {self.schema_version}; recordings={rec}, streams={streams}, gaze_samples={gaze}>"

def new_eye_dataset(recordings=None, streams=None, gaze_samples=None, eye_samples=None, episodes=None, events=None, intervals=None, responses=None, coordinate_spaces=None, aoi_definitions=None, aoi_geometry=None, biometrics=None, calibrations=None, features=None, quality=None, provenance=None, raw=None, vendor_metadata=None, schema_version=SCHEMA_VERSION, validate=True) -> EyeDataset:
    """Construct a canonical dataset preserving R table order and missing-field semantics."""
    vals=locals().copy(); names=canonical_table_names(); tables={}
    for n in names:
        v=vals[n]
        if v is None: v=empty_eye_table(n)
        tables[n]=standardize_eye_table(v,n,keep_extra=True)
    out=EyeDataset(tables,raw=[] if raw is None else raw,vendor_metadata={} if vendor_metadata is None else vendor_metadata,schema_version=schema_version)
    if validate: out.validation=validate_eye_dataset(out,strict=False)
    return out

def is_eye_dataset(x) -> bool:
    return isinstance(x,EyeDataset)

def _assert_eye_dataset(x):
    if not is_eye_dataset(x): raise TypeError("`x` must be an EyeDataset.")

def validate_eye_dataset(x, strict=False, stop_on_error=False) -> pd.DataFrame:
    _assert_eye_dataset(x)
    issue_frames=[validate_eye_table(x[n],n,strict=strict) for n in canonical_table_names()]
    issues=pd.concat(issue_frames,ignore_index=True) if issue_frames else pd.DataFrame(columns=["severity","code","table","field","message"])
    rows=[]
    primary={"recordings":["recording_id"],"streams":["stream_id"],"gaze_samples":["sample_id"],"eye_samples":["recording_id","sample_id","eye"],"episodes":["episode_id"],"events":["event_id"],"intervals":["interval_id"],"responses":["response_id"],"coordinate_spaces":["coordinate_space_id"],"aoi_definitions":["aoi_id"],"calibrations":["calibration_id"],"features":["feature_id"],"quality":["quality_id"],"provenance":["provenance_id"]}
    for n,keys in primary.items():
        d=x[n]
        if d.empty or not all(k in d.columns for k in keys): continue
        missing=d[keys].apply(lambda col: col.map(_missing_scalar)).any(axis=1)
        if bool(missing.any()): rows.append(dict(severity="error",code="missing_primary_key",table=n,field="+".join(keys),message=f"Missing values found in primary key `{' + '.join(keys)}`."))
        if bool((~missing).any()) and d.loc[~missing,keys].astype(str).duplicated().any(): rows.append(dict(severity="error",code="duplicate_primary_key",table=n,field="+".join(keys),message=f"Duplicate values found for primary key `{' + '.join(keys)}`."))
    recording_ids=set(x['recordings']['recording_id'].dropna().astype(str))
    for n in ["streams","gaze_samples","eye_samples","episodes","events","intervals","responses","biometrics","calibrations","features","quality"]:
        d=x[n]
        if not d.empty and 'recording_id' in d and recording_ids:
            orphan=sorted(set(d['recording_id'].dropna().astype(str))-recording_ids)
            if orphan: rows.append(dict(severity="error",code="orphan_recording_id",table=n,field="recording_id",message=f"Unknown recording id(s): {', '.join(orphan)}."))
    stream_ids=set(x['streams']['stream_id'].dropna().astype(str))
    for n in ["gaze_samples","biometrics"]:
        d=x[n]
        if not d.empty and 'stream_id' in d:
            orphan=sorted(set(d['stream_id'].dropna().astype(str))-stream_ids)
            if orphan: rows.append(dict(severity="error",code="orphan_stream_id",table=n,field="stream_id",message=f"Unknown stream id(s): {', '.join(orphan)}."))
    coord_ids=set(x['coordinate_spaces']['coordinate_space_id'].dropna().astype(str))
    for n in ["streams","gaze_samples","episodes","aoi_definitions","aoi_geometry"]:
        d=x[n]
        if not d.empty and 'coordinate_space_id' in d:
            orphan=sorted(set(d['coordinate_space_id'].dropna().astype(str))-coord_ids)
            if orphan: rows.append(dict(severity="error",code="orphan_coordinate_space",table=n,field="coordinate_space_id",message=f"Unknown coordinate space(s): {', '.join(orphan)}."))
    if not x['gaze_samples'].empty:
        t=pd.to_numeric(x['gaze_samples']['timestamp_seconds'],errors='coerce')
        original=x['gaze_samples']['timestamp_seconds']
        bad=original.notna() & ~np.isfinite(t.to_numpy(dtype=float))
        if bool(bad.any()): rows.append(dict(severity="error",code="nonfinite_timestamp",table="gaze_samples",field="timestamp_seconds",message="Non-finite normalized timestamps detected."))
    if not x['intervals'].empty:
        s=pd.to_numeric(x['intervals']['start_time'],errors='coerce'); e=pd.to_numeric(x['intervals']['end_time'],errors='coerce')
        bad=np.isfinite(s)&np.isfinite(e)&(e<s)
        if bool(bad.any()): rows.append(dict(severity="error",code="negative_interval",table="intervals",field="end_time",message="At least one interval ends before it starts."))
    if rows: issues=pd.concat([issues,pd.DataFrame(rows)],ignore_index=True)
    issues=issues.loc[:,["severity","code","table","field","message"]].reset_index(drop=True)
    if stop_on_error and bool((issues['severity']=='error').any()):
        raise EyeProcessValidationError(f"`eye_dataset` validation failed with {int((issues['severity']=='error').sum())} error(s).")
    return issues

def get_eye_table(x, table):
    _assert_eye_dataset(x)
    if not isinstance(table,str): raise TypeError("`table` must be a string.")
    if table not in x: raise EyeProcessSchemaError(f"Unknown component `{table}`.")
    return x[table]

def set_eye_table(x, table, value, validate=True):
    _assert_eye_dataset(x)
    if table not in canonical_table_names(): raise EyeProcessSchemaError(f"Unknown canonical table `{table}`.")
    out=x.copy(); out[table]=standardize_eye_table(value,table,keep_extra=True)
    out=add_provenance(out,"set_table",table,f"Rows: {len(value)}")
    if validate: out.validation=validate_eye_dataset(out)
    return out

def append_eye_table(x, table, value, validate=True):
    old=get_eye_table(x,table)
    both=pd.concat([old,value],ignore_index=True,sort=False)
    return set_eye_table(x,table,both,validate=validate)

def add_provenance(x, action, component=pd.NA, details=pd.NA, source_files=pd.NA, file_hashes=None, software="eyeprocess", software_version=None, reversible=True, warnings=pd.NA):
    _assert_eye_dataset(x); out=x.copy()
    if isinstance(source_files,(list,tuple,set)): source_string='|'.join(map(str,source_files))
    elif pd.isna(source_files): source_string=pd.NA
    else: source_string=str(source_files)
    if file_hashes is None and not pd.isna(source_string) and source_string:
        file_hashes=[_file_md5(p) for p in source_string.split('|')]
    if isinstance(file_hashes,(list,tuple,set)): hash_string='|'.join('NA' if h is None else str(h) for h in file_hashes)
    elif file_hashes is None: hash_string=''
    else: hash_string=str(file_hashes)
    if isinstance(warnings,(list,tuple,set)): warn_string=' | '.join(map(str,warnings))
    elif pd.isna(warnings): warn_string='NA'
    else: warn_string=str(warnings)
    row=pd.DataFrame([dict(provenance_id=_next_id('prov'),timestamp=_now_utc(),action=str(action),component=component,details=details,source_files=source_string,file_hashes=hash_string,software=str(software),software_version=software_version or __import__('eyeprocesspy').__version__,reversible=bool(reversible),warnings=warn_string)])
    out['provenance']=standardize_eye_table(pd.concat([out['provenance'],row],ignore_index=True,sort=False),'provenance')
    return out

def provenance_manifest(x):
    _assert_eye_dataset(x)
    p=x['provenance']
    return {"schema_version":x.schema_version,"created":_now_utc(),"sources":list(pd.unique(p['source_files'].dropna())),"file_hashes":list(pd.unique(p['file_hashes'].dropna())),"actions":p.copy(),"coordinate_spaces":x['coordinate_spaces'].copy(),"streams":x['streams'].copy(),"validation":validate_eye_dataset(x)}

def compact_eye_dataset(x, drop_raw=False, drop_empty=False):
    _assert_eye_dataset(x); out=x.copy()
    if drop_raw: out.raw=[]
    if drop_empty: out.empty_components=[n for n in canonical_table_names() if out[n].empty]
    return add_provenance(out,"compact_dataset","dataset",f"drop_raw={str(bool(drop_raw)).upper()};drop_empty={str(bool(drop_empty)).upper()}",reversible=not drop_raw)
