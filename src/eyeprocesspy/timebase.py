from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .dataset import EyeDataset, _assert_eye_dataset, add_provenance
from .exceptions import EyeProcessTimebaseError

_TIME_UNITS={"seconds":1,"second":1,"s":1,"milliseconds":1e-3,"millisecond":1e-3,"ms":1e-3,"microseconds":1e-6,"microsecond":1e-6,"us":1e-6,"nanoseconds":1e-9,"nanosecond":1e-9,"ns":1e-9,"ticks":1}

def estimate_sampling_rate(timestamp_seconds, trim=0.05):
    a=pd.to_numeric(pd.Series(timestamp_seconds),errors='coerce').to_numpy(dtype=float); t=np.unique(a[np.isfinite(a)])
    if t.size<2:return np.nan
    dt=np.diff(np.sort(t)); dt=dt[(dt>0)&np.isfinite(dt)]
    if not dt.size:return np.nan
    if dt.size>10 and trim>0:
        q=np.quantile(dt,[trim,1-trim]); dt=dt[(dt>=q[0])&(dt<=q[1])]
    return float(1/np.median(dt))

def normalize_timebase(x, component=("gaze_samples","eye_samples","events","biometrics"), native_unit=None, origin="recording_start", overwrite=True):
    _assert_eye_dataset(x); out=x.copy()
    if origin not in {"recording_start","absolute","first_observation"}: raise ValueError("Invalid origin.")
    components=[component] if isinstance(component,str) else list(component)
    for n in components:
        d=out[n].copy()
        if d.empty or 'timestamp_native' not in d: continue
        unit=native_unit
        if unit is None and not out['streams'].empty and 'timestamp_unit' in out['streams']:
            vals=out['streams']['timestamp_unit'].dropna(); unit=str(vals.iloc[0]) if len(vals) else 'seconds'
        unit=(unit or 'seconds').lower()
        if unit not in _TIME_UNITS: raise EyeProcessTimebaseError(f"Unsupported time unit `{unit}`.")
        sec=pd.to_numeric(d['timestamp_native'],errors='coerce')*_TIME_UNITS[unit]
        if origin in {"recording_start","first_observation"}:
            for _,idx in d.groupby('recording_id',dropna=False).groups.items():
                vals=sec.loc[idx].to_numpy(dtype=float); finite=vals[np.isfinite(vals)]
                if finite.size: sec.loc[idx]=sec.loc[idx]-finite.min()
        if overwrite or d['timestamp_seconds'].isna().all(): d['timestamp_seconds']=sec
        out[n]=d
    return add_provenance(out,"normalize_timebase",','.join(components),f"unit={native_unit or 'stream/default'};origin={origin}")

def audit_timebase(x, component="gaze_samples"):
    _assert_eye_dataset(x); d=x[component]
    if d.empty:return pd.DataFrame()
    if not {"recording_id","timestamp_seconds"}.issubset(d.columns): raise EyeProcessTimebaseError(f"`{component}` lacks required timebase columns.")
    rows=[]
    for rid,z in d.groupby('recording_id',dropna=False,sort=True):
        t=pd.to_numeric(z['timestamp_seconds'],errors='coerce').to_numpy(dtype=float); finite=t[np.isfinite(t)]
        ordered=np.sort(finite); dt=np.diff(ordered)
        pos=dt[dt>0]
        rows.append(dict(recording_id=rid,component=component,n=len(z),n_missing=int((~np.isfinite(t)).sum()),n_nonmonotonic=int((np.diff(finite)<0).sum()) if finite.size>1 else 0,n_duplicate_time=int(pd.Series(finite).duplicated().sum()),median_interval_ms=float(np.median(pos)*1000) if pos.size else np.nan,estimated_hz=estimate_sampling_rate(t),max_gap_ms=float(np.max(dt)*1000) if dt.size else np.nan,status="warning" if finite.size>1 and bool((np.diff(finite)<0).any()) else "ok"))
    return pd.DataFrame(rows)

def align_clock(timestamp, offset=0, slope=1):
    a=np.asarray(timestamp,dtype=float)*float(slope)+float(offset)
    return float(a) if a.ndim==0 else a

@dataclass(frozen=True)
class EyeClockTransform:
    method:str; offset:float; slope:float; n_markers:int; residual_sd:float; max_abs_residual:float

def estimate_clock_transform(source_times, target_times, method="linear"):
    if method not in {"linear","offset"}: raise ValueError("Invalid method.")
    s=np.asarray(source_times,dtype=float); t=np.asarray(target_times,dtype=float); ok=np.isfinite(s)&np.isfinite(t); n=int(ok.sum())
    if n<1: raise EyeProcessTimebaseError("No valid marker pairs for clock alignment.")
    ss=s[ok]; tt=t[ok]
    if method=="offset" or n<2:
        slope=1.; offset=float(np.median(tt-ss)); residuals=tt-align_clock(ss,offset,slope)
    else:
        slope,offset=np.polyfit(ss,tt,1); residuals=tt-align_clock(ss,offset,slope)
    sd=float(np.std(residuals,ddof=1)) if residuals.size>1 else np.nan
    return EyeClockTransform(method,float(offset),float(slope),n,sd,float(np.max(np.abs(residuals))))

def apply_clock_transform(x, transform, components=("biometrics",), source_clock=None):
    _assert_eye_dataset(x); out=x.copy(); comps=[components] if isinstance(components,str) else list(components)
    try: offset=transform.offset; slope=transform.slope
    except AttributeError:
        try: offset=transform['offset']; slope=transform['slope']
        except Exception as e: raise EyeProcessTimebaseError("`transform` must be an eye clock transform.") from e
    for n in comps:
        d=out[n].copy()
        if not d.empty and 'timestamp_seconds' in d: d['timestamp_seconds']=align_clock(pd.to_numeric(d['timestamp_seconds'],errors='coerce'),offset,slope); out[n]=d
    return add_provenance(out,"apply_clock_transform",','.join(comps),f"offset={offset};slope={slope}",reversible=np.isfinite(slope) and slope!=0)
