"""Vendor-neutral eye-tracking spatial and sampling data-quality metrics.

The module deliberately keeps accuracy, precision, sampling stability, and data
availability as separate quantities. Thresholds are review rules only: no
function silently excludes samples, trials, sessions, or participants.
"""
from __future__ import annotations

# ruff: noqa: E701, E702
import math
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "validate_gaze_quality_inputs", "compute_gaze_accuracy", "compute_gaze_precision",
    "compute_rms_s2s", "compute_gaze_sd_precision", "compute_bcea",
    "estimate_sampling_interval", "estimate_sampling_jitter", "estimate_effective_sampling_rate",
    "compute_valid_sample_fraction", "compute_gaze_data_loss", "summarise_spatial_quality",
    "summarise_sampling_quality", "create_gaze_quality_report", "compare_gaze_quality_sessions",
    "compare_gaze_quality_conditions", "plot_gaze_accuracy", "plot_gaze_precision", "plot_bcea",
    "plot_sampling_intervals", "plot_gaze_quality_dashboard", "report_gaze_quality",
    "simulate_gaze_quality_calibration",
]

_COORD_UNITS = {"pixels", "normalized", "degrees"}
_TIME_SCALES = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _df(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return pd.DataFrame(data)
    except Exception as exc:  # pragma: no cover - defensive conversion path
        raise TypeError("data must be coercible to a pandas DataFrame") from exc


def _by_list(by: str | Sequence[str] | None) -> list[str]:
    if by is None:
        return []
    return [by] if isinstance(by, str) else [str(x) for x in by]


def _require(d: pd.DataFrame, columns: Iterable[str | None], label: str = "data") -> None:
    needed = [c for c in columns if c]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _numeric(x: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)


def _groups(d: pd.DataFrame, by: str | Sequence[str] | None):
    keys = _by_list(by)
    if not keys:
        yield (), d
        return
    grouper: Any = keys[0] if len(keys) == 1 else keys
    for key, z in d.groupby(grouper, sort=True, dropna=False, observed=False):
        yield key if isinstance(key, tuple) else (key,), z


def _header(keys: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(keys, values, strict=True))


def _group_token(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple("<NA>" if pd.isna(value) else value for value in values)


def _unit_area(unit: str) -> str:
    return {"degrees": "deg^2", "pixels": "px^2", "normalized": "normalized^2"}[unit]


def _unit_linear(unit: str) -> str:
    return {"degrees": "deg", "pixels": "px", "normalized": "normalized"}[unit]


def _geometry_value(geometry: Mapping[str, Any], name: str) -> float:
    value = float(geometry.get(name, math.nan))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"geometry['{name}'] must be a finite positive number")
    return value


def _convert_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    unit: str,
    output_unit: str,
    geometry: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if unit == output_unit:
        return x.astype(float, copy=True), y.astype(float, copy=True)
    if geometry is None:
        raise ValueError("geometry is required when output_unit differs from input unit")
    wpx = _geometry_value(geometry, "screen_width_px")
    hpx = _geometry_value(geometry, "screen_height_px")
    wcm = _geometry_value(geometry, "screen_width_cm")
    hcm = _geometry_value(geometry, "screen_height_cm")
    dist = _geometry_value(geometry, "viewing_distance_cm")

    def to_px(a: np.ndarray, axis: str) -> np.ndarray:
        if unit == "pixels":
            return a
        if unit == "normalized":
            return a * (wpx if axis == "x" else hpx)
        cm = np.tan(np.deg2rad(a)) * dist
        return cm * ((wpx / wcm) if axis == "x" else (hpx / hcm)) + (wpx / 2 if axis == "x" else hpx / 2)

    px = to_px(x, "x")
    py = to_px(y, "y")
    if output_unit == "pixels":
        return px, py
    if output_unit == "normalized":
        return px / wpx, py / hpx
    xcm = (px - wpx / 2) * (wcm / wpx)
    ycm = (py - hpx / 2) * (hcm / hpx)
    return np.rad2deg(np.arctan2(xcm, dist)), np.rad2deg(np.arctan2(ycm, dist))


def _prepare_coordinates(
    d: pd.DataFrame,
    *,
    x: str,
    y: str,
    target_x: str | None = None,
    target_y: str | None = None,
    unit: str,
    output_unit: str | None,
    geometry: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    unit = str(unit).lower()
    out_unit = unit if output_unit is None else str(output_unit).lower()
    if unit not in _COORD_UNITS or out_unit not in _COORD_UNITS:
        raise ValueError("unit and output_unit must be 'pixels', 'normalized', or 'degrees'")
    gx, gy = _convert_xy(_numeric(d[x]), _numeric(d[y]), unit=unit, output_unit=out_unit, geometry=geometry)
    if target_x is None and target_y is None:
        return gx, gy, None, None, out_unit
    if target_x is None or target_y is None:
        raise ValueError("target_x and target_y must be supplied together")
    tx, ty = _convert_xy(_numeric(d[target_x]), _numeric(d[target_y]), unit=unit, output_unit=out_unit, geometry=geometry)
    return gx, gy, tx, ty, out_unit


def _source_fingerprint(d: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = d.loc[:, list(dict.fromkeys(columns))].to_csv(index=False, na_rep="<NA>")
    return sha256(payload.encode("utf-8")).hexdigest()


def _attach_provenance(df: pd.DataFrame, provenance: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out.attrs["gaze_quality_provenance"] = dict(provenance)
    return out


def validate_gaze_quality_inputs(
    data: Any,
    x: str = "gaze_x",
    y: str = "gaze_y",
    time: str | None = None,
    target_x: str | None = None,
    target_y: str | None = None,
    by: str | Sequence[str] | None = None,
    unit: str = "degrees",
    time_unit: str = "ms",
    unit_column: str | None = None,
) -> dict[str, Any]:
    """Validate structural inputs without dropping or repairing observations."""
    d = _df(data)
    keys = _by_list(by)
    _require(d, [x, y, time, target_x, target_y, *keys, unit_column])
    if (target_x is None) != (target_y is None):
        raise ValueError("target_x and target_y must be supplied together")
    unit = str(unit).lower()
    if unit not in _COORD_UNITS:
        raise ValueError("unit must be 'pixels', 'normalized', or 'degrees'")
    time_unit = str(time_unit).lower()
    if time is not None and time_unit not in _TIME_SCALES:
        raise ValueError("time_unit must be one of s, ms, us, ns")
    if unit_column is not None:
        vals = sorted(set(d[unit_column].dropna().astype(str).str.lower()))
        if len(vals) > 1:
            raise ValueError(f"mixed coordinate units are not allowed within one call: {vals}")
        if vals and vals[0] != unit:
            raise ValueError(f"unit='{unit}' conflicts with {unit_column}='{vals[0]}'")

    group_issues: list[dict[str, Any]] = []
    if time is not None:
        scale = _TIME_SCALES[time_unit]
        for g, z in _groups(d, keys):
            t = _numeric(z[time]) * scale
            finite = t[np.isfinite(t)]
            dt = np.diff(finite) if finite.size > 1 else np.array([], dtype=float)
            issues: list[str] = []
            if np.any(dt < 0):
                issues.append("non_monotonic_timestamps")
            if np.any(dt == 0):
                issues.append("duplicate_timestamps")
            if issues:
                group_issues.append({**_header(keys, g), "issues": issues})
    return {
        "n_rows": int(len(d)),
        "coordinate_unit": unit,
        "time_unit": time_unit if time is not None else None,
        "group_issues": group_issues,
        "valid": True,
    }


def compute_gaze_accuracy(
    data: Any,
    x: str = "gaze_x",
    y: str = "gaze_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
    by: str | Sequence[str] | None = None,
    unit: str = "degrees",
    output_unit: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute target-referenced gaze accuracy for each requested analysis unit."""
    d = _df(data); keys = _by_list(by)
    _require(d, [x, y, target_x, target_y, *keys])
    rows = []
    for g, z in _groups(d, keys):
        gx, gy, tx, ty, out_unit = _prepare_coordinates(z, x=x, y=y, target_x=target_x, target_y=target_y, unit=unit, output_unit=output_unit, geometry=geometry)
        assert tx is not None and ty is not None
        ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(tx) & np.isfinite(ty)
        dx, dy = gx[ok] - tx[ok], gy[ok] - ty[ok]
        radial = np.hypot(dx, dy)
        target_pairs = np.column_stack((tx[ok], ty[ok])) if ok.any() else np.empty((0, 2))
        n_targets = len(np.unique(target_pairs, axis=0)) if target_pairs.size else 0
        rows.append({**_header(keys, g), "n_accuracy_samples": int(ok.sum()), "n_accuracy_targets": int(n_targets),
                     "accuracy_mean": float(np.mean(radial)) if radial.size else math.nan,
                     "accuracy_median": float(np.median(radial)) if radial.size else math.nan,
                     "accuracy_horizontal": float(np.mean(np.abs(dx))) if dx.size else math.nan,
                     "accuracy_vertical": float(np.mean(np.abs(dy))) if dy.size else math.nan,
                     "accuracy_euclidean": float(np.mean(radial)) if radial.size else math.nan,
                     "accuracy_bias_x": float(np.mean(dx)) if dx.size else math.nan,
                     "accuracy_bias_y": float(np.mean(dy)) if dy.size else math.nan,
                     "unit": _unit_linear(out_unit)})
    out = pd.DataFrame(rows)
    return _attach_provenance(out, {"function": "compute_gaze_accuracy", "input_unit": unit, "output_unit": output_unit or unit})


def compute_rms_s2s(
    data: Any,
    x: str = "gaze_x",
    y: str = "gaze_y",
    time: str | None = None,
    by: str | Sequence[str] | None = None,
    unit: str = "degrees",
    output_unit: str | None = None,
    geometry: Mapping[str, Any] | None = None,
    dimension: str = "2d",
    time_unit: str = "ms",
    max_gap_ms: float | None = None,
) -> pd.DataFrame:
    """Compute RMS sample-to-sample displacement without bridging missing samples."""
    d = _df(data); keys = _by_list(by); dimension = str(dimension).lower()
    if dimension not in {"horizontal", "vertical", "2d"}:
        raise ValueError("dimension must be 'horizontal', 'vertical', or '2d'")
    _require(d, [x, y, time, *keys])
    if max_gap_ms is not None and (not math.isfinite(float(max_gap_ms)) or float(max_gap_ms) <= 0):
        raise ValueError("max_gap_ms must be a finite positive value when supplied")
    if max_gap_ms is not None and time is None:
        raise ValueError("time is required when max_gap_ms is supplied")
    if time is not None and time_unit not in _TIME_SCALES:
        raise ValueError("time_unit must be one of s, ms, us, ns")
    rows = []
    for g, z in _groups(d, keys):
        gx, gy, _, _, out_unit = _prepare_coordinates(z, x=x, y=y, unit=unit, output_unit=output_unit, geometry=geometry)
        finite_pair = np.isfinite(gx[:-1]) & np.isfinite(gy[:-1]) & np.isfinite(gx[1:]) & np.isfinite(gy[1:]) if len(z) > 1 else np.array([], dtype=bool)
        dx, dy = np.diff(gx), np.diff(gy)
        if dimension == "horizontal":
            step = np.abs(dx)
        elif dimension == "vertical":
            step = np.abs(dy)
        else:
            step = np.hypot(dx, dy)
        keep = finite_pair & np.isfinite(step)
        if time is not None and len(z) > 1:
            t = _numeric(z[time]) * _TIME_SCALES[time_unit]
            dt_ms = np.diff(t) * 1000.0
            keep &= np.isfinite(dt_ms) & (dt_ms > 0)
            if max_gap_ms is not None:
                keep &= dt_ms <= float(max_gap_ms)
        vals = step[keep]
        rows.append({**_header(keys, g), "n_steps": int(vals.size), "precision_rms_s2s": float(np.sqrt(np.mean(vals ** 2))) if vals.size else math.nan,
                     "median_s2s": float(np.median(vals)) if vals.size else math.nan,
                     "p95_s2s": float(np.quantile(vals, .95)) if vals.size else math.nan,
                     "dimension": dimension, "unit": _unit_linear(out_unit), "max_gap_ms": max_gap_ms})
    return _attach_provenance(pd.DataFrame(rows), {"function": "compute_rms_s2s", "missing_gap_policy": "never_bridge", "max_gap_ms": max_gap_ms})


def compute_gaze_sd_precision(
    data: Any,
    x: str = "gaze_x",
    y: str = "gaze_y",
    by: str | Sequence[str] | None = None,
    unit: str = "degrees",
    output_unit: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute spatial standard-deviation precision around a stable target."""
    d = _df(data); keys = _by_list(by); _require(d, [x, y, *keys])
    rows=[]
    for g,z in _groups(d, keys):
        gx,gy,_,_,out_unit=_prepare_coordinates(z,x=x,y=y,unit=unit,output_unit=output_unit,geometry=geometry)
        ok=np.isfinite(gx)&np.isfinite(gy); xx,yy=gx[ok],gy[ok]
        sx=float(np.std(xx,ddof=0)) if xx.size else math.nan; sy=float(np.std(yy,ddof=0)) if yy.size else math.nan
        rows.append({**_header(keys,g),"n_precision_samples":int(ok.sum()),"precision_sd_x":sx,"precision_sd_y":sy,
                     "precision_sd":float(math.hypot(sx,sy)) if math.isfinite(sx) and math.isfinite(sy) else math.nan,"unit":_unit_linear(out_unit)})
    return _attach_provenance(pd.DataFrame(rows), {"function":"compute_gaze_sd_precision"})


def compute_bcea(
    data: Any,
    x: str = "gaze_x",
    y: str = "gaze_y",
    by: str | Sequence[str] | None = None,
    probability: float = .68,
    unit: str = "degrees",
    output_unit: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute bivariate contour ellipse area (BCEA) at an explicit probability."""
    p=float(probability)
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("probability must lie strictly between 0 and 1")
    d=_df(data); keys=_by_list(by); _require(d,[x,y,*keys]); rows=[]; k=-math.log(1-p)
    for g,z in _groups(d,keys):
        gx,gy,_,_,out_unit=_prepare_coordinates(z,x=x,y=y,unit=unit,output_unit=output_unit,geometry=geometry)
        ok=np.isfinite(gx)&np.isfinite(gy); xx,yy=gx[ok],gy[ok]
        if xx.size>1:
            sx=float(np.std(xx,ddof=0)); sy=float(np.std(yy,ddof=0))
            rho=float(np.corrcoef(xx,yy)[0,1]) if sx>0 and sy>0 else 0.0
            rho=max(-1.0,min(1.0,rho))
            area=float(2*math.pi*k*sx*sy*math.sqrt(max(0.0,1-rho*rho)))
        else:
            sx=sy=rho=area=math.nan
        rows.append({**_header(keys,g),"n_bcea_samples":int(ok.sum()),"bcea":area,"bcea_probability":p,"sd_x":sx,"sd_y":sy,
                     "correlation_xy":rho,"unit":_unit_area(out_unit)})
    return _attach_provenance(pd.DataFrame(rows), {"function":"compute_bcea","probability":p})


def compute_gaze_precision(data: Any, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Return the distinct RMS-S2S, SD, and BCEA precision representations."""
    probability=kwargs.pop("probability",.68)
    rms_keys={k:v for k,v in kwargs.items() if k in {"x","y","time","by","unit","output_unit","geometry","dimension","time_unit","max_gap_ms"}}
    spatial_keys={k:v for k,v in kwargs.items() if k in {"x","y","by","unit","output_unit","geometry"}}
    return {"rms_s2s":compute_rms_s2s(data,**rms_keys),"sd":compute_gaze_sd_precision(data,**spatial_keys),"bcea":compute_bcea(data,probability=probability,**spatial_keys)}


def estimate_sampling_interval(data: Any,time: str="timestamp_ms",by: str|Sequence[str]|None=None,time_unit: str="ms") -> pd.DataFrame:
    d=_df(data); keys=_by_list(by); _require(d,[time,*keys])
    if time_unit not in _TIME_SCALES: raise ValueError("time_unit must be one of s, ms, us, ns")
    rows=[]
    for g,z in _groups(d,keys):
        t=_numeric(z[time])*_TIME_SCALES[time_unit]; finite=t[np.isfinite(t)]; dt=np.diff(finite)*1000 if finite.size>1 else np.array([],float)
        positive=dt[dt>0]
        rows.append({**_header(keys,g),"n_observed_timestamps":int(finite.size),"n_intervals":int(positive.size),
                     "median_interval_ms":float(np.median(positive)) if positive.size else math.nan,"mean_interval_ms":float(np.mean(positive)) if positive.size else math.nan,
                     "min_interval_ms":float(np.min(positive)) if positive.size else math.nan,"max_interval_ms":float(np.max(positive)) if positive.size else math.nan,
                     "duplicate_timestamp_count":int(np.sum(dt==0)),"non_monotonic_timestamp_count":int(np.sum(dt<0))})
    return _attach_provenance(pd.DataFrame(rows),{"function":"estimate_sampling_interval","time_unit":time_unit})


def estimate_sampling_jitter(data: Any,time: str="timestamp_ms",by: str|Sequence[str]|None=None,time_unit: str="ms") -> pd.DataFrame:
    d=_df(data); keys=_by_list(by); _require(d,[time,*keys])
    if time_unit not in _TIME_SCALES: raise ValueError("time_unit must be one of s, ms, us, ns")
    rows=[]
    for g,z in _groups(d,keys):
        t=_numeric(z[time])*_TIME_SCALES[time_unit]; finite=t[np.isfinite(t)]; dt=np.diff(finite)*1000 if finite.size>1 else np.array([],float); pos=dt[dt>0]
        med=float(np.median(pos)) if pos.size else math.nan
        rows.append({**_header(keys,g),"sampling_jitter_ms":float(np.std(pos-med,ddof=1)) if pos.size>1 else math.nan,
                     "sampling_jitter_mad_ms":float(np.median(np.abs(pos-med))) if pos.size else math.nan,"median_interval_ms":med})
    return _attach_provenance(pd.DataFrame(rows),{"function":"estimate_sampling_jitter","definition":"SD of positive inter-sample intervals around their median"})


def estimate_effective_sampling_rate(
    data: Any,time: str="timestamp_ms",by: str|Sequence[str]|None=None,time_unit: str="ms",nominal_sampling_hz: float|None=None,
    dropped_interval_factor: float=1.5,x: str|None=None,y: str|None=None,valid: str|None=None,
) -> pd.DataFrame:
    d=_df(data); keys=_by_list(by); _require(d,[time,x,y,valid,*keys])
    if (x is None) != (y is None): raise ValueError("x and y must either both be supplied or both be omitted")
    if valid is not None and x is None: raise ValueError("x and y are required when valid is supplied")
    if time_unit not in _TIME_SCALES: raise ValueError("time_unit must be one of s, ms, us, ns")
    if nominal_sampling_hz is not None and (not math.isfinite(float(nominal_sampling_hz)) or float(nominal_sampling_hz)<=0): raise ValueError("nominal_sampling_hz must be positive")
    if not math.isfinite(float(dropped_interval_factor)) or float(dropped_interval_factor)<=1: raise ValueError("dropped_interval_factor must be > 1")
    rows=[]
    for g,z in _groups(d,keys):
        t=_numeric(z[time])*_TIME_SCALES[time_unit]; finite_t=np.isfinite(t); finite=t[finite_t]
        dt=np.diff(finite) if finite.size>1 else np.array([],float); pos=dt[dt>0]
        med=float(np.median(pos)) if pos.size else math.nan
        span=float(finite[-1]-finite[0]) if finite.size>1 else math.nan
        duration=span+med if math.isfinite(span) and math.isfinite(med) and med>0 else (1/float(nominal_sampling_hz) if finite.size==1 and nominal_sampling_hz is not None else math.nan)
        if x is not None:
            good,_,_=_valid_masks(z,x,y,valid); effective_count=int(np.sum(good & finite_t)); count_rule="valid gaze samples with finite timestamps"
        else:
            effective_count=int(finite.size); count_rule="finite timestamps"
        hz=float(effective_count/duration) if math.isfinite(duration) and duration>0 else math.nan
        long_count=math.nan
        dropped=math.nan
        if nominal_sampling_hz is not None and pos.size:
            expected=1/float(nominal_sampling_hz)
            long_mask=pos>float(dropped_interval_factor)*expected
            long_count=int(np.sum(long_mask))
            if long_count:
                nominal_steps=np.floor((pos[long_mask]/expected)+0.5).astype(int)
                dropped=int(np.sum(np.maximum(nominal_steps-1,0)))
            else:
                dropped=0
        rows.append({**_header(keys,g),"observed_sample_count":int(finite.size),"effective_sample_count":effective_count,
                     "timestamp_span_s":span,"trial_duration_s":duration,"effective_sampling_hz":hz,
                     "median_interval_ms":med*1000 if math.isfinite(med) else math.nan,
                     "long_interval_count":long_count,"dropped_interval_count":dropped,
                     "nominal_sampling_hz":nominal_sampling_hz,"effective_count_rule":count_rule})
    return _attach_provenance(pd.DataFrame(rows),{"function":"estimate_effective_sampling_rate",
        "definition":"effective sample count / estimated recording duration",
        "duration_estimator":"timestamp span plus one median positive inter-sample interval",
        "long_interval_definition":"positive interval exceeding dropped_interval_factor times the nominal interval",
        "dropped_interval_definition":"sum of estimated missing nominal samples within long intervals",
        "dropped_interval_factor":dropped_interval_factor})


def _valid_masks(d: pd.DataFrame,x: str,y: str,valid: str|None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    gx,gy=_numeric(d[x]),_numeric(d[y]); missing=~np.isfinite(gx)|~np.isfinite(gy)
    if valid is None:
        explicit=np.ones(len(d),dtype=bool)
    else:
        s=d[valid]
        if pd.api.types.is_bool_dtype(s.dtype): explicit=s.fillna(False).to_numpy(bool)
        else:
            vals=pd.to_numeric(s,errors="coerce").to_numpy(float); explicit=np.isfinite(vals)&(vals>0)
    good=(~missing)&explicit; invalid=(~missing)&(~explicit)
    return good,invalid,missing


def compute_valid_sample_fraction(data: Any,x: str="gaze_x",y: str="gaze_y",valid: str|None=None,by: str|Sequence[str]|None=None) -> pd.DataFrame:
    d=_df(data); keys=_by_list(by); _require(d,[x,y,valid,*keys]); rows=[]
    for g,z in _groups(d,keys):
        good,invalid,missing=_valid_masks(z,x,y,valid); n=len(z)
        rows.append({**_header(keys,g),"n_samples":n,"valid_sample_fraction":float(good.mean()) if n else math.nan,
                     "invalid_sample_fraction":float(invalid.mean()) if n else math.nan,"missing_sample_fraction":float(missing.mean()) if n else math.nan})
    return _attach_provenance(pd.DataFrame(rows),{"function":"compute_valid_sample_fraction","validity_rule":valid or "finite x and y"})


def compute_gaze_data_loss(
    data: Any,x: str="gaze_x",y: str="gaze_y",time: str|None="timestamp_ms",valid: str|None=None,missing_reason: str|None=None,
    by: str|Sequence[str]|None=None,time_unit: str="ms",
) -> pd.DataFrame:
    d=_df(data); keys=_by_list(by); _require(d,[x,y,time,valid,missing_reason,*keys])
    if time is not None and time_unit not in _TIME_SCALES: raise ValueError("time_unit must be one of s, ms, us, ns")
    rows=[]
    for g,z in _groups(d,keys):
        good,invalid,missing=_valid_masks(z,x,y,valid); lost=~good; n=len(z); runs=[]; start=None
        for i,flag in enumerate(lost):
            if flag and start is None: start=i
            if start is not None and (not flag or i==n-1):
                end=i if flag and i==n-1 else i-1; runs.append((start,end)); start=None
        longest_samples=max((b-a+1 for a,b in runs),default=0); longest_ms=math.nan
        if time is not None and runs:
            t=_numeric(z[time])*_TIME_SCALES[time_unit]*1000
            durations=[]
            pos=np.diff(t); typical=float(np.median(pos[np.isfinite(pos)&(pos>0)])) if np.any(np.isfinite(pos)&(pos>0)) else 0.0
            for a,b in runs:
                if np.isfinite(t[a]) and np.isfinite(t[b]): durations.append(max(0.0,float(t[b]-t[a]))+typical)
            longest_ms=max(durations,default=math.nan)
        row={**_header(keys,g),"n_samples":n,"valid_sample_fraction":float(good.mean()) if n else math.nan,
             "invalid_sample_fraction":float(invalid.mean()) if n else math.nan,"missing_sample_fraction":float(missing.mean()) if n else math.nan,
             "data_loss_fraction":float(lost.mean()) if n else math.nan,"missing_run_count":len(runs),"longest_missing_run_samples":longest_samples,"longest_missing_run_ms":longest_ms}
        if missing_reason is not None:
            reasons=z.loc[lost,missing_reason].fillna("unknown").astype(str).str.lower(); denom=max(1,int(lost.sum()))
            for reason,count in reasons.value_counts().sort_index().items(): row[f"missing_reason_{reason}_fraction"]=float(count/denom)
        rows.append(row)
    return _attach_provenance(pd.DataFrame(rows),{"function":"compute_gaze_data_loss","loss_rule":"missing coordinates or explicit invalidity"})


def _merge_quality_parts(parts: Sequence[pd.DataFrame], keys: Sequence[str]) -> pd.DataFrame:
    """Merge quality tables without manufacturing duplicate metric columns."""
    if not parts:
        return pd.DataFrame()
    out = parts[0].reset_index(drop=True)
    for part in parts[1:]:
        right = part.reset_index(drop=True)
        if keys:
            duplicate_nonkeys = [c for c in right.columns if c in out.columns and c not in keys]
            if duplicate_nonkeys:
                right = right.drop(columns=duplicate_nonkeys)
            out = out.merge(right, on=list(keys), how="outer")
        else:
            duplicate_nonkeys = [c for c in right.columns if c in out.columns]
            if duplicate_nonkeys:
                right = right.drop(columns=duplicate_nonkeys)
            out = pd.concat([out, right], axis=1)
    return out


def summarise_spatial_quality(data: Any, **kwargs: Any) -> pd.DataFrame:
    keys=_by_list(kwargs.get("by")); acc=compute_gaze_accuracy(data,**{k:v for k,v in kwargs.items() if k in {"x","y","target_x","target_y","by","unit","output_unit","geometry"}})
    rms=compute_rms_s2s(data,**{k:v for k,v in kwargs.items() if k in {"x","y","time","by","unit","output_unit","geometry","dimension","time_unit","max_gap_ms"}})
    sd=compute_gaze_sd_precision(data,**{k:v for k,v in kwargs.items() if k in {"x","y","by","unit","output_unit","geometry"}})
    bcea=compute_bcea(data,probability=kwargs.get("probability",.68),**{k:v for k,v in kwargs.items() if k in {"x","y","by","unit","output_unit","geometry"}})
    acc=acc.rename(columns={"unit":"accuracy_unit"})
    rms=rms.rename(columns={"unit":"precision_rms_s2s_unit"})
    sd=sd.rename(columns={"unit":"precision_sd_unit"})
    bcea=bcea.rename(columns={"unit":"bcea_unit"})
    out=_merge_quality_parts([acc,rms,sd,bcea],keys)
    return _attach_provenance(out,{"function":"summarise_spatial_quality"})


def summarise_sampling_quality(data: Any,**kwargs: Any) -> pd.DataFrame:
    keys=_by_list(kwargs.get("by")); inter=estimate_sampling_interval(data,**{k:v for k,v in kwargs.items() if k in {"time","by","time_unit"}})
    jit=estimate_sampling_jitter(data,**{k:v for k,v in kwargs.items() if k in {"time","by","time_unit"}})
    eff=estimate_effective_sampling_rate(data,**{k:v for k,v in kwargs.items() if k in {"time","by","time_unit","nominal_sampling_hz","dropped_interval_factor","x","y","valid"}})
    out=_merge_quality_parts([inter,jit,eff],keys)
    return _attach_provenance(out,{"function":"summarise_sampling_quality"})


def _validate_thresholds(thresholds: Mapping[str,Any]|None, columns: Sequence[str]) -> None:
    if not thresholds:
        return
    unknown=sorted(set(thresholds)-set(columns))
    if unknown:
        raise ValueError(f"threshold metrics are not present in the quality report: {', '.join(unknown)}")
    for metric,rule in thresholds.items():
        if isinstance(rule,Mapping):
            bad=sorted(set(rule)-{"min","max"})
            if bad or not rule:
                raise ValueError(f"threshold rule for {metric} must contain only 'min' and/or 'max'")
            values=list(rule.values())
        else:
            values=[rule]
        for value in values:
            try:
                numeric=float(value)
            except (TypeError,ValueError) as exc:
                raise ValueError(f"threshold value for {metric} must be finite numeric") from exc
            if not math.isfinite(numeric):
                raise ValueError(f"threshold value for {metric} must be finite numeric")


def _threshold_flags(row: pd.Series,thresholds: Mapping[str,Any]|None) -> list[str]:
    flags=[]
    if not thresholds: return flags
    for metric,rule in thresholds.items():
        if not pd.notna(row[metric]): continue
        value=float(row[metric])
        if isinstance(rule,Mapping):
            if "max" in rule and value>float(rule["max"]): flags.append(f"{metric}>max")
            if "min" in rule and value<float(rule["min"]): flags.append(f"{metric}<min")
        else:
            if value>float(rule): flags.append(f"{metric}>max")
    return flags


def create_gaze_quality_report(
    data: Any,x: str="gaze_x",y: str="gaze_y",time: str="timestamp_ms",target_x: str|None="target_x",target_y: str|None="target_y",
    valid: str|None=None,missing_reason: str|None=None,by: str|Sequence[str]|None=None,unit: str="degrees",output_unit: str|None=None,
    geometry: Mapping[str,Any]|None=None,time_unit: str="ms",nominal_sampling_hz: float|None=None,bcea_probability: float=.68,
    max_gap_ms: float|None=None,thresholds: Mapping[str,Any]|None=None,preprocessing_spec: Any=None,event_detector: Any=None,aoi_specification: Any=None,
    quality_rules: Any=None,model_specification: Any=None,software_version: str|None=None,unit_column: str|None=None,
) -> pd.DataFrame:
    """Create a manuscript-ready report; thresholds only flag rows for review."""
    d=_df(data); keys=_by_list(by)
    resolved_unit_column=unit_column
    if resolved_unit_column is None and "coordinate_unit" in d.columns:
        resolved_unit_column="coordinate_unit"
    _require(d,[x,y,time,valid,missing_reason,resolved_unit_column,*keys])
    has_targets=target_x is not None and target_y is not None and target_x in d.columns and target_y in d.columns
    validation=validate_gaze_quality_inputs(d,x=x,y=y,time=time,target_x=target_x if has_targets else None,target_y=target_y if has_targets else None,by=keys,unit=unit,time_unit=time_unit,unit_column=resolved_unit_column)
    if has_targets:
        spatial=summarise_spatial_quality(d,x=x,y=y,time=time,target_x=target_x,target_y=target_y,by=keys,unit=unit,output_unit=output_unit,geometry=geometry,time_unit=time_unit,max_gap_ms=max_gap_ms,probability=bcea_probability)
    else:
        rms=compute_rms_s2s(d,x=x,y=y,time=time,by=keys,unit=unit,output_unit=output_unit,geometry=geometry,time_unit=time_unit,max_gap_ms=max_gap_ms)
        sd=compute_gaze_sd_precision(d,x=x,y=y,by=keys,unit=unit,output_unit=output_unit,geometry=geometry)
        bc=compute_bcea(d,x=x,y=y,by=keys,probability=bcea_probability,unit=unit,output_unit=output_unit,geometry=geometry)
        rms=rms.rename(columns={"unit":"precision_rms_s2s_unit"})
        sd=sd.rename(columns={"unit":"precision_sd_unit"})
        bc=bc.rename(columns={"unit":"bcea_unit"})
        spatial=_merge_quality_parts([rms,sd,bc],keys)
    sampling=summarise_sampling_quality(d,time=time,by=keys,time_unit=time_unit,nominal_sampling_hz=nominal_sampling_hz,x=x,y=y,valid=valid)
    loss=compute_gaze_data_loss(d,x=x,y=y,time=time,valid=valid,missing_reason=missing_reason,by=keys,time_unit=time_unit)
    report=_merge_quality_parts([spatial,sampling,loss],keys)
    _validate_thresholds(thresholds,report.columns)
    issue_map={_group_token([item.get(k) for k in keys]):item["issues"] for item in validation["group_issues"]}
    flags=[]
    for _,row in report.iterrows():
        rf=[]; key=_group_token([row.get(k) for k in keys])
        rf.extend(issue_map.get(key,[]))
        if "n_accuracy_targets" in row and pd.notna(row.get("n_accuracy_targets")) and float(row["n_accuracy_targets"])>1: rf.append("mixed_accuracy_targets")
        if "n_bcea_samples" in row and pd.notna(row.get("n_bcea_samples")) and float(row["n_bcea_samples"])<2: rf.append("insufficient_bcea_samples")
        if "n_steps" in row and pd.notna(row.get("n_steps")) and float(row["n_steps"])<1: rf.append("insufficient_rms_pairs")
        if "valid_sample_fraction" in row and pd.notna(row.get("valid_sample_fraction")) and float(row["valid_sample_fraction"])<=0: rf.append("no_valid_gaze_samples")
        rf.extend(_threshold_flags(row,thresholds)); flags.append(sorted(set(rf)))
    report["quality_flags"]=[";".join(x) for x in flags]
    report["review_required"]=[bool(x) for x in flags]
    provenance={"source_fingerprint":_source_fingerprint(d,[x,y,time,valid,missing_reason,resolved_unit_column,*([target_x,target_y] if has_targets else []),*keys]),
                "preprocessing_spec":preprocessing_spec,"event_detector":event_detector,"aoi_specification":aoi_specification,
                "quality_rules":quality_rules if quality_rules is not None else thresholds,"model_specification":model_specification,
                "software_version":software_version,"input_unit":unit,"output_unit":output_unit or unit,"unit_column":resolved_unit_column,"time_unit":time_unit,
                "nominal_sampling_hz":nominal_sampling_hz,"bcea_probability":bcea_probability,"max_gap_ms":max_gap_ms,
                "automatic_exclusion":False}
    return _attach_provenance(report,provenance)


def _compare_reports(data: Any,factor: str,**kwargs: Any) -> pd.DataFrame:
    if factor not in _df(data).columns: raise ValueError(f"data is missing required column: {factor}")
    by=_by_list(kwargs.pop("by",None)); by=list(dict.fromkeys([*by,factor])); return create_gaze_quality_report(data,by=by,**kwargs)


def compare_gaze_quality_sessions(data: Any,session: str="session_id",**kwargs: Any) -> pd.DataFrame:
    return _compare_reports(data,session,**kwargs)


def compare_gaze_quality_conditions(data: Any,condition: str="condition",**kwargs: Any) -> pd.DataFrame:
    return _compare_reports(data,condition,**kwargs)


def _ax(ax: Any=None):
    import matplotlib.pyplot as plt
    return plt.subplots()[1] if ax is None else ax


def plot_gaze_accuracy(report: Any,metric: str="accuracy_mean",ax: Any=None):
    d=_df(report); _require(d,[metric]); ax=_ax(ax); ax.plot(np.arange(len(d)),pd.to_numeric(d[metric],errors="coerce"),marker="o"); ax.set(ylabel=metric,xlabel="analysis unit",title="Gaze accuracy"); ax.eyeprocess_plot_data=d; return ax


def plot_gaze_precision(report: Any,metric: str="precision_rms_s2s",ax: Any=None):
    d=_df(report); _require(d,[metric]); ax=_ax(ax); ax.plot(np.arange(len(d)),pd.to_numeric(d[metric],errors="coerce"),marker="o"); ax.set(ylabel=metric,xlabel="analysis unit",title="Gaze precision"); ax.eyeprocess_plot_data=d; return ax


def plot_bcea(report: Any,ax: Any=None):
    d=_df(report); _require(d,["bcea"]); ax=_ax(ax); ax.bar(np.arange(len(d)),pd.to_numeric(d.bcea,errors="coerce")); ax.set(ylabel="BCEA",xlabel="analysis unit",title="Bivariate contour ellipse area"); ax.eyeprocess_plot_data=d; return ax


def plot_sampling_intervals(data: Any,time: str="timestamp_ms",time_unit: str="ms",ax: Any=None):
    d=_df(data); _require(d,[time])
    if time_unit not in _TIME_SCALES: raise ValueError("time_unit must be one of s, ms, us, ns")
    t=_numeric(d[time])*_TIME_SCALES[time_unit]; dt=np.diff(t)*1000; ax=_ax(ax); ax.plot(np.arange(1,len(dt)+1),dt,marker="."); ax.set(ylabel="interval (ms)",xlabel="interval",title="Sampling intervals"); ax.eyeprocess_plot_data=pd.DataFrame({"interval_ms":dt}); return ax


def plot_gaze_quality_dashboard(report: Any):
    import matplotlib.pyplot as plt
    d=_df(report); fig,axes=plt.subplots(2,2,figsize=(10,7)); metrics=[("accuracy_mean","Accuracy"),("precision_rms_s2s","RMS-S2S precision"),("bcea","BCEA"),("valid_sample_fraction","Valid fraction")]
    for ax,(m,title) in zip(axes.flat,metrics,strict=True):
        if m in d: ax.bar(np.arange(len(d)),pd.to_numeric(d[m],errors="coerce")); ax.set_title(title); ax.set_xlabel("analysis unit")
        else: ax.text(.5,.5,f"{m} unavailable",ha="center",va="center"); ax.set_title(title)
    fig.tight_layout(); fig.eyeprocess_plot_data=d; return fig


def report_gaze_quality(report: Any,digits: int=3) -> str:
    if isinstance(digits,bool) or not isinstance(digits,(int,np.integer)) or int(digits)<0:
        raise ValueError("digits must be a non-negative integer")
    digits=int(digits)
    d=_df(report)
    if d.empty: return "No gaze-quality rows were available."
    cols=[c for c in ["accuracy_mean","precision_rms_s2s","precision_sd","bcea","effective_sampling_hz","valid_sample_fraction","data_loss_fraction"] if c in d]
    parts=[]
    for c in cols:
        v=pd.to_numeric(d[c],errors="coerce"); v=v[np.isfinite(v)]
        if len(v): parts.append(f"{c}: mean {v.mean():.{digits}f}, range {v.min():.{digits}f}–{v.max():.{digits}f}")
    n_review=int(pd.Series(d.get("review_required",False)).fillna(False).astype(bool).sum()) if len(d) else 0
    return "; ".join(parts)+f". Review required for {n_review}/{len(d)} analysis units. Thresholds, when supplied, are study-specific review rules and never trigger automatic exclusion."


def simulate_gaze_quality_calibration(seed: int=20260918,samples_per_target: int=18,nominal_sampling_hz: float=60.0) -> pd.DataFrame:
    """Generate a reproducible 9-point validation set with six quality profiles."""
    try:
        samples_float=float(samples_per_target)
        sampling_hz=float(nominal_sampling_hz)
    except (TypeError,ValueError) as exc:
        raise ValueError("samples_per_target and nominal_sampling_hz must be numeric") from exc
    if not math.isfinite(samples_float) or samples_float<4 or not samples_float.is_integer():
        raise ValueError("samples_per_target must be an integer of at least 4")
    if not math.isfinite(sampling_hz) or sampling_hz<=0:
        raise ValueError("nominal_sampling_hz must be a finite positive value")
    samples_per_target=int(samples_float)
    nominal_sampling_hz=sampling_hz
    rng=np.random.default_rng(int(seed)); targets=[(-5,-5),(0,-5),(5,-5),(-5,0),(0,0),(5,0),(-5,5),(0,5),(5,5)]
    specs={"good_accuracy_good_precision":(0.0,0.0,.12),"poor_accuracy_good_precision":(.9,-.7,.12),"good_accuracy_poor_precision":(0.0,0.0,.75),"poor_accuracy_poor_precision":(.9,-.7,.75),"irregular_sampling":(.1,-.1,.20),"missingness":(.1,-.1,.20)}
    rows=[]
    for profile,(bx,by,sd) in specs.items():
        t=0.0
        for tid,(tx,ty) in enumerate(targets,1):
            for sample in range(samples_per_target):
                dt=1/nominal_sampling_hz
                if profile=="irregular_sampling": dt*=max(.25,1+rng.normal(0,.35)); dt*=2.5 if sample in {6,13} else 1
                t+=dt; gx=tx+bx+rng.normal(0,sd); gy=ty+by+rng.normal(0,sd); valid=True; reason=None
                if profile=="missingness" and sample in {5,6,7,14}: gx=gy=np.nan; valid=False; reason="blink" if sample in {5,6} else "tracker_invalidity"
                rows.append({"participant_id":"P001","session_id":"S001","profile":profile,"target_id":tid,"sample_in_target":sample+1,
                             "timestamp_ms":t*1000,"target_x":tx,"target_y":ty,"gaze_x":gx,"gaze_y":gy,"valid":valid,"missing_reason":reason,"coordinate_unit":"degrees"})
    return pd.DataFrame(rows)
