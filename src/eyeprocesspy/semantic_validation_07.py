"""Semantic-fidelity and independent validation contracts from eyeprocess 0.7.

These functions preserve the R package's distinction between adapter availability,
semantic fidelity, and empirical validation evidence.  They never silently coerce
clock or coordinate semantics during an audit.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult, _as_df, _req_cols, _result

__all__ = [
    "validation_evidence_levels", "semantic_fidelity_spec", "field_fidelity_report",
    "timestamp_fidelity_audit", "coordinate_fidelity_audit", "pupil_unit_fidelity_audit",
    "eye_stream_fidelity_audit", "event_semantics_audit", "validate_hed_event_semantics",
    "validate_bids_eye_semantics", "semantic_roundtrip_audit", "semantic_loss_map",
    "public_validation_corpus", "compatibility_evidence_matrix",
    "validate_vendor_timestamp_semantics", "plot_eye_semantic_roundtrip",
    "plot_eye_compatibility_evidence_matrix",
]

_FIDELITY_LEVELS = [
    "LOSSLESS", "SEMANTICALLY_EQUIVALENT", "UNIT_TRANSFORMED", "COORDINATE_TRANSFORMED",
    "DERIVED", "INTENTIONALLY_DROPPED", "UNSUPPORTED", "AMBIGUOUS", "MISSING",
]


def _df(x: Any, name: str) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, Mapping):
        for key in ("samples", "data", "gaze"):
            z = x.get(key)
            if isinstance(z, pd.DataFrame):
                return z.copy()
    return _as_df(x, name)


def _safe_cor(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return math.nan
    if np.std(a[ok], ddof=1) == 0 or np.std(b[ok], ddof=1) == 0:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _rank(status: str) -> int | float:
    try:
        return _FIDELITY_LEVELS.index(status) + 1
    except ValueError:
        return math.nan


def _align(source: pd.DataFrame, roundtrip: pd.DataFrame, key: str | Sequence[str] | None,
           allow_row_reorder: bool = True) -> dict[str, Any]:
    if key is None:
        n = min(len(source), len(roundtrip))
        return {"source": source.iloc[:n].reset_index(drop=True),
                "roundtrip": roundtrip.iloc[:n].reset_index(drop=True),
                "source_n": len(source), "roundtrip_n": len(roundtrip), "matched_n": n,
                "alignment": "row-order"}
    keys = [key] if isinstance(key, str) else list(key)
    _req_cols(source, keys, "source"); _req_cols(roundtrip, keys, "roundtrip")
    if not allow_row_reorder and not source[keys].reset_index(drop=True).equals(roundtrip[keys].reset_index(drop=True)):
        raise EyeProcessValidationError("Key columns differ and allow_row_reorder=False.")
    sk = source[keys].astype(str).agg("\r".join, axis=1)
    rk = roundtrip[keys].astype(str).agg("\r".join, axis=1)
    if sk.duplicated().any() or rk.duplicated().any():
        raise EyeProcessValidationError("Round-trip keys must uniquely identify rows.")
    pos = pd.Series(np.arange(len(roundtrip)), index=rk.to_numpy())
    idx = sk.map(pos)
    keep = idx.notna().to_numpy()
    return {"source": source.loc[keep].reset_index(drop=True),
            "roundtrip": roundtrip.iloc[idx.loc[keep].astype(int).to_numpy()].reset_index(drop=True),
            "source_n": len(source), "roundtrip_n": len(roundtrip), "matched_n": int(keep.sum()),
            "alignment": "+".join(keys)}


def validation_evidence_levels() -> pd.DataFrame:
    return pd.DataFrame({
        "rank": range(1, 7),
        "level": ["declared", "synthetic-fixture", "vendor-example", "independent-public-real",
                  "multisession-multidevice-real", "semantic-roundtrip-validated"],
        "requirement": [
            "Adapter or schema support is declared.",
            "Deterministic synthetic or package fixture passes the declared contract.",
            "A vendor-provided example export passes import and semantic checks.",
            "An independently produced public real recording passes the declared checks.",
            "Evidence spans repeated sessions and/or more than one device/model context.",
            "Native-to-canonical-to-interchange-to-canonical round trip has field-level semantic-loss evidence.",
        ],
    })


def semantic_fidelity_spec(timestamp_tolerance: float = 1e-6, coordinate_tolerance: float = 1e-6,
                           pupil_tolerance: float = 1e-6, missingness_tolerance: float = 1e-6,
                           correlation_floor: float = 0.999, allow_row_reorder: bool = True) -> EyeResult:
    vals = np.asarray([timestamp_tolerance, coordinate_tolerance, pupil_tolerance, missingness_tolerance], float)
    if np.any(~np.isfinite(vals)) or np.any(vals < 0):
        raise EyeProcessValidationError("Tolerances must be finite and non-negative.")
    if not np.isfinite(correlation_floor) or not -1 <= correlation_floor <= 1:
        raise EyeProcessValidationError("correlation_floor must lie in [-1, 1].")
    return _result("eye_semantic_fidelity_spec", timestamp_tolerance=float(timestamp_tolerance),
                   coordinate_tolerance=float(coordinate_tolerance), pupil_tolerance=float(pupil_tolerance),
                   missingness_tolerance=float(missingness_tolerance), correlation_floor=float(correlation_floor),
                   allow_row_reorder=bool(allow_row_reorder))


def field_fidelity_report(source: Any, roundtrip: Any, fields: Sequence[str] | str | None = None,
                          mapping: Mapping[str, str] | None = None, key: str | Sequence[str] | None = None,
                          tolerance: float = 1e-8, spec: Any = None) -> EyeResult:
    source, roundtrip = _df(source, "source"), _df(roundtrip, "roundtrip")
    spec = semantic_fidelity_spec() if spec is None else spec
    al = _align(source, roundtrip, key, bool(spec.allow_row_reorder))
    a, b = al["source"], al["roundtrip"]
    if fields is None:
        common = [c for c in a.columns if c in b.columns]
        preferred = ["participant_id", "recording_id", "trial_id", "stimulus_id", "timestamp",
                     "timestamp_native", "x", "y", "pupil_left", "pupil_right", "eye", "event",
                     "event_type", "fixation_id"]
        fields = list(dict.fromkeys([c for c in preferred if c in common] + common))
    elif isinstance(fields, str):
        fields = [fields]
    else:
        fields = list(fields)
    mp = {f: f for f in fields} if mapping is None else dict(mapping)
    rows: list[dict[str, Any]] = []
    for src in fields:
        dst = mp.get(src, src)
        base = {"field": src, "roundtrip_field": dst, "source_present": src in a.columns,
                "roundtrip_present": dst in b.columns, "n": 0, "missingness_delta": math.nan,
                "correlation": math.nan, "max_abs_error": math.nan, "transform_intercept": math.nan,
                "transform_slope": math.nan}
        if src not in a.columns:
            rows.append({**base, "status": "MISSING"}); continue
        if dst not in b.columns:
            rows.append({**base, "status": "UNSUPPORTED", "n": len(a)}); continue
        x, y = a[src].iloc[:min(len(a), len(b))], b[dst].iloc[:min(len(a), len(b))]
        n = len(x); md = abs(float(x.isna().mean()) - float(y.isna().mean())); status = "AMBIGUOUS"
        corv = err = intercept = slope = math.nan
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
        if numeric:
            xa, ya = xn.to_numpy(float), yn.to_numpy(float); ok = np.isfinite(xa) & np.isfinite(ya)
            if ok.any():
                err = float(np.max(np.abs(xa[ok] - ya[ok]))); corv = _safe_cor(xa, ya)
            same_na = np.array_equal(pd.isna(x).to_numpy(), pd.isna(y).to_numpy())
            if same_na and ok.any() and float(np.max(np.abs(xa[ok] - ya[ok]))) <= tolerance:
                status = "LOSSLESS"
            elif ok.sum() >= 3 and np.isfinite(corv) and corv >= float(spec.correlation_floor):
                slope, intercept = np.polyfit(xa[ok], ya[ok], 1)
                residual_error = float(np.max(np.abs((intercept + slope * xa[ok]) - ya[ok])))
                if residual_error <= max(tolerance, float(spec.coordinate_tolerance)):
                    status = "SEMANTICALLY_EQUIVALENT" if abs(intercept) <= tolerance and abs(slope - 1) <= tolerance else "UNIT_TRANSFORMED"
                else:
                    status = "SEMANTICALLY_EQUIVALENT"
        else:
            aa, bb = x.astype("string"), y.astype("string")
            same = (aa.eq(bb) | (x.isna() & y.isna())).fillna(False)
            if bool(same.all()) and np.array_equal(x.isna().to_numpy(), y.isna().to_numpy()):
                status = "LOSSLESS"
        if md > float(spec.missingness_tolerance) and status == "LOSSLESS":
            status = "SEMANTICALLY_EQUIVALENT"
        rows.append({**base, "status": status, "n": n, "missingness_delta": md, "correlation": corv,
                     "max_abs_error": err, "transform_intercept": float(intercept), "transform_slope": float(slope)})
    tab = pd.DataFrame(rows)
    tab["severity_rank"] = tab["status"].map(_rank)
    return _result("eye_field_fidelity_report", fields=tab, alignment=al["alignment"], source_n=al["source_n"],
                   roundtrip_n=al["roundtrip_n"], matched_n=al["matched_n"], spec=spec)


def timestamp_fidelity_audit(source: Any, roundtrip: Any, source_time: str = "timestamp",
                             roundtrip_time: str | None = None, source_unit: str = "seconds",
                             roundtrip_unit: str | None = None, key: str | Sequence[str] | None = None,
                             tolerance: float = 1e-6) -> EyeResult:
    s, r = _df(source, "source"), _df(roundtrip, "roundtrip")
    roundtrip_time = source_time if roundtrip_time is None else roundtrip_time
    roundtrip_unit = source_unit if roundtrip_unit is None else roundtrip_unit
    _req_cols(s, [source_time], "source"); _req_cols(r, [roundtrip_time], "roundtrip")
    mult = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}
    if source_unit not in mult or roundtrip_unit not in mult:
        raise EyeProcessValidationError("Timestamp units must be seconds, milliseconds, microseconds, or nanoseconds.")
    al = _align(s, r, key, True)
    a = pd.to_numeric(al["source"][source_time], errors="coerce").to_numpy(float) * mult[source_unit]
    b = pd.to_numeric(al["roundtrip"][roundtrip_time], errors="coerce").to_numpy(float) * mult[roundtrip_unit]
    ok = np.isfinite(a) & np.isfinite(b); delta = b[ok] - a[ok]
    offset = float(np.median(delta)) if delta.size else math.nan
    centered = float(np.max(np.abs(delta - offset))) if delta.size else math.nan
    intercept = slope = math.nan
    if ok.sum() >= 3 and np.std(a[ok], ddof=1) > 0:
        slope, intercept = np.polyfit(a[ok], b[ok], 1)
    ma = bool(np.all(np.diff(a[np.isfinite(a)]) >= 0)); mb = bool(np.all(np.diff(b[np.isfinite(b)]) >= 0))
    interval_cor = _safe_cor(np.diff(a[ok]), np.diff(b[ok])) if ok.sum() >= 4 else math.nan
    if delta.size and float(np.max(np.abs(delta))) <= tolerance: status = "LOSSLESS"
    elif np.isfinite(centered) and centered <= tolerance and mb: status = "SEMANTICALLY_EQUIVALENT"
    elif np.isfinite(slope) and abs(slope - 1) <= 1e-6 and mb: status = "SEMANTICALLY_EQUIVALENT"
    else: status = "AMBIGUOUS"
    return _result("eye_timestamp_fidelity", status=status, matched_n=int(ok.sum()), source_monotonic=ma,
                   roundtrip_monotonic=mb, offset_seconds=offset, affine_intercept_seconds=float(intercept),
                   affine_slope=float(slope), max_centered_error_seconds=centered,
                   interval_correlation=interval_cor, tolerance_seconds=float(tolerance))


def coordinate_fidelity_audit(source: Any, roundtrip: Any, source_x: str = "x", source_y: str = "y",
                              roundtrip_x: str | None = None, roundtrip_y: str | None = None,
                              key: str | Sequence[str] | None = None, tolerance: float = 1e-6,
                              correlation_floor: float = 0.999) -> EyeResult:
    s, r = _df(source, "source"), _df(roundtrip, "roundtrip")
    roundtrip_x = source_x if roundtrip_x is None else roundtrip_x; roundtrip_y = source_y if roundtrip_y is None else roundtrip_y
    _req_cols(s, [source_x, source_y], "source"); _req_cols(r, [roundtrip_x, roundtrip_y], "roundtrip")
    al = _align(s, r, key, True)
    def axis(a: Any, b: Any) -> dict[str, float]:
        aa, bb = pd.to_numeric(a, errors="coerce").to_numpy(float), pd.to_numeric(b, errors="coerce").to_numpy(float)
        ok = np.isfinite(aa) & np.isfinite(bb); cor = _safe_cor(aa, bb)
        if not ok.any(): return {"intercept": math.nan, "slope": math.nan, "correlation": math.nan, "error": math.nan}
        if ok.sum() >= 3 and np.std(aa[ok], ddof=1) > 0:
            sl, it = np.polyfit(aa[ok], bb[ok], 1); err = float(np.max(np.abs((it + sl * aa[ok]) - bb[ok])))
            return {"intercept": float(it), "slope": float(sl), "correlation": cor, "error": err}
        return {"intercept": math.nan, "slope": math.nan, "correlation": cor, "error": float(np.max(np.abs(aa[ok]-bb[ok])))}
    ax = axis(al["source"][source_x], al["roundtrip"][roundtrip_x]); ay = axis(al["source"][source_y], al["roundtrip"][roundtrip_y])
    sx = pd.to_numeric(al["source"][source_x], errors="coerce").to_numpy(float); rx = pd.to_numeric(al["roundtrip"][roundtrip_x], errors="coerce").to_numpy(float)
    sy = pd.to_numeric(al["source"][source_y], errors="coerce").to_numpy(float); ry = pd.to_numeric(al["roundtrip"][roundtrip_y], errors="coerce").to_numpy(float)
    okx = np.isfinite(sx)&np.isfinite(rx); oky=np.isfinite(sy)&np.isfinite(ry)
    direct = okx.any() and oky.any() and np.max(np.abs(sx[okx]-rx[okx])) <= tolerance and np.max(np.abs(sy[oky]-ry[oky])) <= tolerance
    transformed = all(np.isfinite([ax["correlation"], ay["correlation"], ax["error"], ay["error"]])) and ax["correlation"] >= correlation_floor and ay["correlation"] >= correlation_floor and ax["error"] <= tolerance and ay["error"] <= tolerance
    status = "LOSSLESS" if direct else "COORDINATE_TRANSFORMED" if transformed else "AMBIGUOUS"
    return _result("eye_coordinate_fidelity", status=status, x=ax, y=ay, matched_n=al["matched_n"], tolerance=float(tolerance), correlation_floor=float(correlation_floor))


def pupil_unit_fidelity_audit(source: Any, roundtrip: Any, source_pupil: str = "pupil_size",
                              roundtrip_pupil: str | None = None, key: str | Sequence[str] | None = None,
                              tolerance: float = 1e-6, correlation_floor: float = 0.995) -> EyeResult:
    s, r = _df(source, "source"), _df(roundtrip, "roundtrip"); roundtrip_pupil = source_pupil if roundtrip_pupil is None else roundtrip_pupil
    _req_cols(s,[source_pupil],"source"); _req_cols(r,[roundtrip_pupil],"roundtrip"); al=_align(s,r,key,True)
    a=pd.to_numeric(al["source"][source_pupil],errors="coerce").to_numpy(float); b=pd.to_numeric(al["roundtrip"][roundtrip_pupil],errors="coerce").to_numpy(float); ok=np.isfinite(a)&np.isfinite(b)
    cor=_safe_cor(a,b); ratio=b[ok]/a[ok]; ratio=ratio[np.isfinite(ratio)&(np.abs(a[ok])>np.finfo(float).eps)]; sr=float(np.median(ratio)) if ratio.size else math.nan
    direct=ok.any() and np.max(np.abs(a[ok]-b[ok]))<=tolerance; serr=float(np.max(np.abs(b[ok]-sr*a[ok]))) if np.isfinite(sr) and ok.any() else math.nan
    status="LOSSLESS" if direct else "UNIT_TRANSFORMED" if np.isfinite(cor) and cor>=correlation_floor and np.isfinite(serr) and serr<=tolerance else "AMBIGUOUS"
    return _result("eye_pupil_fidelity",status=status,matched_n=int(ok.sum()),correlation=cor,estimated_scale_ratio=sr,scaled_max_error=serr,tolerance=float(tolerance))


def eye_stream_fidelity_audit(source: Any, roundtrip: Any, source_eye: str = "eye", roundtrip_eye: str | None = None,
                              key: str | Sequence[str] | None = None) -> EyeResult:
    s,r=_df(source,"source"),_df(roundtrip,"roundtrip"); roundtrip_eye=source_eye if roundtrip_eye is None else roundtrip_eye
    _req_cols(s,[source_eye],"source"); _req_cols(r,[roundtrip_eye],"roundtrip"); al=_align(s,r,key,True)
    def norm_eye(x: pd.Series) -> pd.Series:
        z=x.astype("string").str.strip().str.lower(); return z.replace({"l":"left","left_eye":"left","lefteye":"left","r":"right","right_eye":"right","righteye":"right","both":"cyclopean","binocular":"cyclopean","combined":"cyclopean"})
    a,b=norm_eye(al["source"][source_eye]),norm_eye(al["roundtrip"][roundtrip_eye]); same=(a.eq(b)|(a.isna()&b.isna())).fillna(False)
    confusion=pd.crosstab(a,b,dropna=False); status="LOSSLESS" if bool(same.all()) and np.array_equal(a.isna().to_numpy(),b.isna().to_numpy()) else "AMBIGUOUS"
    return _result("eye_stream_fidelity",status=status,matched_n=al["matched_n"],source_streams=sorted(a.dropna().unique().tolist()),roundtrip_streams=sorted(b.dropna().unique().tolist()),confusion=confusion)


def event_semantics_audit(source_events: Any, roundtrip_events: Any, label: str = "event", time: str | None = "timestamp",
                          key: str | Sequence[str] | None = None, tolerance: float = 1e-6) -> EyeResult:
    s,r=_df(source_events,"source_events"),_df(roundtrip_events,"roundtrip_events"); _req_cols(s,[label],"source_events"); _req_cols(r,[label],"roundtrip_events"); al=_align(s,r,key,True)
    a,b=al["source"][label].astype("string"),al["roundtrip"][label].astype("string"); lab=(a.eq(b)|(a.isna()&b.isna())).fillna(False)
    terr=math.nan
    if time and time in al["source"] and time in al["roundtrip"]:
        ta=pd.to_numeric(al["source"][time],errors="coerce").to_numpy(float); tb=pd.to_numeric(al["roundtrip"][time],errors="coerce").to_numpy(float); ok=np.isfinite(ta)&np.isfinite(tb); terr=float(np.max(np.abs(ta[ok]-tb[ok]))) if ok.any() else math.nan
    if bool(lab.all()) and (not np.isfinite(terr) or terr<=tolerance) and al["matched_n"]==al["source_n"]==al["roundtrip_n"]: status="LOSSLESS"
    elif bool(lab.all()): status="SEMANTICALLY_EQUIVALENT"
    else: status="AMBIGUOUS"
    return _result("eye_event_semantics",status=status,source_n=al["source_n"],roundtrip_n=al["roundtrip_n"],matched_n=al["matched_n"],exact_label_fraction=float(lab.mean()) if len(lab) else math.nan,max_time_error=terr)


def validate_hed_event_semantics(events: Any, hed_column: str = "HED") -> pd.DataFrame:
    d=_df(events,"events"); _req_cols(d,[hed_column],"events"); x=d[hed_column].astype("string"); nonempty=x.notna() & x.str.strip().ne("")
    def balanced(z: str) -> bool:
        depth=0
        for ch in z:
            if ch=="(": depth+=1
            elif ch==")": depth-=1
            if depth<0: return False
        return depth==0
    bal=[balanced(str(v)) if bool(ne) else False for v,ne in zip(x,nonempty)]
    return pd.DataFrame({"row":np.arange(1,len(d)+1),"nonempty":nonempty.to_numpy(bool),"balanced_parentheses":bal,"structurally_valid":nonempty.to_numpy(bool)&np.asarray(bal,bool)})


def validate_bids_eye_semantics(data: Any, metadata: Mapping[str, Any], events_metadata: Mapping[str, Any] | None = None) -> EyeResult:
    d=_df(data,"data")
    if not isinstance(metadata,Mapping): raise EyeProcessValidationError("metadata must be a mapping.")
    rows=[]
    def add(name: str, pass_: bool, detail: Any): rows.append({"check":name,"pass":bool(pass_),"detail":str(detail)})
    req=["timestamp","x_coordinate","y_coordinate"]; add("required_columns",all(c in d for c in req),", ".join(c for c in req if c not in d)); add("column_order",list(d.columns[:3])==req,", ".join(map(str,d.columns[:3])))
    add("PhysioType",metadata.get("PhysioType")=="eyetrack",metadata.get("PhysioType")); eye=metadata.get("RecordedEye"); add("RecordedEye",eye in {"left","right","cyclopean"},eye); cs=metadata.get("SampleCoordinateSystem"); add("SampleCoordinateSystem",cs in {"gaze-on-screen","eye-in-head","gaze-in-world","custom"},cs)
    if cs=="gaze-on-screen":
        sp=(events_metadata or {}).get("StimulusPresentation") if isinstance(events_metadata,Mapping) else None; needed=["ScreenDistance","ScreenOrigin","ScreenResolution","ScreenSize"]; present=isinstance(sp,Mapping) and all(k in sp for k in needed); add("gaze_on_screen_stimulus_metadata",present,"complete" if present else "requires "+", ".join(needed))
    if "pupil_size" in d:
        pm=metadata.get("pupil_size"); units=pm.get("Units") if isinstance(pm,Mapping) else None; add("pupil_units_described",units is not None,units or "missing")
    tab=pd.DataFrame(rows); return _result("eye_bids_semantic_audit",valid=bool(tab["pass"].all()),checks=tab,metadata=dict(metadata))


def semantic_roundtrip_audit(source: Any, roundtrip: Any, key: str | Sequence[str] | None = None, fields: Sequence[str] | None = None,
                             timestamp: Mapping[str,Any] | None = None, coordinates: Mapping[str,Any] | None = None,
                             pupil: Mapping[str,Any] | None = None, eye: Mapping[str,Any] | None = None,
                             source_events: Any = None, roundtrip_events: Any = None, event_args: Mapping[str,Any] | None = None) -> EyeResult:
    field=field_fidelity_report(source,roundtrip,fields=fields,key=key)
    def attempt(fn,*args,**kwargs):
        try: return fn(*args,**kwargs)
        except Exception: return None
    tr=attempt(timestamp_fidelity_audit,source,roundtrip,key=key,**dict(timestamp or {})); cr=attempt(coordinate_fidelity_audit,source,roundtrip,key=key,**dict(coordinates or {})); pr=attempt(pupil_unit_fidelity_audit,source,roundtrip,key=key,**dict(pupil)) if pupil is not None else None; er=attempt(eye_stream_fidelity_audit,source,roundtrip,key=key,**dict(eye)) if eye is not None else None
    ev=attempt(event_semantics_audit,source_events,roundtrip_events,**dict(event_args or {})) if source_events is not None and roundtrip_events is not None else None
    fs=field.fields.status.tolist(); component={"fields":"LOSSLESS" if all(s=="LOSSLESS" for s in fs) else "SEMANTICALLY_EQUIVALENT" if all(s in {"LOSSLESS","SEMANTICALLY_EQUIVALENT","UNIT_TRANSFORMED","COORDINATE_TRANSFORMED"} for s in fs) else "AMBIGUOUS","timestamp":"UNSUPPORTED" if tr is None else tr.status,"coordinates":"UNSUPPORTED" if cr is None else cr.status}
    if pupil is not None: component["pupil"]="UNSUPPORTED" if pr is None else pr.status
    if eye is not None: component["eye"]="UNSUPPORTED" if er is None else er.status
    if source_events is not None and roundtrip_events is not None: component["events"]="UNSUPPORTED" if ev is None else ev.status
    good={"LOSSLESS","SEMANTICALLY_EQUIVALENT","UNIT_TRANSFORMED","COORDINATE_TRANSFORMED"}; vals=list(component.values()); overall="LOSSLESS" if all(v=="LOSSLESS" for v in vals) else "SEMANTICALLY_EQUIVALENT" if all(v in good for v in vals) else "AMBIGUOUS"
    return _result("eye_semantic_roundtrip",overall=overall,component_status=component,fields=field,timestamp=tr,coordinates=cr,pupil=pr,eye=er,events=ev)


def semantic_loss_map(x: Any) -> pd.DataFrame:
    if getattr(x,"eyeprocess_class",None)!="eye_semantic_roundtrip": raise EyeProcessValidationError("x must be an eye_semantic_roundtrip object.")
    comp=pd.DataFrame({"scope":"component","name":list(x.component_status),"status":list(x.component_status.values())}); comp["severity_rank"]=comp.status.map(_rank)
    fld=x.fields.fields[["field","status","severity_rank"]].rename(columns={"field":"name"}).copy(); fld.insert(0,"scope","field"); return pd.concat([comp,fld],ignore_index=True)


def public_validation_corpus() -> pd.DataFrame:
    return pd.DataFrame({
        "ecosystem":["Gazepoint","Gazepoint","EyeLink","EyeLink","EyeLink","Tobii","Tobii","Pupil Labs"],
        "device":["GP3 HD 150 Hz","GP3 HD 150 Hz","EyeLink (raw EDF; device metadata verify in corpus)","EyeLink 1000 1000 Hz","EyeLink Portable Duo 1000 Hz","Tobii Pro Fusion 120 Hz","Tobii Pro Glasses 3 ~50 Hz","Neon"],
        "corpus":["Pavia free observation of moving elements","Pavia symmetric dynamic stimuli (2026)","Raw eye-tracking data (EyeLink EDF; 10 subjects, 2026)","GazeBase","Eye movement benchmark data for smooth-pursuit classification (2026)","MCFW-Gaze v3","GroupAffect-4 v3","Pupil Labs official Neon sample recording"],
        "evidence_goal":[
            "independent-public-real; multisession-real; raw Gazepoint CSV",
            "independent-public-real; multisession-real; raw Gazepoint CSV",
            "raw-native-format; EDF parser validation",
            "large multisession longitudinal EyeLink benchmark",
            "raw EDF plus ASC conversion; event-parser benchmark",
            "remote-screen Tobii; binocular continuous raw gaze",
            "wearable Tobii; synchronized multimodal streams",
            "vendor-example; native/CSV semantic roundtrip",
        ],
        "url":[
            "https://vision.unipv.it/research/etanim/",
            "https://vision.unipv.it/research/etsymanim/",
            "https://zenodo.org/records/20780576",
            "https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257",
            "https://osf.io/zx7hc/",
            "https://zenodo.org/records/20300972",
            "https://zenodo.org/records/20796290",
            "https://docs.pupil-labs.com/neon/neon-player/getting-started/",
        ],
        "access":[
            "public","public; research/education/non-commercial terms","public","public CC BY 4.0",
            "public OSF","public Zenodo","public Zenodo; some audio restricted","vendor example",
        ],
        "auto_download":False,"review_terms_before_use":True,
    })


def compatibility_evidence_matrix(compatibility: Any, evidence: Any = None) -> pd.DataFrame:
    if isinstance(compatibility,Mapping) and not isinstance(compatibility,pd.DataFrame):
        compatibility=next((compatibility[k] for k in ("matrix","compatibility","data","table") if isinstance(compatibility.get(k),pd.DataFrame)),compatibility)
    comp=_df(compatibility,"compatibility")
    if evidence is None:
        out=comp.copy(); out["detailed_evidence_level"]="declared"; out["semantic_roundtrip_validated"]=False; out.attrs["eyeprocess_class"]="eye_compatibility_evidence_matrix"; return out
    ev=_df(evidence,"evidence"); _req_cols(ev,["ecosystem","device","evidence_level"],"evidence"); lv=validation_evidence_levels(); allowed=set(lv.level)
    if not set(ev.evidence_level).issubset(allowed): raise EyeProcessValidationError("Unknown evidence_level; use validation_evidence_levels().")
    rank=dict(zip(lv.level,lv["rank"])); ev=ev.copy(); ev[".rank"]=ev.evidence_level.map(rank); ev["semantic_roundtrip_pass"]=ev.get("semantic_roundtrip_pass",False)
    rows=[]
    for (eco,dev),z in ev.groupby(["ecosystem","device"],sort=False):
        best=z.loc[z[".rank"].idxmax()]; rows.append({"ecosystem":eco,"device":dev,"detailed_evidence_level":best.evidence_level,"semantic_roundtrip_validated":bool(z.semantic_roundtrip_pass.fillna(False).any()),"evidence_cases":len(z)})
    if not {"ecosystem","device"}.issubset(comp.columns): raise EyeProcessValidationError("compatibility must contain ecosystem and device columns.")
    out=comp.merge(pd.DataFrame(rows),on=["ecosystem","device"],how="left",sort=False); out["detailed_evidence_level"]=out.detailed_evidence_level.fillna("declared"); out["semantic_roundtrip_validated"]=out.semantic_roundtrip_validated.fillna(False).astype(bool); out["evidence_cases"]=out.evidence_cases.fillna(0).astype(int); out.attrs["eyeprocess_class"]="eye_compatibility_evidence_matrix"; return out


def validate_vendor_timestamp_semantics(data: Any, vendor: str, device_time: str | None = None,
                                        system_time: str | None = None, media_time: str | None = None) -> EyeResult:
    d=_df(data,"data")
    if not isinstance(vendor,str) or not vendor: raise EyeProcessValidationError("vendor must be one non-missing character value.")
    vendor_l=vendor.lower(); rows=[]
    def add(clock,col,expected,required=False):
        present=col is not None and col in d; mono=math.nan
        if present:
            z=pd.to_numeric(d[col],errors="coerce").dropna().to_numpy(float); mono=bool(len(z)<2 or np.all(np.diff(z)>=0))
        rows.append({"clock":clock,"column":col,"present":present,"monotonic":mono,"required":required,"expected_semantics":expected})
    if "tobii" in vendor_l:
        add("device",device_time,"device-origin timestamp preserved without replacement"); add("system",system_time,"host/system timestamp retained separately when exported")
    elif "pupil" in vendor_l or "neon" in vendor_l: add("native",device_time,"native high-resolution recording timestamp retained; unit/origin documented",True)
    elif "gazepoint" in vendor_l: add("native",device_time,"native monotonic recording clock retained",True); add("media",media_time,"media-relative clock retained separately when present")
    else: add("native",device_time,"native timestamp retained with unit and origin metadata",True); add("system",system_time,"system timestamp retained separately when available")
    tab=pd.DataFrame(rows); out=_result("eye_vendor_timestamp_semantics",vendor=vendor_l,clocks=tab); out["pass"]=bool((~tab.required | tab.present).all()); return out


def plot_eye_semantic_roundtrip(x: Any, ax: Any = None):
    import matplotlib.pyplot as plt
    if getattr(x,"eyeprocess_class",None)!="eye_semantic_roundtrip": raise EyeProcessValidationError("x must be an eye_semantic_roundtrip object.")
    loss=semantic_loss_map(x); ax=plt.subplots()[1] if ax is None else ax; score=10-np.minimum(loss.severity_rank.to_numpy(float),9); y=np.arange(len(loss)); ax.barh(y,score); ax.set_yticks(y,loss.name); ax.set_xlabel("Semantic fidelity (higher is better)"); ax.set_title(f"Semantic round trip: {x.overall}"); ax.gp3_data=loss; return ax


def plot_eye_compatibility_evidence_matrix(x: Any, ax: Any = None):
    import matplotlib.pyplot as plt
    d=_df(x,"x"); _req_cols(d,["detailed_evidence_level"],"x"); lv=validation_evidence_levels(); rank=dict(zip(lv.level,lv["rank"])); values=d.detailed_evidence_level.map(rank).to_numpy(float); labels=(d.ecosystem.astype(str)+" - "+d.device.astype(str)).tolist() if {"ecosystem","device"}.issubset(d.columns) else [str(i+1) for i in range(len(d))]; ax=plt.subplots()[1] if ax is None else ax; y=np.arange(len(d)); ax.barh(y,values); ax.set_yticks(y,labels); ax.set_xlabel("Detailed validation evidence level"); ax.gp3_data=d; return ax
