"""Process-measure registry, reliability, calibration uncertainty and gaze quality.

Ports the dependency-light contracts in R/074 and R/076 from eyeprocess 0.11.1.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .exceptions import EyeProcessValidationError
from .irt import EyeResult, _result, _stable_hash

__all__ = [
    "process_measure_registry", "validate_process_measure_registry", "register_process_measure",
    "find_process_measures", "process_measure_card", "process_measure_guardrails",
    "process_measure_coverage", "process_measure_lineage", "process_measure_units",
    "split_half_process_reliability", "process_icc", "process_bland_altman",
    "process_reliability_profile", "process_temporal_stability", "bootstrap_process_reliability",
    "estimate_calibration_error", "gaze_precision_rms_s2s", "effective_sampling_frequency",
    "audit_sampling_irregularity", "calibration_error_model", "gaze_uncertainty_ellipse",
    "propagate_calibration_uncertainty", "aoi_membership_probability", "probabilistic_aoi_assignment",
    "compare_hard_probabilistic_aoi", "calibration_sensitivity_grid", "fixation_boundary_uncertainty",
    "calibration_drift_profile", "gaze_data_quality_profile", "data_quality_reporting_table",
]

_REGISTRY_COLUMNS = ["name", "channel", "unit", "level", "interpretation", "guardrail", "status"]

_NAMES = [
    "fixation_count", "fixation_duration_mean", "first_fixation_duration", "dwell_time", "regression_count",
    "saccade_amplitude", "scanpath_length", "aoi_transition_count", "aoi_transition_entropy", "time_to_first_fixation",
    "pupil_mean", "pupil_peak", "pupil_baseline", "pupil_change", "pupil_velocity_activity",
    "response_time", "omission_indicator", "accuracy_indicator", "gaze_validity", "pupil_validity",
    "sampling_rate_effective", "calibration_error", "gaze_precision_rms_s2s", "data_loss",
    "process_profile_score", "streaming_theta", "process_anomaly_distance", "presentation_sensitivity_index",
]
_CHANNELS = [*["gaze"] * 10, *["pupil"] * 5, "behavior", "behavior", "behavior", *["quality"] * 6,
             "derived", "psychometric", "quality", "presentation"]
_UNITS = [
    "count", "ms", "ms", "ms", "count", "visual-angle/user units", "coordinate units", "count", "bits", "ms",
    "device/user units", "device/user units", "device/user units", "device/user units", "device/user units per time",
    "ms", "binary", "binary", "proportion", "proportion", "Hz", "coordinate/visual-angle units",
    "coordinate/visual-angle units", "proportion", "model-specific", "latent-score units", "distance", "index",
]
_LEVELS = [*["trial/AOI/person"] * 10, *["trial/person"] * 5, "trial", "trial", "trial", *["recording"] * 6,
           "person/trial", "person/step", "person", "trial/person"]
_INTERPRETATIONS = [
    "Observed count of classified fixations.", "Observed mean duration of classified fixations.",
    "Duration of the first classified fixation.", "Observed accumulated dwell within an AOI.",
    "Observed count of backward/revisit movements under the declared rule.",
    "Observed saccade amplitude under declared event detection.", "Observed spatial path length.",
    "Observed AOI-to-AOI transition count.", "Dispersion of observed AOI transitions.",
    "Latency to first qualifying fixation within an AOI.", "Observed pupil-size summary after declared preprocessing.",
    "Observed pupil maximum after declared preprocessing.", "Observed pre-event pupil baseline.",
    "Observed difference from the declared baseline.", "Observed pupil change/activity feature.",
    "Observed response latency.", "Observed response omission.", "Observed scoring indicator.",
    "Proportion of gaze samples meeting the declared validity rule.",
    "Proportion of pupil samples meeting the declared validity rule.",
    "Empirical sample frequency inferred from timestamps.",
    "Observed validation-target offset under the declared coordinate system.",
    "Successive-sample gaze dispersion metric.",
    "Proportion of expected/recorded gaze samples unavailable under the declared denominator.",
    "Model-derived process-profile coordinate.", "Partial latent score from a calibrated response model.",
    "Multivariate review statistic for process-feature unusualness.",
    "Observed sensitivity to presentation variants under the declared audit.",
]
_GENERAL_GUARD = "Do not interpret as attention, cognition, motivation, diagnosis, or intent without external construct-validity evidence."
_QUALITY_GUARD = "Quality metrics characterize measurement conditions and should not be converted into participant labels without a justified protocol."
_GUARDRAILS = [
    *[_GENERAL_GUARD] * 15,
    "Response time is behavioral timing, not a direct measure of latent cognition.",
    "Omission status does not by itself identify disengagement, misconduct, or inability.",
    "Accuracy is a scored response property; construct interpretation depends on the assessment design.",
    *[_QUALITY_GUARD] * 6,
    "Profile scores are model-derived summaries and require external validation before person-level interpretation.",
    "Streaming scores require calibrated banks and operational validation before consequential use.",
    "Large distances are review statistics, not misconduct/diagnostic labels.",
    "Presentation sensitivity is a measurement/presentation diagnostic, not a clinical accessibility classification.",
]
_STATUSES = [*["reference"] * 24, "experimental", "experimental", "reference", "experimental"]


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _req(d: pd.DataFrame, cols: Sequence[str | None], name: str = "data") -> None:
    needed = [str(c) for c in cols if c is not None]
    miss = [c for c in needed if c not in d.columns]
    if miss:
        raise EyeProcessValidationError(f"{name} is missing required columns: {', '.join(miss)}")


def _num(x: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)


def _q(x: Any, p: float) -> float:
    z = np.asarray(x, dtype=float)
    z = z[np.isfinite(z)]
    return float(np.quantile(z, p)) if z.size else math.nan


def _tag(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    out = df.copy()
    out.attrs["eyeprocess_class"] = cls
    return out


def process_measure_registry(include_experimental: bool = True) -> pd.DataFrame:
    out = pd.DataFrame({
        "name": _NAMES, "channel": _CHANNELS, "unit": _UNITS, "level": _LEVELS,
        "interpretation": _INTERPRETATIONS, "guardrail": _GUARDRAILS, "status": _STATUSES,
    })
    if not bool(include_experimental):
        out = out.loc[out.status != "experimental"].reset_index(drop=True)
    return _tag(out, "eye_process_measure_registry")


def validate_process_measure_registry(registry: Any) -> bool:
    reg = _df(registry, "registry")
    _req(reg, _REGISTRY_COLUMNS, "registry")
    if reg["name"].astype(str).duplicated().any():
        raise EyeProcessValidationError("Process registry contains duplicate measure names.")
    for c in _REGISTRY_COLUMNS:
        if reg[c].isna().any() or reg[c].astype(str).str.len().eq(0).any():
            raise EyeProcessValidationError("Every process measure requires non-missing name, channel, unit, level, interpretation, guardrail, and status metadata.")
    return True


def register_process_measure(
    registry: Any = None, name: Any = None, channel: Any = None, unit: Any = None, level: Any = None,
    interpretation: Any = None, guardrail: Any = None, status: Any = "user_defined",
) -> pd.DataFrame:
    reg = process_measure_registry() if registry is None else _df(registry, "registry")
    values = {"name": name, "channel": channel, "unit": unit, "level": level, "interpretation": interpretation, "guardrail": guardrail, "status": status}
    for value in values.values():
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)) and len(value) != 1:
            raise EyeProcessValidationError("Process-measure metadata fields must be scalar.")
    def scalar(v: Any) -> Any:
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            return list(v)[0] if len(v) else None
        return v
    row = pd.DataFrame([{k: scalar(v) for k, v in values.items()}])
    validate_process_measure_registry(row)
    reg = reg.loc[reg.name.astype(str) != str(row.name.iloc[0])]
    out = pd.concat([reg, row], ignore_index=True)
    validate_process_measure_registry(out)
    return _tag(out, "eye_process_measure_registry")


def find_process_measures(registry: Any = None, channel: Any = None, level: Any = None, status: Any = None, query: str | None = None) -> pd.DataFrame:
    reg = process_measure_registry() if registry is None else _df(registry, "registry")
    validate_process_measure_registry(reg)
    keep = pd.Series(True, index=reg.index)
    if channel is not None:
        vals = [channel] if isinstance(channel, str) else list(channel)
        keep &= reg.channel.isin(vals)
    if level is not None:
        vals = [level] if isinstance(level, str) else list(level)
        keep &= reg.level.astype(str).str.contains("|".join(map(str, vals)), case=False, regex=True)
    if status is not None:
        vals = [status] if isinstance(status, str) else list(status)
        keep &= reg.status.isin(vals)
    if query is not None:
        txt = reg[["name", "channel", "interpretation", "guardrail"]].astype(str).agg(" ".join, axis=1)
        keep &= txt.str.contains(str(query), case=False, regex=True)
    return reg.loc[keep].reset_index(drop=True)


def process_measure_card(name: str, registry: Any = None) -> EyeResult:
    reg = process_measure_registry() if registry is None else _df(registry, "registry")
    validate_process_measure_registry(reg)
    if not isinstance(name, str) or not name:
        raise EyeProcessValidationError("name must be one non-empty measure name.")
    hit = reg.loc[reg.name == name]
    if len(hit) != 1:
        raise EyeProcessValidationError(f"Expected exactly one registry entry for '{name}'.")
    return _result("eye_process_measure_card", **hit.iloc[0].to_dict())


def process_measure_guardrails(registry: Any = None) -> pd.DataFrame:
    reg = process_measure_registry() if registry is None else _df(registry, "registry")
    validate_process_measure_registry(reg)
    return reg[["name", "channel", "interpretation", "guardrail", "status"]].reset_index(drop=True)


def process_measure_coverage(data: Any, registry: Any = None) -> pd.DataFrame:
    d = _df(data); reg = process_measure_registry() if registry is None else _df(registry)
    validate_process_measure_registry(reg)
    rows=[]
    for name in reg.name.astype(str):
        present=name in d.columns
        fraction=float(d[name].notna().mean()) if present and len(d) else math.nan
        rows.append({"name":name,"registered":True,"present":present,"nonmissing_fraction":fraction})
    return pd.DataFrame(rows)


def process_measure_lineage(measure: Any, inputs: Any, transformations: Any = (), output_level: Any = None) -> EyeResult:
    inps=list(dict.fromkeys(map(str, [inputs] if isinstance(inputs,str) else inputs)))
    trans=list(map(str, [transformations] if isinstance(transformations,str) else transformations))
    out_level=None if output_level is None or pd.isna(output_level) else str(output_level)
    return _result("eye_process_measure_lineage", measure=str(measure), inputs=inps, transformations=trans,
                   output_level=out_level, lineage_hash=_stable_hash([measure,inps,trans,out_level]))


def process_measure_units(registry: Any = None) -> pd.DataFrame:
    reg=process_measure_registry() if registry is None else _df(registry)
    return reg[["channel","unit"]].drop_duplicates(ignore_index=True)


def _aggregate_measure(d: pd.DataFrame, person: str, session: str, measure: str) -> pd.DataFrame:
    z=d[[person,session,measure]].copy(); z[measure]=pd.to_numeric(z[measure],errors="coerce")
    return z.groupby([person,session],sort=False,dropna=False)[measure].mean().reset_index(name="value")


def split_half_process_reliability(data: Any, person: str, trial: str, measure: str, split: str = "odd_even", repetitions: int = 100, seed: int = 1, aggregate_fun: Callable = np.mean) -> pd.DataFrame:
    if split not in {"odd_even","random"}: raise EyeProcessValidationError("split must be 'odd_even' or 'random'.")
    d=_df(data).reset_index(drop=True); _req(d,[person,trial,measure])
    if not callable(aggregate_fun): raise EyeProcessValidationError("aggregate_fun must be a function.")
    repetitions=int(repetitions); seed=int(seed)
    if repetitions<1: raise EyeProcessValidationError("repetitions must be a positive integer.")
    if seed<0: raise EyeProcessValidationError("seed must be a finite non-negative scalar.")
    def compute(half: np.ndarray) -> tuple[float,float,int]:
        z=d.copy(); z[".half"]=half
        def aggfun(s: pd.Series) -> float:
            vals=pd.to_numeric(s,errors="coerce").dropna().to_numpy(float)
            if vals.size==0: return math.nan
            try: return float(aggregate_fun(vals))
            except TypeError: return float(aggregate_fun(vals, axis=0))
        a=z.groupby([person,".half"],sort=False)[measure].apply(aggfun).reset_index(name="value")
        h1=a.loc[a[".half"]==1,[person,"value"]].rename(columns={"value":"h1"})
        h2=a.loc[a[".half"]==2,[person,"value"]].rename(columns={"value":"h2"})
        m=h1.merge(h2,on=person)
        r=float(m[["h1","h2"]].corr().iloc[0,1]) if len(m)>=3 else math.nan
        sb=2*r/(1+r) if np.isfinite(r) and r>-1 else math.nan
        return r,sb,len(m)
    if split=="odd_even":
        cats=pd.Categorical(d[trial]); half=np.where((cats.codes+1)%2==1,1,2)
        r,sb,n=compute(half); return pd.DataFrame([{"split":"odd_even","replication":1,"raw_r":r,"spearman_brown":sb,"n_persons":n}])
    rng=np.random.default_rng(seed); rows=[]
    for rep in range(1,repetitions+1):
        half=np.zeros(len(d),dtype=int)
        for _, idx in d.groupby(person,sort=False).groups.items():
            idx=np.asarray(list(idx),dtype=int); vals=np.resize(np.array([1,2]),len(idx)); rng.shuffle(vals); half[idx]=vals
        r,sb,n=compute(half); rows.append({"split":"random","replication":rep,"raw_r":r,"spearman_brown":sb,"n_persons":n})
    return pd.DataFrame(rows)


def _icc_a1(Y: np.ndarray) -> dict[str,float]:
    Y=np.asarray(Y,float); Y=Y[np.isfinite(Y).all(axis=1)]; n,k=Y.shape if Y.ndim==2 else (0,0)
    if n<3 or k<2: return {"icc":math.nan,"n":n,"sessions":k,"MSR":math.nan,"MSC":math.nan,"MSE":math.nan}
    gm=Y.mean(); rowm=Y.mean(axis=1); colm=Y.mean(axis=0)
    msr=k*np.sum((rowm-gm)**2)/(n-1); msc=n*np.sum((colm-gm)**2)/(k-1)
    resid=Y-rowm[:,None]-colm[None,:]+gm; mse=np.sum(resid**2)/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n; icc=(msr-mse)/den if den!=0 else math.nan
    return {"icc":float(icc),"n":n,"sessions":k,"MSR":float(msr),"MSC":float(msc),"MSE":float(mse)}


def process_icc(data: Any, person: str, session: str, measure: str) -> pd.DataFrame:
    d=_df(data); _req(d,[person,session,measure]); agg=_aggregate_measure(d,person,session,measure)
    Y=agg.pivot(index=person,columns=session,values="value").to_numpy(float); z=_icc_a1(Y)
    return pd.DataFrame([{"icc_a1":z["icc"],"n_persons":z["n"],"n_sessions":z["sessions"],"ms_person":z["MSR"],"ms_session":z["MSC"],"ms_error":z["MSE"]}])


def process_bland_altman(data: Any, person: str, session: str, measure: str, sessions: Any = None) -> EyeResult:
    d=_df(data); _req(d,[person,session,measure]); sess=list(pd.unique(d[session].dropna())) if sessions is None else list(sessions)
    if len(sess)!=2 or any(pd.isna(v) for v in sess): raise EyeProcessValidationError("Bland-Altman summary requires exactly two non-missing sessions.")
    agg=_aggregate_measure(d.loc[d[session].isin(sess)],person,session,measure)
    a=agg.loc[agg[session]==sess[0],[person,"value"]].rename(columns={"value":"x_1"}); b=agg.loc[agg[session]==sess[1],[person,"value"]].rename(columns={"value":"x_2"}); m=a.merge(b,on=person)
    m["pair_mean"]=m[["x_1","x_2"]].mean(axis=1); m["difference"]=m.x_2-m.x_1
    v=m.difference.to_numpy(float); v=v[np.isfinite(v)]; bias=float(v.mean()) if len(v) else math.nan; sd=float(v.std(ddof=1)) if len(v)>=2 else math.nan
    summary=pd.DataFrame([{"n":len(m),"bias":bias,"sd_difference":sd,"loa_lower":bias-1.96*sd,"loa_upper":bias+1.96*sd}])
    return _result("eye_process_bland_altman",pairs=m,summary=summary,sessions=sess)


def process_reliability_profile(data: Any, person: str, session: str, measure: str) -> EyeResult:
    d=_df(data); _req(d,[person,session,measure]); icc=process_icc(d,person,session,measure); sess=list(pd.unique(d[session].dropna()))
    ba=process_bland_altman(d,person,session,measure,sess[:2]) if len(sess)>=2 else None
    return _result("eye_process_reliability_profile",measure=measure,icc=icc,bland_altman=ba,caveat="Reliability is design- and population-dependent; high reliability does not establish construct validity.")


def process_temporal_stability(data: Any, person: str, session: str, measure: str, method: str = "pearson") -> pd.DataFrame:
    if method not in {"pearson","spearman"}: raise EyeProcessValidationError("method must be 'pearson' or 'spearman'.")
    d=_df(data); _req(d,[person,session,measure]); agg=_aggregate_measure(d,person,session,measure); sess=list(pd.unique(agg[session].dropna())); rows=[]
    for i in range(len(sess)):
        for j in range(i+1,len(sess)):
            a=agg.loc[agg[session]==sess[i],[person,"value"]]; b=agg.loc[agg[session]==sess[j],[person,"value"]]; m=a.merge(b,on=person,suffixes=("_x","_y"))
            corr=float(m[["value_x","value_y"]].corr(method=method).iloc[0,1]) if len(m)>=3 else math.nan
            rows.append({"session_1":str(sess[i]),"session_2":str(sess[j]),"n":len(m),"correlation":corr})
    return pd.DataFrame(rows)


def bootstrap_process_reliability(data: Any, person: str, session: str, measure: str, replications: int = 500, seed: int = 1) -> pd.DataFrame:
    d=_df(data); _req(d,[person,session,measure]); replications=int(replications); seed=int(seed)
    if replications<1: raise EyeProcessValidationError("replications must be a positive integer.")
    if seed<0: raise EyeProcessValidationError("seed must be a finite non-negative scalar.")
    ids=list(pd.unique(d[person].dropna()));
    if not ids: raise EyeProcessValidationError("No non-missing participant identifiers are available.")
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(replications):
        samp=rng.choice(ids,size=len(ids),replace=True); chunks=[]
        for i,pid in enumerate(samp,1):
            z=d.loc[d[person]==pid].copy(); z[person]=f"B{i}"; chunks.append(z)
        boot=pd.concat(chunks,ignore_index=True); vals.append(float(process_icc(boot,person,session,measure).icc_a1.iloc[0]))
    finite=np.asarray(vals,float); finite=finite[np.isfinite(finite)]
    return pd.DataFrame([{"estimate":float(process_icc(d,person,session,measure).icc_a1.iloc[0]),"bootstrap_median":float(np.median(finite)) if len(finite) else math.nan,"lower":_q(finite,.025),"upper":_q(finite,.975),"replications":replications}])


def _groups(d: pd.DataFrame, by: Any) -> list[tuple[dict[str,Any],pd.DataFrame]]:
    if by is None or (not isinstance(by,str) and len(by)==0): return [({".group":"all"},d)]
    cols=[by] if isinstance(by,str) else list(by); _req(d,cols)
    return [({c:z.iloc[0][c] for c in cols},z) for _,z in d.groupby(cols,sort=True,dropna=False)]


def estimate_calibration_error(data: Any, gaze_x: str = "gaze_x", gaze_y: str = "gaze_y", target_x: str = "target_x", target_y: str = "target_y", by: Any = None) -> pd.DataFrame:
    d=_df(data); cols=[gaze_x,gaze_y,target_x,target_y]+([] if by is None else ([by] if isinstance(by,str) else list(by))); _req(d,cols); rows=[]
    for header,z in _groups(d,by):
        dx=_num(z[gaze_x])-_num(z[target_x]); dy=_num(z[gaze_y])-_num(z[target_y]); rad=np.sqrt(dx*dx+dy*dy); ok=np.isfinite(rad); dxf=dx[np.isfinite(dx)]; dyf=dy[np.isfinite(dy)]; rf=rad[ok]
        rows.append({**header,"n":int(ok.sum()),"bias_x":float(dxf.mean()) if len(dxf) else math.nan,"bias_y":float(dyf.mean()) if len(dyf) else math.nan,"mean_radial_error":float(rf.mean()) if len(rf) else math.nan,"median_radial_error":float(np.median(rf)) if len(rf) else math.nan,"rms_radial_error":float(np.sqrt(np.mean(rf**2))) if len(rf) else math.nan,"p95_radial_error":_q(rf,.95)})
    return pd.DataFrame(rows)


def gaze_precision_rms_s2s(data: Any, x: str = "gaze_x", y: str = "gaze_y", time: str | None = None, by: Any = None) -> pd.DataFrame:
    d=_df(data); cols=[x,y]+([] if time is None else [time])+([] if by is None else ([by] if isinstance(by,str) else list(by))); _req(d,cols); rows=[]
    for header,z in _groups(d,by):
        if time is not None: z=z.assign(_ord=pd.to_numeric(z[time],errors="coerce")).sort_values("_ord",kind="stable")
        xx=_num(z[x]); yy=_num(z[y]); step=np.sqrt(np.diff(xx)**2+np.diff(yy)**2); step=step[np.isfinite(step)]
        rows.append({**header,"n_steps":len(step),"rms_s2s":float(np.sqrt(np.mean(step**2))) if len(step) else math.nan,"median_s2s":float(np.median(step)) if len(step) else math.nan,"p95_s2s":_q(step,.95)})
    return pd.DataFrame(rows)


def effective_sampling_frequency(data: Any, time: str = "timestamp_ms", unit: str = "ms", by: Any = None) -> pd.DataFrame:
    if unit not in {"ms","s","us"}: raise EyeProcessValidationError("unit must be 'ms', 's', or 'us'.")
    scale={"ms":1000.0,"s":1.0,"us":1e6}[unit]; d=_df(data); cols=[time]+([] if by is None else ([by] if isinstance(by,str) else list(by))); _req(d,cols); rows=[]
    for header,z in _groups(d,by):
        tt=np.sort(_num(z[time])); dt=np.diff(tt); dt=dt[np.isfinite(dt)&(dt>0)]; med=float(np.median(dt)) if len(dt) else math.nan
        rows.append({**header,"n_intervals":len(dt),"median_interval":med,"effective_hz":scale/med if np.isfinite(med) and med>0 else math.nan,"interval_cv":float(np.std(dt,ddof=1)/np.mean(dt)) if len(dt)>1 and np.mean(dt)!=0 else math.nan})
    return pd.DataFrame(rows)


def audit_sampling_irregularity(data: Any, time: str = "timestamp_ms", unit: str = "ms", by: Any = None, cv_threshold: float = .05) -> EyeResult:
    cv_threshold=float(cv_threshold)
    if not np.isfinite(cv_threshold) or cv_threshold<0: raise EyeProcessValidationError("cv_threshold must be a finite non-negative scalar.")
    tab=effective_sampling_frequency(data,time,unit,by); tab["irregularity_flag"]=np.isfinite(tab.interval_cv)&(tab.interval_cv>cv_threshold)
    return _result("eye_sampling_irregularity_audit",table=tab,cv_threshold=cv_threshold,caveat="Sampling irregularity thresholds are workflow-specific review rules, not universal acceptability cutoffs.")


def calibration_error_model(data: Any, gaze_x: str = "gaze_x", gaze_y: str = "gaze_y", target_x: str = "target_x", target_y: str = "target_y") -> EyeResult:
    d=_df(data); _req(d,[gaze_x,gaze_y,target_x,target_y]); dx=_num(d[gaze_x])-_num(d[target_x]); dy=_num(d[gaze_y])-_num(d[target_y]); ok=np.isfinite(dx)&np.isfinite(dy); E=np.column_stack([dx[ok],dy[ok]])
    if len(E)<3: raise EyeProcessValidationError("At least three complete calibration-error pairs are required.")
    return _result("eye_calibration_error_model",mean_error=E.mean(axis=0),covariance=np.cov(E,rowvar=False,ddof=1),errors=E,n=len(E),metrics=estimate_calibration_error(d,gaze_x,gaze_y,target_x,target_y),coordinate_units="input_coordinate_units",status="empirical_calibration_error_model",caveat="The model approximates observed calibration/validation error and should be estimated in the coordinate system used for downstream AOIs.")


def gaze_uncertainty_ellipse(model: Any, level: float = .95, center: Any = None) -> pd.DataFrame:
    if not isinstance(model,Mapping) or getattr(model,"eyeprocess_class",None)!="eye_calibration_error_model": raise EyeProcessValidationError("model must be an eye_calibration_error_model.")
    level=float(level)
    if not 0<level<1: raise EyeProcessValidationError("level must lie in (0,1).")
    vals,vecs=np.linalg.eigh(np.asarray(model["covariance"],float)); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]; axes=np.sqrt(np.maximum(vals,0)*chi2.ppf(level,df=2)); angle=math.degrees(math.atan2(vecs[1,0],vecs[0,0])); c=np.asarray(model["mean_error"] if center is None else center,float)
    if c.size!=2 or not np.isfinite(c).all(): raise EyeProcessValidationError("center must contain two finite coordinates.")
    return pd.DataFrame([{"center_x":c[0],"center_y":c[1],"major_axis":float(np.max(axes)),"minor_axis":float(np.min(axes)),"angle_deg":angle,"level":level}])


def propagate_calibration_uncertainty(data: Any, model: Any, x: str = "gaze_x", y: str = "gaze_y", draws: int = 500, seed: int = 1) -> pd.DataFrame:
    if not isinstance(model,Mapping) or getattr(model,"eyeprocess_class",None)!="eye_calibration_error_model": raise EyeProcessValidationError("model must be an eye_calibration_error_model.")
    d=_df(data); _req(d,[x,y]); draws=int(draws); seed=int(seed)
    if draws<1: raise EyeProcessValidationError("draws must be a positive integer.")
    if seed<0: raise EyeProcessValidationError("seed must be a finite non-negative scalar.")
    S=np.asarray(model["covariance"],float)
    if not np.isfinite(S).all(): raise EyeProcessValidationError("Calibration covariance contains non-finite values.")
    vals,vecs=np.linalg.eigh(S); root=vecs@np.diag(np.sqrt(np.maximum(vals,0)))@vecs.T; rng=np.random.default_rng(seed); rows=[]
    for i,row in d.reset_index(drop=True).iterrows():
        z=rng.standard_normal((draws,2))@root + np.asarray(model["mean_error"],float)
        rows.append(pd.DataFrame({"sample_id":i+1,"draw_id":np.arange(1,draws+1),"gaze_x":float(pd.to_numeric(pd.Series([row[x]]),errors="coerce").iloc[0])-z[:,0],"gaze_y":float(pd.to_numeric(pd.Series([row[y]]),errors="coerce").iloc[0])-z[:,1]}))
    out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=["sample_id","draw_id","gaze_x","gaze_y"]); out.attrs.update({"eyeprocess_class":"eye_gaze_uncertainty_draws","model_n":model["n"],"draws":draws}); return out


def _aois(aois: Any) -> pd.DataFrame:
    a=_df(aois,"aois"); _req(a,["aoi","x_min","x_max","y_min","y_max"],"aois")
    if a.empty: raise EyeProcessValidationError("aois must contain at least one rectangle.")
    if a.aoi.isna().any() or a.aoi.astype(str).str.len().eq(0).any() or a.aoi.astype(str).duplicated().any(): raise EyeProcessValidationError("AOI names must be non-missing, non-empty, and unique.")
    for c in ["x_min","x_max","y_min","y_max"]: a[c]=pd.to_numeric(a[c],errors="coerce")
    if not np.isfinite(a[["x_min","x_max","y_min","y_max"]].to_numpy(float)).all() or (a.x_min>a.x_max).any() or (a.y_min>a.y_max).any(): raise EyeProcessValidationError("AOI rectangle bounds must be finite and ordered.")
    a["aoi"]=a.aoi.astype(str); return a


def aoi_membership_probability(draws: Any, aois: Any) -> pd.DataFrame:
    d=_df(draws,"draws"); _req(d,["sample_id","gaze_x","gaze_y"],"draws"); a=_aois(aois); rows=[]
    gx=pd.to_numeric(d.gaze_x,errors="coerce"); gy=pd.to_numeric(d.gaze_y,errors="coerce")
    complete=gx.notna() & gy.notna() & np.isfinite(gx.to_numpy(float)) & np.isfinite(gy.to_numpy(float))
    for _,r in a.iterrows():
        inside=pd.Series(pd.NA,index=d.index,dtype="boolean")
        valid_idx=inside.index[complete]
        inside.loc[valid_idx]=(gx.loc[valid_idx]>=r.x_min)&(gx.loc[valid_idx]<=r.x_max)&(gy.loc[valid_idx]>=r.y_min)&(gy.loc[valid_idx]<=r.y_max)
        for sid,z in pd.DataFrame({"sample_id":d.sample_id,"inside":inside}).groupby("sample_id",sort=True):
            vals=z.inside.dropna()
            rows.append({"sample_id":int(sid),"aoi":r.aoi,"probability":float(vals.astype(float).mean()) if len(vals) else math.nan})
    return pd.DataFrame(rows)


def probabilistic_aoi_assignment(data: Any, aois: Any, model: Any, x: str = "gaze_x", y: str = "gaze_y", draws: int = 500, seed: int = 1, min_probability: float = .5) -> EyeResult:
    min_probability=float(min_probability)
    if not 0<=min_probability<=1: raise EyeProcessValidationError("min_probability must lie in [0,1].")
    u=propagate_calibration_uncertainty(data,model,x,y,draws,seed); p=aoi_membership_probability(u,aois); rows=[]
    for sid,z in p.groupby("sample_id",sort=True):
        z=z.loc[np.isfinite(z.probability)].sort_values("probability",ascending=False)
        if z.empty: rows.append({"sample_id":sid,"aoi":pd.NA,"probability":math.nan,"ambiguity":math.nan}); continue
        best=z.iloc[0]; rows.append({"sample_id":sid,"aoi":best.aoi if best.probability>=min_probability else pd.NA,"probability":float(best.probability),"ambiguity":float(best.probability-z.iloc[1].probability) if len(z)>1 else float(best.probability)})
    return _result("eye_probabilistic_aoi_assignment",assignments=pd.DataFrame(rows),probabilities=p,aois=_df(aois),model=model,min_probability=min_probability,caveat="Probabilities quantify propagated calibration uncertainty under the fitted error model; they are not posterior probabilities of psychological attention.")


def compare_hard_probabilistic_aoi(data: Any, aois: Any, probabilistic: Any, x: str = "gaze_x", y: str = "gaze_y") -> pd.DataFrame:
    d=_df(data); a=_aois(aois); _req(d,[x,y])
    if not isinstance(probabilistic,Mapping) or getattr(probabilistic,"eyeprocess_class",None)!="eye_probabilistic_aoi_assignment": raise EyeProcessValidationError("probabilistic must be an eye_probabilistic_aoi_assignment.")
    hard=[]
    for _,r in d.iterrows():
        xx=pd.to_numeric(pd.Series([r[x]]),errors="coerce").iloc[0]; yy=pd.to_numeric(pd.Series([r[y]]),errors="coerce").iloc[0]; hit=a.loc[(xx>=a.x_min)&(xx<=a.x_max)&(yy>=a.y_min)&(yy<=a.y_max)] if np.isfinite(xx) and np.isfinite(yy) else a.iloc[0:0]; hard.append(hit.aoi.iloc[0] if len(hit) else pd.NA)
    out=pd.DataFrame({"sample_id":np.arange(1,len(d)+1),"hard_aoi":hard}).merge(probabilistic["assignments"].rename(columns={"aoi":"probabilistic_aoi"}),on="sample_id",how="left",sort=False)
    out["agreement"]=[(pd.isna(h)&pd.isna(p)) or (not pd.isna(h) and not pd.isna(p) and h==p) for h,p in zip(out.hard_aoi,out.probabilistic_aoi)]; return out


def calibration_sensitivity_grid(offset_x: Any = (-.02,0,.02), offset_y: Any = (-.02,0,.02)) -> pd.DataFrame:
    x=np.atleast_1d(np.asarray(offset_x,float)); y=np.atleast_1d(np.asarray(offset_y,float))
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all(): raise EyeProcessValidationError("offset_x and offset_y must contain finite values.")
    rows=[]; k=1
    for yy in y:
        for xx in x:
            rows.append({"calibration_spec_id":f"CAL{k:03d}","offset_x":xx,"offset_y":yy,"radial_offset":float(math.hypot(xx,yy))}); k+=1
    return pd.DataFrame(rows)


def fixation_boundary_uncertainty(data: Any, aois: Any, x: str = "gaze_x", y: str = "gaze_y") -> pd.DataFrame:
    d=_df(data); a=_aois(aois); _req(d,[x,y]); rows=[]
    for i,r in d.reset_index(drop=True).iterrows():
        xx=pd.to_numeric(pd.Series([r[x]]),errors="coerce").iloc[0]; yy=pd.to_numeric(pd.Series([r[y]]),errors="coerce").iloc[0]
        if not np.isfinite(xx) or not np.isfinite(yy): rows.append({"sample_id":i+1,"nearest_aoi":pd.NA,"signed_boundary_distance":math.nan}); continue
        vals=[]
        for _,q in a.iterrows():
            dx=max(q.x_min-xx,0,xx-q.x_max); dy=max(q.y_min-yy,0,yy-q.y_max); dist=min(xx-q.x_min,q.x_max-xx,yy-q.y_min,q.y_max-yy) if dx==0 and dy==0 else -math.hypot(dx,dy); vals.append(dist)
        j=int(np.argmax(vals)); rows.append({"sample_id":i+1,"nearest_aoi":a.iloc[j].aoi,"signed_boundary_distance":vals[j]})
    return pd.DataFrame(rows)


def calibration_drift_profile(data: Any, by: Any, gaze_x: str = "gaze_x", gaze_y: str = "gaze_y", target_x: str = "target_x", target_y: str = "target_y") -> EyeResult:
    tab=estimate_calibration_error(data,gaze_x,gaze_y,target_x,target_y,by)
    if tab.empty: raise EyeProcessValidationError("No calibration groups are available for drift profiling.")
    tab["baseline_error"]=float(tab.mean_radial_error.iloc[0]); tab["delta_from_first"]=tab.mean_radial_error-tab.baseline_error
    return _result("eye_calibration_drift_profile",table=tab,by=by,caveat="Observed drift may reflect calibration, participant movement, geometry, lighting, hardware, or other acquisition changes.")


def gaze_data_quality_profile(data: Any, x: str = "gaze_x", y: str = "gaze_y", time: str = "timestamp_ms", target_x: str | None = None, target_y: str | None = None, valid: str | None = None, by: Any = None, time_unit: str = "ms") -> EyeResult:
    d=_df(data); cols=[x,y,time]+([] if valid is None else [valid])+([] if by is None else ([by] if isinstance(by,str) else list(by))); _req(d,cols)
    if (target_x is None)!=(target_y is None): raise EyeProcessValidationError("target_x and target_y must be supplied together.")
    if d.empty: return _result("eye_data_quality_profile",table=pd.DataFrame(),coordinate_units="input_coordinate_units",caveat="No samples were supplied; data-quality metrics are undefined.")
    rows=[]
    for header,z in _groups(d,by):
        xx=_num(z[x]); yy=_num(z[y]); complete=np.isfinite(xx)&np.isfinite(yy)
        if valid is not None:
            vv=z[valid].astype("boolean").fillna(False).to_numpy(bool); valid_vec=vv&complete
        else: valid_vec=complete
        pr=gaze_precision_rms_s2s(z.loc[valid_vec],x,y,time,None); ef=effective_sampling_frequency(z,time,time_unit,None)
        acc=estimate_calibration_error(z,x,y,target_x,target_y,None) if target_x is not None and target_x in z and target_y in z else None
        vf=float(valid_vec.mean()) if len(valid_vec) else math.nan
        rows.append({**header,"n_samples":len(z),"valid_fraction":vf,"data_loss":1-vf,"effective_hz":float(ef.effective_hz.iloc[0]),"sampling_interval_cv":float(ef.interval_cv.iloc[0]),"rms_s2s":float(pr.rms_s2s.iloc[0]) if len(pr) else math.nan,"mean_radial_error":float(acc.mean_radial_error.iloc[0]) if acc is not None and len(acc) else math.nan})
    return _result("eye_data_quality_profile",table=pd.DataFrame(rows),coordinate_units="input_coordinate_units",caveat="Data-quality metrics should be reported with their operational definitions and acquisition context. Whether quality is sufficient depends on the scientific question and analysis resolution.")


def data_quality_reporting_table(x: Any) -> pd.DataFrame:
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_data_quality_profile": raise EyeProcessValidationError("x must be an eye_data_quality_profile.")
    return x["table"].copy()
