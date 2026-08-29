"""eyeprocess 0.8 process governance, temporal windows, and advanced pupillometry.

Ports the public contracts from frozen R files 058--061 of eyeprocess 0.11.1.
Review flags are measurement/data-quality signals and never psychological,
clinical, misconduct, motivation, or ability labels.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _result

__all__ = [
    # R/058
    "process_preflight_spec", "audit_biometric_preflight", "preflight_decisions",
    "preflight_failures", "preflight_passed", "preflight_exclusion_manifest",
    "apply_preflight_decision", "audit_process_anomalies", "audit_multivariate_process_quality",
    "process_anomaly_distance", "audit_presentation_accessibility",
    "simulate_presentation_variants", "compare_presentation_fairness",
    # R/059
    "process_drift_spec", "audit_process_drift", "process_drift_alerts",
    "compare_deployment_batches", "drift_by_device", "drift_by_site", "drift_by_vendor",
    "drift_by_stimulus_version",
    # R/060
    "process_window_spec", "extract_process_windows", "summarize_process_windows",
    "bind_process_windows", "validate_process_windows", "audit_process_window_sensitivity",
    "aoi_trajectory_features", "fit_aoi_growth_curve", "predict_aoi_trajectory",
    "compare_aoi_trajectories",
    # R/061
    "pupil_band_power", "pupil_velocity_activity", "pupil_activity_index",
    "pupil_frequency_features", "audit_pupil_frequency_stability", "pupil_response_kernel",
    "pupil_event_regressor", "fit_pupil_event_deconvolution", "pupil_event_effects",
    "compare_pupil_kernels", "fit_pupil_confound_model", "adjust_pupil_confounds",
    "pupil_confound_effects", "audit_pupil_fatigue_drift", "compare_raw_adjusted_pupil",
    "filter_eye_signal", "filter_pupil_signal", "audit_signal_filter", "compare_signal_filters",
]


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _req(d: pd.DataFrame, cols: Sequence[str | None], name: str = "data") -> None:
    needed = [str(c) for c in cols if c is not None and str(c)]
    miss = [c for c in needed if c not in d.columns]
    if miss:
        raise EyeProcessValidationError(f"{name} is missing required column(s): {', '.join(miss)}")


def _num(x: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)


def _mean(x: Any) -> float:
    a = _num(x); a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else math.nan


def _sd(x: Any) -> float:
    a = _num(x); a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if a.size >= 2 else math.nan


def _z(x: Any) -> np.ndarray:
    a = _num(x)
    good = np.isfinite(a)
    if not good.any(): return np.full(len(a), np.nan)
    s = _sd(a)
    if not np.isfinite(s) or s == 0: return np.zeros(len(a), dtype=float)
    return (a - np.nanmean(a)) / s


def _groups(d: pd.DataFrame, by: Sequence[str]) -> list[np.ndarray]:
    _req(d, by)
    if d.empty: raise EyeProcessValidationError("data must contain at least one row.")
    if by and d[list(by)].isna().any().any():
        raise EyeProcessValidationError("Grouping columns must not contain missing values; repair or explicitly label missing IDs before analysis.")
    if not by: return [np.arange(len(d))]
    return [np.asarray(v, dtype=int) for v in d.groupby(list(by), sort=True, dropna=False).indices.values()]


def _group_values(d: pd.DataFrame, idx: np.ndarray, by: Sequence[str]) -> dict[str, Any]:
    if not by: return {".group": "all"}
    return {c: d.iloc[int(idx[0])][c] for c in by}


def _as_bool(x: Any) -> np.ndarray:
    s = pd.Series(x).astype("boolean").fillna(False)
    return s.to_numpy(dtype=bool)


def _class(obj: Any, cls: str) -> bool:
    return isinstance(obj, EyeResult) and getattr(obj, "eyeprocess_class", None) == cls


def _ols(y: np.ndarray, X: pd.DataFrame) -> EyeResult:
    y = np.asarray(y, float)
    Xn = X.copy()
    Xn.insert(0, "(Intercept)", 1.0)
    A = Xn.to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ beta
    resid = y - fitted
    n, p = A.shape
    sigma2 = float(np.sum(resid**2) / max(n - p, 1))
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * stats.t.sf(np.abs(t), df=max(n-p, 1))
    coef = pd.DataFrame({"estimate": beta, "std_error": se, "t_value": t, "p_value": pval}, index=Xn.columns)
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1 - float(np.sum(resid**2))/ss_tot if ss_tot > 0 else math.nan
    return _result("eye_linear_model_reference", coefficients=coef, fitted=fitted, residuals=resid,
                   design_columns=list(Xn.columns), sigma2=sigma2, r_squared=r2)


# ---------------------------------------------------------------------------
# R/058 process pre-flight and anomaly governance
# ---------------------------------------------------------------------------

def process_preflight_spec(min_gaze_validity: float = 0.80, min_pupil_validity: float = 0.70,
                           max_gaze_missingness: float = 0.25, max_pupil_missingness: float = 0.30,
                           min_valid_trial_fraction: float = 0.70, trial_gaze_validity_threshold: float = 0.75,
                           min_rt_ms: float = 200, max_rt_ms: float = 10000,
                           sampling_rate_tolerance: float = 0.20, blink_quantile: float = 0.95,
                           caution_flags: int = 1, review_flags: int = 2) -> EyeResult:
    props = np.asarray([min_gaze_validity,min_pupil_validity,max_gaze_missingness,max_pupil_missingness,
                        min_valid_trial_fraction,trial_gaze_validity_threshold,sampling_rate_tolerance,blink_quantile], float)
    if not np.all(np.isfinite(props)): raise EyeProcessValidationError("All proportion/threshold values must be finite.")
    if np.any((props[:6] < 0) | (props[:6] > 1)): raise EyeProcessValidationError("Validity/missingness proportions must lie in [0, 1].")
    if sampling_rate_tolerance < 0: raise EyeProcessValidationError("sampling_rate_tolerance must be non-negative.")
    if not 0 < blink_quantile < 1: raise EyeProcessValidationError("blink_quantile must lie strictly between 0 and 1.")
    if not (np.isfinite(min_rt_ms) and np.isfinite(max_rt_ms) and min_rt_ms < max_rt_ms): raise EyeProcessValidationError("min_rt_ms must be smaller than max_rt_ms.")
    caution_flags, review_flags = int(caution_flags), int(review_flags)
    if caution_flags < 1 or review_flags <= caution_flags: raise EyeProcessValidationError("Require 1 <= caution_flags < review_flags.")
    return _result("eye_process_preflight_spec", min_gaze_validity=min_gaze_validity,
        min_pupil_validity=min_pupil_validity,max_gaze_missingness=max_gaze_missingness,
        max_pupil_missingness=max_pupil_missingness,min_valid_trial_fraction=min_valid_trial_fraction,
        trial_gaze_validity_threshold=trial_gaze_validity_threshold,min_rt_ms=min_rt_ms,max_rt_ms=max_rt_ms,
        sampling_rate_tolerance=sampling_rate_tolerance,blink_quantile=blink_quantile,
        caution_flags=caution_flags,review_flags=review_flags,
        interpretation="Pre-flight thresholds protect downstream biometric/process analyses. Flags indicate review needs, not behavioral, clinical, or ability labels.")


def audit_biometric_preflight(data: Any, by: Sequence[str] = ("person_id",), spec: Any = None,
                              valid_gaze_prop: str = "valid_gaze_prop", valid_pupil_prop: str = "valid_pupil_prop",
                              missing_gaze: str = "missing_gaze", missing_pupil: str = "missing_pupil",
                              rt_ms: str = "rt_ms", blink_cluster_count: str = "blink_cluster_count",
                              sampling_rate_hz: str = "sampling_rate_hz") -> EyeResult:
    d=_df(data); by=list(by); _req(d,by); spec=process_preflight_spec() if spec is None else spec
    if not _class(spec,"eye_process_preflight_spec"): raise EyeProcessValidationError("spec must be created by process_preflight_spec().")
    rows=[]
    for idx in _groups(d,by):
        z=d.iloc[idx]
        def col(name: str, default=np.nan): return z[name] if name in z else np.repeat(default,len(z))
        gv,pv=_num(col(valid_gaze_prop)),_num(col(valid_pupil_prop))
        mg=col(missing_gaze); mp=col(missing_pupil)
        mg=_as_bool(mg).astype(float) if pd.api.types.is_bool_dtype(pd.Series(mg).dtype) else _num(mg)
        mp=_as_bool(mp).astype(float) if pd.api.types.is_bool_dtype(pd.Series(mp).dtype) else _num(mp)
        rt,bl,sr=_num(col(rt_ms)),_num(col(blink_cluster_count)),_num(col(sampling_rate_hz))
        finite=np.isfinite(gv); acceptable=finite & (gv>=spec.trial_gaze_validity_threshold)
        row={**_group_values(d,idx,by),"n_rows":len(idx),"mean_valid_gaze_prop":_mean(gv),
             "mean_valid_pupil_prop":_mean(pv),"mean_missing_gaze":_mean(mg),"mean_missing_pupil":_mean(mp),
             "mean_rt_ms":_mean(rt),"mean_blink_cluster_count":_mean(bl),"mean_sampling_rate_hz":_mean(sr),
             "proportion_trials_with_acceptable_gaze":float(np.mean(acceptable[finite])) if finite.any() else math.nan}
        rows.append(row)
    tab=pd.DataFrame(rows)
    blink_vals=tab.mean_blink_cluster_count.to_numpy(float); blink_vals=blink_vals[np.isfinite(blink_vals)]
    blink_cutoff=float(np.quantile(blink_vals,spec.blink_quantile)) if blink_vals.size else math.nan
    srvals=tab.mean_sampling_rate_hz.to_numpy(float); srvals=srvals[np.isfinite(srvals)]
    target=float(np.median(srvals)) if srvals.size else math.nan
    tab["low_valid_gaze_flag"]=np.isfinite(tab.mean_valid_gaze_prop)&(tab.mean_valid_gaze_prop<spec.min_gaze_validity)
    tab["low_valid_pupil_flag"]=np.isfinite(tab.mean_valid_pupil_prop)&(tab.mean_valid_pupil_prop<spec.min_pupil_validity)
    tab["excessive_gaze_missingness_flag"]=np.isfinite(tab.mean_missing_gaze)&(tab.mean_missing_gaze>spec.max_gaze_missingness)
    tab["excessive_pupil_missingness_flag"]=np.isfinite(tab.mean_missing_pupil)&(tab.mean_missing_pupil>spec.max_pupil_missingness)
    tab["implausible_rt_flag"]=np.isfinite(tab.mean_rt_ms)&((tab.mean_rt_ms<spec.min_rt_ms)|(tab.mean_rt_ms>spec.max_rt_ms))
    tab["too_few_valid_trials_flag"]=np.isfinite(tab.proportion_trials_with_acceptable_gaze)&(tab.proportion_trials_with_acceptable_gaze<spec.min_valid_trial_fraction)
    tab["extreme_blink_cluster_flag"]=np.isfinite(blink_cutoff)&np.isfinite(tab.mean_blink_cluster_count)&(tab.mean_blink_cluster_count>blink_cutoff)
    tab["sampling_rate_instability_flag"]=np.isfinite(target)&(target>0)&np.isfinite(tab.mean_sampling_rate_hz)&(np.abs(tab.mean_sampling_rate_hz-target)>spec.sampling_rate_tolerance*target)
    flags=[c for c in tab if c.endswith("_flag")]
    tab["preflight_flag_count"]=tab[flags].sum(axis=1).astype(int)
    tab["preflight_decision"]=np.where(tab.preflight_flag_count>=spec.review_flags,"review_or_exclude_from_biometric_models",
                                np.where(tab.preflight_flag_count>=spec.caution_flags,"use_with_caution","pass_preflight"))
    return _result("eye_biometric_preflight",table=tab,flag_columns=flags,by=by,spec=spec,
                   blink_cutoff=blink_cutoff,target_sampling_rate_hz=target,source_n=len(d),
                   interpretation="Pre-flight decisions are quality-governance recommendations. They must not be interpreted as evidence about motivation, cheating, diagnosis, or ability.")


def preflight_decisions(x: Any) -> pd.DataFrame:
    if not _class(x,"eye_biometric_preflight"): raise EyeProcessValidationError("x must be an eye_biometric_preflight object.")
    return x.table.copy()

def preflight_failures(x: Any) -> pd.DataFrame:
    d=preflight_decisions(x); return d.loc[d.preflight_decision!="pass_preflight"].reset_index(drop=True)

def preflight_passed(x: Any) -> pd.DataFrame:
    d=preflight_decisions(x); return d.loc[d.preflight_decision=="pass_preflight"].reset_index(drop=True)

def preflight_exclusion_manifest(x: Any) -> pd.DataFrame:
    d=preflight_decisions(x); keep=[*x.by,*x.flag_columns,"preflight_flag_count","preflight_decision"]
    out=d[keep].copy(); out["recommended_action"]=np.where(out.preflight_decision=="review_or_exclude_from_biometric_models","manual_review_before_biometric_model_inclusion",np.where(out.preflight_decision=="use_with_caution","retain_with_sensitivity_analysis","retain")); return out

def apply_preflight_decision(data: Any, audit: Any, keep_decisions: Sequence[str] = ("pass_preflight","use_with_caution")) -> pd.DataFrame:
    d=_df(data)
    if not _class(audit,"eye_biometric_preflight"): raise EyeProcessValidationError("audit must be eye_biometric_preflight.")
    _req(d,audit.by)
    if d[audit.by].isna().any().any(): raise EyeProcessValidationError("Grouping columns used by the pre-flight audit must not contain missing values.")
    d[".ep08_original_row"]=np.arange(len(d)); tab=audit.table[[*audit.by,"preflight_decision"]]
    m=d.merge(tab,on=audit.by,how="left",sort=False).sort_values(".ep08_original_row")
    if m.preflight_decision.isna().any(): raise EyeProcessValidationError("Some rows could not be matched to a pre-flight decision; ensure the grouping identifiers match the audited data.")
    out=m.loc[m.preflight_decision.isin(keep_decisions)].drop(columns=".ep08_original_row").reset_index(drop=True)
    out.attrs["preflight_application"]={"keep_decisions":list(keep_decisions),"n_input":len(d),"n_output":len(out),"caution":"Rows were filtered only because apply_preflight_decision() was explicitly called."}; return out


def audit_process_anomalies(data: Any, person: str = "person_id", metrics: Sequence[str] | None = None,
                            alpha: float = 0.975, aggregate: bool = True, ridge: float = 1e-6) -> EyeResult:
    d=_df(data); _req(d,[person])
    if not 0<alpha<1: raise EyeProcessValidationError("alpha must be in (0,1).")
    if metrics is None:
        metrics=[c for c in d.select_dtypes(include=np.number).columns if c!=person and np.isfinite(_num(d[c])).sum()>=10 and np.isfinite(_sd(d[c])) and _sd(d[c])>0]
    metrics=[m for m in metrics if m in d]
    if len(metrics)<2: raise EyeProcessValidationError("At least two usable numeric metrics are required.")
    q=d[[person,*metrics]].copy(); q[metrics]=q[metrics].apply(pd.to_numeric,errors="coerce")
    if aggregate: q=q.groupby(person,sort=True,as_index=False)[metrics].mean()
    X=q[metrics].to_numpy(float)
    for j in range(X.shape[1]):
        mu=np.nanmean(X[:,j]) if np.isfinite(X[:,j]).any() else 0.0; X[~np.isfinite(X[:,j]),j]=mu
    center=X.mean(axis=0); V=np.cov(X,rowvar=False)
    if np.ndim(V)!=2 or not np.all(np.isfinite(V)): raise EyeProcessValidationError("Could not estimate a finite covariance matrix.")
    V=V+np.eye(V.shape[0])*ridge; diff=X-center; dist=np.einsum("ij,jk,ik->i",diff,np.linalg.pinv(V),diff)
    threshold=float(stats.chi2.ppf(alpha,df=X.shape[1])); q["mahalanobis_process_distance"]=dist; q["review_threshold"]=threshold; q["review_required"]=dist>threshold
    q["review_label"]=np.where(q.review_required,"review_required_process_or_data_quality_anomaly","no_multivariate_process_flag"); q=q.sort_values("mahalanobis_process_distance",ascending=False).reset_index(drop=True)
    return _result("eye_process_anomaly_audit",table=q,metrics=list(metrics),alpha=alpha,threshold=threshold,center=center,covariance=V,caveat="Multivariate distance is a process/data-quality review statistic. It is not evidence of cheating, spoofing, diagnosis, motivation, or intent.")

def audit_multivariate_process_quality(*args: Any, **kwargs: Any) -> EyeResult: return audit_process_anomalies(*args,**kwargs)

def process_anomaly_distance(x: Any) -> pd.DataFrame:
    if not _class(x,"eye_process_anomaly_audit"): raise EyeProcessValidationError("x must be an eye_process_anomaly_audit object.")
    cols=[c for c in x.table.columns if c not in x.metrics]+list(x.metrics); return x.table[cols].copy()


def audit_presentation_accessibility(data: Any, person: str="person_id", rt: str="rt_ms", dwell: str="dwell_ms", revisits: str="revisits", entropy: str="aoi_entropy", pupil: str="pupil_peak", gaze_validity: str="valid_gaze_prop", review_quantile: float=.90) -> EyeResult:
    d=_df(data); _req(d,[person])
    if not 0<review_quantile<1: raise EyeProcessValidationError("review_quantile must be in (0,1).")
    cols=[c for c in [rt,dwell,revisits,entropy,pupil,gaze_validity] if c in d]
    if len(cols)<3: raise EyeProcessValidationError("At least three process columns are required.")
    q=d[[person,*cols]].copy(); q[cols]=q[cols].apply(pd.to_numeric,errors="coerce"); agg=q.groupby(person,sort=True,as_index=False)[cols].mean()
    get=lambda c: _num(agg[c]) if c in agg else np.full(len(agg),np.nan)
    reading=_z(get(rt))+_z(get(dwell))+_z(get(revisits)); search=_z(get(entropy))-_z(get(gaze_validity)); phys=_z(get(pupil)); comp=np.column_stack([reading,search,phys]); score=np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in comp])
    if not np.isfinite(score).any(): raise EyeProcessValidationError("No finite presentation-sensitivity scores could be computed.")
    threshold=float(np.nanquantile(score,review_quantile)); agg["reading_effort_proxy"]=reading; agg["visual_search_instability_proxy"]=search; agg["physiological_load_proxy"]=phys; agg["presentation_sensitivity_index"]=score; agg["presentation_review_flag"]=np.isfinite(score)&(score>=threshold); agg["interpretation_label"]=np.where(agg.presentation_review_flag,"presentation_accessibility_review_not_clinical_diagnosis","no_presentation_review_flag")
    return _result("eye_presentation_accessibility",table=agg,threshold=threshold,review_quantile=review_quantile,status="experimental_design_audit",caveat="This audit evaluates presentation/accessibility sensitivity only. It must not be used to infer ADHD, dyslexia, visual impairment, neurodivergence, or another diagnosis.")

def simulate_presentation_variants(audit: Any, line_spacing_multiplier: float=1.25, key_term_highlighting: bool=True) -> pd.DataFrame:
    if not _class(audit,"eye_presentation_accessibility"): raise EyeProcessValidationError("audit must be created by audit_presentation_accessibility().")
    t=audit.table.copy(); f=t.presentation_review_flag.astype(bool); t["simulated_variant"]=np.where(f,"calibrated_accessibility_variant_for_review","standard_presentation"); t["simulated_line_spacing_multiplier"]=np.where(f,line_spacing_multiplier,1.0); t["simulated_key_term_highlighting"]=f & bool(key_term_highlighting); t["operational_status"]="simulation_only_requires_calibration_and_fairness_testing"; return t

def compare_presentation_fairness(data: Any, variant: str, outcome: str, person: str | None=None) -> EyeResult:
    d=_df(data); _req(d,[variant,outcome]+([person] if person else [])); q=d[[variant,outcome]+([person] if person else [])].copy(); q[outcome]=pd.to_numeric(q[outcome],errors="coerce"); q=q.dropna(subset=[variant,outcome])
    if person: q=q.groupby([person,variant],sort=True,as_index=False)[outcome].mean()
    if q[variant].nunique()<2: raise EyeProcessValidationError("At least two presentation variants are required.")
    levels=list(pd.unique(q[variant])); X=pd.get_dummies(q[variant],drop_first=True,dtype=float); model=_ols(q[outcome].to_numpy(float),X)
    summ=q.groupby(variant,sort=True)[outcome].agg(["count","mean","std"]).reset_index().rename(columns={"count":"n","std":"sd"})
    return _result("eye_presentation_fairness_comparison",model=model,summary=summ,status="descriptive_fairness_sensitivity",caveat="Presentation differences require calibrated designs and substantive fairness interpretation; this comparison is not causal by itself.")


# ---------------------------------------------------------------------------
# R/059 deployment drift
# ---------------------------------------------------------------------------

def process_drift_spec(baseline: str="first_batch", difficulty_limit: float=.40, discrimination_limit: float=.35, gaze_validity_drop: float=.10, luminance_limit: float=25, relative_metric_quantile: float=.90, min_batches: int=2) -> EyeResult:
    if baseline not in {"first_batch","reference_batch"}: raise EyeProcessValidationError("baseline must be 'first_batch' or 'reference_batch'.")
    vals=np.asarray([difficulty_limit,discrimination_limit,gaze_validity_drop,luminance_limit],float)
    if not np.all(np.isfinite(vals)) or np.any(vals<0): raise EyeProcessValidationError("Drift limits must be finite and non-negative.")
    if not 0<relative_metric_quantile<1: raise EyeProcessValidationError("relative_metric_quantile must be in (0,1).")
    if int(min_batches)<2: raise EyeProcessValidationError("min_batches must be at least 2.")
    return _result("eye_process_drift_spec",baseline=baseline,difficulty_limit=difficulty_limit,discrimination_limit=discrimination_limit,gaze_validity_drop=gaze_validity_drop,luminance_limit=luminance_limit,relative_metric_quantile=relative_metric_quantile,min_batches=int(min_batches),interpretation="Drift flags indicate review needs; they do not establish item compromise or content leakage.")

def audit_process_drift(data: Any,item: str="item_id",batch: str="deployment_batch",metrics: Sequence[str] = ("irt_difficulty","irt_discrimination","rt_ms","dwell_ms","pupil_bc","valid_gaze_prop","screen_luminance"),spec: Any=None,reference_batch: Any=None,aggregate_fun: Callable[[Any],float]=_mean) -> EyeResult:
    d=_df(data); _req(d,[item,batch]); spec=process_drift_spec() if spec is None else spec
    if not _class(spec,"eye_process_drift_spec"): raise EyeProcessValidationError("spec must be process_drift_spec().")
    metrics=[m for m in metrics if m in d and np.isfinite(_num(d[m])).any()]
    if not metrics: raise EyeProcessValidationError("No usable drift metrics were found.")
    q=d[[item,batch,*metrics]].copy(); q[item]=q[item].astype(str)
    if pd.api.types.is_numeric_dtype(q[batch]): q["batch_order"]=pd.to_numeric(q[batch],errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(q[batch]): q["batch_order"]=pd.to_datetime(q[batch]).astype("int64")/1e9
    else:
        lev={v:i+1 for i,v in enumerate(pd.unique(q[batch]))}; q["batch_order"]=q[batch].map(lev).astype(float)
    for m in metrics: q[m]=pd.to_numeric(q[m],errors="coerce")
    agrows=[]
    for (iid,bv),z in q.groupby([item,batch],sort=False,dropna=False):
        r={item:iid,batch:bv,"batch_order":_mean(z.batch_order)}; r.update({m:aggregate_fun(z[m]) for m in metrics}); agrows.append(r)
    ag=pd.DataFrame(agrows)
    rows=[]
    for iid,z in ag.groupby(item,sort=False):
        z=z.sort_values("batch_order"); r={item:iid,"n_batches":len(z)}
        for m in metrics:
            vals=_num(z[m]); finite=vals[np.isfinite(vals)]
            if spec.baseline=="first_batch": base=float(finite[0]) if finite.size else math.nan
            else:
                if reference_batch is None: raise EyeProcessValidationError("reference_batch is required when baseline='reference_batch'.")
                ref=z.loc[z[batch].isin(np.atleast_1d(reference_batch)),m]; base=_mean(ref)
            latest=float(finite[-1]) if finite.size else math.nan; r[f"{m}_baseline"]=base; r[f"{m}_latest"]=latest; r[f"{m}_delta"]=latest-base
        rows.append(r)
    tab=pd.DataFrame(rows)
    def flag(m,fun):
        c=f"{m}_delta"; v=_num(tab[c]) if c in tab else np.full(len(tab),np.nan); return np.isfinite(v)&fun(v)
    tab["difficulty_drift_flag"]=flag("irt_difficulty",lambda v:np.abs(v)>spec.difficulty_limit)
    tab["discrimination_drift_flag"]=flag("irt_discrimination",lambda v:np.abs(v)>spec.discrimination_limit)
    tab["gaze_quality_drift_flag"]=flag("valid_gaze_prop",lambda v:v < -spec.gaze_validity_drop)
    tab["screen_luminance_drift_flag"]=flag("screen_luminance",lambda v:np.abs(v)>spec.luminance_limit)
    for m in [m for m in metrics if m not in {"irt_difficulty","irt_discrimination","valid_gaze_prop","screen_luminance"}]:
        vals=np.abs(_num(tab[f"{m}_delta"])); fin=vals[np.isfinite(vals)]; cutoff=float(np.quantile(fin,spec.relative_metric_quantile)) if fin.size else math.nan; tab[f"{m}_drift_flag"]=np.isfinite(cutoff)&np.isfinite(vals)&(vals>cutoff)
    tab["insufficient_batches_flag"]=tab.n_batches<spec.min_batches; flags=[c for c in tab if c.endswith("_flag")]; review=[c for c in flags if c!="insufficient_batches_flag"]; tab["drift_review_count"]=tab[review].sum(axis=1).astype(int); tab["drift_status"]=np.where(tab.insufficient_batches_flag,"insufficient_batches_for_drift_review",np.where(tab.drift_review_count>0,"review_item_or_stimulus_context","no_review_flag"))
    return _result("eye_process_drift_audit",table=tab,trajectories=ag,item=item,batch=batch,metrics=metrics,spec=spec,reference_batch=reference_batch,flag_columns=flags,caveat="Drift may reflect item exposure, stimulus redesign, device/luminance changes, population shift, or data quality. Flags do not prove compromise or leakage.")

def process_drift_alerts(x: Any) -> pd.DataFrame:
    if not _class(x,"eye_process_drift_audit"): raise EyeProcessValidationError("x must be eye_process_drift_audit.")
    return x.table.loc[x.table.drift_status=="review_item_or_stimulus_context"].reset_index(drop=True)

def compare_deployment_batches(data: Any,batch: str="deployment_batch",batch_a: Any=None,batch_b: Any=None,metrics: Sequence[str] | None=None,item: str="item_id") -> pd.DataFrame:
    d=_df(data); _req(d,[batch]);
    if metrics is None: metrics=[c for c in d.select_dtypes(include=np.number).columns if c not in {batch,item}]
    metrics=[m for m in metrics if m in d and m not in {batch,item} and np.isfinite(_num(d[m])).any()]
    if not metrics: raise EyeProcessValidationError("No usable numeric metrics were found for batch comparison.")
    a=d.loc[d[batch].isin(np.atleast_1d(batch_a))]; b=d.loc[d[batch].isin(np.atleast_1d(batch_b))]
    if a.empty or b.empty: raise EyeProcessValidationError("Both comparison batches must contain data.")
    if item in d:
        aa=a.groupby(item,sort=True)[metrics].mean().reset_index(); bb=b.groupby(item,sort=True)[metrics].mean().reset_index(); m=aa.merge(bb,on=item,suffixes=("_a","_b"));
        for metric in metrics: m[f"{metric}_delta"]=m[f"{metric}_b"]-m[f"{metric}_a"]
        return m
    return pd.DataFrame({"metric":metrics,"batch_a_mean":[_mean(a[m]) for m in metrics],"batch_b_mean":[_mean(b[m]) for m in metrics],"delta":[_mean(b[m])-_mean(a[m]) for m in metrics]})

def _grouped_drift(data: Any, group: str, **kwargs: Any) -> pd.DataFrame:
    d=_df(data); _req(d,[group]); vals=pd.unique(d[group].dropna())
    if not len(vals): raise EyeProcessValidationError("No non-missing grouping values are available for drift stratification.")
    out=[]
    for v in vals:
        t=audit_process_drift(d.loc[d[group]==v],**kwargs).table.copy(); t[group]=v; out.append(t)
    return pd.concat(out,ignore_index=True)
def drift_by_device(data: Any,device: str="device_id",**kwargs: Any)->pd.DataFrame:return _grouped_drift(data,device,**kwargs)
def drift_by_site(data: Any,site: str="site_id",**kwargs: Any)->pd.DataFrame:return _grouped_drift(data,site,**kwargs)
def drift_by_vendor(data: Any,vendor: str="vendor",**kwargs: Any)->pd.DataFrame:return _grouped_drift(data,vendor,**kwargs)
def drift_by_stimulus_version(data: Any,stimulus_version: str="stimulus_version",**kwargs: Any)->pd.DataFrame:return _grouped_drift(data,stimulus_version,**kwargs)


# ---------------------------------------------------------------------------
# R/060 temporal process windows and AOI trajectories
# ---------------------------------------------------------------------------

def process_window_spec(width_ms: float=1000,step_ms: float=500,start_ms: float=0,end_ms: float=3000,align: str="stimulus",min_samples: int=5)->EyeResult:
    if align not in {"stimulus","response","custom"}: raise EyeProcessValidationError("align must be stimulus, response, or custom.")
    vals=np.asarray([width_ms,step_ms,start_ms,end_ms],float)
    if not np.all(np.isfinite(vals)): raise EyeProcessValidationError("Window timing values must be finite.")
    if width_ms<=0 or step_ms<=0 or end_ms<=start_ms: raise EyeProcessValidationError("Require width_ms > 0, step_ms > 0, and end_ms > start_ms.")
    if width_ms>end_ms-start_ms: raise EyeProcessValidationError("width_ms cannot exceed the analysis range.")
    if int(min_samples)<2: raise EyeProcessValidationError("min_samples must be at least 2.")
    return _result("eye_process_window_spec",width_ms=width_ms,step_ms=step_ms,start_ms=start_ms,end_ms=end_ms,align=align,min_samples=int(min_samples))

def _auc(y,t):
    y,t=_num(y),_num(t); ok=np.isfinite(y)&np.isfinite(t); y,t=y[ok],t[ok]
    if len(y)<2:return math.nan
    o=np.argsort(t); return float(np.trapezoid(y[o],t[o]))
def _slope(y,t):
    y,t=_num(y),_num(t); ok=np.isfinite(y)&np.isfinite(t)
    if ok.sum()<3 or _sd(t[ok])==0:return math.nan
    return float(np.polyfit(t[ok],y[ok],1)[0])
def _rmssd(x):
    a=_num(x);a=a[np.isfinite(a)];return float(np.sqrt(np.mean(np.diff(a)**2))) if len(a)>=2 else math.nan
def _entropy_numeric(x,bins=8):
    a=_num(x);a=a[np.isfinite(a)]
    if len(a)<2 or len(np.unique(a))<2:return 0.0
    br=np.unique(np.quantile(a,np.linspace(0,1,bins+1)))
    if len(br)<3:return 0.0
    h=np.histogram(a,bins=br)[0]; p=h[h>0]/h.sum();return float(-np.sum(p*np.log(p)))
def _entropy_factor(x):
    s=pd.Series(x).dropna().astype(str);s=s[s.str.len()>0]
    if s.empty:return math.nan
    p=s.value_counts(normalize=True).to_numpy();return float(-np.sum(p*np.log(p)))
def _switch_count(x):
    s=pd.Series(x).dropna().astype(str);s=s[s.str.len()>0].to_numpy();return int(np.sum(s[1:]!=s[:-1])) if len(s)>=2 else 0
def _path_length(x,y):
    x,y=_num(x),_num(y);ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok];return float(np.sum(np.hypot(np.diff(x),np.diff(y)))) if len(x)>=2 else math.nan
def _velocity_mean(x,y,t):
    x,y,t=_num(x),_num(y),_num(t);ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(t);x,y,t=x[ok],y[ok],t[ok]
    if len(x)<2:return math.nan
    dt=np.diff(t)/1000; good=np.isfinite(dt)&(dt>0); v=np.hypot(np.diff(x),np.diff(y));return _mean(v[good]/dt[good]) if good.any() else math.nan

def extract_process_windows(data: Any,person: str="person_id",trial: str="trial_id",time: str="time_ms",spec: Any=None,align_time: str|None=None,pupil: str="pupil_bc",pupil_tonic: str="pupil_tonic",pupil_phasic: str="pupil_phasic",gaze_x: str="x",gaze_y: str="y",aoi: str="aoi",valid_gaze: str="valid_gaze_prop",valid_pupil: str="valid_pupil_prop",blink: str="blink",trackloss: str="trackloss") -> EyeResult:
    d=_df(data);_req(d,[person,trial,time]);spec=process_window_spec() if spec is None else spec
    if not _class(spec,"eye_process_window_spec"):raise EyeProcessValidationError("spec must be process_window_spec().")
    t=_num(d[time]);
    if align_time is not None:_req(d,[align_time]);t=t-_num(d[align_time])
    d[".ep08_relative_time"]=t; global_aois=sorted(pd.Series(d[aoi]).dropna().astype(str).loc[lambda s:s.str.len()>0].unique()) if aoi in d else []
    starts=np.arange(spec.start_ms,spec.end_ms-spec.width_ms+1e-12,spec.step_ms); rows=[]
    for idx in _groups(d,[person,trial]):
        gd=d.iloc[idx]
        for s in starts:
            e=s+spec.width_ms; mask=np.isfinite(gd[".ep08_relative_time"])&(gd[".ep08_relative_time"]>=s)&(gd[".ep08_relative_time"]<e); z=gd.loc[mask]
            if len(z)<spec.min_samples:continue
            def ncol(c):return _num(z[c]) if c in z else np.full(len(z),np.nan)
            pv,pt,pp,gx,gy=ncol(pupil),ncol(pupil_tonic),ncol(pupil_phasic),ncol(gaze_x),ncol(gaze_y); zt=_num(z[".ep08_relative_time"]); av=z[aoi].astype(object).to_numpy() if aoi in z else np.repeat(None,len(z));gv,qv=ncol(valid_gaze),ncol(valid_pupil);bl=_as_bool(z[blink]).astype(float) if blink in z else np.full(len(z),np.nan);tl=_as_bool(z[trackloss]).astype(float) if trackloss in z else np.full(len(z),np.nan)
            ac=float(pd.Series(pv[:-1]).corr(pd.Series(pv[1:]))) if np.isfinite(pv).sum()>=3 else math.nan
            r={**_group_values(d,idx,[person,trial]),"window_start":float(s),"window_end":float(e),"window_mid":float(s+spec.width_ms/2),"n_samples_window":len(z),"valid_gaze_prop":_mean(gv) if np.isfinite(gv).any() else math.nan,"valid_pupil_prop":_mean(qv) if np.isfinite(qv).any() else float(np.mean(np.isfinite(pv))),"blink_prop":_mean(bl),"trackloss_prop":_mean(tl),"pupil_mean":_mean(pv),"pupil_sd":_sd(pv),"pupil_slope":_slope(pv,zt),"pupil_auc":_auc(pv,zt),"pupil_rmssd":_rmssd(pv),"pupil_entropy":_entropy_numeric(pv),"pupil_peak":float(np.nanmax(pv)) if np.isfinite(pv).any() else math.nan,"pupil_autocorr_lag1":ac,"tonic_mean":_mean(pt),"phasic_mean":_mean(pp),"phasic_sd":_sd(pp),"phasic_auc":_auc(pp,zt),"gaze_velocity_mean":_velocity_mean(gx,gy,zt),"gaze_x_sd":_sd(gx),"gaze_y_sd":_sd(gy),"gaze_path_length":_path_length(gx,gy),"aoi_entropy":_entropy_factor(av),"aoi_switch_count":_switch_count(av)}
            for lv in global_aois:r[f"aoi_prop__{str(lv).replace(' ','_')}"]=float(np.mean(pd.Series(av).dropna().astype(str)==lv)) if pd.Series(av).notna().any() else math.nan
            rows.append(r)
    return _result("eye_process_windows",data=pd.DataFrame(rows),spec=spec,person=person,trial=trial,time=time,align_time=align_time,source_n=len(d),status="windowed_process_representation")

def summarize_process_windows(x: Any,by: Sequence[str]|None=None)->pd.DataFrame:
    if not _class(x,"eye_process_windows"):raise EyeProcessValidationError("x must be eye_process_windows.")
    d=x.data
    if d.empty:return d.copy()
    nums=[c for c in d.select_dtypes(include=np.number).columns if c not in {"window_start","window_end","window_mid"}]
    if not by:return pd.DataFrame({"metric":nums,"mean":[_mean(d[c]) for c in nums],"sd":[_sd(d[c]) for c in nums]})
    _req(d,by);return d.groupby(list(by),sort=True,as_index=False)[nums].mean()
def bind_process_windows(*args: Any, **kwargs: Any)->EyeResult:
    xs=list(args) + list(kwargs.values()); xs=xs[0] if len(xs)==1 and isinstance(xs[0],(list,tuple)) and not _class(xs[0],"eye_process_windows") else xs
    if not xs or not all(_class(z,"eye_process_windows") for z in xs):raise EyeProcessValidationError("All inputs must be eye_process_windows objects.")
    return _result("eye_process_windows",data=pd.concat([z.data for z in xs],ignore_index=True,sort=False),spec=[z.spec for z in xs],source_n=sum(z.source_n for z in xs),status="bound_process_windows")
def validate_process_windows(x: Any)->pd.DataFrame:
    if not _class(x,"eye_process_windows"):raise EyeProcessValidationError("x must be eye_process_windows.")
    d=x.data; req=["window_start","window_end","window_mid","n_samples_window"]; miss=[c for c in req if c not in d];issues=[]
    if miss:issues.append("missing columns: "+", ".join(miss))
    if not d.empty and not miss and (d.window_end<=d.window_start).any():issues.append("non-positive window width")
    if not d.empty and "n_samples_window" in d and (d.n_samples_window<1).any():issues.append("empty windows")
    return pd.DataFrame({"valid":[not issues],"issue":["; ".join(issues) if issues else "none"],"n_windows":[len(d)]})
def audit_process_window_sensitivity(data: Any,widths_ms: Sequence[float]=(250,500,1000,1500),steps_ms: Sequence[float]=(100,250,500),metric: str="pupil_mean",grid: bool=True,**kwargs: Any)->EyeResult:
    settings=[(w,s) for w in widths_ms for s in steps_ms] if grid else list(zip(np.resize(widths_ms,max(len(widths_ms),len(steps_ms))),np.resize(steps_ms,max(len(widths_ms),len(steps_ms)))))
    rows=[]
    for w,s in settings:
        base=kwargs.get("spec"); spec=process_window_spec(w,s,base.start_ms if base else 0,base.end_ms if base else 3000); kw={**kwargs,"spec":spec}; x=extract_process_windows(data,**kw); d=x.data; rows.append({"width_ms":w,"step_ms":s,"metric":metric,"mean_value":_mean(d[metric]) if metric in d else math.nan,"sd_value":_sd(d[metric]) if metric in d else math.nan,"n_windows":len(d)})
    return _result("eye_process_window_sensitivity",table=pd.DataFrame(rows),metric=metric,settings=pd.DataFrame(settings,columns=["width_ms","step_ms"]),caveat="Window sensitivity assesses representation robustness; it does not select a causal or psychologically privileged window.")

def _polyfit_orthogonal(time: Any,y: Any,degree: int):
    t,y=_num(time),_num(y);ok=np.isfinite(t)&np.isfinite(y);t,y=t[ok],y[ok]
    if len(y)<degree+2 or len(np.unique(t))<degree+1:return np.full(degree,np.nan)
    ts=(t-t.mean())/(t.std(ddof=1) or 1); V=np.column_stack([ts**j for j in range(1,degree+1)]); Q,_=np.linalg.qr(V); A=np.column_stack([np.ones(len(y)),Q]); return np.linalg.lstsq(A,y,rcond=None)[0][1:]
def aoi_trajectory_features(data: Any,person: str="person_id",trial: str="trial_id",time: str="time_ms",aoi: str="aoi",bin_ms: float=100,degree: int=3,aois: Sequence[str]|None=None)->EyeResult:
    d=_df(data);_req(d,[person,trial,time,aoi]);
    if d.empty:raise EyeProcessValidationError("data must contain at least one row.")
    degree=int(degree)
    if degree<1 or degree>6:raise EyeProcessValidationError("degree must be between 1 and 6.")
    if not np.isfinite(bin_ms) or bin_ms<=0:raise EyeProcessValidationError("bin_ms must be positive.")
    q=d[[person,trial,time,aoi]].copy();q[".time"]=_num(q[time]);q[".aoi"]=q[aoi].astype(object);q[".bin"]=np.floor(q[".time"]/bin_ms)*bin_ms
    aois=sorted(pd.Series(q[".aoi"]).dropna().astype(str).loc[lambda s:s.str.len()>0].unique()) if aois is None else list(aois)
    if not aois:raise EyeProcessValidationError("No AOI levels were available.")
    rows=[]
    for idx in _groups(q,[person,trial]):
        z=q.iloc[idx];bins=np.sort(z[".bin"].dropna().unique());r=_group_values(q,idx,[person,trial])
        for lv in aois:
            prop=[float(np.mean(z.loc[z[".bin"]==b,".aoi"].astype(str)==lv)) for b in bins];cf=_polyfit_orthogonal(bins,prop,degree)
            for j,v in enumerate(cf,1):r[f"{str(lv).replace(' ','_')}_gca_degree{j}"]=v
        rows.append(r)
    return _result("eye_aoi_trajectory",features=pd.DataFrame(rows),aois=aois,degree=degree,bin_ms=bin_ms,person=person,trial=trial,caveat="AOI trajectory coefficients summarize time-course shape and should not be interpreted causally without a design supporting that claim.")
def fit_aoi_growth_curve(data: Any,time: str,outcome: str,degree: int=3)->EyeResult:
    d=_df(data);degree=int(degree);_req(d,[time,outcome])
    if degree<1 or degree>6:raise EyeProcessValidationError("degree must be between 1 and 6.")
    t,y=_num(d[time]),_num(d[outcome]);ok=np.isfinite(t)&np.isfinite(y);t,y=t[ok],y[ok]
    if len(y)<degree+2:raise EyeProcessValidationError("Insufficient observations for requested degree.")
    center=float(t.mean());scale=float(t.std(ddof=1) or 1);ts=(t-center)/scale;X=pd.DataFrame({f"p{j}":ts**j for j in range(1,degree+1)});model=_ols(y,X);return _result("eye_aoi_growth_curve",model=model,time=t,outcome=y,degree=degree,center=center,scale=scale,range=(float(t.min()),float(t.max())),status="trajectory_shape_model")
def predict_aoi_trajectory(object: Any,time: Any=None)->pd.DataFrame:
    if not _class(object,"eye_aoi_growth_curve"):raise EyeProcessValidationError("object must be eye_aoi_growth_curve.")
    t=np.linspace(*object.range,101) if time is None else _num(time);ts=(t-object.center)/object.scale;A=np.column_stack([np.ones(len(t)),*[ts**j for j in range(1,object.degree+1)]]);beta=object.model.coefficients.estimate.to_numpy(float);return pd.DataFrame({"time":t,"predicted":A@beta})
def compare_aoi_trajectories(*args: Any, **kwargs: Any)->pd.DataFrame:
    xs=list(args) + list(kwargs.values());xs=xs[0] if len(xs)==1 and isinstance(xs[0],(list,tuple)) else xs
    if not xs:raise EyeProcessValidationError("Supply at least one eye_aoi_trajectory object.")
    if not all(_class(x,"eye_aoi_trajectory") for x in xs):raise EyeProcessValidationError("All inputs must be eye_aoi_trajectory objects.")
    return pd.DataFrame({"model":np.arange(1,len(xs)+1),"n_rows":[len(x.features) for x in xs],"n_aois":[len(x.aois) for x in xs],"degree":[x.degree for x in xs],"bin_ms":[x.bin_ms for x in xs]})


# ---------------------------------------------------------------------------
# R/061 advanced pupil representations
# ---------------------------------------------------------------------------

def _interp(y: Any)->np.ndarray:
    a=_num(y);idx=np.arange(len(a));ok=np.isfinite(a)
    if ok.sum()<2:return np.full(len(a),np.nan)
    return np.interp(idx,idx[ok],a[ok])
def pupil_band_power(y: Any,sampling_rate_hz: float,lower_hz: float,upper_hz: float,detrend: bool=True)->float:
    a=_interp(y)
    if len(a)<8 or not np.isfinite(a).any():return math.nan
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz<=0:raise EyeProcessValidationError("sampling_rate_hz must be positive.")
    if not (np.isfinite(lower_hz) and np.isfinite(upper_hz) and 0<=lower_hz<upper_hz):raise EyeProcessValidationError("Require 0 <= lower_hz < upper_hz.")
    if detrend:a=a-np.mean(a)
    if _sd(a)==0:return 0.0
    n=len(a);f=np.fft.fft(a);power=np.abs(f)**2/n;freq=np.arange(n)*sampling_rate_hz/n;keep=(freq>=lower_hz)&(freq<=upper_hz)&(freq<=sampling_rate_hz/2);return float(power[keep].sum()) if keep.any() else math.nan
def pupil_velocity_activity(y: Any,time_ms: Any)->float:
    a,t=_interp(y),_num(time_ms)/1000;ok=np.isfinite(a)&np.isfinite(t);a,t=a[ok],t[ok]
    if len(a)<4:return math.nan
    dt=np.diff(t);dy=np.diff(a);good=np.isfinite(dt)&(dt>0)&np.isfinite(dy);return float(np.sqrt(np.mean((dy[good]/dt[good])**2))) if good.any() else math.nan
def _roll_mean(x,w):return pd.Series(_interp(x)).rolling(int(w),center=True,min_periods=int(w)).mean().to_numpy()
def _width_ms(ms,t):
    tt=np.sort(np.unique(_num(t)));step=np.nanmedian(np.diff(tt));w=5 if not np.isfinite(step) or step<=0 else max(3,int(round(ms/step)));return w+1 if w%2==0 else w
def _deriv_power(y,t,ms):
    a=_interp(y);tt=_num(t)/1000
    if len(a)<6:return math.nan
    sm=_roll_mean(a,_width_ms(ms,t));ok=np.isfinite(sm)&np.isfinite(tt);sm,tt=sm[ok],tt[ok]
    if len(sm)<6:return math.nan
    dt=np.diff(tt);dy=np.diff(sm);good=np.isfinite(dt)&(dt>0)&np.isfinite(dy);return float(np.mean((dy[good]/dt[good])**2)) if good.any() else math.nan
def pupil_activity_index(y: Any,time_ms: Any=None,sampling_rate_hz: float|None=None,method: str="velocity",low_band: Sequence[float]=(.05,.50),high_band: Sequence[float]=(.50,4.0),fast_window_ms: float=250,slow_window_ms: float=750)->float:
    if time_ms is None:time_ms=np.arange(1,len(y)+1)
    if method=="velocity":return pupil_velocity_activity(y,time_ms)
    if method=="frequency_contrast":
        if sampling_rate_hz is None:raise EyeProcessValidationError("sampling_rate_hz is required for frequency_contrast.")
        lo=pupil_band_power(y,sampling_rate_hz,*low_band);hi=pupil_band_power(y,sampling_rate_hz,*high_band);return float(np.log1p(hi)-np.log1p(lo)) if np.isfinite(lo) and np.isfinite(hi) else math.nan
    if method!="ripa_proxy":raise EyeProcessValidationError("method must be velocity, frequency_contrast, or ripa_proxy.")
    fast,slow=_deriv_power(y,time_ms,fast_window_ms),_deriv_power(y,time_ms,slow_window_ms);return float(np.log1p(fast)-np.log1p(slow)) if np.isfinite(fast) and np.isfinite(slow) else math.nan

def pupil_frequency_features(data: Any,by: Sequence[str]=("person_id","trial_id"),time: str="time_ms",pupil: str="pupil_bc",sampling_rate_hz: Any=60,low_band: Sequence[float]=(.05,.50),high_band: Sequence[float]=(.50,4.0))->EyeResult:
    d=_df(data);by=list(by);_req(d,[*by,time,pupil]);rows=[]
    for idx in _groups(d,by):
        z=d.iloc[idx];sr=_mean(z[sampling_rate_hz]) if isinstance(sampling_rate_hz,str) else float(np.atleast_1d(sampling_rate_hz)[0]);y=z[pupil];tt=z[time];lo=pupil_band_power(y,sr,*low_band);hi=pupil_band_power(y,sr,*high_band);rows.append({**_group_values(d,idx,by),"sampling_rate_hz":sr,"pupil_low_frequency_power":lo,"pupil_high_frequency_power":hi,"pupil_frequency_contrast":np.log1p(hi)-np.log1p(lo) if np.isfinite(lo) and np.isfinite(hi) else math.nan,"pupil_velocity_activity":pupil_velocity_activity(y,tt),"pupil_ripa_proxy":pupil_activity_index(y,tt,sr,"ripa_proxy")})
    return _result("eye_pupil_frequency_features",features=pd.DataFrame(rows),low_band=tuple(low_band),high_band=tuple(high_band),by=by,pupil=pupil,time=time,caveat="Frequency/activity features are signal representations, not pure cognitive-load measures. Short windows, luminance, gaze position, blink handling, and filtering can materially affect them.")
def audit_pupil_frequency_stability(data: Any,windows_ms: Sequence[float]=(500,1000,2000),by: Sequence[str]=("person_id","trial_id"),time: str="time_ms",pupil: str="pupil_bc",sampling_rate_hz: Any=60)->EyeResult:
    d=_df(data);_req(d,[*by,time,pupil]);rows=[]
    for idx in _groups(d,list(by)):
        z=d.iloc[idx];tt=_num(z[time]);
        if not np.isfinite(tt).any():continue
        t0=np.nanmin(tt)
        for w in windows_ms:
            zz=z.loc[(pd.to_numeric(z[time],errors="coerce")>=t0)&(pd.to_numeric(z[time],errors="coerce")<t0+w)]
            if len(zz)<8:continue
            f=pupil_frequency_features(zz,by,time,pupil,sampling_rate_hz).features
            if not f.empty:f=f.copy();f["window_ms"]=w;rows.append(f)
    return _result("eye_pupil_frequency_stability",table=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(),windows_ms=list(windows_ms),caveat="Large feature changes across window choices indicate representation instability.")
def pupil_response_kernel(time_since_event_ms: Any,tmax_ms: float=930,shape: float=10.1,normalize: bool=True)->np.ndarray:
    raw=_num(time_since_event_ms);t=np.maximum(raw,0)/1000;tmax=float(tmax_ms)/1000
    if not np.isfinite(tmax) or tmax<=0 or not np.isfinite(shape) or shape<=0:raise EyeProcessValidationError("tmax_ms and shape must be positive.")
    with np.errstate(invalid="ignore",over="ignore"):out=(t/tmax)**shape*np.exp(-shape*((t/tmax)-1))
    out[raw<0]=0;out[~np.isfinite(out)]=0
    if normalize and np.max(out)>0:out=out/np.max(out)
    return out
def pupil_event_regressor(time_ms: Any,event_time_ms: float,tmax_ms: float=930,shape: float=10.1)->np.ndarray:return pupil_response_kernel(_num(time_ms)-float(event_time_ms),tmax_ms,shape)
def fit_pupil_event_deconvolution(data: Any,by: Sequence[str]=("person_id","trial_id"),time: str="time_ms",pupil: str="pupil_bc",events: Mapping[str,Any]|None=None,tmax_ms: float=930,shape: float=10.1,min_samples: int=20)->EyeResult:
    d=_df(data);by=list(by);_req(d,[*by,time,pupil]);events={"stimulus":0} if events is None else dict(events)
    if not events or any(not str(k) for k in events):raise EyeProcessValidationError("events must be a named mapping.")
    fits=[]; effects=[]; fitted=[]
    for idx in _groups(d,by):
        z=d.iloc[idx];tt=_num(z[time]);yy=_num(z[pupil]);X={};ets={};valid=True
        for nm,spec in events.items():
            if isinstance(spec,str):_req(z,[spec]); vals=_num(z[spec]);fin=vals[np.isfinite(vals)];et=float(fin[0]) if fin.size else math.nan
            elif np.size(spec)==1:et=float(spec)
            else:raise EyeProcessValidationError("Each event specification must be a scalar numeric time or a column name.")
            ets[nm]=et
            if not np.isfinite(et):valid=False;break
            X[f"event__{str(nm).replace(' ','_')}"]=pupil_event_regressor(tt,et,tmax_ms,shape)
        if not valid:continue
        q=pd.DataFrame({"pupil":yy,"time":tt,**X}).dropna();
        if len(q)<min_samples or _sd(q.pupil)==0:continue
        model=_ols(q.pupil.to_numpy(float),q[[c for c in q if c.startswith("event__")]]);base=_group_values(d,idx,by);er=dict(base)
        for nm in events:
            cn=f"event__{str(nm).replace(' ','_')}";er[f"beta__{str(nm).replace(' ','_')}"]=float(model.coefficients.loc[cn,"estimate"]);er[f"event_time__{str(nm).replace(' ','_')}"]=ets[nm]
        er["residual_sd"]=_sd(model.residuals);er["r_squared"]=model.r_squared;effects.append(er);fits.append(model);fr=pd.DataFrame({**{k:[v]*len(q) for k,v in base.items()},"time":q.time,"observed":q.pupil,"fitted":model.fitted,"residual":model.residuals});fitted.append(fr)
    return _result("eye_pupil_deconvolution",fits=fits,effects=pd.DataFrame(effects),fitted=pd.concat(fitted,ignore_index=True) if fitted else pd.DataFrame(),events=events,tmax_ms=tmax_ms,shape=shape,by=by,status="transparent_linear_kernel_deconvolution",caveat="Event coefficients depend on the selected response kernel and event timing. Use kernel/timing sensitivity analyses before substantive physiological interpretation.")
def pupil_event_effects(x: Any)->pd.DataFrame:
    if not _class(x,"eye_pupil_deconvolution"):raise EyeProcessValidationError("x must be eye_pupil_deconvolution.")
    return x.effects.copy()
def compare_pupil_kernels(data: Any,tmax_values: Sequence[float]=(512,930),**kwargs: Any)->pd.DataFrame:
    rows=[]
    for tm in tmax_values:
        f=fit_pupil_event_deconvolution(data,tmax_ms=tm,**kwargs);e=f.effects;rows.append({"tmax_ms":tm,"mean_r_squared":_mean(e.r_squared) if "r_squared" in e else math.nan,"mean_residual_sd":_mean(e.residual_sd) if "residual_sd" in e else math.nan,"n_groups":len(e)})
    return pd.DataFrame(rows)

def fit_pupil_confound_model(data: Any,pupil: str="pupil_peak",luminance: str="screen_luminance",trial_order: str="trial_sequence",theta: str|None=None,person: str="person_id",item: str="item_id",engine: str="auto")->EyeResult:
    if engine not in {"auto","mgcv","lm"}:raise EyeProcessValidationError("engine must be auto, mgcv, or lm.")
    d=_df(data);req=[pupil,luminance,trial_order]+([theta] if theta else []);_req(d,req);fitd=d.copy();fitd[".pupil"]=_num(d[pupil]);fitd[".luminance"]=_num(d[luminance]);fitd[".trial"]=_num(d[trial_order]);fitd[".theta"]=_num(d[theta]) if theta else 0.0;fitd[".person"]=d[person].astype(str) if person in d else "all";fitd[".item"]=d[item].astype(str) if item in d else "all";fitd=fitd.loc[np.isfinite(fitd[[".pupil",".luminance",".trial",".theta"]]).all(axis=1)].copy()
    if len(fitd)<30:raise EyeProcessValidationError("At least 30 complete observations are required.")
    nlum,ntrial=fitd[".luminance"].nunique(),fitd[".trial"].nunique()
    if nlum<2 or ntrial<2:raise EyeProcessValidationError("Luminance and trial-order predictors must each contain at least two unique values.")
    chosen="lm" if engine=="auto" else engine
    if chosen=="mgcv":raise EyeProcessBackendError("The frozen R engine='mgcv' requires R package mgcv; no algorithmically identical Python backend is substituted. Use engine='lm' for the transparent reference path.")
    X=pd.DataFrame(index=fitd.index);X["luminance"] = fitd[".luminance"];X["trial"] = fitd[".trial"]
    if nlum>=3:X["luminance2"]=fitd[".luminance"]**2
    if ntrial>=3:X["trial2"]=fitd[".trial"]**2
    if _sd(fitd[".theta"])>0:X["theta"]=fitd[".theta"];X["luminance_theta"]=fitd[".luminance"]*fitd[".theta"]
    if fitd[".person"].nunique()>1 and fitd[".person"].nunique()<=100:X=pd.concat([X,pd.get_dummies(fitd[".person"],prefix="person",drop_first=True,dtype=float)],axis=1)
    if fitd[".item"].nunique()>1 and fitd[".item"].nunique()<=100:X=pd.concat([X,pd.get_dummies(fitd[".item"],prefix="item",drop_first=True,dtype=float)],axis=1)
    model=_ols(fitd[".pupil"].to_numpy(float),X);fitd["pupil_confound_residual"]=model.residuals;fitd["pupil_confound_adjusted"]=model.residuals+fitd[".pupil"].mean()
    return _result("eye_pupil_confound_model",model=model,data=fitd,engine="lm",original_columns={"pupil":pupil,"luminance":luminance,"trial_order":trial_order,"theta":theta,"person":person,"item":item},status="confound_adjustment_sensitivity",caveat="Adjusted pupil values remain model-dependent and should be described as luminance/fatigue-adjusted, not as pure cognition or effort isolated from all confounding.")
def adjust_pupil_confounds(x: Any)->pd.DataFrame:
    if not _class(x,"eye_pupil_confound_model"):raise EyeProcessValidationError("x must be eye_pupil_confound_model.")
    return x.data.copy()
def pupil_confound_effects(x: Any)->pd.DataFrame:
    if not _class(x,"eye_pupil_confound_model"):raise EyeProcessValidationError("x must be eye_pupil_confound_model.")
    return x.model.coefficients.copy()
def audit_pupil_fatigue_drift(data: Any,pupil: str="pupil_peak",trial_order: str="trial_sequence",person: str="person_id",luminance: str|None=None,difficulty: str|None=None,engine: str="auto")->EyeResult:
    if engine not in {"auto","plm","lm_fixed_effects"}:raise EyeProcessValidationError("engine must be auto, plm, or lm_fixed_effects.")
    d=_df(data);_req(d,[pupil,trial_order,person,luminance,difficulty]);q=pd.DataFrame({"pupil":_num(d[pupil]),"trial_order":_num(d[trial_order]),"person":d[person].astype(str)})
    if luminance:q["luminance"]=_num(d[luminance])
    if difficulty:q["difficulty"]=_num(d[difficulty])
    q=q.replace([np.inf,-np.inf],np.nan).dropna()
    if len(q)<20 or q.person.nunique()<2:raise EyeProcessValidationError("At least 20 complete rows from two or more persons are required.")
    chosen="lm_fixed_effects" if engine=="auto" else engine
    if chosen=="plm":raise EyeProcessBackendError("The frozen R engine='plm' requires R package plm; no algorithmically identical Python backend is substituted. Use engine='lm_fixed_effects'.")
    X=q[[c for c in ["trial_order","luminance","difficulty"] if c in q]].copy();X=pd.concat([X,pd.get_dummies(q.person,prefix="person",drop_first=True,dtype=float)],axis=1);model=_ols(q.pupil.to_numpy(float),X)
    return _result("eye_pupil_fatigue_drift",model=model,coefficients=model.coefficients,data=q,engine="lm_fixed_effects",status="within_person_fatigue_drift_sensitivity",caveat="Trial-order association is a fatigue/drift sensitivity signal, not proof of a fatigue mechanism.")
def compare_raw_adjusted_pupil(x: Any)->pd.DataFrame:
    if not _class(x,"eye_pupil_confound_model"):raise EyeProcessValidationError("x must be eye_pupil_confound_model.")
    d=x.data
    def cor(a,b):
        aa,bb=_num(a),_num(b);ok=np.isfinite(aa)&np.isfinite(bb);return float(np.corrcoef(aa[ok],bb[ok])[0,1]) if ok.sum()>=2 else math.nan
    return pd.DataFrame({"metric":["mean","sd","correlation_with_luminance","correlation_with_trial_order"],"raw":[_mean(d[".pupil"]),_sd(d[".pupil"]),cor(d[".pupil"],d[".luminance"]),cor(d[".pupil"],d[".trial"])],"adjusted":[_mean(d.pupil_confound_adjusted),_sd(d.pupil_confound_adjusted),cor(d.pupil_confound_adjusted,d[".luminance"]),cor(d.pupil_confound_adjusted,d[".trial"])]})
def filter_eye_signal(signal: Any,width: int=9,method: str="auto",online: bool=True)->EyeResult:
    if method not in {"auto","robfilter","runmed"}:raise EyeProcessValidationError("method must be auto, robfilter, or runmed.")
    y=_num(signal)
    if np.isfinite(y).sum()<5:raise EyeProcessValidationError("Too few finite signal values.")
    width=max(3,int(width));width=width+1 if width%2==0 else width;maxodd=len(y) if len(y)%2 else len(y)-1;width=min(width,maxodd)
    if width<3:raise EyeProcessValidationError("Signal is too short for median filtering.")
    chosen="runmed" if method=="auto" else method
    if chosen=="robfilter":raise EyeProcessBackendError("The frozen R robfilter backend is unavailable; no algorithmically identical Python substitute is used. Select method='runmed'.")
    yi=_interp(y);filtered=pd.Series(yi).rolling(width,center=True,min_periods=1).median().to_numpy();tab=pd.DataFrame({"sample_index":np.arange(1,len(y)+1),"raw":y,"filtered":filtered,"residual":y-filtered})
    return _result("eye_signal_filter_audit",data=tab,fit=None,method="runmed",width=width,status="base_running_median_reference",caveat="Filtering choices can change downstream pupil/process features; preserve raw data and audit sensitivity.")
def filter_pupil_signal(*args: Any,**kwargs: Any)->EyeResult:return filter_eye_signal(*args,**kwargs)
def audit_signal_filter(x: Any)->pd.DataFrame:
    if not _class(x,"eye_signal_filter_audit"):raise EyeProcessValidationError("x must be eye_signal_filter_audit.")
    d=x.data;aa,bb=_num(d.raw),_num(d.filtered);ok=np.isfinite(aa)&np.isfinite(bb);cor=float(np.corrcoef(aa[ok],bb[ok])[0,1]) if ok.sum()>=2 else math.nan;return pd.DataFrame({"method":[x.method],"width":[x.width],"raw_sd":[_sd(d.raw)],"filtered_sd":[_sd(d.filtered)],"residual_sd":[_sd(d.residual)],"raw_filtered_cor":[cor]})
def compare_signal_filters(signal: Any,widths: Sequence[int]=(5,9,15),methods: Sequence[str]=("runmed","robfilter"))->pd.DataFrame:
    rows=[]
    for m in methods:
        if m=="robfilter":continue
        for w in widths:rows.append(audit_signal_filter(filter_eye_signal(signal,w,m)))
    if not rows:raise EyeProcessValidationError("No requested filter method was available.")
    return pd.concat(rows,ignore_index=True)
