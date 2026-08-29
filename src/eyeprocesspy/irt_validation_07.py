"""Estimator-agnostic IRT validation contracts from frozen eyeprocess 0.7 source."""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .exceptions import EyeProcessValidationError
from .irt import EyeResult, _as_df, _req_cols, _result, _tag

__all__ = [
    "irt_validation_spec", "as_irt_recovery_results", "summarize_parameter_recovery",
    "audit_bias", "audit_rmse", "audit_coverage", "audit_interval_width", "audit_convergence",
    "validation_failure_taxonomy", "audit_identifiability", "validation_mcse",
    "recommended_validation_replications", "run_sbc", "posterior_sbc_contract",
    "run_posterior_sbc", "audit_sbc", "posterior_predictive_discrepancies",
    "stress_test_misspecification", "stress_test_latent_distribution", "stress_test_local_dependence",
    "stress_test_speededness", "stress_test_missingness", "stress_test_preprocessing",
    "external_validate_irt", "leave_device_out_validation", "leave_session_out_validation",
    "leave_site_out_validation", "leave_item_out_validation", "audit_measurement_transportability",
    "compare_validation_engines", "audit_channel_incremental_information",
    "negative_control_process_test", "calibration_transfer_audit", "grade_model_evidence",
]


def _df(x: Any, name: str="data") -> pd.DataFrame:
    return _as_df(x,name)


def _bind(xs: Sequence[pd.DataFrame|None]) -> pd.DataFrame:
    ys=[x for x in xs if x is not None and len(x)>0]
    if not ys: return pd.DataFrame()
    cols=[]
    for x in ys:
        for c in x.columns:
            if c not in cols: cols.append(c)
    return pd.concat([x.reindex(columns=cols) for x in ys],ignore_index=True)


def irt_validation_spec(model_id: str, replications: int=250, parameters: Any=None,
                        metrics: Sequence[str] = ("bias","rmse","coverage","interval_width","convergence"),
                        grouped_validation: Sequence[str] = ("device","session","site"),
                        preprocessing_variants: Any=None, misspecification_scenarios: Any=None,
                        thresholds: Mapping[str,Any]|None=None, seed: int=20260808, notes: str|None=None) -> EyeResult:
    if not isinstance(model_id,str) or not model_id: raise EyeProcessValidationError("model_id must be one non-empty string.")
    replications=int(replications)
    if replications<1: raise EyeProcessValidationError("replications must be >= 1.")
    th={"max_abs_bias":.10,"max_rmse":.30,"min_coverage":.90,"max_failure_rate":.05,"min_external_folds":2}
    if thresholds: th.update(dict(thresholds))
    return _result("eye_irt_validation_spec",model_id=model_id,replications=replications,parameters=parameters,
                   metrics=list(dict.fromkeys(metrics)),grouped_validation=list(dict.fromkeys(grouped_validation)),
                   preprocessing_variants=preprocessing_variants,misspecification_scenarios=misspecification_scenarios,
                   thresholds=th,seed=int(seed),notes=notes,contract_version="0.7.0",created_with="eyeprocess")


def as_irt_recovery_results(results: Any) -> pd.DataFrame:
    d=_df(results,"results"); _req_cols(d,["replicate","parameter","truth","estimate"],"results"); d=d.copy()
    d["truth"]=pd.to_numeric(d.truth,errors="coerce"); d["estimate"]=pd.to_numeric(d.estimate,errors="coerce")
    if "converged" not in d: d["converged"]=np.isfinite(d.estimate)
    d["converged"]=d.converged.astype("boolean")
    if "failure_type" not in d: d["failure_type"]=pd.NA
    if "scenario" not in d: d["scenario"]="baseline"
    if "engine" not in d: d["engine"]="unspecified"
    d["error"]=d.estimate-d.truth
    return _tag(d,"eye_irt_recovery_results")


def summarize_parameter_recovery(results: Any, by: Sequence[str] = ("scenario","engine","parameter"), interval_level: float=.95) -> pd.DataFrame:
    d=as_irt_recovery_results(results); by=[c for c in by if c in d.columns]; rows=[]
    groups=[((),d)] if not by else d.groupby(by,dropna=False,sort=False)
    for key,z in groups:
        ok=z.converged.fillna(False).to_numpy(bool); finite=ok & np.isfinite(z.truth) & np.isfinite(z.estimate); err=z.loc[finite,"error"].to_numpy(float)
        if {"lower","upper"} <= set(z.columns):
            cover=(pd.to_numeric(z.lower,errors="coerce")<=z.truth)&(z.truth<=pd.to_numeric(z.upper,errors="coerce")); width=pd.to_numeric(z.upper,errors="coerce")-pd.to_numeric(z.lower,errors="coerce")
        else: cover=pd.Series(np.nan,index=z.index); width=pd.Series(np.nan,index=z.index)
        row={}
        if by:
            vals=key if isinstance(key,tuple) else (key,); row.update(dict(zip(by,vals)))
        row.update(n=len(z),n_converged=int(ok.sum()),convergence_rate=float(np.mean(ok)),failure_rate=float(np.mean(~ok)),
                   bias=float(np.mean(err)) if len(err) else math.nan,absolute_bias=float(abs(np.mean(err))) if len(err) else math.nan,
                   rmse=float(np.sqrt(np.mean(err**2))) if len(err) else math.nan,mae=float(np.mean(np.abs(err))) if len(err) else math.nan,
                   coverage=float(cover.mean(skipna=True)) if cover.notna().any() else math.nan,
                   interval_width=float(width.mean(skipna=True)) if np.isfinite(width).any() else math.nan,nominal_coverage=float(interval_level))
        rows.append(row)
    return _tag(pd.DataFrame(rows),"eye_irt_recovery_summary")


def _audit_metric(results: Any, col: str, passfun: Callable[[pd.Series],pd.Series], cls: str, extra: Mapping[str,Any], by: Sequence[str]) -> pd.DataFrame:
    s=results.copy() if isinstance(results,pd.DataFrame) and results.attrs.get("eyeprocess_class")=="eye_irt_recovery_summary" else summarize_parameter_recovery(results,by=by)
    for k,v in extra.items(): s[k]=v
    s["pass"]=passfun(pd.to_numeric(s[col],errors="coerce")); return _tag(s,cls)


def audit_bias(results: Any, threshold: float=.10, by: Sequence[str] = ("scenario","engine","parameter")) -> pd.DataFrame:
    return _audit_metric(results,"absolute_bias",lambda x:np.isfinite(x)&(x<=threshold),"eye_irt_bias_audit",{"threshold":threshold},by)

def audit_rmse(results: Any, threshold: float=.30, by: Sequence[str] = ("scenario","engine","parameter")) -> pd.DataFrame:
    return _audit_metric(results,"rmse",lambda x:np.isfinite(x)&(x<=threshold),"eye_irt_rmse_audit",{"threshold":threshold},by)

def audit_coverage(results: Any, minimum: float=.90, maximum: float=1, by: Sequence[str] = ("scenario","engine","parameter")) -> pd.DataFrame:
    return _audit_metric(results,"coverage",lambda x:np.isfinite(x)&(x>=minimum)&(x<=maximum),"eye_irt_coverage_audit",{"minimum":minimum,"maximum":maximum},by)

def audit_interval_width(results: Any, maximum: float=math.inf, by: Sequence[str] = ("scenario","engine","parameter")) -> pd.DataFrame:
    return _audit_metric(results,"interval_width",lambda x:np.isfinite(x)&(x<=maximum),"eye_irt_interval_width_audit",{"maximum":maximum},by)


def audit_convergence(results: Any, minimum: float=.95, by: Sequence[str] = ("scenario","engine")) -> pd.DataFrame:
    d=as_irt_recovery_results(results); by=[c for c in by if c in d]; rows=[]; groups=[((),d)] if not by else d.groupby(by,dropna=False,sort=False)
    for key,z in groups:
        per=z.groupby("replicate",dropna=False).converged.apply(lambda x:bool(x.fillna(False).all())); rate=float(per.mean())
        row={}; vals=key if isinstance(key,tuple) else (key,)
        if by: row.update(dict(zip(by,vals)))
        row.update(n_replicates=len(per),convergence_rate=rate,minimum=minimum,pass_=rate>=minimum); rows.append(row)
    out=pd.DataFrame(rows).rename(columns={"pass_":"pass"}); return _tag(out,"eye_irt_convergence_audit")


def validation_failure_taxonomy(x: Any) -> pd.DataFrame:
    msg=[str(x)] if isinstance(x,(str,BaseException)) else [str(v) for v in x]; out=[]
    for m in msg:
        low=m.lower(); typ="other"
        if any(k in low for k in ("singular","boundary")): typ="singular_fit"
        if any(k in low for k in ("converg","gradient","hessian")): typ="nonconvergence"
        if any(k in low for k in ("divergent","treedepth","rhat","effective sample","ess")): typ="bayesian_sampling"
        if any(k in low for k in ("identif","rank deficient","not positive definite","non-positive")): typ="identifiability"
        if any(k in low for k in ("overflow","underflow","nan","infinite","non-finite")): typ="numerical"
        if any(k in low for k in ("memory","cannot allocate")): typ="resource"
        if any(k in low for k in ("package ","not installed","no module named")): typ="dependency"
        out.append({"message":m,"failure_type":typ})
    return pd.DataFrame(out)


def audit_identifiability(results: Any, max_missing: float=.05, max_sd_ratio: float=10, correlation_matrix: Any=None, max_abs_correlation: float=.995) -> pd.DataFrame:
    d=as_irt_recovery_results(results); rows=[]
    for p,z in d.groupby("parameter",sort=False):
        ts=float(z.truth.std(ddof=1)); es=float(z.estimate.std(ddof=1)); miss=float((~np.isfinite(z.estimate)).mean()); ratio=es/ts if np.isfinite(ts) and ts>0 else math.nan
        rows.append({"parameter":p,"missing_rate":miss,"truth_sd":ts,"estimate_sd":es,"sd_ratio":ratio,"pass":miss<=max_missing and (not np.isfinite(ratio) or (ratio<=max_sd_ratio and ratio>1e-6))})
    out=pd.DataFrame(rows); corr_issue=False; corr_max=math.nan
    if correlation_matrix is not None:
        C=np.asarray(correlation_matrix,float).copy(); np.fill_diagonal(C,np.nan); vals=np.abs(C[np.isfinite(C)]); corr_max=float(vals.max()) if vals.size else math.nan; corr_issue=np.isfinite(corr_max) and corr_max>max_abs_correlation
        if corr_issue: out["pass"]=False
    return _tag(out,"eye_irt_identifiability_audit",max_abs_parameter_correlation=corr_max,correlation_issue=corr_issue)


def validation_mcse(results: Any, metric: str="bias") -> pd.DataFrame:
    if metric not in {"bias","rmse","coverage"}: raise EyeProcessValidationError("metric must be bias, rmse, or coverage.")
    d=as_irt_recovery_results(results)
    if metric=="coverage":
        _req_cols(d,["lower","upper"],"results"); x=((pd.to_numeric(d.lower,errors="coerce")<=d.truth)&(d.truth<=pd.to_numeric(d.upper,errors="coerce"))).astype(float); p=float(x.mean()); n=int(np.isfinite(x).sum()); mc=math.sqrt(p*(1-p)/n) if n else math.nan; return pd.DataFrame([{"metric":metric,"estimate":p,"mcse":mc,"n":n}])
    e=(d.estimate-d.truth).to_numpy(float); e=e[np.isfinite(e)]; n=len(e)
    if metric=="bias": val=float(e.mean()) if n else math.nan; mc=float(e.std(ddof=1)/math.sqrt(n)) if n>1 else math.nan
    else:
        sq=e**2; val=float(np.sqrt(sq.mean())) if n else math.nan; mc=float(sq.std(ddof=1)/math.sqrt(n)/(2*val)) if n>1 and val>0 else (0.0 if val==0 else math.nan)
    return pd.DataFrame([{"metric":metric,"estimate":val,"mcse":mc,"n":n}])


def recommended_validation_replications(target_mcse: float=.01, metric: str="coverage", anticipated_sd: float=1, anticipated_probability: float=.95, minimum: int=100) -> int:
    if metric not in {"coverage","mean"}: raise EyeProcessValidationError("metric must be coverage or mean.")
    n=anticipated_probability*(1-anticipated_probability)/(target_mcse**2) if metric=="coverage" else anticipated_sd**2/(target_mcse**2); return int(max(minimum,math.ceil(n)))


def run_sbc(simulator: Callable[[int],Any], fitter: Callable[[Any],Any], posterior_draws: Callable[[Any],Any], replications: int=100, seed: int=20260808) -> EyeResult:
    if not all(callable(f) for f in (simulator,fitter,posterior_draws)): raise EyeProcessValidationError("simulator, fitter, and posterior_draws must be functions.")
    # Preserve deterministic Python callback behaviour. Replicate callbacks control their own RNG if desired.
    np.random.seed(seed); rows=[]; fails=[]
    for r in range(1,int(replications)+1):
        try:
            s=simulator(r)
            if not isinstance(s,Mapping) or "data" not in s or "truth" not in s: raise ValueError("simulator() must return list/data mapping with data and truth")
            truth=dict(s["truth"]); fit=fitter(s["data"]); dr=posterior_draws(fit); dr=pd.DataFrame(dr)
            keep=[p for p in truth if p in dr]
            if not keep: raise ValueError("No posterior-draw columns match truth names.")
            for p in keep:
                v=pd.to_numeric(dr[p],errors="coerce").to_numpy(float); v=v[np.isfinite(v)]; rk=int(np.sum(v<float(truth[p]))); rows.append({"replicate":r,"parameter":p,"truth":float(truth[p]),"rank":rk,"draws":len(v),"normalized_rank":(rk+.5)/(len(v)+1)})
        except Exception as e:
            z=validation_failure_taxonomy(e).iloc[0].to_dict(); z["replicate"]=r; fails.append(z)
    return _result("eye_irt_sbc",ranks=pd.DataFrame(rows),failures=pd.DataFrame(fails),replications=int(replications),seed=seed,method="prior_simulation_based_calibration")


def posterior_sbc_contract(replication: Callable[[int,Any],Any]) -> EyeResult:
    if not callable(replication): raise EyeProcessValidationError("replication must be a function.")
    return _result("eye_posterior_sbc_contract",replication=replication,requirement="Each replication must be generated from a posterior-SBC self-consistency experiment conditional on the observed data; ordinary prior SBC is not an acceptable substitute.")


def run_posterior_sbc(observed_data: Any, contract: Any, replications: int=100, seed: int=20260808) -> EyeResult:
    if getattr(contract,"eyeprocess_class",None)!="eye_posterior_sbc_contract": raise EyeProcessValidationError("contract must be created by posterior_sbc_contract().")
    np.random.seed(seed); rows=[]; fails=[]
    for r in range(1,int(replications)+1):
        try:
            z=contract.replication(r,observed_data); truth=dict(z["truth"]); dr=pd.DataFrame(z["draws"]); keep=[p for p in truth if p in dr]
            if not keep: raise ValueError("Posterior-SBC draws do not match named truth parameters.")
            for p in keep:
                v=pd.to_numeric(dr[p],errors="coerce").to_numpy(float); v=v[np.isfinite(v)]; rk=int(np.sum(v<float(truth[p]))); rows.append({"replicate":r,"parameter":p,"truth":float(truth[p]),"rank":rk,"draws":len(v),"normalized_rank":(rk+.5)/(len(v)+1)})
        except Exception as e:
            q=validation_failure_taxonomy(e).iloc[0].to_dict(); q["replicate"]=r; fails.append(q)
    out=_result("eye_posterior_sbc",ranks=pd.DataFrame(rows),failures=pd.DataFrame(fails),replications=int(replications),seed=seed,method="posterior_simulation_based_calibration",requirement=contract.requirement); out["superclass"]="eye_irt_sbc"; return out


def audit_sbc(x: Any, bins: int=10, alpha: float=.01) -> pd.DataFrame:
    d=x.ranks.copy() if getattr(x,"eyeprocess_class",None) in {"eye_irt_sbc","eye_posterior_sbc"} else _df(x,"x"); _req_cols(d,["parameter","normalized_rank"],"x"); rows=[]
    for p,z in d.groupby("parameter",sort=False):
        u=pd.to_numeric(z.normalized_rank,errors="coerce").to_numpy(float); u=u[np.isfinite(u)]; counts,_=np.histogram(u,bins=np.linspace(0,1,int(bins)+1)); expected=len(u)/int(bins) if bins else math.nan; chisq=float(np.sum((counts-expected)**2/expected)) if expected>0 else math.nan; df_=int(bins)-1; pv=float(chi2.sf(chisq,df_)) if np.isfinite(chisq) else math.nan
        rows.append({"parameter":p,"n":len(u),"mean_rank":float(np.mean(u)) if len(u) else math.nan,"rank_variance":float(np.var(u,ddof=1)) if len(u)>1 else math.nan,"expected_mean":.5,"expected_variance":1/12,"chisq":chisq,"df":df_,"p_value":pv,"alpha":alpha,"pass_screen":bool(np.isfinite(pv) and pv>=alpha)})
    return _tag(pd.DataFrame(rows),"eye_sbc_audit",bins=int(bins))


def posterior_predictive_discrepancies(observed: Any, replicated: Any, discrepancies: Mapping[str,Callable[[Any],float]]|None=None) -> pd.DataFrame:
    if discrepancies is None:
        discrepancies={"mean":lambda x:float(np.nanmean(x)),"sd":lambda x:float(np.nanstd(x,ddof=1)),"zero_rate":lambda x:float(np.nanmean(np.asarray(x)==0))}
    reps=[np.asarray(r) for r in (replicated if isinstance(replicated,list) else np.asarray(replicated))]
    if not reps: raise EyeProcessValidationError("replicated must contain datasets.")
    rows=[]
    for name,f in discrepancies.items():
        obs=float(f(observed)); rv=np.asarray([f(r) for r in reps],float); lo=float(np.nanmean(rv<=obs)); hi=float(np.nanmean(rv>=obs)); rows.append({"discrepancy":name,"observed":obs,"replicated_mean":float(np.nanmean(rv)),"replicated_sd":float(np.nanstd(rv,ddof=1)),"p_lower":lo,"p_upper":hi,"p_two_sided":min(1.0,2*min(lo,hi))})
    return _tag(pd.DataFrame(rows),"eye_irt_ppc")


def stress_test_misspecification(scenarios: Any, runner: Callable[[Any,int],Any], replications: int=50, seed: int=20260808) -> pd.DataFrame:
    if not callable(runner): raise EyeProcessValidationError("runner must be a function.")
    if isinstance(scenarios,Mapping): s=pd.DataFrame({"scenario":list(scenarios),"value":list(scenarios.values())})
    else: s=_df(scenarios,"scenarios")
    if "scenario" not in s: s["scenario"]=[f"scenario_{i+1}" for i in range(len(s))]
    np.random.seed(seed); rows=[]
    for _,sc in s.iterrows():
        for r in range(1,int(replications)+1):
            try:
                z=_df(runner(sc.to_frame().T,r),"runner result"); z["scenario"]=sc.scenario; z["replicate"]=r; z["failed"]=False; rows.append(z)
            except Exception as e:
                q=validation_failure_taxonomy(e); q["scenario"]=sc.scenario; q["replicate"]=r; q["failed"]=True; rows.append(q)
    return _tag(_bind(rows),"eye_irt_stress_test",seed=seed,replications=int(replications))


def stress_test_latent_distribution(runner: Callable, replications: int=50, seed: int=20260808) -> pd.DataFrame:
    x=["normal","skewed","student_t","mixture_normal","bimodal","heavy_tail"]; return stress_test_misspecification(pd.DataFrame({"scenario":x,"distribution":x}),runner,replications,seed)

def stress_test_local_dependence(runner: Callable, strengths: Sequence[float]=(0,.2,.5,.8), replications: int=50, seed: int=20260808) -> pd.DataFrame:
    return stress_test_misspecification(pd.DataFrame({"scenario":[f"local_dependence_{x}" for x in strengths],"strength":strengths}),runner,replications,seed)

def stress_test_speededness(runner: Callable, proportions: Sequence[float]=(0,.10,.25,.40), replications: int=50, seed: int=20260808) -> pd.DataFrame:
    return stress_test_misspecification(pd.DataFrame({"scenario":[f"speeded_{x}" for x in proportions],"speeded_proportion":proportions}),runner,replications,seed)

def stress_test_missingness(runner: Callable, mechanisms: Sequence[str] = ("MCAR","MAR","MNAR_omission","not_reached"), rates: Sequence[float] = (.05,.15,.30), replications: int=50, seed: int=20260808) -> pd.DataFrame:
    rows=[{"mechanism":m,"rate":r,"scenario":f"{m}_{r}"} for m in mechanisms for r in rates]; return stress_test_misspecification(pd.DataFrame(rows),runner,replications,seed)

def stress_test_preprocessing(runner: Callable, variants: Any, replications: int=25, seed: int=20260808) -> pd.DataFrame:
    s=pd.DataFrame({"scenario":list(variants),"preprocessing":list(variants)}) if isinstance(variants,(list,tuple)) and all(isinstance(x,str) for x in variants) else _df(variants,"variants")
    if "scenario" not in s: s["scenario"]=[f"preprocess_{i+1}" for i in range(len(s))]
    return stress_test_misspecification(s,runner,replications,seed)


def _leave_group_out(data: Any, group: str, fitter: Callable, predictor: Callable, scorer: Callable) -> pd.DataFrame:
    d=_df(data,"data"); _req_cols(d,[group],"data"); rows=[]
    for level in pd.unique(d[group]):
        test=d[d[group]==level].copy(); train=d[d[group]!=level].copy()
        try:
            sc=_df(scorer(test,predictor(fitter(train),test)),"score"); sc[group]=level; sc["n_train"]=len(train); sc["n_test"]=len(test); sc["failed"]=False; rows.append(sc)
        except Exception as e:
            q=validation_failure_taxonomy(e); q[group]=level; q["n_train"]=len(train); q["n_test"]=len(test); q["failed"]=True; rows.append(q)
    return _bind(rows)


def external_validate_irt(train_data: Any, external_data: Any, fitter: Callable, predictor: Callable, scorer: Callable, label: str="external") -> pd.DataFrame:
    try:
        sc=_df(scorer(external_data,predictor(fitter(train_data),external_data)),"score"); sc["validation_set"]=label; sc["n_train"]=len(train_data); sc["n_test"]=len(external_data); sc["failed"]=False
    except Exception as e:
        sc=validation_failure_taxonomy(e); sc["validation_set"]=label; sc["n_train"]=len(train_data); sc["n_test"]=len(external_data); sc["failed"]=True
    return _tag(sc,"eye_external_irt_validation")

def leave_device_out_validation(data: Any, device: str, fitter: Callable, predictor: Callable, scorer: Callable) -> pd.DataFrame: return _tag(_leave_group_out(data,device,fitter,predictor,scorer),"eye_leave_device_out_validation")
def leave_session_out_validation(data: Any, session: str, fitter: Callable, predictor: Callable, scorer: Callable) -> pd.DataFrame: return _tag(_leave_group_out(data,session,fitter,predictor,scorer),"eye_leave_session_out_validation")
def leave_site_out_validation(data: Any, site: str, fitter: Callable, predictor: Callable, scorer: Callable) -> pd.DataFrame: return _tag(_leave_group_out(data,site,fitter,predictor,scorer),"eye_leave_site_out_validation")
def leave_item_out_validation(data: Any, item: str, fitter: Callable, predictor: Callable, scorer: Callable) -> pd.DataFrame: return _tag(_leave_group_out(data,item,fitter,predictor,scorer),"eye_leave_item_out_validation")


def audit_measurement_transportability(validation: Any, metric: str, higher_is_better: bool=True, max_range: float|None=None, minimum: float|None=None, maximum: float|None=None) -> pd.DataFrame:
    d=_df(validation,"validation"); _req_cols(d,[metric],"validation"); x=pd.to_numeric(d[metric],errors="coerce").to_numpy(float); f=x[np.isfinite(x)]; rng=float(np.ptp(f)) if len(f) else math.nan; mean=float(np.mean(f)) if len(f) else math.nan; passed=True
    if max_range is not None: passed=passed and np.isfinite(rng) and rng<=max_range
    if minimum is not None: passed=passed and np.isfinite(mean) and mean>=minimum
    if maximum is not None: passed=passed and np.isfinite(mean) and mean<=maximum
    return _tag(pd.DataFrame([{"metric":metric,"n_groups":len(f),"mean":mean,"sd":float(np.std(f,ddof=1)) if len(f)>1 else math.nan,"min":float(np.min(f)) if len(f) else math.nan,"max":float(np.max(f)) if len(f) else math.nan,"range":rng,"higher_is_better":higher_is_better,"max_allowed_range":math.nan if max_range is None else max_range,"pass":bool(passed)}]),"eye_measurement_transportability_audit")


def compare_validation_engines(results: Any) -> pd.DataFrame:
    s=summarize_parameter_recovery(results,by=("engine","parameter")).sort_values(["parameter","rmse","absolute_bias","coverage"],ascending=[True,True,True,False],na_position="last").reset_index(drop=True); s["rank_within_parameter"]=s.groupby("parameter").cumcount()+1; return _tag(s,"eye_validation_engine_comparison")


def audit_channel_incremental_information(data: Any, fold: str, baseline_fitter: Callable, process_fitter: Callable, predictor: Callable, scorer: Callable, higher_is_better: bool=True) -> pd.DataFrame:
    d=_df(data,"data"); _req_cols(d,[fold],"data"); rows=[]
    for g in pd.unique(d[fold]):
        test=d[d[fold]==g]; train=d[d[fold]!=g]
        try:
            sb=float(np.asarray(scorer(test,predictor(baseline_fitter(train),test))).ravel()[0]); sp=float(np.asarray(scorer(test,predictor(process_fitter(train),test))).ravel()[0]); imp=sp-sb if higher_is_better else sb-sp; rows.append({"fold":str(g),"baseline":sb,"process":sp,"improvement":imp,"n_test":len(test),"failed":False})
        except Exception as e:
            q=validation_failure_taxonomy(e).iloc[0].to_dict(); q.update(fold=str(g),baseline=math.nan,process=math.nan,improvement=math.nan,n_test=len(test),failed=True); rows.append(q)
    out=pd.DataFrame(rows); return _tag(out,"eye_incremental_information_audit",mean_improvement=float(out.improvement.mean()),positive_fold_fraction=float((out.improvement>0).mean()))


def negative_control_process_test(data: Any, process_columns: str|Sequence[str], evaluator: Callable[[pd.DataFrame],float], within: str|Sequence[str]|None=None, permutations: int=100, higher_is_better: bool=True, seed: int=20260808) -> EyeResult:
    d=_df(data,"data"); cols=[process_columns] if isinstance(process_columns,str) else list(process_columns); _req_cols(d,cols,"data"); obs=float(evaluator(d)); rng=np.random.default_rng(seed); w=[] if within is None else ([within] if isinstance(within,str) else list(within)); _req_cols(d,w,"data") if w else None
    groups=[np.arange(len(d))] if not w else [np.asarray(v,int) for v in d.groupby(w,sort=False).groups.values()]; null=[]
    for _ in range(int(permutations)):
        z=d.copy()
        for c in cols:
            for idx in groups: z.loc[z.index[idx],c]=rng.permutation(z.iloc[idx][c].to_numpy())
        null.append(float(evaluator(z)))
    arr=np.asarray(null,float); p=(1+np.sum(arr>=obs))/(len(arr)+1) if higher_is_better else (1+np.sum(arr<=obs))/(len(arr)+1)
    return _result("eye_process_negative_control",observed=obs,null=arr,p_value=float(p),permutations=int(permutations),process_columns=cols,within=w or None,higher_is_better=higher_is_better,seed=seed)


def calibration_transfer_audit(data: Any, group: str, observed: str, predicted: str) -> pd.DataFrame:
    d=_df(data,"data"); _req_cols(d,[group,observed,predicted],"data"); rows=[]
    for g,z in d.groupby(group,dropna=False,sort=False):
        y=pd.to_numeric(z[observed],errors="coerce").to_numpy(float); p=pd.to_numeric(z[predicted],errors="coerce").to_numpy(float); ok=np.isfinite(y)&np.isfinite(p); y,p=y[ok],p[ok]
        if len(y)<3: rows.append({group:g,"n":len(y),"intercept":math.nan,"slope":math.nan,"brier":math.nan}); continue
        if np.all(np.isin(y,[0,1])) and np.all((p>0)&(p<1)):
            lp=np.log(p/(1-p)); X=np.column_stack([np.ones(len(p)),lp])
            # IRLS-ish logistic optimizer.
            from scipy.optimize import minimize
            def nll(b):
                pr=1/(1+np.exp(-np.clip(X@b,-40,40))); return -float(np.sum(y*np.log(np.clip(pr,1e-12,1))+(1-y)*np.log(np.clip(1-pr,1e-12,1))))
            co=minimize(nll,np.zeros(2),method="BFGS").x
        else:
            co=np.linalg.lstsq(np.column_stack([np.ones(len(p)),p]),y,rcond=None)[0]
        rows.append({group:g,"n":len(y),"intercept":float(co[0]),"slope":float(co[1]),"brier":float(np.mean((p-y)**2))})
    return _tag(pd.DataFrame(rows),"eye_calibration_transfer_audit")


def grade_model_evidence(recovery: Any, spec: Any=None, external_validation: Any=None, sbc: Any=None, ppc: Any=None, semantic_roundtrip: Any=None) -> EyeResult:
    spec=irt_validation_spec("unspecified") if spec is None else spec
    if getattr(spec,"eyeprocess_class",None)!="eye_irt_validation_spec": raise EyeProcessValidationError("spec must be an eye_irt_validation_spec.")
    s=recovery if isinstance(recovery,pd.DataFrame) and recovery.attrs.get("eyeprocess_class")=="eye_irt_recovery_summary" else summarize_parameter_recovery(recovery); th=spec.thresholds
    checks=[("absolute_bias",bool((s.absolute_bias<=th.get("max_abs_bias",.1)).all())),("rmse",bool((s.rmse<=th.get("max_rmse",.3)).all())),("coverage",bool(((s.coverage>=th.get("min_coverage",.9))|s.coverage.isna()).all())),("failure_rate",bool((s.failure_rate<=th.get("max_failure_rate",.05)).all()))]
    if sbc is not None:
        sa=sbc if isinstance(sbc,pd.DataFrame) and sbc.attrs.get("eyeprocess_class")=="eye_sbc_audit" else audit_sbc(sbc); checks.append(("sbc_screen",bool(sa.pass_screen.fillna(False).all())))
    if ppc is not None:
        pdx=_df(ppc,"ppc"); checks.append(("ppc_extremes",bool((pdx.p_two_sided>=.01).all()) if "p_two_sided" in pdx else False))
    if external_validation is not None:
        ed=_df(external_validation,"external_validation"); nf=int((~ed.failed).sum()) if "failed" in ed else len(ed); checks.append(("external_folds",nf>=th.get("min_external_folds",2)))
    if semantic_roundtrip is not None:
        rd=semantic_roundtrip if isinstance(semantic_roundtrip,pd.DataFrame) else getattr(semantic_roundtrip,"field_fidelity",pd.DataFrame()); passed=len(rd)>0 and ("status" not in rd or not rd.status.astype(str).str.upper().isin(["UNSUPPORTED","AMBIGUOUS"]).any()); checks.append(("semantic_roundtrip",bool(passed)))
    c=pd.DataFrame(checks,columns=["criterion","pass"]); n=int(c["pass"].sum()); total=len(c); grade="strong_validation_evidence" if n==total and total>=6 else ("moderate_validation_evidence" if n==total else ("provisional_validation_evidence" if n>=math.ceil(total*.75) else "insufficient_validation_evidence"))
    return _result("eye_irt_evidence_grade",model_id=spec.model_id,grade=grade,checks=c,recovery=s,contract=spec,warning="Evidence grades summarize the supplied validation programme; they are not a substitute for substantive validity, independent replication, or model-specific scientific judgment.")
