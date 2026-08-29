"""Explicit 0.7 requested-API completion layer.

This module composes previously ported process/IRT primitives and adds the
remaining semantic, validation, event-time and plotting contracts from
R/055-requested-api-completion-0-7.R.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, t as student_t

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _as_df, _req_cols, _result
from .process_irt_07 import (
    distractor_process_map,
    detect_irt_changepoints,
    facet_effects,
    simulate_irt_model,
)
from .irt_validation_07 import stress_test_latent_distribution
from .irt_validation_07 import as_irt_recovery_results, validation_failure_taxonomy
from .semantic_validation_07 import (
    event_semantics_audit,
    field_fidelity_report,
    semantic_roundtrip_audit,
    validate_hed_event_semantics,
    validate_vendor_timestamp_semantics,
)

__all__ = [
    "plot_distractor_information", "fit_gaze_informed_missingness_irt",
    "device_facet_effects", "session_facet_effects", "algorithm_facet_effects",
    "detect_process_changepoint", "plot_process_changepoint", "plot_person_item_space",
    "explain_latent_interaction", "plot_irf_uncertainty", "audit_latent_distribution",
    "compare_latent_distribution_models", "latent_distribution_stress_test",
    "fit_event_time_irt", "simulate_from_model", "extract_parameter_truth",
    "fit_validation_replicate", "vendor_schema_contract", "validate_vendor_semantics",
    "event_roundtrip_audit", "roundtrip_eye_bids", "cross_version_adapter_regression",
]


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    return _as_df(x, name)


def _dummy_matrix(d: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones(len(d))]; names = ["Intercept"]
    for c in continuous:
        cols.append(pd.to_numeric(d[c], errors="coerce").to_numpy(float)); names.append(c)
    for c in categorical:
        levels = pd.Series(d[c].astype("string")).dropna().unique().tolist()
        for lv in levels[1:]:
            cols.append((d[c].astype("string") == lv).to_numpy(float)); names.append(f"{c}[{lv}]")
    return np.column_stack(cols), names


def _logistic_fit(X: np.ndarray, y: np.ndarray) -> EyeResult:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1); X = X[ok]; y = y[ok]
    if len(y) == 0: raise EyeProcessValidationError("No complete observations are available for logistic fitting.")
    def nll(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))
    res = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
    return _result("eye_reference_logistic_fit", coefficients=np.asarray(res.x,float), fitted=expit(X@res.x), converged=bool(res.success), logLik=-float(res.fun), algorithmic_parity=True)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> EyeResult:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1); X = X[ok]; y = y[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); fitted=X@beta; resid=y-fitted
    return _result("eye_reference_linear_fit", coefficients=beta, fitted=fitted, residuals=resid, converged=True, algorithmic_parity=True)


def plot_distractor_information(object: Any, ax: Any = None):
    import matplotlib.pyplot as plt
    z = distractor_process_map(object) if getattr(object,"eyeprocess_class",None)=="eye_nominal_gaze_irt" else _df(object,"object")
    ax = plt.subplots()[1] if ax is None else ax
    if z.empty:
        ax.set_title("Distractor process information"); ax.text(.5,.5,"No distractor coefficients available",ha="center")
    elif {"gaze_contrast","choice_contrast"}.issubset(z.columns):
        ax.scatter(z.gaze_contrast,z.choice_contrast); ax.axhline(0,ls="--"); ax.axvline(0,ls="--"); ax.set_xlabel("Gaze contrast"); ax.set_ylabel("Choice contrast"); ax.set_title("Distractor process information")
        if "option" in z:
            for _,r in z.iterrows(): ax.annotate(str(r.option),(r.gaze_contrast,r.choice_contrast))
    elif {"response_category","gaze_channel","coefficient"}.issubset(z.columns):
        labels=(z.response_category.astype(str)+" / "+z.gaze_channel.astype(str)).tolist(); y=np.arange(len(z)); ax.scatter(z.coefficient,y); ax.set_yticks(y,labels); ax.axvline(0,ls="--"); ax.set_xlabel("Coefficient")
    else:
        nums=z.select_dtypes(include=np.number).columns.tolist()
        if not nums: raise EyeProcessValidationError("No numeric distractor-information field is available to plot.")
        y=np.arange(len(z)); ax.scatter(z[nums[0]],y); ax.set_xlabel(nums[0])
    ax.gp3_data=z.copy(); return ax


def fit_gaze_informed_missingness_irt(data: Any, response: str="response", person: str="participant_id",
                                      item: str="item_id", gaze_exposure: str="gaze_exposure",
                                      theta: str|None=None, reached: str|None=None) -> EyeResult:
    cols=[response,person,item,gaze_exposure]+([theta] if theta else [])+([reached] if reached else []); d=_df(data,"data").copy(); _req_cols(d,cols,"data")
    ge=pd.to_numeric(d[gaze_exposure],errors="coerce").to_numpy(float)
    if np.any(ge[np.isfinite(ge)]<0): raise EyeProcessValidationError("gaze_exposure must be non-negative.")
    d[".missing"]=d[response].isna().astype(int); d[".log_exposure"]=np.log1p(ge)
    theta_source="supplied"
    if theta is None:
        theta_source="smoothed-person-score-proxy"; y=pd.to_numeric(d[response],errors="coerce")
        finite=y.dropna()
        if not set(finite.unique()).issubset({0,1}): raise EyeProcessValidationError("Without theta, response must be binary.")
        agg=d.assign(_y=y).dropna(subset=[response]).groupby(person,dropna=False)._y.agg(["sum","count"])
        p=(agg["sum"]+.5)/(agg["count"]+1); tm=np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5))); d[".theta"]=d[person].map(tm)
    else: d[".theta"]=pd.to_numeric(d[theta],errors="coerce")
    X,names=_dummy_matrix(d,[".theta",".log_exposure"],[item]); mm=_logistic_fit(X,d[".missing"].to_numpy(float)); mm["design_names"]=names
    obs=d[d[response].notna()].copy(); rm=None
    if len(obs):
        Xo,no=_dummy_matrix(obs,[".theta",".log_exposure"],[item]); yo=pd.to_numeric(obs[response],errors="coerce").to_numpy(float); rm=_logistic_fit(Xo,yo) if set(yo[np.isfinite(yo)]).issubset({0,1}) else _ols_fit(Xo,yo); rm["design_names"]=no
    rs=None
    if reached:
        r=d[reached].astype("boolean"); rows=[]
        for val in [False,True]:
            mask=r.eq(val).fillna(False); rows.append({"reached":val,"n":int(mask.sum()),"missing_rate":float(d.loc[mask,".missing"].mean()) if mask.any() else math.nan})
        rs=pd.DataFrame(rows)
    return _result("eye_gaze_informed_missingness_irt",missingness_model=mm,response_model=rm,theta_source=theta_source,reached_summary=rs,data=d,status="reference-diagnostic",note="Two-part conditional diagnostic; it does not establish MAR/MNAR status and is not a full joint latent missingness IRT estimator.")


def _named_facet_effects(object: Any, facet: str, channel: str="response") -> EyeResult:
    if getattr(object,"eyeprocess_class",None)!="eye_manyfacet_process_irt": raise EyeProcessValidationError("object must come from fit_manyfacet_process_irt().")
    if facet not in object.facets: raise EyeProcessValidationError(f"Facet {facet} was not included in the fitted model.")
    ef=facet_effects(object,channel=channel); col=object.facets[facet]; re=ef.random_effects.get(facet,ef.random_effects.get(col)); vc=ef.variance_components.copy(); vc=vc[(vc.grp==facet)|(vc.var1==col)]
    return _result("eye_process_facet_effects",facet=facet,column=col,channel=channel,random_effects=re,variance_component=vc)

def device_facet_effects(object: Any, channel: str="response") -> EyeResult: return _named_facet_effects(object,"device",channel)
def session_facet_effects(object: Any, channel: str="response") -> EyeResult: return _named_facet_effects(object,"session",channel)
def algorithm_facet_effects(object: Any, channel: str="response") -> EyeResult: return _named_facet_effects(object,"algorithm",channel)
def detect_process_changepoint(*args: Any, **kwargs: Any) -> EyeResult: return detect_irt_changepoints(*args,**kwargs)


def plot_process_changepoint(object: Any, ax: Any=None):
    import matplotlib.pyplot as plt
    cp=object if getattr(object,"eyeprocess_class",None)=="eye_irt_changepoints" else getattr(object,"changepoints",None)
    if getattr(cp,"eyeprocess_class",None)!="eye_irt_changepoints": raise EyeProcessValidationError("Supply a process/change-point object returned by eyeprocesspy.")
    z=cp.results.copy(); ax=plt.subplots()[1] if ax is None else ax
    if z.empty: ax.set_title("Process change points")
    else:
        y=pd.to_numeric(z.changepoint_order,errors="coerce"); det=z.detected.astype(bool).to_numpy(); ax.scatter(np.arange(1,len(z)+1)[~det],y[~det],facecolors="none",edgecolors="black"); ax.scatter(np.arange(1,len(z)+1)[det],y[det]); ax.set_xlabel("Participant"); ax.set_ylabel("Estimated change-point order"); ax.set_title("Detected process change points")
    ax.gp3_data=z; return ax


def plot_person_item_space(object: Any, dimensions: Sequence[int]=(1,2), labels: bool=False, ax: Any=None):
    import matplotlib.pyplot as plt
    if getattr(object,"eyeprocess_class",None)!="eye_latent_space_irt": raise EyeProcessValidationError("object must come from fit_latent_space_irt().")
    dims=[int(x) for x in dimensions]
    if len(dims)!=2 or min(dims)<1: raise EyeProcessValidationError("dimensions must contain two positive indices.")
    p=np.asarray(object.person_coordinates,float); i=np.asarray(object.item_coordinates,float); a,b=dims[0]-1,dims[1]-1
    if max(a,b)>=p.shape[1] or max(a,b)>=i.shape[1]: raise EyeProcessValidationError("Requested latent-space dimension is unavailable.")
    ax=plt.subplots()[1] if ax is None else ax; ax.scatter(p[:,a],p[:,b],facecolors="none",edgecolors="black"); ax.scatter(i[:,a],i[:,b],marker="x"); ax.set_xlabel(f"Latent-space dimension {dims[0]}"); ax.set_ylabel(f"Latent-space dimension {dims[1]}"); ax.set_title("Person-item latent space")
    if labels:
        pn=list(getattr(object,"person_ids",[str(j+1) for j in range(len(p))])); inn=list(getattr(object,"item_ids",[f"I{j+1}" for j in range(len(i))]))
        for q,n in zip(p,pn): ax.annotate(str(n),(q[a],q[b]));
        for q,n in zip(i,inn): ax.annotate(str(n),(q[a],q[b]))
    ax.gp3_data={"person":p,"item":i,"dimensions":dims}; return ax


def explain_latent_interaction(object: Any, person: Any=None, item: Any=None, top: int=10) -> pd.DataFrame:
    if getattr(object,"eyeprocess_class",None)!="eye_latent_space_irt": raise EyeProcessValidationError("object must come from fit_latent_space_irt().")
    p=np.asarray(object.person_coordinates,float); it=np.asarray(object.item_coordinates,float)
    if p.shape[1]!=it.shape[1]: raise EyeProcessValidationError("Person/item coordinates have incompatible dimensions.")
    pn=list(map(str,getattr(object,"person_ids",range(1,len(p)+1)))); inn=list(map(str,getattr(object,"item_ids",range(1,len(it)+1))))
    def picks(x,names,n):
        if x is None:return list(range(n))
        xs=[x] if np.isscalar(x) else list(x); out=[]
        for z in xs:
            if isinstance(z,(int,np.integer)) and 1<=int(z)<=n: out.append(int(z)-1)
            elif str(z) in names: out.append(names.index(str(z)))
        return out
    pi,ii=picks(person,pn,len(p)),picks(item,inn,len(it))
    if not pi or not ii: raise EyeProcessValidationError("No matching person/item coordinates.")
    rows=[]
    for a in pi:
        for b in ii: rows.append({"person":pn[a],"item":inn[b],"distance":float(np.linalg.norm(it[b]-p[a]))})
    return pd.DataFrame(rows).sort_values("distance",kind="stable").head(max(1,int(top))).reset_index(drop=True)


def plot_irf_uncertainty(object: Any, item: int|str=1, theta_grid: Sequence[float]|None=None, level: float=.95, ax: Any=None):
    import matplotlib.pyplot as plt
    if getattr(object,"eyeprocess_class",None)!="eye_gpirt": raise EyeProcessValidationError("object must come from fit_gpirt().")
    if object.engine!="spline_reference": raise EyeProcessBackendError("External GPIRT engines do not expose a common posterior prediction contract.")
    names=list(object.item_names); idx=names.index(item) if isinstance(item,str) and item in names else int(item)-1
    if idx<0 or idx>=len(object.models): raise EyeProcessValidationError("Unknown item.")
    grid=np.linspace(-4,4,101) if theta_grid is None else np.asarray(theta_grid,float); fit=object.models[idx]; deg=int(fit.theta_degree); D=np.column_stack([np.ones(len(grid))]+[grid**k for k in range(1,deg+1)]); eta=D@np.asarray(fit.coefficients,float); est=expit(eta)
    # R uses GLM link-scale SE; Python reference uses Hessian covariance when available, else no-width line.
    se=np.zeros(len(grid)); cov=getattr(fit,"covariance",None)
    if cov is not None:
        C=np.asarray(cov,float)
        if C.shape==(D.shape[1],D.shape[1]): se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',D,C,D),0))
    z=float(norm.ppf(1-(1-level)/2)); tab=pd.DataFrame({"theta":grid,"estimate":est,"lower":expit(eta-z*se),"upper":expit(eta+z*se)})
    ax=plt.subplots()[1] if ax is None else ax; ax.plot(tab.theta,tab.estimate); ax.plot(tab.theta,tab.lower,ls="--"); ax.plot(tab.theta,tab.upper,ls="--"); ax.set_ylim(0,1); ax.set_xlabel("theta"); ax.set_ylabel("Response probability"); ax.set_title(f"Flexible IRF: {names[idx]}"); ax.gp3_data=tab; return ax


def audit_latent_distribution(theta: Any, tail_z: float=3) -> pd.DataFrame:
    x=np.asarray(theta,float); x=x[np.isfinite(x)]
    if len(x)<8: raise EyeProcessValidationError("At least eight finite latent-trait values are required.")
    mu=float(np.mean(x)); s=float(np.std(x,ddof=1))
    if not np.isfinite(s) or s<=0: raise EyeProcessValidationError("Latent-trait variance must be positive.")
    z=(x-mu)/s; skew=float(np.mean(z**3)); ex=float(np.mean(z**4)-3); bc=(skew**2+1)/max(ex+3,np.finfo(float).eps); theo=norm.ppf((np.arange(1,len(x)+1)-.375)/(len(x)+.25)); qq=float(np.corrcoef(np.sort(z),theo)[0,1])
    out=pd.DataFrame([{"n":len(x),"mean":mu,"sd":s,"skewness":skew,"excess_kurtosis":ex,"tail_rate":float(np.mean(np.abs(z)>tail_z)),"tail_z":tail_z,"bimodality_coefficient":bc,"normal_qq_correlation":qq}]); out.attrs["eyeprocess_class"]="eye_latent_distribution_audit"; return out


def compare_latent_distribution_models(theta: Any) -> EyeResult:
    x=np.asarray(theta,float); x=x[np.isfinite(x)]
    if len(x)<20: raise EyeProcessValidationError("At least 20 finite values are recommended for distribution comparison.")
    n=len(x); mu=float(np.mean(x)); sd=float(np.std(x,ddof=1)); ll_n=float(np.sum(norm.logpdf(x,loc=mu,scale=sd)))
    def t_nll(par):
        m,ls,ld=par; s=math.exp(ls); df=2+math.exp(ld); return float(-np.sum(student_t.logpdf((x-m)/s,df=df)-math.log(s)))
    rt=minimize(t_nll,[mu,math.log(sd),math.log(6)],method="BFGS"); tpar={"location":float(rt.x[0]),"scale":math.exp(float(rt.x[1])),"df":2+math.exp(float(rt.x[2]))}
    q=np.quantile(x,[.3,.7]); means=q.astype(float); sig=np.repeat(sd,2); w=np.array([.5,.5]); ll_old=-np.inf
    for _ in range(250):
        dens=np.column_stack([w[k]*norm.pdf(x,means[k],max(sig[k],1e-8)) for k in range(2)]); den=np.maximum(dens.sum(1),np.finfo(float).tiny); resp=dens/den[:,None]; nk=resp.sum(0); w=nk/n; means=(resp*x[:,None]).sum(0)/np.maximum(nk,1e-8); sig=np.sqrt((resp*(x[:,None]-means[None,:])**2).sum(0)/np.maximum(nk,1e-8)); sig=np.maximum(sig,1e-6); ll=float(np.sum(np.log(den))); 
        if np.isfinite(ll_old) and abs(ll-ll_old)<1e-8: break
        ll_old=ll
    tab=pd.DataFrame({"model":["normal","student_t","two_normal_mixture"],"logLik":[ll_n,-float(rt.fun),ll],"k":[2,3,5],"converged":[True,bool(rt.success),np.isfinite(ll)]}); tab["AIC"]=-2*tab.logLik+2*tab.k; tab["BIC"]=-2*tab.logLik+math.log(n)*tab.k; tab["delta_AIC"]=tab.AIC-tab.AIC.min(); tab["delta_BIC"]=tab.BIC-tab.BIC.min(); tab=tab.sort_values("BIC",kind="stable").reset_index(drop=True)
    mix={"weight1":float(w[0]),"mean1":float(means[0]),"sd1":float(sig[0]),"weight2":float(w[1]),"mean2":float(means[1]),"sd2":float(sig[1])}
    return _result("eye_latent_distribution_comparison",comparison=tab,audit=audit_latent_distribution(x),student_t=tpar,mixture=mix,status="distribution-stress-test")


def latent_distribution_stress_test(*args: Any, **kwargs: Any): return stress_test_latent_distribution(*args,**kwargs)


def fit_event_time_irt(data: Any,event_time: str="event_time",event: str="event",theta: str="theta",person: str="participant_id",item: str="item_id",engine: str="cox_reference",external_engine: Callable[...,Any]|None=None,**kwargs: Any) -> EyeResult:
    if engine=="external":
        if not callable(external_engine): raise EyeProcessBackendError("Supply a validated event-time IRT fitter through external_engine.")
        return _result("eye_event_time_irt",model=external_engine(data=data,**kwargs),engine="external",status="experimental-gated")
    if engine!="cox_reference": raise EyeProcessValidationError("engine must be cox_reference or external.")
    d=_df(data,"data").copy(); _req_cols(d,[event_time,event,theta,person,item],"data"); t=pd.to_numeric(d[event_time],errors="coerce").to_numpy(float); e=d[event].astype(bool).to_numpy(int); th=pd.to_numeric(d[theta],errors="coerce").to_numpy(float)
    if np.any(t[np.isfinite(t)]<0): raise EyeProcessValidationError("Event times must be non-negative.")
    X,names=_dummy_matrix(d,[theta],[item]); X=X[:,1:]; names=names[1:]
    ok=np.isfinite(t)&np.isfinite(th)&np.all(np.isfinite(X),axis=1); t=t[ok]; e=e[ok]; X=X[ok]
    def nll(beta):
        eta=X@beta; val=0.0
        for i in np.where(e==1)[0]:
            risk=t>=t[i]; val += eta[i]-np.log(np.exp(np.clip(eta[risk],-700,700)).sum())
        return -float(val)
    res=minimize(nll,np.zeros(X.shape[1]),method="BFGS"); model=_result("eye_cox_reference_fit",coefficients=pd.DataFrame({"term":names,"estimate":res.x}),converged=bool(res.success),logLik=-float(res.fun),robust_cluster_se=False)
    return _result("eye_event_time_irt",model=model,engine="cox_reference",theta_conditioned=True,status="experimental-reference",algorithmic_parity=False,note="Conditioned-on-theta Cox partial-likelihood diagnostic; clustered robust SE from R survival::coxph are not reproduced in the dependency-light Python core.")


def simulate_from_model(model: Any, **kwargs: Any):
    if callable(model): return model(**kwargs)
    if isinstance(model,str) or getattr(model,"eyeprocess_class",None)=="eye_irt_model_spec": return simulate_irt_model(model,**kwargs)
    if isinstance(model,Mapping) and callable(model.get("simulate_fun")): return model["simulate_fun"](**kwargs)
    raise EyeProcessValidationError("Supply a registered IRT model/specification or a simulation function.")


def extract_parameter_truth(simulation: Any) -> pd.DataFrame:
    src=simulation.get("truth") if isinstance(simulation,Mapping) and simulation.get("truth") is not None else simulation.get("parameters") if isinstance(simulation,Mapping) and simulation.get("parameters") is not None else getattr(simulation,"truth",None)
    if src is None: raise EyeProcessValidationError("Simulation object does not expose parameter truth.")
    if isinstance(src,pd.DataFrame):
        _req_cols(src,["parameter","truth"],"truth"); return src[["parameter","truth"]].copy()
    if isinstance(src,Mapping): return pd.DataFrame({"parameter":[str(k) for k in src],"truth":[float(v) for v in src.values()]})
    arr=np.asarray(src,float).ravel(); return pd.DataFrame({"parameter":[f"parameter_{i+1}" for i in range(len(arr))],"truth":arr})


def fit_validation_replicate(replicate: int,generator: Callable[...,Any],fitter: Callable[...,Any],extractor: Callable[...,Any],scenario: Any="baseline",engine: str="unspecified") -> pd.DataFrame:
    if not all(callable(z) for z in (generator,fitter,extractor)): raise EyeProcessValidationError("generator, fitter, and extractor must be functions.")
    try:
        sim=generator(replicate=replicate,scenario=scenario); truth=extract_parameter_truth(sim); dat=sim.get("data",sim) if isinstance(sim,Mapping) else sim; fit=fitter(dat); est=_df(extractor(fit,sim),"extractor result"); _req_cols(est,["parameter","estimate"],"extractor result"); out=truth.merge(est,on="parameter",how="outer",sort=False); out["replicate"]=replicate; out["scenario"]=str(scenario) if np.isscalar(scenario) else "custom"; out["engine"]=engine; out["converged"]=True; return as_irt_recovery_results(out)
    except Exception as exc:
        fail=validation_failure_taxonomy(exc).iloc[0]; out=pd.DataFrame([{"replicate":replicate,"parameter":pd.NA,"truth":math.nan,"estimate":math.nan,"scenario":str(scenario) if np.isscalar(scenario) else "custom","engine":engine,"converged":False,"failure_type":fail.failure_type,"failure_message":fail.message}]); out.attrs["eyeprocess_class"]="eye_irt_recovery_results"; return out


def vendor_schema_contract(vendor: str,version: str|None=None,required_fields: Sequence[str]=(),optional_fields: Sequence[str]=(),aliases: Mapping[str,Sequence[str]]|None=None,timestamp: Mapping[str,Any]|None=None,coordinate: Mapping[str,Any]|None=None,units: Mapping[str,Any]|None=None,eye_streams: Sequence[str]=(),event_fields: Sequence[str]=()) -> EyeResult:
    if not isinstance(vendor,str) or not vendor: raise EyeProcessValidationError("vendor must be one non-missing character value.")
    return _result("eye_vendor_schema_contract",vendor=vendor,version=version,required_fields=list(dict.fromkeys(required_fields)),optional_fields=list(dict.fromkeys(optional_fields)),aliases=dict(aliases or {}),timestamp=dict(timestamp or {}),coordinate=dict(coordinate or {}),units=dict(units or {}),eye_streams=list(dict.fromkeys(eye_streams)),event_fields=list(dict.fromkeys(event_fields)),contract_version="0.7.0")


def validate_vendor_semantics(data: Any,contract: Any,metadata: Mapping[str,Any]|None=None) -> EyeResult:
    d=_df(data,"data"); metadata=dict(metadata or {})
    if getattr(contract,"eyeprocess_class",None)!="eye_vendor_schema_contract": raise EyeProcessValidationError("contract must come from vendor_schema_contract().")
    fields=pd.DataFrame([{"field":f,"present":f in d,"required":True} for f in contract.required_fields]+[{"field":f,"present":f in d,"required":False} for f in contract.optional_fields])
    aliases=pd.DataFrame([{"canonical":c,"matched":any(x in d for x in [c,*map(str,v)]),"matched_field":", ".join(x for x in [c,*map(str,v)] if x in d)} for c,v in contract.aliases.items()]) if contract.aliases else pd.DataFrame()
    ts=contract.timestamp; ta=validate_vendor_timestamp_semantics(d,contract.vendor,device_time=ts.get("device_time",ts.get("native_time")),system_time=ts.get("system_time"),media_time=ts.get("media_time")) if ts else None
    unit_rows=[]
    for nm,expected in contract.units.items():
        mf=metadata.get(nm); observed=mf.get("Units") if isinstance(mf,Mapping) else metadata.get(f"{nm}_units"); unit_rows.append({"field":nm,"expected_unit":str(expected),"observed_unit":observed,"pass":observed is not None and str(observed).lower()==str(expected).lower()})
    units=pd.DataFrame(unit_rows); tpass=True if ta is None else bool(ta["pass"]); pass_=bool((fields.loc[fields.required,"present"].all() if len(fields) else True) and (aliases.matched.all() if len(aliases) else True) and tpass and (units["pass"].all() if len(units) else True)); out=_result("eye_vendor_semantic_validation",vendor=contract.vendor,version=contract.version,fields=fields,aliases=aliases,timestamp=ta,units=units,contract=contract); out["pass"]=pass_; return out


def event_roundtrip_audit(source_events: Any,roundtrip_events: Any,hed_column: str|None=None,**kwargs: Any) -> EyeResult:
    core=event_semantics_audit(source_events,roundtrip_events,**kwargs); hed=None
    s,r=_df(source_events,"source_events"),_df(roundtrip_events,"roundtrip_events")
    if hed_column and hed_column in s and hed_column in r:
        a=validate_hed_event_semantics(s,hed_column); b=validate_hed_event_semantics(r,hed_column); hed={"source":a,"roundtrip":b,"valid_fraction_source":float(a.structurally_valid.mean()),"valid_fraction_roundtrip":float(b.structurally_valid.mean())}
    return _result("eye_event_roundtrip_audit",status=core.status,event_semantics=core,hed=hed)


def _extract_samples(x: Any) -> pd.DataFrame:
    if isinstance(x,pd.DataFrame): return x.copy()
    if isinstance(x,Mapping):
        for k in ("samples","data","gaze"):
            if isinstance(x.get(k),pd.DataFrame): return x[k].copy()
    return _df(x,"object")


def roundtrip_eye_bids(source: Any,exporter: Callable[...,Any],importer: Callable[...,Any],export_args: Mapping[str,Any]|None=None,import_args: Mapping[str,Any]|None=None,extract_samples: Callable[[Any],pd.DataFrame]=_extract_samples,audit_args: Mapping[str,Any]|None=None) -> EyeResult:
    if not all(callable(z) for z in (exporter,importer,extract_samples)): raise EyeProcessValidationError("exporter, importer, and extract_samples must be functions.")
    exported=exporter(source,**dict(export_args or {})); reconstructed=importer(exported,**dict(import_args or {})); audit=semantic_roundtrip_audit(extract_samples(source),extract_samples(reconstructed),**dict(audit_args or {})); return _result("eye_bids_roundtrip",exported=exported,reconstructed=reconstructed,audit=audit,status=audit.overall)


def cross_version_adapter_regression(input: Any,baseline_adapter: Callable[...,Any],candidate_adapter: Callable[...,Any],baseline_version: str="baseline",candidate_version: str="candidate",extract_samples: Callable[[Any],pd.DataFrame]=_extract_samples,audit_args: Mapping[str,Any]|None=None) -> EyeResult:
    if not all(callable(z) for z in (baseline_adapter,candidate_adapter,extract_samples)): raise EyeProcessValidationError("Adapter and extractor arguments must be functions.")
    old,new=baseline_adapter(input),candidate_adapter(input); fidelity=field_fidelity_report(extract_samples(old),extract_samples(new),**dict(audit_args or {})); st=fidelity.fields.status.tolist(); good={"LOSSLESS","UNIT_TRANSFORMED","COORDINATE_TRANSFORMED","SEMANTICALLY_EQUIVALENT"}; status="LOSSLESS" if all(s=="LOSSLESS" for s in st) else "SEMANTICALLY_EQUIVALENT" if all(s in good for s in st) else "REGRESSION_OR_AMBIGUOUS"; return _result("eye_adapter_regression_audit",status=status,baseline_version=baseline_version,candidate_version=candidate_version,fidelity=fidelity,baseline=old,candidate=new)
