"""Measurement-intelligence parity for frozen eyeprocess 0.11.1.

Ports cross-device linking, multi-objective item-bank optimization, dynamic
process-DIF/fairness summaries, and conditional process norms from R sources
036, 043, 044, and 045.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, t as student_t

from .exceptions import EyeProcessValidationError
from .irt import EyeResult, _result


def _df(x: Any, name: str="x") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame): return x.copy()
    try: return pd.DataFrame(x)
    except Exception as exc: raise EyeProcessValidationError(f"`{name}` must be a data frame.") from exc


def _req(d: pd.DataFrame, cols: Sequence[str]) -> None:
    miss=[c for c in cols if c not in d.columns]
    if miss: raise EyeProcessValidationError("Missing required columns: "+", ".join(miss))


def _z(x: Any) -> np.ndarray:
    a=pd.to_numeric(pd.Series(x),errors="coerce").to_numpy(float); m=np.nanmean(a); s=np.nanstd(a,ddof=1)
    return (a-m)/s if np.isfinite(s) and s>0 else np.zeros_like(a)


def _safe_mean(x: Any) -> float:
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]; return float(a.mean()) if a.size else np.nan


def _safe_sd(x: Any) -> float:
    a=np.asarray(x,dtype=float); a=a[np.isfinite(a)]; return float(a.std(ddof=1)) if a.size>1 else np.nan


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    keep=np.isfinite(y)&np.all(np.isfinite(x),axis=1); X=x[keep]; Y=y[keep]
    if len(Y)==0: return {"coef":np.full(x.shape[1],np.nan),"residual_sd":np.nan,"n":0}
    beta=np.linalg.lstsq(X,Y,rcond=None)[0]; resid=Y-X@beta; df=max(1,len(Y)-X.shape[1]); rss=float(resid@resid); sigma2=rss/df
    cov=np.linalg.pinv(X.T@X)*sigma2; se=np.sqrt(np.maximum(np.diag(cov),0)); stat=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0); p=2*student_t.sf(np.abs(stat),df)
    return {"coef":beta,"se":se,"p":p,"residual_sd":float(np.sqrt(sigma2)),"n":len(Y)}


def _logistic_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    keep=np.isfinite(y)&np.all(np.isfinite(x),axis=1); X=x[keep]; Y=y[keep]
    if len(Y)==0: return {"coef":np.full(x.shape[1],np.nan),"se":np.full(x.shape[1],np.nan),"p":np.full(x.shape[1],np.nan),"n":0}
    def objective(b):
        eta=X@b; return float(np.sum(np.logaddexp(0,eta)-Y*eta)+1e-8*np.sum(b[1:]**2))
    fit=minimize(objective,np.zeros(X.shape[1]),method="BFGS")
    b=np.asarray(fit.x); pr=np.clip(expit(X@b),1e-8,1-1e-8); H=X.T@(X*(pr*(1-pr))[:,None]); H[1:,1:]+=np.eye(X.shape[1]-1)*2e-8
    cov=np.linalg.pinv(H); se=np.sqrt(np.maximum(np.diag(cov),0)); z=np.divide(b,se,out=np.full_like(b,np.nan),where=se>0); p=2*norm.sf(np.abs(z))
    return {"coef":b,"se":se,"p":p,"n":len(Y),"converged":bool(fit.success)}


# ---------------------------------------------------------------------------
# 036 device linking
# ---------------------------------------------------------------------------

def fit_device_linking(x: Any, metric: str, reference_device: str, method: str="mixed_bland_altman", device_col: str="device", id_cols: Sequence[str]=("person_id","item_id")) -> EyeResult:
    if method not in {"mixed_bland_altman","hierarchical","equipercentile"}: raise EyeProcessValidationError("invalid linking method.")
    d=_df(x); ids=[c for c in id_cols if c in d.columns];
    if not ids: raise EyeProcessValidationError("At least one pairing identifier is required.")
    _req(d,[metric,device_col,*ids]); devices=d[device_col].astype(str).unique().tolist()
    if reference_device not in devices: raise EyeProcessValidationError("`reference_device` is absent from the data.")
    ref=d.loc[d[device_col].astype(str)==reference_device,[*ids,metric]].rename(columns={metric:"reference_value"})
    oth=d.loc[d[device_col].astype(str)!=reference_device,[*ids,device_col,metric]].rename(columns={metric:"device_value"})
    paired=oth.merge(ref,on=ids,how="inner"); paired["device_value"]=pd.to_numeric(paired.device_value,errors="coerce"); paired["reference_value"]=pd.to_numeric(paired.reference_value,errors="coerce"); paired=paired.dropna(subset=["device_value","reference_value"])
    paired["mean_value"]=(paired.device_value+paired.reference_value)/2; paired["difference"]=paired.device_value-paired.reference_value
    models={}; rows=[]
    for dev,g in paired.groupby(device_col,sort=False):
        x0=g.device_value.to_numpy(float); y0=g.reference_value.to_numpy(float); diff=g.difference.to_numpy(float); mean=g.mean_value.to_numpy(float)
        if method=="equipercentile":
            probs=np.linspace(0,1,min(101,max(5,len(g)))); models[str(dev)]={"device_quantiles":np.quantile(x0,probs),"reference_quantiles":np.quantile(y0,probs),"probabilities":probs}
        else:
            transfer=_linear_fit(np.c_[np.ones(len(g)),x0],y0); bias_model=_linear_fit(np.c_[np.ones(len(g)),mean],diff); sd=_safe_sd(diff); bias=_safe_mean(diff)
            models[str(dev)]={"transfer":transfer,"bias_model":bias_model,"bias":bias,"limits":np.array([bias-1.96*sd,bias+1.96*sd])}
        rows.append({"device":str(dev),"n_pairs":len(g),"bias":_safe_mean(diff),"sd_difference":_safe_sd(diff),"correlation":float(np.corrcoef(x0,y0)[0,1]) if len(g)>1 and np.std(x0)>0 and np.std(y0)>0 else np.nan})
    return _result("eye_device_linking",data=d,paired=paired,metric=metric,device_col=device_col,id_cols=ids,reference_device=reference_device,method=method,models=models,summary=pd.DataFrame(rows),status="Cross-device linking model fitted.")


def apply_device_linking(x: Any, linking_model: Any, metric: str|None=None, device_col: str|None=None, output_col: str|None=None) -> pd.DataFrame:
    if not isinstance(linking_model,Mapping) or getattr(linking_model,"eyeprocess_class",None)!="eye_device_linking": raise EyeProcessValidationError("`linking_model` must be an `eye_device_linking` object.")
    d=_df(x); metric=metric or linking_model["metric"]; device_col=device_col or linking_model["device_col"]; output_col=output_col or f"{metric}_linked"; _req(d,[metric,device_col]); out=d.copy(); out[output_col]=pd.to_numeric(out[metric],errors="coerce")
    for dev,model in linking_model["models"].items():
        rows=out[device_col].astype(str)==str(dev); values=pd.to_numeric(out.loc[rows,metric],errors="coerce").to_numpy(float)
        if linking_model["method"]=="equipercentile": out.loc[rows,output_col]=np.interp(values,model["device_quantiles"],model["reference_quantiles"])
        else: out.loc[rows,output_col]=model["transfer"]["coef"][0]+model["transfer"]["coef"][1]*values
    return out


def audit_device_equivalence(x: Any, equivalence_margin: float, by: Sequence[str]=("metric","task","aoi")) -> EyeResult:
    margin=float(equivalence_margin)
    if not np.isfinite(margin) or margin<=0: raise EyeProcessValidationError("`equivalence_margin` must be a positive scalar.")
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_device_linking": raise EyeProcessValidationError("The dependency-free equivalence audit currently requires an `eye_device_linking` object.")
    d=x["paired"]; groups=list(dict.fromkeys([x["device_col"],*[c for c in by if c in d.columns]])); rows=[]
    for key,g in d.groupby(groups,sort=False,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        diff=g.difference.to_numpy(float); est=_safe_mean(diff); sd=_safe_sd(diff); se=sd/math.sqrt(len(diff)) if len(diff) else np.nan; q=float(student_t.ppf(.95,df=max(1,len(diff)-1))) if len(diff)>1 else float(student_t.ppf(.95,1)); lo=est-q*se; hi=est+q*se
        row=dict(zip(groups,key)); row.update(n=len(diff),mean_difference=est,lower90=lo,upper90=hi,equivalent=bool(lo>-margin and hi<margin)); rows.append(row)
    return _result("eye_device_equivalence",summary=pd.DataFrame(rows),margin=margin,source=x,status="Device-equivalence intervals calculated.")


def estimate_device_specific_error(x: Any) -> pd.DataFrame:
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_device_linking": raise EyeProcessValidationError("`x` must be an `eye_device_linking` object.")
    rows=[]
    for dev,g in x["paired"].groupby(x["device_col"],sort=False):
        diff=g.difference.to_numpy(float); rows.append({"device":str(dev),"residual_sd":_safe_sd(diff),"residual_variance":float(np.nanvar(diff,ddof=1)) if len(diff)>1 else np.nan,"n":len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 043 item-bank optimization
# ---------------------------------------------------------------------------

def item_objective_spec(information: Any, process_burden: Any, fairness: Any, exposure: Any, content_constraints: Any=None, weights: Mapping[str,float]|None=None, directions: Mapping[str,str]|None=None) -> EyeResult:
    weights=dict(weights or {"information":1,"process_burden":1,"fairness":1,"exposure":1}); directions=dict(directions or {"information":"max","process_burden":"min","fairness":"min","exposure":"min"})
    return _result("eye_item_objective_spec",objectives={"information":information,"process_burden":process_burden,"fairness":fairness,"exposure":exposure},constraints=content_constraints,weights=weights,directions=directions)


def _objective_table(x: Any,spec: Any) -> pd.DataFrame:
    d=_df(x); out=pd.DataFrame({"item_id":d.item_id.astype(str) if "item_id" in d else [f"item_{i+1}" for i in range(len(d))]})
    for name,value in spec["objectives"].items():
        if isinstance(value,str) and value in d.columns: v=pd.to_numeric(d[value],errors="coerce").to_numpy(float)
        else: v=np.asarray(value,dtype=float).reshape(-1)
        if v.size!=len(d): raise EyeProcessValidationError(f"Objective `{name}` must have one value per item.")
        out[name]=v
    return out


def item_pareto_front(x: Any, objectives: Any) -> EyeResult:
    if not isinstance(objectives,Mapping) or getattr(objectives,"eyeprocess_class",None)!="eye_item_objective_spec": raise EyeProcessValidationError("`objectives` must be an `eye_item_objective_spec`.")
    tab=_objective_table(x,objectives); names=list(objectives["objectives"]); dirs=[objectives["directions"].get(n,"max") for n in names]; vals=tab[names].to_numpy(float); signs=np.array([1 if q=="max" else -1 for q in dirs]); adj=vals*signs; nd=np.ones(len(tab),dtype=bool)
    for i in range(len(tab)):
        for j in range(len(tab)):
            if i!=j and np.all(adj[j]>=adj[i]) and np.any(adj[j]>adj[i]): nd[i]=False; break
    score=np.zeros(len(tab))
    for k,n in enumerate(names):
        v=vals[:,k]; sd=np.nanstd(v,ddof=1); z=(v-np.nanmean(v))/sd if np.isfinite(sd) and sd>0 else np.zeros(len(v)); score+=(1 if dirs[k]=="max" else -1)*float(objectives["weights"].get(n,1))*z
    tab["pareto_front"]=nd; tab["weighted_score"]=score; summary=tab.assign(_not=~nd).sort_values(["_not","weighted_score"],ascending=[True,False]).drop(columns="_not")
    return _result("eye_item_pareto",data=_df(x),objective_spec=objectives,table=tab,summary=summary,status="Item Pareto front identified.")


def _constraint_mask(tab: pd.DataFrame,constraints: Any) -> np.ndarray:
    if constraints is None: return np.ones(len(tab),dtype=bool)
    if callable(constraints): return np.asarray(constraints(tab),dtype=bool)
    if isinstance(constraints,Mapping):
        keep=np.ones(len(tab),dtype=bool)
        for name,rule in constraints.items():
            if name in tab and isinstance(rule,(list,tuple,np.ndarray)) and len(rule)==2:
                lo,hi=min(rule),max(rule); keep&=(tab[name]>=lo)&(tab[name]<=hi)
        return keep
    return np.ones(len(tab),dtype=bool)


def optimize_item_bank(x: Any,n_items: int,objectives: Any,constraints: Any=None,method: str="integer",iterations: int=500,seed: int=20260807) -> EyeResult:
    if method not in {"integer","evolutionary"}: raise EyeProcessValidationError("invalid optimization method.")
    pareto=x if isinstance(x,Mapping) and getattr(x,"eyeprocess_class",None)=="eye_item_pareto" else item_pareto_front(x,objectives); tab=pareto["table"].copy(); n_items=int(n_items)
    if n_items<1 or n_items>len(tab): raise EyeProcessValidationError("`n_items` is outside the available item count.")
    allowed=_constraint_mask(tab,constraints if constraints is not None else objectives.get("constraints")); candidates=np.where(allowed)[0]
    if len(candidates)<n_items: raise EyeProcessValidationError("Constraints leave fewer items than requested.")
    if method=="integer": selected=sorted(candidates,key=lambda i:(bool(tab.pareto_front.iloc[i]),float(tab.weighted_score.iloc[i])),reverse=True)[:n_items]
    else:
        rng=np.random.default_rng(int(seed)); best=rng.choice(candidates,n_items,replace=False); best_score=float(tab.weighted_score.iloc[best].sum())
        for _ in range(int(iterations)):
            avail=np.setdiff1d(candidates,best)
            if len(avail)==0: break
            proposal=best.copy(); proposal[int(rng.integers(0,n_items))]=int(rng.choice(avail)); sc=float(tab.weighted_score.iloc[proposal].sum())
            if sc>best_score: best,best_score=proposal,sc
        selected=list(best)
    tab["selected"]=False; tab.loc[selected,"selected"]=True; sel=tab.loc[tab.selected].copy()
    return _result("eye_item_bank_optimization",pareto=pareto,table=tab,selected=sel,n_items=n_items,method=method,objective_total=float(sel.weighted_score.sum()),summary=sel,status="Multi-objective item bank selected under declared constraints.")


def audit_bank_decision_stability(x: Any,draws: int=1000,noise_sd: float=.1,seed: int=20260807) -> EyeResult:
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_item_bank_optimization": raise EyeProcessValidationError("`x` must be an item-bank optimization result.")
    tab=x["table"]; draws=int(draws); rng=np.random.default_rng(int(seed)); count=np.zeros(len(tab),dtype=int)
    for _ in range(draws):
        score=tab.weighted_score.to_numpy(float)+rng.normal(0,float(noise_sd),len(tab)); count[np.argsort(score)[::-1][:int(x["n_items"])]]+=1
    summary=pd.DataFrame({"item_id":tab.item_id.astype(str),"selection_probability":count/draws,"selected_originally":tab.selected.to_numpy(bool)})
    return _result("eye_bank_decision_stability",optimization=x,summary=summary,status="Item-bank selection stability audited under objective perturbation.")


# ---------------------------------------------------------------------------
# 044 process DIF / fairness
# ---------------------------------------------------------------------------

def fit_process_dif(x: Any,response: str,process: str,group: str,item: str,ability: str|None=None) -> EyeResult:
    d=_df(x); cols=[response,process,group,item]+([ability] if ability else []); _req(d,cols); rows=[]; models={}
    for iid,g in d.groupby(item,sort=False):
        y=pd.to_numeric(g[response],errors="coerce").to_numpy(float); proc=_z(g[process]); groups=pd.Categorical(g[group]); levels=list(groups.categories); dummy=pd.get_dummies(groups,drop_first=True,dtype=float); Xparts=[np.ones(len(g)),*([dummy[c].to_numpy(float) for c in dummy.columns]),proc,*([dummy[c].to_numpy(float)*proc for c in dummy.columns])]
        names=["(Intercept)",*([f".group{c}" for c in dummy.columns]),".process",*([f".group{c}:.process" for c in dummy.columns])]
        if ability: Xparts.append(pd.to_numeric(g[ability],errors="coerce").to_numpy(float)); names.append(".ability")
        X=np.column_stack(Xparts); binary=set(pd.Series(y).dropna().unique()).issubset({0,1}); model=_logistic_fit(X,y) if binary else _linear_fit(X,y); models[str(iid)]={**model,"terms":names,"binary":binary}
        pvals=np.asarray(model["p"]); coef=np.asarray(model["coef"]); group_idx=[i for i,n in enumerate(names) if n.startswith(".group") and ":.process" not in n]; int_idx=[i for i,n in enumerate(names) if ":.process" in n]
        rows.append({"item_id":str(iid),"n":len(g),"psychometric_dif":float(np.max(np.abs(coef[group_idx]))) if group_idx else np.nan,"process_dif":float(np.max(np.abs(coef[int_idx]))) if int_idx else np.nan,"psychometric_p":float(np.min(pvals[group_idx])) if group_idx else np.nan,"process_p":float(np.min(pvals[int_idx])) if int_idx else np.nan})
    summary=pd.DataFrame(rows); summary["review_flag"]=(summary.psychometric_p<.05)|(summary.process_p<.05)
    return _result("eye_process_dif",models=models,summary=summary,data=d,response=response,process=process,group=group,item=item,status="Psychometric and process DIF models fitted item by item.")


def monitor_dif_drift(x: Any,time: str,group: str,metrics: Sequence[str],item: str|None=None) -> EyeResult:
    d=_df(x); cols=[time,group,*metrics]+([item] if item else []); _req(d,cols); rows=[]; group_cols=[time,group]+([item] if item else [])
    for metric in metrics:
        means=d.groupby(group_cols,dropna=False,sort=False)[metric].apply(lambda s:_safe_mean(pd.to_numeric(s,errors="coerce"))).reset_index(name="mean_value")
        if item: means=means.rename(columns={item:"item_id"})
        else: means["item_id"]="all_items"
        means["metric"]=metric; rows.append(means)
    traj=pd.concat(rows,ignore_index=True); slopes=[]
    for (metric0,iid,grp),g in traj.groupby(["metric","item_id",group],dropna=False,sort=False):
        tv=pd.to_numeric(g[time],errors="coerce").to_numpy(float); mv=pd.to_numeric(g.mean_value,errors="coerce").to_numpy(float); keep=np.isfinite(tv)&np.isfinite(mv); tv=tv[keep]; mv=mv[keep]; slope=np.nan
        if len(tv)>=2 and len(np.unique(tv))>=2: slope=float(np.polyfit(tv,mv,1)[0])
        slopes.append({"metric":metric0,"item_id":iid,"group":str(grp),"slope":slope,"n_time_points":len(np.unique(tv))})
    slopes=pd.DataFrame(slopes); return _result("eye_dif_drift",trajectories=traj,slopes=slopes,time=time,group=group,summary=slopes,status="Process and fairness drift trajectories summarized.")


def decompose_dif_evidence(psychometric: Any,process: Any=None,design_features: Any=None) -> EyeResult:
    tab=psychometric["summary"].copy() if isinstance(psychometric,Mapping) and getattr(psychometric,"eyeprocess_class",None)=="eye_process_dif" else _df(psychometric); _req(tab,["item_id"])
    if process is not None: tab=tab.merge(_df(process),on="item_id",how="outer",suffixes=("_psychometric","_process"))
    if design_features is not None: tab=tab.merge(_df(design_features),on="item_id",how="outer")
    psych=(pd.to_numeric(tab.psychometric_p,errors="coerce")<.05) if "psychometric_p" in tab else tab.get("review_flag",False); proc=(pd.to_numeric(tab.process_p,errors="coerce")<.05) if "process_p" in tab else np.zeros(len(tab),dtype=bool)
    tab["evidence_pattern"]=np.where(psych&proc,"convergent_psychometric_and_process_difference",np.where(psych,"psychometric_only",np.where(proc,"process_only","no_flag")))
    return _result("eye_dif_decomposition",table=tab,summary=tab,status="DIF evidence decomposed. Interpret associations without causal language unless the design supports it.")


def audit_fairness_transportability(x: Any,context: str="device",effect: str="process_dif",item: str="item_id") -> EyeResult:
    data=x["summary"].copy() if isinstance(x,Mapping) and getattr(x,"eyeprocess_class",None)=="eye_process_dif" else _df(x); _req(data,[item,effect]);
    if context not in data: data[context]="single_context"
    wide=data.pivot_table(index=item,columns=context,values=effect,aggfunc="first")
    corr=wide.corr(min_periods=1).to_numpy() if wide.shape[1]>=2 else np.ones((1,1))
    if corr.size>1:
        upper=corr[np.triu_indices_from(corr,1)]; finite=upper[np.isfinite(upper)]; mean=float(finite.mean()) if finite.size else np.nan
    else:
        mean=1.0
    matrix=wide.reset_index(); return _result("eye_fairness_transportability",data=data,effect_matrix=matrix,correlation=corr,correlation_labels=list(map(str,wide.columns)),summary=pd.DataFrame({"contexts":[wide.shape[1]],"mean_cross_context_correlation":[mean]}),status="Fairness transportability audited across available contexts.")


# ---------------------------------------------------------------------------
# 045 process norms
# ---------------------------------------------------------------------------

def _norm_design(data: pd.DataFrame,covariates: Sequence[str],columns: Sequence[str]|None=None) -> tuple[np.ndarray,list[str]]:
    z=pd.get_dummies(data[list(covariates)],drop_first=False,dtype=float); z=z.apply(pd.to_numeric,errors="coerce"); z.insert(0,"(Intercept)",1.0)
    if columns is not None: z=z.reindex(columns=list(columns),fill_value=0.0)
    return z.to_numpy(float),list(z.columns)


def fit_process_norms(x: Any,metric: str,covariates: Sequence[str]|str,family: str="auto") -> EyeResult:
    if family not in {"auto","gaussian","lognormal"}: raise EyeProcessValidationError("invalid norm family.")
    d=_df(x); cov=[covariates] if isinstance(covariates,str) else list(covariates); _req(d,[metric,*cov]); data=d[[metric,*cov]].copy(); data[metric]=pd.to_numeric(data[metric],errors="coerce"); data=data.dropna()
    if len(data)<2: raise EyeProcessValidationError("complete reference data are required.")
    if family=="auto":
        raw=np.sort(data[metric].to_numpy(float)); probs=(np.arange(1,len(raw)+1)-.375)/(len(raw)+.25); q=norm.ppf(probs); cr=np.corrcoef(raw,q)[0,1]; cl=np.corrcoef(np.sort(np.log(raw)),q)[0,1] if np.all(raw>0) else -np.inf; family="lognormal" if np.all(raw>0) and abs(cr)<abs(cl) else "gaussian"
    outcome=np.log(data[metric].to_numpy(float)) if family=="lognormal" else data[metric].to_numpy(float); X,cols=_norm_design(data,cov); mean_fit=_linear_fit(X,outcome); resid=outcome-X@mean_fit["coef"]; logvar=np.log(np.maximum(resid**2,np.finfo(float).eps)); var_fit=_linear_fit(X,logvar); stored=data.copy(); stored[".outcome"]=outcome
    return _result("eye_process_norms",mean_model={**mean_fit,"columns":cols},variance_model={**var_fit,"columns":cols},data=stored,metric=metric,covariates=cov,family=family,summary=pd.DataFrame({"n":[len(data)],"family":[family],"residual_sd":[_safe_sd(resid)]}),status="Conditional process reference distribution fitted. This is not a clinical norm without representative sampling and external validation.")


def _norm_predict(model: Any,newdata: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    X,_=_norm_design(newdata,model["covariates"],model["mean_model"]["columns"]); mean=X@np.asarray(model["mean_model"]["coef"]); Xv,_=_norm_design(newdata,model["covariates"],model["variance_model"]["columns"]); variance=np.exp(Xv@np.asarray(model["variance_model"]["coef"])); return mean,np.sqrt(np.maximum(variance,1e-12))


def predict_process_centiles(model: Any,newdata: Any,centiles: Sequence[float]=(2.5,10,25,50,75,90,97.5)) -> pd.DataFrame:
    if not isinstance(model,Mapping) or getattr(model,"eyeprocess_class",None)!="eye_process_norms": raise EyeProcessValidationError("`model` must be an `eye_process_norms` object.")
    d=_df(newdata,"newdata"); _req(d,model["covariates"]); mean,sd=_norm_predict(model,d); out=d.copy()
    for c in centiles:
        v=mean+norm.ppf(float(c)/100)*sd; v=np.exp(v) if model["family"]=="lognormal" else v; label=(f"{c:g}").replace(".","_"); out[f"centile_{label}"]=v
    return out


def score_process_deviation(model: Any,newdata: Any,type: str="z") -> pd.DataFrame:
    if not isinstance(model,Mapping) or getattr(model,"eyeprocess_class",None)!="eye_process_norms": raise EyeProcessValidationError("`model` must be an `eye_process_norms` object.")
    if type not in {"z","centile","tail_probability"}: raise EyeProcessValidationError("invalid score type.")
    d=_df(newdata,"newdata"); _req(d,[model["metric"],*model["covariates"]]); outcome=pd.to_numeric(d[model["metric"]],errors="coerce").to_numpy(float); outcome=np.log(np.maximum(outcome,np.finfo(float).eps)) if model["family"]=="lognormal" else outcome; mean,sd=_norm_predict(model,d); z=(outcome-mean)/np.maximum(sd,1e-12); score=z if type=="z" else 100*norm.cdf(z) if type=="centile" else 2*norm.cdf(-np.abs(z)); out=d.copy(); out["deviation_score"]=score; out["score_type"]=type; return out


def audit_norm_transportability(model: Any,new_sample: Any) -> EyeResult:
    scored=score_process_deviation(model,new_sample,"z"); z=pd.to_numeric(scored.deviation_score,errors="coerce").to_numpy(float); m=_safe_mean(z); sd=_safe_sd(z); summary=pd.DataFrame({"n":[int(np.isfinite(z).sum())],"mean_z":[m],"sd_z":[sd],"within_95_reference":[float(np.nanmean(np.abs(z)<=1.96))],"calibration_flag":[bool(abs(m)>.25 or abs(sd-1)>.25)]}); return _result("eye_norm_transportability",model=model,scored=scored,summary=summary,status="Norm transportability audited in an external sample.")


# ---------------------------------------------------------------------------
# Frozen public plot aliases from the four source families
# ---------------------------------------------------------------------------

def _ax(ax=None):
    if ax is not None: return ax
    try: import matplotlib.pyplot as plt
    except ImportError as exc: raise ImportError("Plotting requires eyeprocesspy[plots].") from exc
    return plt.subplots()[1]


def _device_data(x,device=None):
    d=x["paired"]; device=device or str(d[x["device_col"]].astype(str).unique()[0]); return device,d.loc[d[x["device_col"]].astype(str)==device]

def plot_device_agreement(x: Any,device: str|None=None,ax=None):
    ax=_ax(ax); dev,d=_device_data(x,device); ax.scatter(d.mean_value,d.difference); ax.axhline(d.difference.mean(),linestyle="--"); ax.set_xlabel("Pair mean"); ax.set_ylabel("Device - reference"); ax.set_title(f"Device agreement: {dev}"); return ax

def plot_device_bias_by_magnitude(x: Any,device: str|None=None,ax=None):
    ax=_ax(ax); dev,d=_device_data(x,device); ax.scatter(d.mean_value,d.difference); coef=np.polyfit(d.mean_value,d.difference,1) if len(d)>=2 else [0,float(d.difference.mean())]; xx=np.array([d.mean_value.min(),d.mean_value.max()]); ax.plot(xx,coef[0]*xx+coef[1]); ax.set_title(f"Magnitude-dependent bias: {dev}"); return ax

def plot_device_transfer_curve(x: Any,device: str|None=None,ax=None):
    ax=_ax(ax); dev,d=_device_data(x,device); ax.scatter(d.device_value,d.reference_value); lo=min(d.device_value.min(),d.reference_value.min()); hi=max(d.device_value.max(),d.reference_value.max()); ax.plot([lo,hi],[lo,hi],linestyle="--");
    if x["method"]!="equipercentile": c=x["models"][dev]["transfer"]["coef"]; xx=np.array([d.device_value.min(),d.device_value.max()]); ax.plot(xx,c[0]+c[1]*xx)
    ax.set_xlabel(dev); ax.set_ylabel(x["reference_device"]); return ax

def plot_device_equivalence_intervals(x: Any,ax=None):
    ax=_ax(ax); s=x["summary"]; y=np.arange(len(s)); ax.scatter(s.mean_difference,y); ax.hlines(y,s.lower90,s.upper90); ax.axvline(-x["margin"],linestyle="--"); ax.axvline(x["margin"],linestyle="--"); ax.set_xlabel("Mean difference"); return ax

def plot_cross_vendor_metric_matrix(x: Any,ax=None):
    ax=_ax(ax); groups=[g.difference.to_numpy(float) for _,g in x["paired"].groupby(x["device_col"],sort=False)]; labels=[str(k) for k,_ in x["paired"].groupby(x["device_col"],sort=False)]; ax.boxplot(groups,tick_labels=labels); ax.set_ylabel("Device - reference"); return ax

def plot_item_pareto(x: Any,x_objective: str="information",y_objective: str="process_burden",ax=None):
    ax=_ax(ax); t=x["table"]; names=list(x["objective_spec"]["objectives"]); xo=x_objective if x_objective in t else names[0]; yo=y_objective if y_objective in t else names[1]; ax.scatter(t[xo],t[yo]);
    for _,r in t.iterrows(): ax.annotate(str(r.item_id),(r[xo],r[yo]))
    ax.set_xlabel(xo); ax.set_ylabel(yo); ax.set_title("Item Pareto front"); return ax

def plot_objective_tradeoffs(x: Any,**kwargs): return plot_item_pareto(x,**kwargs)
def plot_bank_information_coverage(x: Any,ax=None):
    ax=_ax(ax); t=x["table"]; objective=list(x["pareto"]["objective_spec"]["objectives"])[0]; ax.scatter(np.arange(1,len(t)+1),t[objective]); ax.set_xlabel("Item"); ax.set_ylabel(objective); return ax

def plot_decision_stability(x: Any,ax=None):
    ax=_ax(ax); s=x["summary"].sort_values("selection_probability",ascending=False); ax.bar(s.item_id.astype(str),s.selection_probability); ax.tick_params(axis="x",rotation=90); ax.set_ylabel("Selection probability"); return ax

def plot_selected_bank_profile(x: Any,ax=None):
    ax=_ax(ax); names=list(x["pareto"]["objective_spec"]["objectives"]); allm=x["table"][names].mean(); selm=x["selected"][names].mean(); pos=np.arange(len(names)); w=.38; ax.bar(pos-w/2,allm,w,label="all_items"); ax.bar(pos+w/2,selm,w,label="selected"); ax.set_xticks(pos,names,rotation=45); ax.set_ylabel("Objective mean"); ax.legend(); return ax

def plot_group_icc_process_overlay(x: Any,ax=None):
    ax=_ax(ax); s=x["summary"]; y=np.arange(len(s)); ax.scatter(s.psychometric_dif,y); ax.axvline(0,linestyle="--"); ax.set_yticks(y,s.item_id.astype(str)); ax.set_xlabel("Absolute DIF effect"); return ax

def plot_process_dif_forest(x: Any,ax=None):
    ax=_ax(ax); s=x["summary"]; y=np.arange(len(s)); ax.scatter(s.process_dif,y); ax.axvline(0,linestyle="--"); ax.set_yticks(y,s.item_id.astype(str)); ax.set_xlabel("Absolute DIF effect"); return ax

def plot_dif_drift_heatmap(x: Any,metric: str|None=None,ax=None):
    ax=_ax(ax); d=x["trajectories"]; metric=metric or str(d.metric.iloc[0]); z=d.loc[d.metric==metric]; tab=z.pivot_table(index="item_id",columns=x["time"],values="mean_value",aggfunc="mean"); ax.imshow(tab.to_numpy(float),aspect="auto"); ax.set_xlabel("Time"); ax.set_ylabel("Item"); return ax

def plot_fairness_transport_matrix(x: Any,ax=None):
    ax=_ax(ax); ax.imshow(np.asarray(x["correlation"],dtype=float),aspect="auto"); return ax

def plot_item_group_process_curves(x: Any,ax=None): return plot_group_icc_process_overlay(x,ax=ax)
def plot_process_centiles(x: Any,ax=None):
    ax=_ax(ax); cov=x["covariates"];
    if len(cov)==1 and pd.api.types.is_numeric_dtype(x["data"][cov[0]]):
        c=cov[0]; grid=pd.DataFrame({c:np.linspace(x["data"][c].min(),x["data"][c].max(),100)}); cent=predict_process_centiles(x,grid,[2.5,10,50,90,97.5]); ax.scatter(x["data"][c],np.exp(x["data"][".outcome"]) if x["family"]=="lognormal" else x["data"][".outcome"])
        for col in [q for q in cent if q.startswith("centile_")]: ax.plot(grid[c],cent[col])
        ax.set_xlabel(c); ax.set_ylabel(x["metric"])
    return ax

def plot_normative_fan(x: Any,ax=None): return plot_process_centiles(x,ax=ax)
def plot_person_normative_profile(x: Any,newdata: Any=None,ax=None):
    ax=_ax(ax); base=x["data"].copy(); base[x["metric"]]=np.exp(base[".outcome"]) if x["family"]=="lognormal" else base[".outcome"]; scored=score_process_deviation(x,newdata if newdata is not None else base,"z"); ax.bar(np.arange(len(scored)),scored.deviation_score); ax.set_ylabel("Normative z score"); return ax

def plot_item_normative_deviation(x: Any,newdata: Any=None,ax=None): return plot_person_normative_profile(x,newdata=newdata,ax=ax)


__all__=[
"fit_device_linking","apply_device_linking","audit_device_equivalence","estimate_device_specific_error","plot_device_agreement","plot_device_bias_by_magnitude","plot_device_transfer_curve","plot_device_equivalence_intervals","plot_cross_vendor_metric_matrix",
"item_objective_spec","item_pareto_front","optimize_item_bank","audit_bank_decision_stability","plot_item_pareto","plot_objective_tradeoffs","plot_bank_information_coverage","plot_decision_stability","plot_selected_bank_profile",
"fit_process_dif","monitor_dif_drift","decompose_dif_evidence","audit_fairness_transportability","plot_group_icc_process_overlay","plot_process_dif_forest","plot_dif_drift_heatmap","plot_fairness_transport_matrix","plot_item_group_process_curves",
"fit_process_norms","predict_process_centiles","score_process_deviation","audit_norm_transportability","plot_process_centiles","plot_normative_fan","plot_person_normative_profile","plot_item_normative_deviation"]
