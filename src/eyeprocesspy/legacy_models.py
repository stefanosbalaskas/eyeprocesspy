"""Legacy/core modelling and simulation parity for eyeprocess 0.11.1.

Ports the exported API in R/013-models.R, R/014-simulation.R and
R/016-advanced-experimental.R. R-only specialist engines remain explicit gates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence
import math
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2

from .dataset import EyeDataset, is_eye_dataset, new_eye_dataset, add_provenance
from .exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError
from .irt import EyeResult
from .schema import empty_eye_table, new_coordinate_space, standardize_eye_table


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _require_dataset(x: Any) -> EyeDataset:
    if not is_eye_dataset(x):
        raise EyeProcessValidationError("`x` must be an EyeDataset.")
    return x


def _backend_error(engine: str, extra: str | None = None) -> None:
    detail = f" Install/use the corresponding R engine for exact parity{': ' + extra if extra else ''}."
    raise EyeProcessBackendError(f"The frozen `{engine}` backend is not available as a native eyeprocesspy engine.{detail}")


def _statsmodels():
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as exc:
        raise EyeProcessBackendError("This operation requires statsmodels.") from exc
    return sm, smf


def _formula_text(formula: Any) -> str:
    if isinstance(formula, str) and "~" in formula:
        return formula
    raise EyeProcessValidationError("Python formula arguments must be supplied as an R-style formula string such as `score ~ gaze_process`.")


def _features_wide(x: EyeDataset, aggregate: Callable = np.mean) -> pd.DataFrame:
    f = x["features"].copy()
    ids = [c for c in ["recording_id", "participant_id", "trial_id", "item_id"] if c in f.columns]
    if f.empty or "feature_name" not in f or "value" not in f:
        return pd.DataFrame(columns=ids)
    z = f.loc[f["feature_name"].notna()].copy()
    z["value"] = pd.to_numeric(z["value"], errors="coerce")
    if z.empty:
        return pd.DataFrame(columns=ids)
    aggfunc = "mean" if aggregate in {np.mean, np.nanmean} else aggregate
    if not callable(aggfunc) and not isinstance(aggfunc, str):
        aggfunc = "mean"
    try:
        wide = z.pivot_table(index=ids, columns="feature_name", values="value", aggfunc=aggfunc, dropna=False, observed=True).reset_index()
    except Exception as exc:
        raise EyeProcessValidationError("Unable to aggregate feature table into model-data form.") from exc
    wide.columns.name = None
    return wide


def response_matrix(x: Any, value: str = "score", duplicate: str = "error") -> pd.DataFrame:
    x = _require_dataset(x)
    if value not in {"score", "response"}:
        raise EyeProcessValidationError("`value` must be 'score' or 'response'.")
    if duplicate not in {"error", "last", "mean"}:
        raise EyeProcessValidationError("`duplicate` must be 'error', 'last', or 'mean'.")
    d = x["responses"].copy()
    if d.empty:
        raise EyeProcessValidationError("No responses are available.")
    d = d.loc[d["participant_id"].notna() & d["item_id"].notna()].copy()
    dup = d.duplicated(["participant_id", "item_id"], keep=False)
    if dup.any():
        if duplicate == "error":
            raise EyeProcessValidationError("Duplicate participant-item responses detected.")
        if duplicate == "last":
            d = d.drop_duplicates(["participant_id", "item_id"], keep="last")
        else:
            vals = pd.to_numeric(d[value], errors="coerce")
            if vals.notna().sum() != d[value].notna().sum():
                raise EyeProcessValidationError('`duplicate = "mean"` requires a numeric response value.')
            d[value] = vals
            d = d.groupby(["participant_id", "item_id"], as_index=False, sort=False)[value].mean()
    persons = sorted(d["participant_id"].astype(str).unique())
    items = sorted(d["item_id"].astype(str).unique())
    if value == "score":
        vals = pd.to_numeric(d[value], errors="coerce")
    else:
        levels = sorted(d[value].dropna().astype(str).unique())
        mapping = {v: i for i, v in enumerate(levels)}
        vals = d[value].map(lambda v: mapping.get(str(v), np.nan) if pd.notna(v) else np.nan)
    out = pd.DataFrame(np.nan, index=persons, columns=items, dtype=float)
    for p, i, v in zip(d["participant_id"].astype(str), d["item_id"].astype(str), vals):
        out.loc[p, i] = float(v) if pd.notna(v) else np.nan
    out.index.name = "participant_id"; out.columns.name = "item_id"
    return out


def response_time_matrix(x: Any, log_transform: bool = False, duplicate: str = "error") -> pd.DataFrame:
    x = _require_dataset(x)
    if duplicate not in {"error", "last", "mean"}:
        raise EyeProcessValidationError("`duplicate` must be 'error', 'last', or 'mean'.")
    d = x["responses"].copy()
    rt = pd.to_numeric(d.get("response_time"), errors="coerce")
    keep = np.isfinite(rt) & (rt > 0) & d["participant_id"].notna() & d["item_id"].notna()
    d = d.loc[keep].copy(); d["response_time"] = rt.loc[keep]
    if d.empty:
        raise EyeProcessValidationError("No positive response times are available.")
    dup = d.duplicated(["participant_id", "item_id"], keep=False)
    if dup.any():
        if duplicate == "error": raise EyeProcessValidationError("Duplicate participant-item response times detected.")
        if duplicate == "last": d = d.drop_duplicates(["participant_id", "item_id"], keep="last")
        else: d = d.groupby(["participant_id", "item_id"], as_index=False, sort=False)["response_time"].mean()
    persons = sorted(d["participant_id"].astype(str).unique()); items = sorted(d["item_id"].astype(str).unique())
    out = pd.DataFrame(np.nan, index=persons, columns=items, dtype=float)
    for p, i, v in zip(d["participant_id"].astype(str), d["item_id"].astype(str), d["response_time"]): out.loc[p, i] = float(v)
    if log_transform: out = np.log(out)
    out.index.name="participant_id"; out.columns.name="item_id"
    return out


def align_response_matrices(Y: Any, RT: Any) -> EyeResult:
    Y = pd.DataFrame(Y).copy(); RT = pd.DataFrame(RT).copy()
    persons = [p for p in Y.index if p in RT.index]; items = [i for i in Y.columns if i in RT.columns]
    if not persons or not items:
        raise EyeProcessValidationError("Response and response-time matrices have no common persons/items.")
    return _result("eye_aligned_response_matrices", Y=Y.loc[persons, items].copy(), RT=RT.loc[persons, items].copy())


def model_data(x: Any, include_features: bool = True, aggregate_features: Callable = np.mean) -> pd.DataFrame:
    x = _require_dataset(x); d = x["responses"].copy()
    if d.empty: raise EyeProcessValidationError("No responses available.")
    if include_features and not x["features"].empty:
        fw = _features_wide(x, aggregate=aggregate_features)
        keys = [c for c in ["recording_id", "participant_id", "trial_id", "item_id"] if c in fw and c in d]
        if keys: d = d.merge(fw, how="left", on=keys, sort=False)
    d["participant_id"] = pd.Categorical(d["participant_id"])
    d["item_id"] = pd.Categorical(d["item_id"])
    return d


def _new_model(fit: Any, engine: str, model_type: str, data: Any, metadata: Mapping[str, Any] | None = None, experimental: bool = False) -> EyeResult:
    return _result("eyeprocess_model", fit=fit, engine=engine, model_type=model_type, data=data,
                   call=None, metadata=dict(metadata or {}), experimental=bool(experimental),
                   fitted_at=datetime.now(timezone.utc).isoformat())


def _fit_binomial(formula: str, data: pd.DataFrame, **kwargs: Any):
    sm, smf = _statsmodels()
    return smf.glm(formula=formula, data=data, family=sm.families.Binomial(), **kwargs).fit()


def fit_irt(x: Any, engine: str = "mirt", model: Any = 1, itemtype: str = "2PL", value: str = "score", **kwargs: Any) -> EyeResult:
    x = _require_dataset(x); Y = response_matrix(x, value=value)
    if engine in {"mirt", "TAM"}: _backend_error(engine)
    if engine != "rasch_glm": raise EyeProcessValidationError("`engine` must be 'mirt', 'TAM', or 'rasch_glm'.")
    d = x["responses"].copy(); d["score"] = pd.to_numeric(d["score"], errors="coerce"); d = d.loc[np.isfinite(d["score"])].copy()
    if d.empty: raise EyeProcessValidationError("No finite scores are available for Rasch GLM fitting.")
    fit = _fit_binomial("score ~ 0 + C(participant_id) + C(item_id)", d, **kwargs)
    return _new_model(fit, "rasch_glm", "IRT", Y, {"model": model, "itemtype": itemtype, "value": value, "approximation": True})


def fit_explanatory_irt(x: Any, formula: Any, engine: str = "lme4", participant_random: bool = True, item_random: bool = True, family: Any = "binomial", **kwargs: Any) -> EyeResult:
    x = _require_dataset(x); f = _formula_text(formula); d = model_data(x, include_features=True)
    response = f.split("~", 1)[0].strip()
    if response not in d.columns: raise EyeProcessValidationError(f"Formula response `{response}` is not available in model data.")
    if engine in {"lme4", "brms"}: _backend_error(engine)
    if engine != "glm": raise EyeProcessValidationError("`engine` must be 'lme4', 'glm', or 'brms'.")
    rhs = f.split("~",1)[1].strip()
    if participant_random: rhs += " + C(participant_id)"
    if item_random: rhs += " + C(item_id)"
    used = f"{response} ~ {rhs}"
    if isinstance(family, str) and family != "binomial": raise EyeProcessValidationError("The Python GLM parity path currently supports the frozen binomial family contract.")
    fit = _fit_binomial(used, d, **kwargs)
    return _new_model(fit, "glm", "explanatory_irt", d, {"formula": used, "participant_random": bool(participant_random), "item_random": bool(item_random), "warning": "Participant and item effects are fixed, not random."})


def fit_accuracy_rt(x: Any, engine: str = "LNIRT", iterations: int = 1000, burnin: int = 10, residual: bool = False, **kwargs: Any) -> EyeResult:
    x = _require_dataset(x); aligned = align_response_matrices(response_matrix(x), response_time_matrix(x, log_transform=True)); Y, RT = aligned.Y, aligned.RT
    if engine == "LNIRT": _backend_error("LNIRT")
    if engine != "two_stage": raise EyeProcessValidationError("`engine` must be 'LNIRT' or 'two_stage'.")
    irt = fit_irt(x, engine="rasch_glm")
    d = model_data(x, include_features=False).copy(); d["response_time"] = pd.to_numeric(d["response_time"], errors="coerce"); d = d.loc[np.isfinite(d.response_time)&(d.response_time>0)].copy(); d["log_rt"] = np.log(d.response_time)
    _, smf = _statsmodels(); rt_fit = smf.ols("log_rt ~ C(participant_id) + C(item_id)", data=d).fit()
    ability = d.groupby("participant_id", observed=True)["score"].mean(); speed = -d.groupby("participant_id", observed=True)["log_rt"].mean(); common = ability.index.intersection(speed.index)
    corr = float(np.corrcoef(ability.loc[common].astype(float), speed.loc[common].astype(float))[0,1]) if len(common)>1 else np.nan
    fit = _result("eye_two_stage_rt", irt=irt, response_time=rt_fit, ability_speed_correlation=corr)
    return _new_model(fit, "two_stage", "accuracy_response_time", {"Y":Y,"RT":RT}, {"iterations":int(iterations),"RT_is_log":True,"joint":False})


def fit_dif(x: Any, group: Any, engine: str = "logistic", items: Any = None, **kwargs: Any) -> EyeResult:
    x = _require_dataset(x); d = model_data(x, include_features=False).copy()
    if isinstance(group, str) and group in d.columns: d["_group"] = pd.Categorical(d[group])
    else:
        vals = list(group) if not isinstance(group, str) else []
        if len(vals) != len(d): raise EyeProcessValidationError("`group` must be a model-data column name or a vector with one value per response row.")
        d["_group"] = pd.Categorical(vals)
    if engine == "mirt": _backend_error("mirt", "multiple-group DIF")
    if engine != "logistic": raise EyeProcessValidationError("`engine` must be 'logistic' or 'mirt'.")
    d["score"] = pd.to_numeric(d.score, errors="coerce")
    d["total_score"] = d.groupby("participant_id", observed=True)["score"].transform("sum") - d["score"]
    levels = list(d["item_id"].cat.categories if hasattr(d["item_id"].dtype, "categories") else pd.unique(d.item_id)) if items is None else list(items)
    rows=[]
    for item in levels:
        z=d.loc[(d.item_id.astype(str)==str(item)) & np.isfinite(d.score)].copy()
        if len(z)<10 or z["_group"].nunique()<2:
            rows.append({"item_id":item,"status":"insufficient_data"}); continue
        try:
            f0=_fit_binomial("score ~ total_score",z); f1=_fit_binomial("score ~ total_score + C(_group) + total_score:C(_group)",z)
            dev=max(0.0,2*(f1.llf-f0.llf)); df=max(1,int(f1.df_model-f0.df_model)); p=float(chi2.sf(dev,df))
            rows.append({"item_id":item,"chisq":dev,"df":df,"p_value":p,"status":"estimated"})
        except Exception:
            rows.append({"item_id":item,"status":"fit_failed"})
    return _new_model(pd.DataFrame(rows), "logistic", "DIF", d, {"group":group})


def fit_shared_process_factor(x: Any, features: Sequence[str], n_factors: int = 1, center: bool = True, scale_: bool = True, append: bool = True, prefix: str = "process_factor") -> EyeResult:
    x=_require_dataset(x); wide=_features_wide(x); missing=[f for f in features if f not in wide]
    if missing: raise EyeProcessValidationError(f"Requested process feature(s) absent: {', '.join(missing)}.")
    n_factors=int(n_factors); ok=wide[list(features)].apply(pd.to_numeric,errors="coerce").notna().all(axis=1)
    if int(ok.sum())<=n_factors: raise EyeProcessValidationError("Insufficient complete rows for factor extraction.")
    X=wide.loc[ok,list(features)].astype(float).to_numpy(); mean=X.mean(axis=0) if center else np.zeros(X.shape[1]); Xc=X-mean; sd=Xc.std(axis=0,ddof=1) if scale_ else np.ones(X.shape[1]); sd=np.where(sd>0,sd,1); Xs=Xc/sd
    u,s,vt=np.linalg.svd(Xs,full_matrices=False); scores=np.full((len(wide),n_factors),np.nan); scores[ok.to_numpy(),:]=u[:,:n_factors]*s[:n_factors]
    feature_rows=[]; ids=[c for c in ["recording_id","participant_id","trial_id","item_id"] if c in wide]
    for j in range(n_factors):
        for i,row in wide.iterrows():
            feature_rows.append({**{c:row[c] for c in ids},"feature_name":f"{prefix}_{j+1}","value":scores[i,j],"unit":"standardized_score","level":"trial","method":"principal_components","parameters":f"inputs={','.join(features)}"})
    out=x.copy()
    if append and feature_rows: out["features"]=standardize_eye_table(pd.concat([out["features"],pd.DataFrame(feature_rows)],ignore_index=True,sort=False),"features")
    fit=_result("eye_pca_fit",center=mean,scale=sd,rotation=vt[:n_factors].T,singular_values=s[:n_factors],features=list(features))
    model=_new_model(fit,"stats::prcomp","shared_process_factor",wide,{"features":list(features),"n_factors":n_factors,"neutral_label":True})
    return _result("eye_shared_process_factor_result",data=out,model=model)


def item_parameters(model: Any, **kwargs: Any) -> pd.DataFrame:
    del kwargs
    if getattr(model,"eyeprocess_class",None)!="eyeprocess_model": raise EyeProcessValidationError("Expected an `eyeprocess_model`.")
    if model.engine in {"mirt","TAM"}: _backend_error(model.engine)
    if model.engine=="rasch_glm":
        params=model.fit.params
        rows=[]
        for n,v in params.items():
            if n.startswith("C(item_id)"):
                label=n.split("[T.",1)[-1].rstrip("]") if "[T." in n else n.split("[",1)[-1].rstrip("]")
                rows.append({"item_id":label,"difficulty":-float(v)})
        return pd.DataFrame(rows,columns=["item_id","difficulty"])
    return pd.DataFrame()


def person_scores(model: Any, **kwargs: Any) -> pd.DataFrame:
    del kwargs
    if getattr(model,"eyeprocess_class",None)!="eyeprocess_model": raise EyeProcessValidationError("Expected an `eyeprocess_model`.")
    if model.engine in {"mirt","TAM"}: _backend_error(model.engine)
    if model.engine=="rasch_glm":
        rows=[]
        for n,v in model.fit.params.items():
            if n.startswith("C(participant_id)"):
                label=n.split("[T.",1)[-1].rstrip("]") if "[T." in n else n.split("[",1)[-1].rstrip("]")
                rows.append({"participant_id":label,"theta":float(v)})
        return pd.DataFrame(rows,columns=["participant_id","theta"])
    return pd.DataFrame()


def model_fit_statistics(model: Any) -> pd.DataFrame:
    if getattr(model,"eyeprocess_class",None)!="eyeprocess_model": raise EyeProcessValidationError("Expected an `eyeprocess_model`.")
    fit=model.fit
    def val(name):
        try:
            if name == "bic" and hasattr(fit, "bic_llf"):
                return float(fit.bic_llf)
            return float(getattr(fit,name))
        except Exception:
            return np.nan
    try: nobs=float(fit.nobs)
    except Exception:
        try: nobs=float(len(model.data))
        except Exception: nobs=np.nan
    return pd.DataFrame([{"engine":model.engine,"model_type":model.model_type,"logLik":val("llf"),"AIC":val("aic"),"BIC":val("bic"),"nobs":nobs}])


def check_local_dependence(model: Any, **kwargs: Any) -> Any:
    del kwargs
    if getattr(model,"eyeprocess_class",None)!="eyeprocess_model": raise EyeProcessValidationError("Expected an `eyeprocess_model`.")
    if model.engine=="mirt": _backend_error("mirt", "Q3 local-dependence diagnostics")
    warnings.warn("Local-dependence diagnostics currently require the `mirt` engine.", RuntimeWarning, stacklevel=2)
    return None


def fit_joint_process_model(x: Any, accuracy_formula: Any, rt_formula: Any, process_formulas: Any = None, engine: str = "brms", **kwargs: Any) -> EyeResult:
    x=_require_dataset(x); d=model_data(x,include_features=True).copy(); d["log_response_time"]=np.log(pd.to_numeric(d.response_time,errors="coerce"))
    if engine=="brms": _backend_error("brms", "multivariate joint process models")
    if engine!="separate": raise EyeProcessValidationError("`engine` must be 'brms' or 'separate'.")
    acc=_fit_binomial(_formula_text(accuracy_formula),d,**kwargs); _,smf=_statsmodels(); rt=smf.ols(_formula_text(rt_formula),data=d).fit(); proc=[] if process_formulas is None else [smf.ols(_formula_text(f),data=d).fit() for f in process_formulas]
    fit=_result("eye_separate_process_models",accuracy=acc,response_time=rt,process=proc)
    return _new_model(fit,"separate","joint_process_model",d,{"accuracy_formula":_formula_text(accuracy_formula),"rt_formula":_formula_text(rt_formula),"process_formulas":process_formulas,"estimand_warning":"Separate models do not propagate cross-outcome uncertainty."},experimental=True)


def fit_dynamic_aoi_model(x: Any, source: str = "visits", smoothing: float = 0.5) -> EyeResult:
    x=_require_dataset(x)
    if source not in {"visits","fixations","samples"}: raise EyeProcessValidationError("Invalid dynamic AOI source.")
    d=x["gaze_samples"].copy(); label="aoi_id" if "aoi_id" in d.columns else ("true_aoi" if "true_aoi" in d.columns else None)
    if d.empty or label is None: raise EyeProcessValidationError("No AOI transitions are available.")
    order_col="timestamp_seconds" if "timestamp_seconds" in d else None; group_cols=[c for c in ["recording_id","trial_id"] if c in d]
    levels=sorted(d[label].dropna().astype(str).unique()); counts=pd.DataFrame(0.0,index=levels,columns=levels)
    groups=[(_,d)] if not group_cols else d.groupby(group_cols,sort=False,dropna=False)
    seq_rows=[]
    for key,g in groups:
        if order_col: g=g.sort_values(order_col,kind="stable")
        seq=g[label].dropna().astype(str).tolist(); seq=[s for i,s in enumerate(seq) if i==0 or s!=seq[i-1]]
        seq_rows.append({"group":key,"sequence":seq})
        for a,b in zip(seq,seq[1:]): counts.loc[a,b]+=1
    if counts.to_numpy().sum()==0: raise EyeProcessValidationError("No AOI transitions are available.")
    sm=float(smoothing); probs=(counts+sm).div((counts+sm).sum(axis=1),axis=0)
    fit=_result("eye_dynamic_aoi",counts=counts,probabilities=probs,smoothing=sm)
    return _new_model(fit,"empirical_markov","dynamic_aoi",pd.DataFrame(seq_rows),{"source":source,"smoothing":sm,"order":1},experimental=True)


def simulate_eye_dataset(n_person: int = 30, n_item: int = 10, sampling_rate: float = 60, trial_duration: float = 2, samples_per_trial: int | None = None, include_pupil: bool = True, include_biometrics: bool = True, missing_gaze: float = 0.05, missing_pupil: float = 0.08, seed: int | None = None) -> EyeDataset:
    n_person=int(n_person); n_item=int(n_item)
    if n_person<2 or n_item<2: raise EyeProcessValidationError("Simulation requires at least two persons and two items.")
    rng=np.random.default_rng(seed); persons=[f"P{i:03d}" for i in range(1,n_person+1)]; items=[f"I{i:03d}" for i in range(1,n_item+1)]
    ability=rng.normal(size=n_person); speed=-.3*ability+math.sqrt(1-.3**2)*rng.normal(size=n_person); difficulty=rng.normal(size=n_item); discrimination=rng.lognormal(0,.2,n_item); intensity=rng.normal(math.log(trial_duration),.2,n_item)
    recordings=pd.DataFrame({"recording_id":[f"rec_{p}" for p in persons],"participant_id":persons,"session_id":"S001","vendor":"simulated","vendor_family":"eyeprocess","device_model":"synthetic_tracker","firmware_version":"1","software_name":"eyeprocess","software_version":"development","experiment_type":"simulated item assessment","nominal_sampling_rate":sampling_rate,"screen_width_px":1920,"screen_height_px":1080,"recording_start":pd.NA,"source_timezone":"UTC","source_file_set":"<simulated>"})
    streams=[]; gaze=[]; eyes=[]; events=[]; intervals=[]; responses=[]; bio=[]
    for p,pid in enumerate(persons):
        rec=f"rec_{pid}"; stream_types=[("gaze_combined",sampling_rate,pd.NA,"coord_sim_norm")]
        if include_pupil: stream_types += [("pupil_left",sampling_rate,"millimetres",pd.NA),("pupil_right",sampling_rate,"millimetres",pd.NA)]
        if include_biometrics: stream_types += [("eda",10,"microsiemens",pd.NA),("heart_rate",1,"beats_per_minute",pd.NA)]
        for typ,rate,unit,coord in stream_types: streams.append({"stream_id":f"{rec}_{'gaze' if typ=='gaze_combined' else typ}","recording_id":rec,"stream_type":typ,"source_device":"synthetic_tracker","source_clock":"simulation","sampling_type":"sampled","nominal_rate_hz":rate,"observed_rate_hz":rate,"timestamp_unit":"seconds","value_unit":unit,"coordinate_space_id":coord,"processing_level":"simulated_raw"})
        current=0.0
        for j,iid in enumerate(items):
            trial=f"{rec}_trial_{j+1:03d}"; eta=discrimination[j]*(ability[p]-difficulty[j]); score=int(rng.binomial(1,expit(eta))); rt=float(np.exp(intensity[j]-.25*speed[p]+rng.normal(0,.12))); duration=max(float(trial_duration),rt+.2); start=current; end=start+duration; current=end+.5
            intervals.append({"interval_id":f"{trial}_interval","recording_id":rec,"interval_type":"trial","start_time":start,"end_time":end,"trial_id":trial,"participant_id":pid,"item_id":iid,"stimulus_id":f"stim_{iid}","condition_id":"A" if (j+1)%2 else "B","valid_interval":True})
            responses.append({"response_id":f"{trial}_response","recording_id":rec,"participant_id":pid,"trial_id":trial,"item_id":iid,"response":str(score),"score":score,"response_time":rt,"response_timestamp":start+rt,"response_type":"binary","valid_response":True})
            for etype,etime in [("TRIAL_START",start),("TRIAL_END",end),("RESPONSE",start+rt)]: events.append({"event_id":f"{trial}_{etype.lower()}","recording_id":rec,"timestamp_native":etime,"timestamp_seconds":etime,"event_type":etype.lower(),"event_name":etype,"event_value":trial,"source":"simulation","trial_id":trial,"stimulus_id":f"stim_{iid}"})
            n=max(5,int(round(duration*sampling_rate)) if samples_per_trial is None else int(samples_per_trial)); tt=np.linspace(start,end,n); rel=(tt-start)/duration; state=np.where(rel<.35,"prompt",np.where(rel<.75,"options","evidence")); centers={"prompt":(.25,.35),"options":(.70,.55),"evidence":(.50,.82)}; xy=np.array([centers[s] for s in state]); noise=.02+.015*expit(difficulty[j]-ability[p]); gx=xy[:,0]+rng.normal(0,noise,n); gy=xy[:,1]+rng.normal(0,noise,n); valid=rng.random(n)>float(missing_gaze); gx[~valid]=np.nan; gy[~valid]=np.nan
            for k,t in enumerate(tt): gaze.append({"recording_id":rec,"stream_id":f"{rec}_gaze","sample_id":f"{trial}_sample_{k+1:05d}","timestamp_native":t,"timestamp_seconds":t,"gaze_x":gx[k],"gaze_y":gy[k],"valid":bool(valid[k]),"confidence":float(valid[k]),"trial_id":trial,"stimulus_id":f"stim_{iid}","coordinate_space_id":"coord_sim_norm","true_aoi":state[k]})
            if include_pupil:
                load=expit(difficulty[j]-ability[p])
                for eye in ["left","right"]:
                    pupil=3.4+.15*load+.25*(rel**2*np.exp(-4*rel))*10+rng.normal(0,.035,n); miss=rng.random(n)<float(missing_pupil); pupil[miss]=np.nan
                    for k,t in enumerate(tt): eyes.append({"recording_id":rec,"sample_id":f"{trial}_eye_{eye}_{k+1:05d}","timestamp_native":t,"timestamp_seconds":t,"eye":eye,"pupil_diameter":pupil[k],"pupil_unit":"millimetres","pupil_valid":not bool(miss[k]),"eye_openness":0 if miss[k] else 1,"detector_method":"simulation","confidence":0 if miss[k] else 1,"trial_id":trial,"stimulus_id":f"stim_{iid}"})
            if include_biometrics:
                for channel,rate in [("eda",10),("heart_rate",1)]:
                    tb=np.arange(start,end+1e-12,1/rate); values=(2+.25*expit(difficulty[j]-ability[p])+.08*np.sin((tb-start)*2*np.pi)+rng.normal(0,.03,len(tb))) if channel=="eda" else (70+3*expit(difficulty[j]-ability[p])+rng.normal(0,.8,len(tb)))
                    for t,v in zip(tb,values): bio.append({"recording_id":rec,"stream_id":f"{rec}_{channel}","timestamp_native":t,"timestamp_seconds":t,"channel":channel,"value":v,"unit":"microsiemens" if channel=="eda" else "beats_per_minute","valid":True,"processing_level":"simulated_raw","source_device":"synthetic_biometrics","trial_id":trial,"stimulus_id":f"stim_{iid}"})
    coord=new_coordinate_space("coord_sim_norm","display_normalized_top_left",width=1,height=1,reference_object="simulated_display")
    aoi_defs=pd.DataFrame([{"aoi_id":"prompt","aoi_name":"Prompt","shape":"rectangle","coordinate_space_id":"coord_sim_norm"},{"aoi_id":"options","aoi_name":"Options","shape":"rectangle","coordinate_space_id":"coord_sim_norm"},{"aoi_id":"evidence","aoi_name":"Evidence","shape":"rectangle","coordinate_space_id":"coord_sim_norm"}])
    out=new_eye_dataset(recordings=recordings,streams=pd.DataFrame(streams),gaze_samples=pd.DataFrame(gaze),eye_samples=pd.DataFrame(eyes) if eyes else None,events=pd.DataFrame(events),intervals=pd.DataFrame(intervals),responses=pd.DataFrame(responses),coordinate_spaces=coord,aoi_definitions=aoi_defs,biometrics=pd.DataFrame(bio) if bio else None,vendor_metadata={"simulation_truth":{"ability":dict(zip(persons,ability)),"speed":dict(zip(persons,speed)),"difficulty":dict(zip(items,difficulty)),"discrimination":dict(zip(items,discrimination)),"time_intensity":dict(zip(items,intensity))}},validate=False)
    out=add_provenance(out,"simulate_eye_dataset","dataset",f"n_person={n_person};n_item={n_item};sampling_rate={sampling_rate};samples_per_trial={samples_per_trial if samples_per_trial is not None else 'rate_derived'}")
    return out


def simulate_process_irt(n_person: int = 200, n_item: int = 20, gaze_effect: float = 0.4, pupil_effect: float = 0.3, ability_speed_correlation: float = -0.3, missing_process: float = 0.1, seed: int | None = None) -> EyeResult:
    rng=np.random.default_rng(seed); persons=[f"P{i}" for i in range(1,int(n_person)+1)]; items=[f"I{i}" for i in range(1,int(n_item)+1)]; corr=float(ability_speed_correlation); Sigma=np.array([[1,corr],[corr,1]]); z=rng.multivariate_normal([0,0],Sigma,len(persons)); theta=z[:,0]; speed=z[:,1]; b=rng.normal(size=len(items)); a=rng.lognormal(0,.15,len(items)); intensity=rng.normal(.5,.2,len(items)); pi=np.repeat(np.arange(len(persons)),len(items)); ii=np.tile(np.arange(len(items)),len(persons)); shared=rng.normal(size=len(pi)); gaze=float(gaze_effect)*shared+rng.normal(0,math.sqrt(max(.01,1-float(gaze_effect)**2)),len(pi)); pupil=float(pupil_effect)*shared+rng.normal(0,math.sqrt(max(.01,1-float(pupil_effect)**2)),len(pi)); eta=a[ii]*(theta[pi]-b[ii])+.25*shared; score=rng.binomial(1,expit(eta)); rt=np.exp(intensity[ii]-.3*speed[pi]+.15*shared+rng.normal(0,.2,len(pi))); miss=rng.random(len(pi))<float(missing_process); gaze=gaze.astype(float); pupil=pupil.astype(float); gaze[miss]=np.nan; pupil[miss]=np.nan
    d=pd.DataFrame({"participant_id":np.array(persons)[pi],"item_id":np.array(items)[ii],"score":score,"response_time":rt,"gaze_process":gaze,"pupil_process":pupil,"shared_process":shared})
    truth={"theta":dict(zip(persons,theta)),"speed":dict(zip(persons,speed)),"difficulty":dict(zip(items,b)),"discrimination":dict(zip(items,a)),"intensity":dict(zip(items,intensity)),"gaze_effect":float(gaze_effect),"pupil_effect":float(pupil_effect)}
    return _result("eye_process_irt_simulation",data=d,truth=truth)


def parameter_recovery(simulator: Callable, estimator: Callable, extractor: Callable, truth_extractor: Callable, replications: int = 100, seed: int = 1, **kwargs: Any) -> pd.DataFrame:
    if not all(callable(f) for f in [simulator,estimator,extractor,truth_extractor]): raise EyeProcessValidationError("Simulator, estimator, extractor, and truth_extractor must be functions.")
    np.random.seed(int(seed)); rows=[]
    for r in range(1,int(replications)+1):
        try: sim=simulator(**kwargs); fit=estimator(sim); est=dict(extractor(fit)); truth=dict(truth_extractor(sim)); common=[k for k in est if k in truth]; rows.extend({"replication":r,"parameter":k,"estimate":float(est[k]),"truth":float(truth[k]),"error":pd.NA} for k in common)
        except Exception as exc: rows.append({"replication":r,"parameter":pd.NA,"estimate":np.nan,"truth":np.nan,"error":str(exc)})
    out=pd.DataFrame(rows); out["bias"]=pd.to_numeric(out.estimate,errors="coerce")-pd.to_numeric(out.truth,errors="coerce"); out["squared_error"]=out.bias**2; out.attrs["eyeprocess_class"]="eye_parameter_recovery"; return out


def power_process_simulation(n_person: Any, n_item: Any, effect: Any, replications: int = 200, alpha: float = 0.05, seed: int = 1) -> pd.DataFrame:
    rng=np.random.default_rng(int(seed)); nps=np.atleast_1d(n_person); nis=np.atleast_1d(n_item); effs=np.atleast_1d(effect); rows=[]
    for npers in nps:
      for nitems in nis:
       for eff in effs:
        detected=[]; estimates=[]
        for _ in range(int(replications)):
            sim=simulate_process_irt(int(npers),int(nitems),gaze_effect=float(eff),seed=int(rng.integers(0,2**31-1))).data
            try:
                fit=_fit_binomial("score ~ gaze_process + C(participant_id) + C(item_id)",sim); detected.append(float(fit.pvalues.get("gaze_process",1))<float(alpha)); estimates.append(float(fit.params.get("gaze_process",np.nan)))
            except Exception: detected.append(False); estimates.append(np.nan)
        rows.append({"n_person":int(npers),"n_item":int(nitems),"effect":float(eff),"power":float(np.mean(detected)),"mean_estimate":float(np.nanmean(estimates)) if np.isfinite(estimates).any() else np.nan,"replications":int(replications),"alpha":float(alpha)})
    return pd.DataFrame(rows)


def process_irt_spec(response: str = "score", response_time: str = "response_time", gaze_features: Sequence[str] = (), pupil_features: Sequence[str] = (), biometric_features: Sequence[str] = (), participant_effect: bool = True, item_effect: bool = True, estimand: str = "association", confirmatory: bool = False) -> EyeResult:
    return _result("process_irt_spec",response=response,response_time=response_time,gaze_features=list(gaze_features),pupil_features=list(pupil_features),biometric_features=list(biometric_features),participant_effect=bool(participant_effect),item_effect=bool(item_effect),estimand=estimand,confirmatory=bool(confirmatory))


def fit_process_irt(x: Any, spec: Any, engine: str = "lme4", **kwargs: Any) -> EyeResult:
    if getattr(spec,"eyeprocess_class",None)!="process_irt_spec": raise EyeProcessValidationError("`spec` must be created with `process_irt_spec()`.")
    pred=list(spec.gaze_features)+list(spec.pupil_features)+list(spec.biometric_features); rhs=" + ".join(pred) if pred else "1"; return fit_explanatory_irt(x,f"{spec.response} ~ {rhs}",engine=engine,participant_random=spec.participant_effect,item_random=spec.item_effect,**kwargs)


def fit_gaze_informed_irt(x: Any, response: str = "score", gaze_features: Sequence[str] = (), engine: str = "lme4", **kwargs: Any) -> EyeResult:
    return fit_process_irt(x,process_irt_spec(response=response,gaze_features=gaze_features),engine=engine,**kwargs)


def fit_pupil_informed_irt(x: Any, response: str = "score", pupil_features: Sequence[str] = (), engine: str = "lme4", **kwargs: Any) -> EyeResult:
    return fit_process_irt(x,process_irt_spec(response=response,pupil_features=pupil_features),engine=engine,**kwargs)


def fit_multimodal_irt(x: Any, response: str = "score", gaze_features: Sequence[str] = (), pupil_features: Sequence[str] = (), biometric_features: Sequence[str] = (), engine: str = "lme4", **kwargs: Any) -> EyeResult:
    return fit_process_irt(x,process_irt_spec(response=response,gaze_features=gaze_features,pupil_features=pupil_features,biometric_features=biometric_features),engine=engine,**kwargs)


def process_irt_diagnostics(model: Any) -> EyeResult:
    if getattr(model,"eyeprocess_class",None)!="eyeprocess_model": raise EyeProcessValidationError("Expected an `eyeprocess_model`.")
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); ld=check_local_dependence(model)
    warns=[]
    if model.experimental: warns.append("Model is marked experimental.")
    for key in ["warning","estimand_warning"]:
        v=model.metadata.get(key)
        if v: warns.append(v)
    return _result("eye_process_irt_diagnostics",fit=model_fit_statistics(model),local_dependence=ld,warnings=warns)


def functional_pupil_features(x: Any, df: int = 5, grid_points: int = 100, append: bool = True, prefix: str = "pupil_basis") -> Any:
    del grid_points
    x=_require_dataset(x); d=x["eye_samples"].copy(); trials=x["intervals"].copy(); trials=trials.loc[trials.interval_type.astype(str)=="trial"] if "interval_type" in trials else trials
    if d.empty or trials.empty: return x if append else empty_eye_table("features")
    try: from patsy import dmatrix
    except Exception as exc: raise EyeProcessBackendError("functional_pupil_features requires patsy for natural spline basis construction.") from exc
    rows=[]
    for keys,z in d.loc[d.trial_id.notna()].groupby(["recording_id","trial_id","eye"],sort=False,dropna=False):
        rec,trial,eye=keys; tr=trials.loc[(trials.recording_id==rec)&(trials.trial_id==trial)]
        if tr.empty: continue
        t=pd.to_numeric(z.timestamp_seconds,errors="coerce"); y=pd.to_numeric(z.pupil_diameter,errors="coerce"); ok=np.isfinite(t)&np.isfinite(y)
        if ok.sum()<int(df)+2: continue
        start=float(tr.start_time.iloc[0]); end=float(tr.end_time.iloc[0]); rel=(t[ok].to_numpy()-start)/(end-start); B=np.asarray(dmatrix(f"cr(x, df={int(df)}) - 1",{"x":rel},return_type="dataframe")); X=np.column_stack([np.ones(len(B)),B]); coef=np.linalg.lstsq(X,y[ok].to_numpy(float),rcond=None)[0][1:]
        for j,v in enumerate(coef,1): rows.append({"recording_id":rec,"participant_id":tr.participant_id.iloc[0],"trial_id":trial,"item_id":tr.item_id.iloc[0],"stimulus_id":tr.stimulus_id.iloc[0],"aoi_id":pd.NA,"feature_name":f"{prefix}_{j}","value":float(v),"unit":"basis_coefficient","level":"trial_eye","method":"natural_spline_pupil","parameters":f"df={df};eye={eye}","observed_fraction":float(ok.mean())})
    f=standardize_eye_table(pd.DataFrame(rows) if rows else empty_eye_table("features"),"features")
    if not append: return f
    out=x.copy(); out["features"]=standardize_eye_table(pd.concat([out["features"],f],ignore_index=True,sort=False),"features"); return add_provenance(out,"functional_pupil_features","features",f"{len(f)} rows;df={df}")


def fit_strategy_mixture(x: Any, features: Sequence[str], centers: int = 2, response_formula: Any = None, seed: int = 1, append: bool = True) -> EyeResult:
    x=_require_dataset(x); wide=_features_wide(x); missing=[f for f in features if f not in wide]
    if missing: raise EyeProcessValidationError(f"Missing strategy feature(s): {', '.join(missing)}.")
    X=wide[list(features)].apply(pd.to_numeric,errors="coerce"); ok=X.notna().all(axis=1); centers=int(centers)
    if ok.sum()<centers*3: raise EyeProcessValidationError("Insufficient complete observations for the requested number of strategy clusters.")
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as exc: raise EyeProcessBackendError("Strategy-mixture reference fitting requires scikit-learn.") from exc
    scaler=StandardScaler().fit(X.loc[ok]); Z=scaler.transform(X.loc[ok]); km=KMeans(n_clusters=centers,random_state=int(seed),n_init=10).fit(Z); wide["strategy_class"]=np.nan; wide.loc[ok,"strategy_class"]=km.labels_+1
    feature_rows=[]; ids=[c for c in ["recording_id","participant_id","trial_id","item_id"] if c in wide]
    for _,row in wide.iterrows(): feature_rows.append({**{c:row[c] for c in ids},"feature_name":"strategy_class","value":row.strategy_class,"unit":"class","level":"trial","method":"kmeans_strategy","parameters":f"features={','.join(features)};centers={centers}"})
    out=x.copy()
    if append: out["features"]=standardize_eye_table(pd.concat([out["features"],pd.DataFrame(feature_rows)],ignore_index=True,sort=False),"features")
    response_fit=None
    if response_formula is not None:
        d=model_data(out,include_features=False).merge(wide[ids+["strategy_class"]],how="left",on=ids); response_fit=_fit_binomial(_formula_text(response_formula),d)
    fit=_result("eye_strategy_mixture",kmeans=km,response_model=response_fit,features=list(features),scaler=scaler)
    model=_new_model(fit,"kmeans+glm","strategy_mixture",wide,{"centers":centers,"construct_warning":"Classes are descriptive and must not be named as cognitive strategies without external validation."},experimental=True)
    return _result("eye_strategy_mixture_result",data=out,model=model)


def estimate_ez_diffusion(x: Any, accuracy: str = "score", response_time: str = "response_time", by: Sequence[str] = ("item_id",), scale: float = 0.1) -> pd.DataFrame:
    d=x["responses"].copy() if is_eye_dataset(x) else pd.DataFrame(x).copy(); by=[by] if isinstance(by,str) else list(by)
    missing=[c for c in [accuracy,response_time,*by] if c not in d]
    if missing: raise EyeProcessValidationError(f"Missing required columns: {', '.join(missing)}")
    rows=[]
    for key,z in d.groupby(by,sort=False,dropna=False):
        if len(by)==1 and not isinstance(key, tuple): key=(key,)
        acc=pd.to_numeric(z[accuracy],errors="coerce"); rt=pd.to_numeric(z[response_time],errors="coerce"); ok=np.isfinite(acc)&np.isfinite(rt)&(rt>0); a=acc[ok].to_numpy(float); r=rt[ok].to_numpy(float); n=len(r); pc_raw=float(a.mean()) if n else np.nan; vrt=float(r.var(ddof=1)) if n>1 else np.nan; mrt=float(r.mean()) if n else np.nan; drift=boundary=nondecision=np.nan; status="ok"
        if n<3 or not np.isfinite(vrt) or vrt<=0 or not np.isfinite(pc_raw): status="insufficient_data"
        else:
            pc=min(max(pc_raw,1/(2*n)),1-1/(2*n)); lp=float(logit(pc)); xterm=lp*(lp*pc**2-lp*pc+pc-.5)/vrt
            if not np.isfinite(xterm) or xterm<=0 or abs(pc-.5)<math.sqrt(np.finfo(float).eps): status="undefined_at_chance"
            else:
                drift=math.copysign(float(scale)*xterm**.25,pc-.5); boundary=float(scale)**2*lp/drift; yy=-drift*boundary/float(scale)**2; mdt=(boundary/(2*drift))*(1-math.exp(yy))/(1+math.exp(yy)); nondecision=mrt-mdt
        row={c:v for c,v in zip(by,key)}; row.update({"accuracy":pc_raw,"mean_rt":mrt,"variance_rt":vrt,"drift_rate":drift,"boundary_separation":boundary,"nondecision_time":nondecision,"n":n,"status":status}); rows.append(row)
    return pd.DataFrame(rows)


def fit_gaze_weighted_choice(x: Any, response: str = "score", dwell_features: Sequence[str] = (), engine: str = "glm", **kwargs: Any) -> EyeResult:
    return fit_explanatory_irt(x,f"{response} ~ {' + '.join(dwell_features) if dwell_features else '1'}",engine=engine,**kwargs)


def model_missing_process(x: Any, feature_name: str, predictors: Sequence[str] = ("score","response_time"), engine: str = "glm", **kwargs: Any) -> EyeResult:
    x=_require_dataset(x); d=model_data(x,include_features=True).copy()
    if feature_name not in d: raise EyeProcessValidationError(f"Feature `{feature_name}` is absent.")
    d["_missing_process"]=(~np.isfinite(pd.to_numeric(d[feature_name],errors="coerce"))).astype(int); rhs=" + ".join(predictors) if predictors else "1"
    if engine=="lme4": _backend_error("lme4", "multilevel missing-process models")
    if engine!="glm": raise EyeProcessValidationError("`engine` must be 'glm' or 'lme4'.")
    fit=_fit_binomial(f"_missing_process ~ {rhs} + C(participant_id) + C(item_id)",d,**kwargs); return _new_model(fit,"glm","missing_process_model",d,{"feature_name":feature_name})


def sensitivity_missing_process(x: Any, feature_name: str, formula: Any, methods: Sequence[str] = ("complete_case","median_indicator")) -> EyeResult:
    x=_require_dataset(x); d=model_data(x,include_features=True).copy(); f=_formula_text(formula)
    if feature_name not in d: raise EyeProcessValidationError(f"Feature `{feature_name}` is absent.")
    fits={}; vals=pd.to_numeric(d[feature_name],errors="coerce")
    if "complete_case" in methods: fits["complete_case"]=_fit_binomial(f,d.loc[np.isfinite(vals)].copy())
    if "median_indicator" in methods:
        z=d.copy(); miss=~np.isfinite(vals); z[f"{feature_name}_missing"]=miss.astype(int); median=float(np.nanmedian(vals)); z[feature_name]=vals.fillna(median); lhs,rhs=f.split("~",1); fits["median_indicator"]=_fit_binomial(f"{lhs.strip()} ~ {rhs.strip()} + {feature_name}_missing",z)
    return _result("eye_missing_sensitivity",fits=fits,feature_name=feature_name,methods=list(methods))


__all__ = [
    "response_matrix","response_time_matrix","align_response_matrices","model_data","fit_irt","fit_explanatory_irt","fit_accuracy_rt","fit_dif","fit_shared_process_factor","item_parameters","person_scores","model_fit_statistics","check_local_dependence","fit_joint_process_model","fit_dynamic_aoi_model",
    "simulate_eye_dataset","simulate_process_irt","parameter_recovery","power_process_simulation",
    "process_irt_spec","fit_process_irt","fit_gaze_informed_irt","fit_pupil_informed_irt","fit_multimodal_irt","process_irt_diagnostics","functional_pupil_features","fit_strategy_mixture","estimate_ez_diffusion","fit_gaze_weighted_choice","model_missing_process","sensitivity_missing_process",
]
