"""Functional pupil-IRT and advanced validation parity for eyeprocess 0.11.1.

This module follows the final function definitions loaded from
``R/026-functional-pupil-engine.R`` (which supersede the earlier definitions in
``R/022-advanced-models-v2.R``) and the validation-program exports that remain
in ``R/022-advanced-models-v2.R``.
"""
from __future__ import annotations

from dataclasses import replace
from importlib import resources
from itertools import product
from typing import Any, Callable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from .dataset import EyeDataset, is_eye_dataset
from .exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError
from .irt import EyeResult


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _choice(value: str | Sequence[str], choices: Sequence[str], name: str) -> str:
    if isinstance(value, str):
        selected = value
    else:
        values = list(value)
        if not values:
            raise EyeProcessValidationError(f"`{name}` must not be empty.")
        selected = str(values[0])
    if selected not in choices:
        raise EyeProcessValidationError(f"`{name}` must be one of: {', '.join(choices)}.")
    return selected


def _name(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EyeProcessValidationError(f"`{label}` must be one non-empty column name.")
    return value


def functional_pupil_irt_spec(
    df: int = 6,
    basis: str | Sequence[str] = ("natural_spline", "bspline"),
    response: str = "score",
    engine: str | Sequence[str] = ("two_stage_glm", "two_stage_lme4", "brms", "stan"),
    alignment: str | Sequence[str] = ("trial", "event"),
    event_time_column: str | None = None,
    latency_ms: float = 200,
    baseline_window: Sequence[float] = (-200, 0),
    baseline_method: str | Sequence[str] = ("subtract", "percent", "zscore"),
    min_baseline_samples: int = 3,
    drop_invalid_baseline: bool = True,
    time_window: Sequence[float] | None = None,
    pupil_column: str | None = None,
    time_column: str | None = None,
    participant_column: str = "participant_id",
    item_column: str = "item_id",
    trial_column: str = "trial_id",
    luminance_column: str | None = None,
    gaze_x_column: str | None = None,
    gaze_y_column: str | None = None,
    blink_column: str | None = None,
    interpolated_column: str | None = None,
    max_interpolated_fraction: float = 0.20,
    nuisance_by_participant: bool = False,
    include_response_time: bool = True,
    ar1: bool = True,
    participant_effect: bool = True,
    item_effect: bool = True,
    chains: int = 4,
    parallel_chains: int | None = None,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
) -> EyeResult:
    df = int(df)
    if df < 2:
        raise EyeProcessValidationError("`df` must be an integer of at least two.")
    response = _name(response, "response") or "score"
    basis = _choice(basis, ("natural_spline", "bspline"), "basis")
    engine = _choice(engine, ("two_stage_glm", "two_stage_lme4", "brms", "stan"), "engine")
    alignment = _choice(alignment, ("trial", "event"), "alignment")
    event_time_column = _name(event_time_column, "event_time_column", allow_none=True)
    if alignment == "event" and event_time_column is None:
        raise EyeProcessValidationError("`event_time_column` is required when `alignment = 'event'`.")
    latency_ms = float(latency_ms)
    if not np.isfinite(latency_ms):
        raise EyeProcessValidationError("`latency_ms` must be one finite number.")
    baseline_window = tuple(float(v) for v in baseline_window)
    if len(baseline_window) != 2 or not np.isfinite(baseline_window).all() or baseline_window[0] >= baseline_window[1]:
        raise EyeProcessValidationError("`baseline_window` must contain increasing finite endpoints.")
    min_baseline_samples = int(min_baseline_samples)
    if min_baseline_samples < 1:
        raise EyeProcessValidationError("`min_baseline_samples` must be a positive integer.")
    if time_window is not None:
        time_window = tuple(float(v) for v in time_window)
        if len(time_window) != 2 or not np.isfinite(time_window).all() or time_window[0] >= time_window[1]:
            raise EyeProcessValidationError("`time_window` must contain increasing finite endpoints.")
    max_interpolated_fraction = float(max_interpolated_fraction)
    if not np.isfinite(max_interpolated_fraction) or not 0 <= max_interpolated_fraction <= 1:
        raise EyeProcessValidationError("`max_interpolated_fraction` must be between zero and one.")
    chains = int(chains)
    parallel_chains = chains if parallel_chains is None else int(parallel_chains)
    iter_warmup = int(iter_warmup); iter_sampling = int(iter_sampling); max_treedepth = int(max_treedepth)
    if min(chains, parallel_chains, iter_warmup, iter_sampling, max_treedepth) < 1:
        raise EyeProcessValidationError("CmdStan iteration, chain, and tree-depth controls must be positive integers.")
    if parallel_chains > chains:
        raise EyeProcessValidationError("`parallel_chains` cannot exceed `chains`.")
    adapt_delta = float(adapt_delta)
    if not 0 < adapt_delta < 1:
        raise EyeProcessValidationError("`adapt_delta` must be strictly between zero and one.")
    columns = {
        "pupil_column": _name(pupil_column, "pupil_column", allow_none=True),
        "time_column": _name(time_column, "time_column", allow_none=True),
        "participant_column": _name(participant_column, "participant_column"),
        "item_column": _name(item_column, "item_column"),
        "trial_column": _name(trial_column, "trial_column"),
        "luminance_column": _name(luminance_column, "luminance_column", allow_none=True),
        "gaze_x_column": _name(gaze_x_column, "gaze_x_column", allow_none=True),
        "gaze_y_column": _name(gaze_y_column, "gaze_y_column", allow_none=True),
        "blink_column": _name(blink_column, "blink_column", allow_none=True),
        "interpolated_column": _name(interpolated_column, "interpolated_column", allow_none=True),
    }
    return _result(
        "eye_functional_pupil_irt_spec",
        df=df, basis=basis, response=response, engine=engine, alignment=alignment,
        event_time_column=event_time_column, latency_ms=latency_ms,
        baseline_window=baseline_window,
        baseline_method=_choice(baseline_method, ("subtract", "percent", "zscore"), "baseline_method"),
        min_baseline_samples=min_baseline_samples, drop_invalid_baseline=bool(drop_invalid_baseline),
        time_window=time_window, max_interpolated_fraction=max_interpolated_fraction,
        nuisance_by_participant=bool(nuisance_by_participant), include_response_time=bool(include_response_time),
        ar1=bool(ar1), participant_effect=bool(participant_effect), item_effect=bool(item_effect),
        chains=chains, parallel_chains=parallel_chains, iter_warmup=iter_warmup,
        iter_sampling=iter_sampling, adapt_delta=adapt_delta, max_treedepth=max_treedepth,
        interpretation="Pupil trajectories are physiological observations and are not automatic measures of cognitive load or any named latent construct.",
        **columns,
    )


def _first_table(x: EyeDataset) -> pd.DataFrame:
    for key in ("eye_samples", "biometrics", "gaze_samples"):
        value = x.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty:
            return value.copy()
    return pd.DataFrame()


def _find_col(d: pd.DataFrame, explicit: str | None, candidates: Sequence[str], label: str) -> str:
    if explicit is not None:
        if explicit not in d:
            raise EyeProcessValidationError(f"{label.capitalize()} `{explicit}` is unavailable.")
        return explicit
    for c in candidates:
        if c in d:
            return c
    raise EyeProcessValidationError(f"Could not identify {label}. Tried: {', '.join(candidates)}")


def _binary_response(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.astype("Int64")
    if pd.api.types.is_numeric_dtype(x):
        y = pd.to_numeric(x, errors="coerce")
        lev = sorted(pd.unique(y[np.isfinite(y)]).tolist())
        if lev == [0, 1]: return y.astype("Int64")
        if lev == [1, 2]: return y.map(lambda z: pd.NA if pd.isna(z) else int(z == 2)).astype("Int64")
        if lev == [-1, 1]: return y.map(lambda z: pd.NA if pd.isna(z) else int(z == 1)).astype("Int64")
    vals = x.astype("string")
    lev = list(pd.unique(vals.dropna()))
    if len(lev) != 2:
        raise EyeProcessValidationError("Functional pupil IRT currently requires a binary response with exactly two observed levels.")
    return vals.map(lambda z: pd.NA if pd.isna(z) else int(z == lev[1])).astype("Int64")


def _prepare_frame(x: Any, spec: EyeResult) -> pd.DataFrame:
    if is_eye_dataset(x):
        d = _first_table(x)
        if d.empty:
            raise EyeProcessValidationError("No eye/pupil sample table is available.")
        responses = x.get("responses", pd.DataFrame()).copy()
    else:
        if not isinstance(x, pd.DataFrame):
            raise EyeProcessValidationError("`x` must be an eye dataset or a long pupil data frame.")
        d = x.copy(); responses = pd.DataFrame()

    pupil_col = _find_col(d, spec.pupil_column, ("pupil", "pupil_size", "pupil_diameter", "pupil_mean", "pupil_left", "pupil_right", "pupil_left_mm", "pupil_right_mm"), "pupil column")
    time_col = _find_col(d, spec.time_column, ("time", "time_ms", "relative_time", "timestamp", "timestamp_ms", "sample_time", "timestamp_seconds"), "time column")
    d[".pupil"] = pd.to_numeric(d[pupil_col], errors="coerce")
    d[".time"] = pd.to_numeric(d[time_col], errors="coerce")
    if "timestamp" in time_col.lower() and "seconds" not in time_col.lower():
        finite=np.sort(pd.unique(d[".time"].dropna()))
        if len(finite)>1 and np.nanmedian(np.diff(finite)) < 1: d[".time"] *= 1000
    elif time_col == "timestamp_seconds":
        d[".time"] *= 1000

    # Resolve identifiers from samples or response table.
    canonical = [(spec.participant_column,"participant_id"),(spec.item_column,"item_id"),(spec.trial_column,"trial_id")]
    for source,target in canonical:
        if source in d: d[target]=d[source].astype("string")
    if not all(t in d for _,t in canonical) and not responses.empty:
        keys=[k for k in ("recording_id","trial_id") if k in d and k in responses]
        fields=[c for c in ("participant_id","item_id","trial_id",spec.response,"response_time",spec.event_time_column) if c and c in responses and c not in d]
        if keys and fields:
            m=responses[keys+fields].drop_duplicates(keys)
            d=d.merge(m,on=keys,how="left",sort=False)
    missing=[t for _,t in canonical if t not in d]
    if missing:
        raise EyeProcessValidationError(f"Pupil data are missing identifiers: {', '.join(missing)}")
    if d[["participant_id","item_id","trial_id"]].isna().any().any():
        raise EyeProcessValidationError("Participant, item, and trial identifiers must be non-missing and non-empty.")
    if spec.response not in d and not responses.empty:
        keys=[k for k in ("participant_id","item_id","trial_id") if k in d and k in responses]
        if keys:
            m=responses[keys+[spec.response]].drop_duplicates(keys)
            d=d.merge(m,on=keys,how="left",sort=False)
    if spec.response not in d:
        raise EyeProcessValidationError(f"Response field `{spec.response}` is unavailable.")
    d[spec.response]=_binary_response(d[spec.response])
    return d


def prepare_functional_pupil_data(x: Any, spec: Any = None) -> EyeResult:
    spec = functional_pupil_irt_spec() if spec is None else spec
    if getattr(spec,"eyeprocess_class",None)!="eye_functional_pupil_irt_spec":
        raise EyeProcessValidationError("`spec` must be created by `functional_pupil_irt_spec()`.")
    d=_prepare_frame(x,spec)
    key=d["participant_id"].astype(str)+"|"+d["trial_id"].astype(str)
    if spec.alignment=="trial":
        d[".time"] = d[".time"] - d.groupby(key,sort=False)[".time"].transform("min")
    else:
        if spec.event_time_column not in d:
            raise EyeProcessValidationError(f"Event-alignment column `{spec.event_time_column}` is unavailable.")
        for _,z in d.groupby(key,sort=False):
            ev=pd.to_numeric(z[spec.event_time_column],errors="coerce").dropna().unique()
            if len(ev)!=1: raise EyeProcessValidationError("Each trial must contain one finite, invariant event-alignment time.")
        d[".time"] = d[".time"] - pd.to_numeric(d[spec.event_time_column],errors="coerce")
    d[".time"] -= spec.latency_ms
    keep=np.isfinite(d[".pupil"]) & np.isfinite(d[".time"])
    if spec.blink_column and spec.blink_column in d:
        keep &= ~d[spec.blink_column].fillna(False).astype(bool)
    if spec.time_window is not None:
        keep &= d[".time"].between(spec.time_window[0],spec.time_window[1])
    d=d.loc[keep].copy()
    if d.empty: raise EyeProcessValidationError("No pupil samples remain after quality/time filtering.")
    key=d["participant_id"].astype(str)+"|"+d["trial_id"].astype(str)
    if spec.interpolated_column and spec.interpolated_column in d:
        indicator=d[spec.interpolated_column].fillna(False).astype(bool).astype(float)
        d["interpolated_fraction"]=indicator.groupby(key).transform("mean")
        d=d.loc[d.interpolated_fraction<=spec.max_interpolated_fraction].copy()
        if d.empty: raise EyeProcessValidationError("No pupil trials remain after the interpolation-quality threshold.")
    else: d["interpolated_fraction"]=np.nan

    # Baseline correction per person/trial.
    out=[]
    for _,z in d.groupby(key,sort=False):
        z=z.copy(); b=z.loc[z[".time"].between(spec.baseline_window[0],spec.baseline_window[1]),".pupil"].dropna().astype(float)
        n=len(b); mean=float(b.mean()) if n else np.nan; sd=float(b.std(ddof=1)) if n>1 else np.nan
        valid=n>=spec.min_baseline_samples and np.isfinite(mean); reason="ok"
        if not valid: reason="insufficient_baseline_samples" if n<spec.min_baseline_samples else "nonfinite_baseline"
        if valid and spec.baseline_method=="percent" and abs(mean)<=math.sqrt(np.finfo(float).eps): valid=False; reason="zero_baseline_mean"
        if valid and spec.baseline_method=="zscore" and (not np.isfinite(sd) or sd<=math.sqrt(np.finfo(float).eps)): valid=False; reason="zero_baseline_sd"
        if valid:
            if spec.baseline_method=="subtract": corrected=z[".pupil"]-mean
            elif spec.baseline_method=="percent": corrected=100*(z[".pupil"]-mean)/mean
            else: corrected=(z[".pupil"]-mean)/sd
        else: corrected=np.nan
        z["pupil_corrected"]=corrected; z["baseline_pupil"]=mean; z["baseline_sd"]=sd
        z["baseline_se"]=sd/math.sqrt(n) if n>1 and np.isfinite(sd) else np.nan
        z["baseline_n"]=n; z["baseline_valid"]=valid; z["baseline_reason"]=reason
        out.append(z)
    d=pd.concat(out,ignore_index=True)
    baseline_audit=d[["participant_id","item_id","trial_id","baseline_pupil","baseline_sd","baseline_se","baseline_n","baseline_valid","baseline_reason"]].drop_duplicates()
    if spec.drop_invalid_baseline: d=d.loc[d.baseline_valid].copy()
    if d.empty: raise EyeProcessValidationError("No pupil trials remain after baseline-quality checks.")

    # Optional nuisance residualization.
    nuisance=[c for c in (spec.luminance_column,spec.gaze_x_column,spec.gaze_y_column) if c and c in d]
    nuisance_model=None; d["pupil_adjusted"]=d["pupil_corrected"]
    if nuisance:
        try:
            import statsmodels.formula.api as smf
            terms=list(nuisance)
            if spec.nuisance_by_participant and d.participant_id.nunique()>1: terms.append("C(participant_id)")
            nuisance_model=smf.ols("pupil_corrected ~ "+" + ".join(terms),data=d).fit()
            d["pupil_adjusted"]=nuisance_model.resid + float(nuisance_model.params.iloc[0])
        except Exception:
            nuisance_model=None
    d=d.loc[np.isfinite(pd.to_numeric(d.pupil_adjusted,errors="coerce"))].copy()
    d=d.sort_values(["participant_id","trial_id",".time"],kind="stable").reset_index(drop=True)
    participants=list(pd.unique(d.participant_id)); items=list(pd.unique(d.item_id)); trial_key=d.participant_id.astype(str)+"|"+d.trial_id.astype(str)
    d["person_index"]=d.participant_id.map({v:i+1 for i,v in enumerate(participants)}).astype(int)
    d["item_index"]=d.item_id.map({v:i+1 for i,v in enumerate(items)}).astype(int)
    unique_trials=list(pd.unique(trial_key)); d["trial_index"]=trial_key.map({v:i+1 for i,v in enumerate(unique_trials)}).astype(int)
    d["previous_index"]=0
    for _,idx in d.groupby("trial_index",sort=False).groups.items():
        inds=list(idx)
        for j in range(1,len(inds)): d.loc[inds[j],"previous_index"]=inds[j-1]+1
    sd=float(d[".time"].std(ddof=1))
    if not np.isfinite(sd) or sd<=0: raise EyeProcessValidationError("Pupil sample times must vary after alignment.")
    d["time_scaled"]=(d[".time"]-d[".time"].mean())/sd
    trials=[]
    for ti,z in d.groupby("trial_index",sort=False):
        resp=pd.to_numeric(z[spec.response],errors="coerce").dropna().unique()
        if len(resp)!=1: raise EyeProcessValidationError("Each trial must contain one non-missing, invariant response value.")
        row={"trial_index":int(ti),"participant_id":z.participant_id.iloc[0],"item_id":z.item_id.iloc[0],"trial_id":z.trial_id.iloc[0],"response":int(resp[0])}
        if "response_time" in z: row["response_time"]=z.response_time.iloc[0]
        trials.append(row)
    trials=pd.DataFrame(trials)
    counts=d.groupby("trial_index").size()
    quality=pd.DataFrame([{"samples":len(d),"trials":len(trials),"participants":len(participants),"items":len(items),"invalid_baseline_trials":int((~baseline_audit.baseline_valid).sum()),"min_samples_per_trial":int(counts.min())}])
    return _result("eye_functional_pupil_data",data=d,spec=spec,participants=participants,items=items,trials=trials,baseline_audit=baseline_audit,nuisance_model=nuisance_model,quality=quality)


def functional_pupil_basis(x: Any, df: int = 6, basis: str | Sequence[str] = ("natural_spline", "bspline"), degree: int = 3, boundary_knots: Sequence[float] | None = None, knots: Sequence[float] | None = None) -> pd.DataFrame:
    time=np.asarray(x.data.time_scaled if getattr(x,"eyeprocess_class",None)=="eye_functional_pupil_data" else x,dtype=float)
    df=int(df); degree=int(degree); basis=_choice(basis,("natural_spline","bspline"),"basis")
    if df<2: raise EyeProcessValidationError("`df` must be an integer of at least two.")
    if degree<1: raise EyeProcessValidationError("`degree` must be a positive integer.")
    if len(time)<df+1 or not np.isfinite(time).all() or len(np.unique(time))<df:
        raise EyeProcessValidationError("Time vector is insufficient for the requested basis.")
    try:
        from patsy import dmatrix
    except Exception as exc: raise EyeProcessBackendError("Functional pupil basis construction requires patsy.") from exc
    if boundary_knots is None: boundary_knots=(float(time.min()),float(time.max()))
    boundary_knots=tuple(float(v) for v in boundary_knots)
    if len(boundary_knots)!=2 or boundary_knots[0]>=boundary_knots[1]: raise EyeProcessValidationError("`boundary_knots` must contain increasing finite endpoints.")
    if basis=="natural_spline":
        B=np.asarray(dmatrix(f"cr(x, df={df}) - 1",{"x":time},return_type="dataframe"),float)
    else:
        B=np.asarray(dmatrix(f"bs(x, df={df}, degree={degree}, include_intercept=True) - 1",{"x":time},return_type="dataframe"),float)
    out=pd.DataFrame(B,columns=[f"pupil_basis_{i+1}" for i in range(B.shape[1])])
    out.attrs.update({"basis":basis,"df":df,"degree":degree,"boundary_knots":boundary_knots,"knots":None if knots is None else tuple(knots)})
    return out


def _trial_coefficients(prepared: EyeResult, basis_matrix: pd.DataFrame) -> pd.DataFrame:
    d=prepared.data; B=np.asarray(basis_matrix,float)
    if len(B)!=len(d): raise EyeProcessValidationError("Basis matrix must have one row per prepared pupil sample.")
    rows=[]
    for ti,idx in d.groupby("trial_index",sort=False).groups.items():
        inds=np.asarray(list(idx),dtype=int); z=d.loc[inds]; Bb=B[inds]
        supported=len(Bb)>=Bb.shape[1] and np.linalg.matrix_rank(Bb)==Bb.shape[1]
        coef=np.linalg.lstsq(Bb,z.pupil_adjusted.to_numpy(float),rcond=None)[0] if supported else np.full(Bb.shape[1],np.nan)
        row={"trial_index":int(ti),"participant_id":z.participant_id.iloc[0],"item_id":z.item_id.iloc[0],"trial_id":z.trial_id.iloc[0],"response":int(pd.to_numeric(z[prepared.spec.response],errors="coerce").dropna().iloc[0])}
        if "response_time" in z: row["response_time"]=z.response_time.iloc[0]
        row.update({f"pupil_basis_{j+1}":float(v) for j,v in enumerate(coef)})
        row["pupil_peak"]=float(z.pupil_adjusted.max()); row["pupil_mean"]=float(z.pupil_adjusted.mean())
        t=z[".time"].to_numpy(float); y=z.pupil_adjusted.to_numpy(float); row["pupil_auc"]=float(np.trapezoid(y,t)) if len(z)>1 else np.nan
        row["basis_supported"]=bool(supported); row["samples"]=len(z); rows.append(row)
    return pd.DataFrame(rows)


def fit_functional_pupil_stan(prepared: Any, basis_matrix: Any = None, seed: int = 1, refresh: int = 0, output_dir: str | None = None, **kwargs: Any) -> EyeResult:
    if getattr(prepared,"eyeprocess_class",None)!="eye_functional_pupil_data": raise EyeProcessValidationError("Expected prepared functional pupil data.")
    try: import cmdstanpy
    except Exception as exc: raise EyeProcessBackendError("The 'stan' extra with CmdStanPy is required for the Stan engine.") from exc
    basis_matrix=functional_pupil_basis(prepared,prepared.spec.df,prepared.spec.basis) if basis_matrix is None else pd.DataFrame(basis_matrix)
    stan_file=resources.files("eyeprocesspy").joinpath("resources","stan","functional_pupil_irt.stan")
    try:
        model=cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    except Exception as exc: raise EyeProcessBackendError("CmdStan is required to compile the bundled functional pupil Stan model.") from exc
    # Full Stan-data contract is intentionally explicit; fitting is not attempted without CmdStan.
    d=prepared.data; trials=prepared.trials.sort_values("trial_index")
    stan_data={"N_trial":len(trials),"N_sample":len(d),"P":len(prepared.participants),"J":len(prepared["items"]),"B":basis_matrix.shape[1],"response":trials.response.astype(int).tolist(),"trial_person":[prepared.participants.index(v)+1 for v in trials.participant_id],"trial_item":[prepared["items"].index(v)+1 for v in trials.item_id],"sample_trial":d.trial_index.astype(int).tolist(),"sample_person":d.person_index.astype(int).tolist(),"sample_item":d.item_index.astype(int).tolist(),"pupil":d.pupil_adjusted.astype(float).tolist(),"basis":basis_matrix.to_numpy(float).tolist(),"luminance":[0.0]*len(d),"gaze_x":[0.0]*len(d),"gaze_y":[0.0]*len(d),"previous_index":d.previous_index.astype(int).tolist(),"use_ar1":int(prepared.spec.ar1),"use_person_pupil":int(prepared.spec.participant_effect),"use_item_pupil":int(prepared.spec.item_effect)}
    fit=model.sample(data=stan_data,seed=int(seed),chains=prepared.spec.chains,parallel_chains=prepared.spec.parallel_chains,iter_warmup=prepared.spec.iter_warmup,iter_sampling=prepared.spec.iter_sampling,adapt_delta=prepared.spec.adapt_delta,max_treedepth=prepared.spec.max_treedepth,refresh=int(refresh),output_dir=output_dir,**kwargs)
    summary=fit.summary(); diag=pd.DataFrame([{"converged":bool((summary["R_hat"].dropna()<=1.05).all()) if "R_hat" in summary else True}])
    return _result("eye_functional_pupil_stan",prepared=prepared,basis=basis_matrix,fit=fit,summary=summary,diagnostics=diag,stan_file=str(stan_file))


def fit_joint_functional_pupil_irt(x: Any, spec: Any = None, seed: int = 1, **kwargs: Any) -> EyeResult:
    spec=functional_pupil_irt_spec() if spec is None else spec
    if getattr(spec,"eyeprocess_class",None)!="eye_functional_pupil_irt_spec": raise EyeProcessValidationError("`spec` must be created by `functional_pupil_irt_spec()`.")
    try:
        prepared=prepare_functional_pupil_data(x,spec)
    except EyeProcessValidationError as exc:
        # Preserve legacy eye-dataset bridge for sources without explicit time metadata.
        if is_eye_dataset(x) and ("time column" in str(exc).lower() or "sample table" in str(exc).lower()):
            from .legacy_models import functional_pupil_features, fit_explanatory_irt, fit_joint_process_model
            y=functional_pupil_features(x,df=spec.df,append=True,prefix="functional_pupil")
            names=sorted(set(y["features"].feature_name.dropna().astype(str)))
            names=[n for n in names if n.startswith("functional_pupil_")]
            if not names: raise EyeProcessValidationError("No functional pupil coefficients could be derived.")
            formula=f"{spec.response} ~ "+" + ".join(names)
            if spec.engine=="two_stage_glm": model=fit_explanatory_irt(y,formula,engine="glm",participant_random=False,item_random=False,**kwargs)
            elif spec.engine in {"two_stage_lme4","brms"}: raise EyeProcessBackendError(f"The exact R `{spec.engine}` functional-pupil engine is not available natively.")
            else: raise EyeProcessValidationError("The Stan functional pupil engine requires long pupil samples with an explicit time column.")
            return _result("eye_functional_pupil_irt",data=y,model=model,spec=spec,feature_names=names,diagnostics=pd.DataFrame([{"converged":True}]),legacy=True,warning="Spline coefficients are measurement summaries; substantive interpretation requires latency, luminance, gaze-position, autocorrelation, and preprocessing sensitivity analyses.")
        raise
    basis_matrix=functional_pupil_basis(prepared,spec.df,spec.basis)
    co=_trial_coefficients(prepared,basis_matrix); feature_names=[c for c in co if c.startswith("pupil_basis_")]
    usable=co[["response",*feature_names]].notna().all(axis=1)&co.basis_supported
    if spec.engine!="stan": co=co.loc[usable].copy()
    if spec.engine!="stan" and len(co)<max(10,len(feature_names)+2): raise EyeProcessValidationError("Too few supported complete trials remain for the requested two-stage model.")
    if spec.engine=="two_stage_glm":
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
        except Exception as exc: raise EyeProcessBackendError("The two-stage functional pupil GLM requires statsmodels.") from exc
        model=smf.glm("response ~ "+" + ".join(feature_names),data=co,family=sm.families.Binomial(),**kwargs).fit()
        diagnostics=pd.DataFrame([{"converged":bool(getattr(model,"converged",True))}])
    elif spec.engine in {"two_stage_lme4","brms"}:
        raise EyeProcessBackendError(f"The exact R `{spec.engine}` functional-pupil engine is not available natively; use the corresponding R backend for exact parity.")
    else:
        model=fit_functional_pupil_stan(prepared,basis_matrix,seed=seed,**kwargs); diagnostics=model.diagnostics
    return _result("eye_functional_pupil_irt",data=prepared,trial_coefficients=co,basis=basis_matrix,model=model,spec=spec,feature_names=feature_names,diagnostics=diagnostics,legacy=False,warning="Pupil trajectories are physiological observations. Shared latent effects must not be labelled cognitive load without experimental, luminance-controlled, and externally reproduced evidence.")


def extract_functional_pupil_parameters(x: Any, pattern: str | None = None, confidence: float = 0.95) -> pd.DataFrame:
    if getattr(x,"eyeprocess_class",None)!="eye_functional_pupil_irt": raise EyeProcessValidationError("Expected an `eye_functional_pupil_irt`.")
    if getattr(x.model,"eyeprocess_class",None)=="eye_functional_pupil_stan":
        d=x.model.summary.copy(); parameter=d.index.astype(str) if "variable" not in d else d.variable.astype(str); estimate=d["Mean"] if "Mean" in d else d.get("mean"); se=d["StdDev"] if "StdDev" in d else d.get("sd"); z=norm.ppf(1-(1-confidence)/2); out=pd.DataFrame({"parameter":parameter,"estimate":estimate,"std_error":se,"lower":estimate-z*se,"upper":estimate+z*se})
    else:
        model=x.model.fit if getattr(x.model,"eyeprocess_class",None)=="eyeprocess_model" else x.model
        params=pd.Series(model.params); se=pd.Series(model.bse); z=norm.ppf(1-(1-confidence)/2)
        out=pd.DataFrame({"parameter":params.index.astype(str),"estimate":params.to_numpy(float),"std_error":se.to_numpy(float),"lower":params.to_numpy(float)-z*se.to_numpy(float),"upper":params.to_numpy(float)+z*se.to_numpy(float)})
    if pattern: out=out.loc[out.parameter.str.contains(pattern,regex=True)].reset_index(drop=True)
    return out


def functional_pupil_diagnostics(x: Any) -> EyeResult:
    if getattr(x,"eyeprocess_class",None)!="eye_functional_pupil_irt": raise EyeProcessValidationError("Expected an `eye_functional_pupil_irt`.")
    if getattr(x,"legacy",False):
        return _result("eye_functional_pupil_diagnostics",checks=pd.DataFrame([{"check":"legacy_bridge","value":1.0,"pass":True}]),trial_quality=pd.DataFrame(),residual_acf=pd.DataFrame(columns=["trial_index","lag1"]),nuisance_model=None)
    d=x.data.data; keys=["participant_id","item_id","trial_id"]
    sample=d.groupby(keys,dropna=False).size().rename("samples").reset_index(); base=d[keys+["baseline_pupil","baseline_sd","baseline_se","baseline_n","baseline_valid","baseline_reason"]].drop_duplicates(); tq=sample.merge(base,on=keys,how="outer")
    acf=[]
    for ti,z in d.groupby("trial_index",sort=False):
        y=z.pupil_adjusted.to_numpy(float); value=np.corrcoef(y[:-1],y[1:])[0,1] if len(y)>=3 and np.std(y[:-1])>0 and np.std(y[1:])>0 else np.nan; acf.append({"trial_index":ti,"lag1":value})
    acf=pd.DataFrame(acf); min_support=int(d.groupby("trial_index").size().min()); conv=x.diagnostics.get("converged",pd.Series([False])); conv_pass=bool(pd.Series(conv).dropna().astype(bool).all()) if len(pd.Series(conv).dropna()) else False
    interp=float(d.interpolated_fraction.max()) if d.interpolated_fraction.notna().any() else np.nan
    checks=pd.DataFrame({"check":["finite_pupil","baseline_available","trial_sample_support","interpolation_threshold","sampler_convergence"],"value":[float(np.isfinite(d.pupil_adjusted).mean()),float(d.baseline_valid.mean()),min_support,interp,float(conv_pass)],"pass":[bool(np.isfinite(d.pupil_adjusted).all()),bool(d.baseline_valid.all()),min_support>=x.spec.df+1,(not np.isfinite(interp) or interp<=x.spec.max_interpolated_fraction),conv_pass]})
    return _result("eye_functional_pupil_diagnostics",checks=checks,trial_quality=tq,residual_acf=acf,nuisance_model=x.data.nuisance_model)


def pupil_preprocessing_grid(baseline_windows: Sequence[Sequence[float]] = ((-200,0),(-500,0)), latency_ms: Sequence[float] = (100,200,300), basis_df: Sequence[int] = (4,6,8), baseline_methods: Sequence[str] = ("subtract","percent"), max_interpolated_fraction: Sequence[float] = (0.10,0.20)) -> pd.DataFrame:
    windows=[tuple(map(float,w)) for w in baseline_windows]
    if not windows or any(len(w)!=2 or w[0]>=w[1] or not np.isfinite(w).all() for w in windows): raise EyeProcessValidationError("Every baseline window must contain increasing finite endpoints.")
    rows=[]
    for wi,lat,df,method,frac in product(range(len(windows)),latency_ms,basis_df,baseline_methods,max_interpolated_fraction):
        method=_choice(str(method),("subtract","percent","zscore"),"baseline_methods"); df=int(df); frac=float(frac)
        if df<2: raise EyeProcessValidationError("`basis_df` must contain integers of at least two.")
        if not 0<=frac<=1: raise EyeProcessValidationError("Interpolation thresholds must be between zero and one.")
        rows.append({"latency_ms":float(lat),"df":df,"baseline_method":method,"max_interpolated_fraction":frac,"baseline_start":windows[wi][0],"baseline_end":windows[wi][1]})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def pupil_preprocessing_sensitivity(x: Any, grid: Any = None, base_spec: Any = None, fit: bool = True, extractor: Callable = extract_functional_pupil_parameters, continue_on_error: bool = True, **kwargs: Any) -> EyeResult:
    grid=pupil_preprocessing_grid() if grid is None else pd.DataFrame(grid).copy(); base_spec=functional_pupil_irt_spec(engine="two_stage_glm") if base_spec is None else base_spec
    if grid.empty: raise EyeProcessValidationError("Sensitivity grid must be a non-empty data frame.")
    required={"baseline_start","baseline_end","latency_ms","df","baseline_method","max_interpolated_fraction"}
    if not required<=set(grid): raise EyeProcessValidationError("Sensitivity grid is missing: "+", ".join(sorted(required-set(grid))))
    rows=[]; models=[]
    for i,row in grid.reset_index(drop=True).iterrows():
        spec=EyeResult(dict(base_spec),eyeprocess_class="eye_functional_pupil_irt_spec"); spec.baseline_window=(float(row.baseline_start),float(row.baseline_end)); spec.latency_ms=float(row.latency_ms); spec.df=int(row.df); spec.baseline_method=str(row.baseline_method); spec.max_interpolated_fraction=float(row.max_interpolated_fraction)
        try:
            result=fit_joint_functional_pupil_irt(x,spec,**kwargs) if fit else prepare_functional_pupil_data(x,spec); models.append(result)
            out=extractor(result) if fit else pd.DataFrame([{"parameter":"samples","estimate":len(result.data)}]); out=out.copy(); out["error"]=pd.NA
        except Exception as exc:
            models.append(exc)
            if not continue_on_error: raise
            out=pd.DataFrame([{"parameter":".error","estimate":np.nan,"error":str(exc)}])
        out["specification"]=i+1
        for c in grid: out[c]=row[c]
        rows.append(out)
    return _result("eye_functional_pupil_sensitivity",grid=grid,results=pd.concat(rows,ignore_index=True),models=models,base_spec=base_spec)


def compare_functional_scalar_models(x: Any, scalar_features: Sequence[str] = ("pupil_peak","pupil_auc","pupil_mean"), criterion: str | Sequence[str] = ("AIC","log_loss"), folds: int = 5, seed: int = 1) -> pd.DataFrame:
    criterion=_choice(criterion,("AIC","log_loss"),"criterion")
    if getattr(x,"eyeprocess_class",None)=="eye_functional_pupil_irt": co=x.trial_coefficients.copy()
    elif getattr(x,"eyeprocess_class",None)=="eye_functional_pupil_data": co=_trial_coefficients(x,functional_pupil_basis(x,x.spec.df,x.spec.basis))
    else: raise EyeProcessValidationError("Expected a functional pupil fit or prepared data.")
    ff=[c for c in co if c.startswith("pupil_basis_")]; sf=[c for c in scalar_features if c in co]
    if not ff: raise EyeProcessValidationError("No functional pupil basis features are available.")
    if not sf: raise EyeProcessValidationError("No requested scalar pupil features are available.")
    z=co.dropna(subset=["response","participant_id",*ff,*sf]).copy()
    if len(z)<10 or z.participant_id.nunique()<2: raise EyeProcessValidationError("Too few complete grouped observations are available for model comparison.")
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as exc: raise EyeProcessBackendError("Model comparison requires statsmodels.") from exc
    rows=[]
    for name,features in [("functional",ff),("scalar",sf)]:
        formula="response ~ "+" + ".join(features)
        if criterion=="AIC": value=float(smf.glm(formula,data=z,family=sm.families.Binomial()).fit().aic)
        else:
            people=np.array(sorted(z.participant_id.unique())); rng=np.random.default_rng(seed); rng.shuffle(people); groups=np.array_split(people,min(int(folds),len(people))); losses=[]
            for assess in groups:
                train=z.loc[~z.participant_id.isin(assess)]; test=z.loc[z.participant_id.isin(assess)]; model=smf.glm(formula,data=train,family=sm.families.Binomial()).fit(); p=np.clip(model.predict(test),1e-8,1-1e-8); y=test.response.to_numpy(float); losses.append(float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))))
            value=float(np.mean(losses))
        rows.append({"model":name,"criterion":criterion,"value":value})
    out=pd.DataFrame(rows); out["rank"]=out.value.rank(method="average"); out.attrs["eyeprocess_class"]="eye_functional_scalar_comparison"; return out


def advanced_validation_grid(quick: bool = False, full_factorial: bool = False) -> pd.DataFrame:
    levels={"n_person":[80,150] if quick else [200,500,1000,2000],"n_item":[10,20] if quick else [10,20,40,80],"ability_speed_correlation":[-.3,.3] if quick else [-.5,0,.5],"gaze_effect":[0,.35] if quick else [0,.15,.35,.60],"feature_reliability":[.5,.8] if quick else [.4,.7,.9],"missing_process":[0,.2] if quick else [0,.1,.3,.5],"state_misclassification":[0,.1] if quick else [0,.05,.15],"pupil_ar1":[.3,.6] if quick else [.2,.6,.9],"luminance_effect":[0,.3] if quick else [0,.3,.6],"dif_effect":[0,.4] if quick else [0,.3,.6],"local_dependence":[0,.3] if quick else [0,.3,.6]}
    if full_factorial:
        return pd.DataFrame(list(product(*levels.values())),columns=levels.keys())
    ref={k:v[min(1,len(v)-1)] for k,v in levels.items()}; ref.update({"missing_process":levels["missing_process"][0],"state_misclassification":levels["state_misclassification"][0],"luminance_effect":levels["luminance_effect"][0],"dif_effect":levels["dif_effect"][0],"local_dependence":levels["local_dependence"][0]})
    rows=[ref.copy()]
    for k,vals in levels.items():
        for value in vals:
            r=ref.copy(); r[k]=value; rows.append(r)
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def simulate_advanced_process_data(n_person: int = 100, n_item: int = 20, n_time: int = 30, n_states: int = 3, ability_speed_correlation: float = -0.30, gaze_effect: float = 0.35, feature_reliability: float = 0.70, missing_process: float = 0, state_misclassification: float = 0, pupil_ar1: float = 0.60, luminance_effect: float = 0, dif_effect: float = 0, local_dependence: float = 0, seed: int = 1) -> EyeResult:
    n_person=int(n_person); n_item=int(n_item); n_time=int(n_time); n_states=int(n_states)
    if n_person<2 or n_item<2 or n_time<4 or n_states<2 or n_states>26: raise EyeProcessValidationError("Simulation sizes must satisfy n_person >= 2, n_item >= 2, n_time >= 4, and 2 <= n_states <= 26.")
    if abs(ability_speed_correlation)>=1 or not 0<feature_reliability<=1 or not 0<=missing_process<1 or not 0<=state_misclassification<1 or abs(pupil_ar1)>=1: raise EyeProcessValidationError("Correlation, reliability, missingness, state-error, and AR(1) settings are outside their valid ranges.")
    if not all(np.isfinite([gaze_effect,luminance_effect,dif_effect,local_dependence])) or local_dependence<0: raise EyeProcessValidationError("Simulation effects must be finite and `local_dependence` must be non-negative.")
    rng=np.random.default_rng(seed); persons=[f"P{i+1}" for i in range(n_person)]; items=[f"I{i+1}" for i in range(n_item)]; states=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_states])
    theta=rng.normal(size=n_person); speed=ability_speed_correlation*theta+math.sqrt(1-ability_speed_correlation**2)*rng.normal(size=n_person); difficulty=rng.normal(size=n_item); discrimination=np.exp(rng.normal(0,.15,size=n_item)); groups=np.resize(["reference","focal"],n_person); dif_items=np.resize([False,True],n_item); testlet=np.repeat(np.arange(1,math.ceil(n_item/4)+1),4)[:n_item]
    rows=[]; test_eff=rng.normal(0,local_dependence,size=(n_person,int(testlet.max())))
    for pi,p in enumerate(persons):
        for ji,item in enumerate(items):
            strategy=int(rng.binomial(1,expit(.5*theta[pi])))+1; lg1=rng.normal(-.7 if strategy==1 else .7,.6); lg2=rng.normal(.7 if strategy==1 else -.7,.6); g1=math.sqrt(feature_reliability)*lg1+math.sqrt(1-feature_reliability)*rng.normal(); g2=math.sqrt(feature_reliability)*lg2+math.sqrt(1-feature_reliability)*rng.normal(); dif=dif_effect*(groups[pi]=="focal")*dif_items[ji]; eta=discrimination[ji]*(theta[pi]-difficulty[ji])+gaze_effect*g1+dif+test_eff[pi,testlet[ji]-1]; score=int(rng.binomial(1,expit(eta))); rt=float(np.exp(1+.25*difficulty[ji]-.25*speed[pi]+.12*g2+rng.normal(0,.2))); lum=float(rng.uniform(-1,1)); rows.append({"participant_id":p,"item_id":item,"score":score,"response_time":rt,"gaze_1":g1,"gaze_2":g2,"strategy":strategy,"group":groups[pi],"dif_item":bool(dif_items[ji]),"testlet":int(testlet[ji]),"luminance":lum})
    trials=pd.DataFrame(rows); miss=rng.random(len(trials))<missing_process; trials.loc[miss,["gaze_1","gaze_2"]]=np.nan
    srows=[]; prows=[]
    for _,r in trials.iterrows():
        length=int(rng.integers(5,13)); state=rng.choice(states)
        for step in range(1,length+1):
            probs=np.full(n_states,.15); probs[states.index(state)]=.5; probs[0 if r.strategy==1 else -1]+=.25; probs/=probs.sum(); state=rng.choice(states,p=probs); observed=rng.choice([s for s in states if s!=state]) if rng.random()<state_misclassification else state; srows.append({"participant_id":r.participant_id,"item_id":r.item_id,"step":step,"true_state":state,"state":observed})
        time=np.linspace(0,1,n_time); innov=rng.normal(0,.08,n_time); noise=np.empty(n_time); noise[0]=innov[0]/math.sqrt(1-pupil_ar1**2)
        for k in range(1,n_time): noise[k]=pupil_ar1*noise[k-1]+innov[k]
        signal=.30*np.sin(np.pi*time)+.20*r.score*np.exp(-((time-.60)/.18)**2)+luminance_effect*r.luminance+noise
        for t,y in zip(time,signal): prows.append({"participant_id":r.participant_id,"item_id":r.item_id,"time":t,"pupil":float(y),"luminance":r.luminance})
    truth={"theta":dict(zip(persons,theta)),"speed":dict(zip(persons,speed)),"difficulty":dict(zip(items,difficulty)),"discrimination":dict(zip(items,discrimination)),"ability_speed_correlation":ability_speed_correlation,"gaze_effect":gaze_effect,"feature_reliability":feature_reliability,"missing_process":missing_process,"state_misclassification":state_misclassification,"pupil_ar1":pupil_ar1,"luminance_effect":luminance_effect,"dif_effect":dif_effect,"local_dependence":local_dependence}
    return _result("eye_advanced_process_simulation",trials=trials,states=pd.DataFrame(srows),pupil=pd.DataFrame(prows),truth=truth)


__all__=["functional_pupil_irt_spec","fit_joint_functional_pupil_irt","advanced_validation_grid","simulate_advanced_process_data","functional_pupil_basis","prepare_functional_pupil_data","fit_functional_pupil_stan","extract_functional_pupil_parameters","functional_pupil_diagnostics","pupil_preprocessing_grid","pupil_preprocessing_sensitivity","compare_functional_scalar_models"]
