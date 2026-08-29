"""Validation programmes, governed pipelines, API lifecycle, multiverse sensitivity,
and research-decision manifests from eyeprocess 0.11.1.

This module ports the dependency-light public contracts in R/069--073.  The
functions are governance/validation infrastructure: they record explicit
choices and empirical checks, but do not certify substantive construct validity.
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module, resources
import inspect
import itertools
import json
import math
from pathlib import Path
import pickle
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from .exceptions import EyeProcessValidationError
from .irt import EyeResult, _result, _stable_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is(x: Any, cls: str) -> bool:
    return isinstance(x, EyeResult) and getattr(x, "eyeprocess_class", None) == cls


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, Mapping):
        try:
            return pd.DataFrame(x)
        except ValueError:
            return pd.DataFrame([x])
    try:
        return pd.DataFrame(x)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _req(d: pd.DataFrame, cols: Sequence[str | None], label: str = "data") -> None:
    need = [str(c) for c in cols if c is not None]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise EyeProcessValidationError(f"{label} is missing required column(s): {', '.join(miss)}")


def _num(x: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)


def _hash_object(x: Any) -> str:
    if callable(x):
        try:
            return _stable_hash(inspect.getsource(x))
        except (OSError, TypeError):
            return _stable_hash(getattr(x, "__qualname__", repr(x)))
    return _stable_hash(x)


def _rbind_fill(parts: Sequence[Any]) -> pd.DataFrame:
    dfs = []
    for x in parts:
        if x is None:
            continue
        d = _df(x)
        if len(d):
            dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    cols: list[str] = []
    for d in dfs:
        for c in d.columns:
            if c not in cols:
                cols.append(c)
    return pd.concat([d.reindex(columns=cols) for d in dfs], ignore_index=True)


def _capture(fun: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = fun(*args, **kwargs)
            error = None
        except Exception as exc:  # validation runner intentionally captures estimator failures
            value = exc
            error = str(exc)
    return {"value": value, "error": error, "warnings": list(dict.fromkeys(str(w.message) for w in caught))}


# ---------------------------------------------------------------------------
# R/069 validation programme


def process_validation_design(
    n_persons: Any = (50, 150, 500),
    n_trials: Any = (10, 30, 80),
    missingness: Any = (0, .05, .15, .30),
    sampling_rate_hz: Any = (60, 120, 300, 1000),
    aoi_error: Any = ("low", "moderate", "severe"),
    calibration_error: Any = (0, .5, 1),
    pupil_dropout: Any = (0, .10, .30),
    heterogeneity: Any = ("low", "moderate"),
    model_misspecification: Any = (False, True),
    replications: int = 100,
    seed: int = 1,
    label: str = "process_validation",
) -> EyeResult:
    def seq(v: Any) -> list[Any]:
        if isinstance(v, (str, bytes)) or np.isscalar(v):
            return [v]
        return list(v)
    x = _result(
        "eye_process_validation_design",
        n_persons=[int(v) for v in seq(n_persons)],
        n_trials=[int(v) for v in seq(n_trials)],
        missingness=[float(v) for v in seq(missingness)],
        sampling_rate_hz=[float(v) for v in seq(sampling_rate_hz)],
        aoi_error=[str(v) for v in seq(aoi_error)],
        calibration_error=[float(v) for v in seq(calibration_error)],
        pupil_dropout=[float(v) for v in seq(pupil_dropout)],
        heterogeneity=[str(v) for v in seq(heterogeneity)],
        model_misspecification=[bool(v) for v in seq(model_misspecification)],
        replications=int(replications), seed=int(seed), label=str(label), status="validation_design",
    )
    validate_process_validation_design(x)
    return x


def validate_process_validation_design(x: Any) -> bool:
    if not _is(x, "eye_process_validation_design"):
        raise EyeProcessValidationError("x must be an eye_process_validation_design.")
    if not x.n_persons or any(not np.isfinite(v) or int(v) < 2 for v in x.n_persons):
        raise EyeProcessValidationError("n_persons must contain integers >= 2.")
    if not x.n_trials or any(not np.isfinite(v) or int(v) < 2 for v in x.n_trials):
        raise EyeProcessValidationError("n_trials must contain integers >= 2.")
    if not x.missingness or any(not np.isfinite(v) or v < 0 or v >= 1 for v in x.missingness):
        raise EyeProcessValidationError("missingness must lie in [0, 1).")
    if not x.pupil_dropout or any(not np.isfinite(v) or v < 0 or v >= 1 for v in x.pupil_dropout):
        raise EyeProcessValidationError("pupil_dropout must lie in [0, 1).")
    if not x.sampling_rate_hz or any(not np.isfinite(v) or v <= 0 for v in x.sampling_rate_hz):
        raise EyeProcessValidationError("sampling_rate_hz must be positive.")
    if not x.calibration_error or any(not np.isfinite(v) or v < 0 for v in x.calibration_error):
        raise EyeProcessValidationError("calibration_error must contain finite non-negative values.")
    if not x.model_misspecification:
        raise EyeProcessValidationError("model_misspecification must contain non-missing logical values.")
    if not np.isfinite(x.seed) or x.seed < 0:
        raise EyeProcessValidationError("seed must be a finite non-negative integer.")
    if not x.aoi_error or any(not str(v) for v in x.aoi_error):
        raise EyeProcessValidationError("aoi_error cannot be empty.")
    if not x.heterogeneity or any(not str(v) for v in x.heterogeneity):
        raise EyeProcessValidationError("heterogeneity cannot be empty.")
    if not np.isfinite(x.replications) or int(x.replications) < 1:
        raise EyeProcessValidationError("replications must be a positive integer.")
    return True


def expand_process_validation_design(x: Any, max_conditions: float = 250000) -> pd.DataFrame:
    validate_process_validation_design(x)
    if not np.isfinite(max_conditions) and not math.isinf(float(max_conditions)):
        raise EyeProcessValidationError("max_conditions must be a non-negative scalar or Inf.")
    if float(max_conditions) < 0:
        raise EyeProcessValidationError("max_conditions must be a non-negative scalar or Inf.")
    keys = ["n_persons", "n_trials", "missingness", "sampling_rate_hz", "aoi_error", "calibration_error", "pupil_dropout", "heterogeneity", "model_misspecification"]
    vals = [x[k] for k in keys]
    rows = [dict(zip(keys, z)) for z in itertools.product(*vals)]
    if len(rows) > max_conditions:
        raise EyeProcessValidationError(f"Validation design expands to {len(rows)} conditions; max_conditions={max_conditions}.")
    g = pd.DataFrame(rows)
    g.insert(0, "condition_id", [f"C{i:06d}" for i in range(1, len(g) + 1)])
    g["replications"] = int(x.replications)
    g["design_label"] = x.label
    g.attrs["master_seed"] = int(x.seed)
    return g


def validation_condition_id(x: Any) -> list[str]:
    if _is(x, "eye_process_validation_design"):
        x = expand_process_validation_design(x)
    d = _df(x)
    _req(d, ["condition_id"], "x")
    return d["condition_id"].astype(str).tolist()


def simulate_process_validation_data(condition: Any, replication: int = 1, seed: int | None = None, beta: float = .35) -> EyeResult:
    c = _df(condition, "condition")
    if len(c) != 1:
        raise EyeProcessValidationError("condition must contain exactly one row.")
    _req(c, ["n_persons", "n_trials", "missingness", "sampling_rate_hz", "aoi_error", "calibration_error", "pupil_dropout", "heterogeneity", "model_misspecification"], "condition")
    if seed is None:
        seed = 100000 + int(replication)
    if not np.isfinite(seed) or int(seed) < 0:
        raise EyeProcessValidationError("seed must be a finite non-negative scalar.")
    if not np.isfinite(beta):
        raise EyeProcessValidationError("beta must be a finite scalar.")
    rng = np.random.default_rng((int(seed) % (2**31 - 2)) + 1)
    row = c.iloc[0]
    n, t = int(row.n_persons), int(row.n_trials)
    hetero_sd = {"low": .25, "moderate": .60, "high": 1.0}.get(str(row.heterogeneity), .60)
    aoi_sd = {"low": .01, "moderate": .03, "severe": .08}.get(str(row.aoi_error), .03)
    pid = np.repeat(np.arange(1, n + 1), t)
    trial = np.tile(np.arange(1, t + 1), n)
    x = rng.normal(size=n * t)
    u = np.repeat(rng.normal(scale=hetero_sd, size=n), t)
    eps = rng.normal(size=n * t)
    omitted = .60 * x + math.sqrt(1 - .60**2) * rng.normal(size=n * t)
    misspecified = bool(row.model_misspecification)
    y = beta * x + u + eps + (.35 * omitted if misspecified else 0)
    pupil = 3.5 + .12 * x + .15 * u + rng.normal(scale=.20, size=n * t)
    gaze_x = 1 / (1 + np.exp(-(.3 * x + rng.normal(scale=.6, size=n * t)))) + rng.normal(scale=aoi_sd, size=n * t)
    gaze_y = 1 / (1 + np.exp(-(-.2 * x + rng.normal(scale=.6, size=n * t)))) + rng.normal(scale=aoi_sd, size=n * t)
    cal = float(row.calibration_error)
    if cal:
        gaze_x += rng.normal(scale=cal / 100, size=n * t)
        gaze_y += rng.normal(scale=cal / 100, size=n * t)
    miss = rng.random(n * t) < float(row.missingness)
    drop = rng.random(n * t) < float(row.pupil_dropout)
    y[miss] = np.nan; gaze_x[miss] = np.nan; gaze_y[miss] = np.nan; pupil[drop] = np.nan
    interval = 1000 / float(row.sampling_rate_hz)
    timestamp_ms = (trial - 1) * interval + rng.normal(scale=interval * .02, size=n * t)
    d = pd.DataFrame({"person_id": pid, "trial_id": trial, "timestamp_ms": timestamp_ms, "x": x,
                      "omitted_structure": omitted, "process_value": y, "pupil": pupil, "gaze_x": gaze_x,
                      "gaze_y": gaze_y, "valid_gaze": np.isfinite(gaze_x) & np.isfinite(gaze_y)})
    truth = pd.DataFrame({"parameter": ["beta_x"], "truth": [float(beta)]})
    return _result("eye_process_validation_simulation", data=d, truth=truth, condition=c, replication=int(replication))


def _default_validation_fit(simulated: Any, condition: Any = None) -> EyeResult:
    d = simulated.data if isinstance(simulated, EyeResult) else simulated["data"]
    xx = pd.to_numeric(d["x"], errors="coerce").to_numpy(float)
    yy = pd.to_numeric(d["process_value"], errors="coerce").to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 3:
        raise EyeProcessValidationError("Not enough finite observations for validation fit.")
    res = linregress(xx[ok], yy[ok])
    return _result("eye_reference_lm", intercept=float(res.intercept), slope=float(res.slope), se=float(res.stderr))


def _default_validation_extract(fit: Any, simulated: Any, condition: Any = None) -> pd.DataFrame:
    est = float(fit.slope); se = float(fit.se)
    truth = float(simulated.truth.loc[simulated.truth.parameter == "beta_x", "truth"].iloc[0])
    return pd.DataFrame({"parameter": ["beta_x"], "truth": [truth], "estimate": [est], "se": [se],
                         "lower": [est - 1.96 * se], "upper": [est + 1.96 * se], "converged": [np.isfinite(est)]})


def run_process_validation(design: Any, simulate_fun: Callable = simulate_process_validation_data,
                           fit_fun: Callable = _default_validation_fit, extract_fun: Callable = _default_validation_extract,
                           max_conditions: float = math.inf, progress: bool = False) -> EyeResult:
    if _is(design, "eye_process_validation_design"):
        master_seed = int(design.seed); cond = expand_process_validation_design(design)
    else:
        cond = _df(design, "design"); _req(cond, ["condition_id", "replications"], "design")
        master_seed = int(cond.attrs.get("master_seed", 1))
    if float(max_conditions) < 0 or (not np.isfinite(max_conditions) and not math.isinf(float(max_conditions))):
        raise EyeProcessValidationError("max_conditions must be a non-negative scalar or Inf.")
    if np.isfinite(max_conditions):
        cond = cond.head(int(max_conditions)).copy()
    rows: list[pd.DataFrame] = []; failures: list[pd.DataFrame] = []; warns: list[pd.DataFrame] = []
    for i, (_, cnd) in enumerate(cond.iterrows(), start=1):
        reps = int(cnd.replications)
        if reps < 1:
            continue
        one = cnd.to_frame().T
        for r in range(1, reps + 1):
            seed_i = int(((float(master_seed) + i * 100000 + r) % (2**31 - 2)) + 1)
            if progress:
                print(f"validation {cnd.condition_id} replication {r}/{reps}")
            sim = _capture(simulate_fun, one, r, seed_i)
            if sim["error"] is not None:
                failures.append(pd.DataFrame({"condition_id": [cnd.condition_id], "replication": [r], "stage": ["simulate"], "error": [sim["error"]]})); continue
            if sim["warnings"]:
                warns.append(pd.DataFrame({"condition_id": [cnd.condition_id], "replication": [r], "stage": ["simulate"], "warning": [" | ".join(sim["warnings"])]}))
            fit = _capture(fit_fun, sim["value"], one)
            if fit["error"] is not None:
                failures.append(pd.DataFrame({"condition_id": [cnd.condition_id], "replication": [r], "stage": ["fit"], "error": [fit["error"]]})); continue
            if fit["warnings"]:
                warns.append(pd.DataFrame({"condition_id": [cnd.condition_id], "replication": [r], "stage": ["fit"], "warning": [" | ".join(fit["warnings"])]}))
            ext = _capture(extract_fun, fit["value"], sim["value"], one)
            if ext["error"] is not None:
                failures.append(pd.DataFrame({"condition_id": [cnd.condition_id], "replication": [r], "stage": ["extract"], "error": [ext["error"]]})); continue
            if ext["warnings"]:
                warns.append(pd.DataFrame({"condition_id": [cnd.condition_id], "replication": [r], "stage": ["extract"], "warning": [" | ".join(ext["warnings"])]}))
            tab = _df(ext["value"], "extract_fun result")
            if not len(tab):
                continue
            tab["condition_id"] = cnd.condition_id; tab["replication"] = r; tab["seed"] = seed_i
            for nm in [z for z in one.columns if z not in ("condition_id", "replications")]:
                tab[nm] = one.iloc[0][nm]
            rows.append(tab)
    return _result("eye_process_validation_result", design=cond.reset_index(drop=True), estimates=_rbind_fill(rows), failures=_rbind_fill(failures),
                   warnings=_rbind_fill(warns), created_at=_now(), design_hash=_hash_object(cond.reset_index(drop=True)),
                   status="empirical_validation_result",
                   caveat="Recovery and robustness are conditional on the supplied data-generating process, estimator, and extraction rules. Passing a simulation study does not establish validity outside the simulated conditions.")


def summarise_process_validation(x: Any, by: Any = None) -> pd.DataFrame:
    if not _is(x, "eye_process_validation_result"):
        raise EyeProcessValidationError("x must be an eye_process_validation_result.")
    d = x.estimates.copy()
    if not len(d): return pd.DataFrame()
    _req(d, ["parameter", "estimate", "truth"], "x$estimates")
    byv = [] if by is None else ([by] if isinstance(by, str) else list(by))
    groups = [c for c in ["parameter", *byv] if c in d.columns]
    rows = []
    grouper = groups[0] if len(groups) == 1 else groups
    for keys, z in d.groupby(grouper, dropna=False, sort=True):
        if len(groups) == 1: keys = (keys,)
        row = dict(zip(groups, keys))
        est = _num(z.estimate); truth = _num(z.truth); diff = est - truth; finite = np.isfinite(diff)
        if {"lower", "upper"}.issubset(z.columns):
            cv = (_num(z.lower) <= truth) & (_num(z.upper) >= truth); cov = float(np.mean(cv[np.isfinite(truth) & np.isfinite(_num(z.lower)) & np.isfinite(_num(z.upper))])) if np.any(np.isfinite(truth) & np.isfinite(_num(z.lower)) & np.isfinite(_num(z.upper))) else math.nan
        else: cov = math.nan
        if "converged" in z: conv = pd.Series(z.converged).astype("boolean").mean(skipna=True)
        else: conv = float(np.mean(np.isfinite(est)))
        ae = np.abs(diff[finite])
        row.update(n=len(z), bias=float(np.mean(diff[finite])) if finite.any() else math.nan,
                   rmse=float(np.sqrt(np.mean(diff[finite] ** 2))) if finite.any() else math.nan,
                   mae=float(np.mean(ae)) if ae.size else math.nan, coverage=cov,
                   convergence_rate=float(conv) if pd.notna(conv) else math.nan,
                   estimate_sd=float(np.std(est[np.isfinite(est)], ddof=1)) if np.isfinite(est).sum() > 1 else math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def validation_recovery_table(x: Any, by: Any = None) -> pd.DataFrame:
    s = summarise_process_validation(x, by)
    byv = [] if by is None else ([by] if isinstance(by, str) else list(by))
    keep = [c for c in ["parameter", *byv, "n", "bias", "rmse", "mae", "estimate_sd"] if c in s]
    return s[keep]


def validation_coverage_table(x: Any, nominal: float = .95, by: Any = None) -> pd.DataFrame:
    s = summarise_process_validation(x, by)
    if not len(s): return s
    s["nominal"] = float(nominal); s["coverage_error"] = s.coverage - float(nominal)
    byv = [] if by is None else ([by] if isinstance(by, str) else list(by))
    keep = [c for c in ["parameter", *byv, "n", "coverage", "nominal", "coverage_error"] if c in s]
    return s[keep]


def validation_failure_profile(x: Any) -> pd.DataFrame:
    if not _is(x, "eye_process_validation_result"):
        raise EyeProcessValidationError("x must be an eye_process_validation_result.")
    total = int(pd.to_numeric(x.design.replications, errors="coerce").fillna(0).sum())
    if not len(x.failures):
        return pd.DataFrame(columns=["stage", "failures", "total_attempts", "failure_rate"])
    tab = x.failures.groupby("stage").size().rename("failures").reset_index()
    tab["total_attempts"] = total; tab["failure_rate"] = tab.failures / total if total else np.nan
    return tab


def validation_summary_mcse(x: Any, by: Any = None) -> pd.DataFrame:
    d = x.estimates.copy()
    if not len(d): return pd.DataFrame()
    byv = [] if by is None else ([by] if isinstance(by, str) else list(by))
    groups = [c for c in ["parameter", *byv] if c in d]
    rows = []
    grouper = groups[0] if len(groups) == 1 else groups
    for keys, z in d.groupby(grouper, dropna=False, sort=True):
        if len(groups) == 1: keys = (keys,)
        row = dict(zip(groups, keys)); err = _num(z.estimate) - _num(z.truth); err = err[np.isfinite(err)]; est = _num(z.estimate); est = est[np.isfinite(est)]
        n = len(err); row.update(n=n, mcse_bias=float(np.std(err, ddof=1)/math.sqrt(n)) if n > 1 else math.nan,
                                 mcse_mean_estimate=float(np.std(est, ddof=1)/math.sqrt(n)) if n > 1 and len(est)>1 else math.nan); rows.append(row)
    return pd.DataFrame(rows)


def validation_condition_ranking(x: Any, weights: Any = None) -> pd.DataFrame:
    if weights is None: weights = {"rmse": 1, "abs_bias": 1, "coverage_error": 1, "failure_rate": 1}
    if not isinstance(weights, Mapping) or any(k not in weights or not np.isfinite(weights[k]) for k in ["rmse", "abs_bias", "coverage_error", "failure_rate"]):
        raise EyeProcessValidationError("weights must be a finite named numeric vector containing rmse, abs_bias, coverage_error, and failure_rate.")
    s = summarise_process_validation(x, by="condition_id")
    if not len(s): return s
    if len(x.failures): f = x.failures.groupby("condition_id").size().rename("failures").reset_index()
    else: f = pd.DataFrame(columns=["condition_id", "failures"])
    s = s.merge(f, on="condition_id", how="left"); s["failures"] = pd.to_numeric(s["failures"], errors="coerce").fillna(0.0)
    reps = x.design.set_index("condition_id").replications.to_dict(); s["failure_rate"] = [r and f/reps.get(cid, r) for cid, f, r in zip(s.condition_id, s.failures, [reps.get(c,1) for c in s.condition_id])]
    s["abs_bias"] = s.bias.abs(); s["coverage_error"] = (s.coverage - .95).abs()
    def scale(v: pd.Series) -> np.ndarray:
        a = pd.to_numeric(v, errors="coerce").to_numpy(float); fin = a[np.isfinite(a)]
        if not fin.size or np.ptp(fin) == 0: return np.zeros(len(a))
        z=(a-fin.min())/(fin.max()-fin.min()); z[~np.isfinite(z)] = 0; return z
    ks=["rmse","abs_bias","coverage_error","failure_rate"]; denom=sum(abs(float(weights[k])) for k in ks) or 1
    penalty=sum(float(weights[k])*scale(s[k]) for k in ks)/denom; s["robustness_score"] = 1-penalty
    return s.sort_values("robustness_score", ascending=False).reset_index(drop=True)


def validation_robustness_score(x: Any) -> float:
    r=validation_condition_ranking(x); z=pd.to_numeric(r.get("robustness_score", pd.Series(dtype=float)), errors="coerce").dropna()
    return float(z.mean()) if len(z) else math.nan


def freeze_validation_reference(x: Any, path: str | Path | None = None, digits: int = 8) -> EyeResult:
    if not _is(x, "eye_process_validation_result"):
        raise EyeProcessValidationError("x must be an eye_process_validation_result.")
    if int(digits) != digits or digits < 0: raise EyeProcessValidationError("digits must be a non-negative integer.")
    s=summarise_process_validation(x, by="condition_id")
    for c in s.select_dtypes(include=[np.number]).columns: s[c]=s[c].round(int(digits))
    ref=_result("eye_validation_reference", summary=s, failure_profile=validation_failure_profile(x), design_hash=x.design_hash,
                summary_hash=_hash_object(s), created_at=_now(), status="frozen_validation_reference")
    if path is not None:
        with open(path,"wb") as fh: pickle.dump(ref, fh)
    return ref


def validate_against_reference(x: Any, reference: Any, tolerance: float = 1e-6) -> EyeResult:
    if not np.isfinite(tolerance) or tolerance < 0: raise EyeProcessValidationError("tolerance must be a finite non-negative scalar.")
    if isinstance(reference, (str, Path)):
        with open(reference,"rb") as fh: reference=pickle.load(fh)
    if not _is(reference,"eye_validation_reference"): raise EyeProcessValidationError("reference must be an eye_validation_reference or RDS path.")
    cur=summarise_process_validation(x, by="condition_id"); ref=reference.summary.copy(); keys=[k for k in ["condition_id","parameter"] if k in cur and k in ref]
    if not keys: raise EyeProcessValidationError("No common keys between current and frozen summaries.")
    ref[".present_reference"]=True; cur[".present_current"]=True
    m=ref.merge(cur,on=keys,how="outer",suffixes=("_reference","_current")); matched=m[".present_reference"].eq(True)&m[".present_current"].eq(True)
    metrics=[z for z in ["bias","rmse","coverage","convergence_rate"] if z in reference.summary]
    for metric in metrics:
        m[f"{metric}_delta"]=pd.to_numeric(m.get(f"{metric}_current"),errors="coerce")-pd.to_numeric(m.get(f"{metric}_reference"),errors="coerce")
    delta=[c for c in m if c.endswith("_delta")]
    numeric_ok=bool(delta) and all(((m.loc[matched,c].abs()<=tolerance)|m.loc[matched,c].isna()).all() for c in delta)
    out = _result("eye_validation_reference_comparison", table=m, tolerance=float(tolerance),
                  reference_hash=reference.summary_hash, current_hash=_hash_object(cur))
    out["pass"] = bool(len(m) > 0 and matched.all() and numeric_ok)
    return out


def validation_evidence_matrix(**kwargs: Any) -> pd.DataFrame:
    if not kwargs: return pd.DataFrame()
    evidence_types=["recovery","bias","rmse","coverage","convergence","failure","sensitivity","negative_controls","external_validation","provenance"]
    rows=[]
    for name,x in kwargs.items():
        avail={k:False for k in evidence_types}
        if _is(x,"eye_process_validation_result"):
            for k in ["recovery","bias","rmse","coverage","convergence","failure"]: avail[k]=True
        if _is(x,"eye_validation_bundle"):
            for k in set(getattr(x,"evidence",{})).intersection(avail): avail[k]=True
        rows.append({"model":name,**avail})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R/070 governed pipelines


def eye_analysis_spec(blink_correction: Any = None, pupil_baseline: Any = None, fixation_algorithm: Any = None,
                      aoi_rule: Any = None, exclusions: Any = None, model: Any = None, sensitivity: Any = None,
                      **kwargs: Any) -> EyeResult:
    decisions={"blink_correction":blink_correction,"pupil_baseline":pupil_baseline,"fixation_algorithm":fixation_algorithm,
               "aoi_rule":aoi_rule,"exclusions":exclusions,"model":model,"sensitivity":sensitivity,**kwargs}
    return _result("eye_analysis_spec", decisions=decisions, created_at=_now(), hash=_hash_object(decisions), status="explicit_analysis_spec",
                   caveat="The specification records user-selected decisions; eyeprocess does not infer that these choices are substantively optimal.")


def eye_pipeline_step(name: str, fun: Callable[...,Any], requires: Any = (), optional: bool = False,
                      description: str | None = None, decision: str | None = None) -> EyeResult:
    name=str(name)
    if not name or not name.isidentifier(): raise EyeProcessValidationError("name must be a non-empty syntactic R name.")
    if not callable(fun): raise EyeProcessValidationError("fun must be a function.")
    req=[] if requires is None else ([str(requires)] if isinstance(requires,str) else list(map(str,requires)))
    req=list(dict.fromkeys(req))
    if any(not z for z in req): raise EyeProcessValidationError("requires must contain non-empty step names.")
    if name in req: raise EyeProcessValidationError("A step cannot require itself.")
    return _result("eye_pipeline_step", name=name, fun=fun, requires=req, optional=bool(optional), description="" if description is None else str(description),
                   decision=None if decision is None else str(decision), function_hash=_hash_object(fun))


def _toposort(steps: Mapping[str,Any]) -> list[str]:
    names=list(steps); deps={n:list(steps[n].requires) for n in names}; unknown=sorted(set(itertools.chain.from_iterable(deps.values()))-set(names))
    if unknown: raise EyeProcessValidationError(f"Unknown pipeline dependency/dependencies: {', '.join(unknown)}")
    incoming={n:len(deps[n]) for n in names}; ready=[n for n in names if incoming[n]==0]; order=[]
    while ready:
        cur=ready.pop(0); order.append(cur)
        for ch in [n for n in names if cur in deps[n]]:
            incoming[ch]-=1
            if incoming[ch]==0 and ch not in order and ch not in ready: ready.append(ch)
    if len(order)!=len(names): raise EyeProcessValidationError("Pipeline dependency graph contains a cycle.")
    return order


def eye_analysis_pipeline(*args: Any, spec: Any = None, name: str = "eye_analysis", strict: bool = True) -> EyeResult:
    steps=list(args)
    if len(steps)==1 and isinstance(steps[0],(list,tuple)) and not _is(steps[0],"eye_pipeline_step"): steps=list(steps[0])
    if not steps: raise EyeProcessValidationError("At least one eye_pipeline_step is required.")
    if not all(_is(s,"eye_pipeline_step") for s in steps): raise EyeProcessValidationError("All pipeline entries must be eye_pipeline_step objects.")
    names=[s.name for s in steps]
    if len(set(names))!=len(names): raise EyeProcessValidationError("Pipeline step names must be unique.")
    obj=_result("eye_analysis_pipeline", name=str(name), steps=dict(zip(names,steps)), spec=eye_analysis_spec() if spec is None else spec,
                strict=bool(strict), created_at=_now(), status="governed_pipeline")
    validate_eye_pipeline(obj); obj["hash"]=_hash_object(eye_pipeline_manifest(obj)); return obj


def validate_eye_pipeline(x: Any) -> bool:
    if not _is(x,"eye_analysis_pipeline"): raise EyeProcessValidationError("x must be an eye_analysis_pipeline.")
    if not x.steps: raise EyeProcessValidationError("Pipeline has no steps.")
    _toposort(x.steps)
    if not _is(x.spec,"eye_analysis_spec"): raise EyeProcessValidationError("Pipeline spec must be an eye_analysis_spec.")
    if x.strict:
        step_names=set(x.steps); special={"context","spec","_context","_spec"}
        bad=[]
        for nm,step in x.steps.items():
            params=set(inspect.signature(step.fun).parameters)-special
            candidates=params&step_names; undeclared=candidates-set(step.requires)
            if undeclared: bad.append(f"{nm} -> {', '.join(sorted(undeclared))}")
        if bad: raise EyeProcessValidationError("Strict pipeline contains formal step dependencies not declared in `requires`: "+"; ".join(bad)+".")
    return True


def eye_pipeline_graph(x: Any) -> EyeResult:
    validate_eye_pipeline(x); order=_toposort(x.steps)
    vertices=pd.DataFrame({"step":list(x.steps),"optional":[x.steps[n].optional for n in x.steps],
                           "decision":["" if x.steps[n].decision is None else x.steps[n].decision for n in x.steps],
                           "order":[order.index(n)+1 for n in x.steps]}).sort_values("order").reset_index(drop=True)
    edges=_rbind_fill([pd.DataFrame({"from":s.requires,"to":[s.name]*len(s.requires)}) if s.requires else None for s in x.steps.values()])
    return _result("eye_pipeline_graph", vertices=vertices, edges=edges)


def eye_pipeline_manifest(x: Any) -> pd.DataFrame:
    validate_eye_pipeline(x); g=eye_pipeline_graph(x); order=list(g.vertices.step)
    return pd.DataFrame({"pipeline":x.name,"step":list(x.steps),"requires":[";".join(x.steps[n].requires) for n in x.steps],
                         "optional":[x.steps[n].optional for n in x.steps],"decision":["" if x.steps[n].decision is None else x.steps[n].decision for n in x.steps],
                         "description":[x.steps[n].description for n in x.steps],"function_hash":[x.steps[n].function_hash for n in x.steps],
                         "execution_order":[order.index(n)+1 for n in x.steps]})


def _call_step(step: EyeResult, outputs: Mapping[str,Any], context: Mapping[str,Any], spec: Any) -> Any:
    deps={n:outputs[n] for n in step.requires}; sig=inspect.signature(step.fun); params=sig.parameters
    special={"spec":spec,"context":context,"_spec":spec,"_context":context}
    if any(p.kind==inspect.Parameter.VAR_KEYWORD for p in params.values()): return step.fun(**deps,**special)
    cand={**deps,**special}; use={k:v for k,v in cand.items() if k in params}; return step.fun(**use)


def run_eye_pipeline(x: Any, context: Any = None, stop_on_error: bool = True, previous: Any = None) -> EyeResult:
    validate_eye_pipeline(x); context={} if context is None else context
    if not isinstance(context,Mapping): raise EyeProcessValidationError("context must be a list.")
    order=_toposort(x.steps); outputs={}; records={}; errors={}; warns={}; context_hash=_hash_object(context)
    if previous is not None:
        if not _is(previous,"eye_pipeline_run"): raise EyeProcessValidationError("previous must be an eye_pipeline_run.")
        if previous.pipeline_hash!=x.hash: raise EyeProcessValidationError("previous run was generated from a different pipeline manifest.")
        if previous.context_hash!=context_hash: raise EyeProcessValidationError("previous run used a different context; rerun from the beginning rather than reusing context-dependent outputs.")
        outputs=dict(previous.outputs); errors=dict(previous.errors); warns=dict(previous.warnings)
        if len(previous.records): records={r.step:r.to_frame().T for _,r in previous.records.iterrows()}
    for nm in order:
        if nm in outputs: continue
        step=x.steps[nm]; missing=[d for d in step.requires if d not in outputs]
        if missing:
            msg="Required upstream outputs unavailable: "+", ".join(missing); errors[nm]=msg
            if not step.optional and stop_on_error: raise EyeProcessValidationError(f"Step '{nm}' cannot run. {msg}")
            continue
        errors.pop(nm,None); warns.pop(nm,None); start=datetime.now(timezone.utc); cap=_capture(_call_step,step,outputs,context,x.spec); elapsed=(datetime.now(timezone.utc)-start).total_seconds()
        if cap["warnings"]: warns[nm]=cap["warnings"]
        ok=cap["error"] is None
        records[nm]=pd.DataFrame({"step":[nm],"status":["success" if ok else ("optional_error" if step.optional else "error")],"optional":[step.optional],
                                  "elapsed_sec":[elapsed],"error":[None if ok else cap["error"]],"output_hash":[_hash_object(cap["value"]) if ok else None]})
        if ok: outputs[nm]=cap["value"]
        else:
            errors[nm]=cap["error"]
            if not step.optional and stop_on_error: raise EyeProcessValidationError(f"Pipeline step '{nm}' failed: {cap['error']}")
    rec=_rbind_fill(list(records.values())); completed=all(n in outputs or x.steps[n].optional for n in order)
    return _result("eye_pipeline_run",pipeline=x,pipeline_hash=x.hash,outputs=outputs,records=rec,errors=errors,warnings=warns,context_hash=context_hash,
                   completed=completed,created_at=_now(),status="pipeline_complete" if completed else "pipeline_incomplete")


def resume_eye_pipeline(x: Any, previous: Any, context: Any = None, stop_on_error: bool = True) -> EyeResult:
    return run_eye_pipeline(x,context=context,stop_on_error=stop_on_error,previous=previous)


def audit_eye_pipeline(x: Any) -> EyeResult:
    pipeline=x.pipeline if _is(x,"eye_pipeline_run") else x; validate_eye_pipeline(pipeline); man=eye_pipeline_manifest(pipeline); decisions=set(pipeline.spec.decisions)
    man["decision_declared"]=[not d or d in decisions for d in man.decision]; man["has_description"]=man.description.astype(str).str.len()>0
    if _is(x,"eye_pipeline_run"): man=man.merge(pipeline_step_status(x),on="step",how="left",suffixes=("","_run"),sort=False)
    bad=man.loc[man.decision.astype(str).str.len().gt(0)&~man.decision_declared,"decision"].drop_duplicates().tolist()
    return _result("eye_pipeline_audit",table=man,undeclared_decisions=bad,valid=bool(man.decision_declared.all()),pipeline_hash=pipeline.hash)


def pipeline_step_status(x: Any) -> pd.DataFrame:
    if not _is(x,"eye_pipeline_run"): raise EyeProcessValidationError("x must be an eye_pipeline_run.")
    out=pd.DataFrame({"step":list(x.pipeline.steps)}); rec=x.records
    if len(rec): out=out.merge(rec,on="step",how="left",sort=False)
    if "status" not in out: out["status"]="not_run"
    else: out["status"]=out.status.fillna("not_run")
    return out


def pipeline_result(x: Any, step: str) -> Any:
    if not _is(x,"eye_pipeline_run"): raise EyeProcessValidationError("x must be an eye_pipeline_run.")
    if str(step) not in x.outputs: raise EyeProcessValidationError(f"No successful output is available for step '{step}'.")
    return x.outputs[str(step)]


def pipeline_failures(x: Any) -> pd.DataFrame:
    s=pipeline_step_status(x); return s[s.status.isin(["error","optional_error"])].reset_index(drop=True)


def write_eye_pipeline_report(x: Any, path: str | Path) -> str:
    pipeline=x.pipeline if _is(x,"eye_pipeline_run") else x; audit=audit_eye_pipeline(x)
    lines=[f"# eyeprocess pipeline report: {pipeline.name}","",f"Pipeline hash: `{pipeline.hash}`",f"Analysis-spec hash: `{pipeline.spec.hash}`","","## Governance","",
           f"- Steps: {len(audit.table)}",f"- All step-linked decisions declared: {audit.valid}","- The report records execution and provenance; it does not certify substantive model adequacy."]
    if _is(x,"eye_pipeline_run"):
        s=pipeline_step_status(x); lines += ["","## Execution","",f"- Completed: {x.completed}",f"- Successful steps: {(s.status=='success').sum()}",f"- Error/optional-error steps: {s.status.isin(['error','optional_error']).sum()}"]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8"); return str(Path(path))


def export_eye_pipeline(x: Any, path: str | Path) -> str:
    p=x.pipeline if _is(x,"eye_pipeline_run") else x; tab=eye_pipeline_manifest(p)
    if _is(x,"eye_pipeline_run"): tab=tab.merge(pipeline_step_status(x),on="step",how="left",sort=False)
    tab.to_csv(path,index=False); return str(path)


def eye_pipeline_dot(x: Any) -> str:
    g=eye_pipeline_graph(x); nodes=[f'  "{s}";' for s in g.vertices.step]; edges=[f'  "{r["from"]}" -> "{r["to"]}";' for _,r in g.edges.iterrows()]
    return "\n".join(["digraph eyeprocess_pipeline {",*nodes,*edges,"}"])


def eye_pipeline_mermaid(x: Any) -> str:
    g=eye_pipeline_graph(x); lines=[f"  {r['from']} --> {r['to']}" for _,r in g.edges.iterrows()] if len(g.edges) else [f"  {s}" for s in g.vertices.step]
    return "\n".join(["flowchart TD",*lines])


def eye_targets_manifest(x: Any) -> pd.DataFrame:
    man=eye_pipeline_manifest(x); return pd.DataFrame({"target":man.step,"dependencies":man.requires,"function_hash":man.function_hash})


def write_eye_targets_template(x: Any, path: str | Path = "_targets.R") -> str:
    man=eye_targets_manifest(x); lines=["# Generated by eyeprocess::write_eye_targets_template()","# Review every command before execution; substantive decisions are not inferred.","library(targets)","library(eyeprocess)","","list("]
    blocks=[]
    for _,r in man.iterrows():
        deps=[z for z in str(r.dependencies).split(";") if z]; note=", ".join(deps) if deps else "no upstream targets"
        blocks.append(f"  # {r.target} requires: {note}\n  tar_target({r.target}, stop(\"Replace with explicit eyeprocess step call\"))")
    lines += [",\n".join(blocks),")"]; Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8"); return str(path)


# ---------------------------------------------------------------------------
# R/071 API lifecycle

_ALLOWED_API_STATUS={"core","workflow","advanced","experimental","gated","compatibility","deprecated","internal-candidate","unreviewed"}

def _lifecycle_path() -> Path:
    return Path(resources.files("eyeprocesspy").joinpath("resources/extdata/api-lifecycle-registry-0.9.csv"))


def eye_api_lifecycle(registry: Any = None) -> pd.DataFrame:
    if registry is None: registry=pd.read_csv(_lifecycle_path(),dtype=str,keep_default_na=True)
    d=_df(registry,"registry"); _req(d,["name","status"],"registry")
    for c in ["canonical","replacement","since","notes"]:
        if c not in d: d[c]=np.nan
    d=d[["name","status","canonical","replacement","since","notes"]].copy(); d["name"]=d.name.astype(str); d["status"]=d.status.astype(str)
    if d.name.isna().any() or d.name.eq("").any(): raise EyeProcessValidationError("Lifecycle registry API names must be non-missing and non-empty.")
    if d.status.isna().any() or d.status.eq("").any(): raise EyeProcessValidationError("Lifecycle registry statuses must be non-missing and non-empty.")
    if d.name.duplicated().any(): raise EyeProcessValidationError("Lifecycle registry contains duplicate API names.")
    bad=sorted(set(d.status)-_ALLOWED_API_STATUS)
    if bad: raise EyeProcessValidationError("Unknown lifecycle status: "+", ".join(bad))
    d.attrs["eyeprocess_class"]="eye_api_lifecycle"; return d.reset_index(drop=True)


def _api_family(names: Sequence[str]) -> list[str]:
    import re
    out=[]
    for x in names:
        if re.match(r"^plot[._]",x): f="plot"
        elif re.match(r"^(audit|validate|validation)_",x): f="validation"
        elif re.match(r"^(fit|predict|score|update)_",x): f="model"
        elif re.match(r"^(read|write|export|import)_",x): f="io"
        elif re.match(r"^(process|eye)_.*(spec|manifest|registry|design|pipeline)",x): f="workflow"
        elif re.match(r"^(compare|summari[sz]e|summary|extract)_",x): f="summary"
        elif re.match(r"^(simulate|inject|stress|benchmark)_",x): f="simulation"
        else: f="utility"
        out.append(f)
    return out


def eye_api_inventory(package: Any = "eyeprocess", lifecycle: Any = None) -> pd.DataFrame:
    life=eye_api_lifecycle() if lifecycle is None else eye_api_lifecycle(lifecycle)
    if isinstance(package,str) and package=="eyeprocess":
        names=sorted(life.name.astype(str).tolist()); out=pd.DataFrame({"name":names,"kind":"function","family":_api_family(names)})
    else:
        mod=import_module("eyeprocesspy") if isinstance(package,str) and package=="eyeprocesspy" else (import_module(package) if isinstance(package,str) else package)
        names=sorted(set(getattr(mod,"__all__",[n for n in dir(mod) if not n.startswith("_")])))
        out=pd.DataFrame({"name":names,"kind":["function" if callable(getattr(mod,n,None)) else type(getattr(mod,n,None)).__name__ for n in names],"family":_api_family(names)})
    out=out.merge(life,on="name",how="left",sort=False); out["status"]=out.status.fillna("unreviewed")
    return out


def register_eye_api_status(registry: Any = None, name: Any = None, status: Any = None, canonical: Any = np.nan,
                            replacement: Any = np.nan, since: str = "0.9.0.9000", notes: Any = np.nan) -> pd.DataFrame:
    reg=eye_api_lifecycle() if registry is None else eye_api_lifecycle(registry)
    row=eye_api_lifecycle(pd.DataFrame([{"name":str(name),"status":str(status),"canonical":canonical,"replacement":replacement,"since":str(since),"notes":notes}]))
    return eye_api_lifecycle(pd.concat([reg[reg.name!=str(name)],row],ignore_index=True))


def eye_api_status(name: Any, registry: Any = None) -> pd.DataFrame:
    reg=eye_api_lifecycle() if registry is None else eye_api_lifecycle(registry); names=[name] if isinstance(name,str) else list(map(str,name)); idx=reg.set_index("name")
    rows=[]
    for n in names:
        if n in idx.index:
            r=idx.loc[n]; rows.append({"name":n,"status":r.status,"canonical":r.canonical,"replacement":r.replacement})
        else: rows.append({"name":n,"status":"unreviewed","canonical":np.nan,"replacement":np.nan})
    return pd.DataFrame(rows)


def eye_api_superseded(registry: Any = None) -> pd.DataFrame:
    reg=eye_api_lifecycle() if registry is None else eye_api_lifecycle(registry); repl=reg.replacement.notna()&reg.replacement.astype(str).ne("")
    return reg[reg.status.isin(["deprecated","compatibility"])|repl].reset_index(drop=True)


def canonical_eye_api(registry: Any = None) -> pd.DataFrame:
    reg=eye_api_lifecycle() if registry is None else eye_api_lifecycle(registry); can=reg.canonical.notna()&reg.canonical.astype(str).ne("")
    return reg[can|reg.status.isin(["core","workflow"])].reset_index(drop=True)


def api_surface_summary(inventory: Any) -> pd.DataFrame:
    d=_df(inventory,"inventory"); _req(d,["name","family","status"],"inventory")
    return d.groupby(["family","status"],dropna=False).size().rename("Freq").reset_index()


def api_family_map(inventory: Any) -> pd.DataFrame:
    if isinstance(inventory,(list,tuple,np.ndarray,pd.Series)) and not isinstance(inventory,pd.DataFrame): names=list(map(str,inventory)); return pd.DataFrame({"name":names,"family":_api_family(names)})
    d=_df(inventory,"inventory"); _req(d,["name"],"inventory"); return pd.DataFrame({"name":d.name.astype(str),"family":d.family if "family" in d else _api_family(d.name.astype(str))})


def audit_eye_api(inventory: Any = None, registry: Any = None) -> EyeResult:
    inv=eye_api_inventory() if inventory is None else _df(inventory,"inventory"); reg=eye_api_lifecycle() if registry is None else eye_api_lifecycle(registry); _req(inv,["name"],"inventory")
    status=eye_api_status(inv.name.tolist(),reg).rename(columns={"status":"status_registry","canonical":"canonical_registry","replacement":"replacement_registry"}); tab=inv.merge(status,on="name",how="left",sort=False)
    tab["status"]=tab.status_registry
    tab["canonical"]=tab.canonical_registry; tab["replacement"]=tab.replacement_registry
    exported=set(inv.name.astype(str)); repl=tab.replacement.notna()&tab.replacement.astype(str).ne(""); can=tab.canonical.notna()&tab.canonical.astype(str).ne("")
    invalid_repl=repl&~tab.replacement.astype(str).isin(exported); invalid_can=can&~tab.canonical.astype(str).isin(exported)
    tab["replacement_exists"]=~invalid_repl; tab["canonical_exists"]=~invalid_can; unreviewed=tab.loc[tab.status=="unreviewed","name"].tolist()
    return _result("eye_api_audit",table=tab,unreviewed=unreviewed,invalid_replacements=tab.loc[invalid_repl,"name"].tolist(),invalid_canonical=tab.loc[invalid_can,"name"].tolist(),
                   reviewed_fraction=float((tab.status!="unreviewed").mean()) if len(tab) else math.nan,valid=bool(not invalid_repl.any() and not invalid_can.any()))


def eye_api_recommendation(audit: Any) -> pd.DataFrame:
    if not _is(audit,"eye_api_audit"): raise EyeProcessValidationError("audit must be an eye_api_audit.")
    def rec(r: pd.Series)->str:
        if r.status=="unreviewed": return "classify"
        if not bool(r.replacement_exists): return "repair_replacement"
        if not bool(r.canonical_exists): return "repair_canonical"
        return "retain"
    return pd.DataFrame({"name":audit.table.name,"recommendation":audit.table.apply(rec,axis=1)})


def write_api_lifecycle_registry(registry: Any, path: str | Path) -> str:
    eye_api_lifecycle(registry).to_csv(path,index=False); return str(path)


def read_api_lifecycle_registry(path: str | Path) -> pd.DataFrame:
    return eye_api_lifecycle(pd.read_csv(path,dtype=str,keep_default_na=True))


def api_lifecycle_diff(old: Any, new: Any) -> pd.DataFrame:
    a=eye_api_lifecycle(old).set_index("name"); b=eye_api_lifecycle(new).set_index("name"); rows=[]
    for nm in sorted(set(a.index)|set(b.index)):
        av=a.loc[nm,"status"] if nm in a.index else np.nan; bv=b.loc[nm,"status"] if nm in b.index else np.nan
        rows.append({"name":nm,"old_status":av,"new_status":bv,"changed":not ((pd.isna(av) and pd.isna(bv)) or av==bv)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R/072 sensitivity/multiverse


def process_sensitivity_grid(*args: Any, label: str = "process_sensitivity", max_specifications: float = 100000, **kwargs: Any) -> pd.DataFrame:
    if args: raise EyeProcessValidationError("Supply one or more uniquely named analysis-decision vectors.")
    if not kwargs or len(kwargs)!=len(set(kwargs)): raise EyeProcessValidationError("Supply one or more uniquely named analysis-decision vectors.")
    opts={k:([v] if isinstance(v,(str,bytes)) or np.isscalar(v) else list(v)) for k,v in kwargs.items()}
    if any(len(v)==0 for v in opts.values()): raise EyeProcessValidationError("Every analysis-decision vector must contain at least one option.")
    if float(max_specifications)<0: raise EyeProcessValidationError("max_specifications must be a non-negative scalar or Inf.")
    rows=[dict(zip(opts,z)) for z in itertools.product(*opts.values())]
    if len(rows)>max_specifications: raise EyeProcessValidationError(f"Sensitivity grid expands to {len(rows)} specifications; increase max_specifications deliberately if intended.")
    g=pd.DataFrame(rows); g.insert(0,"specification_id",[f"S{i:05d}" for i in range(1,len(g)+1)]); g.attrs.update(label=str(label),eyeprocess_class="eye_process_sensitivity_grid"); return g


def _default_sensitivity_extract(fit: Any, specification: Any) -> pd.DataFrame:
    if np.isscalar(fit) and isinstance(fit,(int,float,np.number)): return pd.DataFrame({"effect":[float(fit)]})
    if isinstance(fit,pd.DataFrame): return fit.copy()
    if isinstance(fit,Mapping):
        keep={k:v for k,v in fit.items() if np.isscalar(v) or isinstance(v,str)}
        if keep: return pd.DataFrame([keep])
    raise EyeProcessValidationError("Provide extract_fun for analysis results that are not scalar/list/data.frame summaries.")


def run_process_sensitivity(data: Any, grid: Any, analysis_fun: Callable, extract_fun: Callable = _default_sensitivity_extract, progress: bool = False) -> EyeResult:
    if not callable(analysis_fun): raise EyeProcessValidationError("analysis_fun must be a function.")
    if not callable(extract_fun): raise EyeProcessValidationError("extract_fun must be a function.")
    g=_df(grid,"grid"); _req(g,["specification_id"],"grid"); rows=[]; fails=[]; warns=[]; decision_cols=[c for c in g if c!="specification_id"]
    for i,(_,specrow) in enumerate(g.iterrows(),start=1):
        spec=specrow.to_frame().T
        if progress: print(f"sensitivity {specrow.specification_id} ({i}/{len(g)})")
        cap=_capture(analysis_fun,data,spec)
        if cap["warnings"]: warns.append(pd.DataFrame({"specification_id":[specrow.specification_id],"stage":["analysis"],"warning":[" | ".join(cap["warnings"])]}))
        if cap["error"] is not None: fails.append(pd.DataFrame({"specification_id":[specrow.specification_id],"error":[cap["error"]]})); continue
        ext=_capture(extract_fun,cap["value"],spec)
        if ext["warnings"]: warns.append(pd.DataFrame({"specification_id":[specrow.specification_id],"stage":["extract"],"warning":[" | ".join(ext["warnings"])]}))
        if ext["error"] is not None: fails.append(pd.DataFrame({"specification_id":[specrow.specification_id],"error":[ext["error"]]})); continue
        tab=_df(ext["value"])
        if not len(tab): continue
        tab["specification_id"]=specrow.specification_id
        for nm in decision_cols: tab[nm]=specrow[nm]
        if cap["warnings"]: tab["warnings"]=" | ".join(cap["warnings"])
        rows.append(tab)
    return _result("eye_process_sensitivity",grid=g,results=_rbind_fill(rows),failures=_rbind_fill(fails),warnings=_rbind_fill(warns),grid_hash=_hash_object(g),created_at=_now(),status="defensible_multiverse",
                   caveat="Sensitivity results describe the supplied defensible specification set; they do not validate specifications omitted from that set.")


def sensitivity_sign_stability(x: Any, effect: str = "effect") -> float:
    d=x.results; _req(d,[effect],"x$results"); e=_num(d[effect]); e=e[np.isfinite(e)&(e!=0)]; return float(max(np.mean(e>0),np.mean(e<0))) if e.size else math.nan


def sensitivity_significance_stability(x: Any, p_value: str = "p_value", alpha: float = .05) -> float:
    if alpha<0 or alpha>1: raise EyeProcessValidationError("alpha must lie in [0, 1].")
    d=x.results; _req(d,[p_value],"x$results"); p=_num(d[p_value]); p=p[np.isfinite(p)]; return float(max(np.mean(p<alpha),np.mean(p>=alpha))) if p.size else math.nan


def sensitivity_threshold_stability(x: Any, effect: str = "effect", threshold: float = 0, direction: str = "above") -> float:
    if direction not in {"above","below","absolute"}: raise EyeProcessValidationError("direction must be one of above, below, absolute.")
    if not np.isfinite(threshold): raise EyeProcessValidationError("threshold must be a finite scalar.")
    d=x.results; _req(d,[effect],"x$results"); e=_num(d[effect]); e=e[np.isfinite(e)]
    if not e.size: return math.nan
    dec=e>=threshold if direction=="above" else (e<=threshold if direction=="below" else np.abs(e)>=abs(threshold)); return float(max(np.mean(dec),np.mean(~dec)))


def summarise_process_sensitivity(x: Any, effect: str = "effect", p_value: str | None = None, threshold: float = 0, alpha: float = .05) -> pd.DataFrame:
    if not _is(x,"eye_process_sensitivity"): raise EyeProcessValidationError("x must be an eye_process_sensitivity.")
    d=x.results
    if not len(d): return pd.DataFrame()
    _req(d,[effect],"x$results"); e=_num(d[effect]); ef=e[np.isfinite(e)]
    row={"specifications":len(d),"failures":len(x.failures),"median_effect":float(np.median(ef)) if ef.size else math.nan,"mean_effect":float(np.mean(ef)) if ef.size else math.nan,
         "min_effect":float(np.min(ef)) if ef.size else math.nan,"max_effect":float(np.max(ef)) if ef.size else math.nan,"sign_stability":sensitivity_sign_stability(x,effect),
         "threshold_stability":sensitivity_threshold_stability(x,effect,threshold)}
    if p_value is not None: row["significance_stability"]=sensitivity_significance_stability(x,p_value,alpha)
    return pd.DataFrame([row])


def decision_stability(x: Any, effect: str = "effect", p_value: str | None = None, alpha: float = .05, threshold: float = 0) -> EyeResult:
    s=summarise_process_sensitivity(x,effect,p_value,threshold,alpha)
    if not len(s): return _result("eye_decision_stability",summary=s,stable_sign=None,stable_threshold=None,stable_significance=None,thresholds={"sign":.9,"threshold":.9,"significance":.9},caveat="No successful specifications were available; stability is undefined.")
    r=s.iloc[0]; return _result("eye_decision_stability",summary=s,stable_sign=bool(r.sign_stability>=.9),stable_threshold=bool(r.threshold_stability>=.9),stable_significance=None if p_value is None else bool(r.significance_stability>=.9),thresholds={"sign":.9,"threshold":.9,"significance":.9},caveat="Stability thresholds are reporting conventions, not universal validity cutoffs.")


def specification_curve_data(x: Any, effect: str = "effect", lower: str | None = None, upper: str | None = None) -> pd.DataFrame:
    d=x.results.copy(); _req(d,[effect],"x$results"); d[".effect"]=_num(d[effect]); d=d.sort_values(".effect").reset_index(drop=True); d["curve_order"]=np.arange(1,len(d)+1)
    if lower is not None and lower in d: d[".lower"]=_num(d[lower])
    if upper is not None and upper in d: d[".upper"]=_num(d[upper])
    return d


def specification_coverage(x: Any) -> float:
    if not _is(x,"eye_process_sensitivity"): raise EyeProcessValidationError("x must be an eye_process_sensitivity.")
    return math.nan if not len(x.grid) else len(set(x.results.get("specification_id",[])))/len(x.grid)


def sensitivity_decision_leverage(x: Any, effect: str = "effect") -> pd.DataFrame:
    d=x.results; _req(d,["specification_id",effect],"x$results"); decision=[c for c in x.grid.columns if c!="specification_id" and c in d]; rows=[]
    for nm in decision:
        means=pd.to_numeric(d[effect],errors="coerce").groupby(d[nm]).mean().dropna().to_numpy(float); rows.append({"decision":nm,"levels":len(means),"effect_range":float(np.ptp(means)) if len(means)>1 else (0.0 if len(means)==1 else math.nan),"effect_sd_across_levels":float(np.std(means,ddof=1)) if len(means)>1 else (0.0 if len(means)==1 else math.nan)})
    return pd.DataFrame(rows).sort_values("effect_range",ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["decision","levels","effect_range","effect_sd_across_levels"])


def sensitivity_fragility_index(x: Any, effect: str = "effect", threshold: float = 0) -> float:
    d=x.results; _req(d,[effect],"x$results"); e=_num(d[effect]); e=e[np.isfinite(e)]
    if not e.size: return math.nan
    majority=np.mean(e>=threshold)>=.5; return float(np.mean((e>=threshold)!=majority))


def sensitivity_rank_stability(x: Any, id: str | None = None, rank: str | None = None, specification: str | None = None) -> float:
    if isinstance(x,(list,tuple)) and not isinstance(x,pd.DataFrame):
        if len(x)<2: return math.nan
        if len({len(z) for z in x})!=1: raise EyeProcessValidationError("Ranking vectors supplied as a list must have equal length.")
        vals=[]
        for a,b in itertools.combinations(x,2):
            r=spearmanr(pd.Series(a).rank().to_numpy(float),pd.Series(b).rank().to_numpy(float),nan_policy="omit").statistic
            if np.isfinite(r): vals.append(float(r))
        return float(np.mean(vals)) if vals else math.nan
    d=_df(x); _req(d,[id,rank,specification],"x"); groups=[z for _,z in d.groupby(specification)]; common=set(map(str,groups[0][id]))
    for z in groups[1:]: common &= set(map(str,z[id]))
    common=sorted(common)
    if len(common)<2 or len(groups)<2: return math.nan
    vec=[]
    for z in groups:
        zz=z.assign(_id=z[id].astype(str)).set_index("_id").loc[common]; vec.append(_num(zz[rank]))
    return sensitivity_rank_stability(vec)


def sensitivity_branch_fingerprint(specification: Any) -> str: return _hash_object(specification)


def sensitivity_multiverse_manifest(x: Any) -> pd.DataFrame:
    if not _is(x,"eye_process_sensitivity"): raise EyeProcessValidationError("x must be an eye_process_sensitivity.")
    g=x.grid; evaluated=set(x.results.get("specification_id",[])); failed=set(x.failures.get("specification_id",[]))
    return pd.DataFrame({"specification_id":g.specification_id,"branch_hash":[_hash_object(g.iloc[[i]]) for i in range(len(g))],"evaluated":g.specification_id.isin(evaluated),"failed":g.specification_id.isin(failed)})


def _compare_methods(data: Any, methods: Any, analysis_fun: Callable, extract_fun: Callable, dimension: str) -> EyeResult:
    if isinstance(methods,Mapping): md=dict(methods)
    else: raise EyeProcessValidationError("methods must be a named list/vector.")
    g=process_sensitivity_grid(method=list(md),label=f"{dimension}_comparison")
    def wrapped(d:Any,spec:pd.DataFrame)->Any: return analysis_fun(d,md[str(spec.iloc[0].method)],spec)
    return run_process_sensitivity(data,g,wrapped,extract_fun)


def compare_aoi_methods(data: Any, methods: Any, analysis_fun: Callable, extract_fun: Callable = _default_sensitivity_extract) -> EyeResult: return _compare_methods(data,methods,analysis_fun,extract_fun,"aoi")
def compare_fixation_methods(data: Any, methods: Any, analysis_fun: Callable, extract_fun: Callable = _default_sensitivity_extract) -> EyeResult: return _compare_methods(data,methods,analysis_fun,extract_fun,"fixation")
def compare_pupil_preprocessing(data: Any, methods: Any, analysis_fun: Callable, extract_fun: Callable = _default_sensitivity_extract) -> EyeResult: return _compare_methods(data,methods,analysis_fun,extract_fun,"pupil_preprocessing")
def compare_process_models(data: Any, methods: Any, analysis_fun: Callable, extract_fun: Callable = _default_sensitivity_extract) -> EyeResult: return _compare_methods(data,methods,analysis_fun,extract_fun,"process_model")


# ---------------------------------------------------------------------------
# R/073 decision manifests


def eye_decision_manifest(sampling: Any = None, validity: Any = None, fixation: Any = None, pupil: Any = None, aoi: Any = None,
                          model: Any = None, sensitivity: Any = None, exclusions: Any = None, provenance: Any = None,
                          notes: Any = None, **kwargs: Any) -> EyeResult:
    domains={"sampling":{} if sampling is None else sampling,"validity":{} if validity is None else validity,"fixation":{} if fixation is None else fixation,
             "pupil":{} if pupil is None else pupil,"aoi":{} if aoi is None else aoi,"model":{} if model is None else model,"sensitivity":{} if sensitivity is None else sensitivity,
             "exclusions":{} if exclusions is None else exclusions,"provenance":{} if provenance is None else provenance,**kwargs}
    x=_result("eye_decision_manifest",domains=domains,notes=notes,created_at=_now(),schema_version="0.9.0.9000",status="research_decision_manifest",caveat="The manifest documents decisions; it does not endorse any decision as universally appropriate.")
    x["hash"]=decision_manifest_hash(x); return x


def validate_decision_manifest(x: Any, required_domains: Any = ("sampling","validity","fixation","pupil","aoi","model","sensitivity","exclusions"), require_nonempty: bool = False) -> bool:
    if not _is(x,"eye_decision_manifest"): raise EyeProcessValidationError("x must be an eye_decision_manifest.")
    req=[] if required_domains is None else list(required_domains); miss=[d for d in req if d not in x.domains]
    if miss: raise EyeProcessValidationError("Manifest is missing required domain(s): "+", ".join(miss))
    if require_nonempty:
        empty=[d for d in req if len(x.domains[d])==0]
        if empty: raise EyeProcessValidationError("Required manifest domain(s) are empty: "+", ".join(empty))
    return True


def _flatten_manifest(x: Any, prefix: str = "") -> list[dict[str,Any]]:
    if isinstance(x,Mapping):
        if not x: return [{"path":prefix,"value":np.nan}]
        rows=[]
        for k,v in x.items(): rows += _flatten_manifest(v,f"{prefix}.{k}" if prefix else str(k))
        return rows
    if isinstance(x,(list,tuple)) and not isinstance(x,(str,bytes)):
        if not x: return [{"path":prefix,"value":np.nan}]
        rows=[]
        for i,v in enumerate(x,1): rows += _flatten_manifest(v,f"{prefix}.[[{i}]]" if prefix else f"[[{i}]]")
        return rows
    if x is None: value=np.nan
    elif isinstance(x,(list,tuple,np.ndarray,pd.Series)): value=";".join(map(str,x))
    else: value=str(x)
    return [{"path":prefix,"value":value}]


def decision_manifest_table(x: Any) -> pd.DataFrame:
    validate_decision_manifest(x,required_domains=()); d=pd.DataFrame(_flatten_manifest(x.domains)); d["manifest_hash"]=x.hash; return d


def decision_manifest_hash(x: Any) -> str:
    payload={"domains":x.domains,"notes":x.notes,"schema_version":x.schema_version} if _is(x,"eye_decision_manifest") else x; return _hash_object(payload)


def lock_decision_manifest(x: Any, label: str = "analysis_decisions") -> EyeResult:
    validate_decision_manifest(x,required_domains=()); return _result("eye_decision_manifest_lock",manifest=x,manifest_hash=decision_manifest_hash(x),label=str(label),locked_at=_now(),status="locked_decision_manifest")


def verify_decision_manifest_lock(x: Any) -> bool:
    if not _is(x,"eye_decision_manifest_lock"): raise EyeProcessValidationError("x must be an eye_decision_manifest_lock.")
    return x.manifest_hash==decision_manifest_hash(x.manifest)


def compare_decision_manifests(old: Any, new: Any) -> pd.DataFrame:
    a=decision_manifest_table(old).set_index("path"); b=decision_manifest_table(new).set_index("path"); rows=[]
    for k in sorted(set(a.index)|set(b.index)):
        av=a.loc[k,"value"] if k in a.index else np.nan; bv=b.loc[k,"value"] if k in b.index else np.nan; changed=not ((pd.isna(av) and pd.isna(bv)) or av==bv)
        rows.append({"path":k,"old":av,"new":bv,"changed":changed})
    out=pd.DataFrame(rows); out.attrs["eyeprocess_class"]="eye_decision_manifest_diff"; return out


def decision_manifest_diff(old: Any, new: Any) -> pd.DataFrame: return compare_decision_manifests(old,new)


def _manifest_payload(x: EyeResult) -> dict[str,Any]:
    return {"domains":x.domains,"notes":x.notes,"created_at":x.created_at,"schema_version":x.schema_version,"status":x.status,"caveat":x.caveat,"hash":x.hash}


def write_decision_manifest(x: Any, path: str | Path, format: str = "rds") -> str:
    validate_decision_manifest(x,required_domains=()); fmt=str(format).lower()
    if fmt not in {"rds","dput","json"}: raise EyeProcessValidationError("format must be rds, dput, or json.")
    if fmt=="rds":
        with open(path,"wb") as fh: pickle.dump(x,fh)
    else:
        Path(path).write_text(json.dumps(_manifest_payload(x),indent=2,default=str),encoding="utf-8")
    return str(path)


def read_decision_manifest(path: str | Path, format: str | None = None) -> EyeResult:
    fmt=(Path(path).suffix.lower().lstrip(".") if format is None else str(format).lower()); fmt="dput" if fmt not in {"rds","json"} else fmt
    if fmt=="rds":
        with open(path,"rb") as fh: x=pickle.load(fh)
    else:
        raw=json.loads(Path(path).read_text(encoding="utf-8")); x=_result("eye_decision_manifest",**raw)
    validate_decision_manifest(x,required_domains=())
    if not getattr(x,"hash",None) or str(x.hash)!=decision_manifest_hash(x): raise EyeProcessValidationError("Decision manifest hash is missing or does not match the imported decision content.")
    return x


def audit_decision_provenance(x: Any, required_domains: Any = ("sampling","validity","fixation","pupil","aoi","model","sensitivity","exclusions"),
                              required_provenance: Any = ("data_source","software_version","analysis_commit")) -> EyeResult:
    validate_decision_manifest(x,required_domains=()); req=list(required_domains); miss=[d for d in req if d not in x.domains]; empty=[d for d in req if d in x.domains and len(x.domains[d])==0]
    prov=x.domains.get("provenance",{}) or {}; mp=[k for k in required_provenance if k not in prov or prov[k] is None or not str(prov[k])]
    return _result("eye_decision_provenance_audit",missing_domains=miss,empty_domains=empty,missing_provenance=mp,complete=not(miss or empty or mp),manifest_hash=x.hash)


def outcome_blind_snapshot(data: Any, outcome: Any, id: Any = None) -> EyeResult:
    d=_df(data); outcomes=[outcome] if isinstance(outcome,str) else list(map(str,outcome)); _req(d,outcomes,"data")
    if id is not None: _req(d,[id] if isinstance(id,str) else list(id),"data")
    blinded=d.drop(columns=outcomes)
    return _result("eye_outcome_blind_snapshot",data=blinded,removed_outcomes=outcomes,id=id,source_columns=list(d),blinded_columns=list(blinded),blinded_hash=_hash_object(blinded),created_at=_now(),caveat="Outcome blinding limits direct access through this snapshot only; it cannot guarantee analysts were otherwise unaware of outcomes.")


def verify_outcome_blind_snapshot(x: Any) -> bool:
    if not _is(x,"eye_outcome_blind_snapshot"): raise EyeProcessValidationError("x must be an eye_outcome_blind_snapshot.")
    return x.blinded_hash==_hash_object(x.data) and not any(c in x.data for c in x.removed_outcomes)


def analysis_decision_entropy(*args: Any, base: float = 2, **kwargs: Any) -> pd.DataFrame:
    if args or not kwargs: raise EyeProcessValidationError("Supply uniquely named decision option vectors.")
    if not np.isfinite(base) or base<=0 or base==1: raise EyeProcessValidationError("base must be finite, positive, and not equal to 1.")
    counts={k:len(set([v] if isinstance(v,(str,bytes)) or np.isscalar(v) else list(v))) for k,v in kwargs.items()}
    if any(v==0 for v in counts.values()): raise EyeProcessValidationError("Decision option vectors cannot be empty.")
    out=pd.DataFrame({"decision":list(counts),"options":list(counts.values()),"max_entropy":[math.log(v,base) for v in counts.values()]}); out.attrs["joint_specifications"]=math.prod(counts.values()); out.attrs["joint_max_entropy"]=float(out.max_entropy.sum()); return out


def decision_space_coverage(grid: Any, evaluated: Any) -> pd.DataFrame:
    g=_df(grid); _req(g,["specification_id"],"grid"); ids=set(evaluated.results.specification_id.astype(str)) if _is(evaluated,"eye_process_sensitivity") else set(map(str,evaluated)); covered=g.specification_id.astype(str).isin(ids)
    return pd.DataFrame({"planned":[len(g)],"evaluated":[int(covered.sum())],"coverage":[float(covered.mean()) if len(covered) else math.nan]})


__all__ = [n for n in globals() if not n.startswith("_") and n not in {
    "annotations","datetime","timezone","import_module","resources","inspect","itertools","json","math","Path","pickle","warnings","Any","Callable","Mapping","Sequence","np","pd","linregress","spearmanr","EyeProcessValidationError","EyeResult"
}]
