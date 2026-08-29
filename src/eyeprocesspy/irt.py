"""Transparent item-response-theory parity layer for eyeprocess 0.11.1.

This module ports the dependency-light 0.9 IRT mathematics, diagnostics, scoring,
linking, process-data contracts, validation, multidimensional/CDM helpers, and
model-governance utilities. External R-engine adapters remain explicit gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import importlib.util
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import chi2, norm

from .exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError

_EPS = np.finfo(float).eps


class EyeResult(dict):
    """Small R-list-like result object with stable class metadata."""

    def __init__(self, *args: Any, eyeprocess_class: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eyeprocess_class = eyeprocess_class

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _tag(df: pd.DataFrame, cls: str, **attrs: Any) -> pd.DataFrame:
    out = df.copy()
    out.attrs["eyeprocess_class"] = cls
    out.attrs.update(attrs)
    return out


def _as_df(x: Any, name: str = "object") -> pd.DataFrame:
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


def _req_cols(df: pd.DataFrame, cols: Sequence[str], name: str = "data") -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise EyeProcessValidationError(f"{name} is missing required columns: {', '.join(miss)}")


def _finite_mean(x: Any) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else math.nan


def _finite_sd(x: Any) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if a.size > 1 else math.nan


def _safe_quantile(x: Any, probs: Any) -> Any:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    p = np.asarray(probs, dtype=float)
    if not a.size:
        z = np.full(p.shape or (), np.nan)
    else:
        z = np.quantile(a, p)
    if np.ndim(probs) == 0:
        return float(z)
    return z


def _stable_hash(x: Any) -> str:
    def conv(v: Any) -> Any:
        if isinstance(v, pd.DataFrame):
            return {"columns": list(v.columns), "records": v.where(pd.notna(v), None).to_dict("records")}
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, Mapping):
            return {str(k): conv(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(z) for z in v]
        return v

    payload = json.dumps(conv(x), sort_keys=True, default=str, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _theta(theta: Any) -> np.ndarray:
    a = np.atleast_1d(np.asarray(theta, dtype=float))
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise EyeProcessValidationError("theta must contain finite numeric values.")
    return a


def _item_pars(items: Any) -> pd.DataFrame:
    df = _as_df(items, "items")
    _req_cols(df, ["item_id", "a", "b"], "items")
    if df["item_id"].isna().any() or df["item_id"].duplicated().any():
        raise EyeProcessValidationError("item_id must be unique and non-missing.")
    for c in ("a", "b"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if (~np.isfinite(df["a"])).any() or (df["a"] <= 0).any() or (~np.isfinite(df["b"])).any():
        raise EyeProcessValidationError("a must be positive finite and b finite.")
    if "c" not in df:
        df["c"] = 0.0
    if "d" not in df:
        df["d"] = 1.0
    df["c"] = pd.to_numeric(df["c"], errors="coerce")
    df["d"] = pd.to_numeric(df["d"], errors="coerce")
    invalid = (~np.isfinite(df["c"])) | (~np.isfinite(df["d"])) | (df["c"] < 0) | (df["c"] >= 1) | (df["d"] <= 0) | (df["d"] > 1) | (df["c"] >= df["d"])
    if invalid.any():
        raise EyeProcessValidationError("c and d must satisfy 0 <= c < d <= 1.")
    df["item_id"] = df["item_id"].astype(str)
    return df


def _binary_matrix(x: Any, name: str = "responses", allow_na: bool = True) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise EyeProcessValidationError(f"{name} must be a two-dimensional matrix.")
    finite = a[~np.isnan(a)]
    if np.any(~np.isin(finite, [0.0, 1.0])):
        suffix = ", and NA" if allow_na else ""
        raise EyeProcessValidationError(f"{name} must contain only 0, 1{suffix}.")
    if not allow_na and np.isnan(a).any():
        raise EyeProcessValidationError(f"{name} cannot contain NA.")
    return a


def _prob_matrix(x: Any, shape: tuple[int, int], name: str = "probabilities") -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.shape != shape:
        raise EyeProcessValidationError(f"{name} must have the same dimensions as responses.")
    z = a[~np.isnan(a)]
    if np.any(~np.isfinite(z)) or np.any((z <= 0) | (z >= 1)):
        raise EyeProcessValidationError(f"{name} must contain probabilities strictly between 0 and 1 (or NA where responses are missing).")
    return a


def _scalar(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a.item()):
        raise EyeProcessValidationError(f"{name} must be a finite scalar.")
    v = float(a.item())
    if positive and v <= 0:
        raise EyeProcessValidationError(f"{name} must be positive.")
    if nonnegative and v < 0:
        raise EyeProcessValidationError(f"{name} must be non-negative.")
    return v


# ---------------------------------------------------------------------------
# 083: foundations and information
# ---------------------------------------------------------------------------

def eyeprocess_irt_model_spec(
    family: str = "rasch",
    dimensions: int = 1,
    identification: str = "theta_standard",
    engine: str = "native_math",
    process_channels: Sequence[str] = (),
    status: str = "reference",
    notes: str | None = None,
) -> EyeResult:
    families = {"rasch", "2pl", "3pl", "4pl", "grm", "gpcm", "nominal", "multidimensional", "testlet", "latent_regression", "cdm", "joint_rt"}
    identifications = {"theta_standard", "item_sum_zero", "anchor"}
    engines = {"native_math", "mirt", "TAM", "GDINA", "LNIRT", "eRm", "custom"}
    statuses = {"reference", "experimental", "gated"}
    if family not in families or identification not in identifications or engine not in engines or status not in statuses:
        raise EyeProcessValidationError("invalid IRT model specification choice.")
    dimensions = int(dimensions)
    if dimensions < 1:
        raise EyeProcessValidationError("dimensions must be a positive integer.")
    channels = list(dict.fromkeys(str(x) for x in process_channels))
    if notes is not None and not isinstance(notes, str):
        raise EyeProcessValidationError("notes must be NULL/None or a scalar string.")
    return _result("eyeprocess_irt_model_spec", family=family, dimensions=dimensions, identification=identification, engine=engine, process_channels=channels, status=status, notes=notes)


def validate_eyeprocess_irt_model_spec(x: Any) -> bool:
    if not isinstance(x, Mapping) or getattr(x, "eyeprocess_class", None) != "eyeprocess_irt_model_spec":
        raise EyeProcessValidationError("x must inherit from eyeprocess_irt_model_spec.")
    required = {"family", "dimensions", "identification", "engine", "process_channels", "status"}
    miss = required - set(x)
    if miss:
        raise EyeProcessValidationError(f"IRT specification is missing fields: {', '.join(sorted(miss))}")
    return True


def eyeprocess_irt_identification_audit(spec: Any, constraints: Mapping[str, Any] | None = None, n_items: int | None = None, n_persons: int | None = None) -> EyeResult:
    validate_eyeprocess_irt_model_spec(spec)
    constraints = {} if constraints is None else dict(constraints)
    fixed_theta_mean = constraints.get("theta_mean_fixed") is True
    fixed_theta_sd = constraints.get("theta_sd_fixed") is True
    sum_zero_items = constraints.get("item_difficulty_sum_zero") is True
    anchors = list(dict.fromkeys(str(x) for x in constraints.get("anchor_items", [])))
    location_identified = fixed_theta_mean or sum_zero_items or bool(anchors)
    scale_identified = fixed_theta_sd or len(anchors) >= 2 or spec["family"] == "rasch"
    if spec["identification"] == "theta_standard":
        location_identified = location_identified or fixed_theta_mean
        scale_identified = scale_identified or fixed_theta_sd
    warnings: list[str] = []
    if not location_identified:
        warnings.append("No explicit location constraint was supplied.")
    if not scale_identified:
        warnings.append("No explicit scale constraint was supplied.")
    if n_items is not None:
        n_items = int(n_items)
        if n_items < 0:
            raise EyeProcessValidationError("n_items must be NULL/None or a finite non-negative scalar.")
        if n_items < 3 * int(spec["dimensions"]):
            warnings.append("Few items relative to declared dimensionality; identification may be weak.")
    if n_persons is not None and int(n_persons) < 0:
        raise EyeProcessValidationError("n_persons must be NULL/None or a finite non-negative scalar.")
    return _result("eye_irt_identification_audit", spec=spec, location_identified=location_identified, scale_identified=scale_identified, anchors=anchors, warnings=warnings, valid=location_identified and scale_identified, n_items=n_items, n_persons=n_persons)


def eyeprocess_irt_sparse_design_audit(data: Any, person: str, item: str, response: str | None = None, min_person_items: int = 3, min_item_persons: int = 10) -> EyeResult:
    df = _as_df(data, "data")
    cols = [person, item] + ([response] if response else [])
    _req_cols(df, cols, "data")
    if not person or not item or (response is not None and not response):
        raise EyeProcessValidationError("person and item must be non-empty scalar column names.")
    min_person_items = int(min_person_items); min_item_persons = int(min_item_persons)
    if min_person_items < 1 or min_item_persons < 1:
        raise EyeProcessValidationError("minimum counts must be positive scalar integers.")
    keep = np.ones(len(df), dtype=bool) if response is None else df[response].notna().to_numpy()
    d = df.loc[keep, [person, item]]
    pc = d[person].astype(str).value_counts(sort=False)
    ic = d[item].astype(str).value_counts(sort=False)
    possible = df[person].nunique(dropna=False) * df[item].nunique(dropna=False)
    return _result(
        "eye_irt_sparse_design_audit",
        n_persons=int(len(pc)), n_items=int(len(ic)), n_observed=int(len(d)), density=float(len(d) / possible) if possible else math.nan,
        person_counts=pd.DataFrame({"person_id": pc.index.astype(str), "n_items": pc.to_numpy(dtype=int)}),
        item_counts=pd.DataFrame({"item_id": ic.index.astype(str), "n_persons": ic.to_numpy(dtype=int)}),
        sparse_persons=pc.index[pc < min_person_items].astype(str).tolist(), sparse_items=ic.index[ic < min_item_persons].astype(str).tolist(),
        min_person_items=min_person_items, min_item_persons=min_item_persons,
    )


def eyeprocess_irt_2pl_probability(theta: Any, a: float = 1, b: float = 0, D: float = 1) -> np.ndarray:
    th = _theta(theta); a = _scalar(a, "a", positive=True); b = _scalar(b, "b"); D = _scalar(D, "D", positive=True)
    return expit(D * a * (th - b))


def eyeprocess_irt_3pl_probability(theta: Any, a: float = 1, b: float = 0, c: float = 0.2, D: float = 1) -> np.ndarray:
    c = _scalar(c, "c")
    if c < 0 or c >= 1:
        raise EyeProcessValidationError("c must lie in [0,1).")
    return c + (1 - c) * eyeprocess_irt_2pl_probability(theta, a=a, b=b, D=D)


def eyeprocess_irt_4pl_probability(theta: Any, a: float = 1, b: float = 0, c: float = 0, d: float = 1, D: float = 1) -> np.ndarray:
    c = _scalar(c, "c"); d = _scalar(d, "d")
    if c < 0 or d > 1 or c >= d:
        raise EyeProcessValidationError("c and d must satisfy 0 <= c < d <= 1.")
    return c + (d - c) * eyeprocess_irt_2pl_probability(theta, a=a, b=b, D=D)


def eyeprocess_irt_grm_probability(theta: Any, a: float = 1, thresholds: Sequence[float] = (), D: float = 1) -> np.ndarray:
    th = _theta(theta); a = _scalar(a, "a", positive=True); D = _scalar(D, "D", positive=True)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0 or not np.all(np.isfinite(thresholds)) or np.any(np.diff(thresholds) <= 0):
        raise EyeProcessValidationError("thresholds must be finite and strictly increasing.")
    rows = []
    for t in th:
        ge = expit(D * a * (t - thresholds))
        p = np.r_[1 - ge[0], ge[:-1] - ge[1:], ge[-1]]
        p = np.maximum(0.0, p); p /= p.sum()
        rows.append(p)
    return np.vstack(rows)


def eyeprocess_irt_gpcm_probability(theta: Any, a: float = 1, steps: Sequence[float] = (), D: float = 1) -> np.ndarray:
    th = _theta(theta); a = _scalar(a, "a", positive=True); D = _scalar(D, "D", positive=True)
    steps = np.asarray(steps, dtype=float)
    if steps.size == 0 or not np.all(np.isfinite(steps)):
        raise EyeProcessValidationError("steps must be finite and non-empty.")
    rows = []
    for t in th:
        eta = np.r_[0.0, np.cumsum(D * a * (t - steps))]
        eta -= eta.max(); z = np.exp(eta); rows.append(z / z.sum())
    return np.vstack(rows)


def eyeprocess_irt_nominal_probability(theta: Any, slopes: Sequence[float], intercepts: Sequence[float]) -> np.ndarray:
    th = _theta(theta); slopes = np.asarray(slopes, dtype=float); intercepts = np.asarray(intercepts, dtype=float)
    if slopes.size < 2 or slopes.shape != intercepts.shape or not np.all(np.isfinite(slopes)) or not np.all(np.isfinite(intercepts)):
        raise EyeProcessValidationError("slopes and intercepts must be finite vectors of equal length >= 2.")
    rows = []
    for t in th:
        eta = intercepts + slopes * t; eta -= eta.max(); z = np.exp(eta); rows.append(z / z.sum())
    return np.vstack(rows)


def _numeric_information(prob_fun: Callable[[float], np.ndarray], theta: Any, h: float = 1e-5) -> np.ndarray:
    th = _theta(theta); out = []
    for t in th:
        p0 = np.asarray(prob_fun(float(t)), dtype=float).reshape(-1)
        pp = np.asarray(prob_fun(float(t + h)), dtype=float).reshape(-1)
        pm = np.asarray(prob_fun(float(t - h)), dtype=float).reshape(-1)
        dp = (pp - pm) / (2 * h)
        out.append(float(np.sum(dp * dp / np.maximum(p0, _EPS))))
    return np.asarray(out)


def eyeprocess_irt_item_information(theta: Any, family: str = "2pl", D: float = 1, **kwargs: Any) -> np.ndarray:
    th = _theta(theta)
    if family not in {"2pl", "3pl", "4pl", "grm", "gpcm", "nominal"}:
        raise EyeProcessValidationError("unsupported IRT family.")
    if family in {"2pl", "3pl", "4pl"}:
        a = float(kwargs.get("a", 1)); b = float(kwargs.get("b", 0)); c = float(kwargs.get("c", 0)); d = float(kwargs.get("d", 1))
        L = eyeprocess_irt_2pl_probability(th, a=a, b=b, D=D)
        if family == "2pl": p = L; deriv = D * a * L * (1 - L)
        elif family == "3pl":
            if c < 0 or c >= 1: raise EyeProcessValidationError("c must lie in [0,1) for 3PL information.")
            p = c + (1 - c) * L; deriv = (1 - c) * D * a * L * (1 - L)
        else:
            if c < 0 or d > 1 or c >= d: raise EyeProcessValidationError("c and d must satisfy 0 <= c < d <= 1 for 4PL information.")
            p = c + (d - c) * L; deriv = (d - c) * D * a * L * (1 - L)
        return deriv**2 / np.maximum(p * (1 - p), _EPS)
    if family == "grm":
        return _numeric_information(lambda t: eyeprocess_irt_grm_probability([t], a=kwargs.get("a", 1), thresholds=kwargs["thresholds"], D=D)[0], th)
    if family == "gpcm":
        return _numeric_information(lambda t: eyeprocess_irt_gpcm_probability([t], a=kwargs.get("a", 1), steps=kwargs["steps"], D=D)[0], th)
    return _numeric_information(lambda t: eyeprocess_irt_nominal_probability([t], slopes=kwargs["slopes"], intercepts=kwargs["intercepts"])[0], th)


def eyeprocess_irt_test_information(theta: Any, items: Any, D: float = 1) -> pd.DataFrame:
    th = _theta(theta); df = _item_pars(items)
    total = np.zeros(th.size)
    for r in df.itertuples(index=False):
        total += eyeprocess_irt_item_information(th, "4pl", a=r.a, b=r.b, c=r.c, d=r.d, D=D)
    out = pd.DataFrame({"theta": th, "information": total, "conditional_sem": 1 / np.sqrt(np.maximum(total, _EPS))})
    return _tag(out, "eye_irt_information_profile")


def eyeprocess_irt_conditional_sem(information: Any) -> np.ndarray:
    x = np.asarray(information, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0): raise EyeProcessValidationError("information must be finite and non-negative.")
    return np.where(x > 0, 1 / np.sqrt(x), np.inf)


def eyeprocess_irt_expected_score(theta: Any, family: str = "2pl", **kwargs: Any) -> np.ndarray:
    th = _theta(theta)
    if family == "2pl": return eyeprocess_irt_2pl_probability(th, a=kwargs.get("a",1), b=kwargs.get("b",0), D=kwargs.get("D",1))
    if family == "3pl": return eyeprocess_irt_3pl_probability(th, a=kwargs.get("a",1), b=kwargs.get("b",0), c=kwargs.get("c",0.2), D=kwargs.get("D",1))
    if family == "4pl": return eyeprocess_irt_4pl_probability(th, a=kwargs.get("a",1), b=kwargs.get("b",0), c=kwargs.get("c",0), d=kwargs.get("d",1), D=kwargs.get("D",1))
    if family == "grm": p = eyeprocess_irt_grm_probability(th, a=kwargs.get("a",1), thresholds=kwargs["thresholds"], D=kwargs.get("D",1))
    elif family == "gpcm": p = eyeprocess_irt_gpcm_probability(th, a=kwargs.get("a",1), steps=kwargs["steps"], D=kwargs.get("D",1))
    elif family == "nominal": p = eyeprocess_irt_nominal_probability(th, slopes=kwargs["slopes"], intercepts=kwargs["intercepts"])
    else: raise EyeProcessValidationError("unsupported IRT family.")
    return p @ np.arange(p.shape[1], dtype=float)


def eyeprocess_irt_test_characteristic_curve(theta: Any, items: Any, D: float = 1) -> pd.DataFrame:
    th = _theta(theta); df = _item_pars(items); total = np.zeros(th.size)
    for r in df.itertuples(index=False): total += eyeprocess_irt_4pl_probability(th, r.a, r.b, r.c, r.d, D)
    return _tag(pd.DataFrame({"theta": th, "expected_score": total, "max_score": len(df)}), "eye_irt_test_characteristic_curve")


def eyeprocess_irt_information_area(theta: Any, information: Any) -> float:
    th = np.asarray(theta,dtype=float); info=np.asarray(information,dtype=float)
    if th.size != info.size or th.size < 2 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(info)): raise EyeProcessValidationError("theta and information must be finite vectors of equal length >= 2.")
    order=np.argsort(th); return float(np.trapezoid(info[order], th[order]))


def eyeprocess_irt_measurement_precision_profile(theta: Any, items: Any, target: Sequence[float]=(-2,2), D: float=1) -> EyeResult:
    target=np.asarray(target,dtype=float)
    if target.size!=2 or not np.all(np.isfinite(target)) or target[0]>=target[1]: raise EyeProcessValidationError("target must be an increasing finite length-2 vector.")
    info=eyeprocess_irt_test_information(theta,items,D=D); keep=(info.theta>=target[0])&(info.theta<=target[1])
    if int(keep.sum())<2: raise EyeProcessValidationError("theta must contain at least two finite grid points inside target.")
    return _result("eye_irt_precision_profile", curve=info, target=target, area=eyeprocess_irt_information_area(info.loc[keep,"theta"],info.loc[keep,"information"]), min_information=float(info.loc[keep,"information"].min()), max_sem=float(info.loc[keep,"conditional_sem"].max()))


# ---------------------------------------------------------------------------
# 084: diagnostics
# ---------------------------------------------------------------------------

def eyeprocess_irt_item_fit_residuals(responses: Any, probabilities: Any, item_ids: Sequence[str] | None=None) -> pd.DataFrame:
    y=_binary_matrix(responses); p=_prob_matrix(probabilities,y.shape)
    if item_ids is None: item_ids=[f"item_{i+1}" for i in range(y.shape[1])]
    if len(item_ids)!=y.shape[1]: raise EyeProcessValidationError("item_ids length must equal number of items.")
    rows=[]
    for j,item_id in enumerate(item_ids):
        keep=~np.isnan(y[:,j])&~np.isnan(p[:,j])
        if not keep.any(): rows.append(dict(item_id=str(item_id),n=0,mean_residual=np.nan,rms_standardized_residual=np.nan,outfit=np.nan,infit=np.nan)); continue
        yy=y[keep,j]; pp=p[keep,j]; r=yy-pp; var=pp*(1-pp); sr=r/np.sqrt(np.maximum(var,_EPS))
        rows.append(dict(item_id=str(item_id),n=int(len(yy)),mean_residual=float(r.mean()),rms_standardized_residual=float(np.sqrt(np.mean(sr**2))),outfit=float(np.mean(sr**2)),infit=float(np.sum(r**2)/np.sum(var))))
    return _tag(pd.DataFrame(rows),"eye_irt_item_fit")


def eyeprocess_irt_person_fit_residuals(responses: Any, probabilities: Any, person_ids: Sequence[str] | None=None) -> pd.DataFrame:
    y=_binary_matrix(responses); p=_prob_matrix(probabilities,y.shape)
    if person_ids is None: person_ids=[f"person_{i+1}" for i in range(y.shape[0])]
    if len(person_ids)!=y.shape[0]: raise EyeProcessValidationError("person_ids length must equal number of persons.")
    rows=[]
    for i,pid in enumerate(person_ids):
        keep=~np.isnan(y[i])&~np.isnan(p[i])
        if not keep.any(): rows.append(dict(person_id=str(pid),n=0,raw_score=np.nan,expected_score=np.nan,outfit=np.nan,infit=np.nan)); continue
        yy=y[i,keep]; pp=p[i,keep]; r=yy-pp; var=pp*(1-pp); sr=r/np.sqrt(np.maximum(var,_EPS))
        rows.append(dict(person_id=str(pid),n=int(len(yy)),raw_score=float(yy.sum()),expected_score=float(pp.sum()),outfit=float(np.mean(sr**2)),infit=float(np.sum(r**2)/np.sum(var))))
    return _tag(pd.DataFrame(rows),"eye_irt_person_fit")


def eyeprocess_irt_q3(responses: Any, probabilities: Any, use: str="pairwise.complete.obs") -> pd.DataFrame:
    y=_binary_matrix(responses); p=_prob_matrix(probabilities,y.shape); resid=y-p
    n=y.shape[1]; out=np.full((n,n),np.nan)
    for i in range(n):
        for j in range(i+1,n):
            keep=np.isfinite(resid[:,i])&np.isfinite(resid[:,j])
            if keep.sum()>=2 and np.std(resid[keep,i],ddof=1)>0 and np.std(resid[keep,j],ddof=1)>0:
                out[i,j]=out[j,i]=np.corrcoef(resid[keep,i],resid[keep,j])[0,1]
    cols = list(responses.columns) if isinstance(responses,pd.DataFrame) else [f"item_{i+1}" for i in range(n)]
    return _tag(pd.DataFrame(out,index=cols,columns=cols),"eye_irt_q3_matrix")


def eyeprocess_irt_local_dependence_pairs(q3: Any, threshold: float=0.20, absolute: bool=True) -> pd.DataFrame:
    df=q3.copy() if isinstance(q3,pd.DataFrame) else pd.DataFrame(np.asarray(q3,dtype=float)); a=df.to_numpy(dtype=float); threshold=_scalar(threshold,"threshold",nonnegative=True)
    if a.ndim!=2 or a.shape[0]!=a.shape[1]: raise EyeProcessValidationError("q3 must be square and threshold non-negative.")
    ids=[str(x) for x in df.columns] if isinstance(q3,pd.DataFrame) else [f"item_{i+1}" for i in range(a.shape[1])]
    rows=[]
    for i in range(a.shape[0]):
        for j in range(i+1,a.shape[1]):
            v=a[i,j]
            if np.isfinite(v) and ((abs(v)>=threshold) if absolute else (v>=threshold)): rows.append((ids[i],ids[j],v,abs(v)))
    return pd.DataFrame(rows,columns=["item_1","item_2","q3","abs_q3"]).sort_values("abs_q3",ascending=False,ignore_index=True) if rows else pd.DataFrame(columns=["item_1","item_2","q3","abs_q3"])


def eyeprocess_irt_extreme_score_audit(responses: Any, lower_fraction: float=.02, upper_fraction: float=.98) -> pd.DataFrame:
    y=_binary_matrix(responses); lower_fraction=float(lower_fraction); upper_fraction=float(upper_fraction)
    if lower_fraction<0 or upper_fraction>1 or lower_fraction>=upper_fraction: raise EyeProcessValidationError("fractions must satisfy 0 <= lower < upper <= 1.")
    n=np.sum(~np.isnan(y),axis=1); score=np.nansum(y,axis=1); frac=np.divide(score,n,out=np.full(len(n),np.nan),where=n>0)
    ids=list(responses.index.astype(str)) if isinstance(responses,pd.DataFrame) else [f"person_{i+1}" for i in range(y.shape[0])]
    return pd.DataFrame({"person_id":ids,"n_answered":n.astype(int),"raw_score":score,"score_fraction":frac,"lower_extreme":frac<=lower_fraction,"upper_extreme":frac>=upper_fraction})


def eyeprocess_irt_threshold_order_audit(item_id: str, thresholds: Sequence[float]) -> EyeResult:
    t=np.asarray(thresholds,dtype=float)
    if not isinstance(item_id,str) or not item_id or t.size==0 or not np.all(np.isfinite(t)): raise EyeProcessValidationError("invalid item_id or thresholds.")
    d=np.diff(t); return _result("eye_irt_threshold_audit",item_id=item_id,thresholds=t,ordered=bool(np.all(d>0)),minimum_gap=float(d.min()) if d.size else np.nan,reversals=(np.where(d<=0)[0]+1).tolist())


def eyeprocess_irt_monotonicity_audit(theta: Any, probability: Any, tolerance: float=1e-8) -> EyeResult:
    th=np.asarray(theta,dtype=float); p=np.asarray(probability,dtype=float); tolerance=_scalar(tolerance,"tolerance",nonnegative=True)
    if th.size!=p.size or th.size<2 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)): raise EyeProcessValidationError("theta/probability must be finite equal-length vectors; probabilities in [0,1].")
    o=np.argsort(th); th=th[o]; p=p[o]; d=np.diff(p)
    return _result("eye_irt_monotonicity_audit",monotone_non_decreasing=bool(np.all(d>=-tolerance)),n_decreases=int(np.sum(d < -tolerance)),largest_decrease=float(d.min()) if np.any(d<0) else 0.0,theta=th,probability=p)


def eyeprocess_irt_category_function_audit(probabilities: Any, tolerance: float=1e-8) -> EyeResult:
    p=np.asarray(probabilities,dtype=float); tolerance=_scalar(tolerance,"tolerance",nonnegative=True)
    if p.ndim!=2 or p.shape[0]==0 or p.shape[1]<2 or not np.all(np.isfinite(p)): raise EyeProcessValidationError("probabilities must be a finite matrix with >= 2 categories.")
    rs=p.sum(axis=1)
    return _result("eye_irt_category_audit",valid_bounds=bool(np.all((p>=-tolerance)&(p<=1+tolerance))),rows_sum_to_one=bool(np.all(np.abs(rs-1)<=tolerance)),max_sum_error=float(np.max(np.abs(rs-1))),min_probability=float(p.min()),max_probability=float(p.max()))


def eyeprocess_irt_parameter_plausibility_audit(items: Any, discrimination: Sequence[float]=(0.2,4), difficulty: Sequence[float]=(-6,6), lower_asymptote: Sequence[float]=(0,0.5), upper_asymptote: Sequence[float]=(0.5,1)) -> pd.DataFrame:
    df=_item_pars(items); ranges=[discrimination,difficulty,lower_asymptote,upper_asymptote]
    if any(len(z)!=2 or not np.all(np.isfinite(z)) or z[0]>=z[1] for z in ranges): raise EyeProcessValidationError("all ranges must be increasing finite length-2 vectors.")
    out=pd.DataFrame({"item_id":df.item_id,"a_flag":(df.a<discrimination[0])|(df.a>discrimination[1]),"b_flag":(df.b<difficulty[0])|(df.b>difficulty[1]),"c_flag":(df.c<lower_asymptote[0])|(df.c>lower_asymptote[1]),"d_flag":(df.d<upper_asymptote[0])|(df.d>upper_asymptote[1])})
    out["any_flag"]=out[["a_flag","b_flag","c_flag","d_flag"]].any(axis=1); out.attrs["guardrail"]="Ranges are review conventions, not universal psychometric cutoffs."; return out


def eyeprocess_irt_ppc_discrepancy(observed: Any, replicated: Any, statistic: str="mean_score") -> EyeResult:
    obs=_binary_matrix(observed); rep=np.asarray(replicated,dtype=float)
    if rep.ndim!=3 or rep.shape[1:]!=obs.shape: raise EyeProcessValidationError("replicated must be an array [draw, person, item] matching observed dimensions.")
    if statistic not in {"mean_score","score_sd","item_means","max_item_residual"}: raise EyeProcessValidationError("invalid statistic.")
    def stat(z: np.ndarray) -> float:
        if statistic=="mean_score": return float(np.mean(np.nansum(z,axis=1)))
        if statistic=="score_sd": return float(np.std(np.nansum(z,axis=1),ddof=1))
        if statistic=="item_means": return float(np.mean(np.nanmean(z,axis=0)))
        return float(np.max(np.abs(np.nanmean(z,axis=0)-np.nanmean(obs,axis=0))))
    obs_stat=0.0 if statistic=="max_item_residual" else stat(obs); rs=np.asarray([stat(z) for z in rep])
    return _result("eye_irt_ppc_discrepancy",statistic=statistic,observed=obs_stat,replicated=rs,posterior_predictive_p=float(np.mean(rs>=obs_stat)),interval=_safe_quantile(rs,[.025,.5,.975]))


def eyeprocess_irt_fit_dashboard(item_fit: Any=None, person_fit: Any=None, q3: Any=None, parameter_audit: Any=None, identification: Any=None) -> EyeResult:
    components={"item_fit":item_fit,"person_fit":person_fit,"q3":q3,"parameter_audit":parameter_audit,"identification":identification}; present=[k for k,v in components.items() if v is not None]
    return _result("eye_irt_fit_dashboard",components=components,present=present,n_components=len(present),interpretation="Diagnostics identify model/data tensions; they are not evidence for behavioral or clinical labels.")


# ---------------------------------------------------------------------------
# 085: scoring/adaptive
# ---------------------------------------------------------------------------

def _response_loglik(theta: float, response: Any, items: Any, D: float=1) -> float:
    df=_item_pars(items); y=np.asarray(response,dtype=float).reshape(-1)
    if y.size!=len(df) or np.any(~np.isnan(y) & ~np.isin(y,[0,1])): raise EyeProcessValidationError("response must contain 0/1/NA and match items.")
    keep=~np.isnan(y)
    if not keep.any(): return 0.0
    probs=np.array([eyeprocess_irt_4pl_probability([theta],r.a,r.b,r.c,r.d,D)[0] for r in df.iloc[np.where(keep)[0]].itertuples(index=False)])
    yy=y[keep]; return float(np.sum(yy*np.log(np.maximum(probs,_EPS))+(1-yy)*np.log(np.maximum(1-probs,_EPS))))


def eyeprocess_irt_eap_score(response: Any, items: Any, theta_grid: Any=np.linspace(-4,4,81), prior_mean: float=0, prior_sd: float=1, D: float=1) -> EyeResult:
    grid=_theta(theta_grid)
    if grid.size<5: raise EyeProcessValidationError("theta_grid must contain at least five points.")
    prior_mean=_scalar(prior_mean,"prior_mean"); prior_sd=_scalar(prior_sd,"prior_sd",positive=True)
    lw=np.array([_response_loglik(float(t),response,items,D)+norm.logpdf(t,prior_mean,prior_sd) for t in grid]); w=np.exp(lw-lw.max()); w/=w.sum(); est=float(np.sum(grid*w)); se=float(np.sqrt(np.sum((grid-est)**2*w)))
    return _result("eye_irt_score",estimate=est,se=se,theta=grid,posterior=w,method="EAP")


def eyeprocess_irt_map_score(response: Any, items: Any, bounds: Sequence[float]=(-6,6), prior_mean: float=0, prior_sd: float=1, D: float=1) -> EyeResult:
    bounds=np.asarray(bounds,dtype=float); prior_mean=_scalar(prior_mean,"prior_mean"); prior_sd=_scalar(prior_sd,"prior_sd",positive=True)
    if bounds.size!=2 or not np.all(np.isfinite(bounds)) or bounds[0]>=bounds[1]: raise EyeProcessValidationError("invalid bounds or normal prior.")
    fit=minimize_scalar(lambda t:-(_response_loglik(float(t),response,items,D)+norm.logpdf(t,prior_mean,prior_sd)),bounds=(bounds[0],bounds[1]),method="bounded")
    return _result("eye_irt_score",estimate=float(fit.x),objective=float(fit.fun),method="MAP",bounds=bounds)


def eyeprocess_irt_mle_score(response: Any, items: Any, bounds: Sequence[float]=(-6,6), D: float=1) -> EyeResult:
    bounds=np.asarray(bounds,dtype=float)
    if bounds.size!=2 or not np.all(np.isfinite(bounds)) or bounds[0]>=bounds[1]: raise EyeProcessValidationError("bounds must be increasing and finite.")
    fit=minimize_scalar(lambda t:-_response_loglik(float(t),response,items,D),bounds=(bounds[0],bounds[1]),method="bounded")
    boundary=bool(abs(fit.x-bounds[0])<=1e-4 or abs(fit.x-bounds[1])<=1e-4)
    return _result("eye_irt_score",estimate=float(fit.x),objective=float(fit.fun),method="ML",bounds=bounds,boundary=boundary)


def eyeprocess_irt_score_table(responses: Any, items: Any, method: str="EAP", person_ids: Sequence[str]|None=None, **kwargs: Any) -> pd.DataFrame:
    y=_binary_matrix(responses); method=method.upper()
    if method not in {"EAP","MAP","ML"}: raise EyeProcessValidationError("method must be EAP, MAP, or ML.")
    if person_ids is None: person_ids=list(responses.index.astype(str)) if isinstance(responses,pd.DataFrame) else [f"person_{i+1}" for i in range(y.shape[0])]
    if len(person_ids)!=y.shape[0]: raise EyeProcessValidationError("person_ids length mismatch.")
    fun={"EAP":eyeprocess_irt_eap_score,"MAP":eyeprocess_irt_map_score,"ML":eyeprocess_irt_mle_score}[method]; rows=[]
    for i,pid in enumerate(person_ids):
        z=fun(y[i],items,**kwargs); rows.append({"person_id":str(pid),"estimate":z["estimate"],"se":z.get("se",np.nan),"method":method})
    return pd.DataFrame(rows)


def eyeprocess_irt_plausible_values(score: Any, n: int=5, seed: int=1) -> np.ndarray:
    if not isinstance(score,Mapping) or "posterior" not in score or "theta" not in score:
        raise EyeProcessValidationError("score must be an EAP eye_irt_score with posterior grid weights.")
    n0 = np.asarray(n)
    seed0 = np.asarray(seed)
    if n0.size != 1 or seed0.size != 1:
        raise EyeProcessValidationError("n and seed must be positive scalar integers.")
    try:
        n = int(n0.item()); seed = int(seed0.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError("n and seed must be positive scalar integers.") from exc
    if n < 1 or seed < 1:
        raise EyeProcessValidationError("n and seed must be positive scalar integers.")
    rng=np.random.default_rng(seed)
    return rng.choice(np.asarray(score["theta"],dtype=float),size=n,replace=True,p=np.asarray(score["posterior"],dtype=float))


def eyeprocess_irt_marginal_reliability(theta_estimate: Any, se: Any) -> float:
    th=np.asarray(theta_estimate,dtype=float); se=np.asarray(se,dtype=float); keep=np.isfinite(th)&np.isfinite(se)&(se>=0)
    if keep.sum()<2: return math.nan
    v=float(np.var(th[keep],ddof=1))
    if not np.isfinite(v) or v<=0: return math.nan
    return float(np.clip(1-np.mean(se[keep]**2)/v,0,1))


def eyeprocess_irt_score_uncertainty(scores: Any) -> EyeResult:
    df=_as_df(scores,"scores"); _req_cols(df,["estimate","se"],"scores"); est=pd.to_numeric(df.estimate,errors="coerce").to_numpy(); se=pd.to_numeric(df.se,errors="coerce").to_numpy()
    return _result("eye_irt_score_uncertainty",n=int(np.isfinite(est).sum()),mean_se=_finite_mean(se),median_se=_safe_quantile(se,.5),p95_se=_safe_quantile(se,.95),marginal_reliability=eyeprocess_irt_marginal_reliability(est,se))


def eyeprocess_irt_information_targeting(items: Any, theta: Any, weights: Any=None, D: float=1) -> EyeResult:
    th=_theta(theta)
    if weights is None: w=np.repeat(1/th.size,th.size)
    else: w=np.asarray(weights,dtype=float)
    if w.size!=th.size or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0: raise EyeProcessValidationError("weights must be non-negative and match theta.")
    w=w/w.sum(); info=eyeprocess_irt_test_information(th,items,D=D)
    return _result("eye_irt_information_targeting",weighted_information=float(np.sum(info.information*w)),weighted_sem=float(np.sum(info.conditional_sem*w)),curve=info,weights=w)


def eyeprocess_irt_item_bank(items: Any, content: Sequence[str]|None=None, exposure_limit: float=1) -> EyeResult:
    df=_item_pars(items); exposure_limit=float(exposure_limit)
    if not (0<exposure_limit<=1): raise EyeProcessValidationError("exposure_limit must lie in (0,1].")
    if content is not None:
        c=list(map(str,content))
        if len(c)!=len(df): raise EyeProcessValidationError("content must match item rows and be non-missing.")
        df["content"]=c
    return _result("eye_irt_item_bank",items=df,exposure_limit=exposure_limit)


def validate_eyeprocess_irt_item_bank(x: Any) -> bool:
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_irt_item_bank": raise EyeProcessValidationError("x must be an eye_irt_item_bank.")
    _item_pars(x["items"]); return True


def eyeprocess_irt_item_selection(bank: Any, theta: float, administered: Sequence[str]=(), exposure: Any=None, content_required: Sequence[str]|None=None, D: float=1) -> EyeResult:
    validate_eyeprocess_irt_item_bank(bank); theta=_scalar(theta,"theta"); items=bank["items"].copy()
    administered_ids = [str(administered)] if isinstance(administered, str) else list(map(str, administered))
    eligible=~items.item_id.isin(administered_ids)
    if exposure is not None:
        ex=_as_df(exposure,"exposure"); _req_cols(ex,["item_id","rate"],"exposure"); rate=ex.set_index(ex.item_id.astype(str)).rate; er=items.item_id.map(rate).fillna(0).astype(float); eligible &= er < float(bank["exposure_limit"])
    if content_required is not None:
        if "content" not in items: raise EyeProcessValidationError("bank has no content labels.")
        eligible &= items.content.astype(str).isin(list(map(str,content_required)))
    idx=np.where(eligible.to_numpy())[0]
    if idx.size==0: return _result("eye_irt_item_selection",selected=None,reason="no_eligible_item",information=np.nan)
    info=np.array([eyeprocess_irt_item_information([theta],"4pl",a=items.iloc[j].a,b=items.iloc[j].b,c=items.iloc[j].c,d=items.iloc[j].d,D=D)[0] for j in idx]); j=idx[int(np.argmax(info))]
    return _result("eye_irt_item_selection",selected=str(items.iloc[j].item_id),information=float(info.max()),theta=theta,reason="maximum_information",candidate_count=int(idx.size))


def eyeprocess_irt_stopping_rule(n_administered: int, se: float=np.nan, min_items: int=5, max_items: int=30, target_se: float=.30) -> dict[str,Any]:
    vals = [np.asarray(v) for v in (n_administered, se, min_items, max_items, target_se)]
    if any(v.size != 1 for v in vals):
        raise EyeProcessValidationError("invalid stopping-rule arguments.")
    try:
        n_administered=int(vals[0].item()); se=float(vals[1].item()); min_items=int(vals[2].item()); max_items=int(vals[3].item()); target_se=float(vals[4].item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError("invalid stopping-rule arguments.") from exc
    if min_items<1 or max_items<min_items or n_administered<0 or not np.isfinite(target_se) or target_se<=0: raise EyeProcessValidationError("invalid stopping-rule arguments.")
    precision=bool(np.isfinite(se) and se<=target_se and n_administered>=min_items); maxmet=n_administered>=max_items
    return {"stop":precision or maxmet,"reason":"target_precision" if precision else "maximum_items" if maxmet else "continue","n_administered":n_administered,"se":se}


def eyeprocess_irt_exposure_summary(administered: Sequence[str], item_bank_ids: Sequence[str]|None=None) -> pd.DataFrame:
    a=list(map(str,administered)); ids=list(dict.fromkeys(map(str,item_bank_ids if item_bank_ids is not None else a))); counts=pd.Series(a).value_counts() if a else pd.Series(dtype=int); total=len(a)
    return pd.DataFrame({"item_id":ids,"count":[int(counts.get(i,0)) for i in ids],"rate":[float(counts.get(i,0)/total) if total else 0.0 for i in ids]})


def eyeprocess_irt_content_balance_audit(administered: Sequence[str], item_bank: Any, target: Mapping[str,float]|None=None) -> pd.DataFrame:
    validate_eyeprocess_irt_item_bank(item_bank); items=item_bank["items"]
    if "content" not in items: raise EyeProcessValidationError("item bank has no content labels.")
    mapping=dict(zip(items.item_id.astype(str),items.content.astype(str))); vals=[]
    for i in map(str,administered):
        if i not in mapping: raise EyeProcessValidationError("administered contains item IDs absent from bank.")
        vals.append(mapping[i])
    if not vals: raise EyeProcessValidationError("administered must contain at least one item.")
    obs=pd.Series(vals).value_counts(normalize=True)
    if target is None: t={c:1/len(obs) for c in obs.index}
    else:
        t={str(k):float(v) for k,v in target.items()}
        if any(not np.isfinite(v) or v<0 for v in t.values()) or sum(t.values())<=0: raise EyeProcessValidationError("target must be a uniquely named, finite, non-negative numeric vector with positive sum.")
        s=sum(t.values()); t={k:v/s for k,v in t.items()}
    cats=list(dict.fromkeys(list(obs.index.astype(str))+list(t))); return pd.DataFrame({"content":cats,"observed":[float(obs.get(c,0)) for c in cats],"target":[float(t.get(c,0)) for c in cats],"deviation":[float(obs.get(c,0)-t.get(c,0)) for c in cats]})


def eyeprocess_irt_adaptive_trace(item_id: Sequence[str], theta_before: Any, theta_after: Any, se_after: Any, information: Any, response: Any=np.nan) -> pd.DataFrame:
    ids=list(map(str,item_id)); n=len(ids); arrays=[np.asarray(v) for v in (theta_before,theta_after,se_after,information)]
    resp=np.asarray([np.nan]*n if np.ndim(response)==0 and pd.isna(response) and n!=1 else response)
    if any(len(np.atleast_1d(v))!=n for v in arrays+[resp]): raise EyeProcessValidationError("all trace vectors must have equal length.")
    return _tag(pd.DataFrame({"step":np.arange(1,n+1),"item_id":ids,"theta_before":np.asarray(theta_before,float),"theta_after":np.asarray(theta_after,float),"se_after":np.asarray(se_after,float),"information":np.asarray(information,float),"response":np.asarray(resp,float)}),"eye_irt_adaptive_trace")


def eyeprocess_irt_information_gain(se_before: Any, se_after: Any) -> np.ndarray:
    a=np.asarray(se_before,dtype=float); b=np.asarray(se_after,dtype=float)
    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a<=0) or np.any(b<=0): raise EyeProcessValidationError("SE values must be positive finite equal-length vectors.")
    return 1/b**2-1/a**2


def eyeprocess_irt_process_aware_selection_penalty(information: Any, burden: Any, burden_weight: float=0, quality_risk: Any=0, quality_weight: float=0) -> np.ndarray:
    info=np.atleast_1d(np.asarray(information,float)); burden=np.atleast_1d(np.asarray(burden,float)); risk=np.atleast_1d(np.asarray(quality_risk,float))
    if burden.size==1 and info.size>1: burden=np.repeat(burden,info.size)
    if risk.size==1 and info.size>1: risk=np.repeat(risk,info.size)
    if info.size==0 or info.size!=burden.size or info.size!=risk.size or not np.all(np.isfinite(np.r_[info,burden,risk])): raise EyeProcessValidationError("inputs must be finite compatible vectors.")
    burden_weight=_scalar(burden_weight,"burden_weight",nonnegative=True); quality_weight=_scalar(quality_weight,"quality_weight",nonnegative=True)
    return info-burden_weight*burden-quality_weight*risk


# ---------------------------------------------------------------------------
# 086: linking/invariance
# ---------------------------------------------------------------------------

def _anchor_merge(reference: Any, focal: Any, anchors: Sequence[str]|None=None) -> tuple[pd.DataFrame,pd.DataFrame,list[str]]:
    r=_item_pars(reference); f=_item_pars(focal); ids=[i for i in r.item_id if i in set(f.item_id)]
    if anchors is not None: ids=[i for i in ids if i in set(map(str,anchors))]
    if len(ids)<2: raise EyeProcessValidationError("At least two common anchor items are required.")
    return r.set_index("item_id").loc[ids].reset_index(),f.set_index("item_id").loc[ids].reset_index(),ids


def eyeprocess_irt_mean_sigma_link(reference: Any, focal: Any, anchors: Sequence[str]|None=None) -> EyeResult:
    r,f,ids=_anchor_merge(reference,focal,anchors); sr=float(r.b.std(ddof=1)); sf=float(f.b.std(ddof=1))
    if not np.isfinite(sr) or not np.isfinite(sf) or sf<=0: raise EyeProcessValidationError("anchor difficulty SDs must be finite and focal SD > 0.")
    A=sr/sf; B=float(r.b.mean()-A*f.b.mean()); return _result("eye_irt_link",A=A,B=B,method="mean-sigma",anchors=ids,objective=np.nan)


def eyeprocess_irt_mean_mean_link(reference: Any, focal: Any, anchors: Sequence[str]|None=None) -> EyeResult:
    r,f,ids=_anchor_merge(reference,focal,anchors); ar=float(r.a.mean()); af=float(f.a.mean())
    if ar<=0 or af<=0: raise EyeProcessValidationError("mean discrimination must be positive.")
    A=af/ar; B=float(r.b.mean()-A*f.b.mean()); return _result("eye_irt_link",A=A,B=B,method="mean-mean",anchors=ids,objective=np.nan)


def _apply_link_df(items: Any,A: float,B: float) -> pd.DataFrame:
    df=_item_pars(items); A=float(A);B=float(B)
    if not np.isfinite(A) or A<=0 or not np.isfinite(B): raise EyeProcessValidationError("A must be positive finite and B finite.")
    df["a"]=df.a/A; df["b"]=A*df.b+B; return df


def eyeprocess_irt_apply_link(items: Any, link: Any) -> pd.DataFrame:
    if not isinstance(link,Mapping) or "A" not in link or "B" not in link: raise EyeProcessValidationError("link must inherit from eye_irt_link.")
    return _apply_link_df(items,float(link["A"]),float(link["B"]))


def _link_opt(reference: Any,focal: Any,anchors: Sequence[str]|None,theta: Any,weights: Any,start: Sequence[float],method: str) -> EyeResult:
    r,f,ids=_anchor_merge(reference,focal,anchors); th=_theta(theta); w=norm.pdf(th) if weights is None else np.asarray(weights,float)
    if w.size!=th.size or not np.all(np.isfinite(w)) or np.any(w<0) or w.sum()<=0: raise EyeProcessValidationError("weights must be finite, non-negative, match theta, and have positive sum.")
    w=w/w.sum(); start=np.asarray(start,float)
    if start.size!=2 or not np.all(np.isfinite(start)) or start[0]<=0: raise EyeProcessValidationError("start must contain positive finite A and finite B.")
    if method=="Stocking-Lord":
        target=eyeprocess_irt_test_characteristic_curve(th,r).expected_score.to_numpy()
        def obj(par):
            ff=_apply_link_df(f,math.exp(par[0]),par[1]); pred=eyeprocess_irt_test_characteristic_curve(th,ff).expected_score.to_numpy(); return float(np.sum(w*(target-pred)**2))
    else:
        pref=np.column_stack([eyeprocess_irt_4pl_probability(th,row.a,row.b,row.c,row.d) for row in r.itertuples(index=False)])
        def obj(par):
            ff=_apply_link_df(f,math.exp(par[0]),par[1]); pf=np.column_stack([eyeprocess_irt_4pl_probability(th,row.a,row.b,row.c,row.d) for row in ff.itertuples(index=False)]); return float(np.sum(w*np.sum((pref-pf)**2,axis=1)))
    fit=minimize(obj,[math.log(start[0]),start[1]],method="BFGS"); return _result("eye_irt_link",A=float(math.exp(fit.x[0])),B=float(fit.x[1]),method=method,anchors=ids,objective=float(fit.fun),convergence=0 if fit.success else 1)


def eyeprocess_irt_stocking_lord_link(reference: Any,focal: Any,anchors: Sequence[str]|None=None,theta: Any=np.linspace(-4,4,81),weights: Any=None,start: Sequence[float]=(1,0)) -> EyeResult: return _link_opt(reference,focal,anchors,theta,weights,start,"Stocking-Lord")
def eyeprocess_irt_haebara_link(reference: Any,focal: Any,anchors: Sequence[str]|None=None,theta: Any=np.linspace(-4,4,81),weights: Any=None,start: Sequence[float]=(1,0)) -> EyeResult: return _link_opt(reference,focal,anchors,theta,weights,start,"Haebara")


def eyeprocess_irt_link_stability(reference: Any,focal: Any,anchor_sets: Sequence[Sequence[str]]|Mapping[str,Sequence[str]],method: str="mean-sigma") -> EyeResult:
    if isinstance(anchor_sets,Mapping): pairs=list(anchor_sets.items())
    else: pairs=[(f"set_{i+1}",v) for i,v in enumerate(anchor_sets)]
    if not pairs: raise EyeProcessValidationError("anchor_sets must be a non-empty list.")
    fun={"mean-sigma":eyeprocess_irt_mean_sigma_link,"mean-mean":eyeprocess_irt_mean_mean_link,"Stocking-Lord":eyeprocess_irt_stocking_lord_link,"Haebara":eyeprocess_irt_haebara_link}.get(method)
    if fun is None: raise EyeProcessValidationError("invalid linking method.")
    rows=[]
    for name,anchors in pairs:
        z=fun(reference,focal,anchors=anchors); rows.append({"set":name,"n_anchors":len(z["anchors"]),"A":z["A"],"B":z["B"],"objective":z.get("objective",np.nan)})
    tab=pd.DataFrame(rows); return _result("eye_irt_link_stability",table=tab,sd_A=_finite_sd(tab.A),sd_B=_finite_sd(tab.B),method=method)


def eyeprocess_irt_anchor_audit(items: Any,dif: Any=None,max_abs_effect: float=.10,min_information: float|None=None) -> pd.DataFrame:
    df=_item_pars(items); max_abs_effect=_scalar(max_abs_effect,"max_abs_effect",nonnegative=True); out=pd.DataFrame({"item_id":df.item_id,"eligible":True,"reason":"eligible"})
    if dif is not None:
        dd=_as_df(dif,"dif"); _req_cols(dd,["item_id","effect"],"dif"); effect=dd.assign(item_id=dd.item_id.astype(str)).set_index("item_id").effect.abs(); e=out.item_id.map(effect); flag=e.notna()&(e>max_abs_effect); out.loc[flag,"eligible"]=False; out.loc[flag,"reason"]="DIF effect exceeds review threshold"
    if min_information is not None:
        mi=float(min_information); info=np.array([eyeprocess_irt_item_information([0],"4pl",a=r.a,b=r.b,c=r.c,d=r.d)[0] for r in df.itertuples(index=False)]); flag=info<mi; out.loc[flag,"eligible"]=False; out.loc[flag,"reason"]="information below review threshold"; out["information_theta0"]=info
    out.attrs["guardrail"]="Anchor eligibility is a screening aid; formal invariance evidence remains model-dependent."; return out


def eyeprocess_irt_anchor_purification(items: Any,effect_fun: Callable[[list[str]],Any],initial: Sequence[str]|None=None,threshold: float=.10,max_iter: int=10) -> EyeResult:
    df=_item_pars(items)
    if not callable(effect_fun): raise EyeProcessValidationError("effect_fun must be a function accepting anchor IDs and returning item_id/effect data.")
    anchors=[i for i in (list(map(str,initial)) if initial is not None else df.item_id.tolist()) if i in set(df.item_id)]; hist=[]
    for it in range(1,int(max_iter)+1):
        eff=_as_df(effect_fun(anchors),"effect_fun result"); _req_cols(eff,["item_id","effect"],"effect_fun result"); bad=[str(r.item_id) for r in eff.itertuples() if np.isfinite(float(r.effect)) and abs(float(r.effect))>threshold and str(r.item_id) in anchors]; hist.append({"iteration":it,"n_anchors":len(anchors),"removed":";".join(bad)})
        if not bad: break
        anchors=[a for a in anchors if a not in set(bad)]
        if len(anchors)<2: raise EyeProcessValidationError("Anchor purification left fewer than two anchors.")
    return _result("eye_irt_anchor_purification",anchors=anchors,history=pd.DataFrame(hist),threshold=threshold)


def eyeprocess_irt_dif_effect_curve(reference_item: Any,focal_item: Any,theta: Any=np.linspace(-4,4,81)) -> pd.DataFrame:
    r=_item_pars(reference_item); f=_item_pars(focal_item)
    if len(r)!=1 or len(f)!=1: raise EyeProcessValidationError("reference_item and focal_item must each contain exactly one item.")
    th=_theta(theta); rr=r.iloc[0];ff=f.iloc[0]; pr=eyeprocess_irt_4pl_probability(th,rr.a,rr.b,rr.c,rr.d); pf=eyeprocess_irt_4pl_probability(th,ff.a,ff.b,ff.c,ff.d)
    return _tag(pd.DataFrame({"theta":th,"reference":pr,"focal":pf,"signed_difference":pf-pr,"absolute_difference":abs(pf-pr)}),"eye_irt_dif_curve")


def eyeprocess_irt_dtf_curve(reference: Any,focal: Any,theta: Any=np.linspace(-4,4,81)) -> pd.DataFrame:
    r=_item_pars(reference); f=_item_pars(focal); ids=[i for i in r.item_id if i in set(f.item_id)]
    if not ids: raise EyeProcessValidationError("No common items.")
    r=r.set_index("item_id").loc[ids].reset_index(); f=f.set_index("item_id").loc[ids].reset_index(); tr=eyeprocess_irt_test_characteristic_curve(theta,r).expected_score.to_numpy(); tf=eyeprocess_irt_test_characteristic_curve(theta,f).expected_score.to_numpy(); th=_theta(theta)
    return _tag(pd.DataFrame({"theta":th,"reference":tr,"focal":tf,"signed_difference":tf-tr,"absolute_difference":abs(tf-tr)}),"eye_irt_dtf_curve")


def eyeprocess_irt_functioning_effect_summary(curve: Any) -> dict[str,float]:
    df=_as_df(curve,"curve"); _req_cols(df,["absolute_difference","signed_difference"],"curve"); a=pd.to_numeric(df.absolute_difference,errors="coerce").to_numpy(); finite=a[np.isfinite(a)]; area=np.nan
    if {"theta","signed_difference"}.issubset(df.columns):
        th=pd.to_numeric(df.theta,errors="coerce").to_numpy(); sd=pd.to_numeric(df.signed_difference,errors="coerce").to_numpy(); keep=np.isfinite(th)&np.isfinite(sd)
        if keep.sum()>=2: area=eyeprocess_irt_information_area(th[keep],sd[keep])
    return {"max_abs":float(finite.max()) if finite.size else np.nan,"mean_abs":float(finite.mean()) if finite.size else np.nan,"signed_area":float(area)}


def eyeprocess_irt_process_dif_concordance(dif: Any,process: Any,item_id: str="item_id",dif_effect: str="effect",process_effect: str="effect") -> EyeResult:
    d=_as_df(dif,"dif"); p=_as_df(process,"process"); _req_cols(d,[item_id,dif_effect],"dif"); _req_cols(p,[item_id,process_effect],"process"); z=d[[item_id,dif_effect]].merge(p[[item_id,process_effect]],on=item_id,suffixes=("_dif","_process"))
    if z.empty: return _result("eye_irt_process_dif_concordance",n=0,correlation=np.nan,table=z)
    a=pd.to_numeric(z[f"{dif_effect}_dif"],errors="coerce").to_numpy(); b=pd.to_numeric(z[f"{process_effect}_process"],errors="coerce").to_numpy(); keep=np.isfinite(a)&np.isfinite(b); cor=np.corrcoef(a[keep],b[keep])[0,1] if keep.sum()>=3 and np.std(a[keep],ddof=1)>0 and np.std(b[keep],ddof=1)>0 else np.nan
    return _result("eye_irt_process_dif_concordance",n=int(keep.sum()),correlation=float(cor),table=z,guardrail="Concordance is descriptive and does not establish a causal explanation of DIF.")


def eyeprocess_irt_session_drift(parameters: Any,item_id: str="item_id",session: str="session",parameter: str="b") -> pd.DataFrame:
    df=_as_df(parameters,"parameters"); _req_cols(df,[item_id,session,parameter],"parameters"); rows=[]
    for ident,z in df.groupby(item_id,sort=False):
        z=z.sort_values(session); v=pd.to_numeric(z[parameter],errors="coerce").to_numpy(); v=v[np.isfinite(v)]; rows.append({"item_id":str(ident),"n_sessions":len(v),"first":v[0] if len(v) else np.nan,"last":v[-1] if len(v) else np.nan,"change":v[-1]-v[0] if len(v) else np.nan,"range":np.ptp(v) if len(v) else np.nan})
    return pd.DataFrame(rows)


def eyeprocess_irt_device_drift(parameters: Any,item_id: str="item_id",device: str="device",parameter: str="b") -> pd.DataFrame:
    df=_as_df(parameters,"parameters"); _req_cols(df,[item_id,device,parameter],"parameters"); rows=[]
    for ident,z in df.groupby(item_id,sort=False):
        v=pd.to_numeric(z[parameter],errors="coerce").to_numpy(); v=v[np.isfinite(v)]; rows.append({"item_id":str(ident),"n_devices":int(z[device].nunique()),"mean":float(v.mean()) if len(v) else np.nan,"sd":float(np.std(v,ddof=1)) if len(v)>1 else np.nan,"range":float(np.ptp(v)) if len(v) else np.nan})
    return pd.DataFrame(rows)


def eyeprocess_irt_invariance_evidence(anchor_audit: Any=None,dif: Any=None,dtf: Any=None,linking: Any=None,process_concordance: Any=None) -> EyeResult:
    c={"anchor_audit":anchor_audit,"dif":dif,"dtf":dtf,"linking":linking,"process_concordance":process_concordance}; present=[k for k,v in c.items() if v is not None]; return _result("eye_irt_invariance_evidence",components=c,present=present,completeness=len(present)/len(c),interpretation="Evidence components characterize scale stability and functioning; no universal invariance cutoff is imposed.")


# ---------------------------------------------------------------------------
# 087: process-aware data contracts
# ---------------------------------------------------------------------------

def eyeprocess_joint_process_irt_spec(response_family: str="2pl",time_model: str="none",process_channels: Sequence[str]=("dwell","pupil","transitions"),person_covariates: Sequence[str]=(),item_covariates: Sequence[str]=(),missingness: str="ignorable",status: str="experimental") -> EyeResult:
    if response_family not in {"2pl","rasch","grm","gpcm"} or time_model not in {"none","lognormal","custom"} or missingness not in {"ignorable","modeled","gated"} or status not in {"experimental","reference","gated"}: raise EyeProcessValidationError("invalid joint-process IRT specification.")
    return _result("eye_joint_process_irt_spec",response_family=response_family,time_model=time_model,process_channels=list(dict.fromkeys(map(str,process_channels))),person_covariates=list(dict.fromkeys(map(str,person_covariates))),item_covariates=list(dict.fromkeys(map(str,item_covariates))),missingness=missingness,status=status)


def validate_eyeprocess_joint_process_irt_spec(x: Any) -> bool:
    if not isinstance(x,Mapping) or getattr(x,"eyeprocess_class",None)!="eye_joint_process_irt_spec": raise EyeProcessValidationError("x must inherit from eye_joint_process_irt_spec.")
    return True


def eyeprocess_process_irt_data_bundle(data: Any,person: str,item: str,response: str,response_time: str|None=None,process: Sequence[str]=(),covariates: Sequence[str]=()) -> EyeResult:
    df=_as_df(data,"data"); cols=list(dict.fromkeys([person,item,response]+([response_time] if response_time else [])+list(process)+list(covariates))); _req_cols(df,cols,"data")
    y=pd.to_numeric(df[response],errors="coerce"); bad=y.notna()&~y.isin([0,1])
    if bad.any(): raise EyeProcessValidationError("response must be binary/NA for this bundle.")
    if response_time is not None:
        rt=pd.to_numeric(df[response_time],errors="coerce");
        if ((rt.notna())&(rt<=0)).any(): raise EyeProcessValidationError("response_time must be positive where observed.")
    return _result("eye_process_irt_data_bundle",data=df[cols].copy(),person=person,item=item,response=response,response_time=response_time,process=list(process),covariates=list(covariates),n_persons=int(df[person].nunique()),n_items=int(df[item].nunique()))


def eyeprocess_response_time_profile(data: Any,person: str,item: str,response_time: str) -> EyeResult:
    df=_as_df(data,"data"); _req_cols(df,[person,item,response_time],"data"); rt=pd.to_numeric(df[response_time],errors="coerce"); good=rt.notna()&(rt>0)
    if not good.any(): raise EyeProcessValidationError("No positive finite response times.")
    z=df.loc[good,[person,item]].copy(); z["rt"]=rt[good].to_numpy(); z["log_rt"]=np.log(z.rt)
    itemtab=z.groupby(item,sort=False).agg(n=("rt","size"),mean_log_rt=("log_rt","mean"),sd_log_rt=("log_rt","std"),median_rt=("rt","median")).reset_index().rename(columns={item:"item_id"}); itemtab["item_id"]=itemtab.item_id.astype(str)
    pers=z.groupby(person,sort=False).agg(n=("rt","size"),mean_log_rt=("log_rt","mean"),median_rt=("rt","median")).reset_index().rename(columns={person:"person_id"}); pers["person_id"]=pers.person_id.astype(str)
    return _result("eye_response_time_profile",item=itemtab,person=pers,n=int(len(z)))


def eyeprocess_speed_accuracy_profile(data: Any,person: str,response: str,response_time: str) -> EyeResult:
    df=_as_df(data,"data"); _req_cols(df,[person,response,response_time],"data"); y=pd.to_numeric(df[response],errors="coerce"); rt=pd.to_numeric(df[response_time],errors="coerce"); good=y.isin([0,1])&rt.notna()&(rt>0); d=pd.DataFrame({person:df.loc[good,person].astype(str),"y":y[good].to_numpy(),"log_rt":np.log(rt[good].to_numpy())}); rows=[]
    for pid,z in d.groupby(person,sort=False):
        cor=np.corrcoef(z.y,z.log_rt)[0,1] if len(z)>=4 and z.y.std(ddof=1)>0 and z.log_rt.std(ddof=1)>0 else np.nan; rows.append({"person_id":str(pid),"n":len(z),"accuracy":float(z.y.mean()),"mean_log_rt":float(z.log_rt.mean()),"within_person_correlation":float(cor)})
    pooled=np.corrcoef(d.y,d.log_rt)[0,1] if len(d)>=4 and d.y.std(ddof=1)>0 else np.nan
    return _result("eye_speed_accuracy_profile",person=pd.DataFrame(rows),pooled_correlation=float(pooled),guardrail="Speed-accuracy associations are descriptive and may differ across levels of aggregation.")


def _group_channel_profile(data: Any,id_col: str,channels: Sequence[str],out_id: str) -> pd.DataFrame:
    df=_as_df(data,"data"); _req_cols(df,[id_col]+list(channels),"data"); rows=[]
    for ident,z in df.groupby(id_col,sort=False):
        row={out_id:str(ident),"n":len(z)}
        for ch in channels: row[f"mean_{ch}"]=_finite_mean(pd.to_numeric(z[ch],errors="coerce"))
        rows.append(row)
    return pd.DataFrame(rows)


def eyeprocess_process_item_profile(data: Any,item: str,channels: Sequence[str]) -> pd.DataFrame: return _group_channel_profile(data,item,channels,"item_id")
def eyeprocess_process_person_profile(data: Any,person: str,channels: Sequence[str]) -> pd.DataFrame: return _group_channel_profile(data,person,channels,"person_id")


def eyeprocess_irt_process_alignment(item_parameters: Any, process_profile: Any, process_columns: Sequence[str] | None = None) -> EyeResult:
    items = _item_pars(item_parameters)
    pp = _as_df(process_profile, "process_profile")
    _req_cols(pp, ["item_id"], "process_profile")
    pp = pp.copy()
    pp["item_id"] = pp["item_id"].astype(str)
    z = items.merge(pp, on="item_id", how="left")
    if process_columns is None:
        process_columns = [c for c in pp.columns if c not in {"item_id", "n"}]
    rows: list[dict[str, Any]] = []
    for ch in process_columns:
        if ch not in z.columns:
            continue
        v = pd.to_numeric(z[ch], errors="coerce").to_numpy(dtype=float)
        a = pd.to_numeric(z["a"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(z["b"], errors="coerce").to_numpy(dtype=float)
        keep_a = np.isfinite(v) & np.isfinite(a)
        keep_b = np.isfinite(v) & np.isfinite(b)
        ca = np.nan
        cb = np.nan
        if keep_a.sum() >= 3 and np.std(v[keep_a], ddof=1) > 0 and np.std(a[keep_a], ddof=1) > 0:
            ca = float(np.corrcoef(v[keep_a], a[keep_a])[0, 1])
        if keep_b.sum() >= 3 and np.std(v[keep_b], ddof=1) > 0 and np.std(b[keep_b], ddof=1) > 0:
            cb = float(np.corrcoef(v[keep_b], b[keep_b])[0, 1])
        rows.append({"channel": str(ch), "correlation_discrimination": ca, "correlation_difficulty": cb})
    return _result(
        "eye_irt_process_alignment",
        table=z,
        correlations=pd.DataFrame(rows, columns=["channel", "correlation_discrimination", "correlation_difficulty"]),
        guardrail="Alignment is descriptive response-process evidence and does not identify cognitive mechanisms.",
    )


def eyeprocess_process_missingness_pattern(data: Any, response: str, channels: Sequence[str]) -> pd.DataFrame:
    df = _as_df(data, "data")
    channels = list(map(str, channels))
    _req_cols(df, [response, *channels], "data")
    miss = pd.DataFrame({"response_missing": df[response].isna()})
    for ch in channels:
        miss[f"{ch}_missing"] = df[ch].isna()
    if len(df) == 0:
        return pd.DataFrame(columns=["pattern", "n", "fraction"])
    pattern = miss.astype(int).astype(str).agg("".join, axis=1)
    counts = pattern.value_counts(sort=True)
    return pd.DataFrame({"pattern": counts.index.astype(str), "n": counts.to_numpy(dtype=int), "fraction": counts.to_numpy(dtype=float) / len(df)})


def eyeprocess_multichannel_measurement_map(
    response: str = "accuracy",
    channels: Sequence[str] = ("response_time", "dwell", "pupil", "transitions"),
    role: Sequence[str] | None = None,
) -> pd.DataFrame:
    # R uses unique(as.character(channels)), preserving first occurrence order.
    channels = list(dict.fromkeys(map(str, channels)))
    roles = ["response_process_measurement"] * len(channels) if role is None else list(map(str, role))
    if len(roles) != len(channels):
        raise EyeProcessValidationError("role must match channels.")
    return pd.DataFrame(
        {
            "channel": [str(response), *channels],
            "role": ["item_response", *roles],
            "inference_boundary": [
                "psychometric response",
                *(["measurement channel; not a direct mental-state label"] * len(channels)),
            ],
        }
    )


# ---------------------------------------------------------------------------
# 088: explicit external-engine adapters
# ---------------------------------------------------------------------------

_ENGINE_ROWS = [
    ("mirt", "multidimensional/polytomous/testlet/multiple-group/mixed IRT", "mirt"),
    ("TAM", "Rasch/2PL/3PL/GPCM/latent regression/plausible values", "TAM"),
    ("GDINA", "cognitive diagnosis and Q-matrix workflows", "GDINA"),
    ("LNIRT", "joint response and lognormal response-time IRT", "LNIRT"),
    ("eRm", "conditional Rasch/PCM/LLTM diagnostics", "eRm"),
    ("equateIRT", "IRT linking/equating and transformation stability", "equateIRT"),
    ("catR", "unidimensional adaptive-testing simulation", "catR"),
    ("mirtCAT", "multidimensional CAT and constrained/shadow testing", "mirtCAT"),
]


def eyeprocess_irt_engine_registry() -> pd.DataFrame:
    out = pd.DataFrame(_ENGINE_ROWS, columns=["engine", "capability", "package"])
    # The frozen registry asks whether the corresponding R namespace is installed.
    # The Python parity layer never treats similarly named Python packages as the same engine.
    out["available"] = False
    return _tag(out, "eye_irt_engine_registry")


def eyeprocess_irt_engine_status(engine: str) -> pd.DataFrame:
    reg = eyeprocess_irt_engine_registry()
    engine = str(engine)
    if engine not in set(reg["engine"]):
        raise EyeProcessValidationError("Unknown engine. Available: " + ", ".join(reg["engine"].tolist()))
    return reg.loc[reg["engine"] == engine].reset_index(drop=True)


def _gated_engine(engine: str, call: str, reason: str | None = None) -> EyeResult:
    return _result(
        "eye_gated_irt_engine",
        status="gated",
        engine=engine,
        call=call,
        fit=None,
        reason=reason or f"Optional engine '{engine}' is not installed.",
    )


def _require_engine(engine: str, expected: str, function_name: str) -> None:
    if engine != expected:
        raise EyeProcessValidationError(f"{function_name}() only accepts engine='{expected}'.")


def fit_eyeprocess_mirt(data: Any, model: Any = 1, itemtype: str = "2PL", engine: str = "mirt", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "mirt", "fit_eyeprocess_mirt")
    return _gated_engine("mirt", "fit_eyeprocess_mirt")


def fit_eyeprocess_tam(resp: Any, model: str = "rasch", engine: str = "TAM", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "TAM", "fit_eyeprocess_tam")
    if model not in {"rasch", "2pl", "gpcm"}:
        raise EyeProcessValidationError("model must be one of 'rasch', '2pl', or 'gpcm'.")
    return _gated_engine("TAM", "fit_eyeprocess_tam")


def fit_eyeprocess_gdina(dat: Any, Q: Any, model: str = "GDINA", engine: str = "GDINA", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "GDINA", "fit_eyeprocess_gdina")
    return _gated_engine("GDINA", "fit_eyeprocess_gdina")


def fit_eyeprocess_lnirt(Y: Any, RT: Any, quadratic: bool = False, engine: str = "LNIRT", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "LNIRT", "fit_eyeprocess_lnirt")
    # R validates positivity only after confirming the optional engine exists. In the
    # Python parity core the exact R engine is unavailable, so the gated result wins.
    return _gated_engine("LNIRT", "fit_eyeprocess_lnirt")


def fit_eyeprocess_erm(data: Any, model: str = "RM", engine: str = "eRm", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "eRm", "fit_eyeprocess_erm")
    if model not in {"RM", "PCM"}:
        raise EyeProcessValidationError("model must be one of 'RM' or 'PCM'.")
    return _gated_engine("eRm", "fit_eyeprocess_erm")


def simulate_eyeprocess_catr(itemBank: Any, trueTheta: float = 0, engine: str = "catR", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "catR", "simulate_eyeprocess_catr")
    return _gated_engine("catR", "simulate_eyeprocess_catr")


def run_eyeprocess_equateirt(function_name: str, engine: str = "equateIRT", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "equateIRT", "run_eyeprocess_equateirt")
    function_name = str(function_name)
    if not function_name:
        raise EyeProcessValidationError("function_name must be a non-empty scalar.")
    return _gated_engine("equateIRT", "run_eyeprocess_equateirt")


def run_eyeprocess_mirtcat(engine: str = "mirtCAT", **kwargs: Any) -> EyeResult:
    _require_engine(engine, "mirtCAT", "run_eyeprocess_mirtcat")
    return _gated_engine("mirtCAT", "run_eyeprocess_mirtcat")


def validate_eyeprocess_external_irt_fit(x: Any, engine: str | None = None) -> bool:
    cls = getattr(x, "eyeprocess_class", None)
    if not isinstance(x, Mapping) or cls not in {"eye_external_irt_fit", "eye_gated_irt_engine"}:
        raise EyeProcessValidationError("x is not an eyeprocess external IRT result.")
    if engine is not None and x.get("engine") != engine:
        raise EyeProcessValidationError("Engine mismatch.")
    if cls == "eye_external_irt_fit" and x.get("fit") is None:
        raise EyeProcessValidationError("Fitted external IRT result has NULL fit.")
    if cls == "eye_gated_irt_engine" and x.get("fit") is not None:
        raise EyeProcessValidationError("Gated external IRT result must have NULL fit.")
    return True


# ---------------------------------------------------------------------------
# 089: simulation/recovery/SBC evidence
# ---------------------------------------------------------------------------

def simulate_eyeprocess_irt_binary(n_persons: int=500,items: Any=None,theta: Any=None,missing_rate: float=0,testlet_sd: float=0,seed: int=1,D: float=1) -> EyeResult:
    n_persons=int(n_persons); seed=int(seed); df=_item_pars(items)
    if n_persons<1: raise EyeProcessValidationError("n_persons must be positive.")
    missing_rate=float(missing_rate); testlet_sd=float(testlet_sd)
    if not (0<=missing_rate<1) or testlet_sd<0: raise EyeProcessValidationError("missing_rate must lie in [0,1) and testlet_sd be non-negative.")
    rng=np.random.default_rng(seed); th=rng.normal(size=n_persons) if theta is None else np.asarray(theta,dtype=float)
    if th.size!=n_persons or not np.all(np.isfinite(th)): raise EyeProcessValidationError("theta must match n_persons and be finite.")
    local=rng.normal(0,testlet_sd,size=n_persons) if testlet_sd>0 else np.zeros(n_persons); y=np.empty((n_persons,len(df)),dtype=float); probs=np.empty_like(y)
    for j,r in enumerate(df.itertuples(index=False)):
        p=eyeprocess_irt_4pl_probability(th+local,r.a,r.b,r.c,r.d,D); probs[:,j]=p; y[:,j]=rng.binomial(1,p)
    if missing_rate>0: y[rng.random(y.shape)<missing_rate]=np.nan
    return _result("eye_irt_simulation",responses=pd.DataFrame(y,columns=df.item_id),probabilities=pd.DataFrame(probs,columns=df.item_id),theta=th,items=df,missing_rate=missing_rate,testlet_sd=testlet_sd,seed=seed)


def eyeprocess_irt_recovery_design(
    sample_size: Sequence[int] = (250, 750),
    n_items: Sequence[int] = (12, 24),
    missing_rate: Sequence[float] = (0, .15),
    testlet_sd: Sequence[float] = (0, .35),
    replications: int = 10,
    seed: int = 20260811,
) -> pd.DataFrame:
    ns = np.asarray(sample_size, dtype=int)
    ni = np.asarray(n_items, dtype=int)
    mr = np.asarray(missing_rate, dtype=float)
    ts = np.asarray(testlet_sd, dtype=float)
    replications = int(replications)
    seed = int(seed)
    if (
        ns.size == 0 or np.any(ns < 20) or ni.size == 0 or np.any(ni < 4)
        or mr.size == 0 or np.any(~np.isfinite(mr)) or np.any((mr < 0) | (mr >= 1))
        or ts.size == 0 or np.any(~np.isfinite(ts)) or np.any(ts < 0)
        or replications < 1 or seed < 1
    ):
        raise EyeProcessValidationError("invalid recovery design.")
    rows: list[dict[str, Any]] = []
    sid = 0
    # R expand.grid() varies the first factor fastest.
    for testlet in ts:
        for missing in mr:
            for nitem in ni:
                for n in ns:
                    sid += 1
                    rows.append(
                        {
                            "scenario_id": f"IRTREC{sid:03d}",
                            "sample_size": int(n),
                            "n_items": int(nitem),
                            "missing_rate": float(missing),
                            "testlet_sd": float(testlet),
                            "replications": replications,
                            "seed": seed,
                        }
                    )
    return _tag(pd.DataFrame(rows), "eye_irt_recovery_design")


def run_eyeprocess_irt_recovery(design: Any, engine: str = "mirt", verbose: bool = True) -> EyeResult:
    if not isinstance(design, pd.DataFrame) or design.attrs.get("eyeprocess_class") != "eye_irt_recovery_design":
        raise EyeProcessValidationError("design must come from eyeprocess_irt_recovery_design().")
    if engine != "mirt":
        raise EyeProcessValidationError("Milestone #2 recovery currently requires engine='mirt'.")
    return _gated_engine("mirt", "run_eyeprocess_irt_recovery", "mirt is required for IRT parameter-recovery evidence.")


def eyeprocess_irt_recovery_summary(x: Any) -> pd.DataFrame:
    if not isinstance(x, Mapping) or getattr(x, "eyeprocess_class", None) != "eye_irt_recovery_result":
        raise EyeProcessValidationError("x must be an eye_irt_recovery_result.")
    d = _as_df(x.get("estimates"), "estimates")
    if d.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sid, z in d.groupby("scenario_id", sort=False):
        for p in ("a", "b"):
            truth = pd.to_numeric(z[f"{p}_truth"], errors="coerce").to_numpy(dtype=float)
            est = pd.to_numeric(z[f"{p}_estimate"], errors="coerce").to_numpy(dtype=float)
            err = est - truth
            keep = np.isfinite(err)
            pair = np.isfinite(truth) & np.isfinite(est)
            rows.append(
                {
                    "scenario_id": sid,
                    "parameter": p,
                    "n": int(keep.sum()),
                    "bias": _finite_mean(err),
                    "rmse": float(np.sqrt(_finite_mean(err ** 2))),
                    "mae": _finite_mean(np.abs(err)),
                    "correlation": float(np.corrcoef(truth[pair], est[pair])[0, 1]) if pair.sum() >= 3 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def eyeprocess_irt_recovery_failures(x: Any) -> pd.DataFrame:
    if not isinstance(x, Mapping) or getattr(x, "eyeprocess_class", None) != "eye_irt_recovery_result":
        raise EyeProcessValidationError("x must be an eye_irt_recovery_result.")
    des = _as_df(x.get("design"), "design")
    fail = _as_df(x.get("failures"), "failures")
    rows = []
    for sc in des.itertuples(index=False):
        nf = int((fail.get("scenario_id", pd.Series(dtype=str)).astype(str) == str(sc.scenario_id)).sum()) if not fail.empty else 0
        rows.append({"scenario_id": sc.scenario_id, "attempted": int(sc.replications), "failures": nf, "failure_rate": nf / int(sc.replications)})
    return pd.DataFrame(rows)


def eyeprocess_irt_sbc_ranks(truth: Any, draws: Any, randomize_ties: bool = True, seed: int = 1) -> np.ndarray:
    truth = np.atleast_1d(np.asarray(truth, dtype=float))
    d = np.asarray(draws, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[0] != truth.size or np.any(~np.isfinite(truth)) or np.any(~np.isfinite(d)):
        raise EyeProcessValidationError("draws rows must match finite truths and contain finite draws.")
    seed = int(seed)
    if seed < 1 or d.shape[1] < 1:
        raise EyeProcessValidationError("seed must be positive and draws must contain at least one posterior draw per truth.")
    rng = np.random.default_rng(seed)
    ranks = []
    for t, row in zip(truth, d):
        less = int(np.sum(row < t))
        equal = int(np.sum(row == t))
        ranks.append(less + (int(rng.integers(0, equal + 1)) if randomize_ties and equal else 0))
    return np.asarray(ranks, dtype=int)


def _sbc_rank_diagnostics(ranks: Any, n_draws: int, bins: int | None = None) -> EyeResult:
    r0 = np.asarray(ranks, dtype=float)
    n_draws = int(n_draws)
    if n_draws < 1:
        raise EyeProcessValidationError("n_draws must be a positive integer.")
    finite = np.isfinite(r0)
    if np.any(finite & (r0 != np.round(r0))):
        raise EyeProcessValidationError("ranks must be integer-valued.")
    r = r0[finite].astype(int)
    if np.any((r < 0) | (r > n_draws)):
        raise EyeProcessValidationError("ranks must lie between 0 and n_draws inclusive.")
    if bins is None:
        bins = max(5, min(20, round(math.sqrt(max(1, len(r))))))
    bins = int(bins)
    if bins < 2:
        raise EyeProcessValidationError("bins must be an integer >= 2.")
    bins = min(bins, n_draws + 1)
    breaks = np.linspace(-0.5, n_draws + 0.5, bins + 1)
    counts, _ = np.histogram(r, bins=breaks)
    support, _ = np.histogram(np.arange(n_draws + 1), bins=breaks)
    expected = len(r) * support / (n_draws + 1)
    ok = expected > 0
    chisq = float(np.sum((counts[ok] - expected[ok]) ** 2 / expected[ok])) if np.any(ok) else np.nan
    df = int(ok.sum() - 1)
    p = float(chi2.sf(chisq, df)) if np.isfinite(chisq) and df > 0 else np.nan
    u = np.sort((r + 0.5) / (n_draws + 1))
    empirical = np.arange(1, len(u) + 1) / len(u) if len(u) else np.array([])
    ecdf = float(np.max(np.abs(empirical - u))) if len(u) else np.nan
    return _result(
        "eye_sbc_diagnostics",
        ranks=r,
        n_draws=n_draws,
        bins=bins,
        counts=counts,
        breaks=breaks,
        expected_count=expected,
        chi_square=chisq,
        chi_square_p=p,
        ecdf_max_deviation=ecdf,
        caveat="SBC diagnoses calibration of the supplied simulator-model-inference workflow; it does not establish substantive model adequacy for empirical data.",
    )


def eyeprocess_irt_sbc_summary(ranks: Any, n_draws: int, bins: int | None = None) -> EyeResult:
    if bins is None:
        bins = min(int(n_draws) + 1, 20)
    diag = _sbc_rank_diagnostics(ranks, n_draws=n_draws, bins=bins)
    return _result(
        "eye_irt_sbc_evidence",
        diagnostics=diag,
        ecdf_deviation=diag.ecdf_max_deviation,
        n=len(np.asarray(ranks).ravel()),
        n_draws=int(n_draws),
    )


def run_eyeprocess_irt_ability_sbc(
    items: Any,
    replications: int = 200,
    posterior_draws: int = 99,
    theta_grid: Any = np.linspace(-5, 5, 401),
    prior_mean: float = 0,
    prior_sd: float = 1,
    interval: float = .95,
    seed: int = 20260811,
    D: float = 1,
) -> EyeResult:
    df = _item_pars(items)
    replications = int(replications)
    posterior_draws = int(posterior_draws)
    seed = int(seed)
    grid = np.asarray(theta_grid, dtype=float)
    prior_mean = float(prior_mean)
    prior_sd = float(prior_sd)
    interval = float(interval)
    if replications < 20:
        raise EyeProcessValidationError("replications must be a scalar integer >= 20.")
    if posterior_draws < 9:
        raise EyeProcessValidationError("posterior_draws must be a scalar integer >= 9.")
    if seed < 1:
        raise EyeProcessValidationError("seed must be a positive scalar integer.")
    if grid.size < 101 or np.any(np.diff(grid) <= 0):
        raise EyeProcessValidationError("theta_grid must be strictly increasing with at least 101 points.")
    if not np.isfinite(prior_mean) or not np.isfinite(prior_sd) or prior_sd <= 0:
        raise EyeProcessValidationError("invalid normal prior.")
    if not 0 < interval < 1:
        raise EyeProcessValidationError("interval must lie in (0,1).")
    rng = np.random.default_rng(seed)
    alpha = (1 - interval) / 2
    ranks: list[int] = []
    rows: list[dict[str, Any]] = []
    for i in range(1, replications + 1):
        truth = float(rng.normal(prior_mean, prior_sd))
        probs = np.array([eyeprocess_irt_4pl_probability([truth], r.a, r.b, r.c, r.d, D)[0] for r in df.itertuples(index=False)])
        response = rng.binomial(1, probs)
        score = eyeprocess_irt_eap_score(response, df, theta_grid=grid, prior_mean=prior_mean, prior_sd=prior_sd, D=D)
        sample = rng.choice(score.theta, size=posterior_draws, replace=True, p=score.posterior)
        ranks.append(int(np.sum(sample < truth)))
        cdf = np.cumsum(score.posterior)
        lo_idx = int(np.argmax(cdf >= alpha))
        hi_idx = int(np.argmax(cdf >= 1 - alpha))
        lower = float(grid[lo_idx])
        upper = float(grid[hi_idx])
        rows.append(
            {
                "replication": i,
                "truth": truth,
                "estimate": float(score.estimate),
                "se": float(score.se),
                "lower": lower,
                "upper": upper,
                "covered": bool(lower <= truth <= upper),
                "raw_score": int(np.sum(response)),
            }
        )
    details = pd.DataFrame(rows)
    out = eyeprocess_irt_sbc_summary(ranks, n_draws=posterior_draws)
    out["coverage"] = float(details["covered"].mean())
    out["nominal_coverage"] = interval
    out["coverage_error"] = out["coverage"] - interval
    out["details"] = details
    out["seed"] = seed
    out["method"] = "known-item grid-posterior ability SBC"
    out["guardrail"] = "SBC validates computational calibration under the declared generative model; it does not establish construct validity or empirical model adequacy."
    return out


def eyeprocess_irt_misspecification_suite() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario": ["reference", "local_dependence", "missingness", "discrimination_heterogeneity", "lower_asymptote", "latent_mixture"],
            "perturbation": ["none", "testlet random effect", "MCAR omission", "wider log-discrimination", "non-zero lower asymptote", "two-component theta mixture"],
            "target": ["calibration baseline", "local independence", "missing-data robustness", "item heterogeneity", "guessing sensitivity", "latent distribution sensitivity"],
        }
    )


def eyeprocess_irt_misspecification_metrics(reference_summary: Any, misspecified_summary: Any) -> pd.DataFrame:
    r = _as_df(reference_summary, "reference_summary")
    m = _as_df(misspecified_summary, "misspecified_summary")
    _req_cols(r, ["parameter", "bias", "rmse"], "reference_summary")
    _req_cols(m, ["parameter", "bias", "rmse"], "misspecified_summary")
    z = r[["parameter", "bias", "rmse"]].merge(m[["parameter", "bias", "rmse"]], on="parameter", suffixes=("_reference", "_misspecified"))
    z["rmse_inflation"] = z["rmse_misspecified"] - z["rmse_reference"]
    z["absolute_bias_inflation"] = np.abs(z["bias_misspecified"]) - np.abs(z["bias_reference"])
    return z


def freeze_eyeprocess_irt_reference(
    recovery_summary: Any = None,
    sbc: Any = None,
    failures: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> EyeResult:
    obj = _result(
        "eye_irt_validation_reference",
        recovery_summary=recovery_summary,
        sbc=sbc,
        failures=failures,
        metadata=dict(metadata or {}),
        scientific_scope="software-validation reference; not construct-validity evidence",
    )
    obj["hash"] = _stable_hash(dict(obj))
    return obj


# ---------------------------------------------------------------------------
# 090: multidimensional/CDM
# ---------------------------------------------------------------------------

def eyeprocess_mirt_loading_spec(items: Sequence[str], loadings: Any, dimension_names: Sequence[str] | None = None, simple_structure: bool = False) -> EyeResult:
    item_names = list(map(str, items))
    L = np.asarray(loadings, dtype=float)
    if L.ndim != 2 or len(item_names) != L.shape[0] or L.shape[1] == 0 or not np.all(np.isfinite(L)):
        raise EyeProcessValidationError("loadings must be a finite item-by-dimension matrix matching items.")
    dims = list(map(str, dimension_names)) if dimension_names is not None else [f"D{i+1}" for i in range(L.shape[1])]
    if len(dims) != L.shape[1] or any(not d for d in dims) or len(set(dims)) != len(dims):
        raise EyeProcessValidationError("invalid dimension_names.")
    violations = [item_names[i] for i, n in enumerate(np.sum(np.abs(L) > 0, axis=1)) if simple_structure and n > 1]
    return _result("eye_mirt_loading_spec", items=item_names, loadings=L, dimensions=dims, simple_structure=bool(simple_structure), violations=violations)


def eyeprocess_mirt_loading_audit(spec: Any, min_items_per_dimension: int = 3) -> pd.DataFrame:
    if not isinstance(spec, Mapping) or getattr(spec, "eyeprocess_class", None) != "eye_mirt_loading_spec":
        raise EyeProcessValidationError("spec must be created by eyeprocess_mirt_loading_spec().")
    minimum = int(min_items_per_dimension)
    if minimum < 1:
        raise EyeProcessValidationError("minimum must be a positive scalar integer.")
    counts = np.sum(np.abs(np.asarray(spec["loadings"], dtype=float)) > 0, axis=0)
    return pd.DataFrame({"dimension": spec["dimensions"], "n_loading_items": counts.astype(int), "meets_minimum": counts >= minimum})


def eyeprocess_mirt_directional_information(theta: Any, discrimination: Any, difficulty: float = 0, direction: Any = None, D: float = 1) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    a = np.asarray(discrimination, dtype=float).reshape(-1)
    if th.size != a.size or th.size == 0 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(a)):
        raise EyeProcessValidationError("theta and discrimination must be finite vectors of equal dimension.")
    u = a.copy() if direction is None else np.asarray(direction, dtype=float).reshape(-1)
    if u.size != th.size or not np.all(np.isfinite(u)) or np.sum(u * u) == 0:
        raise EyeProcessValidationError("invalid direction.")
    u = u / np.linalg.norm(u)
    p = float(expit(float(D) * (float(a @ th) - float(difficulty))))
    info = (float(D) ** 2) * p * (1 - p) * np.outer(a, a)
    return float(u @ info @ u)


def eyeprocess_mirt_information_matrix(theta: Any, discrimination: Any, difficulty: float = 0, D: float = 1) -> np.ndarray:
    th = np.asarray(theta, dtype=float).reshape(-1)
    a = np.asarray(discrimination, dtype=float).reshape(-1)
    if th.size != a.size or not np.all(np.isfinite(th)) or not np.all(np.isfinite(a)):
        raise EyeProcessValidationError("theta/discrimination dimension mismatch.")
    p = float(expit(float(D) * (float(a @ th) - float(difficulty))))
    return (float(D) ** 2) * p * (1 - p) * np.outer(a, a)


def eyeprocess_irt_testlet_spec(item_id: Sequence[str], testlet: Sequence[str], general_dimension: str = "general") -> pd.DataFrame:
    ids = list(map(str, item_id)); groups = list(map(str, testlet))
    if len(ids) != len(groups) or not ids or len(set(ids)) != len(ids) or any(not x for x in ids + groups):
        raise EyeProcessValidationError("item_id/testlet must be non-missing equal-length vectors with unique items.")
    return _tag(pd.DataFrame({"item_id": ids, "testlet": groups, "general_dimension": str(general_dimension)}), "eye_irt_testlet_spec")


def eyeprocess_irt_testlet_audit(spec: Any, min_items: int = 2) -> pd.DataFrame:
    df = _as_df(spec, "spec")
    if getattr(spec, "attrs", {}).get("eyeprocess_class") != "eye_irt_testlet_spec":
        raise EyeProcessValidationError("spec must be an eye_irt_testlet_spec.")
    minimum = int(min_items)
    if minimum < 1:
        raise EyeProcessValidationError("min_items must be a positive scalar integer.")
    tab = df["testlet"].astype(str).value_counts(sort=False)
    return pd.DataFrame({"testlet": tab.index, "n_items": tab.to_numpy(dtype=int), "singleton": tab.to_numpy() == 1, "meets_minimum": tab.to_numpy() >= minimum})


def eyeprocess_irt_latent_regression_design(data: Any, formula: str, center_numeric: bool = True) -> EyeResult:
    df = _as_df(data, "data")
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    terms = [t.strip() for t in rhs.split("+") if t.strip() and t.strip() != "1"]
    mm = pd.DataFrame(index=df.index)
    if "-1" not in rhs and "0" not in [t.strip() for t in rhs.split("+")]:
        mm["(Intercept)"] = 1.0
    for term in terms:
        if term in {"0", "-1"}: continue
        if ":" in term:
            parts = [x.strip() for x in term.split(":")]
            vals = np.ones(len(df), dtype=float)
            for part in parts:
                if part not in df: raise EyeProcessValidationError(f"formula references missing column: {part}")
                vals *= pd.to_numeric(df[part], errors="coerce").to_numpy(dtype=float)
            mm[term] = vals
        elif term not in df:
            raise EyeProcessValidationError(f"formula references missing column: {term}")
        elif pd.api.types.is_numeric_dtype(df[term]):
            mm[term] = pd.to_numeric(df[term], errors="coerce").astype(float)
        else:
            mm = pd.concat([mm, pd.get_dummies(df[term], prefix=term, drop_first=True, dtype=float)], axis=1)
    centers = {c: 0.0 for c in mm.columns}
    if center_numeric:
        for c in mm.columns:
            if c == "(Intercept)": continue
            v = pd.to_numeric(mm[c], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(v)
            centers[c] = float(np.mean(v[finite])) if finite.any() else np.nan
            if np.isfinite(centers[c]): mm[c] = v - centers[c]
    complete = np.isfinite(mm.to_numpy(dtype=float)).all(axis=1)
    return _result("eye_irt_latent_regression_design", matrix=mm, formula=formula, centers=centers, complete=complete)


def eyeprocess_cdm_qmatrix_audit(Q: Any, item_ids: Sequence[str] | None = None, attribute_names: Sequence[str] | None = None) -> EyeResult:
    q = np.asarray(Q, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0 or np.any(~np.isin(q, [0, 1])):
        raise EyeProcessValidationError("Q must be a non-empty binary matrix.")
    ids = list(map(str, item_ids)) if item_ids is not None else [f"item_{i+1}" for i in range(q.shape[0])]
    attrs = list(map(str, attribute_names)) if attribute_names is not None else [f"A{i+1}" for i in range(q.shape[1])]
    if len(ids) != q.shape[0] or len(attrs) != q.shape[1]:
        raise EyeProcessValidationError("identifier lengths mismatch Q dimensions.")
    row_load = q.sum(axis=1).astype(int); attr_load = q.sum(axis=0).astype(int)
    duplicate_rows = np.array([np.sum(np.all(q == row, axis=1)) > 1 for row in q], dtype=bool)
    identity_ok = all(any(np.array_equal(row, np.eye(q.shape[1], dtype=float)[k]) for row in q) for k in range(q.shape[1]))
    return _result(
        "eye_cdm_qmatrix_audit",
        item=pd.DataFrame({"item_id": ids, "n_attributes": row_load, "no_attribute": row_load == 0}),
        attribute=pd.DataFrame({"attribute": attrs, "n_items": attr_load, "unmeasured": attr_load == 0}),
        duplicate_rows=duplicate_rows,
        complete_identity_block=bool(identity_ok),
    )


def eyeprocess_cdm_attribute_profiles(n_attributes: int, attribute_names: Sequence[str] | None = None) -> pd.DataFrame:
    n = int(n_attributes)
    if n < 1 or n > 20:
        raise EyeProcessValidationError("n_attributes must be a scalar integer between 1 and 20.")
    names = list(map(str, attribute_names)) if attribute_names is not None else [f"A{i+1}" for i in range(n)]
    if len(names) != n or any(not x for x in names) or len(set(names)) != len(names):
        raise EyeProcessValidationError("attribute_names must be unique, non-missing, non-empty, and match n_attributes.")
    # R expand.grid changes the first factor fastest.
    vals = np.array([[int((i >> j) & 1) for j in range(n)] for i in range(2 ** n)], dtype=int)
    out = pd.DataFrame(vals, columns=names)
    out.insert(0, "profile_id", np.arange(1, len(out) + 1))
    return out


def eyeprocess_cdm_dina_ideal_response(Q: Any, profiles: Any) -> np.ndarray:
    q = np.asarray(Q, dtype=int); p = np.asarray(profiles, dtype=int)
    if q.ndim != 2 or p.ndim != 2 or q.shape[1] != p.shape[1] or np.any(~np.isin(q,[0,1])) or np.any(~np.isin(p,[0,1])):
        raise EyeProcessValidationError("Q and profiles must be compatible binary matrices.")
    out = np.empty((p.shape[0], q.shape[0]), dtype=int)
    for j, req in enumerate(q): out[:, j] = np.sum(p * req, axis=1) == int(np.sum(req))
    return out


def eyeprocess_cdm_dina_probability(ideal_response: Any, slip: Any = 0.1, guess: Any = 0.2) -> np.ndarray:
    eta = np.asarray(ideal_response, dtype=float)
    if eta.ndim != 2 or np.any(~np.isin(eta,[0,1])): raise EyeProcessValidationError("ideal_response must be binary.")
    s = np.atleast_1d(np.asarray(slip, dtype=float)); g = np.atleast_1d(np.asarray(guess, dtype=float))
    if s.size == 1: s = np.repeat(s, eta.shape[1])
    if g.size == 1: g = np.repeat(g, eta.shape[1])
    if s.size != eta.shape[1] or g.size != eta.shape[1] or not np.all(np.isfinite(s)) or not np.all(np.isfinite(g)) or np.any((s<0)|(s>=1)) or np.any((g<0)|(g>=1)):
        raise EyeProcessValidationError("invalid slip/guess.")
    return np.where(eta == 1, 1 - s[None,:], g[None,:])


def eyeprocess_cdm_classification_uncertainty(profile_probabilities: Any) -> pd.DataFrame:
    p = np.asarray(profile_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2 or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(np.abs(p.sum(axis=1)-1) > 1e-6):
        raise EyeProcessValidationError("profile probabilities must have at least two columns, be non-negative, finite, and row-normalized.")
    entropy = -np.sum(np.where(p > 0, p*np.log(p), 0.0), axis=1)
    return pd.DataFrame({"case": np.arange(1,p.shape[0]+1), "max_probability": p.max(axis=1), "entropy": entropy, "normalized_entropy": entropy/np.log(p.shape[1])})


# ---------------------------------------------------------------------------
# 093: advanced diagnostics/governance
# ---------------------------------------------------------------------------

def eyeprocess_irt_infit_outfit(observed: Any, expected: Any, by: str = "item", min_variance: float = 1e-8) -> pd.DataFrame:
    y = _binary_matrix(observed, "observed")
    p = _prob_matrix(expected, y.shape, "expected")
    min_variance = _scalar(min_variance, "min_variance", positive=True)
    if by not in {"item", "person"}: raise EyeProcessValidationError("by must be 'item' or 'person'.")
    v = p * (1-p); missing = np.isnan(y) | np.isnan(p); v = v.copy(); v[missing] = np.nan
    r2 = (y-p)**2; z2 = r2 / np.maximum(v, min_variance)
    if by == "item":
        n = np.sum(np.isfinite(y)&np.isfinite(p), axis=0); units = list(observed.columns.astype(str)) if isinstance(observed,pd.DataFrame) else [f"item_{i+1}" for i in range(y.shape[1])]
        infit_num=np.nansum(r2,axis=0); infit_den=np.nansum(v,axis=0)
        outfit=np.array([np.nanmean(z2[:,j]) if n[j]>0 else np.nan for j in range(y.shape[1])])
    else:
        n = np.sum(np.isfinite(y)&np.isfinite(p), axis=1); units = list(observed.index.astype(str)) if isinstance(observed,pd.DataFrame) else [f"person_{i+1}" for i in range(y.shape[0])]
        infit_num=np.nansum(r2,axis=1); infit_den=np.nansum(v,axis=1)
        outfit=np.array([np.nanmean(z2[i,:]) if n[i]>0 else np.nan for i in range(y.shape[0])])
    infit=infit_num/np.maximum(infit_den,min_variance); infit=np.where(n==0,np.nan,infit)
    return pd.DataFrame({"unit":units,"n":n.astype(int),"infit":infit.astype(float),"outfit":outfit.astype(float)})


def eyeprocess_irt_person_fit_lz(observed: Any, expected: Any, min_probability: float = 1e-8) -> pd.DataFrame:
    y=_binary_matrix(observed,"observed"); p=_prob_matrix(expected,y.shape,"expected").copy(); mp=_scalar(min_probability,"min_probability",positive=True)
    if mp>=.5: raise EyeProcessValidationError("min_probability must lie in (0, 0.5).")
    p=np.clip(p,mp,1-mp); p[np.isnan(y)]=np.nan; logp1=np.log(p); logp0=np.log1p(-p)
    ll=np.nansum(y*logp1+(1-y)*logp0,axis=1); mu=np.nansum(p*logp1+(1-p)*logp0,axis=1); var=np.nansum(p*(1-p)*(logp1-logp0)**2,axis=1); lz=(ll-mu)/np.sqrt(np.maximum(var,_EPS)); n=np.sum(np.isfinite(y)&np.isfinite(p),axis=1); lz=np.where(n==0,np.nan,lz)
    units=list(observed.index.astype(str)) if isinstance(observed,pd.DataFrame) else [f"person_{i+1}" for i in range(y.shape[0])]
    return pd.DataFrame({"person":units,"n_observed":n.astype(int),"log_likelihood":ll,"expected_log_likelihood":mu,"lz":lz})


def eyeprocess_irt_bank_coverage(items: Any, theta: Any = np.linspace(-4,4,161), target_information: float = 5, target: Sequence[float] = (-2,2)) -> EyeResult:
    th=_theta(theta); ti=_scalar(target_information,"target_information",nonnegative=True); target=np.asarray(target,dtype=float)
    if target.size!=2 or not np.all(np.isfinite(target)) or target[0]>=target[1]: raise EyeProcessValidationError("target must be an increasing finite length-2 vector.")
    curve=eyeprocess_irt_test_information(th,items); keep=(curve.theta>=target[0])&(curve.theta<=target[1])
    if not keep.any(): raise EyeProcessValidationError("theta grid does not overlap the target region.")
    gaps=curve.loc[keep & (curve.information<ti)].copy()
    return _result("eye_irt_bank_coverage",curve=curve,target=target,target_information=ti,fraction_target_met=float(np.mean(curve.loc[keep,"information"]>=ti)),minimum_information=float(curve.loc[keep,"information"].min()),maximum_sem=float(curve.loc[keep,"conditional_sem"].max()),gaps=gaps)


def eyeprocess_irt_targeting_gap(theta: Any, items: Any, breaks: Any = np.arange(-4,4.01,.5)) -> EyeResult:
    th=np.asarray(theta,dtype=float); th=th[np.isfinite(th)]
    if th.size==0: raise EyeProcessValidationError("theta must contain at least one finite value.")
    br=np.asarray(breaks,dtype=float)
    if br.size<3 or not np.all(np.isfinite(br)) or np.any(np.diff(br)<=0): raise EyeProcessValidationError("breaks must be a strictly increasing finite vector.")
    if th.min()<br.min() or th.max()>br.max(): raise EyeProcessValidationError("breaks must span all finite theta values.")
    mids=br[:-1]+np.diff(br)/2; counts,_=np.histogram(th,bins=br); density=counts/counts.sum(); info=eyeprocess_irt_test_information(mids,items); info_mass=info.information.to_numpy(); info_mass=info_mass/info_mass.sum() if info_mass.sum()>0 else np.zeros_like(info_mass); tab=pd.DataFrame({"theta":mids,"person_mass":density,"information_mass":info_mass,"gap":density-info_mass})
    return _result("eye_irt_targeting_gap",table=tab,absolute_gap=float(np.sum(abs(tab.gap))/2),interpretation="Targeting gap compares empirical score-location mass with normalized information; it is not a validity coefficient.")


def eyeprocess_irt_classification_precision(theta_estimate: Any, standard_error: Any, cut_score: Any = 0, confidence: float = .95) -> pd.DataFrame:
    th=np.atleast_1d(np.asarray(theta_estimate,dtype=float)); se=np.atleast_1d(np.asarray(standard_error,dtype=float))
    if se.size==1 and th.size>1: se=np.repeat(se,th.size)
    if th.size!=se.size or not np.all(np.isfinite(th)) or not np.all(np.isfinite(se)) or np.any(se<0): raise EyeProcessValidationError("theta_estimate and standard_error must be compatible finite vectors with non-negative SEs.")
    cuts=np.atleast_1d(np.asarray(cut_score,dtype=float));
    if cuts.size==0 or not np.all(np.isfinite(cuts)): raise EyeProcessValidationError("cut_score must be finite.")
    confidence=float(confidence)
    if not 0<confidence<1: raise EyeProcessValidationError("confidence must lie in (0,1).")
    z=norm.ppf((1+confidence)/2); rows=[]
    for c in cuts:
        with np.errstate(divide='ignore',invalid='ignore'):
            pa=np.where(se>0,1-norm.cdf((c-th)/se),np.where(th>c,1.0,np.where(th==c,.5,0.0)))
        for i in range(th.size): rows.append({"case":i+1,"cut_score":float(c),"theta":float(th[i]),"se":float(se[i]),"lower":float(th[i]-z*se[i]),"upper":float(th[i]+z*se[i]),"probability_above":float(pa[i]),"classification":"above" if pa[i]>=.5 else "below","confidence_in_classification":float(max(pa[i],1-pa[i]))})
    out=pd.DataFrame(rows); out.attrs["guardrail"]="Classification precision quantifies uncertainty relative to declared cut scores; it does not justify the substantive meaning of those cuts."; return out


def eyeprocess_irt_missing_by_design_audit(responses: Any, design: Any = None, min_administered: int = 1) -> EyeResult:
    y=np.asarray(responses,dtype=float)
    if y.ndim!=2: raise EyeProcessValidationError("responses must be a numeric/integer matrix.")
    minimum=int(min_administered)
    if minimum<1: raise EyeProcessValidationError("min_administered must be a positive integer.")
    observed=~np.isnan(y)
    if design is not None:
        d=np.asarray(design,dtype=float)
        if d.shape!=y.shape: raise EyeProcessValidationError("design must have the same dimensions as responses.")
        if np.any(~np.isnan(d)&~np.isin(d,[0,1])): raise EyeProcessValidationError("design must contain only 0/1/NA.")
        expected=d==1; structural=(~observed)&(~np.isnan(d))&(~expected); unexpected=(~observed)&expected
    else: structural=np.zeros(y.shape,dtype=bool); unexpected=~observed
    return _result("eye_irt_missing_design_audit",n_persons=y.shape[0],n_items=y.shape[1],observed_fraction=float(observed.mean()),administered_per_person=observed.sum(axis=1),administered_per_item=observed.sum(axis=0),sparse_persons=(np.where(observed.sum(axis=1)<minimum)[0]+1),structural_missing=int(structural.sum()),unexpected_missing=int(unexpected.sum()),has_declared_design=design is not None)


def eyeprocess_irt_prior_spec(discrimination: str = "lognormal", difficulty: str = "normal", guessing: str = "beta", location: float = 0, scale: float = 1, guessing_shape: Sequence[float] = (5,17), label: str = "default") -> EyeResult:
    if discrimination not in {"lognormal","normal"} or difficulty not in {"normal","student-t"} or guessing not in {"beta","logit-normal"}: raise EyeProcessValidationError("invalid prior family.")
    loc=_scalar(location,"location"); sc=_scalar(scale,"scale",positive=True); gs=np.asarray(guessing_shape,dtype=float)
    if gs.size!=2 or not np.all(np.isfinite(gs)) or np.any(gs<=0): raise EyeProcessValidationError("guessing_shape must contain two positive finite values.")
    return _result("eye_irt_prior_spec",discrimination=discrimination,difficulty=difficulty,guessing=guessing,location=loc,scale=sc,guessing_shape=gs,label=str(label))


def eyeprocess_irt_prior_sensitivity_grid(discrimination_scale: Sequence[float] = (.5,1,1.5), difficulty_scale: Sequence[float] = (1,2), guessing_mean: Sequence[float] = (.10,.20)) -> pd.DataFrame:
    a=np.asarray(discrimination_scale,dtype=float); b=np.asarray(difficulty_scale,dtype=float); g=np.asarray(guessing_mean,dtype=float)
    if np.any(~np.isfinite(a)) or np.any(a<=0) or np.any(~np.isfinite(b)) or np.any(b<=0) or np.any(~np.isfinite(g)) or np.any((g<=0)|(g>=1)): raise EyeProcessValidationError("Prior grid values are outside their admissible ranges.")
    rows=[{"discrimination_scale":float(x),"difficulty_scale":float(y),"guessing_mean":float(z)} for x in a for y in b for z in g]; out=pd.DataFrame(rows); out["prior_id"]=[f"P{i+1:03d}" for i in range(len(out))]; return out[["prior_id","discrimination_scale","difficulty_scale","guessing_mean"]]


def eyeprocess_irt_prior_sensitivity_summary(results: Any, prior_id: str = "prior_id", estimate: str = "estimate") -> EyeResult:
    df=_as_df(results,"results"); _req_cols(df,[prior_id,estimate],"results"); v=pd.to_numeric(df[estimate],errors="coerce").to_numpy(); keep=np.isfinite(v)
    return _result("eye_irt_prior_sensitivity",n_specifications=len(df),n_finite=int(keep.sum()),median=float(np.median(v[keep])) if keep.any() else np.nan,range=float(np.ptp(v[keep])) if keep.any() else np.nan,sd=float(np.std(v[keep],ddof=1)) if keep.sum()>1 else np.nan,table=df,guardrail="Prior sensitivity describes specification dependence; it is not a license for selective prior choice.")


def eyeprocess_irt_model_card(spec: Any, engine_status: Any = None, identification: Any = None, fit_evidence: Any = None, invariance: Any = None, validation: Any = None, intended_use: Any = None, excluded_interpretations: Sequence[str] = ("diagnosis","cheating inference","mental-state inference")) -> EyeResult:
    if not isinstance(spec,Mapping) or getattr(spec,"eyeprocess_class",None) not in {"eyeprocess_irt_model_spec","eye_joint_process_irt_spec"}: raise EyeProcessValidationError("spec must be an eyeprocess IRT or joint-process IRT specification.")
    payload={"specification":spec,"engine_status":engine_status,"identification":identification,"fit_evidence":fit_evidence,"invariance":invariance,"validation":validation,"intended_use":intended_use,"excluded_interpretations":list(dict.fromkeys(map(str,excluded_interpretations))),"created":str(date.today())}; payload["hash"]=_stable_hash(payload); return EyeResult(payload,eyeprocess_class="eye_irt_model_card")


def eyeprocess_irt_model_card_audit(card: Any) -> pd.DataFrame:
    if not isinstance(card,Mapping) or getattr(card,"eyeprocess_class",None)!="eye_irt_model_card": raise EyeProcessValidationError("card must be created by eyeprocess_irt_model_card().")
    fields=["engine_status","identification","fit_evidence","invariance","validation","intended_use"]
    present=[]
    for f in fields:
        v=card.get(f); present.append(not (v is None or (hasattr(v,"__len__") and len(v)==0)))
    return pd.DataFrame({"field":fields,"present":present})


# public exports for this module
__all__ = [n for n,v in globals().items() if (n.startswith("eyeprocess_") or n.startswith("fit_eyeprocess_") or n.startswith("run_eyeprocess_") or n.startswith("simulate_eyeprocess_") or n.startswith("validate_eyeprocess_") or n.startswith("freeze_eyeprocess_")) and callable(v)]
