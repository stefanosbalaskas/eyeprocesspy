"""Process-aware IRT parity layer from eyeprocess 0.7-era source families.

The dependency-light contracts are direct translations. Functions whose R
reference implementation requires lme4/brms/nnet/survival use auditable
NumPy/SciPy reference estimators and mark that backend difference explicitly;
they are not reported as algorithmically identical until R-oracle validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm

from .exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError
from .irt import EyeResult, _as_df, _req_cols, _result

__all__ = [
    # 049 registry/channels
    "irt_response_channel", "irt_rt_channel", "irt_count_channel", "irt_survival_channel",
    "irt_nominal_channel", "irt_compositional_channel", "irt_sequence_channel",
    "irt_functional_channel", "irt_model_spec", "register_irt_model", "list_irt_models",
    "get_irt_model", "fit_irt_model", "simulate_irt_model", "validate_irt_model",
    "compare_irt_models", "promote_irt_model",
    # 050 process models
    "fit_joint_gaze_rt_irt", "fit_speed_accuracy_engagement_irt",
    "fit_joint_graded_rt_process_irt", "fit_nominal_gaze_irt", "option_process_information",
    "distractor_process_map", "audit_distractor_attention", "classify_item_missingness",
    "estimate_visual_exposure_probability", "fit_omission_survival_irt",
    "fit_manyfacet_process_irt", "facet_effects", "audit_process_measurement_invariance",
    "detect_irt_changepoints", "fit_changepoint_rt_irt", "fit_changepoint_multimodal_irt",
    "recalibrate_after_changepoint",
    # 054 additional measurement
    "irt_continuous_channel", "fit_censored_normal_process_irt",
    "predict_eye_censored_normal_process_irt", "process_dependent_discrimination_audit",
    "process_channel_ablation", "fit_multimodal_trait_irt", "generalizability_process_study",
    "cross_device_process_equating_audit",
    # 057 emerging
    "encode_response_combinations", "audit_process_local_dependence",
    "fit_multiple_response_process_irt", "fit_revisit_process_cdm",
    # Python plot counterparts for R S3 methods
    "plot_eye_process_dependent_discrimination", "plot_eye_process_channel_ablation",
    "plot_eye_process_g_study", "plot_eye_process_local_dependence_audit",
]

_EPS = np.finfo(float).eps
_REGISTRY: dict[str, EyeResult] = {}


def _scalar_chr(x: Any, arg: str) -> str:
    if not isinstance(x, str) or not x:
        raise EyeProcessValidationError(f"{arg} must be a non-empty scalar string.")
    return x


def _choice(x: str, allowed: Sequence[str], arg: str) -> str:
    if x not in allowed:
        raise EyeProcessValidationError(f"{arg} must be one of: {', '.join(allowed)}.")
    return x


def _df(data: Any, cols: Sequence[str] = (), name: str = "data") -> pd.DataFrame:
    d = _as_df(data, name)
    _req_cols(d, list(dict.fromkeys(c for c in cols if c)), name)
    return d


def _channel(type_: str, family: str, role: str, *, link: Any = None, variables: Any = None,
             latent: Any = None, options: Mapping[str, Any] | None = None) -> EyeResult:
    return _result(
        f"eye_irt_{type_}_channel",
        type=type_, family=family, role=role, link=link, variables=variables,
        latent=latent, options=dict(options or {}), superclass="eye_irt_channel",
    )


def irt_response_channel(family: str = "2pl", response: str = "response", latent: str = "ability",
                         options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("2pl", "rasch", "graded", "partial_credit"), "family")
    return _channel("response", family, "measurement", variables=response, latent=latent, options=options)


def irt_rt_channel(family: str = "lognormal", rt: str = "rt", latent: str = "speed",
                   options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("lognormal", "gaussian_log", "shifted_lognormal"), "family")
    return _channel("rt", family, "process", variables=rt, latent=latent, options=options)


def irt_count_channel(family: str = "negative_binomial", value: str = "fixation_count",
                      latent: str = "engagement", options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("negative_binomial", "poisson"), "family")
    return _channel("count", family, "process", variables=value, latent=latent, options=options)


def irt_survival_channel(family: str = "cox", time: str = "time", event: str = "event",
                         latent: str | None = None, options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("cox", "weibull", "exponential"), "family")
    return _channel("survival", family, "process", variables={"time": time, "event": event}, latent=latent, options=options)


def irt_nominal_channel(choice: str = "response_option", categories: Sequence[str] | None = None,
                        latent: str = "ability", options: Mapping[str, Any] | None = None) -> EyeResult:
    opts = dict(options or {})
    opts["categories"] = None if categories is None else list(categories)
    return _channel("nominal", "nominal", "measurement", variables=choice, latent=latent, options=opts)


def irt_compositional_channel(parts: Sequence[str], family: str = "logratio_gaussian", latent: str = "process",
                              options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("logratio_gaussian", "dirichlet"), "family")
    parts = list(parts)
    if len(parts) < 2:
        raise EyeProcessValidationError("parts must name at least two compositional variables.")
    return _channel("compositional", family, "process", variables=[str(x) for x in parts], latent=latent, options=options)


def irt_sequence_channel(sequence: str = "sequence", family: str = "ngram", latent: str = "strategy",
                         options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("ngram", "hmm", "embedding"), "family")
    return _channel("sequence", family, "process", variables=sequence, latent=latent, options=options)


def irt_functional_channel(value: str = "pupil", time: str = "time", family: str = "basis_gaussian",
                           latent: str = "process", options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("basis_gaussian", "functional_factor"), "family")
    return _channel("functional", family, "process", variables={"value": value, "time": time}, latent=latent, options=options)


def irt_continuous_channel(family: str = "censored_normal", value: str = "process_value",
                           lower: float = 0, upper: float = 1, latent: str = "process",
                           options: Mapping[str, Any] | None = None) -> EyeResult:
    family = _choice(family, ("censored_normal", "beta", "gaussian"), "family")
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise EyeProcessValidationError("lower and upper must be finite with lower < upper.")
    opts = dict(options or {})
    opts.update(lower=lower, upper=upper)
    return _channel("continuous", family, "process", variables=value, latent=latent, options=opts)


def irt_model_spec(id: str, latent: Sequence[str] | str, channels: Mapping[str, Any] | Sequence[Any],
                   status: str = "experimental", fit_fun: Callable[..., Any] | None = None,
                   simulate_fun: Callable[..., Any] | None = None, validate_fun: Callable[..., Any] | None = None,
                   citation: Sequence[str] = (), description: str | None = None,
                   requirements: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> EyeResult:
    id = _scalar_chr(id, "id")
    status = _choice(status, ("experimental", "reference", "gated"), "status")
    if isinstance(channels, Mapping):
        ch = dict(channels)
    elif isinstance(channels, Sequence):
        ch = {f"channel_{i+1}": z for i, z in enumerate(channels)}
    else:
        ch = {}
    if not ch or not all(isinstance(z, Mapping) and str(z.get("superclass", "")) == "eye_irt_channel" for z in ch.values()):
        raise EyeProcessValidationError("channels must be a non-empty list/mapping of irt_*_channel() objects.")
    lat = [latent] if isinstance(latent, str) else list(latent)
    lat = list(dict.fromkeys(str(z) for z in lat))
    if not lat or any(not z for z in lat):
        raise EyeProcessValidationError("latent must contain named latent dimensions.")
    for fn in (fit_fun, simulate_fun, validate_fun):
        if fn is not None and not callable(fn):
            raise EyeProcessValidationError("Model callbacks must be functions or None.")
    return _result("eye_irt_model_spec", id=id, latent=lat, channels=ch, status=status,
                   fit_fun=fit_fun, simulate_fun=simulate_fun, validate_fun=validate_fun,
                   citation=list(citation), description=description, requirements=list(requirements),
                   metadata=dict(metadata or {}))


def register_irt_model(spec: Any, overwrite: bool = False) -> EyeResult:
    if getattr(spec, "eyeprocess_class", None) != "eye_irt_model_spec":
        raise EyeProcessValidationError("spec must come from irt_model_spec().")
    if spec.id in _REGISTRY and not overwrite:
        raise EyeProcessValidationError(f"IRT model `{spec.id}` is already registered. Use overwrite=True deliberately.")
    _REGISTRY[spec.id] = spec
    return spec


def _unavailable(name: str) -> Callable[..., Any]:
    def f(*args: Any, **kwargs: Any) -> Any:
        raise EyeProcessBackendError(
            f"Model callback `{name}` belongs to a later eyeprocess process-IRT tranche and is not yet parity-validated."
        )
    f.__name__ = name
    return f


def _register_builtins() -> None:
    builtins = [
        irt_model_spec("joint_gaze_rt", ["ability", "speed", "engagement"],
                       {"response": irt_response_channel("2pl"), "rt": irt_rt_channel("lognormal"), "gaze": irt_count_channel("negative_binomial")},
                       status="reference", fit_fun=fit_joint_gaze_rt_irt,
                       citation=["10.1177/01466216221089344"], description="Response + RT + gaze-count joint measurement architecture."),
        irt_model_spec("nominal_gaze", ["ability", "option_process"],
                       {"response": irt_nominal_channel(), "gaze": irt_compositional_channel(["option_A", "option_B"])},
                       status="experimental", fit_fun=fit_nominal_gaze_irt,
                       description="Nominal response choices integrated with option-level gaze evidence."),
        irt_model_spec("omission_survival", ["ability", "speed", "omission_process"],
                       {"response": irt_response_channel("2pl"), "time": irt_survival_channel()},
                       status="experimental", fit_fun=fit_omission_survival_irt),
        irt_model_spec("manyfacet_process", ["person", "item", "process"],
                       {"response": irt_response_channel("rasch"), "gaze": irt_count_channel()},
                       status="reference", fit_fun=fit_manyfacet_process_irt),
        irt_model_spec("process_hmm", ["ability", "process_state"],
                       {"response": irt_response_channel("2pl"), "process": irt_sequence_channel(family="hmm")},
                       status="experimental", fit_fun=_unavailable("fit_process_hmm_irt")),
        irt_model_spec("graded_rt_process", ["ability", "speed", "process"],
                       {"response": irt_response_channel("graded"), "rt": irt_rt_channel("lognormal"), "process": irt_count_channel("negative_binomial")},
                       status="experimental", fit_fun=fit_joint_graded_rt_process_irt),
        irt_model_spec("latent_space_process", ["ability", "interaction_space"],
                       {"response": irt_response_channel("graded")}, status="experimental",
                       fit_fun=_unavailable("fit_latent_space_irt"), requirements=["LSMjml"]),
        irt_model_spec("gpirt_shape_audit", ["ability"], {"response": irt_response_channel("2pl")},
                       status="gated", fit_fun=_unavailable("fit_gpirt")),
        irt_model_spec("flow_mirt", ["ability_1", "ability_2"], {"response": irt_response_channel("2pl")},
                       status="gated", fit_fun=_unavailable("fit_flow_mirt")),
        irt_model_spec("multiple_response_process", ["ability", "option_process"],
                       {"response": irt_nominal_channel(), "process": irt_compositional_channel(["selected", "not_selected"])},
                       status="experimental", fit_fun=fit_multiple_response_process_irt),
        irt_model_spec("bounded_continuous_process", ["process_trait"], {"process": irt_continuous_channel("censored_normal")},
                       status="experimental", fit_fun=fit_censored_normal_process_irt),
    ]
    for sp in builtins:
        _REGISTRY.setdefault(sp.id, sp)


def list_irt_models() -> pd.DataFrame:
    _register_builtins()
    rows = []
    for id_ in sorted(_REGISTRY):
        z = _REGISTRY[id_]
        rows.append(dict(id=z.id, status=z.status, latent=", ".join(z.latent), channels=", ".join(z.channels),
                         requirements=", ".join(z.requirements), description=z.description or ""))
    return pd.DataFrame(rows)


def get_irt_model(id: str) -> EyeResult:
    _register_builtins()
    id = _scalar_chr(id, "id")
    if id not in _REGISTRY:
        raise EyeProcessValidationError(f"Unknown IRT model `{id}`. See list_irt_models().")
    return _REGISTRY[id]


def fit_irt_model(spec: Any, data: Any, *args: Any, allow_experimental: bool = False, **kwargs: Any) -> Any:
    if isinstance(spec, str):
        spec = get_irt_model(spec)
    if getattr(spec, "eyeprocess_class", None) != "eye_irt_model_spec":
        raise EyeProcessValidationError("spec must be a model id or eye_irt_model_spec.")
    if spec.status in {"experimental", "gated"} and not allow_experimental:
        raise EyeProcessModelError(f"Model `{spec.id}` is {spec.status}. Set allow_experimental=True only for validation/research use.")
    if spec.fit_fun is None:
        raise EyeProcessBackendError(f"Model `{spec.id}` has no bundled fitter.")
    fit = spec.fit_fun(data, *args, **kwargs)
    if isinstance(fit, EyeResult):
        fit["eye_irt_model_spec"] = spec
    return fit


def simulate_irt_model(spec: Any, *args: Any, allow_experimental: bool = True, **kwargs: Any) -> Any:
    if isinstance(spec, str):
        spec = get_irt_model(spec)
    if getattr(spec, "eyeprocess_class", None) != "eye_irt_model_spec":
        raise EyeProcessValidationError("spec must be a model id or eye_irt_model_spec.")
    if spec.simulate_fun is None:
        raise EyeProcessBackendError(f"Model `{spec.id}` does not register a simulator. Supply one through irt_model_spec().")
    if not allow_experimental and spec.status != "reference":
        raise EyeProcessModelError("Experimental model simulation is disabled.")
    return spec.simulate_fun(*args, **kwargs)


def _grade_evidence(validation: Any) -> EyeResult:
    if isinstance(validation, Mapping):
        if "pass" in validation:
            passed = bool(validation["pass"])
        elif "grade" in validation:
            passed = str(validation["grade"]).lower() in {"pass", "reference", "strong", "a", "green"}
        else:
            passed = False
        grade = validation.get("grade", "pass" if passed else "review")
        return _result("eye_irt_evidence_grade", pass_=passed, pass_value=passed, grade=str(grade))
    passed = bool(validation)
    return _result("eye_irt_evidence_grade", pass_=passed, pass_value=passed, grade="pass" if passed else "review")


def validate_irt_model(spec: Any, validation: Any = None, **kwargs: Any) -> Any:
    if isinstance(spec, str):
        spec = get_irt_model(spec)
    if getattr(spec, "eyeprocess_class", None) != "eye_irt_model_spec":
        raise EyeProcessValidationError("spec must be a model id or eye_irt_model_spec.")
    if spec.validate_fun is not None:
        return spec.validate_fun(validation=validation, **kwargs)
    if validation is None:
        raise EyeProcessValidationError("Supply a validation object or register a model-specific validator.")
    return _grade_evidence(validation)


def _model_metric(fit: Any, name: str) -> float:
    if isinstance(fit, Mapping):
        for key in (name, name.lower(), name.upper()):
            if key in fit and np.isscalar(fit[key]):
                try: return float(fit[key])
                except Exception: pass
        model = fit.get("model")
        if model is not None:
            fit = model
    for attr in (name, name.lower(), name.upper()):
        if hasattr(fit, attr):
            v = getattr(fit, attr)
            try: return float(v() if callable(v) else v)
            except Exception: pass
    return math.nan


def compare_irt_models(*fits: Any, names: Sequence[str] | None = None) -> pd.DataFrame:
    if not fits:
        raise EyeProcessValidationError("Supply at least one fitted model.")
    if names is None:
        names = [f"model_{i+1}" for i in range(len(fits))]
    if len(names) != len(fits):
        raise EyeProcessValidationError("names must match the number of fitted models.")
    out = pd.DataFrame({"model": list(names),
                        "logLik": [_model_metric(f, "logLik") for f in fits],
                        "AIC": [_model_metric(f, "AIC") for f in fits],
                        "BIC": [_model_metric(f, "BIC") for f in fits]})
    out.attrs["eyeprocess_class"] = "eye_irt_model_comparison"
    return out


def promote_irt_model(spec: Any, evidence: Any, target: str = "experimental", update_registry: bool = False) -> EyeResult:
    target = _choice(target, ("experimental", "reference"), "target")
    if isinstance(spec, str): spec = get_irt_model(spec)
    if getattr(spec, "eyeprocess_class", None) != "eye_irt_model_spec":
        raise EyeProcessValidationError("spec must be a registered model or model spec.")
    grade = _grade_evidence(evidence)
    passed = bool(grade.pass_value)
    if target == "reference" and not passed:
        raise EyeProcessModelError(f"Model `{spec.id}` cannot be promoted to reference: evidence grade is {grade.grade}.")
    promoted = EyeResult(spec.copy(), eyeprocess_class="eye_irt_model_spec")
    promoted["status"] = target
    if update_registry:
        register_irt_model(promoted, overwrite=True)
    return _result("eye_irt_promotion", model=spec.id, **{"from": spec.status, "to": target},
                   evidence_grade=grade.grade, evidence_pass=passed,
                   timestamp=datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# compact numerical helpers used by process reference estimators
# ---------------------------------------------------------------------------

def _dummy_design(d: pd.DataFrame, continuous: Sequence[str] = (), categorical: Sequence[str] = (), intercept: bool = True) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    if intercept:
        parts.append(np.ones((len(d), 1))); names.append("Intercept")
    for c in continuous:
        v = pd.to_numeric(d[c], errors="coerce").to_numpy(float).reshape(-1, 1)
        parts.append(v); names.append(c)
    for c in categorical:
        dm = pd.get_dummies(d[c].astype("category"), prefix=c, drop_first=True, dtype=float)
        if dm.shape[1]:
            parts.append(dm.to_numpy(float)); names.extend(dm.columns.astype(str).tolist())
    return (np.hstack(parts) if parts else np.empty((len(d), 0))), names


def _logistic_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> EyeResult:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        b = np.zeros(X.shape[1])
        p = np.full(len(y), np.mean(y) if len(y) else np.nan)
        return _result("eye_binary_reference_fit", coefficients=b, fitted=p, logLik=math.nan, convergence=1)
    def obj(b: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ b; p = expit(eta)
        nll = -np.sum(y*np.log(np.clip(p,_EPS,1)) + (1-y)*np.log(np.clip(1-p,_EPS,1))) + .5*ridge*np.sum(b[1:]**2)
        grad = X.T @ (p-y); grad[1:] += ridge*b[1:]
        return float(nll), grad
    res = minimize(lambda b: obj(b)[0], np.zeros(X.shape[1]), jac=lambda b: obj(b)[1], method="BFGS")
    p = expit(X @ res.x)
    ll = float(np.sum(y*np.log(np.clip(p,_EPS,1)) + (1-y)*np.log(np.clip(1-p,_EPS,1))))
    return _result("eye_binary_reference_fit", coefficients=res.x, fitted=p, logLik=ll,
                   AIC=-2*ll+2*len(res.x), BIC=-2*ll+len(res.x)*math.log(max(len(y),1)), convergence=0 if res.success else 1)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> EyeResult:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    if not len(yo):
        return _result("eye_gaussian_reference_fit", coefficients=np.full(X.shape[1], np.nan), fitted=np.full(len(y),np.nan), residuals=np.full(len(y),np.nan), logLik=math.nan)
    b, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    fitted_all = X @ b
    resid = y - fitted_all
    sigma = max(float(np.nanstd(resid[ok], ddof=min(1,max(len(yo)-1,0)))), 1e-8)
    ll = float(np.sum(norm.logpdf(resid[ok], scale=sigma)))
    return _result("eye_gaussian_reference_fit", coefficients=b, fitted=fitted_all, residuals=resid,
                   sigma=sigma, logLik=ll, AIC=-2*ll+2*len(b), BIC=-2*ll+len(b)*math.log(max(len(yo),1)))


def _multinomial_fit(X: np.ndarray, y: Sequence[Any], labels: Sequence[str] | None = None, ridge: float = 1e-6) -> EyeResult:
    cat = pd.Categorical(y, categories=labels)
    codes = cat.codes
    keep = codes >= 0
    X, codes = X[keep], codes[keep]
    classes = [str(z) for z in cat.categories]
    k, p = len(classes), X.shape[1]
    if k < 2:
        raise EyeProcessValidationError("At least two response categories are required.")
    def unpack(v: np.ndarray) -> np.ndarray:
        B = np.zeros((p, k)); B[:,1:] = v.reshape(p,k-1); return B
    def obj(v: np.ndarray) -> float:
        B=unpack(v); eta=X@B; lse=logsumexp(eta,axis=1); ll=np.sum(eta[np.arange(len(codes)),codes]-lse)
        return float(-ll + .5*ridge*np.sum(v*v))
    res=minimize(obj,np.zeros(p*(k-1)),method="BFGS")
    B=unpack(res.x); probs=np.exp(X@B-logsumexp(X@B,axis=1,keepdims=True)); ll=-obj(res.x)+.5*ridge*np.sum(res.x*res.x)
    return _result("eye_multinomial_reference_fit", coefficients=B, classes=classes, fitted_probability=probs,
                   logLik=float(ll), AIC=-2*ll+2*len(res.x), BIC=-2*ll+len(res.x)*math.log(max(len(codes),1)), convergence=0 if res.success else 1)


def fit_joint_gaze_rt_irt(data: Any, response: str = "response", rt: str = "rt", gaze: str = "fixation_count",
                          person: str = "participant_id", item: str = "item_id", gaze_family: str = "negative_binomial",
                          engine: str = "reference", iter: int = 2000, chains: int = 4, cores: int = 1, seed: int = 1,
                          **kwargs: Any) -> EyeResult:
    gaze_family = _choice(gaze_family,("negative_binomial","poisson"),"gaze_family")
    engine = _choice(engine,("reference","brms"),"engine")
    if engine == "brms":
        raise EyeProcessBackendError("The R `brms` engine has no exact Python identity; use the canonical Stan/PyMC extension only after model-specific validation.")
    d=_df(data,[response,rt,gaze,person,item]).dropna(subset=[response,rt,gaze,person,item]).copy()
    y=pd.to_numeric(d[response],errors="coerce").to_numpy(float)
    if not np.all(np.isin(y,[0,1])): raise EyeProcessValidationError("response must be coded 0/1 for fit_joint_gaze_rt_irt().")
    r=pd.to_numeric(d[rt],errors="coerce").to_numpy(float); g=pd.to_numeric(d[gaze],errors="coerce").to_numpy(float)
    if np.any(r<=0): raise EyeProcessValidationError("Response times must be strictly positive.")
    if np.any(g<0): raise EyeProcessValidationError("Gaze counts must be non-negative.")
    X,names=_dummy_design(d,categorical=[person,item])
    response_fit=_logistic_fit(X,y)
    rt_fit=_ols_fit(X,np.log(r))
    gaze_fit=_ols_fit(X,np.log1p(g))
    # auditable empirical person/item scores from grouped channel residual means
    work=pd.DataFrame({"person":d[person].astype(str),"item":d[item].astype(str),"rres":y-response_fit.fitted,
                       "tres":np.log(r)-rt_fit.fitted,"gres":np.log1p(g)-gaze_fit.fitted})
    ps=work.groupby("person",sort=False)[["rres","tres","gres"]].mean().reset_index().rename(columns={"person":"id","rres":"ability","tres":"speed_logtime","gres":"gaze_propensity"})
    is_=work.groupby("item",sort=False)[["rres","tres","gres"]].mean().reset_index().rename(columns={"item":"id","rres":"response_intercept","tres":"time_intensity","gres":"gaze_intensity"})
    return _result("eye_joint_gaze_rt_irt", engine="reference", response_model=response_fit, rt_model=rt_fit, gaze_model=gaze_fit,
                   person_scores=ps,item_scores=is_,person_covariance=ps.iloc[:,1:].cov().to_numpy(),item_covariance=is_.iloc[:,1:].cov().to_numpy(),
                   data_n=len(d),gaze_family=gaze_family,columns={"response":response,"rt":rt,"gaze":gaze,"person":person,"item":item},
                   status="python-reference-estimator",algorithmic_parity=False,
                   interpretation="Auditable fixed-effect decomposition corresponding to the R lme4 reference architecture; not an identical random-effects estimator.")


def fit_speed_accuracy_engagement_irt(data: Any, *args: Any, engine: str = "reference", **kwargs: Any) -> EyeResult:
    fit=fit_joint_gaze_rt_irt(data,*args,engine=engine,**kwargs); fit.eyeprocess_class="eye_speed_accuracy_engagement_irt"; return fit


def fit_joint_graded_rt_process_irt(data: Any,response: str="response",rt: str="rt",process: str="fixation_count",person: str="participant_id",item: str="item_id",
                                    engine: str="reference",process_family: str="negative_binomial",iter: int=2000,chains: int=4,cores: int=1,seed: int=1,**kwargs: Any) -> EyeResult:
    engine=_choice(engine,("reference","brms"),"engine"); process_family=_choice(process_family,("negative_binomial","poisson","gaussian"),"process_family")
    if engine=="brms": raise EyeProcessBackendError("The R `brms` graded-response engine is backend-gated in Python pending model-specific validation.")
    d=_df(data,[response,rt,process,person,item]).dropna().copy(); r=pd.to_numeric(d[rt],errors="coerce").to_numpy(float)
    if np.any(r<=0): raise EyeProcessValidationError("RT must be positive.")
    cats=pd.Categorical(d[response],ordered=True); X,names=_dummy_design(d,categorical=[item]); response_fit=_multinomial_fit(X,cats.astype(str),labels=[str(x) for x in cats.categories])
    resp_codes=cats.codes.astype(float); Xrt,nrt=_dummy_design(d,continuous=[],categorical=[person,item]); Xrt=np.column_stack([Xrt,resp_codes]); rt_fit=_ols_fit(Xrt,np.log(r))
    Xp,npn=_dummy_design(d,categorical=[person,item]); pv=pd.to_numeric(d[process],errors="coerce").to_numpy(float); process_fit=_ols_fit(Xp,np.log1p(np.maximum(pv,0)) if process_family!="gaussian" else pv)
    return _result("eye_joint_graded_rt_process_irt",engine="reference",response_model=response_fit,rt_model=rt_fit,process_model=process_fit,data_n=len(d),status="experimental-python-reference",algorithmic_parity=False,
                   note="Transparent Python decomposition; not the published SAEM estimator and not identical to the R MASS/lme4 reference.")


def fit_nominal_gaze_irt(data: Any,response_option: str="response_option",option_gaze: Sequence[str]=(),person: str="participant_id",item: str="item_id",ability: str|None=None,correct_option: Any=None,add_item_effects: bool=True,**kwargs: Any) -> EyeResult:
    option_gaze=list(option_gaze)
    if len(option_gaze)<2: raise EyeProcessValidationError("Supply at least two option-level gaze columns.")
    cols=[response_option,*option_gaze,person,item]+([ability] if ability else [])
    d=_df(data,cols).copy(); gaze=d[option_gaze].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float); rs=gaze.sum(axis=1); prop=gaze/np.maximum(rs[:,None],_EPS)
    prop_cols=[f"gaze_prop_{c}" for c in option_gaze]
    for j,c in enumerate(prop_cols): d[c]=prop[:,j]
    if ability is None:
        if correct_option is None:
            modal=d.groupby(item,dropna=False)[response_option].agg(lambda z:z.astype(str).value_counts().index[0]); correct=d[response_option].astype(str).to_numpy()==d[item].map(modal).astype(str).to_numpy()
        elif isinstance(correct_option,str) and correct_option in d.columns: correct=d[response_option].astype(str).to_numpy()==d[correct_option].astype(str).to_numpy()
        elif np.isscalar(correct_option): correct=d[response_option].astype(str).to_numpy()==str(correct_option)
        else: correct=d[response_option].astype(str).to_numpy()==np.asarray(correct_option).astype(str)
        tmp=pd.DataFrame({"person":d[person].astype(str),"correct":correct.astype(float)}); agg=tmp.groupby("person")["correct"].agg(["mean","count"]); adj=(agg["mean"]*agg["count"]+.5)/(agg["count"]+1); theta=np.log(np.clip(adj,1e-6,1-1e-6)/np.clip(1-adj,1e-6,1)); d[".ability_proxy"]=d[person].astype(str).map(theta); ability_term=".ability_proxy"
    else: ability_term=ability
    X,names=_dummy_design(d,continuous=[ability_term,*prop_cols],categorical=[item] if add_item_effects else []); y=d[response_option].astype(str)
    fit=_multinomial_fit(X,y)
    Xb,basenames=_dummy_design(d,continuous=[ability_term],categorical=[item] if add_item_effects else []); base=_multinomial_fit(Xb,y,labels=fit.classes)
    return _result("eye_nominal_gaze_irt",model=fit,baseline_model=base,data=d,option_gaze=option_gaze,gaze_proportion_columns=prop_cols,ability=ability_term,person=person,item=item,
                   design_names=names,baseline_design_names=basenames,logLik_gain=fit.logLik-base.logLik,status="experimental-two-stage",algorithmic_parity=True,
                   note="Option gaze is process evidence; the fitted model does not prove a cognitive meaning for gaze allocation.")


def option_process_information(object: Any) -> pd.DataFrame:
    if getattr(object,"eyeprocess_class",None)!="eye_nominal_gaze_irt": raise EyeProcessValidationError("object must be an eye_nominal_gaze_irt.")
    pf=np.asarray(object.model.fitted_probability,float); pb=np.asarray(object.baseline_model.fitted_probability,float)
    entropy=lambda p:-np.sum(np.clip(p,_EPS,1)*np.log(np.clip(p,_EPS,1)),axis=1)
    full,base=entropy(pf),entropy(pb); out=pd.DataFrame({"row":np.arange(1,len(full)+1),"entropy_reduction":base-full,"full_entropy":full,"baseline_entropy":base}); out.attrs["eyeprocess_class"]="eye_option_process_information"; return out


def distractor_process_map(object: Any) -> pd.DataFrame:
    if getattr(object,"eyeprocess_class",None)!="eye_nominal_gaze_irt": raise EyeProcessValidationError("object must be an eye_nominal_gaze_irt.")
    names=list(object.design_names); idx=[i for i,n in enumerate(names) if n.startswith("gaze_prop_")]; rows=[]
    B=np.asarray(object.model.coefficients)
    for k,cls in enumerate(object.model.classes):
        for i in idx: rows.append({"response_category":cls,"gaze_channel":names[i],"coefficient":float(B[i,k])})
    return pd.DataFrame(rows)


def audit_distractor_attention(data: Any,response_option: str="response_option",option_gaze: Sequence[str]=(),chosen_suffix: str|None=None) -> pd.DataFrame:
    # named R vector is represented by mapping category->column, otherwise names equal columns
    if isinstance(option_gaze,Mapping): map_names=list(map(str,option_gaze.keys())); cols=list(map(str,option_gaze.values()))
    else: cols=list(map(str,option_gaze)); map_names=cols
    d=_df(data,[response_option,*cols]); cats=d[response_option].astype(str).to_numpy(); chosen=[]; unchosen=[]
    for i,cat in enumerate(cats):
        z=pd.to_numeric(d.iloc[i][cols],errors="coerce").fillna(0).to_numpy(float); idx=map_names.index(cat) if cat in map_names else None
        chosen.append(np.nan if idx is None else z[idx]); unchosen.append(np.nan if idx is None or len(z)<2 else float(np.mean(np.delete(z,idx))))
    chosen=np.asarray(chosen); unchosen=np.asarray(unchosen); diff=chosen-unchosen
    return pd.DataFrame([{"n":int(np.isfinite(diff).sum()),"mean_chosen":float(np.nanmean(chosen)),"mean_unchosen":float(np.nanmean(unchosen)),"mean_difference":float(np.nanmean(diff)),"median_difference":float(np.nanmedian(diff))}])


def classify_item_missingness(data: Any,response: str="response",reached: str="reached",inspected: str|None=None,started: str|None=None) -> pd.Categorical:
    cols=[response,reached]+([inspected] if inspected else [])+([started] if started else []); d=_df(data,cols); ym=d[response].isna().to_numpy(); rv=d[reached].astype("boolean").to_numpy(dtype=object)
    iv=np.array([None]*len(d),object) if inspected is None else d[inspected].astype("boolean").to_numpy(dtype=object); sv=np.array([None]*len(d),object) if started is None else d[started].astype("boolean").to_numpy(dtype=object)
    state=np.array(["answered"]*len(d),object)
    for i in range(len(d)):
        if not ym[i]: continue
        r=False if rv[i] is pd.NA or rv[i] is None else bool(rv[i])
        if not r: state[i]="not_reached"; continue
        state[i]="omitted_after_reach"
        if iv[i] is not pd.NA and iv[i] is not None: state[i]="inspected_omission" if bool(iv[i]) else "reached_not_inspected"
        if sv[i] is not pd.NA and sv[i] is not None and bool(sv[i]): state[i]="started_unanswered"
    levels=["answered","not_reached","reached_not_inspected","inspected_omission","started_unanswered","omitted_after_reach"]
    return pd.Categorical(state,categories=levels,ordered=False)


def estimate_visual_exposure_probability(data: Any,exposed: str="reached",predictors: Sequence[str]=(),family: Any=None) -> EyeResult:
    predictors=list(predictors); d=_df(data,[exposed,*predictors]).dropna(subset=[exposed,*predictors]).copy(); y=d[exposed].astype(float).to_numpy(); X,names=_dummy_design(d,continuous=[p for p in predictors if pd.api.types.is_numeric_dtype(d[p])],categorical=[p for p in predictors if not pd.api.types.is_numeric_dtype(d[p])]); fit=_logistic_fit(X,y)
    return _result("eye_visual_exposure_model",model=fit,fitted_probability=fit.fitted,exposed=exposed,predictors=predictors,design_names=names)


def fit_omission_survival_irt(data: Any,response: str="response",response_time: str="response_time",omission_time: str|None=None,reached: str="reached",person: str="participant_id",item: str="item_id",gaze_exposure: str|None=None,first_fixation_latency: str|None=None,**kwargs: Any) -> EyeResult:
    cols=[response,response_time,reached,person,item]+[c for c in (omission_time,gaze_exposure,first_fixation_latency) if c]; d=_df(data,cols).copy(); states=classify_item_missingness(d,response=response,reached=reached,inspected=gaze_exposure); d[".missing_state"]=states.astype(str); times=pd.to_numeric(d[response_time],errors="coerce").to_numpy(dtype=float, copy=True)
    if omission_time: use=d[response].isna().to_numpy() & np.isfinite(pd.to_numeric(d[omission_time],errors="coerce").to_numpy(float)); times[use]=pd.to_numeric(d.loc[use,omission_time],errors="coerce")
    finite=times[np.isfinite(times)&(times>0)]; fallback=float(np.max(finite)) if finite.size else 1.0; times[~np.isfinite(times)|(times<=0)]=fallback; d[".time"]=times; d[".omission_event"]=d[".missing_state"].isin(["omitted_after_reach","inspected_omission","started_unanswered"]).astype(int); d[".not_reached_event"]=(d[".missing_state"]=="not_reached").astype(int)
    # dependency-light exponential cause-specific reference: rate per item (+ optional covariates are retained for audit)
    def rates(event: str) -> pd.DataFrame:
        return d.groupby(item,dropna=False).apply(lambda z: pd.Series({"events":int(z[event].sum()),"exposure":float(z[".time"].sum()),"hazard":float(z[event].sum()/max(z[".time"].sum(),_EPS))}),include_groups=False).reset_index()
    answered=d[d[response].notna() & d[reached].astype(bool)].copy(); response_fit=None
    if len(answered) and set(pd.to_numeric(answered[response],errors="coerce").dropna().unique()).issubset({0,1}):
        X,nm=_dummy_design(answered,categorical=[person,item]); response_fit=_logistic_fit(X,pd.to_numeric(answered[response]).to_numpy(float))
    counts=d[".missing_state"].value_counts(dropna=False)
    return _result("eye_omission_survival_irt",response_model=response_fit,omission_model=rates(".omission_event"),not_reached_model=rates(".not_reached_event"),classified_data=d,state_counts=counts,
                   status="experimental-python-exponential-reference",algorithmic_parity=False,note="Cause-specific exponential-hazard reference; R uses clustered Cox PH through survival::coxph.")


def _facet_components(d: pd.DataFrame,outcome: str,facets: Mapping[str,str]) -> tuple[EyeResult,pd.DataFrame]:
    y=pd.to_numeric(d[outcome],errors="coerce").to_numpy(float); grand=float(np.nanmean(y)); rows=[]; effects={}
    explained=np.zeros(len(d))
    for role,col in facets.items():
        means=d.assign(_y=y).groupby(col,dropna=False)["_y"].mean()-grand; eff=d[col].map(means).to_numpy(float); effects[role]=means; explained+=np.nan_to_num(eff); rows.append({"grp":role,"var1":col,"var2":np.nan,"vcov":float(np.nanvar(eff,ddof=1)) if len(np.unique(d[col]))>1 else 0.0,"sdcor":float(np.nanstd(eff,ddof=1)) if len(np.unique(d[col]))>1 else 0.0})
    resid=y-grand-explained; rv=float(np.nanvar(resid,ddof=1)) if np.isfinite(resid).sum()>1 else 0.0; rows.append({"grp":"Residual","var1":np.nan,"var2":np.nan,"vcov":rv,"sdcor":math.sqrt(max(rv,0))})
    return _result("eye_manyfacet_reference_fit",grand_mean=grand,effects=effects,residuals=resid),pd.DataFrame(rows)


def fit_manyfacet_process_irt(data: Any,response: str="response",process: str|None=None,person: str="participant_id",item: str="item_id",device: str|None=None,session: str|None=None,site: str|None=None,algorithm: str|None=None,aoi_definition: str|None=None,process_family: str="gaussian") -> EyeResult:
    process_family=_choice(process_family,("gaussian","poisson","negative_binomial"),"process_family"); facets={k:v for k,v in {"person":person,"item":item,"device":device,"session":session,"site":site,"algorithm":algorithm,"aoi_definition":aoi_definition}.items() if v}; cols=[response]+([process] if process else [])+list(facets.values()); d=_df(data,cols).dropna(subset=[response,*facets.values()]).copy(); response_fit,response_vc=_facet_components(d,response,facets); process_fit=process_vc=None
    if process: process_fit,process_vc=_facet_components(d.dropna(subset=[process]),process,facets)
    return _result("eye_manyfacet_process_irt",response_model=response_fit,process_model=process_fit,facets=facets,process_family=process_family,status="python-reference-estimator",variance_components={"response":response_vc,"process":process_vc},algorithmic_parity=False)


def facet_effects(object: Any,channel: str="response") -> EyeResult:
    if getattr(object,"eyeprocess_class",None)!="eye_manyfacet_process_irt": raise EyeProcessValidationError("object must be an eye_manyfacet_process_irt.")
    channel=_choice(channel,("response","process"),"channel"); fit=object.response_model if channel=="response" else object.process_model
    if fit is None: raise EyeProcessValidationError(f"No {channel} model was fitted.")
    return _result("eye_facet_effects",random_effects=fit.effects,variance_components=object.variance_components[channel].copy())


def audit_process_measurement_invariance(object: Any,channel: str="process",relative_sd_threshold: float=.25) -> EyeResult:
    ef=facet_effects(object,channel); vc=ef.variance_components.copy(); residual=vc.loc[vc.grp=="Residual","sdcor"]; base=float(residual.iloc[0]) if len(residual) and np.isfinite(residual.iloc[0]) else float(vc.sdcor.max()); base=max(base,_EPS); vc["relative_sd"]=vc.sdcor/base; vc["flag"]=(vc.grp!="Residual")&np.isfinite(vc.relative_sd)&(vc.relative_sd>relative_sd_threshold)
    return _result("eye_process_measurement_invariance",pass_value=not bool(vc.flag.any()),threshold=relative_sd_threshold,components=vc,channel=channel,note="Facet variance is evidence of transportability differences, not proof of vendor bias or causal device effects.")


def _segment_loglik(y: np.ndarray|None=None,rt: np.ndarray|None=None,gaze: np.ndarray|None=None) -> tuple[float,int]:
    ll=0.0;k=0
    if y is not None:
        z=y[np.isfinite(y)]
        if len(z): p=float(np.clip(np.mean(z),1e-6,1-1e-6)); ll+=float(np.sum(z*np.log(p)+(1-z)*np.log(1-p))); k+=1
    for z in (rt,gaze):
        if z is not None:
            zz=z[np.isfinite(z)]
            if len(zz)>=2:
                s=max(float(np.std(zz,ddof=1)),1e-6); ll+=float(np.sum(norm.logpdf(zz,loc=float(np.mean(zz)),scale=s))); k+=2
    return ll,k


def _best_split(y: np.ndarray|None,rt: np.ndarray|None,gaze: np.ndarray|None,min_segment: int) -> dict[str,Any]:
    lengths=[len(z) for z in (y,rt,gaze) if z is not None]; n=max(lengths) if lengths else 0; ll0,k0=_segment_loglik(y,rt,gaze); null=-2*ll0+k0*math.log(max(n,2)); cand=list(range(min_segment,n-min_segment+1))
    if not cand: return {"index":None,"delta_sic":0.0,"null_sic":null,"split_sic":null}
    scores=[]
    for k in cand:
        a=_segment_loglik(y[:k] if y is not None else None,rt[:k] if rt is not None else None,gaze[:k] if gaze is not None else None); b=_segment_loglik(y[k:] if y is not None else None,rt[k:] if rt is not None else None,gaze[k:] if gaze is not None else None); scores.append(-2*(a[0]+b[0])+(a[1]+b[1]+1)*math.log(max(n,2)))
    j=int(np.argmin(scores)); return {"index":cand[j],"delta_sic":float(null-scores[j]),"null_sic":float(null),"split_sic":float(scores[j])}


def detect_irt_changepoints(data: Any,person: str="participant_id",order: str="item_order",response: str="response",rt: str="rt",gaze: str|None=None,min_segment: int=5,min_delta_sic: float=2,max_changes: int=2) -> EyeResult:
    cols=[person,order]+[c for c in (response,rt,gaze) if c]; d=_df(data,cols); min_segment=int(min_segment)
    if min_segment<2: raise EyeProcessValidationError("min_segment must be at least 2.")
    rows=[]
    for pid,z in d.groupby(person,sort=False,dropna=False):
        z=z.sort_values(order,kind="stable"); y=pd.to_numeric(z[response],errors="coerce").to_numpy(float) if response else None; r=np.log(np.maximum(pd.to_numeric(z[rt],errors="coerce").to_numpy(float),_EPS)) if rt else None
        g=None
        if gaze:
            raw=pd.to_numeric(z[gaze],errors="coerce").to_numpy(float); sd=np.nanstd(raw,ddof=1); g=(raw-np.nanmean(raw))/sd if np.isfinite(sd) and sd>0 else np.zeros_like(raw)
        best=_best_split(y,r,g,min_segment); idx=best["index"]; cp_order=np.nan if idx is None else z[order].iloc[idx-1]
        rows.append({"participant_id":str(pid),"changepoint_index":np.nan if idx is None else idx,"changepoint_order":cp_order,"delta_sic":best["delta_sic"],"detected":bool(np.isfinite(best["delta_sic"]) and best["delta_sic"]>=min_delta_sic),"n":len(z)})
    return _result("eye_irt_changepoints",results=pd.DataFrame(rows),channels={"response":response,"rt":rt,"gaze":gaze},method="sic-inspired-reference",min_segment=min_segment,min_delta_sic=min_delta_sic,max_changes=int(max_changes))


def fit_changepoint_rt_irt(data: Any,*args: Any,refit: bool=True,**kwargs: Any) -> EyeResult:
    kwargs.pop("gaze",None); return _result("eye_changepoint_rt_irt",changepoints=detect_irt_changepoints(data,*args,gaze=None,**kwargs),refit_requested=bool(refit),status="experimental-reference")


def fit_changepoint_multimodal_irt(data: Any,*args: Any,gaze: str="fixation_count",refit: bool=True,**kwargs: Any) -> EyeResult:
    return _result("eye_changepoint_multimodal_irt",changepoints=detect_irt_changepoints(data,*args,gaze=gaze,**kwargs),refit_requested=bool(refit),status="experimental-reference")


def recalibrate_after_changepoint(data: Any,fitter: Callable[[pd.DataFrame],Any],person: str="participant_id",order: str="item_order",policy: str="flag",**kwargs: Any) -> EyeResult:
    policy=_choice(policy,("flag","exclude_post_change","add_regime"),"policy")
    if not callable(fitter): raise EyeProcessValidationError("fitter must be a function.")
    cp=detect_irt_changepoints(data,person=person,order=order,**kwargs); d=_as_df(data).copy(); mapping=dict(zip(cp.results.participant_id.astype(str),cp.results.changepoint_order)); regime=[]
    for _,row in d.iterrows():
        cut=mapping.get(str(row[person]),np.nan); regime.append("post" if np.isfinite(cut) and row[order]>cut else "pre")
    d[".process_regime"]=pd.Categorical(regime,categories=["pre","post"],ordered=True); used=d if policy!="exclude_post_change" else d[d[".process_regime"]!="post"].copy(); fit=fitter(used)
    return _result("eye_changepoint_recalibration",changepoints=cp,data=used,fit=fit,policy=policy)


# ---------------------------------------------------------------------------
# bounded continuous/process audits
# ---------------------------------------------------------------------------

def _cn_nll(par: np.ndarray,y: np.ndarray,theta: np.ndarray,lower: float,upper: float) -> float:
    alpha,beta,ls=par; sigma=math.exp(ls); mu=alpha*theta+beta; eps=math.sqrt(_EPS); left=y<=lower+eps; right=y>=upper-eps; mid=~(left|right); ll=np.zeros(len(y)); ll[left]=norm.logcdf(lower,loc=mu[left],scale=sigma); ll[right]=norm.logsf(upper,loc=mu[right],scale=sigma); ll[mid]=norm.logpdf(y[mid],loc=mu[mid],scale=sigma); return float(-np.sum(ll[np.isfinite(ll)]))


def fit_censored_normal_process_irt(response_matrix: Any,theta: Any,lower: float=0,upper: float=1,control: Mapping[str,Any]|None=None) -> EyeResult:
    if isinstance(response_matrix,pd.DataFrame): X=response_matrix.to_numpy(float); items=response_matrix.columns.astype(str).tolist()
    else: X=np.asarray(response_matrix,float); items=[f"item{i+1}" for i in range(X.shape[1] if X.ndim==2 else 0)]
    if X.ndim!=2: raise EyeProcessValidationError("response_matrix must be a two-dimensional matrix.")
    th=np.asarray(theta,float).reshape(-1)
    if X.shape[0]!=len(th): raise EyeProcessValidationError("length(theta) must equal nrow(response_matrix).")
    if lower>=upper: raise EyeProcessValidationError("lower must be < upper.")
    if np.any((X<lower)|(X>upper)): raise EyeProcessValidationError("All observed values must lie within [lower, upper].")
    fits=[]; rows=[]; maxiter=int((control or {}).get("maxit",1000))
    for j,item in enumerate(items):
        y=X[:,j]; ok=np.isfinite(y)&np.isfinite(th); yy=y[ok]; tt=th[ok]
        if len(yy)<10 or len(np.unique(yy))<2: fits.append(None); rows.append({"item":item,"discrimination":np.nan,"intercept":np.nan,"sigma":np.nan,"n":len(yy),"convergence":99,"logLik":np.nan}); continue
        slope,intercept=np.polyfit(tt,yy,1); sigma=float(np.std(yy,ddof=1)); sigma=sigma if np.isfinite(sigma) and sigma>0 else .1*(upper-lower); start=np.array([slope if np.isfinite(slope) else 1,intercept if np.isfinite(intercept) else np.mean(yy),math.log(max(sigma,1e-3))]); res=minimize(_cn_nll,start,args=(yy,tt,float(lower),float(upper)),method="BFGS",options={"maxiter":maxiter}); fits.append(res); rows.append({"item":item,"discrimination":float(res.x[0]),"intercept":float(res.x[1]),"sigma":float(math.exp(res.x[2])),"n":len(yy),"convergence":0 if res.success else int(getattr(res,"status",1)),"logLik":float(-res.fun)})
    return _result("eye_censored_normal_process_irt",coefficients=pd.DataFrame(rows),fits=fits,theta=th,lower=float(lower),upper=float(upper),engine="conditional_censored_normal_mle",status="experimental",citation="10.1007/s41237-026-00292-x",caveat="Conditional item calibration given supplied theta; not the full marginal EM estimator.")


def predict_eye_censored_normal_process_irt(object: Any,theta: Any=None,items: Sequence[str]|None=None) -> np.ndarray:
    if getattr(object,"eyeprocess_class",None)!="eye_censored_normal_process_irt": raise EyeProcessValidationError("object must be an eye_censored_normal_process_irt.")
    co=object.coefficients.copy();
    if items is not None: co=co[co.item.isin(list(items))]
    th=np.asarray(object.theta if theta is None else theta,float).reshape(-1); out=[]
    for _,r in co.iterrows():
        mu=float(r.discrimination)*th+float(r.intercept); s=float(r.sigma); a=(object.lower-mu)/s; b=(object.upper-mu)/s; out.append(object.lower*norm.cdf(a)+mu*(norm.cdf(b)-norm.cdf(a))+s*(norm.pdf(a)-norm.pdf(b))+object.upper*(1-norm.cdf(b)))
    return np.column_stack(out) if out else np.empty((len(th),0))


def process_dependent_discrimination_audit(data: Any,response: str,theta: str,process: str,person: str,item: str,nonlinear: bool=True) -> EyeResult:
    d=_df(data,[response,theta,process,person,item]).copy(); y=pd.to_numeric(d[response],errors="coerce").to_numpy(float)
    if not set(y[np.isfinite(y)]).issubset({0,1}): raise EyeProcessValidationError("response must be binary 0/1.")
    d[".process_log"]=np.log(np.maximum(pd.to_numeric(d[process],errors="coerce"),_EPS)); Xp,nm=_dummy_design(d,categorical=[person,item]); pm=_ols_fit(Xp,d[".process_log"].to_numpy(float)); d["process_residual"]=pm.residuals; d[".theta"]=pd.to_numeric(d[theta],errors="coerce"); d[".interaction"]=d[".theta"]*d["process_residual"]; X,names=_dummy_design(d,continuous=[".theta","process_residual",".interaction"],categorical=[person,item]); rm=_logistic_fit(X,y); idx=names.index(".interaction"); interaction=pd.DataFrame([{"Estimate":float(rm.coefficients[idx])}]); residual=d[[person,item,".theta","process_residual",response]].rename(columns={person:"person",item:"item",".theta":"theta",response:"y"})
    return _result("eye_process_dependent_discrimination",process_model=pm,response_model=rm,interaction=interaction,smooth_model=None,residual_process=residual,status="diagnostic-python-reference",algorithmic_parity=False,caveat="Fixed-effect residualization/logistic diagnostic corresponding to the R lme4 interaction architecture; not identical mixed-effects estimation.")


def process_channel_ablation(data: Any,channels: Mapping[str,Sequence[str]],evaluator: Callable[[Any,Sequence[str],str],float],baseline: Sequence[str]=(),higher_is_better: bool=True) -> pd.DataFrame:
    if not isinstance(channels,Mapping) or not channels: raise EyeProcessValidationError("channels must be a named mapping of column vectors.")
    if not callable(evaluator): raise EyeProcessValidationError("evaluator must be a function.")
    full=list(dict.fromkeys([*baseline,*[c for cols in channels.values() for c in cols]])); _df(data,full); full_score=float(evaluator(data,full,"full")); rows=[]
    for nm,cols in channels.items(): active=[c for c in full if c not in cols]; score=float(evaluator(data,active,f"minus_{nm}")); loss=full_score-score if higher_is_better else score-full_score; rows.append({"channel":nm,"full_score":full_score,"ablated_score":score,"information_loss":loss,"columns_removed":", ".join(cols)})
    out=pd.DataFrame(rows); out.attrs["eyeprocess_class"]="eye_process_channel_ablation"; return out


def fit_multimodal_trait_irt(data: Any,response: str,rt: str,gaze: str,person: str,item: str,trait_label: str="trait",process_label: str="process",**kwargs: Any) -> EyeResult:
    fit=fit_joint_gaze_rt_irt(data,response=response,rt=rt,gaze=gaze,person=person,item=item,**kwargs); fit.eyeprocess_class="eye_multimodal_trait_irt"; fit["trait_label"]=trait_label; fit["process_label"]=process_label; fit["status"]="experimental"; fit["citation_additional"]="10.1177/10944281261457337"; fit["caveat_multimodal_trait"]="Process channels for noncognitive traits require construct-specific validation and measurement-invariance evidence."; return fit


def generalizability_process_study(data: Any,outcome: str,facets: Sequence[str],REML: bool=True) -> EyeResult:
    facets=list(dict.fromkeys(map(str,facets))); d=_df(data,[outcome,*facets]).dropna(subset=[outcome,*facets]).copy(); fit,vc=_facet_components(d,outcome,{f:f for f in facets}); out=pd.DataFrame({"facet":vc.grp,"variance":vc.vcov,"sd":vc.sdcor}); total=float(out.variance.sum()); out["proportion"]=out.variance/total if total>0 else np.nan
    return _result("eye_process_g_study",model=fit,variance_components=out,facets=facets,outcome=outcome,status="python-reference-estimator",algorithmic_parity=False)


def cross_device_process_equating_audit(data: Any,value: str,reference_value: str,device: str,anchor: str|None=None) -> pd.DataFrame:
    d=_df(data,[value,reference_value,device]+([anchor] if anchor else [])); rows=[]
    for dev,z in d.groupby(device,sort=False,dropna=False):
        if anchor: z=z[z[anchor].notna()]
        x=pd.to_numeric(z[value],errors="coerce").to_numpy(float); y=pd.to_numeric(z[reference_value],errors="coerce").to_numpy(float); ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
        if len(x)<3: rows.append({"device":dev,"n":len(x),"A":np.nan,"B":np.nan,"bias":np.nan,"rmse":np.nan}); continue
        A,B=np.polyfit(x,y,1); pr=A*x+B; rows.append({"device":dev,"n":len(x),"A":float(A),"B":float(B),"bias":float(np.mean(pr-y)),"rmse":float(np.sqrt(np.mean((pr-y)**2)))})
    out=pd.DataFrame(rows); out.attrs["eyeprocess_class"]="eye_cross_device_equating_audit"; return out


# ---------------------------------------------------------------------------
# emerging multiple-response/process contracts
# ---------------------------------------------------------------------------

def encode_response_combinations(data: Any,person: str="participant_id",item: str="item_id",option: str="option_id",selected: str="selected",sort_options: bool=True,empty_code: str="<none>") -> pd.DataFrame:
    d=_df(data,[person,item,option,selected]); rows=[]
    for keys,z in d.groupby([person,item],sort=False,dropna=False):
        keep=z[selected].astype("boolean").fillna(False).to_numpy(bool); opts=z.loc[keep,option].astype(str).tolist(); opts=sorted(opts) if sort_options else opts; rows.append({person:str(keys[0]),item:str(keys[1]),"response_combination":"|".join(opts) if opts else empty_code,"n_selected":len(opts)})
    return pd.DataFrame(rows)


def _pairwise_corr(X: np.ndarray) -> np.ndarray:
    p=X.shape[1]; out=np.eye(p)
    for i in range(p):
        for j in range(i+1,p):
            ok=np.isfinite(X[:,i])&np.isfinite(X[:,j]); r=np.corrcoef(X[ok,i],X[ok,j])[0,1] if ok.sum()>=2 and np.nanstd(X[ok,i])>0 and np.nanstd(X[ok,j])>0 else np.nan; out[i,j]=out[j,i]=r
    return out


def audit_process_local_dependence(response_residuals: Any,process_residuals: Any=None,threshold: float=.20) -> EyeResult:
    if isinstance(response_residuals,pd.DataFrame): R=response_residuals.to_numpy(float); nm=response_residuals.columns.astype(str).tolist()
    else: R=np.asarray(response_residuals,float); nm=[f"V{i+1}" for i in range(R.shape[1] if R.ndim==2 else 0)]
    if R.ndim!=2 or R.shape[1]<2: raise EyeProcessValidationError("At least two residual columns are required.")
    rc=_pairwise_corr(R); rows=[]
    for i in range(R.shape[1]):
        for j in range(i+1,R.shape[1]): rows.append({"first":nm[i],"second":nm[j],"response_residual_correlation":rc[i,j],"response_flag":bool(np.isfinite(rc[i,j]) and abs(rc[i,j])>=threshold)})
    out=pd.DataFrame(rows)
    if process_residuals is not None:
        P=np.asarray(process_residuals,float)
        if P.shape!=R.shape: raise EyeProcessValidationError("process_residuals must have the same dimensions as response_residuals.")
        pc=_pairwise_corr(P); vals=[]
        for i in range(R.shape[1]):
            for j in range(i+1,R.shape[1]): vals.append(pc[i,j])
        out["process_residual_correlation"]=vals; out["process_flag"]=np.isfinite(vals)&(np.abs(vals)>=threshold); out["concordant_direction"]=np.sign(out.response_residual_correlation)==np.sign(out.process_residual_correlation)
    maxabs=float(np.nanmax(np.abs(out.response_residual_correlation))) if len(out) and np.isfinite(out.response_residual_correlation).any() else np.nan
    return _result("eye_process_local_dependence_audit",pairs=out,threshold=float(threshold),max_absolute_response=maxabs,note="Q3-style residual correlations diagnose local dependence; threshold flags are descriptive and require model/design context.")


def fit_multiple_response_process_irt(data: Any,selected: str="selected",theta: str="theta",person: str="participant_id",item: str="item_id",option: str="option_id",gaze: str|None=None,engine: str="reference",external_engine: Callable[...,Any]|None=None,**kwargs: Any) -> EyeResult:
    engine=_choice(engine,("reference","external"),"engine")
    if engine=="external":
        if not callable(external_engine): raise EyeProcessBackendError("Supply a validated multiple-response IRT fitter through external_engine.")
        return _result("eye_multiple_response_process_irt",model=external_engine(data=data,**kwargs),engine="external",exact_multiple_response=True,status="experimental-external")
    cols=[selected,theta,person,item,option]+([gaze] if gaze else []); d=_df(data,cols).dropna(subset=[selected,theta,person,item,option]).copy(); d[".selected"]=d[selected].astype(bool).astype(int); d[".theta"]=pd.to_numeric(d[theta],errors="coerce"); d[".item_option"]=d[item].astype(str)+"::"+d[option].astype(str); continuous=[".theta"]
    if gaze:
        gv=np.log1p(np.maximum(pd.to_numeric(d[gaze],errors="coerce").to_numpy(float),0)); sd=np.nanstd(gv,ddof=1); d[".gaze"]=(gv-np.nanmean(gv))/sd if np.isfinite(sd) and sd>0 else 0.0; d[".interaction"]=d[".theta"]*d[".gaze"]; continuous += [".gaze",".interaction"]
    X,names=_dummy_design(d,continuous=continuous,categorical=[".item_option",person]); fit=_logistic_fit(X,d[".selected"].to_numpy(float))
    return _result("eye_multiple_response_process_irt",model=fit,data=d,gaze=gaze,engine="reference",exact_multiple_response=False,status="experimental-python-reference",algorithmic_parity=False,design_names=names,note="Option-level fixed-effect logistic reference; not the full MRM/MRM-LD likelihood.")


def fit_revisit_process_cdm(response_matrix: Any,q_matrix: Any,process_data: Any,person_id: str="participant_id",revisited: str="revisited",rt: str="response_time",gaze: Sequence[str]|str|None=None,**kwargs: Any) -> EyeResult:
    R=np.asarray(response_matrix,float); Q=np.asarray(q_matrix,int)
    if R.ndim!=2 or Q.ndim!=2 or R.shape[1]!=Q.shape[0]: raise EyeProcessValidationError("response_matrix columns must align with q_matrix rows.")
    if not np.all(np.isin(Q,[0,1])): raise EyeProcessValidationError("q_matrix must contain 0/1 indicators.")
    pd0=_df(process_data,[person_id,revisited,rt]+(([gaze] if isinstance(gaze,str) else list(gaze)) if gaze is not None else [])).copy(); pd0[revisited]=pd.to_numeric(pd0[revisited],errors="coerce"); pd0[rt]=np.log1p(np.maximum(pd.to_numeric(pd0[rt],errors="coerce"),0)); features=[revisited,rt]+(([gaze] if isinstance(gaze,str) else list(gaze)) if gaze is not None else [])
    k=Q.shape[1]; profiles=np.array([[int((m>>j)&1) for j in range(k)] for m in range(2**k)],int); ideal=(profiles[:,None,:]>=Q[None,:,:]).all(axis=2).astype(float); slip=float(kwargs.get("slip",.1)); guess=float(kwargs.get("guess",.2)); P=ideal*(1-slip)+(1-ideal)*guess; scores=[]
    for row in R:
        ll=[]
        for p in P:
            ok=np.isfinite(row); ll.append(float(np.sum(row[ok]*np.log(np.clip(p[ok],_EPS,1))+(1-row[ok])*np.log(np.clip(1-p[ok],_EPS,1)))))
        scores.append(int(np.argmax(ll)))
    mastery=profiles[np.asarray(scores)]; proc=pd0.groupby(person_id,dropna=False)[features].mean(numeric_only=True).reset_index()
    return _result("eye_revisit_process_cdm",mastery_profiles=mastery,profile_index=np.asarray(scores),attribute_profiles=profiles,process_summary=proc,revisit_process_features=features,status="experimental-reference",note="Revisiting and response time are collateral process evidence; mastery labels remain defined by the Q-matrix and response model.")


# ---------------------------------------------------------------------------
# Python plot counterparts for S3-only R methods
# ---------------------------------------------------------------------------

def _plt_ax(ax: Any=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise EyeProcessBackendError("Plotting requires the 'plots' extra: pip install eyeprocesspy[plots]") from exc
    if ax is None: _,ax=plt.subplots()
    return ax


def plot_eye_process_dependent_discrimination(x: Any,ax: Any=None):
    if getattr(x,"eyeprocess_class",None)!="eye_process_dependent_discrimination": raise EyeProcessValidationError("x must be an eye_process_dependent_discrimination.")
    ax=_plt_ax(ax); d=x.residual_process; q=np.nanquantile(d.process_residual,[.1,.5,.9]); b=np.asarray(x.response_model.coefficients,float); # first 4 coefficients are intercept/theta/process/interact by construction
    th=np.linspace(float(np.nanmin(d.theta)),float(np.nanmax(d.theta)),101)
    for label,z in zip(["low","median","high"],q): ax.plot(th,expit(b[0]+b[1]*th+b[2]*z+b[3]*th*z),label=label)
    ax.set(xlabel="Theta",ylabel="Response probability",title="Process-dependent effective discrimination"); ax.legend(); ax.gp3_data=pd.DataFrame({"theta":np.tile(th,3),"process_band":np.repeat(["low","median","high"],len(th))}); return ax


def plot_eye_process_channel_ablation(x: Any,ax: Any=None):
    d=pd.DataFrame(x); ax=_plt_ax(ax); ax.bar(d.channel,d.information_loss); ax.axhline(0,linestyle="--"); ax.set(ylabel="Out-of-sample information loss",title="Process-channel ablation"); ax.tick_params(axis="x",rotation=45); ax.gp3_data=d; return ax


def plot_eye_process_g_study(x: Any,ax: Any=None):
    if getattr(x,"eyeprocess_class",None)!="eye_process_g_study": raise EyeProcessValidationError("x must be an eye_process_g_study.")
    d=x.variance_components; ax=_plt_ax(ax); ax.bar(d.facet,d.proportion); ax.set(ylabel="Variance proportion",title="Process-measure generalizability decomposition"); ax.tick_params(axis="x",rotation=45); ax.gp3_data=d; return ax


def plot_eye_process_local_dependence_audit(x: Any,ax: Any=None):
    if getattr(x,"eyeprocess_class",None)!="eye_process_local_dependence_audit": raise EyeProcessValidationError("x must be an eye_process_local_dependence_audit.")
    d=x.pairs; ax=_plt_ax(ax); labels=(d["first"].astype(str)+" / "+d["second"].astype(str)).tolist(); y=np.arange(len(d)); ax.scatter(d.response_residual_correlation,y); ax.set_yticks(y,labels); ax.axvline(-x.threshold,linestyle="--"); ax.axvline(x.threshold,linestyle="--"); ax.set(xlabel="Response residual correlation",title="Q3-style local-dependence audit"); ax.gp3_data=d; return ax
