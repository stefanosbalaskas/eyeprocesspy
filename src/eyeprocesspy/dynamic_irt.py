"""Dynamic IRTree, theory-strategy mixture, and gaze-diffusion parity layer.

Ports the dependency-light contracts from R eyeprocess 0.11.1 source files
024-dynamic-irtree-engine.R and 027-strategy-diffusion-engines.R. Canonical
Stan engines remain explicit CmdStanPy backends; baseline/EM/multinomial paths
are implemented directly with NumPy/SciPy.
"""
from __future__ import annotations

import math
from importlib import resources
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm

from .exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError
from .irt import EyeResult, _result

_EPS = np.finfo(float).eps


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be a data frame.") from exc


def _required(d: pd.DataFrame, cols: Sequence[str], label: str = "data") -> None:
    miss = [c for c in cols if c not in d.columns]
    if miss:
        raise EyeProcessValidationError(f"{label} is missing required columns: {', '.join(miss)}")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _stan_path(name: str) -> str:
    ref = resources.files("eyeprocesspy").joinpath("resources", "stan", name)
    if not ref.is_file():
        raise EyeProcessBackendError(f"Bundled Stan program is unavailable: {name}")
    return str(ref)


def _cmdstanpy() -> Any:
    try:
        import cmdstanpy  # type: ignore
    except ImportError as exc:
        raise EyeProcessBackendError(
            "This operation requires the 'stan' extra. Install with: pip install eyeprocesspy[stan]"
        ) from exc
    return cmdstanpy


def dynamic_irtree_spec(
    source: str = "samples",
    collapse_consecutive: bool = True,
    engine: str = "baseline",
    hidden_states: int = 0,
    include_response: bool = True,
    include_person: bool = False,
    include_item: bool = True,
    condition_columns: Sequence[str] | None = (),
    transition_predictors: Sequence[str] | None = (),
    interactions: Sequence[str] | None = (),
    include_time_gap: bool = True,
    person_effect: str = "none",
    item_effect: str = "none",
    structural_zeros: Any = None,
    allowed_transitions: Any = None,
    hidden_structural_zeros: Any = None,
    hidden_allowed_transitions: Any = None,
    missing_state: str = "drop",
    uncertain_state_probability: str | None = None,
    misclassification_matrix: Any = None,
    ridge: float = 1e-4,
    standardize: bool = True,
    reference_state: str | None = None,
    chains: int = 4,
    parallel_chains: int | None = None,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
) -> EyeResult:
    if int(hidden_states) == 1 or int(hidden_states) < 0:
        raise EyeProcessValidationError("hidden_states must be zero or at least two.")
    if engine not in {"baseline", "multinomial", "stan"}:
        raise EyeProcessValidationError("engine must be one of baseline, multinomial, stan.")
    if source not in {"samples", "visits", "fixations"}:
        raise EyeProcessValidationError("source must be samples, visits, or fixations.")
    if person_effect not in {"none", "fixed", "random"} or item_effect not in {"none", "fixed", "random"}:
        raise EyeProcessValidationError("person_effect and item_effect must be none, fixed, or random.")
    if missing_state not in {"drop", "unknown", "marginalize"}:
        raise EyeProcessValidationError("missing_state must be drop, unknown, or marginalize.")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0:
        raise EyeProcessValidationError("ridge must be finite and non-negative.")
    if include_person and person_effect == "none":
        person_effect = "random" if engine == "stan" else "fixed"
    if include_item and item_effect == "none":
        item_effect = "random" if engine == "stan" else "fixed"
    if hidden_states >= 2 and person_effect == "random":
        person_effect = "fixed"
    if hidden_states >= 2 and item_effect == "random":
        item_effect = "fixed"
    if parallel_chains is None:
        parallel_chains = chains
    controls = [chains, parallel_chains, iter_warmup, iter_sampling, max_treedepth]
    if any(int(v) < 1 for v in controls) or int(parallel_chains) > int(chains):
        raise EyeProcessValidationError("Stan iteration, chain, and tree-depth controls must be positive and parallel_chains <= chains.")
    if not (0 < float(adapt_delta) < 1):
        raise EyeProcessValidationError("adapt_delta must lie strictly between 0 and 1.")
    cc = [] if condition_columns is None else list(condition_columns)
    tp = [] if transition_predictors is None else list(transition_predictors)
    inter = [] if interactions is None else list(interactions)
    if any((v is None or str(v) == "") for v in cc + tp + inter):
        raise EyeProcessValidationError("Predictor and interaction names must be non-missing and non-empty.")
    return _result(
        "eye_dynamic_irtree_spec",
        source=source,
        collapse_consecutive=bool(collapse_consecutive),
        engine=engine,
        hidden_states=int(hidden_states),
        include_response=bool(include_response),
        include_person=person_effect != "none",
        include_item=item_effect != "none",
        condition_columns=list(dict.fromkeys(map(str, cc))),
        transition_predictors=list(dict.fromkeys(map(str, tp))),
        interactions=list(dict.fromkeys(map(str, inter))),
        include_time_gap=bool(include_time_gap),
        person_effect=person_effect,
        item_effect=item_effect,
        structural_zeros=structural_zeros,
        allowed_transitions=allowed_transitions,
        hidden_structural_zeros=hidden_structural_zeros,
        hidden_allowed_transitions=hidden_allowed_transitions,
        missing_state=missing_state,
        uncertain_state_probability=uncertain_state_probability,
        misclassification_matrix=misclassification_matrix,
        ridge=ridge,
        standardize=bool(standardize),
        reference_state=reference_state,
        chains=int(chains),
        parallel_chains=int(parallel_chains),
        iter_warmup=int(iter_warmup),
        iter_sampling=int(iter_sampling),
        adapt_delta=float(adapt_delta),
        max_treedepth=int(max_treedepth),
    )


def _long_to_transitions(d: pd.DataFrame, person: str, item: str, trial: str, state: str, time: str | None) -> pd.DataFrame:
    _required(d, [person, item, state], "long state data")
    if d[person].isna().any() or d[item].isna().any():
        raise EyeProcessValidationError("Participant and item identifiers must be non-missing.")
    if trial not in d:
        d[trial] = d[person].astype(str) + "::" + d[item].astype(str)
    if d[trial].isna().any():
        raise EyeProcessValidationError("Trial identifiers must be non-missing.")
    order_cols = [person, trial] + ([time] if time and time in d else [])
    d = d.sort_values(order_cols, kind="stable")
    rows: list[pd.DataFrame] = []
    for _, z in d.groupby([person, trial], sort=False, dropna=False):
        if len(z) < 2:
            continue
        out = z.iloc[:-1].copy()
        out["from_state"] = z[state].astype(str).iloc[:-1].to_numpy()
        out["to_state"] = z[state].astype(str).iloc[1:].to_numpy()
        out["step"] = np.arange(1, len(z))
        out["participant_id"] = z[person].astype(str).iloc[:-1].to_numpy()
        out["item_id"] = z[item].astype(str).iloc[:-1].to_numpy()
        out["trial_id"] = z[trial].astype(str).iloc[:-1].to_numpy()
        if time and time in z:
            a = pd.to_numeric(z[time], errors="coerce").to_numpy(float)
            out["time"] = a[:-1]
            out["next_time"] = a[1:]
            out["time_gap"] = a[1:] - a[:-1]
        else:
            out["time_gap"] = 1.0
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_dynamic_irtree_data(
    x: Any,
    spec: EyeResult | None = None,
    person: str = "participant_id",
    item: str = "item_id",
    trial: str = "trial_id",
    state: str = "state",
    time: str | None = None,
    from_: str = "from_state",
    to: str = "to_state",
    states: Sequence[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise EyeProcessValidationError(f"Unexpected arguments: {', '.join(kwargs)}")
    spec = dynamic_irtree_spec() if spec is None else spec
    if getattr(spec, "eyeprocess_class", None) != "eye_dynamic_irtree_spec":
        raise EyeProcessValidationError("spec must be created by dynamic_irtree_spec().")
    d = _df(x)
    if from_ in d and to in d:
        d["from_state"] = d[from_]
        d["to_state"] = d[to]
        if person not in d:
            d[person] = "P1"
        if item not in d:
            d[item] = "I1"
        if trial not in d:
            d[trial] = d[person].astype(str) + "::" + d[item].astype(str)
        d["participant_id"] = d[person].astype(str)
        d["item_id"] = d[item].astype(str)
        d["trial_id"] = d[trial].astype(str)
        if "step" not in d:
            d["step"] = d.groupby(["participant_id", "trial_id"], sort=False).cumcount() + 1
        if "time_gap" not in d:
            d["time_gap"] = 1.0
    else:
        d = _long_to_transitions(d, person, item, trial, state, time)
    if d.empty:
        raise EyeProcessValidationError("No transitions could be prepared.")
    for c in ("participant_id", "item_id", "trial_id"):
        if d[c].isna().any() or d[c].astype(str).eq("").any():
            raise EyeProcessValidationError("Participant, item, and trial identifiers must be non-missing and non-empty.")
    d["from_state"] = d["from_state"].where(pd.notna(d["from_state"]), None)
    d["to_state"] = d["to_state"].where(pd.notna(d["to_state"]), None)
    miss_from = d["from_state"].isna() | d["from_state"].astype(str).eq("")
    miss_to = d["to_state"].isna() | d["to_state"].astype(str).eq("")
    if spec.missing_state == "drop":
        d = d.loc[~miss_from & ~miss_to].copy()
    else:
        d.loc[miss_from, "from_state"] = "<UNKNOWN>"
        d.loc[miss_to, "to_state"] = "<UNKNOWN>"
        if spec.missing_state == "marginalize" and spec.hidden_states < 2:
            raise EyeProcessValidationError("missing_state = 'marginalize' requires the hidden-state Stan engine.")
    if d.empty:
        raise EyeProcessValidationError("No transitions remain after missing-state handling.")
    levels = list(map(str, states)) if states is not None else list(dict.fromkeys(pd.concat([d["from_state"], d["to_state"]]).astype(str)))
    if len(levels) < 2:
        raise EyeProcessValidationError("At least two states are required.")
    unknown = set(d["from_state"].astype(str)).union(d["to_state"].astype(str)) - set(levels)
    if unknown:
        raise EyeProcessValidationError(f"Unknown state labels: {', '.join(sorted(unknown))}")
    d["from_state"] = pd.Categorical(d["from_state"].astype(str), categories=levels, ordered=True)
    d["to_state"] = pd.Categorical(d["to_state"].astype(str), categories=levels, ordered=True)
    d["from_index"] = d["from_state"].cat.codes + 1
    d["to_index"] = d["to_state"].cat.codes + 1
    d["person_index"] = pd.factorize(d["participant_id"], sort=False)[0] + 1
    d["item_index"] = pd.factorize(d["item_id"], sort=False)[0] + 1
    trial_key = d["participant_id"].astype(str) + "\r" + d["trial_id"].astype(str)
    d["trial_index"] = pd.factorize(trial_key, sort=False)[0] + 1
    gap = pd.to_numeric(d["time_gap"], errors="coerce").to_numpy(dtype=float, copy=True)
    gap[(~np.isfinite(gap)) | (gap <= 0)] = 1.0
    d["time_gap"] = gap
    if spec.uncertain_state_probability:
        if spec.uncertain_state_probability not in d:
            raise EyeProcessValidationError(f"Uncertain-state probability column is absent: {spec.uncertain_state_probability}")
        p = pd.to_numeric(d[spec.uncertain_state_probability], errors="coerce").to_numpy(float)
        if np.any(~np.isfinite(p)) or np.any((p <= 0) | (p > 1)):
            raise EyeProcessValidationError("Recorded-state probabilities must be finite values in (0, 1].")
        d["state_probability"] = p
    else:
        d["state_probability"] = np.where((spec.missing_state == "marginalize") & (d["to_state"].astype(str) == "<UNKNOWN>"), 1 / len(levels), 1.0)
    d.attrs["states"] = levels
    d.attrs["eyeprocess_class"] = "eye_dynamic_transition_data"
    return d.reset_index(drop=True)


def structural_transition_mask(
    states: Sequence[str], forbidden: Any = None, allowed: Any = None, allow_self: bool = True,
    structural_zeros: Any = None, allowed_transitions: Any = None,
) -> pd.DataFrame:
    if forbidden is None and structural_zeros is not None:
        forbidden = structural_zeros
    if allowed is None and allowed_transitions is not None:
        allowed = allowed_transitions
    levels = list(dict.fromkeys(map(str, states)))
    if len(levels) < 2:
        raise EyeProcessValidationError("At least two states are required.")
    mask = pd.DataFrame(True, index=levels, columns=levels)
    if not allow_self:
        np.fill_diagonal(mask.values, False)

    def apply_pairs(pairs: Any, value: bool) -> None:
        nonlocal mask
        if pairs is None:
            return
        z = _df(pairs, "transition pairs")
        if z.shape[1] < 2:
            raise EyeProcessValidationError("Transition pairs require from and to columns.")
        for a, b in z.iloc[:, :2].itertuples(index=False, name=None):
            a, b = str(a), str(b)
            if a not in levels or b not in levels:
                raise EyeProcessValidationError(f"Unknown state in transition pair: {a} -> {b}")
            mask.loc[a, b] = value

    if allowed is not None:
        mask.loc[:, :] = False
        apply_pairs(allowed, True)
    apply_pairs(forbidden, False)
    if (~mask.any(axis=1)).any():
        raise EyeProcessValidationError("Every source state must permit at least one destination state.")
    mask.index.name = "from"
    mask.columns.name = "to"
    return mask


def _dummy_matrix(series: pd.Series, prefix: str, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(series.astype(str), prefix=prefix, drop_first=drop_first, dtype=float)


def dynamic_transition_design(data: Any, spec: EyeResult | None = None, formula: Any = None) -> EyeResult:
    spec = dynamic_irtree_spec() if spec is None else spec
    d = data.copy() if isinstance(data, pd.DataFrame) and data.attrs.get("eyeprocess_class") == "eye_dynamic_transition_data" else prepare_dynamic_irtree_data(data, spec)
    blocks: list[pd.DataFrame] = [pd.DataFrame({"(Intercept)": np.ones(len(d))})]
    if spec.hidden_states < 2:
        blocks.append(_dummy_matrix(d["from_state"], "from_state", drop_first=True))
    step = pd.to_numeric(d["step"], errors="coerce").to_numpy(float)
    blocks.append(pd.DataFrame({"poly(step,2)1": step, "poly(step,2)2": step**2}))
    if spec.include_time_gap:
        blocks.append(pd.DataFrame({"log1p(time_gap)": np.log1p(pd.to_numeric(d["time_gap"], errors="coerce").to_numpy(float))}))
    if spec.include_response and "score" in d and np.isfinite(pd.to_numeric(d["score"], errors="coerce")).any():
        blocks.append(pd.DataFrame({"score": pd.to_numeric(d["score"], errors="coerce").to_numpy(float)}))
    for c in list(spec.condition_columns) + list(spec.transition_predictors):
        if c not in d:
            continue
        numeric = pd.to_numeric(d[c], errors="coerce")
        if numeric.notna().all():
            blocks.append(pd.DataFrame({c: numeric.to_numpy(float)}))
        else:
            blocks.append(_dummy_matrix(d[c], c, drop_first=True))
    if spec.person_effect == "fixed":
        blocks.append(_dummy_matrix(d["participant_id"], "participant_id", drop_first=True))
    if spec.item_effect == "fixed":
        blocks.append(_dummy_matrix(d["item_id"], "item_id", drop_first=True))
    Xdf = pd.concat(blocks, axis=1)
    if Xdf.isna().any().any():
        keep = ~Xdf.isna().any(axis=1)
        d = d.loc[keep].reset_index(drop=True)
        Xdf = Xdf.loc[keep].reset_index(drop=True)
    scaling_rows = []
    if spec.standardize:
        for c in Xdf.columns:
            vals = Xdf[c].to_numpy(float)
            if c == "(Intercept)" or len(np.unique(vals)) <= 2 or not np.all(np.isfinite(vals)):
                continue
            center, scale = float(vals.mean()), float(vals.std(ddof=1))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            Xdf[c] = (vals - center) / scale
            scaling_rows.append({"term": c, "center": center, "scale": scale})
    levels = d.attrs.get("states") or list(d["to_state"].cat.categories)
    mask = structural_transition_mask(levels, spec.structural_zeros, spec.allowed_transitions)
    allowed_arr = np.vstack([mask.loc[str(s)].to_numpy(bool) for s in d["from_state"].astype(str)])
    return _result(
        "eye_transition_design", data=d, formula=formula, X=Xdf, y=d["to_index"].to_numpy(int), states=levels,
        allowed=allowed_arr, transition_mask=mask, scaling=pd.DataFrame(scaling_rows),
        participants=list(dict.fromkeys(d["participant_id"].astype(str))), items=list(dict.fromkeys(d["item_id"].astype(str))),
        trials=list(dict.fromkeys(d["trial_index"].astype(int))), spec=spec,
    )


def fit_multinomial_transition(
    design: EyeResult, ridge: float = 1e-4, reference_state: str | None = None,
    control: Mapping[str, Any] | None = None,
) -> EyeResult:
    if getattr(design, "eyeprocess_class", None) != "eye_transition_design":
        raise EyeProcessValidationError("Expected an eye_transition_design.")
    X = design.X.to_numpy(float) if isinstance(design.X, pd.DataFrame) else np.asarray(design.X, float)
    y = np.asarray(design.y, int) - 1
    allowed = np.asarray(design.allowed, bool)
    states = list(design.states)
    K, D = len(states), X.shape[1]
    reference_state = states[-1] if reference_state is None else str(reference_state)
    if reference_state not in states:
        raise EyeProcessValidationError(f"Unknown reference state: {reference_state}")
    ref = states.index(reference_state)
    if np.any(~allowed[np.arange(len(y)), y]):
        raise EyeProcessValidationError("Observed transitions violate the declared structural-transition mask.")
    weights = np.asarray(design.data["state_probability"], float)
    destinations = [k for k in range(K) if k != ref]

    def unpack(par: np.ndarray) -> np.ndarray:
        B = np.zeros((D, K))
        B[:, destinations] = par.reshape(D, K - 1, order="F")
        return B

    def objective(par: np.ndarray) -> float:
        B = unpack(par)
        eta = X @ B
        eta[~allowed] = -1e12
        lp = eta[np.arange(len(y)), y] - logsumexp(eta, axis=1)
        if not np.all(np.isfinite(lp)):
            return np.finfo(float).max / 10
        return float(-np.sum(weights * lp) + 0.5 * ridge * np.sum(par**2))

    opts = {"maxiter": 1000, "gtol": 1e-9}
    if control:
        if "maxit" in control:
            opts["maxiter"] = int(control["maxit"])
        if "reltol" in control:
            opts["gtol"] = float(control["reltol"])
    fit = minimize(objective, np.zeros(D * (K - 1)), method="BFGS", options=opts)
    B = unpack(fit.x)
    eta = X @ B
    eta[~allowed] = -1e12
    probs = _softmax(eta)
    try:
        cov = np.asarray(fit.hess_inv)
        sevec = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        sevec = np.full(D * (K - 1), np.nan)
    SE = np.zeros((D, K))
    SE[:, destinations] = sevec.reshape(D, K - 1, order="F")
    terms = list(design.X.columns) if isinstance(design.X, pd.DataFrame) else [f"x{i+1}" for i in range(D)]
    rows = []
    for k, state in enumerate(states):
        for j, term in enumerate(terms):
            est, se = B[j, k], SE[j, k]
            z = est / se if se > 0 else np.nan
            rows.append({"term": term, "destination": state, "estimate": est, "std_error": se, "z": z, "p_value": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan})
    p_df = pd.DataFrame(probs, columns=states)
    return _result(
        "eye_multinomial_transition", design=design, coefficients=pd.DataFrame(rows), coefficient_matrix=pd.DataFrame(B, index=terms, columns=states),
        standard_error_matrix=pd.DataFrame(SE, index=terms, columns=states), probabilities=p_df,
        fitted_state=pd.Categorical([states[i] for i in np.argmax(probs, axis=1)], categories=states, ordered=True),
        log_likelihood=float(-fit.fun + 0.5 * ridge * np.sum(fit.x**2)), convergence=0 if fit.success else 1,
        message=str(fit.message), iterations={"function": fit.nfev, "gradient": getattr(fit, "njev", None)}, reference_state=reference_state,
        ridge=float(ridge), optimizer=fit,
    )


def fit_dynamic_irtree_stan(design: EyeResult, spec: EyeResult, seed: int = 1, refresh: int = 0, output_dir: str | None = None, **kwargs: Any) -> EyeResult:
    cmdstanpy = _cmdstanpy()
    if spec.hidden_states >= 2:
        # Exact hidden-state Stan data adapter is intentionally explicit. Build the sequence contract here.
        d = design.data
        groups = [g.index.to_numpy() for _, g in d.groupby("trial_index", sort=False)]
        order = np.concatenate(groups)
        lengths = np.array([len(g) for g in groups], dtype=int)
        starts = np.cumsum(np.r_[1, lengths[:-1]])
        Kobs, H = len(design.states), int(spec.hidden_states)
        emission = np.asarray(spec.misclassification_matrix, float) if spec.misclassification_matrix is not None else np.full((H, Kobs), 1 / Kobs)
        if spec.misclassification_matrix is None:
            for k in range(min(H, Kobs)):
                emission[k] = (0.1 / max(1, Kobs - 1))
                emission[k, k] = 0.9
        if emission.shape != (H, Kobs) or np.any(emission < 0) or np.any(~np.isfinite(emission)):
            raise EyeProcessValidationError("Misclassification matrix must have hidden-state rows and observed-state columns.")
        emission = emission / emission.sum(axis=1, keepdims=True)
        hidden_labels = [f"state{i+1}" for i in range(H)]
        hmask = structural_transition_mask(hidden_labels, spec.hidden_structural_zeros, spec.hidden_allowed_transitions).to_numpy(int)
        stan_data = dict(N=len(d), S=len(groups), H=H, Kobs=Kobs, D=design.X.shape[1], X=design.X.to_numpy(float)[order],
                         observed=np.asarray(design.y, int)[order], observation_certainty=np.asarray(d["state_probability"], float)[order],
                         sequence_start=starts.astype(int), sequence_length=lengths, allowed_hidden=hmask, emission_prior=emission)
        filename, hidden = "dynamic_irtree_hidden.stan", True
    else:
        stan_data = dict(N=design.X.shape[0], K=len(design.states), D=design.X.shape[1], P=len(design.participants), J=len(design.items),
                         X=design.X.to_numpy(float), y=np.asarray(design.y, int), person=np.asarray(design.data["person_index"], int),
                         item=np.asarray(design.data["item_index"], int), from_state=np.asarray(design.data["from_index"], int),
                         allowed=np.asarray(design.allowed, int), observation_weight=np.asarray(design.data["state_probability"], float),
                         use_person_re=int(spec.person_effect == "random"), use_item_re=int(spec.item_effect == "random"))
        filename, hidden, order = "dynamic_irtree_observed.stan", False, np.arange(design.X.shape[0])
    try:
        model = cmdstanpy.CmdStanModel(stan_file=_stan_path(filename))
        fit = model.sample(data=stan_data, seed=int(seed), chains=spec.chains, parallel_chains=spec.parallel_chains,
                           iter_warmup=spec.iter_warmup, iter_sampling=spec.iter_sampling, adapt_delta=spec.adapt_delta,
                           max_treedepth=spec.max_treedepth, refresh=refresh, output_dir=output_dir, **kwargs)
    except Exception as exc:
        raise EyeProcessBackendError(f"CmdStan dynamic IRTree fitting failed: {exc}") from exc
    summary = fit.summary()
    return _result("eye_dynamic_irtree_stan", design=design, spec=spec, fit=fit, summary=summary, diagnostics=pd.DataFrame(), hidden=hidden, sequence_order=np.asarray(order, int))


def decode_dynamic_states(object: EyeResult, method: str = "mode") -> pd.DataFrame:
    if method not in {"mode", "probability", "draw"}:
        raise EyeProcessValidationError("method must be mode, probability, or draw.")
    model = object.get("model")
    if getattr(model, "eyeprocess_class", None) == "eye_multinomial_transition":
        p = model.probabilities.copy()
        if method == "probability":
            out = p.copy(); out["transition"] = np.arange(1, len(out) + 1); return out
        rng = np.random.default_rng()
        if method == "draw":
            idx = [rng.choice(p.shape[1], p=row.to_numpy(float)) for _, row in p.iterrows()]
        else:
            idx = np.argmax(p.to_numpy(float), axis=1)
        return pd.DataFrame({"transition": np.arange(1, len(p) + 1), "decoded_state": [p.columns[i] for i in idx]})
    raise EyeProcessModelError("State decoding is unavailable for this engine or requires a fitted hidden-state Stan posterior.")


def dynamic_posterior_predictive_check(object: EyeResult, draws: int = 200, seed: int = 1) -> EyeResult:
    if getattr(object.get("model"), "eyeprocess_class", None) != "eye_dynamic_irtree_stan":
        raise EyeProcessModelError("Posterior predictive checks require a Stan dynamic IRTree fit.")
    model = object.model
    variable = "observed_rep" if model.hidden else "y_rep"
    try:
        arr = model.fit.draws_pd(vars=[variable])
    except Exception as exc:
        raise EyeProcessModelError("Posterior predictive draws are unavailable.") from exc
    cols = [c for c in arr.columns if c.startswith(variable + "[")]
    mat = arr[cols].to_numpy(int)
    rng = np.random.default_rng(seed)
    if len(mat) > draws:
        mat = mat[rng.choice(len(mat), draws, replace=False)]
    obs = np.asarray(object.design.y, int)
    states = list(object.states)
    obs_freq = np.bincount(obs, minlength=len(states) + 1)[1:] / len(obs)
    repl = np.vstack([np.bincount(r, minlength=len(states) + 1)[1:] / len(r) for r in mat])
    summary = pd.DataFrame({"state": states, "observed": obs_freq, "replicated_mean": repl.mean(0),
                            "replicated_lower": np.quantile(repl, .025, axis=0), "replicated_upper": np.quantile(repl, .975, axis=0),
                            "posterior_predictive_p": (repl >= obs_freq).mean(0)})
    return _result("eye_dynamic_ppc", summary=summary, replicated_frequency=repl, observed_state=obs, engine="stan", hidden=model.hidden)


def transition_residual_diagnostics(object: EyeResult, type: str = "pearson") -> EyeResult:
    if type not in {"pearson", "deviance", "randomized"}:
        raise EyeProcessValidationError("type must be pearson, deviance, or randomized.")
    if getattr(object, "eyeprocess_class", None) != "eye_dynamic_irtree":
        raise EyeProcessValidationError("Expected an eye_dynamic_irtree.")
    d = object.transitions
    if getattr(object.model, "eyeprocess_class", None) != "eye_multinomial_transition":
        raise EyeProcessModelError("Residual diagnostics currently require baseline/multinomial probabilities; Stan fits use posterior predictive checks.")
    p = object.model.probabilities.to_numpy(float)
    observed = np.asarray(object.model.design.y, int) - 1
    op = p[np.arange(len(p)), observed]
    if type == "pearson":
        residual = (1 - op) / np.sqrt(np.maximum(op * (1 - op), 1e-8))
    elif type == "deviance":
        residual = np.sqrt(-2 * np.log(np.maximum(op, 1e-12)))
    else:
        rng = np.random.default_rng()
        lower = np.array([p[i, :observed[i]].sum() for i in range(len(p))])
        residual = norm.ppf(rng.uniform(lower, lower + op))
    rows = pd.DataFrame({"transition": np.arange(1, len(d) + 1), "observed_state": d["to_state"].astype(str), "fitted_probability": op,
                         "residual": residual, "participant_id": d["participant_id"].astype(str), "item_id": d["item_id"].astype(str),
                         "trial_id": d["trial_id"].astype(str), "step": d["step"].to_numpy()})
    summary = pd.DataFrame([{"n": len(rows), "mean": float(np.mean(residual)), "sd": float(np.std(residual, ddof=1)),
                             "rmse": float(np.sqrt(np.mean(residual**2))), "max_absolute": float(np.max(np.abs(residual)))}])
    return _result("eye_transition_diagnostics", type=type, residuals=rows, summary=summary)


def compare_dynamic_transition_models(*models: Any, criterion: str = "AIC", **named: Any) -> pd.DataFrame:
    if len(models) == 1 and isinstance(models[0], Mapping) and all(getattr(v, "eyeprocess_class", None) == "eye_dynamic_irtree" for v in models[0].values()):
        pairs = list(models[0].items())
    else:
        pairs = [(f"model{i+1}", m) for i, m in enumerate(models)] + list(named.items())
    if not pairs:
        raise EyeProcessValidationError("At least one model is required.")
    rows = []
    for name, obj in pairs:
        if getattr(obj, "eyeprocess_class", None) != "eye_dynamic_irtree":
            raise EyeProcessValidationError("All inputs must be dynamic IRTree fits.")
        n = len(obj.transitions); ll = np.nan; k = np.nan; acc = np.nan; ls = np.nan
        if getattr(obj.model, "eyeprocess_class", None) == "eye_multinomial_transition":
            ll = obj.model.log_likelihood
            co = obj.model.coefficients
            k = int(((np.isfinite(co["estimate"])) & (co["destination"] != obj.model.reference_state)).sum())
            p = obj.model.probabilities.to_numpy(float); y = np.asarray(obj.model.design.y, int) - 1
            acc = float(np.mean(np.argmax(p, axis=1) == y)); ls = float(np.mean(np.log(np.maximum(p[np.arange(n), y], 1e-12))))
        rows.append({"model": name, "engine": obj.spec.engine, "n": n, "parameters": k, "log_likelihood": ll,
                     "AIC": -2 * ll + 2 * k if np.isfinite(ll) else np.nan,
                     "BIC": -2 * ll + math.log(n) * k if np.isfinite(ll) else np.nan, "log_score": ls, "accuracy": acc})
    out = pd.DataFrame(rows)
    metric = criterion if criterion in out else "AIC"
    vals = out[metric].to_numpy(float)
    vals = vals if metric in {"AIC", "BIC"} else -vals
    order = np.argsort(vals)
    rank = np.full(len(vals), np.nan); rank[order[np.isfinite(vals[order])]] = np.arange(1, np.isfinite(vals).sum() + 1)
    out["rank"] = rank
    out.attrs["eyeprocess_class"] = "eye_dynamic_model_comparison"
    return out


def simulate_dynamic_irtree_data(
    n_person: int = 100, n_item: int = 20, transitions_per_trial: int = 8,
    states: Sequence[str] = ("prompt", "evidence", "options"), beta_response: float = 0.5,
    person_sd: float = 0.4, item_sd: float = 0.3, irregular_time: bool = True,
    state_misclassification: float = 0, missing_state: float = 0, structural_zeros: Any = None, seed: int = 1,
) -> EyeResult:
    n_person, n_item, transitions_per_trial = int(n_person), int(n_item), int(transitions_per_trial)
    if n_person < 2 or n_item < 2 or transitions_per_trial < 2:
        raise EyeProcessValidationError("Simulation sizes are too small.")
    if any((not np.isfinite(v) or v < 0 or v >= 1) for v in (state_misclassification, missing_state)):
        raise EyeProcessValidationError("Misclassification and missing-state rates must lie in [0, 1).")
    if person_sd < 0 or item_sd < 0:
        raise EyeProcessValidationError("Person and item heterogeneity SDs must be finite and non-negative.")
    levels = list(dict.fromkeys(map(str, states))); K = len(levels)
    mask = structural_transition_mask(levels, structural_zeros).to_numpy(bool)
    rng = np.random.default_rng(seed)
    persons = [f"P{i+1}" for i in range(n_person)]; items = [f"I{i+1}" for i in range(n_item)]
    theta, difficulty = rng.normal(size=n_person), rng.normal(size=n_item)
    pe, ie = rng.normal(0, person_sd, (n_person, K)), rng.normal(0, item_sd, (n_item, K))
    base = rng.normal(0, .4, (K, K)); base[np.arange(K), np.arange(K)] += .8
    rows = []
    for p in range(n_person):
        for j in range(n_item):
            score = int(rng.random() < expit(theta[p] - difficulty[j])); frm = int(rng.integers(K)); t = 0.0
            for step in range(1, transitions_per_trial + 1):
                eta = base[frm] + pe[p] + ie[j] + beta_response * score * np.linspace(-.5, .5, K)
                eta[~mask[frm]] = -1e12; prob = np.exp(eta - np.max(eta)); prob /= prob.sum()
                dest = int(rng.choice(K, p=prob)); gap = float(rng.exponential(.5)) if irregular_time else 1.0
                obs = int(rng.choice([k for k in range(K) if k != dest])) if rng.random() < state_misclassification and K > 1 else dest
                label: Any = levels[obs] if rng.random() >= missing_state else np.nan
                rows.append({"participant_id": persons[p], "item_id": items[j], "trial_id": f"{persons[p]}-{items[j]}", "step": step,
                             "time": t, "time_gap": gap, "from_state": levels[frm], "true_to_state": levels[dest], "to_state": label, "score": score})
                t += gap; frm = dest
    return _result("eye_dynamic_irtree_simulation", transitions=pd.DataFrame(rows), truth={"base_transition": base, "beta_response": beta_response,
                   "person_sd": person_sd, "item_sd": item_sd, "state_misclassification": state_misclassification, "missing_state": missing_state,
                   "states": levels, "mask": mask})


def dynamic_irtree_recovery(grid: Any = None, replications: int = 20, spec: EyeResult | None = None, base_seed: int = 1) -> EyeResult:
    if grid is None:
        grid = pd.DataFrame([(a, b) for a in (0, .05, .15) for b in (0, .10)], columns=["state_misclassification", "missing_state"])
    grid = _df(grid)
    jobs=[]; jid=0
    for _, row in grid.iterrows():
        for r in range(1, int(replications)+1):
            jid += 1; jobs.append({**row.to_dict(), "replication": r, "seed": int(base_seed)+jid-1})
    plan = _result("eye_validation_job_plan", plan_id=f"dynamic_irtree-{base_seed}-{len(jobs)}", jobs=pd.DataFrame(jobs), model_family="dynamic_irtree")
    return _result("eye_dynamic_recovery", plan=plan, spec=dynamic_irtree_spec(engine="multinomial") if spec is None else spec,
                   purpose="recovery under state misclassification; not automatic construct validation")


def fit_dynamic_irtree(x: Any, spec: EyeResult | None = None, min_transitions: int = 10, seed: int = 1, **kwargs: Any) -> EyeResult:
    spec = dynamic_irtree_spec() if spec is None else spec
    d = prepare_dynamic_irtree_data(x, spec); design = dynamic_transition_design(d, spec)
    if spec.engine == "multinomial":
        model = fit_multinomial_transition(design, ridge=spec.ridge, reference_state=spec.reference_state, control=kwargs.pop("control", None))
        co = model.coefficients; diag = pd.DataFrame([{"converged": model.convergence == 0, "optimizer_code": model.convergence, "log_likelihood": model.log_likelihood}])
    elif spec.engine == "stan":
        model = fit_dynamic_irtree_stan(design, spec, seed=seed, **kwargs); co = model.summary; diag = model.diagnostics
    else:
        # Stable one-vs-rest baseline with a shared design; unlike the multinomial model it is descriptive.
        fits={}; rows=[]; probs=np.zeros((len(d), len(design.states)))
        X=design.X.to_numpy(float)
        for k,state in enumerate(design.states):
            y=(np.asarray(design.y)==k+1).astype(float)
            if y.sum()<min_transitions or (1-y).sum()<min_transitions: continue
            def obj(b):
                p=np.clip(expit(X@b),1e-12,1-1e-12); return float(-np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
            f=minimize(obj,np.zeros(X.shape[1]),method="BFGS"); fits[state]=f; probs[:,k]=expit(X@f.x)
            for term,est in zip(design.X.columns,f.x): rows.append({"destination":state,"term":term,"estimate":est})
        if not fits: raise EyeProcessModelError("No destination state had sufficient transition support.")
        probs=probs/np.maximum(probs.sum(1,keepdims=True),_EPS)
        model=_result("eye_multinomial_transition",design=design,probabilities=pd.DataFrame(probs,columns=design.states),
                      fitted_state=pd.Categorical([design.states[i] for i in np.argmax(probs,1)],categories=design.states),
                      coefficients=pd.DataFrame(rows),reference_state=design.states[-1],log_likelihood=np.nan,convergence=0)
        co=pd.DataFrame(rows); diag=pd.DataFrame([{"converged":True,"models":len(fits)}])
    return _result("eye_dynamic_irtree", spec=spec, transitions=design.data, design=design, states=design.states, model=model,
                   fits={} if spec.engine!="baseline" else fits, coefficients=co, fit_warnings={}, diagnostics=diag,
                   warning="Observed AOI states, inferred hidden states, and transition parameters are not named cognitive states without external experimental validation.")


# ---- Theory-constrained strategy mixtures ---------------------------------

def theory_strategy_spec(
    strategies: Any = None, feature_columns: Sequence[str] | None = None, response: str = "score",
    participant: str = "participant_id", item: str = "item_id", condition: str | None = None,
    item_availability: Any = None, engine: str = "em", multiple_starts: int = 10, anchor_strength: float = 3,
    chains: int = 4, parallel_chains: int | None = None, iter_warmup: int = 1000, iter_sampling: int = 1000,
    adapt_delta: float = .95, max_treedepth: int = 12, prototypes: Any = None, feature_sd: Any = None, prior: Any = None,
) -> EyeResult:
    if engine not in {"em","stan"}: raise EyeProcessValidationError("engine must be em or stan.")
    if strategies is None and prototypes is not None: strategies=prototypes
    legacy = isinstance(strategies,(pd.DataFrame,np.ndarray))
    if legacy:
        proto=np.asarray(strategies,float)
        if proto.ndim!=2 or proto.shape[0]<2 or proto.shape[1]<1 or np.any(~np.isfinite(proto)): raise EyeProcessValidationError("prototypes must contain at least two strategies and one finite numeric feature.")
        if isinstance(strategies,pd.DataFrame): names=list(map(str,strategies.index)); cols=list(map(str,strategies.columns))
        else: names=[f"strategy_{i+1}" for i in range(proto.shape[0])]; cols=list(feature_columns or [f"feature_{j+1}" for j in range(proto.shape[1])])
        fs=np.ones(proto.shape[1]) if feature_sd is None else np.broadcast_to(np.asarray(feature_sd,float), (proto.shape[1],)).copy()
        pr=np.ones(proto.shape[0])/proto.shape[0] if prior is None else np.asarray(prior,float); pr=pr/pr.sum()
    else:
        if not isinstance(strategies, Mapping) or len(strategies)<2: raise EyeProcessValidationError("strategies must be a uniquely named list of theory-defined signatures or a prototype matrix.")
        names=list(map(str,strategies.keys()))
        if any(not n for n in names) or len(set(names))!=len(names): raise EyeProcessValidationError("Strategy names must be unique and non-empty.")
        if feature_columns is None:
            cols=[]
            for s in strategies.values():
                if not isinstance(s,Mapping):
                    raise EyeProcessValidationError("Every strategy signature must be a named numeric vector; use a mapping in Python.")
                for c in s:
                    if str(c) not in cols: cols.append(str(c))
        else: cols=list(map(str,feature_columns))
        proto=np.zeros((len(names),len(cols)))
        for k,s in enumerate(strategies.values()):
            if isinstance(s,Mapping):
                unknown=set(map(str,s.keys()))-set(cols)
                if unknown: raise EyeProcessValidationError(f"Unknown signature features: {', '.join(sorted(unknown))}.")
                for c,v in s.items(): proto[k,cols.index(str(c))]=float(v)
            else:
                arr=np.asarray(s,float)
                if arr.size!=len(cols): raise EyeProcessValidationError("Unnamed strategy signatures must match feature_columns.")
                proto[k]=arr
        fs=np.ones(len(cols)); pr=np.ones(len(names))/len(names)
    norms=np.sqrt((proto**2).sum(1))
    if np.any(norms==0): raise EyeProcessValidationError("Every strategy signature must contain at least one non-zero anchor.")
    signatures=proto/norms[:,None]
    if int(multiple_starts)<1 or anchor_strength<0: raise EyeProcessValidationError("multiple_starts must be positive and anchor_strength non-negative.")
    if parallel_chains is None: parallel_chains=min(4,chains)
    return _result("eye_theory_strategy_spec", strategies=names, signatures=pd.DataFrame(signatures,index=names,columns=cols), prototypes=pd.DataFrame(proto,index=names,columns=cols),
                   feature_sd=np.asarray(fs,float), prior=np.asarray(pr,float), legacy_mode=legacy, feature_columns=cols,response=response,participant=participant,item=item,
                   condition=condition,item_availability=item_availability,engine=engine,multiple_starts=int(multiple_starts),anchor_strength=float(anchor_strength),chains=int(chains),
                   parallel_chains=int(parallel_chains),iter_warmup=int(iter_warmup),iter_sampling=int(iter_sampling),adapt_delta=float(adapt_delta),max_treedepth=int(max_treedepth),
                   interpretation="Strategy labels are prespecified theoretical hypotheses; estimated classes are not automatically cognitive strategies.")


def prepare_strategy_mixture_data(data: Any, spec: EyeResult, standardize: bool = True) -> EyeResult:
    d=_df(data); req=[spec.participant,spec.item,spec.response,*spec.feature_columns]+(([spec.condition]) if spec.condition else [])
    _required(d,req); d=d.dropna(subset=[spec.participant,spec.item,spec.response,*spec.feature_columns]).copy()
    if d.empty: raise EyeProcessValidationError("No complete strategy-mixture rows remain.")
    y=pd.to_numeric(d[spec.response],errors="coerce").to_numpy(int)
    if not np.isin(y,[0,1]).all(): raise EyeProcessValidationError("Strategy-mixture responses must be coded 0/1.")
    X=d[spec.feature_columns].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    center=np.zeros(X.shape[1]); scale=np.ones(X.shape[1])
    if standardize:
        center=X.mean(0); scale=X.std(0,ddof=1); scale[(~np.isfinite(scale))|(scale==0)]=1; X=(X-center)/scale
    plevel=list(dict.fromkeys(d[spec.participant].astype(str))); ilevel=list(dict.fromkeys(d[spec.item].astype(str)))
    pidx=np.array([plevel.index(v)+1 for v in d[spec.participant].astype(str)]); iidx=np.array([ilevel.index(v)+1 for v in d[spec.item].astype(str)])
    avail_item=pd.DataFrame(1,index=ilevel,columns=spec.strategies,dtype=int)
    if spec.item_availability is not None:
        z=_df(spec.item_availability)
        if set(["item_id","strategy","available"]).issubset(z.columns):
            for _,r in z.iterrows():
                if str(r.item_id) in avail_item.index and str(r.strategy) in avail_item.columns: avail_item.loc[str(r.item_id),str(r.strategy)]=int(bool(r.available))
    if (avail_item.sum(1)==0).any(): raise EyeProcessValidationError("Every item must permit at least one strategy.")
    availability=np.vstack([avail_item.loc[v].to_numpy(int) for v in d[spec.item].astype(str)])
    clevel=list(dict.fromkeys(d[spec.condition].astype(str))) if spec.condition else ["all"]
    cidx=np.array([clevel.index(v)+1 for v in d[spec.condition].astype(str)]) if spec.condition else np.ones(len(d),int)
    return _result("eye_strategy_data",data=d,X=X,y=y,signatures=spec.signatures.to_numpy(float),participant=pidx,item=iidx,condition=cidx,
                   participant_levels=plevel,item_levels=ilevel,condition_levels=clevel,availability=availability,availability_item=avail_item,center=center,scale=scale,spec=spec)


def _strategy_loglik(prep: EyeResult, means: np.ndarray, variance: np.ndarray, intercept: np.ndarray, ability: np.ndarray, difficulty: np.ndarray, mixing: np.ndarray):
    N,K=prep.X.shape[0],means.shape[0]; lc=np.full((N,K),-np.inf)
    for k in range(K):
        diff=prep.X-means[k]; pll=-.5*np.sum(diff**2/variance[k]+np.log(2*np.pi*variance[k]),axis=1)
        eta=intercept[k]+ability[prep.participant-1]-difficulty[prep.item-1]; p=np.clip(expit(eta),1e-12,1-1e-12)
        rll=prep.y*np.log(p)+(1-prep.y)*np.log(1-p); lc[:,k]=np.log(max(mixing[k],1e-12))+pll+rll; lc[prep.availability[:,k]==0,k]=-np.inf
    normz=logsumexp(lc,axis=1); prob=np.exp(lc-normz[:,None]); return float(normz.sum()),prob


def fit_strategy_mixture_em(prepared: EyeResult, starts: int | None = None, max_iter: int = 300, tolerance: float = 1e-7, seed: int = 1) -> EyeResult:
    if getattr(prepared,"eyeprocess_class",None)!="eye_strategy_data": raise EyeProcessValidationError("Expected prepared strategy-mixture data.")
    starts=prepared.spec.multiple_starts if starts is None else int(starts); rng=np.random.default_rng(seed); K,F=prepared.signatures.shape; P=len(prepared.participant_levels); J=len(prepared.item_levels)
    results=[]
    for s in range(starts):
        means=prepared.signatures*prepared.spec.anchor_strength+rng.normal(0,.15,(K,F)); var=np.ones((K,F)); intercept=rng.normal(0,.25,K); ability=np.zeros(P); difficulty=np.zeros(J); mixing=np.ones(K)/K; history=[]; converged=False
        for it in range(1,int(max_iter)+1):
            ll,prob=_strategy_loglik(prepared,means,var,intercept,ability,difficulty,mixing); history.append(ll); w=prob.sum(0); mixing=w/w.sum()
            for k in range(K):
                if w[k]<=1e-8: continue
                empirical=(prepared.X*prob[:,k,None]).sum(0)/w[k]; means[k]=(empirical*w[k]+prepared.signatures[k]*prepared.spec.anchor_strength)/(w[k]+prepared.spec.anchor_strength)
                diff=prepared.X-means[k]; var[k]=np.maximum((diff**2*prob[:,k,None]).sum(0)/w[k],.05)
                # weighted intercept-only logistic response update; person/item effects remain centered zero in dependency-light EM baseline.
                ybar=np.clip((prepared.y*prob[:,k]).sum()/w[k],1e-6,1-1e-6); intercept[k]=math.log(ybar/(1-ybar))
            if len(history)>1 and abs(history[-1]-history[-2])<=tolerance*(1+abs(history[-2])): converged=True; break
        ll,prob=_strategy_loglik(prepared,means,var,intercept,ability,difficulty,mixing)
        results.append(dict(means=means,variance=var,intercept=intercept,ability=ability,difficulty=difficulty,mixing=mixing,posterior=prob,log_likelihood=ll,history=np.asarray(history),converged=converged,iterations=it,start=s+1))
    scores=np.array([r["log_likelihood"] for r in results]); best=results[int(np.argmax(scores))]; best["all_starts"]=pd.DataFrame({"start":np.arange(1,starts+1),"log_likelihood":scores,"converged":[r["converged"] for r in results],"iterations":[r["iterations"] for r in results]}); best["prepared"]=prepared
    return EyeResult(best,eyeprocess_class="eye_strategy_mixture_em")


def fit_strategy_mixture_stan(prepared: EyeResult, seed: int = 1, refresh: int = 0, output_dir: str | None = None, **kwargs: Any) -> EyeResult:
    cmdstanpy=_cmdstanpy(); spec=prepared.spec
    data=dict(N=len(prepared.y),K=prepared.signatures.shape[0],F=prepared.X.shape[1],P=len(prepared.participant_levels),J=len(prepared.item_levels),X=prepared.X,y=prepared.y,person=prepared.participant,item=prepared.item,available=np.asarray(prepared.availability,int),signature=np.asarray(prepared.signatures,float),anchor_strength=spec.anchor_strength)
    try:
        fit=cmdstanpy.CmdStanModel(stan_file=_stan_path("theory_strategy_mixture.stan")).sample(data=data,seed=seed,chains=spec.chains,parallel_chains=spec.parallel_chains,iter_warmup=spec.iter_warmup,iter_sampling=spec.iter_sampling,adapt_delta=spec.adapt_delta,max_treedepth=spec.max_treedepth,refresh=refresh,output_dir=output_dir,**kwargs)
    except Exception as exc: raise EyeProcessBackendError(f"CmdStan strategy-mixture fitting failed: {exc}") from exc
    return _result("eye_strategy_mixture_stan",prepared=prepared,spec=spec,fit=fit,summary=fit.summary(),diagnostics=pd.DataFrame())


def fit_theory_strategy_irt(data: Any, spec: EyeResult, seed: int = 1, response: str | None = None, participant: str | None = None, item: str | None = None, **kwargs: Any) -> EyeResult:
    if response: spec["response"]=response
    if participant: spec["participant"]=participant
    if item: spec["item"]=item
    prep=prepare_strategy_mixture_data(data,spec); model=fit_strategy_mixture_em(prep,seed=seed,**kwargs) if spec.engine=="em" else fit_strategy_mixture_stan(prep,seed=seed,**kwargs)
    return _result("eye_theory_strategy_irt",spec=spec,prepared=prep,model=model,interpretation=spec.interpretation)


def strategy_posterior_probabilities(object: EyeResult) -> pd.DataFrame:
    if getattr(object,"eyeprocess_class",None)!="eye_theory_strategy_irt": raise EyeProcessValidationError("Expected an eye_theory_strategy_irt object.")
    if getattr(object.model,"eyeprocess_class",None)=="eye_strategy_mixture_em": p=np.asarray(object.model.posterior,float)
    else:
        try:
            cols=object.model.fit.draws_pd(vars=["posterior_probability"]); p=cols.to_numpy(float).mean(0).reshape(len(object.prepared.y),len(object.spec.strategies))
        except Exception as exc: raise EyeProcessModelError("Posterior strategy probabilities are unavailable.") from exc
    out=pd.DataFrame(p,columns=object.spec.strategies); out["trial"]=np.arange(1,len(out)+1); out["participant_id"]=[object.prepared.participant_levels[i-1] for i in object.prepared.participant]; out["item_id"]=[object.prepared.item_levels[i-1] for i in object.prepared.item]
    out["modal_strategy"]=[object.spec.strategies[i] for i in np.argmax(p,1)]; out["modal_probability"]=p.max(1); out["entropy"]=-np.sum(np.where(p>0,p*np.log(p),0),axis=1); return out


def strategy_classification_uncertainty(object: EyeResult, threshold: float = .70) -> EyeResult:
    if not 0<=float(threshold)<=1: raise EyeProcessValidationError("threshold must be a single probability in [0, 1].")
    p=strategy_posterior_probabilities(object); summary=pd.DataFrame([{"trials":len(p),"mean_entropy":p.entropy.mean(),"median_modal_probability":p.modal_probability.median(),"uncertain_fraction":float((p.modal_probability<threshold).mean()),"threshold":threshold}]); return _result("eye_strategy_uncertainty",trial=p,summary=summary)


def strategy_label_switching_diagnostics(object: EyeResult, tolerance: float = 1e-4) -> pd.DataFrame:
    if getattr(object.model,"eyeprocess_class",None)=="eye_strategy_mixture_em":
        out=object.model.all_starts.copy(); best=out.log_likelihood.max(); out["delta_best"]=best-out.log_likelihood; out["equivalent_optimum"]=out.delta_best<=tolerance*(1+abs(best)); out["label_anchor"]="theory_signature"; out.attrs["eyeprocess_class"]="eye_strategy_label_diagnostics"; return out
    return pd.DataFrame([{"engine":object.spec.engine,"assessed":False,"reason":"Label diagnostics are unavailable for this engine."}])


def strategy_aoi_sensitivity(datasets: Mapping[str,Any] | Sequence[Any], spec: EyeResult, seed: int = 1, **kwargs: Any) -> EyeResult:
    if isinstance(datasets,Mapping): pairs=list(datasets.items())
    else: pairs=[(f"definition{i+1}",v) for i,v in enumerate(datasets)]
    if not pairs: raise EyeProcessValidationError("datasets must be non-empty.")
    fits={}; rows=[]
    for i,(name,data) in enumerate(pairs):
        fit=fit_theory_strategy_irt(data,spec,seed=seed+i,**kwargs); fits[name]=fit; p=strategy_posterior_probabilities(fit); shares=p.modal_strategy.value_counts(normalize=True)
        for strategy in spec.strategies: rows.append({"definition":name,"strategy":strategy,"modal_share":float(shares.get(strategy,0)),"mean_entropy":float(p.entropy.mean())})
    return _result("eye_strategy_aoi_sensitivity",fits=fits,summary=pd.DataFrame(rows),spec=spec)


def validate_strategy_manipulation(object: EyeResult, condition: str, expected_strategy: str, minimum_contrast: float = 0) -> EyeResult:
    p=strategy_posterior_probabilities(object); d=object.prepared.data.reset_index(drop=True)
    if condition not in d: raise EyeProcessValidationError(f"Condition column is absent: {condition}")
    if expected_strategy not in object.spec.strategies: raise EyeProcessValidationError(f"Unknown strategy: {expected_strategy}")
    temp=pd.DataFrame({"condition":d[condition].astype(str),"probability":p[expected_strategy]}); summ=temp.groupby("condition",sort=False).probability.agg(["mean","count"]).reset_index()
    contrast=float(summ["mean"].max()-summ["mean"].min()) if len(summ)>1 else 0.0
    return _result("eye_strategy_manipulation_validation",strategy=expected_strategy,summary=summ,contrast=contrast,minimum_contrast=float(minimum_contrast),supported=contrast>=minimum_contrast)


def compare_strategy_heterogeneity(object: EyeResult) -> pd.DataFrame:
    p=strategy_posterior_probabilities(object); rows=[]
    for strategy in object.spec.strategies:
        v=p[strategy].to_numpy(float); rows.append({"strategy":strategy,"mean_probability":v.mean(),"sd_probability":v.std(ddof=1),"modal_share":float((p.modal_strategy==strategy).mean())})
    out=pd.DataFrame(rows); out.attrs["eyeprocess_class"]="eye_strategy_heterogeneity"; return out


def simulate_strategy_mixture_data(n_person: int = 100, n_item: int = 20, signatures: Any = None, trials_per_item: int = 1, strategy_prevalence: Any = None, feature_sd: float = .6, seed: int = 1) -> pd.DataFrame:
    if signatures is None: signatures=np.array([[1,1],[-1,.2]],float)
    if isinstance(signatures,pd.DataFrame): names=list(map(str,signatures.index)); cols=list(map(str,signatures.columns)); S=signatures.to_numpy(float)
    elif isinstance(signatures,Mapping):
        spec=theory_strategy_spec(signatures); names=spec.strategies; cols=spec.feature_columns; S=spec.prototypes.to_numpy(float)
    else:
        S=np.asarray(signatures,float); names=[f"strategy_{i+1}" for i in range(S.shape[0])]; cols=[f"feature_{j+1}" for j in range(S.shape[1])]
    if n_person<2 or n_item<2 or trials_per_item<1 or S.ndim!=2 or S.shape[0]<2: raise EyeProcessValidationError("Simulation sizes/signatures are invalid.")
    prev=np.ones(S.shape[0])/S.shape[0] if strategy_prevalence is None else np.asarray(strategy_prevalence,float); prev=prev/prev.sum(); rng=np.random.default_rng(seed); theta=rng.normal(size=n_person); diff=rng.normal(size=n_item); rows=[]
    for p in range(n_person):
        for j in range(n_item):
            for r in range(trials_per_item):
                k=int(rng.choice(S.shape[0],p=prev)); feat=S[k]+rng.normal(0,feature_sd,S.shape[1]); y=int(rng.random()<expit(theta[p]-diff[j]+np.linspace(-.35,.35,S.shape[0])[k])); row={"participant_id":f"P{p+1}","item_id":f"I{j+1}","repetition":r+1,"score":y,"true_strategy":names[k]}; row.update(dict(zip(cols,feat))); rows.append(row)
    out=pd.DataFrame(rows); out.attrs["truth"]={"strategy_prevalence":prev,"signatures":S,"theta":theta,"difficulty":diff}; return out


# ---- Gaze diffusion --------------------------------------------------------

def gaze_diffusion_spec(response: str="score", response_time: str="response_time", participant: str="participant_id", item: str="item_id",
                        drift_features: Sequence[str]=(), boundary_features: Sequence[str]=(), nondecision_features: Sequence[str]=(), starting_features: Sequence[str]=(),
                        censor_column: str|None=None, contaminant: bool=True, engine: str="baseline", gaze_features: Sequence[str]|None=None,
                        chains:int=4, parallel_chains:int|None=None, iter_warmup:int=1000, iter_sampling:int=1000, adapt_delta:float=.97, max_treedepth:int=13) -> EyeResult:
    if engine not in {"baseline","stan","ez_regression","diffIRT","brms"}: raise EyeProcessValidationError("Unknown diffusion engine.")
    drift=list(map(str,drift_features)); boundary=list(map(str,boundary_features)); nondec=list(map(str,nondecision_features)); starting=list(map(str,starting_features))
    if gaze_features is not None and not drift: drift=list(map(str,gaze_features))
    allf=drift+boundary+nondec+starting
    if len(set(allf))!=len(allf): raise EyeProcessValidationError("A feature may map to only one diffusion parameter in a confirmatory specification.")
    if parallel_chains is None: parallel_chains=min(4,chains)
    return _result("eye_gaze_diffusion_spec",response=response,response_time=response_time,participant=participant,item=item,drift_features=drift,gaze_features=list(dict.fromkeys(allf)),boundary_features=boundary,nondecision_features=nondec,starting_features=starting,censor_column=censor_column,contaminant=bool(contaminant),engine=engine,normalized_engine=engine,legacy_mode=engine in {"ez_regression","diffIRT","brms"},chains=int(chains),parallel_chains=int(parallel_chains),iter_warmup=int(iter_warmup),iter_sampling=int(iter_sampling),adapt_delta=float(adapt_delta),max_treedepth=int(max_treedepth),interpretation="Gaze covariates are parameter predictors, not direct measures of attention or evidence quality.")


def _std_matrix(d: pd.DataFrame, cols: Sequence[str], prefix: str) -> np.ndarray:
    if not cols: return np.zeros((len(d),0))
    X=d[list(cols)].apply(pd.to_numeric,errors="coerce").to_numpy(float); center=X.mean(0); scale=X.std(0,ddof=1); scale[(~np.isfinite(scale))|(scale==0)]=1; z=(X-center)/scale
    return z


def prepare_gaze_diffusion_data(data: Any, spec: EyeResult, minimum_rt: float=.05) -> EyeResult:
    d=_df(data); features=spec.drift_features+spec.boundary_features+spec.nondecision_features+spec.starting_features; req=[spec.response,spec.response_time,spec.participant,spec.item,*features]+(([spec.censor_column]) if spec.censor_column else []); _required(d,req); d=d.dropna(subset=req).copy()
    rt=pd.to_numeric(d[spec.response_time],errors="coerce").to_numpy(float)
    if len(d)==0 or np.any(~np.isfinite(rt)) or np.any(rt<=minimum_rt): raise EyeProcessValidationError("Response times must be finite seconds greater than minimum_rt.")
    if np.median(rt)>30: raise EyeProcessValidationError("Response times appear to be milliseconds; convert them to seconds before diffusion modelling.")
    y=pd.to_numeric(d[spec.response],errors="coerce").to_numpy(int)
    if not np.isin(y,[0,1]).all(): raise EyeProcessValidationError("Diffusion responses must be coded 0/1.")
    if spec.censor_column:
        cmap={"observed":0,"right":1,"left":2}; vals=d[spec.censor_column].astype(str).str.lower(); censor=vals.map(cmap).to_numpy()
        if pd.isna(censor).any(): raise EyeProcessValidationError("Censoring values must be observed, right, or left.")
        censor=censor.astype(int)
    else: censor=np.zeros(len(d),int)
    pl=list(dict.fromkeys(d[spec.participant].astype(str))); il=list(dict.fromkeys(d[spec.item].astype(str))); pidx=np.array([pl.index(v)+1 for v in d[spec.participant].astype(str)]); iidx=np.array([il.index(v)+1 for v in d[spec.item].astype(str)])
    return _result("eye_gaze_diffusion_data",data=d,y=y,rt=rt,censor=censor,participant=pidx,item=iidx,participant_levels=pl,item_levels=il,X_drift=_std_matrix(d,spec.drift_features,"drift:"),X_boundary=_std_matrix(d,spec.boundary_features,"boundary:"),X_nondecision=_std_matrix(d,spec.nondecision_features,"nondecision:"),X_starting=_std_matrix(d,spec.starting_features,"starting:"),rt_lower=float(minimum_rt),rt_upper=float(rt.max()*1.25),minimum_observed_rt=float(rt.min()),spec=spec)


def _logistic_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    def obj(b):
        p=np.clip(expit(X@b),1e-12,1-1e-12); return float(-np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
    return minimize(obj,np.zeros(X.shape[1]),method="BFGS").x


def _baseline_diffusion(prep: EyeResult) -> EyeResult:
    d=prep.data; features=list(dict.fromkeys(prep.spec.drift_features+prep.spec.boundary_features+prep.spec.nondecision_features+prep.spec.starting_features)); base=pd.DataFrame({"Intercept":np.ones(len(d))})
    for c in features: base[c]=pd.to_numeric(d[c],errors="coerce").to_numpy(float)
    base=pd.concat([base,pd.get_dummies(d[prep.spec.participant].astype(str),prefix="participant",drop_first=True,dtype=float),pd.get_dummies(d[prep.spec.item].astype(str),prefix="item",drop_first=True,dtype=float)],axis=1); X=base.to_numpy(float); bacc=_logistic_fit(X,prep.y); blog=np.linalg.lstsq(X,np.log(prep.rt),rcond=None)[0]
    return _result("eye_gaze_diffusion_baseline",accuracy=_result("eye_logistic_model",terms=list(base.columns),coef=bacc),timing=_result("eye_linear_model",terms=list(base.columns),coef=blog),prepared=prep,item_parameters=pd.DataFrame({"item_id":prep.item_levels,"difficulty":np.nan,"boundary_proxy":np.nan,"nondecision_proxy":np.nan}))


def fit_gaze_diffusion_stan(prepared: EyeResult, seed:int=1, refresh:int=0, output_dir:str|None=None, **kwargs:Any)->EyeResult:
    cmdstanpy=_cmdstanpy(); spec=prepared.spec
    try:
        ver=cmdstanpy.cmdstan_version();
        if ver is None or tuple(ver)<(2,38): raise EyeProcessBackendError("The censored Wiener engine requires CmdStan 2.38.0 or newer.")
    except EyeProcessBackendError: raise
    except Exception as exc: raise EyeProcessBackendError("CmdStan is not available; install/configure eyeprocesspy[stan].") from exc
    data=dict(N=len(prepared.y),P=len(prepared.participant_levels),J=len(prepared.item_levels),Fd=prepared.X_drift.shape[1],Fb=prepared.X_boundary.shape[1],Fn=prepared.X_nondecision.shape[1],Fs=prepared.X_starting.shape[1],Xd=prepared.X_drift,Xb=prepared.X_boundary,Xn=prepared.X_nondecision,Xs=prepared.X_starting,y=prepared.y,rt=prepared.rt,censor=prepared.censor,person=prepared.participant,item=prepared.item,min_rt=prepared.minimum_observed_rt,rt_lower=prepared.rt_lower,rt_upper=prepared.rt_upper,use_contaminant=int(spec.contaminant))
    try: fit=cmdstanpy.CmdStanModel(stan_file=_stan_path("gaze_diffusion_irt.stan")).sample(data=data,seed=seed,chains=spec.chains,parallel_chains=spec.parallel_chains,iter_warmup=spec.iter_warmup,iter_sampling=spec.iter_sampling,adapt_delta=spec.adapt_delta,max_treedepth=spec.max_treedepth,refresh=refresh,output_dir=output_dir,**kwargs)
    except Exception as exc: raise EyeProcessBackendError(f"CmdStan gaze-diffusion fitting failed: {exc}") from exc
    return _result("eye_gaze_diffusion_stan",prepared=prepared,spec=spec,fit=fit,summary=fit.summary(),diagnostics=pd.DataFrame())


def fit_gaze_diffusion_irt(data:Any,spec:EyeResult|None=None,seed:int=1,**kwargs:Any)->EyeResult:
    spec=gaze_diffusion_spec() if spec is None else spec
    if spec.legacy_mode: raise EyeProcessBackendError(f"Legacy R diffusion engine '{spec.engine}' is not silently substituted in Python; use an explicit validated adapter.")
    prep=prepare_gaze_diffusion_data(data,spec); model=_baseline_diffusion(prep) if spec.engine=="baseline" else fit_gaze_diffusion_stan(prep,seed=seed,**kwargs); return _result("eye_gaze_diffusion_irt",spec=spec,prepared=prep,model=model,interpretation=spec.interpretation)


def extract_diffusion_parameters(object:EyeResult,variables:Sequence[str]=("beta_drift","beta_boundary","beta_nondecision","beta_starting","person_drift","item_difficulty","boundary","nondecision","starting","contaminant_probability"))->pd.DataFrame:
    if getattr(object.model,"eyeprocess_class",None)=="eye_gaze_diffusion_baseline":
        a=object.model.accuracy; t=object.model.timing; return pd.concat([pd.DataFrame({"component":"accuracy","term":a.terms,"estimate":a.coef}),pd.DataFrame({"component":"log_response_time","term":t.terms,"estimate":t.coef})],ignore_index=True)
    s=object.model.summary.copy(); keep=np.zeros(len(s),bool)
    namecol="variable" if "variable" in s else s.index.astype(str)
    vals=s["variable"].astype(str) if "variable" in s else pd.Series(s.index.astype(str),index=s.index)
    for prefix in variables:
        keep |= vals.str.startswith(prefix).to_numpy()
    return s.loc[keep].copy()


def diffusion_parameter_diagnostics(object:EyeResult,correlation_threshold:float=.85)->EyeResult:
    if getattr(object.model,"eyeprocess_class",None)!="eye_gaze_diffusion_stan": return _result("eye_diffusion_diagnostics",engine="baseline",sampling=None,correlations=pd.DataFrame(),flagged=pd.DataFrame(),message="Posterior identification diagnostics require the Stan engine.")
    return _result("eye_diffusion_diagnostics",engine="stan",sampling=object.model.diagnostics,correlations=pd.DataFrame(),flagged=pd.DataFrame(),threshold=correlation_threshold)


def diffusion_posterior_predictive(object:EyeResult,draws:int=200,method:str="rtdists",seed:int=1)->EyeResult:
    if getattr(object.model,"eyeprocess_class",None)=="eye_gaze_diffusion_baseline":
        p=expit(np.column_stack([np.ones(len(object.prepared.data))]).dot(np.array([0.]))) if False else object.prepared.y
        observed=pd.DataFrame([{"accuracy":float(np.mean(object.prepared.y)),"mean_rt":float(np.mean(object.prepared.rt)),"median_rt":float(np.median(object.prepared.rt))}]); return _result("eye_diffusion_ppc",observed=observed,replicated=pd.DataFrame(),method="baseline descriptive")
    return _result("eye_diffusion_ppc",observed=pd.DataFrame([{"accuracy":float(np.mean(object.prepared.y)),"mean_rt":float(np.mean(object.prepared.rt)),"median_rt":float(np.median(object.prepared.rt))}]),replicated=pd.DataFrame(),method=method)


def compare_diffusion_accuracy_rt(object:EyeResult)->pd.DataFrame:
    if getattr(object,"eyeprocess_class",None)!="eye_gaze_diffusion_irt": raise EyeProcessValidationError("Expected a gaze-diffusion fit.")
    return pd.DataFrame({"model":["gaze_diffusion","separate_accuracy_log_rt"],"log_likelihood_proxy":[np.nan,np.nan],"parameters":[np.nan,len(extract_diffusion_parameters(object))],"purpose":["joint cognitive-process likelihood","descriptive accuracy and timing baseline"]})


def _wiener_trial(rng:np.random.Generator,drift:float,boundary:float,nondecision:float,starting:float=.5,dt:float=.002,max_time:float=10)->tuple[int,float]:
    pos=boundary*starting; limit=max(1,int(math.ceil(max_time/dt)))
    for step in range(1,limit+1):
        pos+=drift*dt+math.sqrt(dt)*rng.normal()
        if pos>=boundary:return 1,nondecision+step*dt
        if pos<=0:return 0,nondecision+step*dt
    return int(pos>=boundary/2),nondecision+max_time


def simulate_gaze_diffusion_data(n_person:int=80,n_item:int=20,trials_per_item:int=1,gaze_effect:float=.35,contaminant_fraction:float=.02,time_step:float=.002,max_decision_time:float=10,seed:int=1)->pd.DataFrame:
    if n_person<2 or n_item<2 or trials_per_item<1: raise EyeProcessValidationError("Simulation requires at least two persons, two items, and one trial per item.")
    if not 0<=contaminant_fraction<1: raise EyeProcessValidationError("contaminant_fraction must be in [0, 1).")
    rng=np.random.default_rng(seed); theta=rng.normal(size=n_person); difficulty=rng.normal(size=n_item); person_boundary=np.exp(.15+rng.normal(0,.12,n_person)); item_nd=.18+expit(rng.normal(-2,.3,n_item))*.18; person_start=expit(rng.normal(0,.15,n_person)); rows=[]
    for p in range(n_person):
        for j in range(n_item):
            for rep in range(trials_per_item):
                gaze=float(rng.normal()); drift=float(theta[p]-difficulty[j]+gaze_effect*gaze); boundary=float(person_boundary[p]); nd=float(item_nd[j]); starting=float(person_start[p]); y,rt=_wiener_trial(rng,drift,boundary,nd,starting,time_step,max_decision_time); contaminant=rng.random()<contaminant_fraction
                if contaminant: rt=float(rng.uniform(max(.05,nd),max(rt,nd+.05))); y=int(rng.integers(2))
                rows.append({"participant_id":f"P{p+1}","item_id":f"I{j+1}","repetition":rep+1,"score":y,"response_time":rt,"gaze_balance":gaze,"true_drift":drift,"true_boundary":boundary,"true_nondecision":nd,"true_starting":starting,"contaminant":contaminant})
    out=pd.DataFrame(rows); out.attrs["truth"]={"theta":theta,"difficulty":difficulty,"gaze_effect":gaze_effect,"contaminant_fraction":contaminant_fraction}; return out


def diffusion_identification_study(conditions:Any=None,replications:int=20,base_seed:int=20260805,spec:EyeResult|None=None)->EyeResult:
    if conditions is None: conditions={"n_person":[50,150],"n_item":[10,30],"gaze_effect":[0,.35],"contaminant_fraction":[0,.05]}
    if isinstance(conditions,Mapping):
        import itertools
        keys=list(conditions); grid=pd.DataFrame([dict(zip(keys,v)) for v in itertools.product(*[conditions[k] for k in keys])])
    else:grid=_df(conditions)
    jobs=[]; j=0
    for _,r in grid.iterrows():
        for rep in range(1,int(replications)+1): j+=1;jobs.append({**r.to_dict(),"replication":rep,"seed":base_seed+j-1})
    plan=_result("eye_validation_job_plan",plan_id=f"gaze_diffusion-{base_seed}-{len(jobs)}",jobs=pd.DataFrame(jobs),model_family="gaze_diffusion")
    return _result("eye_diffusion_identification_study",plan=plan,spec=gaze_diffusion_spec(drift_features=["gaze_balance"],engine="stan") if spec is None else spec,metadata={"purpose":"parameter identification, recovery, censoring, contaminants, and trade-off diagnostics"})


__all__ = [
    "dynamic_irtree_spec","prepare_dynamic_irtree_data","structural_transition_mask","dynamic_transition_design","fit_multinomial_transition",
    "fit_dynamic_irtree_stan","decode_dynamic_states","dynamic_posterior_predictive_check","transition_residual_diagnostics","compare_dynamic_transition_models",
    "simulate_dynamic_irtree_data","dynamic_irtree_recovery","fit_dynamic_irtree",
    "theory_strategy_spec","prepare_strategy_mixture_data","fit_strategy_mixture_em","fit_strategy_mixture_stan","fit_theory_strategy_irt",
    "strategy_posterior_probabilities","strategy_classification_uncertainty","strategy_label_switching_diagnostics","strategy_aoi_sensitivity",
    "validate_strategy_manipulation","compare_strategy_heterogeneity","simulate_strategy_mixture_data","gaze_diffusion_spec","prepare_gaze_diffusion_data",
    "fit_gaze_diffusion_stan","fit_gaze_diffusion_irt","extract_diffusion_parameters","diffusion_parameter_diagnostics","diffusion_posterior_predictive",
    "compare_diffusion_accuracy_rt","simulate_gaze_diffusion_data","diffusion_identification_study",
]
