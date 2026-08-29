"""Visual-context, multiblock, profile, external-validity and item-seeding APIs.

Parity target: frozen R ``062-context-process-structure-0-8.R`` from
``eyeprocess`` 0.11.1.  Functions whose R implementation is inseparable from
``mirt`` remain explicit backend gates; dependency-light reference algorithms
are direct NumPy/SciPy translations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult, _result


def _df(x: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception as exc:  # pragma: no cover - defensive
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _req(d: pd.DataFrame, cols: Sequence[str], name: str = "data") -> None:
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required columns: {', '.join(missing)}")


def _num(s: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(s), errors="coerce")


def _sd(s: Any) -> float:
    a = _num(s).to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if a.size > 1 else np.nan


def _mean(s: Any) -> float:
    a = _num(s).to_numpy(float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 2:
        return np.nan
    yy = y[ok]
    pp = pred[ok]
    den = float(np.sum((yy - yy.mean()) ** 2))
    return 1.0 - float(np.sum((yy - pp) ** 2)) / den if den > 0 else np.nan


def _lm_fit(d: pd.DataFrame, response: str, predictors: Sequence[str]) -> EyeResult:
    cols = [response, *predictors]
    z = d[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if z.empty:
        raise EyeProcessValidationError("No complete cases available for linear model.")
    X = np.column_stack([np.ones(len(z)), z[list(predictors)].to_numpy(float)])
    y = z[response].to_numpy(float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    return _result(
        "eye_lm_reference",
        response=response,
        predictors=list(predictors),
        coefficients=pd.Series(coef, index=["(Intercept)", *predictors], dtype=float),
        fitted=pred,
        residuals=y - pred,
        r_squared=_r2(y, pred),
        training_index=z.index.to_numpy(),
    )


def _lm_predict(model: Mapping[str, Any], newdata: pd.DataFrame) -> np.ndarray:
    preds = list(model["predictors"])
    _req(newdata, preds, "newdata")
    Xn = newdata[preds].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    beta = np.asarray(model["coefficients"], dtype=float)
    return np.column_stack([np.ones(len(newdata)), Xn]) @ beta


def visual_context_registry(
    item_metadata: Any,
    item: str = "item_id",
    context: str | None = None,
    context_candidates: Sequence[str] = (
        "visual_anchor_id", "stimulus_id", "stimulus_page", "page_id",
        "layout_id", "screen_id", "diagram_id",
    ),
    min_items_per_context: int = 3,
) -> EyeResult:
    d = _df(item_metadata, "item_metadata")
    _req(d, [item], "item_metadata")
    if int(min_items_per_context) < 2:
        raise EyeProcessValidationError("min_items_per_context must be at least 2.")
    ids = d[item].astype(str)
    if ids.duplicated().any():
        raise EyeProcessValidationError(
            "item_metadata must contain one row per item for visual-context registration."
        )
    if context is None:
        found = [x for x in context_candidates if x in d.columns]
        if not found:
            raise EyeProcessValidationError("No visual-context column found; supply `context` explicitly.")
        context = found[0]
    _req(d, [context], "item_metadata")
    ctx = d[context].astype("string")
    missing = ctx.isna() | ctx.fillna("").eq("")
    ctx = ctx.astype(object)
    ctx.loc[missing] = "unique_context__" + ids.loc[missing]
    tab = pd.DataFrame({"item_id": ids, "visual_context_id": ctx.astype(str)})
    counts = tab["visual_context_id"].value_counts(dropna=False)
    tab["n_items"] = tab["visual_context_id"].map(counts).astype(int)
    tab["shared_context"] = tab["n_items"] >= int(min_items_per_context)
    return _result(
        "eye_visual_context_registry",
        mapping=tab[["item_id", "visual_context_id", "n_items", "shared_context"]].reset_index(drop=True),
        source_item_column=item,
        source_context_column=context,
        min_items_per_context=int(min_items_per_context),
        caveat=("Visual-context factors represent shared presentation context unless substantive "
                "theory justifies another interpretation."),
    )


def _context_positions(registry: Any, item_names: Sequence[str], selected_context: str | None = None) -> list[int]:
    if getattr(registry, "eyeprocess_class", None) != "eye_visual_context_registry":
        raise EyeProcessValidationError("registry must be created by visual_context_registry().")
    m = registry["mapping"].copy()
    pos = {str(name): i + 1 for i, name in enumerate(item_names)}  # R positions
    m["item_position"] = m["item_id"].map(pos)
    m = m[m["item_position"].notna()]
    shared = m.loc[m["shared_context"], "visual_context_id"].drop_duplicates().tolist()
    if not shared:
        raise EyeProcessValidationError("No shared visual context has enough items.")
    if selected_context is None:
        selected_context = str(shared[0])
    if selected_context not in shared:
        raise EyeProcessValidationError("selected context is not a valid shared context.")
    return sorted(m.loc[m["visual_context_id"] == selected_context, "item_position"].astype(int).unique().tolist())


def fit_visual_context_irt(
    response_matrix: Any,
    registry: Any,
    context: str | None = None,
    itemtype: str = "2PL",
    model_dimension: str = "Ability",
    context_dimension: str = "VisualContextFactor",
    SE: bool = False,
) -> EyeResult:
    """Gate the exact frozen ``mirt`` testlet model rather than substitute it.

    The registry/positions and model string are validated before raising, so the
    failure is an actionable backend boundary rather than a placeholder.
    """
    X = _df(response_matrix, "response_matrix")
    if X.shape[1] < 4:
        raise EyeProcessValidationError("At least four items are required.")
    item_names = [str(c) if str(c) else f"Item{i+1}" for i, c in enumerate(X.columns)]
    if all(str(c).isdigit() for c in X.columns):
        item_names = [f"Item{i+1}" for i in range(X.shape[1])]
    pos = _context_positions(registry, item_names, context)
    if len(pos) < 3 or len(pos) >= X.shape[1]:
        raise EyeProcessValidationError("Context factor must include at least three but not all items.")
    raise EyeProcessBackendError(
        "fit_visual_context_irt() requires the exact R `mirt` multidimensional/testlet engine; "
        "eyeprocesspy does not silently substitute a different estimator."
    )


def compare_visual_context_irt(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_visual_context_irt":
        raise EyeProcessValidationError("x must be eye_visual_context_irt.")
    cmp = x.get("comparison")
    if isinstance(cmp, pd.DataFrame):
        return cmp.copy()
    if cmp is None:
        return pd.DataFrame()
    return _df(cmp, "comparison")


def context_factor_effects(x: Any, IRTpars: bool = False) -> pd.DataFrame:
    del IRTpars
    if getattr(x, "eyeprocess_class", None) != "eye_visual_context_irt":
        raise EyeProcessValidationError("x must be eye_visual_context_irt.")
    if isinstance(x.get("context_factor_effects"), pd.DataFrame):
        return x["context_factor_effects"].copy()
    raise EyeProcessBackendError("Context-factor coefficient extraction requires the exact fitted mirt backend.")


def audit_visual_context_dependence(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_visual_context_irt":
        raise EyeProcessValidationError("x must be eye_visual_context_irt.")
    mapping = x["registry"]["mapping"]
    positions = list(x["positions"])
    total = mapping["item_id"].nunique()
    return pd.DataFrame([{
        "context": x["context"],
        "n_context_items": len(positions),
        "total_items": total,
        "context_fraction": len(positions) / total if total else np.nan,
        "comparison_available": x.get("comparison") is not None,
        "interpretation": ("Context factor models known shared presentation dependence; it is not "
                           "automatically a substantive trait."),
    }])


def process_feature_blocks(data: Any, blocks: Mapping[str, Sequence[str] | str], id: str | None = None,
                           drop_constant: bool = True) -> EyeResult:
    d = _df(data)
    if not isinstance(blocks, Mapping) or not blocks or any(not str(k) for k in blocks):
        raise EyeProcessValidationError("blocks must be a named list of column names.")
    norm = {str(k): ([v] if isinstance(v, str) else list(v)) for k, v in blocks.items()}
    flat = [v for vals in norm.values() for v in vals]
    if len(flat) != len(set(flat)):
        raise EyeProcessValidationError("Each feature must belong to only one block for this multiblock map.")
    _req(d, ([id] if id else []) + flat)
    clean: dict[str, list[str]] = {}
    for name, vals in norm.items():
        vv = [v for v in vals if v in d.columns]
        if drop_constant:
            vv = [v for v in vv if np.isfinite(_sd(d[v])) and _sd(d[v]) > 0]
        if vv:
            clean[name] = vv
    if len(clean) < 2:
        raise EyeProcessValidationError("At least two non-empty process-feature blocks are required.")
    return _result("eye_process_feature_blocks", data=d, blocks=clean, id=id,
                   block_sizes={k: len(v) for k, v in clean.items()},
                   status="conceptual_process_feature_blocks")


def fit_multiblock_process_map(x: Any, blocks: Mapping[str, Sequence[str]] | None = None,
                               id: str | None = None, engine: str = "auto", ncp: int = 5) -> EyeResult:
    if engine not in {"auto", "FactoMineR", "pca_block_scaled"}:
        raise EyeProcessValidationError("engine must be one of auto, FactoMineR, pca_block_scaled.")
    if getattr(x, "eyeprocess_class", None) != "eye_process_feature_blocks":
        x = process_feature_blocks(x, blocks or {}, id=id)
    d, b, id_col = x["data"], x["blocks"], x["id"]
    if int(ncp) < 1:
        raise EyeProcessValidationError("ncp must be at least 1.")
    if len(d) < 3:
        raise EyeProcessValidationError("At least three rows are required for multiblock mapping.")
    if engine == "FactoMineR":
        raise EyeProcessBackendError("The exact FactoMineR MFA backend is an R-specific optional engine.")
    # `auto` deterministically selects the transparent dependency-light fallback in Python.
    variables = [v for vals in b.values() for v in vals]
    df = d[variables].apply(pd.to_numeric, errors="coerce").copy()
    for v in variables:
        mu = _mean(df[v])
        if not np.isfinite(mu):
            mu = 0.0
        df[v] = df[v].fillna(mu)
    pieces = []
    for _, vv in b.items():
        arr = df[vv].to_numpy(float)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        z = (arr - mu) / sd
        z /= np.sqrt(z.shape[1])
        pieces.append(z)
    Z = np.column_stack(pieces)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    k = min(int(ncp), Z.shape[1], max(1, Z.shape[0] - 1))
    rotation = vt[:k].T
    scores = Z @ rotation
    dim_names = [f"PC{i+1}" for i in range(k)]
    person = pd.DataFrame(scores, columns=dim_names)
    person["id"] = d[id_col].astype(str).to_numpy() if id_col else [str(i + 1) for i in range(len(d))]
    var = pd.DataFrame(rotation, columns=dim_names)
    var["variable"] = variables
    rows = []
    for name, vv in b.items():
        idx = [variables.index(v) for v in vv]
        vals = np.mean(np.abs(rotation[idx, :]), axis=0)
        rows.append({"block": name, **dict(zip(dim_names, vals))})
    block_coord = pd.DataFrame(rows)
    model = _result("eye_pca_reference", singular_values=s[:k], rotation=rotation, scores=scores)
    return _result(
        "eye_multiblock_process_map", model=model, person_coordinates=person,
        variable_coordinates=var, block_coordinates=block_coord, blocks=b,
        engine="pca_block_scaled", status="exploratory_block_scaled_PCA_fallback_not_MFA",
        caveat=("Multiblock mapping is exploratory structure description and does not replace IRT "
                "calibration, DIF analysis, or external validation."),
    )


def multiblock_contributions(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_multiblock_process_map":
        raise EyeProcessValidationError("x must be eye_multiblock_process_map.")
    return x["block_coordinates"].copy()


def multiblock_person_coordinates(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_multiblock_process_map":
        raise EyeProcessValidationError("x must be eye_multiblock_process_map.")
    return x["person_coordinates"].copy()


def multiblock_variable_coordinates(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_multiblock_process_map":
        raise EyeProcessValidationError("x must be eye_multiblock_process_map.")
    return x["variable_coordinates"].copy()


def _soft_cluster_prob(z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = np.column_stack([np.sum((z - c) ** 2, axis=1) for c in centers])
    d2 -= np.min(d2, axis=1, keepdims=True)
    s = np.exp(-0.5 * d2)
    return s / s.sum(axis=1, keepdims=True)


def fit_process_profile_mixture(data: Any, variables: Sequence[str], k: int = 3,
                                id: str = "person_id", engine: str = "auto", seed: int = 777) -> EyeResult:
    if engine not in {"auto", "tidyLPA", "kmeans_reference"}:
        raise EyeProcessValidationError("Invalid profile engine.")
    d = _df(data)
    variables = list(variables)
    if not variables:
        raise EyeProcessValidationError("Supply at least one process-profile variable.")
    _req(d, variables)
    if int(k) < 2:
        raise EyeProcessValidationError("k must be at least 2.")
    X = d[variables].apply(pd.to_numeric, errors="coerce")
    usable = [v for v in variables if np.isfinite(_sd(X[v])) and _sd(X[v]) > 0]
    if len(usable) < 2:
        raise EyeProcessValidationError("At least two varying numeric process variables are required.")
    X = X[usable]
    ok = X.notna().all(axis=1)
    Xc = X.loc[ok]
    if len(Xc) < max(20, 4 * int(k)):
        raise EyeProcessValidationError("Too few complete cases for requested profile count.")
    arr = Xc.to_numpy(float)
    z = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=1)
    if engine == "tidyLPA":
        raise EyeProcessBackendError("The exact tidyLPA latent-profile backend is R-specific.")
    # scipy kmeans2 is a deterministic reference once its seed is fixed.
    rng = np.random.default_rng(int(seed))
    init_idx = rng.choice(len(z), size=int(k), replace=False)
    centers, labels = kmeans2(z, z[init_idx], minit="matrix", iter=100)
    probs = _soft_cluster_prob(z, centers)
    ids = d.loc[ok, id].astype(str).to_numpy() if id in d.columns else (np.flatnonzero(ok) + 1).astype(str)
    assignment = pd.DataFrame({"id": ids, "profile": [f"profile_{i+1}" for i in labels]})
    for j in range(probs.shape[1]):
        assignment[f"profile_probability_{j+1}"] = probs[:, j]
    complete = d.loc[ok, usable].copy()
    complete[".profile"] = assignment["profile"].to_numpy()
    summary = complete.groupby(".profile", sort=True)[usable].mean().reset_index().rename(columns={".profile": "profile"})
    model = _result("eye_kmeans_reference", centers=centers, labels=labels + 1)
    return _result(
        "eye_process_profile_mixture", model=model, assignment=assignment, summary=summary,
        variables=usable, k=int(k), engine="kmeans_reference", scaled_data=z,
        status="descriptive_kmeans_reference_not_finite_mixture",
        caveat=("Process profiles are exploratory descriptive groupings; they are not clinical, cheating, "
                "engagement, or cognitive-strategy labels without external validation."),
    )


def process_profile_probabilities(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_process_profile_mixture":
        raise EyeProcessValidationError("x must be eye_process_profile_mixture.")
    return x["assignment"].copy()


def process_profile_summary(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_process_profile_mixture":
        raise EyeProcessValidationError("x must be eye_process_profile_mixture.")
    return x["summary"].copy()


def compare_process_profile_solutions(data: Any, variables: Sequence[str], k_values: Sequence[int] = range(2, 7),
                                      seed: int = 777) -> pd.DataFrame:
    d = _df(data)
    _req(d, list(variables))
    X = d[list(variables)].apply(pd.to_numeric, errors="coerce")
    usable = [v for v in variables if np.isfinite(_sd(X[v])) and _sd(X[v]) > 0]
    if len(usable) < 2:
        raise EyeProcessValidationError("At least two varying numeric variables are required.")
    X = X[usable].dropna().to_numpy(float)
    z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    ks = sorted(set(int(k) for k in k_values if int(k) >= 2 and int(k) < len(z)))
    if not ks:
        raise EyeProcessValidationError("No valid k_values for the available complete cases.")
    total = float(np.sum((z - z.mean(axis=0)) ** 2))
    rows = []
    for off, k in enumerate(ks):
        rng = np.random.default_rng(int(seed) + off)
        init = z[rng.choice(len(z), k, replace=False)]
        centers, labels = kmeans2(z, init, minit="matrix", iter=100)
        within = float(sum(np.sum((z[labels == j] - centers[j]) ** 2) for j in range(k)))
        rows.append({"k": k, "total_withinss": within,
                     "between_over_total": (total - within) / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def audit_process_external_validity(data: Any, criterion: str, predictors: Sequence[str],
                                    baseline_predictors: Sequence[str] | None = None) -> EyeResult:
    d = _df(data)
    predictors = list(predictors)
    baseline = list(baseline_predictors or [])
    if not predictors:
        raise EyeProcessValidationError("Supply at least one process predictor.")
    req = list(dict.fromkeys([criterion, *predictors, *baseline]))
    _req(d, req)
    z = d[req].apply(pd.to_numeric, errors="coerce").dropna()
    if len(z) < 20:
        raise EyeProcessValidationError("At least 20 complete cases are required.")
    full = _lm_fit(z, criterion, list(dict.fromkeys([*baseline, *predictors])))
    base = _lm_fit(z, criterion, baseline)
    assoc = []
    for p in predictors:
        a, b = z[criterion].to_numpy(float), z[p].to_numpy(float)
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
        assoc.append({"predictor": p, "correlation": corr})
    comparison = pd.DataFrame([{
        "baseline_r2": base["r_squared"], "full_r2": full["r_squared"],
        "incremental_r2": full["r_squared"] - base["r_squared"],
    }])
    return _result(
        "eye_process_external_validity", full_model=full, baseline_model=base,
        comparison=comparison, associations=pd.DataFrame(assoc), criterion=criterion,
        predictors=predictors, baseline_predictors=baseline, data=z,
        incremental_r2=full["r_squared"] - base["r_squared"],
        status="external_structural_validation",
        caveat="Association with an external criterion supports validity evidence but does not establish causal mechanisms.",
    )


def process_criterion_associations(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_process_external_validity":
        raise EyeProcessValidationError("x must be eye_process_external_validity.")
    return x["associations"].copy()


def incremental_process_validity(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_process_external_validity":
        raise EyeProcessValidationError("x must be eye_process_external_validity.")
    return pd.DataFrame([{
        "baseline_r2": x["baseline_model"]["r_squared"],
        "full_r2": x["full_model"]["r_squared"],
        "incremental_r2": x["incremental_r2"],
    }])


def compare_process_criterion_models(x: Any) -> pd.DataFrame:
    if getattr(x, "eyeprocess_class", None) != "eye_process_external_validity":
        raise EyeProcessValidationError("x must be eye_process_external_validity.")
    return x["comparison"].copy() if isinstance(x.get("comparison"), pd.DataFrame) else pd.DataFrame()


def fit_item_parameter_seed_model(item_data: Any, difficulty: str = "irt_difficulty",
                                  discrimination: str = "irt_discrimination", predictors: Sequence[str] = (),
                                  engine: str = "auto", seed: int = 2221) -> EyeResult:
    del seed
    if engine not in {"auto", "ranger", "lm"}:
        raise EyeProcessValidationError("Invalid seed-model engine.")
    predictors = list(predictors)
    if not predictors:
        raise EyeProcessValidationError("Supply at least one item design/process predictor.")
    d = _df(item_data, "item_data")
    _req(d, [difficulty, discrimination, *predictors], "item_data")
    z = d[[difficulty, discrimination, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
    if any(not np.isfinite(_sd(z[p])) or _sd(z[p]) == 0 for p in predictors):
        raise EyeProcessValidationError("All item-seeding predictors must vary in the complete training data.")
    if len(z) < max(8, len(predictors) + 3):
        raise EyeProcessValidationError("Too few complete calibrated items for parameter seeding.")
    if engine == "ranger":
        raise EyeProcessBackendError("The exact R `ranger` seed-model backend is not a core Python dependency.")
    md = _lm_fit(z, difficulty, predictors)
    ma = _lm_fit(z, discrimination, predictors)
    return _result(
        "eye_item_parameter_seed", difficulty_model=md, discrimination_model=ma,
        difficulty=difficulty, discrimination=discrimination, predictors=predictors,
        engine="lm", training_data=z, status="experimental_pre_pilot_screening",
        caveat=("Predicted item parameters are screening priors/cold-start estimates only. Operational "
                "use requires expert review, pilot data, bias/accessibility review, and formal IRT calibration."),
    )


def predict_item_parameter_priors(object: Any, newdata: Any) -> pd.DataFrame:
    if getattr(object, "eyeprocess_class", None) != "eye_item_parameter_seed":
        raise EyeProcessValidationError("object must be eye_item_parameter_seed.")
    d = _df(newdata, "newdata")
    _req(d, object["predictors"], "newdata")
    pdiff = _lm_predict(object["difficulty_model"], d)
    pdisc = _lm_predict(object["discrimination_model"], d)
    out = d.copy()
    out["predicted_pre_pilot_difficulty"] = pdiff
    out["predicted_pre_pilot_discrimination"] = np.maximum(pdisc, 0.05)
    out["operational_status"] = "not_operational_requires_review_pilot_calibration"
    return out


def audit_candidate_item_bank(object: Any, candidate_data: Any, difficulty_range: Sequence[float] = (-3, 3),
                              discrimination_min: float = 0.3) -> EyeResult:
    dr = np.asarray(difficulty_range, dtype=float)
    if dr.size != 2 or not np.all(np.isfinite(dr)) or dr[0] >= dr[1]:
        raise EyeProcessValidationError("difficulty_range must contain two increasing finite values.")
    if not np.isfinite(discrimination_min) or discrimination_min <= 0:
        raise EyeProcessValidationError("discrimination_min must be positive.")
    p = predict_item_parameter_priors(object, candidate_data)
    p["difficulty_review_flag"] = (p["predicted_pre_pilot_difficulty"] < dr[0]) | (p["predicted_pre_pilot_difficulty"] > dr[1])
    p["discrimination_review_flag"] = p["predicted_pre_pilot_discrimination"] < float(discrimination_min)
    p["review_required"] = p["difficulty_review_flag"] | p["discrimination_review_flag"]
    return _result("eye_candidate_item_bank_audit", table=p, seed_model=object,
                   status="experimental_candidate_item_screening", caveat=object["caveat"])


__all__ = [
    "visual_context_registry", "fit_visual_context_irt", "compare_visual_context_irt",
    "context_factor_effects", "audit_visual_context_dependence", "process_feature_blocks",
    "fit_multiblock_process_map", "multiblock_contributions", "multiblock_person_coordinates",
    "multiblock_variable_coordinates", "fit_process_profile_mixture", "process_profile_probabilities",
    "process_profile_summary", "compare_process_profile_solutions", "audit_process_external_validity",
    "process_criterion_associations", "incremental_process_validity", "compare_process_criterion_models",
    "fit_item_parameter_seed_model", "predict_item_parameter_priors", "audit_candidate_item_bank",
]
