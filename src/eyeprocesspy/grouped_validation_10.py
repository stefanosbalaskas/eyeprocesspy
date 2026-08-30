"""Grouped validation and leakage audits from frozen ``R/021-validation-program.R``."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError

__all__ = [
    "crossed_grouped_cv",
    "crossed_grouped_folds",
    "grouped_cv",
    "grouped_folds",
    "quantify_process_leakage",
]


class EyeGroupedFolds(dict):
    """Dictionary-like grouped-fold object preserving the R contract."""

    eyeprocess_class = "eye_grouped_folds"


class EyeGroupedCV(dict):
    """Dictionary-like grouped-CV result preserving the R contract."""

    eyeprocess_class = "eye_grouped_cv"


class EyeCrossedGroupedFolds(dict):
    """Cross-classified fold object with analysis/assessment/buffer partitions."""

    eyeprocess_class = "eye_crossed_grouped_folds"


class EyeCrossedGroupedCV(dict):
    """Cross-classified grouped-CV result preserving the R contract."""

    eyeprocess_class = "eye_crossed_grouped_cv"


def _load_model_backend():
    """Load optional GLM/formula dependencies only when grouped CV is used."""
    try:
        import patsy
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise EyeProcessValidationError(
            "Grouped cross-validation requires the optional psychometrics "
            "dependencies. Install `eyeprocesspy[psychometrics]`."
        ) from exc
    return patsy, sm, smf


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _require_frame(data: Any) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        _stop("`data` must be a pandas DataFrame.")
    return data


def _names(value: Any, name: str, minimum: int = 1) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    else:
        try:
            values = tuple(str(item).strip() for item in value)
        except TypeError as exc:
            raise EyeProcessValidationError(
                f"`{name}` must contain non-empty column names."
            ) from exc
    if len(values) < minimum or any(not item or item.lower() == "nan" for item in values):
        if minimum > 1:
            _stop(
                f"`{name}` must contain at least {minimum} non-empty crossed "
                "grouping columns."
            )
        _stop(f"`{name}` must contain non-empty column names.")
    return values


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        _stop("Missing required columns: " + ", ".join(missing))


def _fold_count(v: Any) -> int:
    try:
        number = int(v)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError(
            "`v` must be an integer of at least two."
        ) from exc
    try:
        exact = float(v) == float(number)
    except (TypeError, ValueError):
        exact = False
    if number < 2 or not exact:
        _stop("`v` must be an integer of at least two.")
    return number


def _group_key(data: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Python equivalent of R ``interaction(..., drop=TRUE, lex.order=TRUE)``."""
    values = data.loc[:, list(columns)].astype("string")
    missing = values.isna().any(axis=1)
    key = values.fillna("<NA>").agg("\x1f".join, axis=1).astype("string")
    key.loc[missing] = pd.NA
    return key


def _balanced_assignment(levels: Sequence[str], v: int, seed: int) -> dict[str, int]:
    folds = np.resize(np.arange(1, v + 1, dtype=int), len(levels))
    rng = np.random.default_rng(int(seed))
    folds = rng.permutation(folds)
    return dict(zip(levels, folds, strict=True))


def grouped_folds(
    data: pd.DataFrame,
    group=("participant_id",),
    v=5,
    seed=1,
):
    """Create grouped folds so declared independent groups never cross folds."""
    data = _require_frame(data)
    groups = _names(group, "group")
    _require_columns(data, groups)
    v = _fold_count(v)

    key = _group_key(data, groups)
    levels = key.dropna().astype(str).drop_duplicates().tolist()
    if len(levels) < v:
        _stop("There are fewer independent groups than requested folds.")

    assignment = _balanced_assignment(levels, v, int(seed))
    fold_id = key.astype("string").map(assignment).astype("Int64")

    folds = []
    for fold in range(1, v + 1):
        assessment_mask = fold_id.eq(fold).fillna(False).to_numpy(dtype=bool)
        analysis_mask = fold_id.ne(fold).fillna(False).to_numpy(dtype=bool)
        folds.append(
            {
                "analysis": np.flatnonzero(analysis_mask),
                "assessment": np.flatnonzero(assessment_mask),
            }
        )

    return EyeGroupedFolds(
        folds=folds,
        fold_id=fold_id,
        group=groups,
        v=v,
    )


def _family(value: Any):
    _, sm, _ = _load_model_backend()
    if value is None:
        return sm.families.Binomial()
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"binomial", "logit", "logistic"}:
            return sm.families.Binomial()
        if normalized in {"gaussian", "normal"}:
            return sm.families.Gaussian()
        _stop(f"Unsupported statsmodels family `{value}`.")
    return value


def _metric(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"log_loss", "brier", "accuracy"}:
        _stop("`metric` must be one of: log_loss, brier, accuracy.")
    return normalized


def _response_values(formula: Any, data: pd.DataFrame) -> np.ndarray:
    patsy, _, _ = _load_model_backend()
    try:
        response, _ = patsy.dmatrices(
            formula,
            data=data,
            return_type="dataframe",
            NA_action="drop",
        )
    except Exception as exc:
        raise EyeProcessValidationError(
            f"Could not evaluate model formula: {exc}"
        ) from exc
    if response.shape[1] != 1:
        _stop("Grouped validation currently requires a scalar response.")
    values = pd.to_numeric(response.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    if len(values) != len(data):
        _stop("Formula evaluation removed rows; grouped validation requires complete model fields.")
    return values


def _score(y: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    finite = np.isfinite(y) & np.isfinite(prediction)
    if not finite.any():
        return np.nan
    y = y[finite]
    prediction = prediction[finite]

    if metric == "log_loss":
        clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
        values = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        return float(np.mean(values))
    if metric == "brier":
        return float(np.mean((y - prediction) ** 2))
    return float(np.mean((prediction >= 0.5) == y))


def _fit_and_score(
    data: pd.DataFrame,
    formula: Any,
    family: Any,
    analysis: np.ndarray,
    assessment: np.ndarray,
    metric: str,
) -> float:
    train = data.iloc[analysis].copy()
    test = data.iloc[assessment].copy()
    if train.empty or test.empty:
        raise ValueError("empty analysis or assessment set")
    _, _, smf = _load_model_backend()
    fitted = smf.glm(
        formula=formula,
        data=train,
        family=_family(family),
    ).fit()
    prediction = np.asarray(fitted.predict(test), dtype=float)
    y = _response_values(formula, test)
    return _score(y, prediction, metric)


def grouped_cv(
    data: pd.DataFrame,
    formula,
    family=None,
    group="participant_id",
    v=5,
    metric="log_loss",
    seed=1,
):
    """Evaluate a GLM with grouped cross-validation."""
    data = _require_frame(data)
    metric = _metric(metric)
    folds = grouped_folds(data, group=group, v=v, seed=seed)

    rows = []
    for index, fold in enumerate(folds["folds"], start=1):
        analysis = fold["analysis"]
        assessment = fold["assessment"]
        try:
            score = _fit_and_score(
                data,
                formula,
                family,
                analysis,
                assessment,
                metric,
            )
            error = pd.NA
        except Exception as exc:
            score = np.nan
            error = str(exc)
        rows.append(
            {
                "fold": index,
                "n_analysis": int(len(analysis)),
                "n_assessment": int(len(assessment)),
                "score": score,
                "error": error,
            }
        )

    return EyeGroupedCV(
        results=pd.DataFrame(rows),
        folds=folds,
        metric=metric,
        formula=formula,
    )


def crossed_grouped_folds(
    data: pd.DataFrame,
    groups=("participant_id", "item_id"),
    v=5,
    seed=1,
):
    """Create cross-classified folds with a deliberate mixed-level buffer."""
    data = _require_frame(data)
    groups = _names(groups, "groups", minimum=2)
    _require_columns(data, groups)
    v = _fold_count(v)

    assignments: dict[str, pd.Series] = {}
    row_assignments: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(seed))

    for group in groups:
        raw = data[group]
        as_text = raw.astype("string")
        if as_text.isna().any() or as_text.fillna("").str.len().eq(0).any():
            _stop("Crossed grouping columns cannot contain missing or empty values.")
        levels = as_text.astype(str).drop_duplicates().tolist()
        if len(levels) < v:
            _stop(
                f"Grouping column `{group}` has fewer levels than requested folds."
            )

        fold_values = np.resize(np.arange(1, v + 1, dtype=int), len(levels))
        fold_values = rng.permutation(fold_values)
        level_assignment = pd.Series(
            fold_values,
            index=pd.Index(levels, name=group),
            dtype="int64",
        )
        assignments[group] = level_assignment
        row_assignments[group] = (
            as_text.astype(str).map(level_assignment.to_dict()).to_numpy(dtype=int)
        )

    folds = []
    for fold in range(1, v + 1):
        held = np.column_stack(
            [row_assignments[group] == fold for group in groups]
        )
        assessment_mask = held.all(axis=1)
        analysis_mask = (~held).all(axis=1)
        buffer_mask = ~(assessment_mask | analysis_mask)
        folds.append(
            {
                "analysis": np.flatnonzero(analysis_mask),
                "assessment": np.flatnonzero(assessment_mask),
                "buffer": np.flatnonzero(buffer_mask),
            }
        )

    return EyeCrossedGroupedFolds(
        folds=folds,
        assignments=assignments,
        groups=groups,
        v=v,
    )


def crossed_grouped_cv(
    data: pd.DataFrame,
    formula,
    family=None,
    groups=("participant_id", "item_id"),
    v=5,
    metric="log_loss",
    seed=1,
):
    """Evaluate a GLM with cross-classified grouped cross-validation."""
    data = _require_frame(data)
    metric = _metric(metric)
    folds = crossed_grouped_folds(data, groups=groups, v=v, seed=seed)

    rows = []
    for index, fold in enumerate(folds["folds"], start=1):
        analysis = fold["analysis"]
        assessment = fold["assessment"]
        buffer = fold["buffer"]
        if not len(analysis) or not len(assessment):
            rows.append(
                {
                    "fold": index,
                    "n_analysis": int(len(analysis)),
                    "n_assessment": int(len(assessment)),
                    "n_buffer": int(len(buffer)),
                    "score": np.nan,
                    "error": "empty analysis or assessment set",
                }
            )
            continue

        try:
            score = _fit_and_score(
                data,
                formula,
                family,
                analysis,
                assessment,
                metric,
            )
            error = pd.NA
        except Exception as exc:
            score = np.nan
            error = str(exc)

        rows.append(
            {
                "fold": index,
                "n_analysis": int(len(analysis)),
                "n_assessment": int(len(assessment)),
                "n_buffer": int(len(buffer)),
                "score": score,
                "error": error,
            }
        )

    return EyeCrossedGroupedCV(
        results=pd.DataFrame(rows),
        folds=folds,
        metric=metric,
        formula=formula,
    )


def quantify_process_leakage(
    data: pd.DataFrame,
    formula,
    group=("participant_id", "item_id"),
    v=5,
    seed=1,
):
    """Compare row-wise and progressively stricter grouped validation schemes."""
    data = _require_frame(data)
    groups = _names(group, "group")
    _require_columns(data, groups)

    row_data = data.copy()
    row_data.insert(0, ".row_group", np.arange(1, len(row_data) + 1))

    jobs: dict[str, Any] = {
        "row_wise": lambda: grouped_cv(
            row_data,
            formula,
            group=".row_group",
            v=v,
            metric="log_loss",
            seed=seed,
        ),
        "combined_group": lambda: grouped_cv(
            data,
            formula,
            group=groups,
            v=v,
            metric="log_loss",
            seed=seed,
        ),
    }
    for name in groups:
        jobs[f"held_{name}"] = (
            lambda group_name=name: grouped_cv(
                data,
                formula,
                group=group_name,
                v=v,
                metric="log_loss",
                seed=seed,
            )
        )
    if len(groups) >= 2:
        jobs["cross_classified"] = lambda: crossed_grouped_cv(
            data,
            formula,
            groups=groups,
            v=v,
            metric="log_loss",
            seed=seed,
        )

    rows = []
    for scheme, job in jobs.items():
        try:
            result = job()
            scores = pd.to_numeric(
                result["results"]["score"],
                errors="coerce",
            ).to_numpy(dtype=float)
            finite = np.isfinite(scores)
            rows.append(
                {
                    "scheme": scheme,
                    "folds": int(len(scores)),
                    "successful_folds": int(finite.sum()),
                    "mean_log_loss": (
                        float(np.mean(scores[finite])) if finite.any() else np.nan
                    ),
                    "error": pd.NA,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scheme": scheme,
                    "folds": 0,
                    "successful_folds": 0,
                    "mean_log_loss": np.nan,
                    "error": str(exc),
                }
            )

    output = pd.DataFrame(rows)
    row_reference = pd.to_numeric(
        output.loc[output["scheme"].eq("row_wise"), "mean_log_loss"],
        errors="coerce",
    )
    reference = (
        float(row_reference.iloc[0])
        if len(row_reference) and np.isfinite(row_reference.iloc[0])
        else np.nan
    )
    output["optimistic_difference"] = (
        output["mean_log_loss"] - reference if np.isfinite(reference) else np.nan
    )
    return output
