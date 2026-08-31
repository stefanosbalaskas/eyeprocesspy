"""Compositional AOI attention for frozen ``R/032-compositional-aoi.R``.

This module source-ports the ten public R/032 exports:

- derive_aoi_composition
- transform_aoi_composition
- fit_aoi_compositional_model
- compare_aoi_compositions
- aoi_balance_coordinates
- plot_aoi_ternary
- plot_aoi_balance_biplot
- plot_aoi_variation_matrix
- plot_compositional_group_difference
- plot_aoi_composition_trajectory

The closure/zero-replacement and Helmert ILR mathematics follow the frozen
eyeprocess 0.11.1 source. Group comparison follows its pseudo-F permutation
algorithm. NumPy provides the seeded permutation stream, which is explicitly
not represented as byte-identical to R's ``set.seed()`` stream.

The frozen R fixed-effects model uses ``stats::lm``. eyeprocesspy implements
ordinary least squares directly with NumPy/SciPy so the base package remains
limited to NumPy, pandas, and SciPy; common R-style additive and interaction
formulae are supported without promoting statsmodels/patsy to core
dependencies.

Matplotlib is imported lazily.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult

__all__ = [
    "aoi_balance_coordinates",
    "compare_aoi_compositions",
    "derive_aoi_composition",
    "fit_aoi_compositional_model",
    "plot_aoi_balance_biplot",
    "plot_aoi_composition_trajectory",
    "plot_aoi_ternary",
    "plot_aoi_variation_matrix",
    "plot_compositional_group_difference",
    "transform_aoi_composition",
]


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _require_frame(value: Any, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise EyeProcessValidationError(f"`{name}` must be a data frame.")
    if value.empty:
        raise EyeProcessValidationError(f"`{name}` must contain at least one row.")
    return value


def _require_class(value: Any, cls: str, *, name: str = "x") -> None:
    if getattr(value, "eyeprocess_class", None) != cls:
        raise EyeProcessValidationError(f"`{name}` must be an `{cls}` object.")


def _as_numeric_frame(value: Any, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        array = np.asarray(value)
        if array.ndim != 2:
            raise EyeProcessValidationError("A composition must be a two-dimensional matrix.")
        names = list(columns) if columns is not None else [f"part_{index + 1}" for index in range(array.shape[1])]
        frame = pd.DataFrame(array, columns=names)

    if frame.shape[1] < 2:
        raise EyeProcessValidationError("A composition requires at least two AOI parts.")

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.astype(float)


def _safe_mean(values: Any) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else np.nan


def _close_composition(matrix: Any, zero_method: str = "multiplicative") -> pd.DataFrame:
    if zero_method not in {"multiplicative", "bayesian"}:
        raise EyeProcessValidationError("`zero_method` must be 'multiplicative' or 'bayesian'.")

    frame = _as_numeric_frame(matrix)
    values = frame.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values) | (values < 0)] = np.nan

    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        replacement = float(finite.mean()) if finite.size else 0.0
        values[~np.isfinite(values[:, column]), column] = replacement

    positive = values[np.isfinite(values) & (values > 0)]
    global_delta = float(positive.min() * 0.5) if positive.size else 1e-6

    for row in range(values.shape[0]):
        zeros = ~np.isfinite(values[row]) | (values[row] <= 0)
        if zeros.any():
            if zero_method == "bayesian":
                nonzero = values[row, ~zeros]
                delta = (float(np.nansum(nonzero)) + 1.0) / (1000.0 * values.shape[1])
            else:
                delta = global_delta
            values[row, zeros] = delta

        total = float(np.sum(values[row]))
        if not np.isfinite(total) or total <= 0:
            values[row] = 1.0 / values.shape[1]
        else:
            values[row] = values[row] / total

    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _ilr_basis(parts: int) -> np.ndarray:
    parts = int(parts)
    if parts < 2:
        raise EyeProcessValidationError("ILR transformation requires at least two parts.")

    basis = np.zeros((parts, parts - 1), dtype=float)
    for column in range(parts - 1):
        basis[: column + 1, column] = -1.0
        basis[column + 1, column] = float(column + 1)
        norm = float(np.sqrt(np.sum(basis[:, column] ** 2)))
        basis[:, column] /= norm
    return basis


def derive_aoi_composition(
    x,
    aois,
    denominator=("total_aoi_dwell", "trial_duration"),
    zero_method=("multiplicative", "bayesian"),
    id_cols=None,
    aoi_col="aoi",
    value_col="dwell_ms",
    trial_duration_col=None,
):
    """Derive a closed AOI dwell composition from wide or long data."""
    x = _require_frame(x, name="x")

    if isinstance(denominator, (tuple, list)):
        denominator = denominator[0] if denominator else "total_aoi_dwell"
    if denominator not in {"total_aoi_dwell", "trial_duration"}:
        raise EyeProcessValidationError("`denominator` must be 'total_aoi_dwell' or 'trial_duration'.")

    if isinstance(zero_method, (tuple, list)):
        zero_method = zero_method[0] if zero_method else "multiplicative"
    if zero_method not in {"multiplicative", "bayesian"}:
        raise EyeProcessValidationError("`zero_method` must be 'multiplicative' or 'bayesian'.")

    parts = [str(part) for part in aois]
    if len(parts) < 2:
        raise EyeProcessValidationError("`aois` must identify at least two AOI parts.")

    requested_ids = [] if id_cols is None else list(id_cols)
    valid_ids = [column for column in requested_ids if column in x.columns]

    if all(part in x.columns for part in parts):
        raw = pd.DataFrame(
            {part: pd.to_numeric(x[part], errors="coerce") for part in parts},
            index=x.index,
        )
        ids = x.loc[:, valid_ids].reset_index(drop=True)

        if trial_duration_col is not None and trial_duration_col in x.columns:
            duration = pd.to_numeric(
                x[trial_duration_col],
                errors="coerce",
            ).to_numpy(dtype=float)
        else:
            duration = raw.sum(axis=1, skipna=True).to_numpy(dtype=float)
    else:
        missing = [column for column in (aoi_col, value_col) if column not in x.columns]
        if missing:
            raise EyeProcessValidationError(
                "`x` is missing required long-format column(s): " + ", ".join(missing) + "."
            )

        data = x.copy()
        if not valid_ids:
            data[".composition_row_id"] = np.arange(1, len(data) + 1)
            valid_ids = [".composition_row_id"]

        group_key = valid_ids[0] if len(valid_ids) == 1 else valid_ids
        grouped = list(
            data.groupby(
                group_key,
                sort=True,
                dropna=False,
                observed=True,
            )
        )

        raw_rows: list[list[float]] = []
        id_rows: list[dict[str, Any]] = []
        duration_values: list[float] = []

        for _, group in grouped:
            raw_rows.append(
                [
                    float(
                        pd.to_numeric(
                            group.loc[
                                group[aoi_col].astype(str).eq(part),
                                value_col,
                            ],
                            errors="coerce",
                        ).sum(skipna=True)
                    )
                    for part in parts
                ]
            )
            id_rows.append({column: group.iloc[0][column] for column in valid_ids})

            if trial_duration_col is not None and trial_duration_col in group.columns:
                duration_values.append(_safe_mean(group[trial_duration_col]))
            else:
                duration_values.append(float(np.nansum(raw_rows[-1])))

        raw = pd.DataFrame(raw_rows, columns=parts)
        ids = pd.DataFrame(id_rows)
        duration = np.asarray(duration_values, dtype=float)

    proportions = _close_composition(raw, zero_method=zero_method)

    if denominator == "trial_duration":
        raw_values = raw.to_numpy(dtype=float)
        fallback = np.nansum(raw_values, axis=1)
        invalid = ~np.isfinite(duration) | (duration <= 0)
        duration = duration.copy()
        duration[invalid] = fallback[invalid]
        scaled = raw_values / np.maximum(duration[:, None], 1e-8)
        proportions = _close_composition(
            pd.DataFrame(scaled, columns=parts),
            zero_method=zero_method,
        )

    table = pd.concat(
        [
            ids.reset_index(drop=True),
            proportions.reset_index(drop=True),
        ],
        axis=1,
    )

    return _result(
        "eye_aoi_composition",
        raw=raw.reset_index(drop=True),
        proportions=proportions.reset_index(drop=True),
        table=table,
        parts=parts,
        id_cols=list(ids.columns),
        denominator=denominator,
        zero_method=zero_method,
        status="AOI dwell composition derived and closed to one.",
    )


def transform_aoi_composition(
    x,
    method=("ilr", "clr", "alr"),
    reference=None,
):
    """Transform a composition using ILR, CLR, or ALR coordinates."""
    if isinstance(method, (tuple, list)):
        method = method[0] if method else "ilr"
    if method not in {"ilr", "clr", "alr"}:
        raise EyeProcessValidationError("`method` must be 'ilr', 'clr', or 'alr'.")

    if getattr(x, "eyeprocess_class", None) == "eye_aoi_composition":
        composition = pd.DataFrame(x["proportions"]).copy()
    else:
        composition = _close_composition(x)

    parts = list(map(str, composition.columns))
    log_composition = np.log(np.maximum(composition.to_numpy(dtype=float), 1e-15))

    if method == "clr":
        transformed_values = log_composition - log_composition.mean(axis=1, keepdims=True)
        columns = [f"clr_{part}" for part in parts]
        basis = None
    elif method == "alr":
        reference = str(reference) if reference is not None else parts[-1]
        if reference not in parts:
            raise EyeProcessValidationError("`reference` is not an AOI part.")
        reference_index = parts.index(reference)
        keep = [part for part in parts if part != reference]
        keep_index = [parts.index(part) for part in keep]
        transformed_values = log_composition[:, keep_index] - log_composition[:, [reference_index]]
        columns = [f"alr_{part}_vs_{reference}" for part in keep]
        basis = reference
    else:
        basis = _ilr_basis(len(parts))
        transformed_values = log_composition @ basis
        columns = [f"ilr_{index + 1}" for index in range(transformed_values.shape[1])]

    transformed = pd.DataFrame(
        transformed_values,
        index=composition.index,
        columns=columns,
    )

    return _result(
        "eye_aoi_logratio",
        source=x,
        transformed=transformed,
        method=method,
        basis=basis,
        parts=parts,
        status=f"{method.upper()} AOI log-ratio transform completed.",
    )


def _expand_formula_terms(rhs: str) -> tuple[bool, list[str]]:
    raw_terms = [term.strip() for term in rhs.split("+") if term.strip()]
    intercept = True
    expanded: list[str] = []

    for term in raw_terms:
        if term in {"0", "-1"}:
            intercept = False
            continue
        if term == "1":
            continue

        if "*" in term:
            factors = [part.strip() for part in term.split("*") if part.strip()]
            if len(factors) != 2:
                raise EyeProcessValidationError("The dependency-free formula engine supports two-way `*` interactions.")
            expanded.extend([factors[0], factors[1], f"{factors[0]}:{factors[1]}"])
        else:
            expanded.append(term)

    return intercept, list(dict.fromkeys(expanded))


def _design_for_term(data: pd.DataFrame, term: str) -> pd.DataFrame:
    categorical_match = re.fullmatch(r"C\(([^)]+)\)", term)
    if categorical_match:
        column = categorical_match.group(1).strip()
        if column not in data.columns:
            raise EyeProcessValidationError(f"Formula column `{column}` is unavailable.")
        return pd.get_dummies(
            data[column].astype("category"),
            prefix=column,
            drop_first=True,
            dtype=float,
        )

    if ":" in term:
        left_name, right_name = [part.strip() for part in term.split(":", 1)]
        left = _design_for_term(data, left_name)
        right = _design_for_term(data, right_name)
        columns: dict[str, pd.Series] = {}
        for left_column in left.columns:
            for right_column in right.columns:
                name = f"{left_column}:{right_column}"
                columns[name] = pd.to_numeric(left[left_column], errors="coerce") * pd.to_numeric(
                    right[right_column], errors="coerce"
                )
        return pd.DataFrame(columns, index=data.index)

    if term not in data.columns:
        raise EyeProcessValidationError(f"Formula column `{term}` is unavailable.")

    series = data[term]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return pd.DataFrame({term: numeric.astype(float)}, index=data.index)

    return pd.get_dummies(
        series.astype("category"),
        prefix=term,
        drop_first=True,
        dtype=float,
    )


@dataclass
class _OLSFit:
    params: pd.Series
    fittedvalues: pd.Series
    resid: pd.Series
    response: pd.Series
    design: pd.DataFrame
    df_resid: int
    sigma2: float
    covariance: pd.DataFrame
    summary_table: pd.DataFrame

    def coef(self) -> pd.Series:
        return self.params.copy()


def _fit_formula_ols(formula: Any, data: pd.DataFrame) -> _OLSFit:
    if not isinstance(formula, str) or "~" not in formula:
        raise EyeProcessValidationError("Python formula arguments must be R-style strings such as `outcome ~ ilr_1`.")

    response_name, rhs = [part.strip() for part in formula.split("~", 1)]
    if not response_name or response_name not in data.columns:
        raise EyeProcessValidationError(f"Formula response `{response_name}` is not available in model data.")

    intercept, terms = _expand_formula_terms(rhs)
    blocks = [_design_for_term(data, term) for term in terms]

    design = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=data.index)
    if intercept:
        design.insert(0, "(Intercept)", 1.0)

    if design.shape[1] == 0:
        raise EyeProcessValidationError("The model formula produced no design columns.")

    response = pd.to_numeric(data[response_name], errors="coerce")
    design = design.apply(pd.to_numeric, errors="coerce").astype(float)

    finite = np.isfinite(response.to_numpy(dtype=float))
    finite &= np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
    if finite.sum() <= design.shape[1]:
        raise EyeProcessValidationError("Insufficient complete observations for the requested compositional model.")

    y = response.loc[finite].to_numpy(dtype=float)
    X = design.loc[finite].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefficients
    residuals = y - fitted
    df_resid = max(0, len(y) - int(rank))
    rss = float(residuals @ residuals)
    sigma2 = rss / df_resid if df_resid > 0 else np.nan
    xtx_inverse = np.linalg.pinv(X.T @ X)
    covariance_values = xtx_inverse * sigma2

    standard_error = np.sqrt(np.maximum(np.diag(covariance_values), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_value = coefficients / standard_error
    if df_resid > 0:
        p_value = 2.0 * student_t.sf(np.abs(t_value), df=df_resid)
    else:
        p_value = np.full_like(t_value, np.nan, dtype=float)

    names = list(design.columns)
    params = pd.Series(coefficients, index=names, name="Estimate")
    fitted_series = pd.Series(
        fitted,
        index=data.index[finite],
        name="fitted",
    )
    residual_series = pd.Series(
        residuals,
        index=data.index[finite],
        name="residual",
    )
    response_series = pd.Series(
        y,
        index=data.index[finite],
        name=response_name,
    )
    covariance = pd.DataFrame(
        covariance_values,
        index=names,
        columns=names,
    )
    summary_table = pd.DataFrame(
        {
            "Estimate": coefficients,
            "Std. Error": standard_error,
            "t value": t_value,
            "Pr(>|t|)": p_value,
        },
        index=names,
    )

    return _OLSFit(
        params=params,
        fittedvalues=fitted_series,
        resid=residual_series,
        response=response_series,
        design=design.loc[finite].copy(),
        df_resid=df_resid,
        sigma2=sigma2,
        covariance=covariance,
        summary_table=summary_table,
    )


def fit_aoi_compositional_model(
    composition,
    formula,
    random=None,
    data=None,
    method="ilr",
):
    """Fit the frozen fixed-effects compositional AOI regression contract."""
    if getattr(composition, "eyeprocess_class", None) == "eye_aoi_logratio":
        transformed = composition
    else:
        transformed = transform_aoi_composition(composition, method=method)

    model_data = pd.DataFrame(transformed["transformed"]).reset_index(drop=True)

    if data is not None:
        data = _require_frame(data, name="data").reset_index(drop=True)
        if len(data) != len(model_data):
            raise EyeProcessValidationError("`data` must have one row per composition.")
        model_data = pd.concat([data, model_data], axis=1)

    model = _fit_formula_ols(formula, model_data)

    if random is not None:
        warnings.warn(
            "`random` is recorded for audit but the dependency-free engine "
            "fits a fixed-effects model. Use an external mixed-model adapter "
            "for confirmatory random effects.",
            RuntimeWarning,
            stacklevel=2,
        )

    return _result(
        "eye_aoi_composition_model",
        model=model,
        composition=transformed,
        formula=formula,
        random=random,
        data=model_data,
        summary=model.summary_table.copy(),
        status="Compositional AOI model fitted on log-ratio coordinates.",
    )


def _composition_pseudo_f(z: Any, group: Any) -> float:
    matrix = np.asarray(z, dtype=float)
    group_series = pd.Series(group, dtype="object")
    levels = list(pd.unique(group_series))

    grand = matrix.mean(axis=0)
    within = 0.0
    between = 0.0

    for level in levels:
        rows = group_series.eq(level).to_numpy()
        center = matrix[rows].mean(axis=0)
        between += float(rows.sum()) * float(np.sum((center - grand) ** 2))
        within += float(np.sum((matrix[rows] - center) ** 2))

    df_between = max(1, len(levels) - 1)
    df_within = max(1, len(matrix) - len(levels))
    return float((between / df_between) / max(within / df_within, 1e-12))


def compare_aoi_compositions(
    x,
    group,
    method=("permanova", "compositional_manova"),
    permutations=499,
    seed=20260807,
):
    """Compare AOI compositions with the frozen pseudo-F permutation scheme."""
    _require_class(x, "eye_aoi_composition")

    if isinstance(method, (tuple, list)):
        method = method[0] if method else "permanova"
    if method not in {"permanova", "compositional_manova"}:
        raise EyeProcessValidationError("`method` must be 'permanova' or 'compositional_manova'.")

    table = x["table"]
    if isinstance(group, str) and group in table.columns:
        group_values = table[group].reset_index(drop=True)
    else:
        if isinstance(group, str):
            group_values = pd.Series([group])
        else:
            group_values = pd.Series(list(group))

    if len(group_values) != len(x["proportions"]):
        raise EyeProcessValidationError("`group` must contain one value per composition.")
    if group_values.nunique(dropna=False) < 2:
        raise EyeProcessValidationError("`group` must contain at least two groups.")

    try:
        permutations = int(permutations)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError("`permutations` must be a positive integer.") from exc
    if permutations < 1:
        raise EyeProcessValidationError("`permutations` must be a positive integer.")

    z = transform_aoi_composition(x, "ilr")["transformed"].to_numpy(dtype=float)
    observed = _composition_pseudo_f(z, group_values)

    rng = np.random.default_rng(seed)
    original = group_values.to_numpy(dtype=object)
    null = np.asarray(
        [_composition_pseudo_f(z, rng.permutation(original)) for _ in range(permutations)],
        dtype=float,
    )
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))

    proportions = pd.DataFrame(x["proportions"]).copy()
    proportions["__group"] = original
    centroids = (
        proportions.groupby(
            "__group",
            as_index=False,
            sort=True,
            dropna=False,
        )
        .mean(numeric_only=True)
        .rename(columns={"__group": "group"})
    )

    summary = pd.DataFrame(
        {
            "statistic": [observed],
            "p_value": [p_value],
            "permutations": [len(null)],
        }
    )

    return _result(
        "eye_aoi_composition_comparison",
        observed=observed,
        null=null,
        p_value=p_value,
        centroids=centroids,
        method=method,
        summary=summary,
        status="Group comparison of AOI compositions completed.",
    )


def aoi_balance_coordinates(x, balances):
    """Calculate user-defined AOI balance coordinates."""
    if getattr(x, "eyeprocess_class", None) == "eye_aoi_composition":
        composition = pd.DataFrame(x["proportions"]).copy()
    else:
        composition = _close_composition(x)

    log_composition = np.log(composition.to_numpy(dtype=float))
    parts = list(map(str, composition.columns))

    if isinstance(balances, pd.DataFrame):
        matrix = balances.to_numpy(dtype=float)
        names = list(map(str, balances.columns))
    elif isinstance(balances, np.ndarray):
        matrix = np.asarray(balances, dtype=float)
        names = [f"balance_{index + 1}" for index in range(matrix.shape[1])] if matrix.ndim == 2 else []
    else:
        matrix = None
        names = []

    if matrix is not None:
        if matrix.ndim != 2 or matrix.shape[0] != len(parts):
            raise EyeProcessValidationError("A balance matrix must have one row per AOI part.")
        values = log_composition @ matrix
        if not names:
            names = [f"balance_{index + 1}" for index in range(values.shape[1])]
        return pd.DataFrame(values, columns=names)

    if isinstance(balances, Mapping):
        balance_items = list(balances.items())
    elif isinstance(balances, Sequence) and not isinstance(
        balances,
        (str, bytes),
    ):
        balance_items = [(f"balance_{index + 1}", value) for index, value in enumerate(balances)]
    else:
        raise EyeProcessValidationError("`balances` must be a named mapping, sequence, or matrix.")

    if not balance_items:
        raise EyeProcessValidationError("`balances` must contain at least one balance.")

    output: dict[str, np.ndarray] = {}
    for name, balance in balance_items:
        if isinstance(balance, Mapping):
            numerator = balance.get("numerator")
            denominator = balance.get("denominator")
        elif (
            isinstance(balance, Sequence)
            and not isinstance(
                balance,
                (str, bytes),
            )
            and len(balance) >= 2
        ):
            numerator, denominator = balance[0], balance[1]
        else:
            raise EyeProcessValidationError("Each balance requires numerator and denominator AOI parts.")

        numerator = [
            str(part)
            for part in ([numerator] if isinstance(numerator, str) else list(numerator or []))
            if str(part) in parts
        ]
        denominator = [
            str(part)
            for part in ([denominator] if isinstance(denominator, str) else list(denominator or []))
            if str(part) in parts
        ]

        if not numerator or not denominator:
            raise EyeProcessValidationError("Each balance requires valid numerator and denominator parts.")

        r = len(numerator)
        s = len(denominator)
        coefficient = float(np.sqrt((r * s) / (r + s)))
        numerator_index = [parts.index(part) for part in numerator]
        denominator_index = [parts.index(part) for part in denominator]

        output[str(name)] = coefficient * (
            log_composition[:, numerator_index].mean(axis=1) - log_composition[:, denominator_index].mean(axis=1)
        )

    return pd.DataFrame(output)


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EyeProcessBackendError(
            "Compositional AOI plotting requires matplotlib. Install `eyeprocesspy[plots]` or the development extras."
        ) from exc
    return plt


def _axis(ax=None):
    if ax is not None:
        return ax
    return _get_plt().subplots()[1]


def _require_composition(x):
    _require_class(x, "eye_aoi_composition")
    return pd.DataFrame(x["proportions"]).copy()


def plot_aoi_ternary(x, ax=None, **kwargs):
    """Plot a three-part AOI dwell composition on a ternary simplex."""
    del kwargs
    composition = _require_composition(x)
    if composition.shape[1] != 3:
        axis = _axis(ax)
        axis.text(
            0.5,
            0.5,
            "Ternary plots require exactly three AOI parts.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.eyeprocess_plot_data = composition
        return axis

    a = composition.iloc[:, 0].to_numpy(dtype=float)
    b = composition.iloc[:, 1].to_numpy(dtype=float)
    c = composition.iloc[:, 2].to_numpy(dtype=float)
    xx = b + 0.5 * c
    yy = np.sqrt(3.0) / 2.0 * c

    axis = _axis(ax)
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0) / 2.0, 0.0]
    axis.plot(triangle_x, triangle_y)
    axis.scatter(xx, yy)
    labels = list(map(str, composition.columns))
    vertices = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, np.sqrt(3.0) / 2.0),
    ]
    for label, (x_value, y_value) in zip(labels, vertices):
        axis.text(x_value, y_value, label)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("AOI dwell composition")
    axis.eyeprocess_plot_data = composition
    return axis


def plot_aoi_balance_biplot(x, ax=None, **kwargs):
    """Plot a CLR-PCA compositional balance biplot."""
    del kwargs
    composition = _require_composition(x)
    clr = transform_aoi_composition(x, "clr")["transformed"]
    matrix = clr.to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    dimensions = min(2, vt.shape[0])
    scores = centered @ vt[:dimensions].T
    if dimensions == 1:
        scores = np.column_stack([scores[:, 0], np.zeros(len(scores))])

    axis = _axis(ax)
    axis.scatter(scores[:, 0], scores[:, 1])

    if singular_values.size:
        scale = float(np.nanmax(np.sqrt(np.sum(scores[:, :2] ** 2, axis=1))))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
    else:
        scale = 1.0

    loadings = vt[:dimensions].T
    if dimensions == 1:
        loadings = np.column_stack([loadings[:, 0], np.zeros(loadings.shape[0])])

    for part, vector in zip(composition.columns, loadings):
        axis.arrow(
            0.0,
            0.0,
            vector[0] * scale,
            vector[1] * scale,
            length_includes_head=True,
            head_width=0.03 * scale,
        )
        axis.text(
            vector[0] * scale,
            vector[1] * scale,
            str(part),
        )

    axis.set(
        xlabel="CLR PC1",
        ylabel="CLR PC2",
        title="AOI compositional balance biplot",
    )
    axis.eyeprocess_plot_data = pd.DataFrame(
        scores[:, :2],
        columns=["PC1", "PC2"],
    )
    return axis


def plot_aoi_variation_matrix(x, ax=None, **kwargs):
    """Plot pairwise variance of AOI log-ratios."""
    del kwargs
    composition = _require_composition(x)
    log_comp = np.log(composition.to_numpy(dtype=float))
    parts = list(map(str, composition.columns))
    variation = np.zeros((len(parts), len(parts)), dtype=float)

    for i in range(len(parts)):
        for j in range(len(parts)):
            difference = log_comp[:, i] - log_comp[:, j]
            variation[i, j] = float(np.var(difference, ddof=1)) if len(difference) > 1 else np.nan

    matrix = pd.DataFrame(variation, index=parts, columns=parts)
    axis = _axis(ax)
    axis.imshow(variation, origin="lower", aspect="auto")
    ticks = np.arange(len(parts))
    axis.set_xticks(ticks, labels=parts, rotation=90)
    axis.set_yticks(ticks, labels=parts)
    axis.set(
        xlabel="AOI part",
        ylabel="AOI part",
        title="AOI log-ratio variation matrix",
    )
    axis.eyeprocess_plot_data = matrix
    axis.eyeprocess_plot_matrix = variation
    return axis


def plot_compositional_group_difference(x, ax=None, **kwargs):
    """Plot group centroid differences from a composition comparison."""
    del kwargs
    _require_class(x, "eye_aoi_composition_comparison")
    centroids = pd.DataFrame(x["centroids"]).copy()
    if "group" not in centroids.columns:
        raise EyeProcessValidationError("Composition comparison centroids do not contain `group`.")

    value_columns = [column for column in centroids.columns if column != "group"]
    values = centroids[value_columns].to_numpy(dtype=float)
    n_groups = len(centroids)
    n_parts = len(value_columns)

    axis = _axis(ax)
    x_position = np.arange(n_groups, dtype=float)
    width = 0.8 / max(1, n_parts)

    for index, part in enumerate(value_columns):
        offset = (index - (n_parts - 1) / 2.0) * width
        axis.bar(
            x_position + offset,
            values[:, index],
            width=width,
            label=str(part),
        )

    axis.set_xticks(
        x_position,
        labels=centroids["group"].astype(str).tolist(),
        rotation=90,
    )
    axis.set(
        ylabel="Mean AOI proportion",
        title="Compositional group differences",
    )
    if n_parts:
        axis.legend(fontsize="small")
    axis.eyeprocess_plot_data = centroids
    return axis


def plot_aoi_composition_trajectory(x, ax=None, **kwargs):
    """Plot ordered AOI composition trajectories."""
    del kwargs
    composition = _require_composition(x)
    axis = _axis(ax)
    order = np.arange(1, len(composition) + 1)

    for part in composition.columns:
        axis.plot(
            order,
            composition[part].to_numpy(dtype=float),
            label=str(part),
        )

    axis.set(
        xlabel="Ordered observation",
        ylabel="AOI proportion",
        title="AOI composition trajectory",
    )
    axis.legend(fontsize="small")
    axis.eyeprocess_plot_data = composition
    return axis
