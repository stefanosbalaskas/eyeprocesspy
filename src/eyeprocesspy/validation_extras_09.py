"""Frozen R/081 validation extras for eyeprocesspy.

This module source-ports the eight public contracts from
``R/081-validation-extras-0-9.R`` in frozen eyeprocess 0.11.1:

- ``simulation_rank_statistic``
- ``sbc_rank_diagnostics``
- ``sbc_ecdf_deviation``
- ``coverage_calibration_curve``
- ``measurement_error_budget``
- ``analysis_resolution_guard``
- ``audit_pupil_preprocessing_order``
- ``pupil_baseline_sensitivity``

The frozen source also defines S3 print/plot methods; those are method
surfaces rather than additional public function rows in the Python parity ledger.

Core numerical/data dependencies only are imported at module load.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .exceptions import EyeProcessValidationError

__all__ = [
    "analysis_resolution_guard",
    "audit_pupil_preprocessing_order",
    "coverage_calibration_curve",
    "measurement_error_budget",
    "pupil_baseline_sensitivity",
    "sbc_ecdf_deviation",
    "sbc_rank_diagnostics",
    "simulation_rank_statistic",
]


def _numeric_vector(value: Any) -> np.ndarray:
    if isinstance(value, pd.Series):
        raw = value.to_numpy()
    elif isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (str, bytes)):
        raw = np.asarray([value], dtype=object)
    else:
        try:
            raw = np.asarray(value)
            if raw.ndim == 0:
                raw = raw.reshape(1)
        except Exception:
            raw = np.asarray([value], dtype=object)

    return pd.to_numeric(
        pd.Series(np.ravel(raw)),
        errors="coerce",
    ).to_numpy(dtype=float)


def _first_numeric(value: Any) -> float:
    numeric = _numeric_vector(value)
    return float(numeric[0]) if numeric.size else math.nan


def _as_frame(value: Any, *, name: str = "data") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"`{name}` must be coercible to a data frame.") from exc


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EyeProcessValidationError("Missing required column(s): " + ", ".join(missing) + ".")


def _recycle(values: Any, length: int) -> list[Any]:
    if isinstance(values, (str, bytes)):
        seq = [values]
    else:
        try:
            seq = list(values)
        except TypeError:
            seq = [values]

    if not seq:
        return [None] * length
    return [seq[index % len(seq)] for index in range(length)]


def simulation_rank_statistic(
    truth,
    draws,
    seed=None,
):
    """Compute the frozen simulation-based calibration rank statistic.

    Ties are randomized uniformly over ranks ``less`` through
    ``less + equal``. Seeded NumPy draws preserve the frozen algorithm and
    seed contract, but are not claimed to reproduce R's RNG stream exactly.
    """
    truth_value = _first_numeric(truth)
    draws_value = _numeric_vector(draws)
    draws_value = draws_value[np.isfinite(draws_value)]

    if not np.isfinite(truth_value) or draws_value.size == 0:
        return None

    normalized_seed = None
    if seed is not None:
        seed_value = _first_numeric(seed)
        if not np.isfinite(seed_value) or seed_value < 0:
            raise EyeProcessValidationError("seed must be a non-negative finite scalar.")
        normalized_seed = int(seed_value % np.iinfo(np.int32).max)

    less = int(np.sum(draws_value < truth_value))
    equal = int(np.sum(draws_value == truth_value))

    if equal == 0:
        return less

    rng = np.random.default_rng(normalized_seed)
    tie_offset = int(rng.integers(0, equal + 1))
    return less + tie_offset


def sbc_rank_diagnostics(
    ranks,
    n_draws,
    bins=None,
):
    """Build frozen SBC rank diagnostics."""
    ranks0 = _numeric_vector(ranks)
    n_draws0 = _first_numeric(n_draws)

    if not np.isfinite(n_draws0) or n_draws0 < 1 or n_draws0 != round(n_draws0):
        raise EyeProcessValidationError("n_draws must be a positive integer.")

    finite_ranks = ranks0[np.isfinite(ranks0)]
    if np.any(finite_ranks != np.round(finite_ranks)):
        raise EyeProcessValidationError("ranks must be integer-valued.")

    n_draws_int = int(n_draws0)
    ranks_int = finite_ranks.astype(int)

    if np.any((ranks_int < 0) | (ranks_int > n_draws_int)):
        raise EyeProcessValidationError("ranks must lie between 0 and n_draws inclusive.")

    if bins is None:
        bins_int = max(
            5,
            min(
                20,
                int(round(math.sqrt(max(1, len(ranks_int))))),
            ),
        )
    else:
        bins_value = _first_numeric(bins)
        if not np.isfinite(bins_value):
            raise EyeProcessValidationError("bins must be an integer >= 2.")
        # Frozen R applies as.integer() before validating, so finite
        # non-integers are truncated toward zero.
        bins_int = int(bins_value)
        if bins_int < 2:
            raise EyeProcessValidationError("bins must be an integer >= 2.")

    bins_int = min(bins_int, n_draws_int + 1)

    breaks = np.linspace(
        -0.5,
        n_draws_int + 0.5,
        bins_int + 1,
        dtype=float,
    )

    def r_hist_counts(values):
        values = np.asarray(values, dtype=float)
        # R hist() defaults to right=TRUE: (a, b], with the lowest endpoint
        # included in the first interval. Rank support never equals -0.5,
        # but the explicit clipping preserves the complete rule.
        indices = (
            np.searchsorted(
                breaks,
                values,
                side="left",
            )
            - 1
        )
        indices[values == breaks[0]] = 0
        valid = (indices >= 0) & (indices < bins_int)
        return np.bincount(
            indices[valid],
            minlength=bins_int,
        )[:bins_int]

    counts = r_hist_counts(ranks_int)
    support = r_hist_counts(np.arange(n_draws_int + 1, dtype=int))

    expected = len(ranks_int) * support.astype(float) / (n_draws_int + 1)
    ok_expected = expected > 0

    if np.any(ok_expected):
        chi_square = float(np.sum((counts[ok_expected] - expected[ok_expected]) ** 2 / expected[ok_expected]))
    else:
        chi_square = math.nan

    degrees_freedom = int(np.sum(ok_expected) - 1)
    chi_square_p = (
        float(chi2.sf(chi_square, degrees_freedom)) if np.isfinite(chi_square) and degrees_freedom > 0 else math.nan
    )

    uniformized = np.sort((ranks_int.astype(float) + 0.5) / (n_draws_int + 1))
    if uniformized.size:
        empirical = np.arange(1, uniformized.size + 1, dtype=float) / uniformized.size
        ecdf_max_deviation = float(np.max(np.abs(empirical - uniformized)))
    else:
        ecdf_max_deviation = math.nan

    return {
        "ranks": ranks_int,
        "n_draws": n_draws_int,
        "bins": bins_int,
        "counts": counts.astype(int),
        "breaks": breaks,
        "expected_count": expected,
        "chi_square": chi_square,
        "chi_square_p": chi_square_p,
        "ecdf_max_deviation": ecdf_max_deviation,
        "caveat": (
            "SBC diagnoses calibration of the supplied "
            "simulator-model-inference workflow; it does not establish "
            "substantive model adequacy for empirical data."
        ),
        "eyeprocess_class": "eye_sbc_diagnostics",
    }


def sbc_ecdf_deviation(
    x,
    n_draws=None,
):
    """Return the frozen maximum ECDF deviation for SBC ranks."""
    if isinstance(x, Mapping) and (x.get("eyeprocess_class") == "eye_sbc_diagnostics" or "ecdf_max_deviation" in x):
        return float(x["ecdf_max_deviation"])

    diagnostics = sbc_rank_diagnostics(
        x,
        n_draws,
    )
    return float(diagnostics["ecdf_max_deviation"])


def _interval_matrix(value: Any) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        matrix = value.to_numpy()
    elif isinstance(value, pd.Series):
        matrix = value.to_numpy().reshape(-1, 1)
    else:
        matrix = np.asarray(value)
        if matrix.ndim == 0:
            matrix = matrix.reshape(1, 1)
        elif matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

    if matrix.ndim != 2:
        raise EyeProcessValidationError("interval limits must be vectors or two-dimensional tables.")

    numeric = np.empty(matrix.shape, dtype=float)
    for column in range(matrix.shape[1]):
        numeric[:, column] = pd.to_numeric(
            pd.Series(matrix[:, column]),
            errors="coerce",
        ).to_numpy(dtype=float)
    return numeric


def coverage_calibration_curve(
    truth,
    lower,
    upper,
    nominal=None,
):
    """Build the frozen interval coverage calibration curve."""
    truth_value = _numeric_vector(truth)
    lower_value = _interval_matrix(lower)
    upper_value = _interval_matrix(upper)

    if lower_value.shape != upper_value.shape:
        raise EyeProcessValidationError("lower and upper must have the same dimensions.")

    if lower_value.shape[0] != len(truth_value):
        raise EyeProcessValidationError("truth length must match interval rows.")

    n_intervals = lower_value.shape[1]
    if nominal is None:
        if n_intervals == 1:
            nominal_value = np.asarray([0.95], dtype=float)
        else:
            nominal_value = np.linspace(
                0.5,
                0.95,
                n_intervals,
                dtype=float,
            )
    else:
        nominal_value = _numeric_vector(nominal)

    if (
        len(nominal_value) != n_intervals
        or np.any(~np.isfinite(nominal_value))
        or np.any(nominal_value < 0)
        or np.any(nominal_value > 1)
    ):
        raise EyeProcessValidationError("nominal must contain one finite probability in [0,1] per interval column.")

    empirical = np.full(n_intervals, np.nan, dtype=float)
    for column in range(n_intervals):
        valid = ~np.isnan(truth_value) & ~np.isnan(lower_value[:, column]) & ~np.isnan(upper_value[:, column])
        if np.any(valid):
            covered = (lower_value[valid, column] <= truth_value[valid]) & (
                upper_value[valid, column] >= truth_value[valid]
            )
            empirical[column] = float(np.mean(covered))

    return pd.DataFrame(
        {
            "nominal": nominal_value,
            "empirical": empirical,
            "error": empirical - nominal_value,
        }
    )


def measurement_error_budget(
    accuracy=math.nan,
    precision=math.nan,
    data_loss=math.nan,
    effective_hz=math.nan,
    calibration_drift=math.nan,
    units=None,
):
    """Build the frozen non-collapsed measurement-error budget."""
    values = [
        _first_numeric(accuracy),
        _first_numeric(precision),
        _first_numeric(data_loss),
        _first_numeric(effective_hz),
        _first_numeric(calibration_drift),
    ]

    output = pd.DataFrame(
        {
            "component": [
                "accuracy_error",
                "precision_error",
                "data_loss",
                "effective_sampling_hz",
                "calibration_drift",
            ],
            "value": values,
            "direction": [
                "lower_better",
                "lower_better",
                "lower_better",
                "context_dependent",
                "lower_better",
            ],
        }
    )

    if units is not None:
        output["unit"] = _recycle(units, len(output))

    output.attrs["interpretation"] = (
        "Components remain separate because they describe different "
        "measurement limitations and should not be collapsed without a "
        "study-specific justification."
    )
    output.attrs["eyeprocess_class"] = "eye_measurement_error_budget"
    return output


def analysis_resolution_guard(
    event_duration_ms,
    effective_hz,
    spatial_feature_size=math.nan,
    radial_error=math.nan,
    min_samples=3,
    max_error_fraction=0.5,
):
    """Audit compatibility between measurement resolution and analysis target."""
    event_duration = _first_numeric(event_duration_ms)
    sampling_hz = _first_numeric(effective_hz)
    minimum_samples = _first_numeric(min_samples)
    maximum_error_fraction = _first_numeric(max_error_fraction)

    if not np.isfinite(event_duration) or event_duration <= 0:
        raise EyeProcessValidationError("event_duration_ms must be positive.")
    if not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise EyeProcessValidationError("effective_hz must be positive.")
    if not np.isfinite(minimum_samples) or minimum_samples <= 0:
        raise EyeProcessValidationError("min_samples must be positive.")
    if not np.isfinite(maximum_error_fraction) or maximum_error_fraction < 0:
        raise EyeProcessValidationError("max_error_fraction must be non-negative.")

    feature_size = _first_numeric(spatial_feature_size)
    radial = _first_numeric(radial_error)

    if np.isfinite(feature_size) and feature_size <= 0:
        raise EyeProcessValidationError("spatial_feature_size must be positive when supplied.")
    if np.isfinite(radial) and radial < 0:
        raise EyeProcessValidationError("radial_error must be non-negative when supplied.")

    expected_samples = event_duration / 1000 * sampling_hz
    temporal_ok = bool(expected_samples >= minimum_samples)

    with np.errstate(divide="ignore", invalid="ignore"):
        spatial_ratio = float(radial / feature_size)

    if np.isfinite(spatial_ratio):
        spatial_ok: bool | None = bool(spatial_ratio <= maximum_error_fraction)
    else:
        spatial_ok = None

    overall = bool(temporal_ok and (spatial_ok is None or spatial_ok is True))

    return {
        "expected_samples": expected_samples,
        "temporal_ok": temporal_ok,
        "spatial_error_fraction": spatial_ratio,
        "spatial_ok": spatial_ok,
        "min_samples": minimum_samples,
        "max_error_fraction": maximum_error_fraction,
        "overall": overall,
        "caveat": (
            "Thresholds are researcher-declared compatibility rules. "
            "eyeprocess does not impose universal temporal or spatial "
            "quality cutoffs."
        ),
        "eyeprocess_class": "eye_analysis_resolution_guard",
    }


def audit_pupil_preprocessing_order(
    steps,
    cleaning_patterns=(
        "blink",
        "missing",
        "interpol",
        "artifact",
        "smooth",
        "filter",
    ),
    baseline_pattern="baseline",
):
    """Audit the declared order of pupil preprocessing steps."""
    if isinstance(steps, (str, bytes)):
        steps_value = [str(steps)]
    else:
        try:
            steps_value = list(steps)
        except TypeError as exc:
            raise EyeProcessValidationError("steps must contain non-empty step labels.") from exc

    if not steps_value or any(pd.isna(step) or str(step) == "" for step in steps_value):
        raise EyeProcessValidationError("steps must contain non-empty step labels.")

    steps_value = [str(step) for step in steps_value]

    if isinstance(cleaning_patterns, (str, bytes)):
        cleaning_value = [str(cleaning_patterns)]
    else:
        try:
            cleaning_value = list(cleaning_patterns)
        except TypeError as exc:
            raise EyeProcessValidationError("cleaning_patterns must be non-empty strings.") from exc

    if not cleaning_value or any(pd.isna(pattern) or str(pattern) == "" for pattern in cleaning_value):
        raise EyeProcessValidationError("cleaning_patterns must be non-empty strings.")
    cleaning_value = [str(pattern) for pattern in cleaning_value]

    if isinstance(
        baseline_pattern,
        (str, bytes),
    ):
        baseline_raw = baseline_pattern
    else:
        try:
            baseline_values = list(baseline_pattern)
        except TypeError:
            baseline_values = [baseline_pattern]
        baseline_raw = baseline_values[0] if baseline_values else None

    baseline_value = str(baseline_raw) if baseline_raw is not None and not pd.isna(baseline_raw) else ""
    if baseline_value == "":
        raise EyeProcessValidationError("baseline_pattern must be a non-empty string.")

    lowered = [step.lower() for step in steps_value]

    baseline_positions = [
        index + 1 for index, step in enumerate(lowered) if re.search(baseline_value, step) is not None
    ]

    cleaning_positions = []
    for pattern in cleaning_value:
        for index, step in enumerate(lowered):
            position = index + 1
            if re.search(pattern, step) is not None and position not in cleaning_positions:
                cleaning_positions.append(position)

    if baseline_positions and cleaning_positions:
        first_baseline = min(baseline_positions)
        late_positions = [position for position in cleaning_positions if position > first_baseline]
    else:
        late_positions = []

    cleaning_after_baseline = [steps_value[position - 1] for position in late_positions]

    if not baseline_positions:
        status = "baseline_not_declared"
    elif late_positions:
        status = "review"
    else:
        status = "pass"

    return {
        "steps": steps_value,
        "baseline_positions": baseline_positions,
        "cleaning_positions": cleaning_positions,
        "cleaning_after_baseline": cleaning_after_baseline,
        "status": status,
        "caveat": (
            "This is an order audit. Appropriate preprocessing still "
            "depends on the signal, task, device, and analysis plan."
        ),
        "eyeprocess_class": "eye_pupil_preprocessing_order_audit",
    }


def _normalize_windows(windows: Any) -> list[tuple[str, Any]]:
    if isinstance(windows, Mapping):
        items = list(windows.items())
        if not items:
            raise EyeProcessValidationError("windows must be a non-empty named list.")
        if any(str(name) == "" for name, _ in items):
            return [
                (f"W{index}", value)
                for index, (_, value) in enumerate(
                    items,
                    start=1,
                )
            ]
        return [(str(name), value) for name, value in items]

    if isinstance(windows, (str, bytes)):
        raise EyeProcessValidationError("windows must be a non-empty named list.")

    try:
        values = list(windows)
    except TypeError as exc:
        raise EyeProcessValidationError("windows must be a non-empty named list.") from exc

    if not values:
        raise EyeProcessValidationError("windows must be a non-empty named list.")

    return [
        (f"W{index}", value)
        for index, value in enumerate(
            values,
            start=1,
        )
    ]


def pupil_baseline_sensitivity(
    data,
    time="time_ms",
    pupil="pupil",
    windows=None,
    by=None,
    correction="subtractive",
):
    """Evaluate frozen pupil baseline-window sensitivity."""
    if correction not in {
        "subtractive",
        "divisive",
    }:
        raise EyeProcessValidationError("`correction` must be 'subtractive' or 'divisive'.")

    frame = _as_frame(data)

    if by is None:
        group_columns: list[str] = []
    elif isinstance(by, str):
        group_columns = [by]
    else:
        group_columns = list(by)

    _require_columns(
        frame,
        [time, pupil] + group_columns,
    )

    window_items = _normalize_windows(windows)

    if group_columns:
        grouped = list(
            frame.groupby(
                group_columns,
                sort=False,
                dropna=False,
                observed=False,
            )
        )
    else:
        grouped = [(None, frame)]

    rows: list[dict[str, Any]] = []
    for window_name, window_raw in window_items:
        window = _numeric_vector(window_raw)
        if len(window) != 2 or np.any(~np.isfinite(window)):
            raise EyeProcessValidationError("Each baseline window must have two finite endpoints.")

        start = float(np.min(window))
        end = float(np.max(window))

        for _, group in grouped:
            time_value = pd.to_numeric(
                group[time],
                errors="coerce",
            ).to_numpy(dtype=float)
            pupil_value = pd.to_numeric(
                group[pupil],
                errors="coerce",
            ).to_numpy(dtype=float)

            baseline_values = pupil_value[(time_value >= start) & (time_value <= end) & np.isfinite(pupil_value)]
            baseline = float(np.mean(baseline_values)) if baseline_values.size else math.nan

            post_values = pupil_value[(time_value > end) & np.isfinite(pupil_value)]

            if post_values.size == 0 or not np.isfinite(baseline):
                corrected = math.nan
            elif correction == "subtractive":
                corrected = float(np.mean(post_values - baseline))
            elif baseline != 0:
                corrected = float(np.mean(post_values / baseline))
            else:
                corrected = math.nan

            row: dict[str, Any] = {}
            if group_columns:
                first = group.iloc[0]
                for column in group_columns:
                    row[column] = first[column]

            row.update(
                {
                    "window": window_name,
                    "start": start,
                    "end": end,
                    "baseline": baseline,
                    "corrected_post_mean": corrected,
                    "correction": correction,
                }
            )
            rows.append(row)

    output = pd.DataFrame(rows)
    output.attrs["eyeprocess_class"] = "eye_pupil_baseline_sensitivity"
    return output
