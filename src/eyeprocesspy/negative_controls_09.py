"""Frozen R/077 temporal leakage and negative-control utilities."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError

__all__ = [
    "audit_temporal_leakage",
    "negative_control_concordance",
    "outcome_blind_feature_audit",
    "placebo_window_audit",
    "process_feature_time_provenance",
    "process_negative_control_permute",
    "process_negative_control_shift",
    "process_null_benchmark",
    "run_process_negative_controls",
    "summarise_process_negative_controls",
    "validate_feature_availability",
]


def _as_frame(value: Any, *, name: str = "data") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"`{name}` must be coercible to a data frame.") from exc


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str = "data",
) -> None:
    missing = [column for column in columns if column is not None and column not in frame.columns]
    if missing:
        raise EyeProcessValidationError(
            f"`{name}` is missing required column(s): " + ", ".join(map(str, missing)) + "."
        )


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
    vector = _numeric_vector(value)
    return float(vector[0]) if vector.size else math.nan


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, pd.Series):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _recycle(value: Any, n: int, *, missing: Any = None) -> list[Any]:
    values = _as_list(value)
    if not values:
        return [missing] * n
    return [values[index % len(values)] for index in range(n)]


def _normalize_group_columns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(column) for column in list(value)]


def _group_indices(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[np.ndarray]:
    if not columns:
        return [np.arange(len(frame), dtype=int)]
    _require_columns(frame, columns)
    groups = frame.groupby(
        list(columns),
        sort=True,
        dropna=False,
        observed=False,
    ).indices
    return [np.asarray(indices, dtype=int) for indices in groups.values()]


def _capture_call(
    function: Callable[[Any], Any],
    argument: Any,
) -> dict[str, Any]:
    caught_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = function(argument)
        caught_messages.extend(str(item.message) for item in caught)
        return {"value": value, "error": None, "warnings": caught_messages}
    except Exception as exc:
        return {"value": None, "error": str(exc), "warnings": caught_messages}


def _default_control_extract(value: Any) -> pd.DataFrame:
    if np.isscalar(value) and not isinstance(value, (str, bytes)):
        return pd.DataFrame({"effect": [_first_numeric(value)]})
    if isinstance(value, Mapping) and "effect" in value:
        return pd.DataFrame({"effect": [_first_numeric(value["effect"])]})
    if isinstance(value, pd.DataFrame):
        return value.copy()
    raise EyeProcessValidationError(
        "analysis_fun output must be numeric, a list with $effect, or a data.frame, unless extract_fun is supplied."
    )


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    try:
        return float(np.quantile(finite, probability, method="linear"))
    except TypeError:
        return float(np.quantile(finite, probability, interpolation="linear"))


def process_feature_time_provenance(
    feature,
    available_at,
    outcome_at,
    source=None,
    transformation=None,
    unit="ms",
):
    lengths = [
        len(_as_list(feature)),
        len(_as_list(available_at)),
        len(_as_list(outcome_at)),
        len(_as_list(source)),
        len(_as_list(transformation)),
    ]
    n = max(lengths)
    if n < 1:
        raise EyeProcessValidationError("At least one feature must be supplied.")

    features = _recycle(feature, n)
    available = _numeric_vector(_recycle(available_at, n))
    outcome = _numeric_vector(_recycle(outcome_at, n))
    sources = _recycle(source, n)
    transformations = _recycle(transformation, n)
    unit_values = _as_list(unit)
    unit_value = str(unit_values[0]) if unit_values else "None"

    feature_strings = [None if value is None or pd.isna(value) else str(value) for value in features]
    if any(value is None or value == "" for value in feature_strings):
        raise EyeProcessValidationError("feature names cannot be missing or empty.")
    if np.any(~np.isfinite(available)) or np.any(~np.isfinite(outcome)):
        raise EyeProcessValidationError("available_at and outcome_at must be finite.")

    lead = outcome - available
    output = pd.DataFrame(
        {
            "feature": feature_strings,
            "available_at": available,
            "outcome_at": outcome,
            "source": sources,
            "transformation": transformations,
            "unit": [unit_value] * n,
            "lead": lead,
            "available_before_outcome": np.isfinite(lead) & (lead >= 0),
        }
    )
    output.attrs["eyeprocess_class"] = "eye_feature_time_provenance"
    return output


def audit_temporal_leakage(
    provenance,
    allow_equal=True,
    tolerance=0,
):
    frame = _as_frame(provenance, name="provenance")
    _require_columns(
        frame,
        ["feature", "available_at", "outcome_at"],
        name="provenance",
    )
    tolerance_value = _first_numeric(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0:
        raise EyeProcessValidationError("tolerance must be a non-negative finite scalar.")

    available = pd.to_numeric(frame["available_at"], errors="coerce").to_numpy(dtype=float)
    outcome = pd.to_numeric(frame["outcome_at"], errors="coerce").to_numpy(dtype=float)
    delta = available - outcome
    leak = delta > tolerance_value if bool(allow_equal) else delta >= -tolerance_value

    detail = pd.DataFrame(
        {
            "feature": frame["feature"].astype(str).to_numpy(),
            "available_at": available,
            "outcome_at": outcome,
            "temporal_delta": delta,
            "leakage_flag": leak,
        }
    )
    n_features = len(detail)
    n_flagged = int(np.sum(leak))
    return {
        "status": "flagged" if n_flagged else "pass",
        "n_features": n_features,
        "n_flagged": n_flagged,
        "flagged_fraction": float(np.mean(leak)) if n_features else math.nan,
        "detail": detail,
        "interpretation": (
            "A leakage flag denotes temporal/information contamination "
            "relative to the declared outcome boundary; it is not a "
            "misconduct label."
        ),
        "eyeprocess_class": "eye_temporal_leakage_audit",
    }


def validate_feature_availability(
    provenance,
    cutoff,
):
    frame = _as_frame(provenance, name="provenance")
    _require_columns(
        frame,
        ["feature", "available_at"],
        name="provenance",
    )
    if cutoff is None:
        raise EyeProcessValidationError("cutoff cannot be empty.")

    if isinstance(cutoff, pd.Series):
        cutoff_items = cutoff.tolist()
        cutoff_names = [str(value) for value in cutoff.index]
    elif isinstance(cutoff, Mapping):
        cutoff_names = [str(value) for value in cutoff.keys()]
        cutoff_items = list(cutoff.values())
    else:
        cutoff_items = _as_list(cutoff)
        cutoff_names = []

    if not cutoff_items:
        raise EyeProcessValidationError("cutoff cannot be empty.")

    if len(cutoff_items) == 1:
        limit = np.repeat(_first_numeric(cutoff_items[0]), len(frame))
    else:
        if not cutoff_names or len(cutoff_names) != len(cutoff_items) or any(name == "" for name in cutoff_names):
            raise EyeProcessValidationError("A multi-value cutoff must be named by feature.")
        mapping = dict(zip(cutoff_names, cutoff_items))
        features = frame["feature"].astype(str).tolist()
        if any(feature not in mapping for feature in features):
            raise EyeProcessValidationError("A named cutoff is required for every feature.")
        limit = np.asarray(
            [_first_numeric(mapping[feature]) for feature in features],
            dtype=float,
        )

    if np.any(~np.isfinite(limit)):
        raise EyeProcessValidationError("cutoff values must be finite.")

    available = pd.to_numeric(frame["available_at"], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "feature": frame["feature"].to_numpy(),
            "available_at": available,
            "cutoff": limit,
            "available": available <= limit,
        }
    )


def outcome_blind_feature_audit(
    data,
    outcome,
    feature_fun,
):
    frame = _as_frame(data)
    if outcome not in frame.columns:
        raise EyeProcessValidationError("outcome column not found.")
    if not callable(feature_fun):
        raise EyeProcessValidationError("feature_fun must be a function.")

    hidden = frame.drop(columns=[outcome])
    captured = _capture_call(feature_fun, hidden)
    passed = captured["error"] is None
    return {
        "status": "pass" if passed else "failed",
        "outcome": outcome,
        "feature_result": captured["value"] if passed else None,
        "error": captured["error"],
        "warnings": captured["warnings"],
        "input_columns": list(hidden.columns),
        "interpretation": (
            "Passing establishes that this feature-construction call "
            "executed without the supplied outcome column; it does not "
            "prove absence of all indirect leakage."
        ),
        "eyeprocess_class": "eye_outcome_blind_feature_audit",
    }


def process_negative_control_permute(
    data,
    outcome,
    seed=1,
    within=None,
):
    frame = _as_frame(data)
    if not isinstance(outcome, str) or outcome == "":
        raise EyeProcessValidationError("outcome must name one column.")
    _require_columns(frame, [outcome])

    seed_value = _first_numeric(seed)
    if not np.isfinite(seed_value) or seed_value < 0:
        raise EyeProcessValidationError("seed must be a non-negative finite scalar.")
    seed_int = int(seed_value % np.iinfo(np.int32).max)

    group_columns = _normalize_group_columns(within)
    _require_columns(frame, group_columns)
    rng = np.random.default_rng(seed_int)
    source = frame[outcome].to_numpy(copy=True)
    result = source.copy()

    if not group_columns:
        result = rng.permutation(source)
    else:
        for indices in _group_indices(frame, group_columns):
            result[indices] = rng.permutation(source[indices])

    frame[outcome] = result
    frame.attrs["negative_control"] = {
        "type": "permutation",
        "outcome": outcome,
        "seed": seed_int,
        "within": group_columns if group_columns else None,
    }
    return frame


def _shift_values(values: pd.Series, lag: int) -> pd.Series:
    n = len(values)
    if n == 0 or lag == 0:
        return values.copy()
    if abs(lag) >= n:
        return pd.Series([np.nan] * n, index=values.index, dtype=object)

    raw = values.tolist()
    shifted = [np.nan] * lag + raw[: n - lag] if lag > 0 else raw[-lag:] + [np.nan] * (-lag)
    return pd.Series(shifted, index=values.index)


def process_negative_control_shift(
    data,
    column,
    lag=1,
    by=None,
):
    frame = _as_frame(data)
    group_columns = _normalize_group_columns(by)
    _require_columns(frame, [column] + group_columns)

    lag_value = _first_numeric(lag)
    if not np.isfinite(lag_value) or lag_value != round(lag_value):
        raise EyeProcessValidationError("lag must be a finite integer.")
    lag_int = int(lag_value)

    if not group_columns:
        frame[column] = _shift_values(frame[column], lag_int).to_numpy()
    else:
        output = frame[column].astype(object).copy()
        for indices in _group_indices(frame, group_columns):
            shifted = _shift_values(frame.iloc[indices][column], lag_int)
            output.iloc[indices] = shifted.to_numpy()
        frame[column] = output

    frame.attrs["negative_control"] = {
        "type": "temporal_shift",
        "column": column,
        "lag": lag_int,
        "by": group_columns if group_columns else None,
    }
    return frame


def _placebo_summary(
    frame: pd.DataFrame,
    value: str,
    expected: float,
) -> dict[str, float | int]:
    values = _numeric_vector(frame[value])
    values = values[np.isfinite(values)]
    n = int(len(values))
    mean_value = float(np.mean(values)) if n else math.nan
    if n > 1:
        sd_value = float(np.std(values, ddof=1))
        se_value = sd_value / math.sqrt(n)
    else:
        sd_value = math.nan
        se_value = math.nan
    return {
        "n": n,
        "mean": mean_value,
        "sd": sd_value,
        "se": se_value,
        "difference_from_expected": (mean_value - expected if np.isfinite(mean_value) else math.nan),
    }


def placebo_window_audit(
    data,
    time,
    value,
    window,
    expected=0,
    by=None,
):
    frame = _as_frame(data)
    group_columns = _normalize_group_columns(by)
    _require_columns(frame, [time, value] + group_columns)

    window_values = _numeric_vector(window)
    if len(window_values) != 2 or np.any(~np.isfinite(window_values)):
        raise EyeProcessValidationError("window must contain two finite values.")
    lower = float(np.min(window_values))
    upper = float(np.max(window_values))

    expected_value = _first_numeric(expected)
    if not np.isfinite(expected_value):
        raise EyeProcessValidationError("expected must be a finite scalar.")

    time_values = pd.to_numeric(frame[time], errors="coerce").to_numpy(dtype=float)
    subset = frame.loc[(time_values >= lower) & (time_values <= upper)].copy()

    rows = []
    if not group_columns:
        rows.append(_placebo_summary(subset, value, expected_value))
    elif len(subset):
        for _, group in subset.groupby(
            group_columns,
            sort=True,
            dropna=False,
            observed=False,
        ):
            row = {column: group.iloc[0][column] for column in group_columns}
            row.update(_placebo_summary(group, value, expected_value))
            rows.append(row)

    output = pd.DataFrame(rows)
    output.attrs["placebo_window"] = [lower, upper]
    output.attrs["expected"] = expected_value
    output.attrs["eyeprocess_class"] = "eye_placebo_window_audit"
    return output


def run_process_negative_controls(
    data,
    outcome,
    analysis_fun,
    controls=("permutation", "shift"),
    replications=100,
    seed=1,
    extract_fun=None,
    shift_lags=(-3, -2, -1, 1, 2, 3),
    within=None,
):
    frame = _as_frame(data)
    _require_columns(frame, [outcome])

    controls_value = [controls] if isinstance(controls, str) else list(controls)
    if not controls_value or any(c not in {"permutation", "shift"} for c in controls_value):
        raise EyeProcessValidationError("controls must contain 'permutation' and/or 'shift'.")

    rep_value = _first_numeric(replications)
    if not np.isfinite(rep_value) or rep_value < 1 or rep_value != round(rep_value):
        raise EyeProcessValidationError("replications must be a positive integer.")
    rep_int = int(rep_value)

    lags = _numeric_vector(shift_lags)
    if "shift" in controls_value and len(lags) == 0:
        raise EyeProcessValidationError("shift_lags cannot be empty when shift controls are requested.")
    if "shift" in controls_value and (
        np.any(~np.isfinite(lags)) or np.any(lags != np.round(lags)) or np.any(lags == 0)
    ):
        raise EyeProcessValidationError("shift_lags must contain non-zero finite integers.")
    lags = lags.astype(int)

    extractor = _default_control_extract if extract_fun is None else extract_fun
    if not callable(analysis_fun) or not callable(extractor):
        raise EyeProcessValidationError("analysis_fun and extract_fun must be functions.")

    seed_value = _first_numeric(seed)
    if not np.isfinite(seed_value) or seed_value < 0:
        raise EyeProcessValidationError("seed must be a non-negative finite scalar.")
    seed_int = int(seed_value % np.iinfo(np.int32).max)

    rows = []
    counter = 0
    for control in controls_value:
        for replication in range(1, rep_int + 1):
            counter += 1
            if control == "permutation":
                controlled = process_negative_control_permute(
                    frame,
                    outcome,
                    seed=(seed_int + counter) % np.iinfo(np.int32).max,
                    within=within,
                )
            else:
                lag = int(lags[(replication - 1) % len(lags)])
                controlled = process_negative_control_shift(
                    frame,
                    outcome,
                    lag=lag,
                    by=within,
                )

            captured = _capture_call(analysis_fun, controlled)
            if captured["error"] is None:
                extracted = _capture_call(extractor, captured["value"])
            else:
                extracted = {
                    "value": None,
                    "error": captured["error"],
                    "warnings": captured["warnings"],
                }

            if extracted["error"] is None:
                try:
                    result = _as_frame(
                        extracted["value"],
                        name="extract_fun output",
                    )
                except Exception:
                    result = pd.DataFrame()
            else:
                result = pd.DataFrame()

            if len(result) == 0:
                result = pd.DataFrame({"effect": [math.nan]})

            result = result.copy()
            result["control"] = control
            result["replication"] = replication
            result["error"] = extracted["error"]
            all_warnings = list(dict.fromkeys(captured["warnings"] + extracted["warnings"]))
            result["warnings"] = " | ".join(all_warnings)
            rows.append(result)

    return {
        "results": pd.concat(rows, ignore_index=True, sort=False),
        "outcome": outcome,
        "controls": controls_value,
        "replications": rep_int,
        "seed": seed_int,
        "interpretation": (
            "Negative controls probe whether the analysis produces "
            "signal after deliberately disrupting the declared "
            "outcome/process relation; they do not prove validity when "
            "null-like."
        ),
        "eyeprocess_class": "eye_process_negative_controls",
    }


def _is_negative_control_object(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("eyeprocess_class") == "eye_process_negative_controls"
        and "results" in value
    )


def summarise_process_negative_controls(
    x,
    effect="effect",
    threshold=0,
):
    if not _is_negative_control_object(x):
        raise EyeProcessValidationError("x must be an eye_process_negative_controls object.")
    frame = _as_frame(x["results"])
    _require_columns(frame, ["control", effect])

    threshold_value = _first_numeric(threshold)
    if not np.isfinite(threshold_value):
        raise EyeProcessValidationError("threshold must be a finite scalar.")

    rows = []
    for control, group in frame.groupby(
        "control",
        sort=True,
        dropna=False,
        observed=False,
    ):
        values = pd.to_numeric(group[effect], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "control": control,
                "n": int(len(values)),
                "n_finite": int(len(finite)),
                "mean": float(np.mean(finite)) if len(finite) else math.nan,
                "sd": (float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan),
                "median": (float(np.median(finite)) if len(finite) else math.nan),
                "q025": _quantile(values, 0.025),
                "q975": _quantile(values, 0.975),
                "exceedance_rate": (float(np.mean(np.abs(finite) > abs(threshold_value))) if len(finite) else math.nan),
            }
        )
    return pd.DataFrame(rows)


def process_null_benchmark(
    observed,
    controls,
    effect="effect",
):
    if _is_negative_control_object(controls):
        frame = _as_frame(controls["results"])
        _require_columns(frame, [effect])
        null = _numeric_vector(frame[effect])
    else:
        null = _numeric_vector(controls)

    null = null[np.isfinite(null)]
    observed_value = _first_numeric(observed)
    if len(null) == 0 or not np.isfinite(observed_value):
        return {
            "observed": observed_value,
            "n_null": int(len(null)),
            "percentile": math.nan,
            "two_sided_tail": math.nan,
        }

    mean_value = float(np.mean(null))
    sd_value = float(np.std(null, ddof=1)) if len(null) > 1 else math.nan
    return {
        "observed": observed_value,
        "n_null": int(len(null)),
        "null_mean": mean_value,
        "null_sd": sd_value,
        "percentile": float(np.mean(null <= observed_value)),
        "two_sided_tail": float(np.mean(np.abs(null) >= abs(observed_value))),
        "standardized_distance": (
            (observed_value - mean_value) / sd_value
            if len(null) > 1 and np.isfinite(sd_value) and sd_value > 0
            else math.nan
        ),
    }


def negative_control_concordance(
    x,
    effect="effect",
    tolerance=0.05,
):
    tolerance_value = _first_numeric(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0:
        raise EyeProcessValidationError("tolerance must be a non-negative finite scalar.")
    summary = summarise_process_negative_controls(x, effect=effect)
    means = pd.to_numeric(summary["mean"], errors="coerce").to_numpy(dtype=float)

    within = np.empty(len(summary), dtype=object)
    for index, mean_value in enumerate(means):
        within[index] = bool(abs(mean_value) <= tolerance_value) if np.isfinite(mean_value) else None

    summary = summary.copy()
    summary["within_tolerance"] = within
    finite = [value for value in within if value is not None]
    return {
        "summary": summary,
        "all_within_tolerance": bool(all(finite)) if finite else None,
        "tolerance": tolerance_value,
    }
