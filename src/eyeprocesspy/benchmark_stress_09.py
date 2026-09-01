"""Frozen R/075 computational benchmarking and synthetic stress tests."""

from __future__ import annotations

import hashlib
import math
import pickle
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError

__all__ = [
    "apply_synthetic_corruption",
    "benchmark_memory_estimate",
    "benchmark_scaling_curve",
    "eye_benchmark_design",
    "inject_aoi_label_noise",
    "inject_calibration_offset",
    "inject_device_shift",
    "inject_eye_missingness",
    "inject_pupil_dropout",
    "inject_sampling_jitter",
    "inject_trial_imbalance",
    "run_eye_benchmark",
    "stress_test_process_pipeline",
    "stress_test_summary",
    "stress_tolerance_frontier",
    "summarise_eye_benchmark",
    "synthetic_corruption_plan",
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


def _numeric(value: Any) -> np.ndarray:
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


def _first_number(value: Any) -> float:
    values = _numeric(value)
    return float(values[0]) if values.size else math.nan


def _positive_integer(value: Any, *, name: str) -> int:
    number = _first_number(value)
    if not np.isfinite(number):
        raise EyeProcessValidationError(f"{name} must be a positive integer.")
    integer = int(number)
    if integer < 1:
        raise EyeProcessValidationError(f"{name} must be a positive integer.")
    return integer


def _proportion(value: Any, *, name: str = "proportion") -> float:
    number = _first_number(value)
    if not np.isfinite(number) or number < 0 or number >= 1:
        raise EyeProcessValidationError(f"{name} must lie in [0, 1).")
    return number


def _seed(value: Any) -> int:
    values = _as_list(value)
    number = _first_number(value)
    if len(values) != 1 or not np.isfinite(number) or number < 0:
        raise EyeProcessValidationError("seed must be a finite non-negative scalar.")
    # Frozen R code maps seeds into 1..(.Machine$integer.max - 1).
    return int(number % (np.iinfo(np.int32).max - 1)) + 1


def _capture(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = function(*args, **kwargs)
        messages.extend(str(item.message) for item in caught)
        return {"value": value, "error": None, "warnings": messages}
    except Exception as exc:
        return {"value": None, "error": str(exc), "warnings": messages}


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Approximate recursive Python object size in bytes."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)

    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        size += sum(_deep_size(key, seen) + _deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_size(item, seen) for item in value)
    return int(size)


def _stable_hash(value: Any) -> str:
    try:
        payload = pickle.dumps(value, protocol=5)
    except Exception:
        payload = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _default_benchmark_generator(
    n: int,
    row: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng()
    persons = max(1, int(math.ceil(n / 50)))
    person_id = np.resize(
        np.repeat(np.arange(1, persons + 1), 50),
        n,
    )
    return pd.DataFrame(
        {
            "person_id": person_id,
            "timestamp_ms": np.arange(1, n + 1) * 1000.0 / 60.0,
            "gaze_x": rng.random(n),
            "gaze_y": rng.random(n),
            "pupil": rng.normal(3.5, 0.3, n),
            "valid": rng.random(n) > 0.05,
        }
    )


def _default_benchmark_operation(
    data: pd.DataFrame,
    row: pd.DataFrame | None = None,
) -> dict[str, float]:
    pupil = pd.to_numeric(data["pupil"], errors="coerce").to_numpy(dtype=float)
    valid = pd.Series(data["valid"]).astype(float).to_numpy()
    return {
        "mean_pupil": float(np.nanmean(pupil)),
        "valid_fraction": float(np.nanmean(valid)),
    }


def _default_sensitivity_extract(
    fit: Any,
    plan: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if np.isscalar(fit) and not isinstance(fit, (str, bytes)):
        return pd.DataFrame({"effect": [_first_number(fit)]})
    if isinstance(fit, pd.DataFrame):
        return fit.copy()
    if isinstance(fit, Mapping):
        row = {}
        for key, value in fit.items():
            values = _as_list(value)
            if len(values) == 1 and isinstance(
                values[0],
                (str, bytes, int, float, bool, np.generic),
            ):
                row[str(key)] = values[0]
        if row:
            return pd.DataFrame([row])
    raise EyeProcessValidationError(
        "Provide metric_fun for analysis results that are not scalar/list/data.frame summaries."
    )


def eye_benchmark_design(
    n_obs=(10_000, 100_000, 1_000_000),
    repetitions=3,
    label="eyeprocess_scaling",
):
    """Define a computational benchmark design."""
    values = _numeric(n_obs)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise EyeProcessValidationError("n_obs must contain positive integers.")
    integers = values.astype(int)
    if np.any(integers < 1):
        raise EyeProcessValidationError("n_obs must contain positive integers.")

    repetitions_int = _positive_integer(
        repetitions,
        name="repetitions",
    )
    labels = _as_list(label)
    label_value = str(labels[0]) if labels else ""

    rows = []
    counter = 0
    # R expand.grid: first supplied vector varies fastest.
    for repetition in range(1, repetitions_int + 1):
        for n_value in integers:
            counter += 1
            rows.append(
                {
                    "benchmark_id": f"B{counter:05d}",
                    "n_obs": int(n_value),
                    "repetition": repetition,
                    "label": label_value,
                }
            )

    output = pd.DataFrame(rows)
    output.attrs["eyeprocess_class"] = "eye_benchmark_design"
    return output


def run_eye_benchmark(
    design=None,
    generator=None,
    operation=None,
    gc_before=True,
    progress=False,
):
    """Run the frozen computational scaling-benchmark workflow."""
    if design is None:
        design = eye_benchmark_design()
    frame = _as_frame(design, name="design")
    _require_columns(frame, ["benchmark_id", "n_obs"], name="design")

    generator_fun = _default_benchmark_generator if generator is None else generator
    operation_fun = _default_benchmark_operation if operation is None else operation
    if not callable(generator_fun):
        raise EyeProcessValidationError("generator must be a function.")
    if not callable(operation_fun):
        raise EyeProcessValidationError("operation must be a function.")

    rows = []
    for _, raw_row in frame.iterrows():
        row = raw_row.to_frame().T
        benchmark_id = str(raw_row["benchmark_id"])
        n_value = int(_first_number(raw_row["n_obs"]))
        if bool(progress):
            print(f"benchmark {benchmark_id} n={n_value}")

        generated = _capture(generator_fun, n_value, row)
        if generated["error"] is not None:
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "n_obs": n_value,
                    "elapsed_sec": math.nan,
                    "input_bytes": math.nan,
                    "output_bytes": math.nan,
                    "status": "generator_error",
                    "error": generated["error"],
                }
            )
            continue

        input_value = generated["value"]
        if bool(gc_before):
            import gc

            gc.collect()

        started = time.perf_counter()
        operated = _capture(operation_fun, input_value, row)
        elapsed = time.perf_counter() - started

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "n_obs": n_value,
                "repetition": (raw_row["repetition"] if "repetition" in frame.columns else math.nan),
                "elapsed_sec": float(elapsed),
                "input_bytes": float(_deep_size(input_value)),
                "output_bytes": (float(_deep_size(operated["value"])) if operated["error"] is None else math.nan),
                "status": ("success" if operated["error"] is None else "operation_error"),
                "error": operated["error"],
            }
        )

    return {
        "design": frame,
        "results": pd.DataFrame(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "computational_benchmark",
        "caveat": ("Benchmark timings are hardware-, operating-system-, Python-version-, and workload-dependent."),
        "eyeprocess_class": "eye_benchmark_result",
    }


def _is_class(value: Any, class_name: str) -> bool:
    return isinstance(value, Mapping) and value.get("eyeprocess_class") == class_name


def summarise_eye_benchmark(x):
    """Summarise benchmark timing and memory by problem size."""
    if not _is_class(x, "eye_benchmark_result"):
        raise EyeProcessValidationError("x must be an eye_benchmark_result.")

    data = _as_frame(x["results"])
    success = data.loc[data["status"] == "success"].copy()
    if success.empty:
        return pd.DataFrame()

    rows = []
    for n_value in sorted(pd.unique(success["n_obs"])):
        group = success.loc[success["n_obs"] == n_value]
        elapsed = pd.to_numeric(group["elapsed_sec"], errors="coerce").to_numpy(dtype=float)
        input_bytes = pd.to_numeric(group["input_bytes"], errors="coerce").to_numpy(dtype=float)
        output_bytes = pd.to_numeric(group["output_bytes"], errors="coerce").to_numpy(dtype=float)

        median_elapsed = float(np.nanmedian(elapsed))
        rows.append(
            {
                "n_obs": n_value,
                "runs": len(group),
                "median_elapsed_sec": median_elapsed,
                "min_elapsed_sec": float(np.nanmin(elapsed)),
                "max_elapsed_sec": float(np.nanmax(elapsed)),
                "median_input_mb": float(np.nanmedian(input_bytes)) / 1024**2,
                "median_output_mb": float(np.nanmedian(output_bytes)) / 1024**2,
                "throughput_obs_per_sec": (
                    float(n_value) / median_elapsed if np.isfinite(median_elapsed) and median_elapsed > 0 else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def benchmark_scaling_curve(x):
    """Estimate the log-log scaling exponent from benchmark results."""
    summary = summarise_eye_benchmark(x)
    if summary.empty:
        return pd.DataFrame([{"exponent": math.nan, "intercept": math.nan, "n_sizes": 0}])

    valid = summary.loc[
        np.isfinite(summary["median_elapsed_sec"]) & (summary["median_elapsed_sec"] > 0) & (summary["n_obs"] > 0)
    ].copy()

    if len(valid) < 2:
        return pd.DataFrame(
            [
                {
                    "exponent": math.nan,
                    "intercept": math.nan,
                    "n_sizes": len(valid),
                }
            ]
        )

    slope, intercept = np.polyfit(
        np.log(valid["n_obs"].to_numpy(dtype=float)),
        np.log(valid["median_elapsed_sec"].to_numpy(dtype=float)),
        1,
    )
    return pd.DataFrame(
        [
            {
                "exponent": float(slope),
                "intercept": float(intercept),
                "n_sizes": len(valid),
            }
        ]
    )


def benchmark_memory_estimate(x, generator=None):
    """Estimate recursive Python memory for an object or generated size."""
    value = generator(int(_first_number(x))) if callable(generator) else x
    byte_count = float(_deep_size(value))
    return pd.DataFrame(
        [
            {
                "bytes": byte_count,
                "kb": byte_count / 1024,
                "mb": byte_count / 1024**2,
                "gb": byte_count / 1024**3,
            }
        ]
    )


def synthetic_corruption_plan(
    missingness=0,
    pupil_dropout=0,
    gaze_offset_x=0,
    gaze_offset_y=0,
    sampling_jitter_sd=0,
    aoi_label_noise=0,
    device_shift=0,
    trial_drop=0,
    seed=1,
):
    """Define explicit synthetic measurement corruptions."""
    probability_names = [
        "missingness",
        "pupil_dropout",
        "aoi_label_noise",
        "trial_drop",
    ]
    probability_values = [
        missingness,
        pupil_dropout,
        aoi_label_noise,
        trial_drop,
    ]
    probabilities = {}
    for name, value in zip(probability_names, probability_values):
        number = _first_number(value)
        if not np.isfinite(number) or number < 0 or number >= 1:
            raise EyeProcessValidationError("Corruption proportions must lie in [0, 1).")
        probabilities[name] = number

    offset_names = [
        "gaze_offset_x",
        "gaze_offset_y",
        "sampling_jitter_sd",
        "device_shift",
    ]
    offset_values = [
        gaze_offset_x,
        gaze_offset_y,
        sampling_jitter_sd,
        device_shift,
    ]
    offsets = {}
    for name, value in zip(offset_names, offset_values):
        number = _first_number(value)
        if not np.isfinite(number):
            raise EyeProcessValidationError("Corruption offsets, jitter, and device shift must be finite.")
        offsets[name] = number

    if offsets["sampling_jitter_sd"] < 0:
        raise EyeProcessValidationError("sampling_jitter_sd must be non-negative.")

    seed_value = _first_number(seed)
    if len(_as_list(seed)) != 1 or not np.isfinite(seed_value) or seed_value < 0:
        raise EyeProcessValidationError("seed must be a finite non-negative scalar.")

    return {
        **probabilities,
        **offsets,
        "seed": int(seed_value),
        "status": "synthetic_measurement_corruption",
        "eyeprocess_class": "eye_synthetic_corruption_plan",
    }


def inject_eye_missingness(
    data,
    columns,
    proportion,
    seed=1,
):
    """Inject generic missingness into selected columns."""
    frame = _as_frame(data)
    columns_value = [columns] if isinstance(columns, str) else list(columns)
    _require_columns(frame, columns_value, name="data")
    probability = _proportion(proportion)
    rng = np.random.default_rng(_seed(seed))

    for column in columns_value:
        mask = rng.random(len(frame)) < probability
        frame.loc[mask, column] = np.nan
    return frame


def inject_pupil_dropout(
    data,
    pupil="pupil",
    proportion=None,
    seed=1,
):
    """Inject pupil dropout."""
    if proportion is None:
        raise EyeProcessValidationError("proportion must lie in [0, 1).")
    return inject_eye_missingness(
        data,
        pupil,
        proportion,
        seed,
    )


def inject_calibration_offset(
    data,
    x="gaze_x",
    y="gaze_y",
    offset_x=0,
    offset_y=0,
):
    """Inject additive gaze calibration offset."""
    frame = _as_frame(data)
    _require_columns(frame, [x, y], name="data")
    dx = _first_number(offset_x)
    dy = _first_number(offset_y)
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise EyeProcessValidationError("offset_x and offset_y must be finite scalars.")

    frame[x] = pd.to_numeric(frame[x], errors="coerce") + dx
    frame[y] = pd.to_numeric(frame[y], errors="coerce") + dy
    return frame


def inject_sampling_jitter(
    data,
    time="timestamp_ms",
    sd=None,
    seed=1,
):
    """Inject Gaussian timestamp jitter."""
    frame = _as_frame(data)
    _require_columns(frame, [time], name="data")
    sd_value = _first_number(sd)
    if not np.isfinite(sd_value) or sd_value < 0:
        raise EyeProcessValidationError("sd must be a finite non-negative scalar.")
    rng = np.random.default_rng(_seed(seed))
    frame[time] = pd.to_numeric(frame[time], errors="coerce").to_numpy(dtype=float) + rng.normal(
        0.0, sd_value, len(frame)
    )
    return frame


def inject_aoi_label_noise(
    data,
    aoi="aoi",
    proportion=None,
    seed=1,
):
    """Randomly reassign a proportion of observed AOI labels."""
    frame = _as_frame(data)
    _require_columns(frame, [aoi], name="data")
    probability = _proportion(proportion)
    rng = np.random.default_rng(_seed(seed))

    observed = [str(value) for value in pd.unique(frame[aoi].dropna())]
    if len(observed) < 2:
        return frame

    draw = rng.random(len(frame))
    eligible = frame[aoi].notna().to_numpy() & (draw < probability)
    for index in np.flatnonzero(eligible):
        current = str(frame.iloc[index][aoi])
        choices = [value for value in observed if value != current]
        frame.iat[index, frame.columns.get_loc(aoi)] = rng.choice(choices)
    return frame


def inject_device_shift(
    data,
    column,
    shift,
    rows=None,
):
    """Inject an additive device/site shift in a numeric feature."""
    frame = _as_frame(data)
    _require_columns(frame, [column], name="data")
    shift_value = _first_number(shift)
    if not np.isfinite(shift_value):
        raise EyeProcessValidationError("shift must be a finite scalar.")

    if rows is None:
        indices = np.arange(len(frame))
    else:
        rows_array = np.asarray(rows)
        if rows_array.dtype == bool:
            if len(rows_array) != len(frame):
                raise EyeProcessValidationError("Logical rows must have length nrow(data).")
            indices = np.flatnonzero(rows_array)
        else:
            indices = rows_array.astype(int)

    numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=True)
    numeric[indices] = numeric[indices] + shift_value
    frame[column] = numeric
    return frame


def inject_trial_imbalance(
    data,
    proportion,
    seed=1,
):
    """Inject row/trial imbalance by dropping observations."""
    frame = _as_frame(data)
    probability = _proportion(proportion)
    rng = np.random.default_rng(_seed(seed))
    keep = rng.random(len(frame)) >= probability
    return frame.loc[keep].copy()


def _is_plan(plan: Any) -> bool:
    return isinstance(plan, Mapping) and plan.get("eyeprocess_class") == "eye_synthetic_corruption_plan"


def apply_synthetic_corruption(
    data,
    plan,
    gaze_columns=("gaze_x", "gaze_y"),
    pupil="pupil",
    time="timestamp_ms",
    aoi=None,
    device_column=None,
):
    """Apply one explicit synthetic corruption plan."""
    if not _is_plan(plan):
        raise EyeProcessValidationError("plan must be an eye_synthetic_corruption_plan.")

    frame = _as_frame(data)
    seed = int(plan["seed"])

    if plan["trial_drop"] > 0:
        frame = inject_trial_imbalance(
            frame,
            plan["trial_drop"],
            seed=seed + 1,
        )

    gaze = [column for column in gaze_columns if column in frame.columns]
    if gaze and plan["missingness"] > 0:
        frame = inject_eye_missingness(
            frame,
            gaze,
            plan["missingness"],
            seed=seed + 2,
        )

    if len(gaze) >= 2 and (plan["gaze_offset_x"] != 0 or plan["gaze_offset_y"] != 0):
        frame = inject_calibration_offset(
            frame,
            gaze[0],
            gaze[1],
            plan["gaze_offset_x"],
            plan["gaze_offset_y"],
        )

    if pupil in frame.columns and plan["pupil_dropout"] > 0:
        frame = inject_pupil_dropout(
            frame,
            pupil,
            plan["pupil_dropout"],
            seed=seed + 3,
        )

    if time in frame.columns and plan["sampling_jitter_sd"] > 0:
        frame = inject_sampling_jitter(
            frame,
            time,
            plan["sampling_jitter_sd"],
            seed=seed + 4,
        )

    if aoi is not None and aoi in frame.columns and plan["aoi_label_noise"] > 0:
        frame = inject_aoi_label_noise(
            frame,
            aoi,
            plan["aoi_label_noise"],
            seed=seed + 5,
        )

    if device_column is not None and device_column in frame.columns and plan["device_shift"] != 0:
        frame = inject_device_shift(
            frame,
            device_column,
            plan["device_shift"],
        )

    frame.attrs["eyeprocess_corruption_plan"] = dict(plan)
    return frame


def stress_test_process_pipeline(
    data,
    plans,
    analysis_fun,
    metric_fun=None,
    **corruption_kwargs,
):
    """Stress-test an analysis under explicit synthetic corruptions."""
    if not isinstance(plans, (list, tuple)) or not plans:
        raise EyeProcessValidationError("plans must be a non-empty list of corruption plans.")
    if not callable(analysis_fun):
        raise EyeProcessValidationError("analysis_fun must be a function.")

    metric = _default_sensitivity_extract if metric_fun is None else metric_fun
    if not callable(metric):
        raise EyeProcessValidationError("metric_fun must be a function.")

    rows = []
    for index, plan in enumerate(plans, start=1):
        corrupted = _capture(
            apply_synthetic_corruption,
            data,
            plan,
            **corruption_kwargs,
        )
        if corrupted["error"] is not None:
            rows.append(
                {
                    "plan_id": index,
                    "status": "corruption_error",
                    "error": corrupted["error"],
                }
            )
            continue

        analyzed = _capture(analysis_fun, corrupted["value"], plan)
        if analyzed["error"] is not None:
            rows.append(
                {
                    "plan_id": index,
                    "status": "analysis_error",
                    "error": analyzed["error"],
                }
            )
            continue

        measured = _capture(metric, analyzed["value"], plan)
        if measured["error"] is not None:
            rows.append(
                {
                    "plan_id": index,
                    "status": "metric_error",
                    "error": measured["error"],
                }
            )
            continue

        metrics = _as_frame(measured["value"], name="metric_fun output")
        if metrics.empty:
            rows.append(
                {
                    "plan_id": index,
                    "status": "empty_metric",
                    "error": None,
                }
            )
            continue

        metrics = metrics.copy()
        metrics["plan_id"] = index
        metrics["status"] = "success"
        metrics["missingness"] = plan["missingness"]
        metrics["pupil_dropout"] = plan["pupil_dropout"]
        metrics["gaze_offset"] = math.hypot(
            plan["gaze_offset_x"],
            plan["gaze_offset_y"],
        )
        metrics["sampling_jitter_sd"] = plan["sampling_jitter_sd"]
        metrics["aoi_label_noise"] = plan["aoi_label_noise"]
        metrics["device_shift"] = plan["device_shift"]
        metrics["trial_drop"] = plan["trial_drop"]
        rows.extend(metrics.to_dict(orient="records"))

    return {
        "plans": list(plans),
        "results": pd.DataFrame(rows),
        "baseline_hash": _stable_hash(data),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "caveat": (
            "Synthetic stress tests probe declared perturbations only and do not replace empirical external validation."
        ),
        "eyeprocess_class": "eye_process_stress_test",
    }


def stress_test_summary(
    x,
    metric="effect",
):
    """Summarise stress-test metrics."""
    if not _is_class(x, "eye_process_stress_test"):
        raise EyeProcessValidationError("x must be an eye_process_stress_test.")

    data = _as_frame(x["results"])
    _require_columns(data, [metric], name="x$results")
    values = pd.to_numeric(data[metric], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    return pd.DataFrame(
        [
            {
                "plans": len(data),
                "successful": int(np.sum(data["status"].astype(str) == "success")),
                "median": (float(np.median(finite)) if len(finite) else math.nan),
                "min": float(np.min(finite)) if len(finite) else math.nan,
                "max": float(np.max(finite)) if len(finite) else math.nan,
            }
        ]
    )


def stress_tolerance_frontier(
    x,
    severity,
    metric,
    acceptable,
):
    """Identify the empirical stress frontier for a metric."""
    if not callable(acceptable):
        raise EyeProcessValidationError("acceptable must be a function.")

    data = _as_frame(x["results"], name="x$results")
    _require_columns(data, [severity, metric], name="x$results")

    accepted = []
    for value in data[metric]:
        result = acceptable(value)
        if result is None or (isinstance(result, float) and np.isnan(result)):
            accepted.append(None)
        elif isinstance(result, (bool, np.bool_)):
            accepted.append(bool(result))
        else:
            raise EyeProcessValidationError("acceptable must return one TRUE/FALSE value per metric value.")

    severity_values = pd.to_numeric(
        data[severity],
        errors="coerce",
    ).to_numpy(dtype=float)

    known_true = [
        severity_values[index]
        for index, status in enumerate(accepted)
        if status is True and np.isfinite(severity_values[index])
    ]
    known_false = [
        severity_values[index]
        for index, status in enumerate(accepted)
        if status is False and np.isfinite(severity_values[index])
    ]

    return pd.DataFrame(
        [
            {
                "max_acceptable_severity": (float(max(known_true)) if known_true else math.nan),
                "first_unacceptable_severity": (float(min(known_false)) if known_false else math.nan),
            }
        ]
    )
