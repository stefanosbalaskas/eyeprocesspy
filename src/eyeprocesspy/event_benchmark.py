"""Ground-truth-aware event-detector benchmarking utilities."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _prepare(value: Any, name: str, start: str, end: str, event_type: str, group: str | None) -> pd.DataFrame:
    data = _frame(value, name)
    required = [start, end, event_type] + ([] if group is None else [group])
    missing = [c for c in required if c not in data]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required column(s): {', '.join(missing)}.")
    data[start] = pd.to_numeric(data[start], errors="coerce")
    data[end] = pd.to_numeric(data[end], errors="coerce")
    if not np.isfinite(data[[start, end]].to_numpy(float)).all() or (data[end] < data[start]).any():
        raise EyeProcessValidationError(f"{name} has invalid event intervals.")
    if data[event_type].isna().any():
        raise EyeProcessValidationError(f"{name} contains missing event labels.")
    if group is not None and data[group].isna().any():
        raise EyeProcessValidationError(f"{name} contains missing group identifiers.")
    return data.reset_index(drop=True)


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap / union if union > 0 else float(a0 == b0 and a1 == b1)


def _match(
    truth: pd.DataFrame,
    detected: pd.DataFrame,
    start: str,
    end: str,
    event_type: str,
    group: str | None,
    minimum_iou: float,
) -> pd.DataFrame:
    candidates = []
    for ti, tr in truth.iterrows():
        for di, dr in detected.iterrows():
            if str(tr[event_type]) != str(dr[event_type]):
                continue
            if group is not None and tr[group] != dr[group]:
                continue
            score = _iou(float(tr[start]), float(tr[end]), float(dr[start]), float(dr[end]))
            if score >= minimum_iou:
                candidates.append((score, ti, di))
    used_t, used_d, rows = set(), set(), []
    for score, ti, di in sorted(candidates, reverse=True):
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        tr, dr = truth.iloc[ti], detected.iloc[di]
        rows.append({
            "truth_index": ti,
            "detected_index": di,
            "event_type": tr[event_type],
            "iou": score,
            "onset_error": float(dr[start] - tr[start]),
            "offset_error": float(dr[end] - tr[end]),
        })
    return pd.DataFrame(rows)


def benchmark_event_detector(
    truth: Any,
    detected: Any,
    *,
    start: str = "start_time",
    end: str = "end_time",
    event_type: str = "event_type",
    group: str | None = None,
    minimum_iou: float = 0.5,
) -> EyeResult:
    """Benchmark detected events against an explicit annotated event catalogue."""
    if not 0 <= float(minimum_iou) <= 1:
        raise EyeProcessValidationError("minimum_iou must lie in [0, 1].")
    tr = _prepare(truth, "truth", start, end, event_type, group)
    de = _prepare(detected, "detected", start, end, event_type, group)
    matches = _match(tr, de, start, end, event_type, group, float(minimum_iou))
    tp = len(matches)
    fp, fn = len(de) - tp, len(tr) - tp
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
    summary = pd.DataFrame([{
        "truth_events": len(tr),
        "detected_events": len(de),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(matches.iou.mean()) if tp else np.nan,
        "mean_abs_onset_error": float(matches.onset_error.abs().mean()) if tp else np.nan,
        "mean_abs_offset_error": float(matches.offset_error.abs().mean()) if tp else np.nan,
    }])
    return EyeResult(
        {"summary": summary, "matches": matches, "truth": tr, "detected": de, "minimum_iou": float(minimum_iou)},
        eyeprocess_class="eye_event_detector_benchmark",
    )


def event_boundary_error(benchmark: Any) -> pd.DataFrame:
    """Return signed and absolute onset/offset errors for matched events."""
    if getattr(benchmark, "eyeprocess_class", None) != "eye_event_detector_benchmark":
        raise EyeProcessValidationError("benchmark must be an eye_event_detector_benchmark.")
    out = benchmark["matches"].copy()
    if out.empty:
        return pd.DataFrame(columns=["truth_index", "detected_index", "event_type", "onset_error", "offset_error", "abs_onset_error", "abs_offset_error"])
    out["abs_onset_error"] = out.onset_error.abs()
    out["abs_offset_error"] = out.offset_error.abs()
    return out


def event_confusion_matrix(
    truth: Any,
    detected: Any,
    *,
    start: str = "start_time",
    end: str = "end_time",
    event_type: str = "event_type",
    sample_step: float = 1.0,
    none_label: str = "none",
) -> pd.DataFrame:
    """Compute a time-sampled event-label confusion matrix at an explicit resolution."""
    tr = _prepare(truth, "truth", start, end, event_type, None)
    de = _prepare(detected, "detected", start, end, event_type, None)
    sample_step = float(sample_step)
    if not np.isfinite(sample_step) or sample_step <= 0:
        raise EyeProcessValidationError("sample_step must be finite and positive.")
    if tr.empty and de.empty:
        return pd.DataFrame()
    starts = [d[start].min() for d in (tr, de) if not d.empty]
    ends = [d[end].max() for d in (tr, de) if not d.empty]
    grid = np.arange(float(min(starts)), float(max(ends)) + sample_step / 2, sample_step)
    def labels(events: pd.DataFrame) -> list[str]:
        out = []
        for t in grid:
            active = events.loc[(events[start] <= t) & (events[end] >= t), event_type]
            states = list(dict.fromkeys(active.astype(str)))
            if not states:
                out.append(none_label)
            elif len(states) == 1:
                out.append(states[0])
            else:
                out.append("overlap")
        return out
    return pd.crosstab(pd.Series(labels(tr), name="truth"), pd.Series(labels(de), name="detected"), dropna=False)


def compare_event_detectors(
    truth: Any,
    detectors: Mapping[str, Any],
    **kwargs: Any,
) -> pd.DataFrame:
    """Compare named detector catalogues against the same truth annotations."""
    if not detectors:
        raise EyeProcessValidationError("detectors must contain at least one named catalogue.")
    rows = []
    for name, detected in detectors.items():
        row = benchmark_event_detector(truth, detected, **kwargs)["summary"].iloc[0].to_dict()
        row["detector"] = str(name)
        rows.append(row)
    return pd.DataFrame(rows)


def detector_parameter_sensitivity(
    data: Any,
    truth: Any,
    detector: Callable[..., Any],
    parameter_grid: Mapping[str, Sequence[Any]],
    *,
    benchmark_kwargs: Mapping[str, Any] | None = None,
) -> EyeResult:
    """Run a declared detector parameter grid and keep failed branches visible."""
    if not callable(detector):
        raise EyeProcessValidationError("detector must be callable.")
    if not parameter_grid:
        raise EyeProcessValidationError("parameter_grid must not be empty.")
    keys = list(parameter_grid)
    rows = []
    for values in itertools.product(*(parameter_grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        try:
            result = detector(data, **params)
            events = result["events"] if isinstance(result, Mapping) and "events" in result else result
            summary = benchmark_event_detector(truth, events, **dict(benchmark_kwargs or {}))["summary"].iloc[0].to_dict()
            rows.append({**params, **summary, "status": "success", "error": None})
        except Exception as exc:
            rows.append({**params, "status": "failed", "error": str(exc)})
    table = pd.DataFrame(rows)
    return EyeResult(
        {
            "table": table,
            "planned_branches": len(table),
            "successful_branches": int(table.status.eq("success").sum()),
            "failed_branches": int(table.status.eq("failed").sum()),
        },
        eyeprocess_class="eye_detector_parameter_sensitivity",
    )


def plot_detector_benchmark(comparison: Any, *, metric: str = "f1", ax: Any = None) -> Any:
    """Plot one benchmark metric across detectors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    table = _frame(comparison, "comparison")
    if "detector" not in table or metric not in table:
        raise EyeProcessValidationError("comparison must contain detector and requested metric.")
    axis = plt.subplots()[1] if ax is None else ax
    axis.bar(table.detector.astype(str), pd.to_numeric(table[metric], errors="coerce"))
    axis.set_ylabel(metric)
    axis.set_title("Event-detector benchmark")
    setattr(axis, "eyeprocess_plot_data", table.copy())
    return axis


__all__ = [
    "benchmark_event_detector",
    "compare_event_detectors",
    "detector_parameter_sensitivity",
    "event_boundary_error",
    "event_confusion_matrix",
    "plot_detector_benchmark",
]
