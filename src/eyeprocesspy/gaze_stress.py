"""Eye-tracking-specific simulation and measurement stress testing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _frame(value: Any) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)


def simulate_known_gaze_process(
    *,
    n_cycles: int = 4,
    sampling_hz: float = 300.0,
    fixation_ms: float = 300.0,
    saccade_ms: float = 30.0,
    pursuit_ms: float = 500.0,
    noise_sd_deg: float = 0.03,
    seed: int = 1,
) -> pd.DataFrame:
    """Generate labelled fixation-saccade-pursuit cycles for method validation."""
    n_cycles, seed = int(n_cycles), int(seed)
    if n_cycles < 1 or sampling_hz <= 0 or min(fixation_ms, saccade_ms, pursuit_ms) <= 0 or noise_sd_deg < 0 or seed < 0:
        raise EyeProcessValidationError("Simulation durations/rate must be positive; noise and seed must be non-negative.")
    rng = np.random.default_rng(seed)
    dt_ms = 1000.0 / float(sampling_hz)
    rows = []
    t = 0.0
    x, y = 0.0, 0.0
    for cycle in range(n_cycles):
        for label, duration in (("fixation", fixation_ms), ("saccade", saccade_ms), ("pursuit", pursuit_ms)):
            n = max(2, int(round(duration / dt_ms)))
            if label == "fixation":
                xs = np.full(n, x)
                ys = np.full(n, y)
            elif label == "saccade":
                target_x, target_y = x + 4.0, y + 1.0
                xs = np.linspace(x, target_x, n)
                ys = np.linspace(y, target_y, n)
                x, y = target_x, target_y
            else:
                target_x, target_y = x + 3.0, y
                xs = np.linspace(x, target_x, n)
                ys = np.linspace(y, target_y, n)
                x, y = target_x, target_y
            for xx, yy in zip(xs, ys, strict=True):
                rows.append({"timestamp_ms": t, "gaze_x_deg": xx + rng.normal(0, noise_sd_deg), "gaze_y_deg": yy + rng.normal(0, noise_sd_deg), "event_type": label, "cycle": cycle + 1})
                t += dt_ms
    out = pd.DataFrame(rows)
    out.attrs["eyeprocess_class"] = "eye_known_gaze_simulation"
    out.attrs["simulation"] = {"sampling_hz": float(sampling_hz), "seed": seed}
    return out


def inject_spatial_drift(
    data: Any,
    *,
    x: str = "gaze_x_deg",
    y: str = "gaze_y_deg",
    max_offset_x: float = 1.0,
    max_offset_y: float = 0.0,
) -> pd.DataFrame:
    """Inject a linear session drift while retaining original columns."""
    frame = _frame(data)
    if x not in frame or y not in frame:
        raise EyeProcessValidationError("Required gaze columns are missing.")
    progress = np.linspace(0.0, 1.0, len(frame))
    out = frame.copy()
    out[f"{x}_clean"] = pd.to_numeric(frame[x], errors="coerce")
    out[f"{y}_clean"] = pd.to_numeric(frame[y], errors="coerce")
    out[x] = out[f"{x}_clean"] + progress * float(max_offset_x)
    out[y] = out[f"{y}_clean"] + progress * float(max_offset_y)
    out.attrs["corruption"] = {"type": "spatial_drift", "max_offset_x": float(max_offset_x), "max_offset_y": float(max_offset_y)}
    return out


def inject_spatial_noise(
    data: Any,
    *,
    x: str = "gaze_x_deg",
    y: str = "gaze_y_deg",
    sd: float = 0.1,
    seed: int = 1,
) -> pd.DataFrame:
    """Inject zero-mean Gaussian spatial noise."""
    frame = _frame(data)
    if x not in frame or y not in frame or float(sd) < 0 or int(seed) < 0:
        raise EyeProcessValidationError("Gaze columns must exist; sd and seed must be non-negative.")
    rng = np.random.default_rng(int(seed))
    out = frame.copy()
    out[x] = pd.to_numeric(out[x], errors="coerce") + rng.normal(0, float(sd), len(out))
    out[y] = pd.to_numeric(out[y], errors="coerce") + rng.normal(0, float(sd), len(out))
    out.attrs["corruption"] = {"type": "spatial_noise", "sd": float(sd), "seed": int(seed)}
    return out


def inject_blink_gaps(
    data: Any,
    *,
    columns: Sequence[str] = ("gaze_x_deg", "gaze_y_deg"),
    fraction: float = 0.05,
    mean_gap_samples: int = 10,
    seed: int = 1,
) -> pd.DataFrame:
    """Inject contiguous missing gaps rather than independent random missingness."""
    frame = _frame(data)
    missing = [c for c in columns if c not in frame]
    fraction, mean_gap_samples, seed = float(fraction), int(mean_gap_samples), int(seed)
    if missing or not 0 <= fraction < 1 or mean_gap_samples < 1 or seed < 0:
        raise EyeProcessValidationError("Invalid columns, fraction, gap length, or seed.")
    rng = np.random.default_rng(seed)
    target = int(round(len(frame) * fraction))
    mask = np.zeros(len(frame), bool)
    attempts = 0
    while mask.sum() < target and attempts < max(20, len(frame) * 2):
        start = int(rng.integers(0, max(len(frame), 1)))
        gap = max(1, int(rng.poisson(mean_gap_samples)))
        mask[start : min(len(frame), start + gap)] = True
        attempts += 1
    out = frame.copy()
    out.loc[mask, list(columns)] = np.nan
    out["synthetic_blink_gap"] = mask
    out.attrs["corruption"] = {"type": "blink_gaps", "requested_fraction": fraction, "realized_fraction": float(mask.mean()) if len(mask) else 0.0, "seed": seed}
    return out


def inject_event_misclassification(
    data: Any,
    *,
    event_type: str = "event_type",
    probability: float = 0.05,
    seed: int = 1,
) -> pd.DataFrame:
    """Perturb event labels while preserving the clean label for recovery analyses."""
    frame = _frame(data)
    probability, seed = float(probability), int(seed)
    if event_type not in frame or not 0 <= probability < 1 or seed < 0:
        raise EyeProcessValidationError("event_type must exist; probability in [0,1), seed non-negative.")
    rng = np.random.default_rng(seed)
    labels = frame[event_type].astype(str)
    states = sorted(labels.dropna().unique())
    if len(states) < 2:
        raise EyeProcessValidationError("At least two event states are required for misclassification.")
    change = rng.random(len(frame)) < probability
    new = labels.copy()
    for i in np.flatnonzero(change):
        alternatives = [state for state in states if state != labels.iloc[i]]
        new.iloc[i] = alternatives[int(rng.integers(0, len(alternatives)))]
    out = frame.copy()
    out[f"{event_type}_clean"] = labels
    out[event_type] = new
    out.attrs["corruption"] = {"type": "event_misclassification", "probability": probability, "seed": seed}
    return out


def run_gaze_stress_suite(
    data: Any,
    evaluator: Callable[[pd.DataFrame], Mapping[str, float]],
    *,
    corruption: str,
    severities: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    seed: int = 1,
) -> EyeResult:
    """Quantify downstream metric degradation across an explicit corruption curve."""
    if not callable(evaluator):
        raise EyeProcessValidationError("evaluator must be callable.")
    frame = _frame(data)
    rows = []
    for severity in severities:
        severity = float(severity)
        try:
            if corruption == "drift":
                stressed = inject_spatial_drift(frame, max_offset_x=severity)
            elif corruption == "noise":
                stressed = inject_spatial_noise(frame, sd=severity, seed=seed)
            elif corruption == "missingness":
                stressed = inject_blink_gaps(frame, fraction=min(severity, 0.95), seed=seed)
            elif corruption == "event_misclassification":
                stressed = inject_event_misclassification(frame, probability=min(severity, 0.95), seed=seed)
            else:
                raise EyeProcessValidationError("corruption must be drift, noise, missingness, or event_misclassification.")
            metrics = dict(evaluator(stressed))
            rows.append({"severity": severity, **{str(k): float(v) for k, v in metrics.items()}, "status": "success", "error": None})
        except Exception as exc:
            rows.append({"severity": severity, "status": "failed", "error": str(exc)})
    return EyeResult(
        {"table": pd.DataFrame(rows), "corruption": corruption, "seed": int(seed)},
        eyeprocess_class="eye_gaze_stress_suite",
    )


def plot_gaze_stress(result: Any, *, metric: str, ax: Any = None) -> Any:
    """Plot a downstream metric against corruption severity."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(result, "eyeprocess_class", None) != "eye_gaze_stress_suite" or metric not in result["table"]:
        raise EyeProcessValidationError("Provide a gaze stress result and an available metric.")
    table = result["table"]
    axis = plt.subplots()[1] if ax is None else ax
    axis.plot(table.severity, pd.to_numeric(table[metric], errors="coerce"), marker="o")
    axis.set_xlabel("Corruption severity")
    axis.set_ylabel(metric)
    axis.set_title(f"Gaze stress: {result['corruption']}")
    axis.eyeprocess_plot_data = table.copy()
    return axis


__all__ = [
    "inject_blink_gaps",
    "inject_event_misclassification",
    "inject_spatial_drift",
    "inject_spatial_noise",
    "plot_gaze_stress",
    "run_gaze_stress_suite",
    "simulate_known_gaze_process",
]
