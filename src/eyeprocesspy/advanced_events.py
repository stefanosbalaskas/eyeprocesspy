"""Transparent smooth-pursuit and microsaccade event analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _frame(value: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _require(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in data]
    if missing:
        raise EyeProcessValidationError(f"Missing required column(s): {', '.join(missing)}.")


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


def _time_velocity(data: pd.DataFrame, time: str, x: str, y: str, time_unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t, gx, gy = _num(data[time]), _num(data[x]), _num(data[y])
    scale = {"s": 1.0, "ms": 1000.0}.get(time_unit)
    if scale is None:
        raise EyeProcessValidationError("time_unit must be 's' or 'ms'.")
    ok_t = t[np.isfinite(t)]
    if len(ok_t) > 1 and np.any(np.diff(ok_t) <= 0):
        raise EyeProcessValidationError("Timestamps must be strictly increasing.")
    ts = t / scale
    vx = np.gradient(gx, ts) if len(data) > 1 else np.full(len(data), np.nan)
    vy = np.gradient(gy, ts) if len(data) > 1 else np.full(len(data), np.nan)
    hz = float(1 / np.median(np.diff(ts))) if len(ts) > 1 else np.nan
    return vx, vy, np.hypot(vx, vy), hz


def _episodes(
    labels: np.ndarray,
    t: np.ndarray,
    speed: np.ndarray,
    minimum_duration_ms: float,
    duration_scale_to_ms: float = 1.0,
) -> pd.DataFrame:
    rows = []
    start = 0
    event_id = 0
    for i in range(1, len(labels) + 1):
        if i < len(labels) and labels[i] == labels[start]:
            continue
        label = str(labels[start])
        if label != "unclassified":
            duration = (
                0.0
                if i - start < 2
                else float(t[i - 1] - t[start]) * float(duration_scale_to_ms)
            )
            if duration >= minimum_duration_ms:
                event_id += 1
                rows.append(
                    {
                        "event_id": event_id,
                        "event_type": label,
                        "start_time": float(t[start]),
                        "end_time": float(t[i - 1]),
                        "duration_ms": duration,
                        "mean_velocity": float(np.nanmean(speed[start:i])),
                        "peak_velocity": float(np.nanmax(speed[start:i])),
                    }
                )
        start = i
    return pd.DataFrame(rows)


def detect_events_ivvt(
    data: Any,
    *,
    time: str = "timestamp_ms",
    x: str = "gaze_x_deg",
    y: str = "gaze_y_deg",
    time_unit: str = "ms",
    fixation_velocity_threshold: float = 5.0,
    saccade_velocity_threshold: float = 30.0,
    minimum_duration_ms: float = 20.0,
    minimum_sampling_hz: float = 30.0,
) -> EyeResult:
    """Classify fixation, pursuit, and saccade samples with explicit velocity thresholds."""
    frame = _frame(data)
    _require(frame, [time, x, y])
    fthr, sthr = float(fixation_velocity_threshold), float(saccade_velocity_threshold)
    if not 0 <= fthr < sthr:
        raise EyeProcessValidationError("Require 0 <= fixation threshold < saccade threshold.")
    vx, vy, speed, hz = _time_velocity(frame, time, x, y, time_unit)
    if not np.isfinite(hz) or hz < float(minimum_sampling_hz):
        raise EyeProcessValidationError("Sampling rate is below the declared pursuit-analysis minimum.")
    labels = np.full(len(frame), "unclassified", dtype=object)
    finite = np.isfinite(speed)
    labels[finite & (speed <= fthr)] = "fixation"
    labels[finite & (speed > fthr) & (speed < sthr)] = "pursuit"
    labels[finite & (speed >= sthr)] = "saccade"
    sample_table = frame.copy()
    sample_table["velocity_x"] = vx
    sample_table["velocity_y"] = vy
    sample_table["velocity"] = speed
    sample_table["event_type"] = labels
    duration_scale = 1000.0 if time_unit == "s" else 1.0
    events = _episodes(
        labels,
        _num(frame[time]),
        speed,
        float(minimum_duration_ms),
        duration_scale_to_ms=duration_scale,
    )
    return EyeResult(
        {"samples": sample_table, "events": events, "sampling_hz": hz, "method": "ivvt", "time_unit": time_unit},
        eyeprocess_class="eye_ivvt_events",
    )


def detect_events_directional(
    data: Any,
    *,
    time: str = "timestamp_ms",
    x: str = "gaze_x_deg",
    y: str = "gaze_y_deg",
    time_unit: str = "ms",
    minimum_velocity: float = 2.0,
    maximum_velocity: float = 30.0,
    maximum_direction_change_deg: float = 35.0,
    minimum_duration_ms: float = 40.0,
    minimum_sampling_hz: float = 30.0,
) -> EyeResult:
    """Detect directionally coherent moderate-speed smooth pursuit."""
    frame = _frame(data)
    _require(frame, [time, x, y])
    lo, hi = float(minimum_velocity), float(maximum_velocity)
    if not 0 <= lo < hi:
        raise EyeProcessValidationError("Require 0 <= minimum_velocity < maximum_velocity.")
    vx, vy, speed, hz = _time_velocity(frame, time, x, y, time_unit)
    if not np.isfinite(hz) or hz < float(minimum_sampling_hz):
        raise EyeProcessValidationError("Sampling rate is below the declared pursuit-analysis minimum.")
    angle = np.degrees(np.unwrap(np.arctan2(vy, vx)))
    change = np.r_[np.nan, np.abs(np.diff(angle))]
    pursuit = (
        np.isfinite(speed)
        & (speed >= lo)
        & (speed <= hi)
        & (np.isnan(change) | (change <= float(maximum_direction_change_deg)))
    )
    labels = np.where(pursuit, "pursuit", "unclassified").astype(object)
    sample_table = frame.copy()
    sample_table["velocity_x"] = vx
    sample_table["velocity_y"] = vy
    sample_table["velocity"] = speed
    sample_table["direction_change_deg"] = change
    sample_table["event_type"] = labels
    duration_scale = 1000.0 if time_unit == "s" else 1.0
    events = _episodes(
        labels,
        _num(frame[time]),
        speed,
        float(minimum_duration_ms),
        duration_scale_to_ms=duration_scale,
    )
    return EyeResult(
        {"samples": sample_table, "events": events, "sampling_hz": hz, "method": "directional", "time_unit": time_unit},
        eyeprocess_class="eye_directional_pursuit_events",
    )


def detect_smooth_pursuits(data: Any, *, method: str = "directional", **kwargs: Any) -> EyeResult:
    """Dispatch to a transparent pursuit detector."""
    method = str(method).lower()
    if method == "directional":
        return detect_events_directional(data, **kwargs)
    if method == "ivvt":
        return detect_events_ivvt(data, **kwargs)
    raise EyeProcessValidationError("method must be 'directional' or 'ivvt'.")


def compute_pursuit_gain(
    data: Any,
    *,
    time: str = "timestamp_ms",
    gaze_x: str = "gaze_x_deg",
    gaze_y: str = "gaze_y_deg",
    target_x: str = "target_x_deg",
    target_y: str = "target_y_deg",
    time_unit: str = "ms",
) -> pd.DataFrame:
    """Compute sample-wise gaze/target speed ratio on synchronized trajectories."""
    frame = _frame(data)
    _require(frame, [time, gaze_x, gaze_y, target_x, target_y])
    _, _, gs, _ = _time_velocity(frame, time, gaze_x, gaze_y, time_unit)
    _, _, ts, _ = _time_velocity(frame, time, target_x, target_y, time_unit)
    gain = np.divide(gs, ts, out=np.full_like(gs, np.nan), where=np.isfinite(ts) & (ts > 0))
    return pd.DataFrame({"timestamp": frame[time].to_numpy(), "gaze_speed": gs, "target_speed": ts, "pursuit_gain": gain})


def compute_pursuit_velocity_error(data: Any, **kwargs: Any) -> pd.DataFrame:
    """Compute gaze minus target velocity error on synchronized trajectories."""
    gain = compute_pursuit_gain(data, **kwargs)
    gain["velocity_error"] = gain.gaze_speed - gain.target_speed
    gain["absolute_velocity_error"] = np.abs(gain.velocity_error)
    return gain


def summarise_pursuits(result: Any) -> pd.DataFrame:
    """Summarize pursuit episodes from IVVT or directional output."""
    if getattr(result, "eyeprocess_class", None) not in {"eye_ivvt_events", "eye_directional_pursuit_events"}:
        raise EyeProcessValidationError("result must be a pursuit detector result.")
    events = result["events"]
    pursuits = events.loc[events.event_type.eq("pursuit")] if not events.empty else events
    if pursuits.empty:
        return pd.DataFrame([{"n_pursuits": 0, "total_duration_ms": 0.0, "mean_duration_ms": np.nan}])
    return pd.DataFrame(
        [{
            "n_pursuits": int(len(pursuits)),
            "total_duration_ms": float(pursuits.duration_ms.sum()),
            "mean_duration_ms": float(pursuits.duration_ms.mean()),
            "mean_velocity": float(pursuits.mean_velocity.mean()),
        }]
    )


def validate_pursuit_detection(result: Any) -> pd.DataFrame:
    """Return review diagnostics rather than declaring detector validity."""
    summary = summarise_pursuits(result)
    summary["sampling_hz"] = float(result["sampling_hz"])
    summary["method"] = str(result["method"])
    summary["status"] = np.where(summary.n_pursuits.gt(0), "events_detected", "review_no_events")
    return summary


def _robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    estimate = np.sqrt(max(float(np.median(values**2) - np.median(values) ** 2), 0.0))
    if estimate > 0:
        return estimate
    return float(np.std(values, ddof=1))


def detect_microsaccades(
    data: Any,
    *,
    time: str = "timestamp_ms",
    x: str = "gaze_x_deg",
    y: str = "gaze_y_deg",
    time_unit: str = "ms",
    lambda_threshold: float = 6.0,
    minimum_duration_ms: float = 6.0,
    maximum_amplitude_deg: float = 2.0,
    minimum_sampling_hz: float = 200.0,
    right_x: str | None = None,
    right_y: str | None = None,
    binocular_tolerance_ms: float = 10.0,
) -> EyeResult:
    """Detect microsaccades using robust velocity ellipses and optional binocular overlap."""
    frame = _frame(data)
    if (right_x is None) != (right_y is None):
        raise EyeProcessValidationError("right_x and right_y must be supplied together.")
    required = [time, x, y] + ([] if right_x is None else [right_x, right_y])
    _require(frame, [c for c in required if c is not None])
    if float(lambda_threshold) <= 0 or float(minimum_duration_ms) < 0 or float(maximum_amplitude_deg) <= 0:
        raise EyeProcessValidationError("Microsaccade thresholds must be scientifically meaningful positive values.")

    vx, vy, speed, hz = _time_velocity(frame, time, x, y, time_unit)
    if not np.isfinite(hz) or hz < float(minimum_sampling_hz):
        raise EyeProcessValidationError("Sampling rate is below the declared microsaccade-analysis minimum.")
    sx, sy = _robust_sigma(vx), _robust_sigma(vy)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
        raise EyeProcessValidationError("Velocity dispersion is insufficient for robust microsaccade thresholds.")
    candidate = (vx / (float(lambda_threshold) * sx)) ** 2 + (vy / (float(lambda_threshold) * sy)) ** 2 > 1
    labels = np.where(candidate, "microsaccade", "unclassified").astype(object)
    duration_scale = 1000.0 if time_unit == "s" else 1.0
    events = _episodes(
        labels,
        _num(frame[time]),
        speed,
        float(minimum_duration_ms),
        duration_scale_to_ms=duration_scale,
    )
    gx, gy = _num(frame[x]), _num(frame[y])
    amplitudes = []
    for _, event in events.iterrows():
        mask = (_num(frame[time]) >= event.start_time) & (_num(frame[time]) <= event.end_time)
        idx = np.flatnonzero(mask)
        amplitudes.append(float(np.hypot(gx[idx[-1]] - gx[idx[0]], gy[idx[-1]] - gy[idx[0]])) if len(idx) else np.nan)
    if len(events):
        events["amplitude_deg"] = amplitudes
        events = events.loc[events.amplitude_deg.le(float(maximum_amplitude_deg))].reset_index(drop=True)
    if right_x is not None and right_y is not None:
        right = detect_microsaccades(
            frame,
            time=time,
            x=right_x,
            y=right_y,
            time_unit=time_unit,
            lambda_threshold=lambda_threshold,
            minimum_duration_ms=minimum_duration_ms,
            maximum_amplitude_deg=maximum_amplitude_deg,
            minimum_sampling_hz=minimum_sampling_hz,
        )
        right_events = right["events"]
        keep = []
        for _, event in events.iterrows():
            if right_events.empty:
                keep.append(False)
                continue
            separation = np.maximum(
                np.maximum(right_events.start_time.to_numpy(float) - float(event.end_time), float(event.start_time) - right_events.end_time.to_numpy(float)),
                0.0,
            )
            keep.append(
                bool(
                    np.min(separation) * duration_scale
                    <= float(binocular_tolerance_ms)
                )
            )
        events = events.loc[keep].reset_index(drop=True)
    return EyeResult(
        {
            "events": events,
            "sampling_hz": hz,
            "threshold_x": float(lambda_threshold) * sx,
            "threshold_y": float(lambda_threshold) * sy,
            "binocular": right_x is not None,
        },
        eyeprocess_class="eye_microsaccades",
    )


def summarise_microsaccades(result: Any) -> pd.DataFrame:
    """Summarize microsaccade count, duration, amplitude, and peak velocity."""
    if getattr(result, "eyeprocess_class", None) != "eye_microsaccades":
        raise EyeProcessValidationError("result must be an eye_microsaccades result.")
    events = result["events"]
    if events.empty:
        return pd.DataFrame([{"n_microsaccades": 0, "mean_duration_ms": np.nan, "mean_amplitude_deg": np.nan, "mean_peak_velocity": np.nan}])
    return pd.DataFrame([{
        "n_microsaccades": int(len(events)),
        "mean_duration_ms": float(events.duration_ms.mean()),
        "mean_amplitude_deg": float(events.amplitude_deg.mean()),
        "mean_peak_velocity": float(events.peak_velocity.mean()),
    }])


def microsaccade_main_sequence(result: Any) -> pd.DataFrame:
    """Return amplitude and peak velocity for main-sequence inspection."""
    if getattr(result, "eyeprocess_class", None) != "eye_microsaccades":
        raise EyeProcessValidationError("result must be an eye_microsaccades result.")
    events = result["events"]
    return events[["event_id", "amplitude_deg", "peak_velocity"]].copy() if not events.empty else pd.DataFrame(columns=["event_id", "amplitude_deg", "peak_velocity"])


def plot_pursuit_velocity(result: Any, ax: Any = None) -> Any:
    """Plot sample velocity and pursuit-labelled samples."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(result, "eyeprocess_class", None) not in {"eye_ivvt_events", "eye_directional_pursuit_events"}:
        raise EyeProcessValidationError("result must be a pursuit detector result.")
    data = result["samples"]
    axis = plt.subplots()[1] if ax is None else ax
    axis.plot(np.arange(len(data)), data.velocity)
    pursuit = data.event_type.eq("pursuit")
    axis.scatter(np.flatnonzero(pursuit), data.loc[pursuit, "velocity"])
    axis.set_title("Smooth-pursuit classification")
    axis.set_ylabel("Velocity")
    axis.eyeprocess_plot_data = data.copy()
    return axis


def plot_microsaccade_main_sequence(result: Any, ax: Any = None) -> Any:
    """Plot microsaccade amplitude against peak velocity."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    table = microsaccade_main_sequence(result)
    axis = plt.subplots()[1] if ax is None else ax
    axis.scatter(table.amplitude_deg, table.peak_velocity)
    axis.set_xlabel("Amplitude (deg)")
    axis.set_ylabel("Peak velocity (deg/s)")
    axis.set_title("Microsaccade main sequence")
    axis.eyeprocess_plot_data = table.copy()
    return axis


__all__ = [
    "compute_pursuit_gain",
    "compute_pursuit_velocity_error",
    "detect_events_directional",
    "detect_events_ivvt",
    "detect_microsaccades",
    "detect_smooth_pursuits",
    "microsaccade_main_sequence",
    "plot_microsaccade_main_sequence",
    "plot_pursuit_velocity",
    "summarise_microsaccades",
    "summarise_pursuits",
    "validate_pursuit_detection",
]
