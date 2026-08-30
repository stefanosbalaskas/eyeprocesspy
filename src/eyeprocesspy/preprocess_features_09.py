from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .dataset import (
    EyeDataset,
    _assert_eye_dataset,
    _next_id,
    _now_utc,
    add_provenance,
    is_eye_dataset,
)
from .exceptions import EyeProcessValidationError
from .schema import empty_eye_table, standardize_eye_table


def _finite(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=True)


def _mode_value(values: Any):
    series = pd.Series(values).dropna()
    if series.empty:
        return pd.NA
    counts = series.value_counts(sort=False)
    return counts.index[int(np.argmax(counts.to_numpy()))]


def _first_nonmissing(values: Any, default=pd.NA):
    for value in pd.Series(values).tolist():
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _safe_span(values: Any) -> float:
    arr = _finite(values)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr) - np.min(arr)) if arr.size else np.nan


def _safe_max(values: Any) -> float:
    arr = _finite(values)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


def _mad(values: Any) -> float:
    arr = _finite(values)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan
    center = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - center)))


def _trapz(x: Any, y: Any) -> float:
    xx = _finite(x)
    yy = _finite(y)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if xx.size < 2:
        return 0.0 if xx.size == 1 else np.nan
    order = np.argsort(xx, kind="stable")
    return float(np.trapezoid(yy[order], xx[order]))


def _group_frames(data: pd.DataFrame, keys: Sequence[str], *, dropna: bool = True) -> list[pd.DataFrame]:
    if data.empty:
        return []
    keys = [key for key in keys if key in data.columns]
    if not keys:
        return [data.copy()]
    if dropna and any(not data[key].notna().any() for key in keys):
        return []
    return [group.copy() for _, group in data.groupby(keys, sort=False, dropna=dropna, observed=True)]


# R/009-preprocessing.R -----------------------------------------------------


@dataclass(slots=True)
class EyePreprocessSpec:
    gaze_filter: str = "none"
    gaze_window: int = 5
    pupil_interpolation: str = "linear"
    pupil_max_gap_ms: float = 150.0
    pupil_filter: str = "median"
    pupil_window: int = 5
    pupil_baseline: str = "subtract"
    pupil_baseline_window: tuple[float, float] = (-0.2, 0.0)
    fixation_algorithm: str = "none"
    fixation_parameters: dict[str, Any] = field(default_factory=dict)
    blink_detection: bool = True
    exclusions: dict[str, Any] = field(default_factory=dict)


def preprocess_spec(
    gaze_filter="none",
    gaze_window=5,
    pupil_interpolation="linear",
    pupil_max_gap_ms=150,
    pupil_filter="median",
    pupil_window=5,
    pupil_baseline="subtract",
    pupil_baseline_window=(-0.2, 0),
    fixation_algorithm="none",
    fixation_parameters=None,
    blink_detection=True,
    exclusions=None,
):
    """Create the frozen-R preprocessing specification."""
    return EyePreprocessSpec(
        gaze_filter=str(gaze_filter),
        gaze_window=int(gaze_window),
        pupil_interpolation=str(pupil_interpolation),
        pupil_max_gap_ms=float(pupil_max_gap_ms),
        pupil_filter=str(pupil_filter),
        pupil_window=int(pupil_window),
        pupil_baseline=str(pupil_baseline),
        pupil_baseline_window=tuple(float(v) for v in pupil_baseline_window),
        fixation_algorithm=str(fixation_algorithm),
        fixation_parameters=dict(fixation_parameters or {}),
        blink_detection=bool(blink_detection),
        exclusions=dict(exclusions or {}),
    )


def rolling_apply(x, width=5, FUN=None, na_rm=True):
    """Centered rolling apply matching R ``rolling_apply()``."""
    width = int(width)
    if width < 1:
        raise EyeProcessValidationError("`width` must be positive.")
    if width % 2 == 0:
        width += 1
    values = _finite(x)
    half = width // 2
    out = np.full(values.size, np.nan, dtype=float)
    if FUN is None:
        FUN = np.nanmedian
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        chunk = values[lo:hi]
        if na_rm:
            finite = chunk[np.isfinite(chunk)]
            out[i] = float(FUN(finite)) if finite.size else np.nan
        else:
            out[i] = float(FUN(chunk))
    return out


def filter_gaze(
    x,
    method=("median", "mean", "moving_median", "moving_average", "none"),
    window=5,
    component="gaze_samples",
):
    """Filter gaze coordinates within recording using the frozen-R contract."""
    _assert_eye_dataset(x)
    if isinstance(method, (list, tuple)):
        method = method[0]
    method = str(method)
    aliases = {"moving_median": "median", "moving_average": "mean"}
    method = aliases.get(method, method)
    if method not in {"median", "mean", "none"}:
        raise EyeProcessValidationError("Invalid gaze filter method.")
    if component not in x:
        raise EyeProcessValidationError(f"Unknown component `{component}`.")
    if method == "none" or x[component].empty:
        return x
    d = x[component].copy()
    required = {"recording_id", "timestamp_seconds", "gaze_x", "gaze_y"}
    missing = required - set(d.columns)
    if missing:
        raise EyeProcessValidationError(f"Missing gaze field(s): {', '.join(sorted(missing))}.")
    for _, idx in d.groupby("recording_id", sort=False, dropna=False).groups.items():
        idx = list(idx)
        order = d.loc[idx, "timestamp_seconds"].astype(float).sort_values(kind="stable").index
        fun = np.nanmedian if method == "median" else np.nanmean
        d.loc[order, "gaze_x"] = rolling_apply(d.loc[order, "gaze_x"], window, fun)
        d.loc[order, "gaze_y"] = rolling_apply(d.loc[order, "gaze_y"], window, fun)
    out = x.copy()
    out[component] = d
    return add_provenance(
        out,
        "filter_gaze",
        component,
        f"method={method};window={int(window)}",
        reversible=False,
    )


def gaze_velocity(data):
    """Return per-sample displacement and velocity within recording."""
    if is_eye_dataset(data):
        data = data["gaze_samples"]
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame or EyeDataset.")
    required = {"recording_id", "sample_id", "timestamp_seconds", "gaze_x", "gaze_y"}
    missing = required - set(data.columns)
    if missing:
        raise EyeProcessValidationError(f"Missing gaze field(s): {', '.join(sorted(missing))}.")
    rows = []
    for recording_id, z in data.groupby("recording_id", sort=False, dropna=False):
        z = z.sort_values("timestamp_seconds", kind="stable")
        t = _finite(z["timestamp_seconds"])
        gx = _finite(z["gaze_x"])
        gy = _finite(z["gaze_y"])
        dt = np.r_[np.nan, np.diff(t)]
        dx = np.r_[np.nan, np.diff(gx)]
        dy = np.r_[np.nan, np.diff(gy)]
        distance = np.sqrt(dx**2 + dy**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = distance / dt
        rows.append(
            pd.DataFrame(
                {
                    "recording_id": recording_id,
                    "sample_id": z["sample_id"].to_numpy(),
                    "timestamp_seconds": t,
                    "dx": dx,
                    "dy": dy,
                    "distance": distance,
                    "dt": dt,
                    "velocity": velocity,
                }
            )
        )
    columns = [
        "recording_id",
        "sample_id",
        "timestamp_seconds",
        "dx",
        "dy",
        "distance",
        "dt",
        "velocity",
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def flag_gaze_outliers(x, method=("mad", "velocity", "bounds"), threshold=6, max_velocity=None):
    """Flag gaze outliers using MAD, velocity, or coordinate-space bounds."""
    _assert_eye_dataset(x)
    if isinstance(method, (list, tuple)):
        method = method[0]
    if method not in {"mad", "velocity", "bounds"}:
        raise EyeProcessValidationError("Invalid gaze-outlier method.")
    d = x["gaze_samples"].copy()
    if d.empty:
        return x
    flag = np.zeros(len(d), dtype=bool)
    if method == "mad":
        for _, idx in d.groupby("recording_id", sort=False, dropna=False).groups.items():
            idx = np.asarray(list(idx))
            gx = _finite(d.loc[idx, "gaze_x"])
            gy = _finite(d.loc[idx, "gaze_y"])
            mx = np.nanmedian(gx)
            my = np.nanmedian(gy)
            sx = _mad(gx)
            sy = _mad(gy)
            local = (np.abs(gx - mx) > float(threshold) * sx) | (np.abs(gy - my) > float(threshold) * sy)
            local[~np.isfinite(gx) | ~np.isfinite(gy)] = False
            flag[idx] = local
    elif method == "velocity":
        cutoff = float(threshold if max_velocity is None else max_velocity)
        vel = gaze_velocity(d)
        lookup = vel.set_index(["recording_id", "sample_id"])["velocity"]
        values = []
        for row in d[["recording_id", "sample_id"]].itertuples(index=False):
            try:
                values.append(float(lookup.loc[(row.recording_id, row.sample_id)]))
            except (KeyError, TypeError, ValueError):
                values.append(np.nan)
        flag = np.asarray(values, dtype=float) > cutoff
    else:
        spaces = x["coordinate_spaces"]
        for space_id, idx in d.groupby("coordinate_space_id", sort=False, dropna=False).groups.items():
            if pd.isna(space_id):
                continue
            row = spaces[spaces["coordinate_space_id"].eq(space_id)]
            if row.empty:
                continue
            idx = np.asarray(list(idx))
            gx = _finite(d.loc[idx, "gaze_x"])
            gy = _finite(d.loc[idx, "gaze_y"])
            x_unit = row.iloc[0]["x_unit"]
            if x_unit == "normalized":
                flag[idx] = (gx < 0) | (gx > 1) | (gy < 0) | (gy > 1)
            elif x_unit == "pixels":
                width = pd.to_numeric(pd.Series([row.iloc[0]["width"]]), errors="coerce").iloc[0]
                height = pd.to_numeric(pd.Series([row.iloc[0]["height"]]), errors="coerce").iloc[0]
                if np.isfinite(width) and np.isfinite(height):
                    flag[idx] = (gx < 0) | (gx > width) | (gy < 0) | (gy > height)
    d["outlier_flag"] = flag
    out = x.copy()
    out["gaze_samples"] = d
    return add_provenance(
        out,
        "flag_gaze_outliers",
        "gaze_samples",
        f"method={method};n={int(flag.sum())}",
    )


def _missing_runs(mask: np.ndarray) -> list[np.ndarray]:
    runs = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value and i == len(mask) - 1 else i - 1
            runs.append(np.arange(start, end + 1))
            start = None
    return runs


def interpolate_pupil(x, method=("linear", "constant", "none"), max_gap_ms=150, mark=True):
    """Interpolate bounded pupil gaps within recording and eye."""
    _assert_eye_dataset(x)
    if isinstance(method, (list, tuple)):
        method = method[0]
    if method not in {"linear", "constant", "none"}:
        raise EyeProcessValidationError("Invalid pupil interpolation method.")
    if method == "none" or x["eye_samples"].empty:
        return x
    d = x["eye_samples"].copy()
    if "interpolated" not in d.columns:
        d["interpolated"] = False
    for _, idx in d.groupby(["recording_id", "eye"], sort=False, dropna=False).groups.items():
        idx = list(idx)
        order = d.loc[idx, "timestamp_seconds"].astype(float).sort_values(kind="stable").index
        t = _finite(d.loc[order, "timestamp_seconds"])
        y = _finite(d.loc[order, "pupil_diameter"])
        missing = ~np.isfinite(y)
        if not missing.any() or missing.all():
            continue
        order_arr = np.asarray(order)
        for pos in _missing_runs(missing):
            if pos.min() == 0 or pos.max() == len(y) - 1:
                continue
            before = int(pos.min() - 1)
            after = int(pos.max() + 1)
            gap = (t[after] - t[before]) * 1000
            if not np.isfinite(gap) or gap > float(max_gap_ms):
                continue
            if method == "linear":
                y[pos] = np.interp(t[pos], [t[before], t[after]], [y[before], y[after]])
            else:
                y[pos] = y[before]
            if mark:
                d.loc[order_arr[pos], "interpolated"] = True
        d.loc[order, "pupil_diameter"] = y
    out = x.copy()
    out["eye_samples"] = d
    return add_provenance(
        out,
        "interpolate_pupil",
        "eye_samples",
        f"method={method};max_gap_ms={float(max_gap_ms):g}",
        reversible=False,
    )


def filter_pupil(
    x,
    method=("median", "mean", "moving_median", "moving_average", "none"),
    window=5,
):
    """Filter pupil diameter within recording and eye."""
    _assert_eye_dataset(x)
    if isinstance(method, (list, tuple)):
        method = method[0]
    method = {"moving_median": "median", "moving_average": "mean"}.get(method, method)
    if method not in {"median", "mean", "none"}:
        raise EyeProcessValidationError("Invalid pupil filter method.")
    if method == "none" or x["eye_samples"].empty:
        return x
    d = x["eye_samples"].copy()
    if "pupil_raw" not in d.columns:
        d["pupil_raw"] = d["pupil_diameter"]
    fun = np.nanmedian if method == "median" else np.nanmean
    for _, idx in d.groupby(["recording_id", "eye"], sort=False, dropna=False).groups.items():
        idx = list(idx)
        order = d.loc[idx, "timestamp_seconds"].astype(float).sort_values(kind="stable").index
        d.loc[order, "pupil_diameter"] = rolling_apply(d.loc[order, "pupil_diameter"], width=window, FUN=fun)
    out = x.copy()
    out["eye_samples"] = d
    return add_provenance(
        out,
        "filter_pupil",
        "eye_samples",
        f"method={method};window={int(window)}",
        reversible=False,
    )


def baseline_pupil(
    x,
    method=("subtract", "divide", "percent", "zscore", "none"),
    baseline_window=(-0.2, 0),
    anchor=("trial_start", "recording_start"),
    minimum_samples=3,
):
    """Apply trial- or recording-anchored pupil baseline correction."""
    _assert_eye_dataset(x)
    if isinstance(method, (list, tuple)):
        method = method[0]
    if isinstance(anchor, (list, tuple)):
        anchor = anchor[0]
    if method not in {"subtract", "divide", "percent", "zscore", "none"}:
        raise EyeProcessValidationError("Invalid pupil baseline method.")
    if anchor not in {"trial_start", "recording_start"}:
        raise EyeProcessValidationError("Invalid pupil baseline anchor.")
    if method == "none" or x["eye_samples"].empty:
        return x
    d = x["eye_samples"].copy()
    if "pupil_uncorrected" not in d.columns:
        d["pupil_uncorrected"] = d["pupil_diameter"]
    if anchor == "trial_start":
        trials = x["intervals"].loc[
            x["intervals"]["interval_type"].eq("trial"),
            ["recording_id", "trial_id", "start_time"],
        ]
        if trials.empty:
            raise EyeProcessValidationError("Trial intervals are required for trial-start baseline correction.")
        start_map = {(row.recording_id, row.trial_id): row.start_time for row in trials.itertuples(index=False)}
        starts = [
            start_map.get((row.recording_id, row.trial_id), np.nan)
            for row in d[["recording_id", "trial_id"]].itertuples(index=False)
        ]
        start = np.asarray(starts, dtype=float)
    else:
        start = d.groupby("recording_id", dropna=False)["timestamp_seconds"].transform("min").astype(float).to_numpy()
    relative = _finite(d["timestamp_seconds"]) - start
    d["pupil_baseline"] = np.nan
    lo, hi = (float(baseline_window[0]), float(baseline_window[1]))
    keys = ["recording_id", "trial_id", "eye"]
    for _, idx in d.groupby(keys, sort=False, dropna=False).groups.items():
        idx = np.asarray(list(idx))
        y = _finite(d.loc[idx, "pupil_diameter"])
        rel = relative[idx]
        bmask = (rel >= lo) & (rel <= hi) & np.isfinite(y)
        if int(bmask.sum()) < int(minimum_samples):
            continue
        base = float(np.mean(y[bmask]))
        d.loc[idx, "pupil_baseline"] = base
        if method == "subtract":
            y = y - base
        elif method == "divide":
            y = y / base
        elif method == "percent":
            y = (y - base) / base * 100
        else:
            sd = float(np.std(y[bmask], ddof=1)) if int(bmask.sum()) > 1 else np.nan
            y = (y - base) / sd
        d.loc[idx, "pupil_diameter"] = y
    out = x.copy()
    out["eye_samples"] = d
    return add_provenance(
        out,
        "baseline_pupil",
        "eye_samples",
        f"method={method};window={lo:g},{hi:g}",
        reversible=method in {"subtract", "divide", "percent"},
    )


def pupil_deconvolve(x, tau=0.9, regularization=0.01, output_column="pupil_phasic"):
    """Apply the frozen exploratory discrete pupil deconvolution."""
    _assert_eye_dataset(x)
    d = x["eye_samples"].copy()
    if d.empty:
        return x
    d[output_column] = np.nan
    for _, idx in d.groupby(["recording_id", "eye"], sort=False, dropna=False).groups.items():
        idx = list(idx)
        order = d.loc[idx, "timestamp_seconds"].astype(float).sort_values(kind="stable").index
        y = _finite(d.loc[order, "pupil_diameter"])
        t = _finite(d.loc[order, "timestamp_seconds"])
        if np.isfinite(y).sum() < 3:
            continue
        dt = float(np.nanmedian(np.diff(t)))
        alpha = math.exp(-dt / float(tau))
        yy = y.copy()
        yy[~np.isfinite(yy)] = float(np.nanmedian(yy))
        innovation = np.r_[0.0, np.diff(yy) + (1 - alpha) * yy[:-1]]
        innovation = innovation / (1 + float(regularization))
        d.loc[order, output_column] = innovation
    out = x.copy()
    out["eye_samples"] = d
    return add_provenance(
        out,
        "pupil_deconvolve",
        "eye_samples",
        f"tau={float(tau):g};regularization={float(regularization):g}",
        reversible=False,
        warnings=("Exploratory discrete deconvolution; validate assumptions before substantive interpretation."),
    )


def _episode_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return empty_eye_table("episodes")
    return standardize_eye_table(pd.DataFrame(rows), "episodes")


def detect_blinks(
    x,
    min_duration_ms=50,
    max_duration_ms=1000,
    source=("pupil_missing", "validity"),
    overwrite=False,
):
    """Detect blink episodes from missing pupil or validity runs."""
    _assert_eye_dataset(x)
    if isinstance(source, (list, tuple)):
        source = source[0]
    if source not in {"pupil_missing", "validity"}:
        raise EyeProcessValidationError("Invalid blink source.")
    d = x["eye_samples"]
    if d.empty:
        return x
    rows = []
    k = 0
    for _, z in d.groupby(["recording_id", "eye"], sort=False, dropna=False):
        z = z.sort_values("timestamp_seconds", kind="stable")
        if source == "pupil_missing":
            missing = ~np.isfinite(_finite(z["pupil_diameter"]))
        else:
            valid = z["pupil_valid"].astype("boolean")
            missing = (~valid.fillna(False)).to_numpy(dtype=bool)
        t = _finite(z["timestamp_seconds"])
        for pos in _missing_runs(missing):
            duration = (t[pos.max()] - t[pos.min()]) * 1000
            if not np.isfinite(duration):
                continue
            if duration < float(min_duration_ms) or duration > float(max_duration_ms):
                continue
            k += 1
            zz = z.iloc[pos]
            rows.append(
                {
                    "episode_id": (f"{z.iloc[0]['recording_id']}_blink_{z.iloc[0]['eye']}_{k:06d}"),
                    "recording_id": z.iloc[0]["recording_id"],
                    "episode_type": "blink",
                    "eye": z.iloc[0]["eye"],
                    "start_time": float(t[pos.min()]),
                    "end_time": float(t[pos.max()]),
                    "duration_ms": float(duration),
                    "start_x": np.nan,
                    "start_y": np.nan,
                    "end_x": np.nan,
                    "end_y": np.nan,
                    "centroid_x": np.nan,
                    "centroid_y": np.nan,
                    "amplitude": np.nan,
                    "peak_velocity": np.nan,
                    "dispersion": np.nan,
                    "coordinate_space_id": pd.NA,
                    "source_algorithm": f"eyeprocess_{source}",
                    "source_parameters": (f"min={float(min_duration_ms):g};max={float(max_duration_ms):g}"),
                    "derived_by": "eyeprocess",
                    "trial_id": _mode_value(zz["trial_id"]),
                    "stimulus_id": _mode_value(zz["stimulus_id"]),
                    "aoi_id": pd.NA,
                }
            )
    out = x.copy()
    episodes = out["episodes"].copy()
    if overwrite and not episodes.empty:
        keep = ~(episodes["episode_type"].eq("blink") & episodes["derived_by"].eq("eyeprocess"))
        episodes = episodes.loc[keep].copy()
    if rows:
        new_episodes = _episode_frame(rows)
        episodes = (
            new_episodes
            if episodes.empty
            else standardize_eye_table(
                pd.concat([episodes, new_episodes], ignore_index=True, sort=False),
                "episodes",
            )
        )
    out["episodes"] = episodes
    return add_provenance(out, "detect_blinks", "episodes", f"{k} blinks; source={source}")


def detect_fixations_ivt(
    x,
    velocity_threshold=30,
    minimum_duration_ms=60,
    maximum_gap_ms=75,
    coordinate_units=("degrees", "pixels", "normalized"),
    overwrite=False,
):
    """Detect I-VT fixations with frozen-R threshold semantics."""
    _assert_eye_dataset(x)
    if isinstance(coordinate_units, (list, tuple)):
        coordinate_units = coordinate_units[0]
    if coordinate_units not in {"degrees", "pixels", "normalized"}:
        raise EyeProcessValidationError("Invalid coordinate units.")
    if coordinate_units != "degrees":
        warnings.warn(
            f"I-VT threshold is being applied in `{coordinate_units}` per second, not visual degrees per second.",
            RuntimeWarning,
            stacklevel=2,
        )
    d = x["gaze_samples"]
    if d.empty:
        return x
    rows = []
    k = 0
    for _, z in d.groupby(["recording_id", "trial_id"], sort=False, dropna=False):
        z = z.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
        t = _finite(z["timestamp_seconds"])
        gx = _finite(z["gaze_x"])
        gy = _finite(z["gaze_y"])
        dt = np.r_[np.nan, np.diff(t)]
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.r_[np.nan, np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2) / np.diff(t)]
        valid = z["valid"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        is_fix = np.isfinite(velocity) & (velocity <= float(velocity_threshold)) & valid
        if len(is_fix):
            is_fix[0] = bool(is_fix[1]) if len(is_fix) > 1 else False
        run_ids = np.zeros(len(z), dtype=int)
        current = 0
        for i in range(len(z)):
            if i == 0:
                current += 1
            elif not is_fix[i] or not is_fix[i - 1] or (np.isfinite(dt[i]) and dt[i] * 1000 > float(maximum_gap_ms)):
                current += 1
            run_ids[i] = current
        for run in pd.unique(run_ids[is_fix]):
            pos = np.where((run_ids == run) & is_fix)[0]
            if not pos.size:
                continue
            duration = (np.max(t[pos]) - np.min(t[pos])) * 1000
            if duration < float(minimum_duration_ms):
                continue
            k += 1
            zz = z.iloc[pos]
            rows.append(
                {
                    "episode_id": f"{z.iloc[0]['recording_id']}_ivt_fix_{k:07d}",
                    "recording_id": z.iloc[0]["recording_id"],
                    "episode_type": "fixation",
                    "eye": "combined",
                    "start_time": float(np.min(t[pos])),
                    "end_time": float(np.max(t[pos])),
                    "duration_ms": float(duration),
                    "start_x": float(gx[pos.min()]),
                    "start_y": float(gy[pos.min()]),
                    "end_x": float(gx[pos.max()]),
                    "end_y": float(gy[pos.max()]),
                    "centroid_x": float(np.nanmean(gx[pos])),
                    "centroid_y": float(np.nanmean(gy[pos])),
                    "amplitude": np.nan,
                    "peak_velocity": _safe_max(velocity[pos]),
                    "dispersion": _safe_span(gx[pos]) + _safe_span(gy[pos]),
                    "coordinate_space_id": z.iloc[0]["coordinate_space_id"],
                    "source_algorithm": "I-VT",
                    "source_parameters": (
                        f"velocity_threshold={float(velocity_threshold):g};"
                        f"units={coordinate_units};"
                        f"minimum_duration_ms={float(minimum_duration_ms):g}"
                    ),
                    "derived_by": "eyeprocess",
                    "trial_id": z.iloc[0]["trial_id"],
                    "stimulus_id": _mode_value(zz["stimulus_id"]),
                    "aoi_id": pd.NA,
                }
            )
    out = x.copy()
    episodes = out["episodes"].copy()
    if overwrite and not episodes.empty:
        keep = ~(episodes["episode_type"].eq("fixation") & episodes["derived_by"].eq("eyeprocess"))
        episodes = episodes.loc[keep].copy()
    if rows:
        new_episodes = _episode_frame(rows)
        episodes = (
            new_episodes
            if episodes.empty
            else standardize_eye_table(
                pd.concat([episodes, new_episodes], ignore_index=True, sort=False),
                "episodes",
            )
        )
    out["episodes"] = episodes
    warning_text = "Threshold units are not visual degrees." if coordinate_units != "degrees" else pd.NA
    return add_provenance(
        out,
        "detect_fixations_ivt",
        "episodes",
        f"{k} fixations",
        warnings=warning_text,
    )


def detect_fixations_idt(
    x,
    dispersion_threshold=1,
    minimum_duration_ms=100,
    coordinate_units=("degrees", "pixels", "normalized"),
    overwrite=False,
):
    """Detect I-DT fixations using the frozen-R expanding-window algorithm."""
    _assert_eye_dataset(x)
    if isinstance(coordinate_units, (list, tuple)):
        coordinate_units = coordinate_units[0]
    if coordinate_units not in {"degrees", "pixels", "normalized"}:
        raise EyeProcessValidationError("Invalid coordinate units.")
    d = x["gaze_samples"]
    if d.empty:
        return x
    rows = []
    k = 0
    for _, z in d.groupby(["recording_id", "trial_id"], sort=False, dropna=False):
        z = z.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
        t = _finite(z["timestamp_seconds"])
        gx = _finite(z["gaze_x"])
        gy = _finite(z["gaze_y"])
        n = len(z)
        i = 0
        while i < n:
            j = i
            while j < n - 1 and (t[j] - t[i]) * 1000 < float(minimum_duration_ms):
                j += 1
            if j >= n:
                break
            pos = np.arange(i, j + 1)
            dispersion = _safe_span(gx[pos]) + _safe_span(gy[pos])
            if not np.isfinite(dispersion) or dispersion > float(dispersion_threshold):
                i += 1
                continue
            while j < n - 1:
                cand = np.arange(i, j + 2)
                d2 = _safe_span(gx[cand]) + _safe_span(gy[cand])
                if not np.isfinite(d2) or d2 > float(dispersion_threshold):
                    break
                j += 1
                dispersion = d2
            pos = np.arange(i, j + 1)
            k += 1
            rows.append(
                {
                    "episode_id": f"{z.iloc[0]['recording_id']}_idt_fix_{k:07d}",
                    "recording_id": z.iloc[0]["recording_id"],
                    "episode_type": "fixation",
                    "eye": "combined",
                    "start_time": float(t[i]),
                    "end_time": float(t[j]),
                    "duration_ms": float((t[j] - t[i]) * 1000),
                    "start_x": float(gx[i]),
                    "start_y": float(gy[i]),
                    "end_x": float(gx[j]),
                    "end_y": float(gy[j]),
                    "centroid_x": float(np.nanmean(gx[pos])),
                    "centroid_y": float(np.nanmean(gy[pos])),
                    "amplitude": np.nan,
                    "peak_velocity": np.nan,
                    "dispersion": float(dispersion),
                    "coordinate_space_id": z.iloc[0]["coordinate_space_id"],
                    "source_algorithm": "I-DT",
                    "source_parameters": (
                        f"dispersion_threshold={float(dispersion_threshold):g};"
                        f"units={coordinate_units};"
                        f"minimum_duration_ms={float(minimum_duration_ms):g}"
                    ),
                    "derived_by": "eyeprocess",
                    "trial_id": z.iloc[0]["trial_id"],
                    "stimulus_id": _mode_value(z.iloc[pos]["stimulus_id"]),
                    "aoi_id": pd.NA,
                }
            )
            i = j + 1
    out = x.copy()
    episodes = out["episodes"].copy()
    if overwrite and not episodes.empty:
        keep = ~(episodes["episode_type"].eq("fixation") & episodes["derived_by"].eq("eyeprocess"))
        episodes = episodes.loc[keep].copy()
    if rows:
        new_episodes = _episode_frame(rows)
        episodes = (
            new_episodes
            if episodes.empty
            else standardize_eye_table(
                pd.concat([episodes, new_episodes], ignore_index=True, sort=False),
                "episodes",
            )
        )
    out["episodes"] = episodes
    warning_text = "Dispersion threshold units are not visual degrees." if coordinate_units != "degrees" else pd.NA
    return add_provenance(
        out,
        "detect_fixations_idt",
        "episodes",
        f"{k} fixations",
        warnings=warning_text,
    )


def detect_saccades(x, velocity_threshold=30, minimum_duration_ms=10, overwrite=False):
    """Detect contiguous high-velocity saccade episodes."""
    _assert_eye_dataset(x)
    d = x["gaze_samples"]
    if d.empty:
        return x
    vel = gaze_velocity(d)
    rows = []
    k = 0
    for recording_id, z in vel.groupby("recording_id", sort=False, dropna=False):
        z = z.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
        high = np.isfinite(_finite(z["velocity"])) & (_finite(z["velocity"]) > float(velocity_threshold))
        run_ids = np.cumsum(np.r_[True, high[1:] != high[:-1]])
        for run in pd.unique(run_ids[high]):
            pos = np.where((run_ids == run) & high)[0]
            duration = (z.loc[pos, "timestamp_seconds"].max() - z.loc[pos, "timestamp_seconds"].min()) * 1000
            if duration < float(minimum_duration_ms):
                continue
            sample_ids = z.loc[pos, "sample_id"].tolist()
            orig = d[d["recording_id"].eq(recording_id) & d["sample_id"].isin(sample_ids)].copy()
            order_map = {sample_id: i for i, sample_id in enumerate(sample_ids)}
            orig["_order"] = orig["sample_id"].map(order_map)
            orig = orig.sort_values("_order", kind="stable")
            if orig.empty:
                continue
            k += 1
            sx, sy = float(orig.iloc[0]["gaze_x"]), float(orig.iloc[0]["gaze_y"])
            ex, ey = float(orig.iloc[-1]["gaze_x"]), float(orig.iloc[-1]["gaze_y"])
            rows.append(
                {
                    "episode_id": f"{recording_id}_saccade_{k:07d}",
                    "recording_id": recording_id,
                    "episode_type": "saccade",
                    "eye": "combined",
                    "start_time": float(z.loc[pos, "timestamp_seconds"].min()),
                    "end_time": float(z.loc[pos, "timestamp_seconds"].max()),
                    "duration_ms": float(duration),
                    "start_x": sx,
                    "start_y": sy,
                    "end_x": ex,
                    "end_y": ey,
                    "centroid_x": np.nan,
                    "centroid_y": np.nan,
                    "amplitude": float(math.hypot(ex - sx, ey - sy)),
                    "peak_velocity": _safe_max(z.loc[pos, "velocity"]),
                    "dispersion": np.nan,
                    "coordinate_space_id": orig.iloc[0]["coordinate_space_id"],
                    "source_algorithm": "velocity threshold",
                    "source_parameters": f"threshold={float(velocity_threshold):g}",
                    "derived_by": "eyeprocess",
                    "trial_id": _mode_value(orig["trial_id"]),
                    "stimulus_id": _mode_value(orig["stimulus_id"]),
                    "aoi_id": pd.NA,
                }
            )
    out = x.copy()
    episodes = out["episodes"].copy()
    if overwrite and not episodes.empty:
        keep = ~(episodes["episode_type"].eq("saccade") & episodes["derived_by"].eq("eyeprocess"))
        episodes = episodes.loc[keep].copy()
    if rows:
        new_episodes = _episode_frame(rows)
        episodes = (
            new_episodes
            if episodes.empty
            else standardize_eye_table(
                pd.concat([episodes, new_episodes], ignore_index=True, sort=False),
                "episodes",
            )
        )
    out["episodes"] = episodes
    return add_provenance(out, "detect_saccades", "episodes", f"{k} saccades")


def preprocess_eye(x, spec=None):
    """Run the frozen-R preprocessing pipeline."""
    _assert_eye_dataset(x)
    spec = preprocess_spec() if spec is None else spec
    if not isinstance(spec, EyePreprocessSpec):
        raise EyeProcessValidationError("`spec` must be created with `preprocess_spec()`.")
    out = filter_gaze(x, spec.gaze_filter, spec.gaze_window)
    out = interpolate_pupil(out, spec.pupil_interpolation, spec.pupil_max_gap_ms)
    out = filter_pupil(out, spec.pupil_filter, spec.pupil_window)
    if spec.pupil_baseline != "none" and not out["intervals"].empty:
        out = baseline_pupil(
            out,
            spec.pupil_baseline,
            spec.pupil_baseline_window,
        )
    if spec.blink_detection:
        out = detect_blinks(out)
    if spec.fixation_algorithm == "ivt":
        out = detect_fixations_ivt(out, **spec.fixation_parameters)
    elif spec.fixation_algorithm == "idt":
        out = detect_fixations_idt(out, **spec.fixation_parameters)
    elif spec.fixation_algorithm != "none":
        raise EyeProcessValidationError("Unknown fixation algorithm in preprocessing spec.")
    return add_provenance(
        out,
        "preprocess_eye",
        "dataset",
        repr(spec),
        reversible=False,
    )


# R/010-features.R ---------------------------------------------------------


@dataclass(slots=True)
class EyeFeatureSpec:
    level: str = "trial"
    window: Any = None
    include_post_response: bool = False
    minimum_observed_fraction: float = 0.5
    gaze: tuple[str, ...] = (
        "fixation_count",
        "fixation_duration",
        "dwell_time",
        "first_fixation_latency",
        "revisits",
        "entropy",
    )
    pupil: tuple[str, ...] = ("mean", "peak", "auc", "slope", "latency_peak")
    response_time: bool = True
    biometrics: bool = True


def feature_spec(
    level=("trial", "trial_aoi", "recording", "participant_item"),
    window=None,
    include_post_response=False,
    minimum_observed_fraction=0.5,
    gaze=(
        "fixation_count",
        "fixation_duration",
        "dwell_time",
        "first_fixation_latency",
        "revisits",
        "entropy",
    ),
    pupil=("mean", "peak", "auc", "slope", "latency_peak"),
    response_time=True,
    biometrics=True,
):
    """Create the frozen-R feature specification."""
    if isinstance(level, (list, tuple)):
        level = level[0]
    if level not in {"trial", "trial_aoi", "recording", "participant_item"}:
        raise EyeProcessValidationError("Invalid feature level.")
    return EyeFeatureSpec(
        level=level,
        window=window,
        include_post_response=bool(include_post_response),
        minimum_observed_fraction=float(minimum_observed_fraction),
        gaze=tuple(gaze),
        pupil=tuple(pupil),
        response_time=bool(response_time),
        biometrics=bool(biometrics),
    )


def _feature_rows(
    base: dict[str, Any],
    values: dict[str, Any],
    units: Sequence[str] | str,
    level: str,
    method: str,
    parameters=pd.NA,
    window_start=np.nan,
    window_end=np.nan,
    observed_fraction=np.nan,
) -> pd.DataFrame:
    if not values:
        return empty_eye_table("features")
    names = list(values)
    if isinstance(units, str):
        units = [units] * len(names)
    elif len(units) == 1 and len(names) > 1:
        units = list(units) * len(names)
    rows = []
    derived_at = _now_utc()
    for name, unit in zip(names, units):
        rows.append(
            {
                "feature_id": _next_id("feature"),
                "recording_id": base.get("recording_id", pd.NA),
                "participant_id": base.get("participant_id", pd.NA),
                "trial_id": base.get("trial_id", pd.NA),
                "item_id": base.get("item_id", pd.NA),
                "stimulus_id": base.get("stimulus_id", pd.NA),
                "aoi_id": base.get("aoi_id", pd.NA),
                "feature_name": name,
                "value": pd.to_numeric(pd.Series([values[name]]), errors="coerce").iloc[0],
                "unit": unit,
                "level": level,
                "window_start": window_start,
                "window_end": window_end,
                "observed_fraction": observed_fraction,
                "method": method,
                "parameters": parameters,
                "derived_at": derived_at,
            }
        )
    return standardize_eye_table(pd.DataFrame(rows), "features")


def trial_table(x):
    """Return trial intervals."""
    _assert_eye_dataset(x)
    return x["intervals"].loc[x["intervals"]["interval_type"].eq("trial")].copy()


def summarize_fixations(
    x,
    by=("recording_id", "trial_id", "aoi_id"),
    source=("all", "vendor", "eyeprocess"),
):
    """Summarize fixation episodes by requested grouping fields."""
    _assert_eye_dataset(x)
    if isinstance(source, (list, tuple)):
        source = source[0]
    d = x["episodes"].loc[x["episodes"]["episode_type"].eq("fixation")].copy()
    if source == "vendor":
        d = d[d["derived_by"].eq("vendor")]
    elif source == "eyeprocess":
        d = d[d["derived_by"].eq("eyeprocess")]
    elif source != "all":
        raise EyeProcessValidationError("Invalid fixation source.")
    by = [col for col in by if col in d.columns]
    if d.empty or not by:
        return pd.DataFrame()
    groups = _group_frames(d, by, dropna=True)
    if not groups:
        return pd.DataFrame()
    rows = []
    for z in groups:
        row = {col: z.iloc[0][col] for col in by}
        durations = _finite(z["duration_ms"])
        starts = _finite(z["start_time"])
        ends = _finite(z["end_time"])
        row.update(
            {
                "fixation_count": len(z),
                "fixation_duration_total_ms": float(np.nansum(durations)),
                "fixation_duration_mean_ms": float(np.nanmean(durations)),
                "fixation_duration_median_ms": float(np.nanmedian(durations)),
                "first_fixation_time": float(np.nanmin(starts)),
                "last_fixation_time": float(np.nanmax(ends)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def scanpath_sequence(
    x,
    trial_id=None,
    recording_id=None,
    source=("visits", "fixations", "samples"),
    collapse_consecutive=True,
):
    """Return AOI sequences by recording and trial."""
    _assert_eye_dataset(x)
    if isinstance(source, (list, tuple)):
        source = source[0]
    if source == "samples":
        d = x["gaze_samples"].copy()
        if "aoi_id" not in d.columns:
            raise EyeProcessValidationError("AOIs have not been assigned to gaze samples.")
        d = pd.DataFrame(
            {
                "recording_id": d["recording_id"],
                "trial_id": d["trial_id"],
                "time": d["timestamp_seconds"],
                "aoi_id": d["aoi_id"],
            }
        )
    elif source in {"visits", "fixations"}:
        episode_type = "aoi_visit" if source == "visits" else "fixation"
        d = (
            x["episodes"]
            .loc[
                x["episodes"]["episode_type"].eq(episode_type),
                ["recording_id", "trial_id", "start_time", "aoi_id"],
            ]
            .copy()
        )
        d = d.rename(columns={"start_time": "time"})
    else:
        raise EyeProcessValidationError("Invalid scanpath source.")
    if trial_id is not None:
        ids = {trial_id} if isinstance(trial_id, str) else set(trial_id)
        d = d[d["trial_id"].isin(ids)]
    if recording_id is not None:
        ids = {recording_id} if isinstance(recording_id, str) else set(recording_id)
        d = d[d["recording_id"].isin(ids)]
    d = d[d["aoi_id"].notna()].sort_values(["recording_id", "trial_id", "time"], kind="stable")
    rows = []
    for (rec, trial), z in d.groupby(["recording_id", "trial_id"], sort=False, dropna=True, observed=True):
        seq = z["aoi_id"].astype(str).tolist()
        if collapse_consecutive and seq:
            seq = [state for i, state in enumerate(seq) if i == 0 or state != seq[i - 1]]
        rows.append(
            {
                "recording_id": rec,
                "trial_id": trial,
                "sequence": " > ".join(seq),
                "length": len(seq),
            }
        )
    return pd.DataFrame(rows, columns=["recording_id", "trial_id", "sequence", "length"])


def transition_matrix(
    x,
    normalize=("none", "row", "all"),
    source=("visits", "fixations", "samples"),
    include_self=False,
):
    """Return AOI transition matrix from scanpath sequences."""
    if isinstance(normalize, (list, tuple)):
        normalize = normalize[0]
    if isinstance(source, (list, tuple)):
        source = source[0]
    if normalize not in {"none", "row", "all"}:
        raise EyeProcessValidationError("Invalid transition normalization.")
    seqs = scanpath_sequence(
        x,
        source=source,
        collapse_consecutive=not bool(include_self),
    )
    transitions = []
    for sequence in seqs.get("sequence", pd.Series(dtype=str)):
        states = str(sequence).split(" > ") if sequence else []
        transitions.extend(zip(states[:-1], states[1:]))
    if not include_self:
        transitions = [(a, b) for a, b in transitions if a != b]
    if not transitions:
        return pd.DataFrame(dtype=float)
    levels = sorted({state for pair in transitions for state in pair})
    matrix = pd.DataFrame(0.0, index=levels, columns=levels)
    for from_state, to_state in transitions:
        matrix.loc[from_state, to_state] += 1.0
    if normalize == "row":
        sums = matrix.sum(axis=1).replace(0, 1)
        matrix = matrix.div(sums, axis=0)
    elif normalize == "all":
        total = float(matrix.to_numpy().sum())
        if total:
            matrix = matrix / total
    return matrix


def gaze_entropy(
    x,
    level=("trial", "recording"),
    source=("visits", "fixations", "samples"),
    base=2,
):
    """Compute Shannon entropy of AOI occupancy."""
    _assert_eye_dataset(x)
    if isinstance(level, (list, tuple)):
        level = level[0]
    if isinstance(source, (list, tuple)):
        source = source[0]
    if source == "samples":
        d = x["gaze_samples"].copy()
        if "aoi_id" not in d.columns:
            raise EyeProcessValidationError("AOIs have not been assigned.")
    elif source in {"visits", "fixations"}:
        episode_type = "aoi_visit" if source == "visits" else "fixation"
        d = x["episodes"].loc[x["episodes"]["episode_type"].eq(episode_type)].copy()
    else:
        raise EyeProcessValidationError("Invalid entropy source.")
    d = d[d["aoi_id"].notna()]
    keys = ["recording_id", "trial_id"] if level == "trial" else ["recording_id"]
    rows = []
    for key, z in d.groupby(keys, sort=False, dropna=True, observed=True):
        counts = z["aoi_id"].value_counts().to_numpy(dtype=float)
        p = counts / counts.sum()
        h = float(-np.sum(p * (np.log(p) / np.log(float(base)))))
        row = {}
        if len(keys) == 1:
            row[keys[0]] = key[0] if isinstance(key, tuple) else key
        else:
            if not isinstance(key, tuple):
                key = (key,)
            row.update(dict(zip(keys, key)))
        row.update({"entropy": h, "n_states": len(p)})
        rows.append(row)
    return pd.DataFrame(rows)


def transition_entropy(x, source=("visits", "fixations", "samples"), base=2):
    """Compute row-wise entropy of the normalized AOI transition matrix."""
    if isinstance(source, (list, tuple)):
        source = source[0]
    matrix = transition_matrix(x, normalize="row", source=source)
    if matrix.empty:
        return pd.DataFrame(columns=["aoi_id", "transition_entropy"])
    rows = []
    for aoi_id, row in matrix.iterrows():
        p = row.to_numpy(dtype=float)
        p = p[p > 0]
        h = 0.0 if not p.size else float(-np.sum(p * (np.log(p) / np.log(float(base)))))
        rows.append({"aoi_id": aoi_id, "transition_entropy": h})
    return pd.DataFrame(rows)


def _append_features(x: EyeDataset, features: pd.DataFrame, append: bool, action: str, details: str):
    out = x.copy()
    if append and not out["features"].empty:
        combined = pd.concat([out["features"], features], ignore_index=True, sort=False)
    else:
        combined = features.copy()
    out["features"] = standardize_eye_table(combined, "features")
    return add_provenance(out, action, "features", details)


def derive_gaze_features(
    x,
    level=("trial_aoi", "trial"),
    source=("fixations", "visits", "samples"),
    append=True,
):
    """Derive frozen-R gaze feature rows."""
    _assert_eye_dataset(x)
    if isinstance(level, (list, tuple)):
        level = level[0]
    if isinstance(source, (list, tuple)):
        source = source[0]
    trials = trial_table(x)
    if trials.empty:
        raise EyeProcessValidationError("Trial intervals are required for trial-level gaze features.")
    if source == "fixations":
        d = x["episodes"].loc[x["episodes"]["episode_type"].eq("fixation")].copy()
    elif source == "visits":
        d = x["episodes"].loc[x["episodes"]["episode_type"].eq("aoi_visit")].copy()
    elif source == "samples":
        d = x["gaze_samples"].copy()
        if "aoi_id" not in d.columns:
            d["aoi_id"] = pd.NA
        d["start_time"] = d["timestamp_seconds"]
        d["end_time"] = d["timestamp_seconds"]
        rate_map = {}
        streams = x["streams"]
        gaze_streams = streams[streams["stream_type"].eq("gaze_combined")]
        for rec, z in gaze_streams.groupby("recording_id", sort=False, dropna=False):
            rate_map[rec] = pd.to_numeric(z["observed_rate_hz"], errors="coerce").iloc[0]
        rates = pd.to_numeric(d["recording_id"].map(rate_map), errors="coerce")
        d["duration_ms"] = 1000 / rates
    else:
        raise EyeProcessValidationError("Invalid gaze-feature source.")
    if d.empty:
        return x
    keys = ["recording_id", "trial_id"] + (["aoi_id"] if level == "trial_aoi" else [])
    frames = _group_frames(d[d["trial_id"].notna()], keys, dropna=True)
    feature_frames = []
    for z in frames:
        rec = z.iloc[0]["recording_id"]
        trial = z.iloc[0]["trial_id"]
        tr = trials[trials["recording_id"].eq(rec) & trials["trial_id"].eq(trial)]
        if tr.empty:
            continue
        tr0 = tr.iloc[0]
        trial_duration = (float(tr0["end_time"]) - float(tr0["start_time"])) * 1000
        seq = z.sort_values("start_time", kind="stable")["aoi_id"].dropna().astype(str).tolist()
        revisits = sum(seq[i] != seq[i - 1] for i in range(1, len(seq))) if seq else 0
        if seq:
            counts = pd.Series(seq).value_counts().to_numpy(dtype=float)
            p = counts / counts.sum()
            entropy = float(-np.sum(p * np.log2(p)))
        else:
            entropy = np.nan
        duration = _finite(z["duration_ms"])
        start_time = _finite(z["start_time"])
        dwell = float(np.nansum(duration))
        values = {
            "fixation_count": len(z),
            "fixation_duration_total_ms": dwell,
            "fixation_duration_mean_ms": float(np.nanmean(duration)),
            "fixation_duration_median_ms": float(np.nanmedian(duration)),
            "dwell_time_ms": dwell,
            "dwell_proportion": dwell / trial_duration if trial_duration else np.nan,
            "first_fixation_latency_ms": (float(np.nanmin(start_time)) - float(tr0["start_time"])) * 1000,
            "revisits": revisits,
            "gaze_entropy": entropy,
        }
        base = {
            "recording_id": rec,
            "participant_id": tr0["participant_id"],
            "trial_id": trial,
            "item_id": tr0["item_id"],
            "stimulus_id": tr0["stimulus_id"],
            "aoi_id": z.iloc[0]["aoi_id"] if level == "trial_aoi" else pd.NA,
        }
        units = [
            "count",
            "milliseconds",
            "milliseconds",
            "milliseconds",
            "milliseconds",
            "proportion",
            "milliseconds",
            "count",
            "bits",
        ]
        feature_frames.append(
            _feature_rows(
                base,
                values,
                units,
                level,
                f"derive_gaze_features:{source}",
                observed_fraction=float(np.isfinite(start_time).mean()),
            )
        )
    features = (
        pd.concat(feature_frames, ignore_index=True, sort=False) if feature_frames else empty_eye_table("features")
    )
    return _append_features(
        x,
        features,
        bool(append),
        "derive_gaze_features",
        f"{len(features)} feature rows; level={level};source={source}",
    )


def derive_pupil_features(
    x,
    level=("trial", "trial_aoi"),
    append=True,
    pupil_column="pupil_diameter",
):
    """Derive trial-eye pupil features."""
    _assert_eye_dataset(x)
    if isinstance(level, (list, tuple)):
        level = level[0]
    d = x["eye_samples"]
    if d.empty:
        return x
    if pupil_column not in d.columns:
        raise EyeProcessValidationError(f"Pupil column `{pupil_column}` is absent.")
    trials = trial_table(x)
    if trials.empty:
        raise EyeProcessValidationError("Trial intervals are required.")
    if level == "trial_aoi":
        warnings.warn(
            "Pupil samples are not directly AOI-labelled; trial-level features will "
            "be returned unless AOI labels exist in `eye_samples`.",
            RuntimeWarning,
            stacklevel=2,
        )
    feature_frames = []
    for z in _group_frames(d[d["trial_id"].notna()], ["recording_id", "trial_id", "eye"]):
        tr = trials[trials["recording_id"].eq(z.iloc[0]["recording_id"]) & trials["trial_id"].eq(z.iloc[0]["trial_id"])]
        if tr.empty:
            continue
        tr0 = tr.iloc[0]
        y = _finite(z[pupil_column])
        t = _finite(z["timestamp_seconds"])
        ok = np.isfinite(y) & np.isfinite(t)
        if not ok.any():
            continue
        yy = y[ok]
        tt = t[ok]
        order = np.argsort(tt, kind="stable")
        yy = yy[order]
        tt = tt[order]
        peak_idx = int(np.argmax(yy))
        slope = float(np.polyfit(tt, yy, 1)[0]) if len(yy) >= 2 else np.nan
        values = {
            "pupil_mean": float(np.mean(yy)),
            "pupil_sd": float(np.std(yy, ddof=1)) if len(yy) > 1 else np.nan,
            "pupil_peak": float(np.max(yy)),
            "pupil_minimum": float(np.min(yy)),
            "pupil_auc": _trapz(tt - np.min(tt), yy),
            "pupil_slope": slope,
            "pupil_latency_peak_ms": (float(tt[peak_idx]) - float(tr0["start_time"])) * 1000,
            "pupil_observed_fraction": float(np.isfinite(y).mean()),
            "pupil_interpolated_fraction": (
                float(z["interpolated"].astype("boolean").fillna(False).mean()) if "interpolated" in z.columns else 0.0
            ),
        }
        unit = _first_nonmissing(z.loc[ok, "pupil_unit"], "unknown")
        units = [
            unit,
            unit,
            unit,
            unit,
            f"{unit}*seconds",
            f"{unit}/second",
            "milliseconds",
            "proportion",
            "proportion",
        ]
        base = {
            "recording_id": z.iloc[0]["recording_id"],
            "participant_id": tr0["participant_id"],
            "trial_id": z.iloc[0]["trial_id"],
            "item_id": tr0["item_id"],
            "stimulus_id": tr0["stimulus_id"],
            "aoi_id": pd.NA,
        }
        feature_frames.append(
            _feature_rows(
                base,
                values,
                units,
                "trial_eye",
                f"derive_pupil_features:{pupil_column}",
                observed_fraction=float(np.isfinite(y).mean()),
                parameters=f"eye={z.iloc[0]['eye']}",
            )
        )
    features = (
        pd.concat(feature_frames, ignore_index=True, sort=False) if feature_frames else empty_eye_table("features")
    )
    return _append_features(
        x,
        features,
        bool(append),
        "derive_pupil_features",
        f"{len(features)} feature rows",
    )


def derive_rt_features(x, append=True):
    """Derive response-time and score feature rows."""
    _assert_eye_dataset(x)
    responses = x["responses"]
    if responses.empty:
        return x
    frames = []
    for _, row in responses.iterrows():
        values = {"response_time": row["response_time"], "score": row["score"]}
        base = {
            "recording_id": row["recording_id"],
            "participant_id": row["participant_id"],
            "trial_id": row["trial_id"],
            "item_id": row["item_id"],
            "stimulus_id": pd.NA,
            "aoi_id": pd.NA,
        }
        frames.append(
            _feature_rows(
                base,
                values,
                ["seconds", "score"],
                "response",
                "derive_rt_features",
            )
        )
    features = pd.concat(frames, ignore_index=True, sort=False)
    return _append_features(
        x,
        features,
        bool(append),
        "derive_rt_features",
        f"{len(features)} feature rows",
    )


def derive_biometric_features(x, append=True):
    """Derive trial-channel biometric summary features."""
    _assert_eye_dataset(x)
    d = x["biometrics"]
    if d.empty:
        return x
    trials = trial_table(x)
    frames = []
    grouped = d[d["trial_id"].notna()].groupby(
        ["recording_id", "trial_id", "channel"],
        sort=False,
        dropna=True,
        observed=True,
    )
    for _, z in grouped:
        y = _finite(z["value"])
        t = _finite(z["timestamp_seconds"])
        ok = np.isfinite(y) & np.isfinite(t)
        if not ok.any():
            continue
        tr = trials[trials["recording_id"].eq(z.iloc[0]["recording_id"]) & trials["trial_id"].eq(z.iloc[0]["trial_id"])]
        channel = str(z.iloc[0]["channel"])
        values = {
            f"{channel}_mean": float(np.mean(y[ok])),
            f"{channel}_sd": float(np.std(y[ok], ddof=1)) if int(ok.sum()) > 1 else np.nan,
            f"{channel}_min": float(np.min(y[ok])),
            f"{channel}_max": float(np.max(y[ok])),
            f"{channel}_auc": _trapz(t[ok] - np.min(t[ok]), y[ok]),
            f"{channel}_observed_fraction": float(ok.mean()),
        }
        tr0 = tr.iloc[0] if not tr.empty else None
        base = {
            "recording_id": z.iloc[0]["recording_id"],
            "participant_id": tr0["participant_id"] if tr0 is not None else pd.NA,
            "trial_id": z.iloc[0]["trial_id"],
            "item_id": tr0["item_id"] if tr0 is not None else pd.NA,
            "stimulus_id": (tr0["stimulus_id"] if tr0 is not None else _first_nonmissing(z["stimulus_id"], pd.NA)),
            "aoi_id": pd.NA,
        }
        unit = _first_nonmissing(z["unit"], "unknown")
        units = [unit, unit, unit, unit, f"{unit}*seconds", "proportion"]
        frames.append(
            _feature_rows(
                base,
                values,
                units,
                "trial_channel",
                "derive_biometric_features",
                observed_fraction=float(ok.mean()),
                parameters=f"channel={channel}",
            )
        )
    features = pd.concat(frames, ignore_index=True, sort=False) if frames else empty_eye_table("features")
    return _append_features(
        x,
        features,
        bool(append),
        "derive_biometric_features",
        f"{len(features)} feature rows",
    )


def derive_all_features(x, spec=None, reset=False):
    """Derive gaze, pupil, RT, and biometric features using a feature spec."""
    _assert_eye_dataset(x)
    spec = feature_spec() if spec is None else spec
    if not isinstance(spec, EyeFeatureSpec):
        raise EyeProcessValidationError("`spec` must be created with `feature_spec()`.")
    out = x.copy()
    if reset:
        out["features"] = empty_eye_table("features")
    if not out["episodes"].empty or not out["gaze_samples"].empty:
        source = "fixations" if bool(out["episodes"]["episode_type"].eq("fixation").any()) else "samples"
        out = derive_gaze_features(
            out,
            level="trial_aoi" if spec.level == "trial_aoi" else "trial",
            source=source,
        )
    if not out["eye_samples"].empty:
        out = derive_pupil_features(out)
    if spec.response_time and not out["responses"].empty:
        out = derive_rt_features(out)
    if spec.biometrics and not out["biometrics"].empty:
        out = derive_biometric_features(out)
    return add_provenance(
        out,
        "derive_all_features",
        "features",
        f"total_rows={len(out['features'])}",
    )


def features_wide(
    x,
    id_cols=("recording_id", "participant_id", "trial_id", "item_id", "stimulus_id", "aoi_id"),
    aggregate=np.mean,
):
    """Pivot long canonical feature rows to one wide row per identifier combination."""
    _assert_eye_dataset(x)
    d = x["features"]
    if d.empty:
        return pd.DataFrame()
    id_cols = [col for col in id_cols if col in d.columns]
    rows = []
    grouped = d.groupby(id_cols, sort=False, dropna=False, observed=True) if id_cols else [(None, d)]
    for key, z in grouped:
        base = {}
        if id_cols:
            if not isinstance(key, tuple):
                key = (key,)
            base.update(dict(zip(id_cols, key)))
        values = {}
        for feature_name, zz in z.groupby("feature_name", sort=False, dropna=False):
            numeric = pd.to_numeric(zz["value"], errors="coerce").dropna().to_numpy(dtype=float)
            values[feature_name] = float(aggregate(numeric)) if numeric.size else np.nan
        rows.append({**base, **values})
    return pd.DataFrame(rows)


def feature_dictionary(x):
    """Return unique feature metadata rows."""
    _assert_eye_dataset(x)
    d = x["features"]
    if d.empty:
        return pd.DataFrame()
    return d[["feature_name", "unit", "level", "method", "parameters"]].drop_duplicates(ignore_index=True)


__all__ = [
    "baseline_pupil",
    "derive_all_features",
    "derive_biometric_features",
    "derive_gaze_features",
    "derive_pupil_features",
    "derive_rt_features",
    "detect_blinks",
    "detect_fixations_idt",
    "detect_fixations_ivt",
    "detect_saccades",
    "feature_dictionary",
    "feature_spec",
    "features_wide",
    "filter_gaze",
    "filter_pupil",
    "flag_gaze_outliers",
    "gaze_entropy",
    "gaze_velocity",
    "interpolate_pupil",
    "preprocess_eye",
    "preprocess_spec",
    "pupil_deconvolve",
    "rolling_apply",
    "scanpath_sequence",
    "summarize_fixations",
    "transition_entropy",
    "transition_matrix",
    "trial_table",
]
