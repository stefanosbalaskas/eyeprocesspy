"""Core Matplotlib plotting surface for frozen ``R/012-plots.R``.

The nineteen public plotting helpers mirror the frozen eyeprocess 0.11.1
selection, aggregation, and diagnostic contracts while following eyeprocesspy's
established plotting convention: return a Matplotlib Axes and attach the
underlying data to ``ax.eyeprocess_plot_data`` (and, where relevant,
``ax.eyeprocess_plot_matrix``).

Matplotlib is lazy-loaded so importing a base eyeprocesspy wheel does not
promote plotting dependencies into the core runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import numpy as np
import pandas as pd

from .coordinates import audit_coordinate_spaces
from .dataset import _assert_eye_dataset
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .foundation_09 import (
    audit_clock_sync,
    audit_missingness,
    audit_sampling_rate,
    audit_signal_quality,
)
from .legacy_models import item_parameters
from .preprocess_features_09 import features_wide, transition_matrix, trial_table

__all__ = [
    "plot_aoi_dwell",
    "plot_biometrics",
    "plot_clock_alignment",
    "plot_coordinate_spaces",
    "plot_eye_overview",
    "plot_eye_trace",
    "plot_feature_correlation",
    "plot_feature_distribution",
    "plot_fixations",
    "plot_gaze_heatmap",
    "plot_item_difficulty",
    "plot_missingness",
    "plot_model_diagnostics",
    "plot_pupil_timeseries",
    "plot_sampling_rate",
    "plot_scanpath",
    "plot_signal_quality",
    "plot_transition_matrix",
    "plot_trial_timeline",
]


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EyeProcessBackendError(
            "Core plotting requires matplotlib. Install `eyeprocesspy[plots]` or the development extras."
        ) from exc
    return plt


def _ax(ax=None):
    if ax is not None:
        return ax
    return _get_plt().subplots()[1]


def _empty(message: str, main: str | None = None, ax=None):
    axis = _ax(ax)
    axis.clear()
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    if main is not None:
        axis.set_title(main)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.eyeprocess_plot_data = pd.DataFrame()
    return axis


def _as_values(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _select_trial_data(
    data: pd.DataFrame,
    trial_id=None,
    recording_id=None,
) -> pd.DataFrame:
    out = data.copy()
    recordings = _as_values(recording_id)
    trials = _as_values(trial_id)
    if recordings and "recording_id" in out.columns:
        out = out[out["recording_id"].isin(recordings)]
    if trials and "trial_id" in out.columns:
        out = out[out["trial_id"].isin(trials)]
    return out.copy()


def _finite(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _set_axis_data(axis, data, matrix=None):
    axis.eyeprocess_plot_data = data
    if matrix is not None:
        axis.eyeprocess_plot_matrix = matrix
    return axis


def _validate_choice(value, allowed, name):
    if isinstance(value, (list, tuple)):
        value = value[0]
    if value not in allowed:
        raise EyeProcessValidationError(f"`{name}` must be one of {', '.join(map(repr, allowed))}.")
    return value


def plot_eye_overview(x, ax=None, **kwargs):
    """Plot canonical table counts, matching frozen ``plot_eye_overview``."""
    del kwargs
    _assert_eye_dataset(x)
    intervals = x["intervals"]
    trials = (
        int(intervals["interval_type"].eq("trial").sum()) if not intervals.empty and "interval_type" in intervals else 0
    )
    counts = pd.Series(
        {
            "recordings": len(x["recordings"]),
            "gaze_samples": len(x["gaze_samples"]),
            "eye_samples": len(x["eye_samples"]),
            "episodes": len(x["episodes"]),
            "events": len(x["events"]),
            "trials": trials,
            "responses": len(x["responses"]),
            "biometrics": len(x["biometrics"]),
            "features": len(x["features"]),
        },
        dtype=float,
    )
    axis = _ax(ax)
    axis.bar(np.arange(len(counts)), counts.to_numpy())
    axis.set_xticks(
        np.arange(len(counts)),
        labels=list(counts.index),
        rotation=90,
    )
    axis.set_ylabel("Rows / counts")
    axis.set_title("eyeprocess dataset overview")
    return _set_axis_data(axis, counts)


def plot_eye_trace(
    x,
    trial_id=None,
    recording_id=None,
    valid_only=True,
    reverse_y=True,
    main="Gaze trace",
    ax=None,
    **kwargs,
):
    """Plot selected gaze samples in timestamp order."""
    del kwargs
    _assert_eye_dataset(x)
    data = _select_trial_data(
        x["gaze_samples"],
        trial_id,
        recording_id,
    )
    if valid_only and not data.empty:
        valid = data["valid"].fillna(False).astype(bool) if "valid" in data else pd.Series(True, index=data.index)
        gx = _finite(data["gaze_x"])
        gy = _finite(data["gaze_y"])
        data = data[valid & gx.notna() & gy.notna()].copy()
    if not data.empty:
        data = data.sort_values("timestamp_seconds", kind="stable")
    if data.empty:
        return _empty("No gaze samples match the selection.", main, ax)
    gx = _finite(data["gaze_x"])
    gy = _finite(data["gaze_y"])
    axis = _ax(ax)
    axis.plot(gx, gy)
    axis.scatter(gx, gy, s=6)
    axis.set(xlabel="Gaze x", ylabel="Gaze y", title=main)
    if reverse_y:
        axis.invert_yaxis()
    return _set_axis_data(axis, data)


def plot_fixations(
    x,
    trial_id=None,
    recording_id=None,
    source=("all", "vendor", "eyeprocess"),
    scale=0.03,
    reverse_y=True,
    main="Fixations",
    ax=None,
    **kwargs,
):
    """Plot fixation centroids with duration-scaled markers."""
    del kwargs
    _assert_eye_dataset(x)
    source = _validate_choice(
        source,
        ("all", "vendor", "eyeprocess"),
        "source",
    )
    data = x["episodes"].copy()
    if "episode_type" in data:
        data = data[data["episode_type"].eq("fixation")]
    data = _select_trial_data(data, trial_id, recording_id)
    if source != "all" and "derived_by" in data:
        data = data[data["derived_by"].eq(source)]
    if not data.empty:
        cx = _finite(data["centroid_x"])
        cy = _finite(data["centroid_y"])
        data = data[cx.notna() & cy.notna()].copy()
    if data.empty:
        return _empty("No fixations match the selection.", main, ax)
    duration = _finite(data["duration_ms"]).fillna(1).clip(lower=1)
    sizes = np.maximum(10.0, np.sqrt(duration.to_numpy()) * float(scale) * 800)
    axis = _ax(ax)
    axis.scatter(
        _finite(data["centroid_x"]),
        _finite(data["centroid_y"]),
        s=sizes,
    )
    axis.set(xlabel="Fixation x", ylabel="Fixation y", title=main)
    if reverse_y:
        axis.invert_yaxis()
    return _set_axis_data(axis, data)


def plot_scanpath(
    x,
    trial_id=None,
    recording_id=None,
    reverse_y=True,
    label=True,
    main="Scanpath",
    ax=None,
    **kwargs,
):
    """Plot ordered AOI visits or, when unavailable, fixation centroids."""
    del kwargs
    _assert_eye_dataset(x)
    data = x["episodes"].copy()
    if data.empty:
        return _empty("No scanpath episodes match the selection.", main, ax)
    episode_type = data["episode_type"].astype("string")
    if episode_type.eq("aoi_visit").any():
        data = data[episode_type.eq("aoi_visit")].copy()
    else:
        data = data[episode_type.eq("fixation")].copy()
    data = _select_trial_data(data, trial_id, recording_id)
    data = data.sort_values("start_time", kind="stable")
    cx = _finite(data["centroid_x"])
    cy = _finite(data["centroid_y"])
    data = data[cx.notna() & cy.notna()].copy()
    if data.empty:
        return _empty("No scanpath episodes match the selection.", main, ax)

    cx = _finite(data["centroid_x"])
    cy = _finite(data["centroid_y"])
    duration = _finite(data["duration_ms"]).fillna(1).clip(lower=1)
    sizes = np.maximum(14.0, np.sqrt(duration.to_numpy()) * 4)
    axis = _ax(ax)
    axis.plot(cx, cy, marker="o")
    axis.scatter(cx, cy, s=sizes)
    if label:
        for index, (xx, yy) in enumerate(zip(cx, cy), start=1):
            axis.annotate(
                str(index),
                (xx, yy),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
            )
    axis.set(xlabel="X", ylabel="Y", title=main)
    if reverse_y:
        axis.invert_yaxis()
    return _set_axis_data(axis, data)


def plot_gaze_heatmap(
    x,
    trial_id=None,
    recording_id=None,
    bins=(50, 50),
    valid_only=True,
    main="Gaze density",
    ax=None,
    **kwargs,
):
    """Plot a two-dimensional gaze-density histogram."""
    del kwargs
    _assert_eye_dataset(x)
    data = _select_trial_data(
        x["gaze_samples"],
        trial_id,
        recording_id,
    )
    if valid_only and not data.empty and "valid" in data:
        data = data[data["valid"].fillna(False).astype(bool)]
    if not data.empty:
        gx = _finite(data["gaze_x"])
        gy = _finite(data["gaze_y"])
        data = data[gx.notna() & gy.notna()].copy()
    if data.empty:
        return _empty("No gaze samples match the selection.", main, ax)

    bins = tuple(int(value) for value in bins)
    if len(bins) != 2 or min(bins) < 1:
        raise EyeProcessValidationError("`bins` must contain two positive integers.")
    gx = _finite(data["gaze_x"]).to_numpy(dtype=float)
    gy = _finite(data["gaze_y"]).to_numpy(dtype=float)
    density, x_edges, y_edges = np.histogram2d(gx, gy, bins=bins)
    axis = _ax(ax)
    axis.imshow(
        density.T,
        origin="lower",
        aspect="auto",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    )
    axis.set(xlabel="Gaze x", ylabel="Gaze y", title=main)
    payload = {
        "x_breaks": x_edges,
        "y_breaks": y_edges,
        "density": density,
    }
    return _set_axis_data(axis, payload, density)


def plot_aoi_dwell(
    x,
    feature=("dwell_time_ms", "dwell_proportion"),
    aggregate: Callable = np.mean,
    main=None,
    ax=None,
    **kwargs,
):
    """Plot AOI-level dwell feature aggregates."""
    del kwargs
    _assert_eye_dataset(x)
    feature = _validate_choice(
        feature,
        ("dwell_time_ms", "dwell_proportion"),
        "feature",
    )
    features = x["features"].copy()
    if features.empty:
        return _empty(
            "No AOI dwell features available.",
            main or "AOI dwell",
            ax,
        )
    data = features[features["feature_name"].eq(feature) & features["aoi_id"].notna()].copy()
    if data.empty:
        return _empty(
            "No AOI dwell features available.",
            main or "AOI dwell",
            ax,
        )
    data["value"] = _finite(data["value"])
    values = data.groupby(
        "aoi_id",
        sort=False,
        dropna=False,
    )["value"].agg(aggregate)
    axis = _ax(ax)
    axis.bar(
        np.arange(len(values)),
        values.to_numpy(dtype=float),
    )
    axis.set_xticks(
        np.arange(len(values)),
        labels=[str(value) for value in values.index],
        rotation=90,
    )
    axis.set_ylabel(feature)
    axis.set_title(main or f"AOI {feature}")
    return _set_axis_data(axis, values)


def plot_transition_matrix(
    x,
    normalize=("row", "none", "all"),
    source=("visits", "fixations", "samples"),
    main="AOI transition matrix",
    ax=None,
    **kwargs,
):
    """Plot the canonical AOI transition matrix."""
    del kwargs
    normalize = _validate_choice(
        normalize,
        ("row", "none", "all"),
        "normalize",
    )
    source = _validate_choice(
        source,
        ("visits", "fixations", "samples"),
        "source",
    )
    matrix = transition_matrix(
        x,
        normalize=normalize,
        source=source,
    )
    if matrix is None or len(matrix) == 0:
        return _empty("No transitions available.", main, ax)
    matrix = pd.DataFrame(matrix)
    axis = _ax(ax)
    axis.imshow(
        matrix.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(matrix.shape[1]),
        labels=[str(value) for value in matrix.columns],
        rotation=90,
    )
    axis.set_yticks(
        np.arange(matrix.shape[0]),
        labels=[str(value) for value in matrix.index],
    )
    axis.set(xlabel="From AOI", ylabel="To AOI", title=main)
    return _set_axis_data(axis, matrix, matrix.to_numpy(dtype=float))


def plot_pupil_timeseries(
    x,
    trial_id=None,
    recording_id=None,
    eye=None,
    column="pupil_diameter",
    main="Pupil time series",
    ax=None,
    **kwargs,
):
    """Plot pupil observations by recording × eye group."""
    del kwargs
    _assert_eye_dataset(x)
    data = _select_trial_data(
        x["eye_samples"],
        trial_id,
        recording_id,
    )
    eyes = _as_values(eye)
    if eyes and "eye" in data:
        data = data[data["eye"].isin(eyes)]
    if column not in data.columns:
        raise EyeProcessValidationError(f"Pupil column `{column}` not found.")
    if not data.empty:
        data = data.sort_values(
            ["recording_id", "eye", "timestamp_seconds"],
            kind="stable",
        )
    if data.empty:
        return _empty("No pupil observations match the selection.", main, ax)

    values = _finite(data[column])
    times = _finite(data["timestamp_seconds"])
    finite = values.notna() & times.notna()
    if not finite.any():
        return _empty("Pupil values are unavailable.", main, ax)
    axis = _ax(ax)
    group_cols = ["recording_id", "eye"]
    for key, group in data.loc[finite].groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):
        axis.plot(
            _finite(group["timestamp_seconds"]),
            _finite(group[column]),
            label=":".join(map(str, key if isinstance(key, tuple) else (key,))),
        )
    axis.set(
        xlabel="Time (seconds)",
        ylabel=column,
        title=main,
    )
    if len(axis.lines) > 1:
        axis.legend(fontsize="small")
    return _set_axis_data(axis, data)


def plot_biometrics(
    x,
    channels=None,
    trial_id=None,
    recording_id=None,
    main="Biometric streams",
    ax=None,
    **kwargs,
):
    """Plot selected biometric channels on a shared time axis."""
    del kwargs
    _assert_eye_dataset(x)
    data = _select_trial_data(
        x["biometrics"],
        trial_id,
        recording_id,
    )
    selected = _as_values(channels)
    if selected and "channel" in data:
        data = data[data["channel"].isin(selected)]
    if data.empty:
        return _empty(
            "No biometric observations match the selection.",
            main,
            ax,
        )
    axis = _ax(ax)
    for channel, group in data.groupby(
        "channel",
        sort=False,
        dropna=False,
    ):
        group = group.sort_values("timestamp_seconds", kind="stable")
        axis.plot(
            _finite(group["timestamp_seconds"]),
            _finite(group["value"]),
            label=str(channel),
        )
    axis.set(
        xlabel="Time (seconds)",
        ylabel="value",
        title=main,
    )
    if data["channel"].nunique(dropna=False) > 1:
        axis.legend(fontsize="small")
    return _set_axis_data(axis, data)


def plot_signal_quality(
    x,
    by_trial=False,
    main="Signal quality",
    ax=None,
    **kwargs,
):
    """Plot signal-quality fractions returned by ``audit_signal_quality``."""
    del kwargs
    quality = audit_signal_quality(x, by_trial=by_trial)
    if quality.empty:
        return _empty("No signal-quality metrics available.", main, ax)
    labels = []
    for row in quality.itertuples(index=False):
        parts = [str(getattr(row, "recording_id", ""))]
        if by_trial and hasattr(row, "trial_id"):
            parts.append(str(row.trial_id))
        parts.append(str(getattr(row, "metric", "")))
        labels.append(":".join(parts))
    values = _finite(quality["value"])
    axis = _ax(ax)
    axis.bar(np.arange(len(quality)), values)
    axis.set_xticks(
        np.arange(len(quality)),
        labels=labels,
        rotation=90,
    )
    max_value = values.max(skipna=True)
    axis.set_ylim(0, max(1.0, float(max_value) if pd.notna(max_value) else 1.0))
    axis.set(ylabel="Fraction", title=main)
    return _set_axis_data(axis, quality)


def plot_sampling_rate(
    x,
    expected_hz=None,
    main="Estimated sampling rate",
    ax=None,
    **kwargs,
):
    """Plot estimated gaze sampling rates."""
    del kwargs
    quality = audit_sampling_rate(x, expected_hz=expected_hz)
    if quality.empty:
        return _empty("No gaze sampling rates available.", main, ax)
    values = _finite(quality["value"])
    axis = _ax(ax)
    axis.bar(np.arange(len(quality)), values)
    labels = (
        quality["recording_id"].astype("string").tolist()
        if "recording_id" in quality
        else [str(index + 1) for index in range(len(quality))]
    )
    axis.set_xticks(
        np.arange(len(quality)),
        labels=labels,
        rotation=90,
    )
    axis.set(ylabel="Hz", title=main)
    if expected_hz is not None:
        axis.axhline(float(expected_hz), linestyle="--")
    return _set_axis_data(axis, quality)


def plot_missingness(
    x,
    component=("gaze_samples", "eye_samples", "biometrics"),
    top=20,
    main=None,
    ax=None,
    **kwargs,
):
    """Plot mean missing fraction by field for one canonical component."""
    del kwargs
    component = _validate_choice(
        component,
        ("gaze_samples", "eye_samples", "biometrics"),
        "component",
    )
    data = audit_missingness(x, component=component)
    if data.empty:
        return _empty(
            "No component data available.",
            main or "Missingness",
            ax,
        )
    aggregate = (
        data.groupby("field", as_index=False, sort=False)["missing_fraction"]
        .mean()
        .sort_values("missing_fraction", ascending=False, kind="stable")
        .head(int(top))
        .reset_index(drop=True)
    )
    axis = _ax(ax)
    axis.bar(
        np.arange(len(aggregate)),
        _finite(aggregate["missing_fraction"]),
    )
    axis.set_xticks(
        np.arange(len(aggregate)),
        labels=aggregate["field"].astype(str).tolist(),
        rotation=90,
    )
    axis.set(
        ylabel="Missing fraction",
        title=main or f"{component} missingness",
    )
    return _set_axis_data(axis, aggregate)


def plot_trial_timeline(
    x,
    recording_id=None,
    main="Trial timeline",
    ax=None,
    **kwargs,
):
    """Plot trial start/end intervals."""
    del kwargs
    _assert_eye_dataset(x)
    data = trial_table(x)
    recordings = _as_values(recording_id)
    if recordings and not data.empty:
        data = data[data["recording_id"].isin(recordings)].copy()
    if not data.empty:
        data = data.sort_values(
            ["recording_id", "start_time"],
            kind="stable",
        )
    if data.empty:
        return _empty("No trials available.", main, ax)

    axis = _ax(ax)
    y = np.arange(len(data))
    start = _finite(data["start_time"]).to_numpy(dtype=float)
    end = _finite(data["end_time"]).to_numpy(dtype=float)
    for pos, lo, hi in zip(y, start, end):
        axis.hlines(pos, lo, hi, linewidth=3)
    axis.set_yticks(
        y,
        labels=data["trial_id"].astype(str).tolist(),
    )
    axis.set(xlabel="Time (seconds)", ylabel="Trial", title=main)
    return _set_axis_data(axis, data)


def plot_feature_distribution(
    x,
    feature_name,
    group=None,
    main=None,
    ax=None,
    **kwargs,
):
    """Plot one feature as a histogram or grouped boxplot."""
    del kwargs
    _assert_eye_dataset(x)
    names = _as_values(feature_name)
    data = x["features"].copy()
    if data.empty:
        return _empty(
            "Feature not found.",
            main or str(feature_name),
            ax,
        )
    data = data[data["feature_name"].isin(names)].copy()
    if data.empty:
        return _empty(
            "Feature not found.",
            main or str(feature_name),
            ax,
        )
    axis = _ax(ax)
    values = _finite(data["value"])
    if group is None or group not in data.columns:
        axis.hist(values.dropna().to_numpy(dtype=float))
        axis.set(
            xlabel=str(feature_name),
            title=main or f"Distribution of {feature_name}",
        )
    else:
        grouped = [
            _finite(frame["value"]).dropna().to_numpy(dtype=float)
            for _, frame in data.groupby(
                group,
                sort=False,
                dropna=False,
            )
        ]
        labels = [
            str(key)
            for key, _ in data.groupby(
                group,
                sort=False,
                dropna=False,
            )
        ]
        axis.boxplot(grouped, tick_labels=labels)
        axis.set(
            xlabel=group,
            ylabel=str(feature_name),
            title=main or str(feature_name),
        )
    return _set_axis_data(axis, data)


def plot_feature_correlation(
    x,
    features=None,
    main="Feature correlations",
    ax=None,
    **kwargs,
):
    """Plot pairwise feature correlations from ``features_wide``."""
    del kwargs
    _assert_eye_dataset(x)
    wide = features_wide(x)
    requested = _as_values(features)
    if requested:
        identifiers = [
            name
            for name in (
                "recording_id",
                "participant_id",
                "trial_id",
                "item_id",
            )
            if name in wide.columns
        ]
        selected = [name for name in requested if name in wide.columns]
        wide = wide[identifiers + selected]
    numeric = wide.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return _empty(
            "At least two numeric features are required.",
            main,
            ax,
        )
    matrix = numeric.corr()
    axis = _ax(ax)
    axis.imshow(
        matrix.to_numpy(dtype=float),
        vmin=-1,
        vmax=1,
        origin="lower",
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(matrix.shape[1]),
        labels=list(map(str, matrix.columns)),
        rotation=90,
    )
    axis.set_yticks(
        np.arange(matrix.shape[0]),
        labels=list(map(str, matrix.index)),
    )
    axis.set_title(main)
    return _set_axis_data(axis, matrix, matrix.to_numpy(dtype=float))


def plot_coordinate_spaces(
    x,
    main="Coordinate-space usage",
    ax=None,
    **kwargs,
):
    """Plot canonical coordinate-space usage counts."""
    del kwargs
    data = audit_coordinate_spaces(x)
    if data.empty:
        return _empty("No coordinate spaces are in use.", main, ax)
    columns = [name for name in ("n_gaze", "n_episodes", "n_aoi_geometry") if name in data.columns]
    if not columns:
        return _empty("No coordinate spaces are in use.", main, ax)
    axis = _ax(ax)
    bottom = np.zeros(len(data), dtype=float)
    x_pos = np.arange(len(data))
    for column in columns:
        values = _finite(data[column]).fillna(0).to_numpy(dtype=float)
        axis.bar(x_pos, values, bottom=bottom, label=column)
        bottom += values
    labels = (
        data["coordinate_space_id"].astype(str).tolist()
        if "coordinate_space_id" in data
        else [str(index + 1) for index in range(len(data))]
    )
    axis.set_xticks(x_pos, labels=labels, rotation=90)
    axis.set(ylabel="Rows", title=main)
    if len(columns) > 1:
        axis.legend(fontsize="small")
    return _set_axis_data(axis, data)


def plot_clock_alignment(
    x,
    channel=None,
    main="Gaze and biometric clocks",
    ax=None,
    **kwargs,
):
    """Plot gaze and biometric clock overlap by recording."""
    del kwargs
    data = audit_clock_sync(x, channel=channel)
    required = {
        "gaze_start",
        "gaze_end",
        "biometric_start",
        "biometric_end",
    }
    if data.empty or not required.issubset(data.columns):
        return _empty(
            "Clock overlap cannot be evaluated.",
            main,
            ax,
        )
    axis = _ax(ax)
    y = np.arange(len(data), dtype=float)
    for position, row in zip(y, data.itertuples(index=False)):
        axis.hlines(
            position + 0.12,
            float(row.gaze_start),
            float(row.gaze_end),
            linewidth=4,
            label="Gaze" if position == 0 else None,
        )
        axis.hlines(
            position - 0.12,
            float(row.biometric_start),
            float(row.biometric_end),
            linewidth=4,
            linestyles="--",
            label="Biometrics" if position == 0 else None,
        )
    labels = (
        data["recording_id"].astype(str).tolist()
        if "recording_id" in data
        else [str(index + 1) for index in range(len(data))]
    )
    axis.set_yticks(y, labels=labels)
    axis.set(xlabel="Time (seconds)", ylabel="Recording", title=main)
    axis.legend(fontsize="small")
    return _set_axis_data(axis, data)


def plot_item_difficulty(model, ax=None, **kwargs):
    """Plot item difficulty from an eyeprocess model."""
    del kwargs
    parameters = item_parameters(model)
    if parameters.empty:
        return _empty("No item parameters available.", "Item difficulty", ax)
    difficulty = next(
        (column for column in ("difficulty", "b", "d") if column in parameters.columns),
        None,
    )
    if difficulty is None:
        return _empty(
            "Difficulty parameter not identified.",
            "Item difficulty",
            ax,
        )
    values = _finite(parameters[difficulty])
    labels = (
        parameters["item_id"].astype(str).tolist()
        if "item_id" in parameters
        else [str(index + 1) for index in range(len(parameters))]
    )
    axis = _ax(ax)
    y = np.arange(len(parameters))
    axis.scatter(values, y)
    axis.set_yticks(y, labels=labels)
    axis.set(xlabel="Difficulty", title="Item difficulty")
    return _set_axis_data(axis, parameters)


def plot_model_diagnostics(model, ax=None, **kwargs):
    """Plot generic fitted-versus-residual diagnostics when available."""
    del kwargs
    fit = None
    if getattr(model, "eyeprocess_class", None) == "eyeprocess_model":
        try:
            fit = model.fit
        except Exception:
            fit = None
    if fit is None:
        return _empty(
            "No generic diagnostic plot is available for this model class.",
            "Model diagnostics",
            ax,
        )

    fitted = getattr(fit, "fittedvalues", None)
    residual = None
    for name in ("resid_deviance", "resid_pearson", "resid"):
        value = getattr(fit, name, None)
        if value is not None:
            residual = value
            break
    if fitted is None or residual is None:
        return _empty(
            "No generic diagnostic plot is available for this model class.",
            "Model diagnostics",
            ax,
        )

    fitted = pd.Series(fitted, dtype=float)
    residual = pd.Series(residual, dtype=float)
    data = pd.DataFrame(
        {
            "fitted": fitted.reset_index(drop=True),
            "residual": residual.reset_index(drop=True),
        }
    ).replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    if data.empty:
        return _empty(
            "No generic diagnostic plot is available for this model class.",
            "Model diagnostics",
            ax,
        )
    axis = _ax(ax)
    axis.scatter(data["fitted"], data["residual"])
    axis.axhline(0, linestyle="--")
    axis.set(
        xlabel="Fitted",
        ylabel="Residual",
        title="Model diagnostics",
    )
    return _set_axis_data(axis, data)
