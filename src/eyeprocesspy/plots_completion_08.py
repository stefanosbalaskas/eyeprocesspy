"""Completion of frozen R/064 next-generation plotting utilities.

This module source-ports the five public contracts in
``R/064-next-generation-plots-0-8.R`` that remained unimplemented after the
earlier 0.8 plotting tranche:

- ``plot_aoi_transition_matrix``
- ``plot_aoi_transition_rank``
- ``plot_process_channel_ablation_delta``
- ``plot_pupil_components``
- ``plot_pupil_preprocessing_audit``

The tabulation, normalization, filtering, and channel-ablation calculations
follow frozen eyeprocess 0.11.1. Matplotlib remains lazy/optional. As elsewhere
in eyeprocesspy, plotting functions return an Axes and retain the source-derived
quantity on ``eyeprocess_plot_data``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessBackendError, EyeProcessValidationError

__all__ = [
    "plot_aoi_transition_matrix",
    "plot_aoi_transition_rank",
    "plot_process_channel_ablation_delta",
    "plot_pupil_components",
    "plot_pupil_preprocessing_audit",
]


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EyeProcessBackendError(
            "R/064 plotting requires matplotlib. Install `eyeprocesspy[plots]` or the development extras."
        ) from exc
    return plt


def _axis(ax=None):
    if ax is not None:
        return ax
    return _get_plt().subplots()[1]


def _as_frame(value: Any, *, name: str = "data") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        frame = pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"`{name}` must be coercible to a data frame.") from exc
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EyeProcessValidationError("Missing required column(s): " + ", ".join(missing) + ".")


def _normalize_choice(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else "from"
    value = str(value)
    if value not in {"from", "all", "none"}:
        raise EyeProcessValidationError("`normalize` must be 'from', 'all', or 'none'.")
    return value


def _transition_table(
    data,
    from_col: str = "from",
    to_col: str = "to",
    normalize: str = "from",
) -> pd.DataFrame:
    """Frozen ``.ep08_transition_table`` semantics."""
    normalize = _normalize_choice(normalize)
    frame = _as_frame(data)
    _require_columns(frame, [from_col, to_col])

    from_values = frame[from_col].astype("string")
    to_values = frame[to_col].astype("string")
    ok = (
        from_values.notna()
        & from_values.str.len().fillna(0).gt(0)
        & to_values.notna()
        & to_values.str.len().fillna(0).gt(0)
    )

    if not bool(ok.any()):
        raise EyeProcessValidationError("No complete AOI transitions are available.")

    clean = pd.DataFrame(
        {
            "from": from_values.loc[ok].astype(str),
            "to": to_values.loc[ok].astype(str),
        }
    )

    levels_from = sorted(clean["from"].unique().tolist())
    levels_to = sorted(clean["to"].unique().tolist())
    table = pd.crosstab(clean["from"], clean["to"], dropna=False)
    table = table.reindex(
        index=levels_from,
        columns=levels_to,
        fill_value=0,
    )

    if table.shape[0] == 0 or table.shape[1] == 0:
        raise EyeProcessValidationError("No complete AOI transitions are available.")

    if normalize == "from":
        denominator = table.sum(axis=1).replace(0, 1)
        table = table.div(denominator, axis=0)
    elif normalize == "all":
        total = float(table.to_numpy(dtype=float).sum())
        if total > 0:
            table = table / total

    table.index.name = None
    table.columns.name = None
    return table


def plot_pupil_preprocessing_audit(
    data,
    time="time_ms",
    signals=(
        "pupil_raw",
        "pupil_interpolated",
        "pupil_smoothed",
        "pupil_bc",
    ),
    ax=None,
    **kwargs,
):
    """Plot frozen raw-to-processed pupil preprocessing stages."""
    frame = _as_frame(data)
    _require_columns(frame, [time])

    requested = [signals] if isinstance(signals, str) else list(signals)
    available = [signal for signal in requested if signal in frame.columns]
    if not available:
        raise EyeProcessValidationError("No requested pupil preprocessing signals were found.")

    time_values = pd.to_numeric(frame[time], errors="coerce").to_numpy(dtype=float)
    plotted = frame.loc[:, [time] + available].copy()

    axis = _axis(ax)
    for signal in available:
        values = pd.to_numeric(
            frame[signal],
            errors="coerce",
        ).to_numpy(dtype=float)
        axis.plot(time_values, values, label=str(signal), **kwargs)

    axis.set(
        xlabel="Time",
        ylabel="Pupil signal",
        title="Pupil preprocessing audit",
    )
    axis.legend()
    axis.eyeprocess_plot_data = plotted
    return axis


def plot_pupil_components(
    data,
    time="time_ms",
    smoothed="pupil_smoothed",
    tonic="pupil_tonic",
    phasic="pupil_phasic",
    ax=None,
    **kwargs,
):
    """Plot tonic/phasic pupil components via the frozen preprocessing helper."""
    return plot_pupil_preprocessing_audit(
        data,
        time=time,
        signals=[smoothed, tonic, phasic],
        ax=ax,
        **kwargs,
    )


def plot_aoi_transition_matrix(
    data,
    from_col="from",
    to_col="to",
    normalize="from",
    ax=None,
    **kwargs,
):
    """Plot the frozen AOI transition matrix representation."""
    del kwargs
    normalize = _normalize_choice(normalize)
    matrix = _transition_table(
        data,
        from_col=from_col,
        to_col=to_col,
        normalize=normalize,
    )

    axis = _axis(ax)
    values = matrix.to_numpy(dtype=float)
    axis.imshow(
        values[::-1, :],
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
        labels=list(map(str, reversed(matrix.index.tolist()))),
    )
    axis.set(
        xlabel="To AOI",
        ylabel="From AOI",
        title=f"AOI transition matrix -- normalize: {normalize}",
    )
    axis.eyeprocess_plot_data = matrix.copy()
    axis.eyeprocess_plot_matrix = values.copy()
    return axis


def plot_aoi_transition_rank(
    data,
    from_col="from",
    to_col="to",
    normalize="from",
    top_n=20,
    ax=None,
    **kwargs,
):
    """Plot the highest-ranked AOI transitions by probability or count."""
    del kwargs
    normalize = _normalize_choice(normalize)

    try:
        top_n = int(top_n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError("`top_n` must be a positive integer.") from exc
    if top_n < 1:
        raise EyeProcessValidationError("`top_n` must be a positive integer.")

    matrix = _transition_table(
        data,
        from_col=from_col,
        to_col=to_col,
        normalize=normalize,
    )

    rows = []
    for to_value in matrix.columns:
        for from_value in matrix.index:
            rows.append(
                {
                    "from": str(from_value),
                    "to": str(to_value),
                    "value": float(matrix.loc[from_value, to_value]),
                }
            )

    ranked = pd.DataFrame(rows)
    ranked["transition"] = ranked["from"] + " -> " + ranked["to"]
    ranked = (
        ranked.sort_values(
            "value",
            ascending=False,
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    display = ranked.iloc[::-1].reset_index(drop=True)
    axis = _axis(ax)
    y = np.arange(len(display))
    axis.barh(y, display["value"].to_numpy(dtype=float))
    axis.set_yticks(y, labels=display["transition"].tolist())
    axis.set(
        xlabel=("Count" if normalize == "none" else "Transition probability"),
        title="Top AOI transitions",
    )
    axis.eyeprocess_plot_data = ranked
    axis.eyeprocess_plot_matrix = matrix.copy()
    return axis


def _extract_ablation_table(x: Any, table: Any) -> pd.DataFrame:
    if table is not None:
        return _as_frame(table, name="ablation table")

    if isinstance(x, pd.DataFrame):
        return x.copy()

    if isinstance(x, Mapping):
        for candidate in ("results", "table", "summary", "metrics"):
            if candidate in x and x[candidate] is not None:
                return _as_frame(
                    x[candidate],
                    name="ablation table",
                )

    for candidate in ("results", "table", "summary", "metrics"):
        value = getattr(x, candidate, None)
        if value is not None:
            return _as_frame(
                value,
                name="ablation table",
            )

    raise EyeProcessValidationError("`table` or a compatible ablation result must be supplied.")


def plot_process_channel_ablation_delta(
    x=None,
    table=None,
    channel_col="channel",
    metric_col="metric",
    value_col="value",
    full_label="full",
    metric=None,
    ax=None,
    **kwargs,
):
    """Plot frozen channel-ablation differences from a full/reference model."""
    del kwargs
    frame = _extract_ablation_table(x, table)
    _require_columns(frame, [channel_col, value_col])

    if metric is not None and metric_col in frame.columns:
        requested = [metric] if isinstance(metric, str) else list(metric)
        frame = frame.loc[frame[metric_col].isin(requested)].copy()

    if frame.empty:
        raise EyeProcessValidationError("No ablation rows to plot.")

    channel = frame[channel_col].astype("string").astype(str)
    value = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)

    reference = value[channel.eq(str(full_label)).to_numpy() & np.isfinite(value)]
    if not reference.size:
        reference = value[np.isfinite(value)]
    if not reference.size:
        raise EyeProcessValidationError("No finite ablation metric values are available.")

    reference_value = float(np.max(reference))
    delta = value - reference_value

    result = pd.DataFrame(
        {
            "channel": channel.to_numpy(dtype=object),
            "value": value,
            "delta_from_reference": delta,
        }
    )
    order = np.argsort(delta, kind="stable")
    ordered = result.iloc[order].reset_index(drop=True)

    axis = _axis(ax)
    y = np.arange(len(ordered))
    axis.scatter(
        ordered["delta_from_reference"].to_numpy(dtype=float),
        y,
    )
    axis.set_yticks(y, labels=ordered["channel"].tolist())
    axis.axvline(0, linestyle="--")
    axis.set(
        xlabel="Metric difference from full/reference",
        title="Process-channel ablation delta",
    )
    axis.eyeprocess_plot_data = result
    axis.eyeprocess_plot_reference = reference_value
    return axis
