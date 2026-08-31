"""Shared measurement-intelligence plotting infrastructure.

Ports the five public exports in frozen ``R/030-measurement-intelligence-utils.R``:

- ``eye_plot_spec``
- ``plot_diagnostics``
- ``plot_evidence``
- ``plot_sensitivity``
- ``autoplot_eyeprocess``

The frozen R package dispatches through S3 ``plot()`` methods.  Python has no
direct S3 equivalent, so eyeprocesspy resolves class-specific ``plot_<class>``
helpers when available and otherwise falls back to a transparent generic
Matplotlib representation.  Matplotlib remains lazy/optional at import time.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessBackendError, EyeProcessValidationError

__all__ = [
    "EyePlotSpec",
    "eye_plot_spec",
    "plot_diagnostics",
    "plot_evidence",
    "plot_sensitivity",
    "autoplot_eyeprocess",
]


class EyePlotSpec(dict):
    """R-list-like plot specification with stable class metadata."""

    eyeprocess_class = "eye_plot_spec"

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        lines = ["eyeprocess plot specification", f"Type: {self['type']}"]
        if self.get("title") is not None:
            lines.append(f"Title: {self['title']}")
        if self.get("caption") is not None:
            lines.append(f"Note: {self['caption']}")
        return "\n".join(lines)


def eye_plot_spec(
    type="default",
    title=None,
    xlab=None,
    ylab=None,
    caption=None,
    show_uncertainty=True,
    show_raw=True,
    facet_by=None,
    label_items=False,
    interactive=False,
):
    """Create the frozen shared eyeprocess plot specification."""
    return EyePlotSpec(
        {
            "type": str(type),
            "title": title,
            "xlab": xlab,
            "ylab": ylab,
            "caption": caption,
            "show_uncertainty": bool(show_uncertainty),
            "show_raw": bool(show_raw),
            "facet_by": facet_by,
            "label_items": bool(label_items),
            "interactive": bool(interactive),
        }
    )


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EyeProcessBackendError(
            "Measurement-intelligence plotting requires matplotlib. "
            "Install `eyeprocesspy[plots]` or the development extras."
        ) from exc
    return plt


def _new_axis(ax=None):
    if ax is not None:
        return ax
    return _get_plt().subplots()[1]


def _set_plot_data(axis, data):
    try:
        axis.eyeprocess_plot_data = data
    except Exception:
        pass
    return axis


def _call_supported(function, x, **kwargs):
    """Call a plot helper without leaking unsupported Python-only keywords."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(x, **kwargs)

    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_kwargs:
        return function(x, **kwargs)

    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(x, **supported)


def _class_name(x) -> str | None:
    value = getattr(x, "eyeprocess_class", None)
    if value:
        return str(value)
    if isinstance(x, pd.DataFrame):
        value = x.attrs.get("eyeprocess_class")
        if value:
            return str(value)
    return None


def _specialized_plotter(x):
    cls = _class_name(x)
    if not cls:
        return None

    # Resolve lazily after package initialization.  A large fraction of the
    # post-0.9 modules already expose ``plot_eye_<result-class>`` helpers.
    import eyeprocesspy as ep

    candidates = [f"plot_{cls}"]
    if cls.startswith("eye_"):
        candidates.append(f"plot_{cls[4:]}")

    # Core legacy model family predates the class-named plot convention.
    if cls == "eyeprocess_model":
        candidates.insert(0, "plot_model_diagnostics")

    for name in candidates:
        function = getattr(ep, name, None)
        if callable(function) and function not in {
            plot_diagnostics,
            plot_evidence,
            plot_sensitivity,
            autoplot_eyeprocess,
        }:
            return function
    return None


def _candidate_frame(x, plot_type="default"):
    if isinstance(x, pd.DataFrame):
        return x.copy()

    if not isinstance(x, Mapping):
        return None

    priorities = {
        "diagnostics": [
            "diagnostics",
            "summary",
            "table",
            "data",
            "results",
            "observations",
        ],
        "evidence": [
            "evidence",
            "nodes",
            "summary",
            "table",
            "data",
            "results",
        ],
        "sensitivity": [
            "sensitivity",
            "summary",
            "table",
            "data",
            "results",
            "components",
        ],
        "default": [
            "summary",
            "table",
            "data",
            "results",
            "components",
            "nodes",
            "observations",
            "probabilities",
        ],
    }
    keys = priorities.get(plot_type, priorities["default"])
    for key in keys:
        value = x.get(key)
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame(name=value.name or "value")
    return None


def _generic_frame_plot(frame, *, plot_type="default", ax=None, **kwargs):
    del kwargs
    axis = _new_axis(ax)
    data = pd.DataFrame(frame).copy()
    numeric = data.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if numeric.empty:
        axis.text(
            0.5,
            0.5,
            "No plottable numeric data.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(str(plot_type).replace("_", " ").title())
        return _set_plot_data(axis, data)

    columns = list(numeric.columns)
    if plot_type == "diagnostics" and len(columns) >= 2:
        x_col, y_col = columns[:2]
        axis.scatter(numeric[x_col], numeric[y_col])
        axis.axhline(0, linestyle="--")
        axis.set(xlabel=str(x_col), ylabel=str(y_col))
    elif len(columns) == 1:
        column = columns[0]
        axis.plot(np.arange(len(numeric)), numeric[column])
        axis.set(xlabel="Index", ylabel=str(column))
    else:
        x_col, y_col = columns[:2]
        axis.scatter(numeric[x_col], numeric[y_col])
        axis.set(xlabel=str(x_col), ylabel=str(y_col))

    axis.set_title(str(plot_type).replace("_", " ").title())
    return _set_plot_data(axis, data)


def _generic_mapping_plot(x, *, plot_type="default", ax=None, **kwargs):
    frame = _candidate_frame(x, plot_type)
    if frame is not None:
        return _generic_frame_plot(
            frame,
            plot_type=plot_type,
            ax=ax,
            **kwargs,
        )

    numeric = {
        str(key): float(value)
        for key, value in x.items()
        if np.isscalar(value)
        and not isinstance(value, (str, bytes, bool))
        and pd.notna(value)
        and np.isfinite(float(value))
    }
    axis = _new_axis(ax)
    if numeric:
        labels = list(numeric)
        values = np.asarray(list(numeric.values()), dtype=float)
        axis.bar(np.arange(len(values)), values)
        axis.set_xticks(
            np.arange(len(values)),
            labels=labels,
            rotation=90,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No generic plot representation is available.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title(str(plot_type).replace("_", " ").title())
    return _set_plot_data(axis, dict(x))


def _dispatch_plot(x, *, plot_type="default", ax=None, **kwargs):
    plotter = _specialized_plotter(x)
    if plotter is not None:
        return _call_supported(
            plotter,
            x,
            type=plot_type,
            ax=ax,
            **kwargs,
        )

    if isinstance(x, pd.DataFrame):
        return _generic_frame_plot(
            x,
            plot_type=plot_type,
            ax=ax,
            **kwargs,
        )

    if isinstance(x, Mapping):
        return _generic_mapping_plot(
            x,
            plot_type=plot_type,
            ax=ax,
            **kwargs,
        )

    method = getattr(x, "plot", None)
    if callable(method):
        try:
            return method(**kwargs)
        except TypeError:
            return method()

    raise EyeProcessValidationError(f"No eyeprocess plotting method is available for class `{type(x).__name__}`.")


def _plot_with_fallback(x, plot_type, *, ax=None, **kwargs):
    try:
        return _dispatch_plot(
            x,
            plot_type=plot_type,
            ax=ax,
            **kwargs,
        )
    except (
        EyeProcessValidationError,
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
    ) as exc:
        warnings.warn(
            "Plot type "
            f"`{plot_type}` is not specialized for class "
            f"`{_class_name(x) or type(x).__name__}`; "
            "using the default plot for that result.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            return _dispatch_plot(
                x,
                plot_type="default",
                ax=ax,
                **kwargs,
            )
        except Exception as fallback_exc:
            raise EyeProcessValidationError(
                f"No default eyeprocess plot is available after the `{plot_type}` fallback."
            ) from fallback_exc


def plot_diagnostics(x, ax=None, **kwargs):
    """Plot diagnostic evidence, using class-specific dispatch when available."""
    return _plot_with_fallback(
        x,
        "diagnostics",
        ax=ax,
        **kwargs,
    )


def plot_evidence(x, ax=None, **kwargs):
    """Plot scientific evidence, using class-specific dispatch when available."""
    return _plot_with_fallback(
        x,
        "evidence",
        ax=ax,
        **kwargs,
    )


def plot_sensitivity(x, ax=None, **kwargs):
    """Plot sensitivity evidence, using class-specific dispatch when available."""
    return _plot_with_fallback(
        x,
        "sensitivity",
        ax=ax,
        **kwargs,
    )


def autoplot_eyeprocess(object, ax=None, **kwargs):
    """Autoplot-compatible wrapper around eyeprocesspy plot dispatch."""
    return _dispatch_plot(
        object,
        plot_type="default",
        ax=ax,
        **kwargs,
    )
