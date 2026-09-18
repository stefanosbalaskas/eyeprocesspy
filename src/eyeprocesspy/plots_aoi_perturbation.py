"""Plotting helpers for AOI perturbation sensitivity analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._aoi_geometry_primitives import _frame, _polygon_array
from .exceptions import EyeProcessValidationError


def _mpl():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as PolygonPatch
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - optional plotting backend
        raise EyeProcessValidationError("Plotting requires the optional `matplotlib` dependency.") from exc
    return plt, Rectangle, PolygonPatch


def _draw_geometry(ax: Any, geometry: pd.DataFrame, *, alpha: float = 1.0, linestyle: str = "-") -> None:
    _, Rectangle, PolygonPatch = _mpl()
    for _, row in geometry.iterrows():
        if row["shape_type"] == "rectangle":
            ax.add_patch(Rectangle((row["xmin"], row["ymin"]), row["xmax"] - row["xmin"], row["ymax"] - row["ymin"], fill=False, alpha=alpha, linestyle=linestyle))
            cx, cy = (row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2
        else:
            poly = _polygon_array(row["polygon"])
            ax.add_patch(PolygonPatch(poly, fill=False, closed=True, alpha=alpha, linestyle=linestyle))
            cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        ax.text(cx, cy, str(row["aoi_id"]), ha="center", va="center", alpha=alpha)


def plot_aoi_perturbations(x: Any, *, perturbation_id: str | None = None, data: pd.DataFrame | None = None, x_col: str | None = None, y_col: str | None = None, ax: Any = None):
    plt, _, _ = _mpl()
    axis = ax or plt.subplots()[1]
    nominal = x["nominal_aois"] if "nominal_aois" in x else x["nominal_geometry"]
    _draw_geometry(axis, nominal, alpha=0.6, linestyle="--")
    if "grid_result" in x:
        if perturbation_id is None:
            candidates = [p for p in x["grid_result"]["geometries"] if p != "baseline"]
            perturbation_id = candidates[0] if candidates else "baseline"
        geom = x["grid_result"]["geometries"][perturbation_id]
    else:
        geom = x["perturbed_geometry"]
        perturbation_id = x["perturbation_id"]
    _draw_geometry(axis, geom, alpha=1.0, linestyle="-")
    if data is not None:
        if not x_col or not y_col:
            raise EyeProcessValidationError("`x_col` and `y_col` are required when plotting reassigned observations.")
        frame = _frame(data, "data")
        bx = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
        by = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
        if "assignments" in x and perturbation_id in x["assignments"]:
            changed = ~pd.Series(x["assignments"]["baseline"]).reset_index(drop=True).eq(
                pd.Series(x["assignments"][perturbation_id]).reset_index(drop=True)
            ).to_numpy()
            axis.scatter(bx[~changed], by[~changed], s=12, alpha=0.35)
            axis.scatter(bx[changed], by[changed], s=32, marker="x", label="reassigned")
            axis.legend()
        else:
            axis.scatter(bx, by, s=12, alpha=0.35)
    axis.set_title(f"AOI perturbation: {perturbation_id}")
    axis.set_xlabel(x_col or "x")
    axis.set_ylabel(y_col or "y")
    axis.set_aspect("equal", adjustable="datalim")
    return axis


def plot_aoi_assignment_stability(x: Any, *, ax: Any = None):
    plt, _, _ = _mpl()
    axis = ax or plt.subplots()[1]
    data = x["stability"]["overall"].copy() if "stability" in x else x["overall"].copy()
    axis.plot(np.arange(len(data)), data["proportion_unchanged"].to_numpy(dtype=float), marker="o")
    axis.set_xticks(np.arange(len(data)), labels=data["perturbation_id"].astype(str).tolist(), rotation=45, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Proportion unchanged")
    axis.set_title("AOI assignment stability")
    return axis


def plot_aoi_coefficient_stability(x: Any, *, term: str, ax: Any = None):
    plt, _, _ = _mpl()
    axis = ax or plt.subplots()[1]
    data = x["models"].loc[x["models"]["term"].astype(str).eq(str(term))].copy()
    if data.empty:
        raise EyeProcessValidationError(f"No coefficient rows found for term `{term}`.")
    data = data.reset_index(drop=True)
    y = pd.to_numeric(data["estimate"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(data["CI_low"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(data["CI_high"], errors="coerce").to_numpy(dtype=float)
    axis.errorbar(np.arange(len(data)), y, yerr=np.vstack([y - lo, hi - y]), fmt="o-")
    failed = ~data["model_converged"].fillna(False).astype(bool).to_numpy()
    if failed.any():
        axis.scatter(np.where(failed)[0], y[failed], marker="x", s=80, label="not converged")
        axis.legend()
    axis.axhline(0, linewidth=1)
    axis.set_xticks(np.arange(len(data)), labels=data["perturbation_id"].astype(str).tolist(), rotation=45, ha="right")
    axis.set_ylabel("Coefficient estimate")
    axis.set_title(f"Coefficient stability: {term}")
    return axis


def plot_aoi_robustness_surface(
    x: Any,
    *,
    value_col: str = "proportion_unchanged",
    x_col: str = "margin_x",
    y_col: str = "margin_y",
    ax: Any = None,
):
    plt, _, _ = _mpl()
    axis = ax or plt.subplots()[1]
    stability = x["stability"]["overall"].copy()
    grid_table = x["grid"]["table"].copy()
    data = stability.merge(grid_table, on="perturbation_id", how="left")
    if x_col not in data.columns or y_col not in data.columns or value_col not in data.columns:
        raise EyeProcessValidationError("Requested robustness-surface columns are unavailable.")
    pivot = data.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="mean")
    if pivot.empty:
        raise EyeProcessValidationError("Robustness surface requires at least one finite x/y/value combination.")
    im = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    axis.set_xlabel(x_col)
    axis.set_ylabel(y_col)
    axis.set_title("AOI robustness surface")
    plt.colorbar(im, ax=axis, label=value_col)
    return axis


__all__ = [
    "plot_aoi_perturbations", "plot_aoi_assignment_stability",
    "plot_aoi_coefficient_stability", "plot_aoi_robustness_surface",
]
