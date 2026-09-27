"""Vendor-neutral time-varying AOI geometry and assignment."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult

OUTSIDE = "__outside__"
AMBIGUOUS = "__ambiguous__"


def _frame(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _require(data: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required column(s): {', '.join(missing)}.")


def _vertices(value: Any) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise EyeProcessValidationError("Polygon vertices could not be parsed.") from exc
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3 or not np.isfinite(array).all():
        raise EyeProcessValidationError("Polygon vertices must be a finite n by 2 array with n >= 3.")
    return array


def validate_dynamic_aoi_spec(
    aois: Any,
    *,
    aoi: str = "aoi",
    time: str = "timestamp",
    shape: str = "shape",
) -> EyeResult:
    """Validate rectangle, polygon, or mask keyframes without filling missing geometry."""
    table = _frame(aois, "aois")
    _require(table, [aoi, time], "aois")
    if table[aoi].isna().any() or table[aoi].astype(str).str.strip().eq("").any():
        raise EyeProcessValidationError("Dynamic AOI names must be non-missing and non-empty.")
    if shape not in table:
        table[shape] = "rectangle"
    table[shape] = table[shape].astype(str).str.lower()
    allowed = {"rectangle", "polygon", "mask"}
    if not set(table[shape]).issubset(allowed):
        raise EyeProcessValidationError("Dynamic AOI shape must be rectangle, polygon, or mask.")
    table[time] = pd.to_numeric(table[time], errors="coerce")
    if not np.isfinite(table[time].to_numpy(float)).all():
        raise EyeProcessValidationError("Dynamic AOI timestamps must be finite.")
    if table.duplicated([aoi, time]).any():
        raise EyeProcessValidationError("Each AOI may have only one geometry keyframe per timestamp.")

    for _, row in table.iterrows():
        kind = row[shape]
        if kind == "rectangle":
            cols = ["x_min", "x_max", "y_min", "y_max"]
            _require(table, cols, "aois")
            vals = pd.to_numeric(pd.Series([row[c] for c in cols]), errors="coerce").to_numpy(float)
            if not np.isfinite(vals).all() or vals[0] > vals[1] or vals[2] > vals[3]:
                raise EyeProcessValidationError("Rectangle bounds must be finite and ordered.")
        elif kind == "polygon":
            if "vertices" not in table:
                raise EyeProcessValidationError("Polygon AOIs require a vertices column.")
            _vertices(row["vertices"])
        else:
            _require(table, ["mask", "x_min", "x_max", "y_min", "y_max"], "aois")
            mask = np.asarray(row["mask"])
            vals = pd.to_numeric(
                pd.Series([row.x_min, row.x_max, row.y_min, row.y_max]), errors="coerce"
            ).to_numpy(float)
            if (
                mask.ndim != 2
                or mask.size == 0
                or not np.isfinite(vals).all()
                or vals[0] >= vals[1]
                or vals[2] >= vals[3]
            ):
                raise EyeProcessValidationError(
                    "Mask AOIs require a 2D mask and finite, non-zero extent."
                )
    aoi_order = tuple(dict.fromkeys(table[aoi].astype(str).tolist()))
    table = table.sort_values([aoi, time], kind="stable").reset_index(drop=True)
    return EyeResult(
        {
            "keyframes": table,
            "aoi_order": aoi_order,
            "aoi_column": aoi,
            "time_column": time,
            "shape_column": shape,
            "caveat": "Dynamic AOIs are geometry definitions, not evidence that the participant attended to their semantic content.",
        },
        eyeprocess_class="eye_dynamic_aoi_spec",
    )


def _geometry_at(group: pd.DataFrame, t: float, time_col: str, shape_col: str, interpolation: str, lag: float) -> pd.Series | None:
    times = group[time_col].to_numpy(float)
    if t < times[0] - lag or t > times[-1] + lag:
        return None
    if t <= times[0]:
        return group.iloc[0].copy()
    if t >= times[-1]:
        return group.iloc[-1].copy()
    right = int(np.searchsorted(times, t, side="right"))
    left = right - 1
    if interpolation == "step":
        return group.iloc[left].copy()
    a, b = group.iloc[left].copy(), group.iloc[right].copy()
    if a[shape_col] != b[shape_col]:
        raise EyeProcessValidationError("Shape type cannot change between interpolated keyframes.")
    alpha = float((t - times[left]) / (times[right] - times[left]))
    kind = str(a[shape_col])
    out = a.copy()
    out[time_col] = t
    if kind == "rectangle":
        for column in ["x_min", "x_max", "y_min", "y_max"]:
            out[column] = (1 - alpha) * float(a[column]) + alpha * float(b[column])
    elif kind == "polygon":
        va, vb = _vertices(a["vertices"]), _vertices(b["vertices"])
        if va.shape != vb.shape:
            raise EyeProcessValidationError("Interpolated polygon keyframes require matching vertices.")
        out["vertices"] = (1 - alpha) * va + alpha * vb
    else:
        raise EyeProcessValidationError("Mask AOIs support step interpolation only.")
    return out


def _inside_polygon(x: float, y: float, vertices: np.ndarray) -> bool:
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        crossing = (yi > y) != (yj > y)
        if crossing and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _contains(row: pd.Series, x: float, y: float, shape_col: str) -> bool:
    kind = str(row[shape_col])
    if kind == "rectangle":
        return bool(row.x_min <= x <= row.x_max and row.y_min <= y <= row.y_max)
    if kind == "polygon":
        return _inside_polygon(x, y, _vertices(row["vertices"]))
    mask = np.asarray(row["mask"]).astype(bool)
    if not (row.x_min <= x <= row.x_max and row.y_min <= y <= row.y_max):
        return False
    col = min(mask.shape[1] - 1, int((x - row.x_min) / (row.x_max - row.x_min) * mask.shape[1]))
    rr = min(mask.shape[0] - 1, int((y - row.y_min) / (row.y_max - row.y_min) * mask.shape[0]))
    return bool(mask[rr, col])


def assign_dynamic_aoi(
    samples: Any,
    aois: Any,
    *,
    time: str = "timestamp",
    x: str = "gaze_x",
    y: str = "gaze_y",
    interpolation: str = "linear",
    lag_tolerance: float = 0.0,
    overlap: str = "ambiguous",
) -> EyeResult:
    """Assign samples to time-indexed AOIs with explicit overlap and coverage states."""
    data = _frame(samples, "samples")
    _require(data, [time, x, y], "samples")
    spec = aois if getattr(aois, "eyeprocess_class", None) == "eye_dynamic_aoi_spec" else validate_dynamic_aoi_spec(aois, time=time)
    interpolation = str(interpolation).lower()
    overlap = str(overlap).lower()
    if interpolation not in {"linear", "step"}:
        raise EyeProcessValidationError("interpolation must be 'linear' or 'step'.")
    if overlap not in {"ambiguous", "first", "all"}:
        raise EyeProcessValidationError("overlap must be 'ambiguous', 'first', or 'all'.")
    lag_tolerance = float(lag_tolerance)
    if not np.isfinite(lag_tolerance) or lag_tolerance < 0:
        raise EyeProcessValidationError("lag_tolerance must be finite and non-negative.")

    keyframes = spec["keyframes"]
    ac, tc, sc = spec["aoi_column"], spec["time_column"], spec["shape_column"]
    grouped = {
        name: keyframes.loc[keyframes[ac].astype(str).eq(name)].reset_index(drop=True)
        for name in spec["aoi_order"]
    }
    rows = []
    for sample_id, row in data.reset_index(drop=False).iterrows():
        tt = pd.to_numeric(pd.Series([row[time]]), errors="coerce").iloc[0]
        xx = pd.to_numeric(pd.Series([row[x]]), errors="coerce").iloc[0]
        yy = pd.to_numeric(pd.Series([row[y]]), errors="coerce").iloc[0]
        if not np.isfinite(tt) or not np.isfinite(xx) or not np.isfinite(yy):
            rows.append({"sample_id": sample_id + 1, "aoi": pd.NA, "status": "missing_sample", "n_hits": 0})
            continue
        hits = []
        active = 0
        for name, group in grouped.items():
            geom = _geometry_at(group, float(tt), tc, sc, interpolation, lag_tolerance)
            if geom is None:
                continue
            active += 1
            if _contains(geom, float(xx), float(yy), sc):
                hits.append(str(name))
        if not active:
            assigned, status = pd.NA, "no_active_geometry"
        elif not hits:
            assigned, status = OUTSIDE, "outside"
        elif len(hits) == 1:
            assigned, status = hits[0], "assigned"
        elif overlap == "ambiguous":
            assigned, status = AMBIGUOUS, "ambiguous"
        elif overlap == "first":
            assigned, status = hits[0], "overlap_priority"
        else:
            assigned, status = "|".join(hits), "overlap_all"
        rows.append({"sample_id": sample_id + 1, "aoi": assigned, "status": status, "n_hits": len(hits)})
    assignments = pd.DataFrame(rows)
    return EyeResult(
        {
            "assignments": assignments,
            "spec": spec,
            "interpolation": interpolation,
            "lag_tolerance": lag_tolerance,
            "overlap": overlap,
        },
        eyeprocess_class="eye_dynamic_aoi_assignment",
    )


def audit_dynamic_aoi_coverage(assignment: Any) -> pd.DataFrame:
    """Summarize assignment, uncovered time, outside geometry, and ambiguity."""
    if getattr(assignment, "eyeprocess_class", None) != "eye_dynamic_aoi_assignment":
        raise EyeProcessValidationError("assignment must be an eye_dynamic_aoi_assignment.")
    data = assignment["assignments"]
    counts = data.status.value_counts(dropna=False).rename_axis("status").reset_index(name="n")
    counts["fraction"] = counts.n / max(len(data), 1)
    return counts


def dynamic_aoi_sensitivity(
    samples: Any,
    aois: Any,
    *,
    lag_tolerances: Sequence[float] = (0.0,),
    interpolations: Sequence[str] = ("linear", "step"),
    **kwargs: Any,
) -> pd.DataFrame:
    """Compare dynamic-AOI assignments across declared timing/interpolation choices."""
    branches = []
    baseline = None
    for interpolation in interpolations:
        for lag in lag_tolerances:
            result = assign_dynamic_aoi(
                samples,
                aois,
                interpolation=interpolation,
                lag_tolerance=float(lag),
                **kwargs,
            )
            labels = result["assignments"].aoi.astype("string")
            if baseline is None:
                baseline = labels
            agreement = float((labels.fillna("<NA>") == baseline.fillna("<NA>")).mean())
            coverage = audit_dynamic_aoi_coverage(result)
            no_geometry = float(
                coverage.loc[coverage.status.eq("no_active_geometry"), "fraction"].sum()
            )
            branches.append(
                {
                    "interpolation": interpolation,
                    "lag_tolerance": float(lag),
                    "agreement_with_first": agreement,
                    "no_active_geometry_fraction": no_geometry,
                }
            )
    return pd.DataFrame(branches)


def plot_dynamic_aoi_alignment(assignment: Any, ax: Any = None) -> Any:
    """Plot assignment status counts for a dynamic-AOI result."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    table = audit_dynamic_aoi_coverage(assignment)
    axis = plt.subplots()[1] if ax is None else ax
    axis.bar(table.status.astype(str), table.fraction)
    axis.set_ylabel("Fraction of samples")
    axis.set_title("Dynamic AOI alignment audit")
    axis.tick_params(axis="x", rotation=30)
    setattr(axis, "eyeprocess_plot_data", table.copy())
    return axis


__all__ = [
    "AMBIGUOUS",
    "OUTSIDE",
    "assign_dynamic_aoi",
    "audit_dynamic_aoi_coverage",
    "dynamic_aoi_sensitivity",
    "plot_dynamic_aoi_alignment",
    "validate_dynamic_aoi_spec",
]
