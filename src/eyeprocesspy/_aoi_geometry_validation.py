"""Validation and point-membership helpers for AOI perturbation analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._aoi_geometry_primitives import (
    _finite_scalar,
    _frame,
    _on_segment,
    _polygon_array,
    _polygon_self_intersects,
    _signed_polygon_area,
    _software_provenance,
    _stable_frame_hash,
)
from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _infer_shape(row: pd.Series) -> str:
    if "shape_type" in row and pd.notna(row["shape_type"]):
        shape = str(row["shape_type"]).lower()
    elif "polygon" in row and row["polygon"] is not None and not (isinstance(row["polygon"], float) and np.isnan(row["polygon"])):
        shape = "polygon"
    else:
        shape = "rectangle"
    if shape not in {"rectangle", "polygon"}:
        raise EyeProcessValidationError("AOI `shape_type` must be 'rectangle' or 'polygon' for perturbation analysis.")
    return shape


def validate_aoi_geometry(aois: pd.DataFrame, *, allow_overlap: bool = True) -> EyeResult:
    """Validate rectangular and polygonal AOI geometry without resolving overlap."""
    frame = _frame(aois, "aois")
    if frame.empty:
        raise EyeProcessValidationError("`aois` must contain at least one AOI.")
    id_col = next((c for c in ("aoi_id", "aoi", "name", "label") if c in frame.columns), None)
    if id_col is None:
        raise EyeProcessValidationError("`aois` must include an AOI identifier column such as `aoi_id`.")
    ids = frame[id_col].astype("string")
    if ids.isna().any() or (ids.str.len() == 0).any():
        raise EyeProcessValidationError("AOI identifiers must be non-missing and non-empty.")
    if ids.duplicated().any():
        raise EyeProcessValidationError("AOI identifiers must be unique within an AOI specification.")

    rows: list[dict[str, Any]] = []
    normalized = frame.copy()
    normalized["aoi_id"] = ids.astype(str)
    for coordinate_column in ("xmin", "xmax", "ymin", "ymax"):
        if coordinate_column in normalized.columns:
            normalized[coordinate_column] = pd.to_numeric(
                normalized[coordinate_column], errors="coerce"
            ).astype(float)
    normalized["shape_type"] = [_infer_shape(row) for _, row in normalized.iterrows()]
    if "polygon" not in normalized.columns:
        normalized["polygon"] = [None] * len(normalized)

    for i, row in normalized.iterrows():
        aoi_id = row["aoi_id"]
        shape = row["shape_type"]
        if shape == "rectangle":
            required = ["xmin", "xmax", "ymin", "ymax"]
            missing = [c for c in required if c not in normalized.columns]
            if missing:
                raise EyeProcessValidationError("Rectangular AOIs require xmin, xmax, ymin, and ymax columns.")
            vals = {c: _finite_scalar(row[c], f"{aoi_id}.{c}") for c in required}
            if not vals["xmin"] < vals["xmax"] or not vals["ymin"] < vals["ymax"]:
                raise EyeProcessValidationError(f"AOI `{aoi_id}` has zero or negative rectangle area.")
            area = (vals["xmax"] - vals["xmin"]) * (vals["ymax"] - vals["ymin"])
            normalized.loc[i, required] = [vals[c] for c in required]
            rows.append({"aoi_id": aoi_id, "shape_type": shape, "area": area, "valid": True})
        else:
            poly = _polygon_array(row["polygon"])
            if _polygon_self_intersects(poly):
                raise EyeProcessValidationError(f"AOI `{aoi_id}` polygon self-intersects.")
            area = abs(_signed_polygon_area(poly))
            if area <= 1e-12:
                raise EyeProcessValidationError(f"AOI `{aoi_id}` polygon has zero area.")
            normalized.at[i, "polygon"] = poly
            rows.append({"aoi_id": aoi_id, "shape_type": shape, "area": area, "valid": True})

    overlap = _pairwise_overlap(normalized)
    overlapping = overlap.loc[overlap["overlap"]]
    if len(overlapping) and not allow_overlap:
        pairs = ", ".join(f"{r.aoi_1}/{r.aoi_2}" for r in overlapping.itertuples())
        raise EyeProcessValidationError(f"AOIs overlap but `allow_overlap=False`: {pairs}.")

    return _result(
        "eye_aoi_geometry_validation",
        geometry=normalized.reset_index(drop=True),
        audit=pd.DataFrame(rows),
        overlap=overlap,
        overlap_present=bool(len(overlapping)),
        source_hash=_stable_frame_hash(normalized),
        software=_software_provenance(),
        status="valid_with_overlap" if len(overlapping) else "valid",
    )


def _point_in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Boundary-inclusive point-in-polygon test."""
    inside = np.zeros(len(x), dtype=bool)
    boundary = np.zeros(len(x), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
        on = (
            np.isclose(cross, 0.0, atol=1e-12, rtol=0.0)
            & (x >= min(xi, xj) - 1e-12)
            & (x <= max(xi, xj) + 1e-12)
            & (y >= min(yi, yj) - 1e-12)
            & (y <= max(yi, yj) + 1e-12)
        )
        boundary |= on
        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + np.finfo(float).eps) + xi
        )
        inside ^= intersects
        j = i
    return inside | boundary


def _aoi_contains(row: pd.Series, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if row["shape_type"] == "rectangle":
        return (x >= float(row["xmin"])) & (x <= float(row["xmax"])) & (y >= float(row["ymin"])) & (y <= float(row["ymax"]))
    return _point_in_polygon(x, y, _polygon_array(row["polygon"]))


def _row_polygon(row: pd.Series) -> np.ndarray:
    if row["shape_type"] == "rectangle":
        xmin, xmax = float(row["xmin"]), float(row["xmax"])
        ymin, ymax = float(row["ymin"]), float(row["ymax"])
        return np.asarray(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            dtype=float,
        )
    return _polygon_array(row["polygon"])


def _point_strictly_inside_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    for i in range(len(poly)):
        if _on_segment(poly[i], poly[(i + 1) % len(poly)], point):
            return False
    return bool(_point_in_polygon(
        np.asarray([point[0]], dtype=float),
        np.asarray([point[1]], dtype=float),
        poly,
    )[0])


def _proper_segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    # Segment contact at endpoints/collinear boundaries has zero intersection
    # area and is therefore not counted as AOI area overlap.
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    tol = 1e-12
    return (
        ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol))
        and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol))
    )


def _polygons_overlap_area(a: np.ndarray, b: np.ndarray) -> bool:
    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            b1, b2 = b[j], b[(j + 1) % len(b)]
            if _proper_segments_cross(a1, a2, b1, b2):
                return True
    if any(_point_strictly_inside_polygon(p, b) for p in a):
        return True
    if any(_point_strictly_inside_polygon(p, a) for p in b):
        return True
    if len(a) == len(b):
        # Covers identical polygons, whose vertices all lie on boundaries.
        a_sorted = np.asarray(sorted(map(tuple, np.round(a, 12))))
        b_sorted = np.asarray(sorted(map(tuple, np.round(b, 12))))
        if np.array_equal(a_sorted, b_sorted) and abs(_signed_polygon_area(a)) > 1e-12:
            return True
    return False


def _pairwise_overlap(geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(len(geometry) - 1):
        for j in range(i + 1, len(geometry)):
            a = geometry.iloc[i]
            b = geometry.iloc[j]
            pa = _row_polygon(a)
            pb = _row_polygon(b)
            ax0, ax1 = float(pa[:, 0].min()), float(pa[:, 0].max())
            ay0, ay1 = float(pa[:, 1].min()), float(pa[:, 1].max())
            bx0, bx1 = float(pb[:, 0].min()), float(pb[:, 0].max())
            by0, by1 = float(pb[:, 1].min()), float(pb[:, 1].max())
            if min(ax1, bx1) <= max(ax0, bx0) or min(ay1, by1) <= max(ay0, by0):
                flag = False
            else:
                flag = _polygons_overlap_area(pa, pb)
            rows.append(
                {"aoi_1": str(a["aoi_id"]), "aoi_2": str(b["aoi_id"]), "overlap": bool(flag)}
            )
    return pd.DataFrame(rows, columns=["aoi_1", "aoi_2", "overlap"])
