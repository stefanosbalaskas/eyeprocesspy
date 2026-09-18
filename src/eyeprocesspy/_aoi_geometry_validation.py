"""Validation and point-membership helpers for AOI perturbation analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._aoi_geometry_primitives import (
    _finite_scalar,
    _frame,
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
    inside = np.zeros(len(x), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + np.finfo(float).eps) + xi
        )
        inside ^= intersects
        j = i
    return inside


def _aoi_contains(row: pd.Series, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if row["shape_type"] == "rectangle":
        return (x >= float(row["xmin"])) & (x <= float(row["xmax"])) & (y >= float(row["ymin"])) & (y <= float(row["ymax"]))
    return _point_in_polygon(x, y, _polygon_array(row["polygon"]))


def _pairwise_overlap(geometry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Deterministic grid probe is used only to flag likely overlap for arbitrary
    # polygons. Assignment ambiguity itself is determined exactly at observed
    # points and is never resolved by this diagnostic.
    for i in range(len(geometry) - 1):
        for j in range(i + 1, len(geometry)):
            a = geometry.iloc[i]
            b = geometry.iloc[j]
            if a["shape_type"] == b["shape_type"] == "rectangle":
                ox = min(float(a["xmax"]), float(b["xmax"])) - max(float(a["xmin"]), float(b["xmin"]))
                oy = min(float(a["ymax"]), float(b["ymax"])) - max(float(a["ymin"]), float(b["ymin"]))
                flag = ox > 0 and oy > 0
            else:

                def bounds(row: pd.Series) -> tuple[float, float, float, float]:
                    if row["shape_type"] == "rectangle":
                        return float(row["xmin"]), float(row["xmax"]), float(row["ymin"]), float(row["ymax"])
                    p = _polygon_array(row["polygon"])
                    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())

                ax0, ax1, ay0, ay1 = bounds(a)
                bx0, bx1, by0, by1 = bounds(b)
                x0, x1 = max(ax0, bx0), min(ax1, bx1)
                y0, y1 = max(ay0, by0), min(ay1, by1)
                if x1 <= x0 or y1 <= y0:
                    flag = False
                else:
                    gx = np.linspace(x0, x1, 21)
                    gy = np.linspace(y0, y1, 21)
                    xx, yy = np.meshgrid(gx, gy)
                    px, py = xx.ravel(), yy.ravel()
                    flag = bool(np.any(_aoi_contains(a, px, py) & _aoi_contains(b, px, py)))
            rows.append({"aoi_1": str(a["aoi_id"]), "aoi_2": str(b["aoi_id"]), "overlap": bool(flag)})
    return pd.DataFrame(rows, columns=["aoi_1", "aoi_2", "overlap"])
