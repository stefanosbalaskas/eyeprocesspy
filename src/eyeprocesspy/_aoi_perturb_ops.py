"""Internal AOI geometry perturbation operations."""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._aoi_geometry_primitives import (
    _normalise_pair,
    _offset_convex_polygon,
    _polygon_array,
    _polygon_self_intersects,
    _signed_polygon_area,
    _software_provenance,
    _stable_frame_hash,
)
from ._aoi_geometry_units import _spec_to_px, _spec_value, aoi_perturbation_spec
from ._aoi_geometry_validation import validate_aoi_geometry
from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _geometry_bounds(row: pd.Series) -> tuple[float, float, float, float]:
    if row["shape_type"] == "rectangle":
        return float(row["xmin"]), float(row["xmax"]), float(row["ymin"]), float(row["ymax"])
    p = _polygon_array(row["polygon"])
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())


def _clip_geometry(geometry: pd.DataFrame, width: float, height: float) -> pd.DataFrame:
    out = geometry.copy()
    for i, row in out.iterrows():
        if row["shape_type"] == "rectangle":
            out.loc[i, "xmin"] = min(max(float(row["xmin"]), 0.0), width)
            out.loc[i, "xmax"] = min(max(float(row["xmax"]), 0.0), width)
            out.loc[i, "ymin"] = min(max(float(row["ymin"]), 0.0), height)
            out.loc[i, "ymax"] = min(max(float(row["ymax"]), 0.0), height)
            if not out.loc[i, "xmin"] < out.loc[i, "xmax"] or not out.loc[i, "ymin"] < out.loc[i, "ymax"]:
                raise EyeProcessValidationError(f"Screen clipping collapsed AOI `{row['aoi_id']}`.")
        else:
            p = _polygon_array(row["polygon"]).copy()
            p[:, 0] = np.clip(p[:, 0], 0.0, width)
            p[:, 1] = np.clip(p[:, 1], 0.0, height)
            if _polygon_self_intersects(p) or abs(_signed_polygon_area(p)) <= 1e-12:
                raise EyeProcessValidationError(f"Screen clipping invalidated polygon AOI `{row['aoi_id']}`.")
            out.at[i, "polygon"] = p
    return out


def _apply_boundary_policy(geometry: pd.DataFrame, spec_px: dict[str, Any]) -> pd.DataFrame:
    width = spec_px.get("screen_width_px")
    height = spec_px.get("screen_height_px")
    if width is None or height is None:
        return geometry
    outside = []
    for row in geometry.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        xmin, xmax, ymin, ymax = _geometry_bounds(row_s)
        if xmin < 0 or ymin < 0 or xmax > float(width) or ymax > float(height):
            outside.append(str(row_s["aoi_id"]))
    if not outside:
        return geometry
    policy = spec_px["boundary_policy"]
    msg = "Perturbed AOIs extend beyond the declared screen: " + ", ".join(outside)
    if policy == "error":
        raise EyeProcessValidationError(msg)
    if policy == "warn":
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return geometry
    if policy == "clip":
        warnings.warn(msg + "; clipping was explicitly requested.", RuntimeWarning, stacklevel=3)
        return _clip_geometry(geometry, float(width), float(height))
    return geometry


def _transform_geometry(geometry: pd.DataFrame, spec: Any) -> pd.DataFrame:
    validation = validate_aoi_geometry(geometry)
    out = validation["geometry"].copy()
    sp = _spec_to_px(spec)
    op = sp["operation"]
    if op == "baseline":
        return _apply_boundary_policy(out, sp)
    rng = np.random.default_rng(sp["seed"]) if op == "jitter" else None
    for i, row in out.iterrows():
        shape = row["shape_type"]
        mx, my, tx, ty = sp["margin_x"], sp["margin_y"], sp["translation_x"], sp["translation_y"]
        if op == "jitter":
            tx = float(rng.uniform(-abs(tx if tx else mx), abs(tx if tx else mx))) if rng is not None else 0.0
            ty = float(rng.uniform(-abs(ty if ty else my), abs(ty if ty else my))) if rng is not None else 0.0
        if shape == "rectangle":
            xmin, xmax, ymin, ymax = map(float, (row["xmin"], row["xmax"], row["ymin"], row["ymax"]))
            if op == "dilation":
                xmin -= mx
                xmax += mx
                ymin -= my
                ymax += my
            elif op == "erosion":
                xmin += mx
                xmax -= mx
                ymin += my
                ymax -= my
            elif op in {"translate", "jitter"}:
                xmin += tx
                xmax += tx
                ymin += ty
                ymax += ty
            elif op == "anisotropic_expansion":
                xmin -= mx
                xmax += mx
                ymin -= my
                ymax += my
            if not xmin < xmax or not ymin < ymax:
                raise EyeProcessValidationError(f"Perturbation `{sp['perturbation_id']}` collapsed AOI `{row['aoi_id']}`.")
            out.loc[i, ["xmin", "xmax", "ymin", "ymax"]] = [xmin, xmax, ymin, ymax]
        else:
            poly = _polygon_array(row["polygon"]).copy()
            if op in {"dilation", "erosion"}:
                if str(_spec_value(spec, "unit")) == "deg":
                    dpp = _spec_value(spec, "degrees_per_pixel")
                    if dpp is None:
                        raise EyeProcessValidationError(
                            "Degree-based polygon perturbation lacks degrees-per-pixel provenance."
                        )
                    poly_angle = poly.copy()
                    poly_angle[:, 0] *= float(dpp[0])
                    poly_angle[:, 1] *= float(dpp[1])
                    signed_margin = float(_spec_value(spec, "margin_x"))
                    if op == "erosion":
                        signed_margin *= -1.0
                    poly_angle = _offset_convex_polygon(poly_angle, signed_margin)
                    poly = poly_angle.copy()
                    poly[:, 0] /= float(dpp[0])
                    poly[:, 1] /= float(dpp[1])
                else:
                    poly = _offset_convex_polygon(poly, mx if op == "dilation" else -mx)
            elif op in {"translate", "jitter"}:
                poly[:, 0] += tx
                poly[:, 1] += ty
            elif op == "anisotropic_expansion":
                cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                bx0, bx1 = float(poly[:, 0].min()), float(poly[:, 0].max())
                by0, by1 = float(poly[:, 1].min()), float(poly[:, 1].max())
                width, height = bx1 - bx0, by1 - by0
                if width <= 0 or height <= 0 or width + 2 * mx <= 0 or height + 2 * my <= 0:
                    raise EyeProcessValidationError(f"Anisotropic expansion collapsed polygon AOI `{row['aoi_id']}`.")
                poly[:, 0] = cx + (poly[:, 0] - cx) * ((width + 2 * mx) / width)
                poly[:, 1] = cy + (poly[:, 1] - cy) * ((height + 2 * my) / height)
            if _polygon_self_intersects(poly) or abs(_signed_polygon_area(poly)) <= 1e-12:
                raise EyeProcessValidationError(f"Perturbation `{sp['perturbation_id']}` invalidated polygon AOI `{row['aoi_id']}`.")
            out.at[i, "polygon"] = poly
    out = _apply_boundary_policy(out, sp)
    validate_aoi_geometry(out)
    return out


def dilate_aoi(aois: pd.DataFrame, margin: Any, **kwargs: Any) -> pd.DataFrame:
    mx, my = _normalise_pair(margin, "margin", nonnegative=True)
    spec = aoi_perturbation_spec("dilation", "dilation", margin_x=mx, margin_y=my, **kwargs)
    return _transform_geometry(aois, spec)


def erode_aoi(aois: pd.DataFrame, margin: Any, **kwargs: Any) -> pd.DataFrame:
    mx, my = _normalise_pair(margin, "margin", nonnegative=True)
    spec = aoi_perturbation_spec("erosion", "erosion", margin_x=mx, margin_y=my, **kwargs)
    return _transform_geometry(aois, spec)


def translate_aoi(aois: pd.DataFrame, x: float = 0.0, y: float = 0.0, **kwargs: Any) -> pd.DataFrame:
    spec = aoi_perturbation_spec("translation", "translate", translation_x=x, translation_y=y, **kwargs)
    return _transform_geometry(aois, spec)


def jitter_aoi(aois: pd.DataFrame, x: float, y: float | None = None, *, seed: int = 20260918, **kwargs: Any) -> pd.DataFrame:
    yy = x if y is None else y
    spec = aoi_perturbation_spec("jitter", "jitter", translation_x=x, translation_y=yy, seed=seed, **kwargs)
    return _transform_geometry(aois, spec)


def perturb_aoi_geometry(aois: pd.DataFrame, spec: Any) -> EyeResult:
    """Apply one perturbation and preserve nominal and transformed provenance."""
    nominal = validate_aoi_geometry(aois)
    transformed = _transform_geometry(nominal["geometry"], spec)
    return _result(
        "eye_aoi_perturbation",
        perturbation_id=_spec_value(spec, "perturbation_id"),
        specification=dict(spec),
        nominal_geometry=nominal["geometry"],
        perturbed_geometry=transformed,
        reference_geometry_hash=nominal["source_hash"],
        perturbed_geometry_hash=_stable_frame_hash(transformed),
        software=_software_provenance(),
    )


def create_aoi_perturbation_grid(
    *,
    dilations: Iterable[float] | None = None,
    erosions: Iterable[float] | None = None,
    translations_x: Iterable[float] | None = None,
    translations_y: Iterable[float] | None = None,
    jitters: Iterable[float] | None = None,
    anisotropic: Iterable[Sequence[float]] | None = None,
    unit: str = "px",
    include_baseline: bool = True,
    seed: int = 20260918,
    **geometry_kwargs: Any,
) -> EyeResult:
    """Create a deterministic perturbation grid without silently crossing choices."""
    specs: list[EyeResult] = []
    if include_baseline:
        specs.append(aoi_perturbation_spec("baseline", "baseline", unit=unit, **geometry_kwargs))
    for value in dilations or []:
        specs.append(aoi_perturbation_spec(f"dilate_{value:g}_{unit}", "dilation", margin_x=value, unit=unit, **geometry_kwargs))
    for value in erosions or []:
        specs.append(aoi_perturbation_spec(f"erode_{value:g}_{unit}", "erosion", margin_x=value, unit=unit, **geometry_kwargs))
    for value in translations_x or []:
        specs.append(aoi_perturbation_spec(f"shift_x_{value:g}_{unit}", "translate", translation_x=value, unit=unit, **geometry_kwargs))
    for value in translations_y or []:
        specs.append(aoi_perturbation_spec(f"shift_y_{value:g}_{unit}", "translate", translation_y=value, unit=unit, **geometry_kwargs))
    for idx, value in enumerate(jitters or []):
        specs.append(aoi_perturbation_spec(f"jitter_{value:g}_{unit}_{idx+1}", "jitter", translation_x=value, translation_y=value, unit=unit, seed=seed + idx, **geometry_kwargs))
    for values in anisotropic or []:
        mx, my = _normalise_pair(values, "anisotropic")
        specs.append(aoi_perturbation_spec(f"anisotropic_{mx:g}_{my:g}_{unit}", "anisotropic_expansion", margin_x=mx, margin_y=my, unit=unit, **geometry_kwargs))
    if not specs:
        raise EyeProcessValidationError("The perturbation grid is empty.")
    ids = [str(s["perturbation_id"]) for s in specs]
    if len(set(ids)) != len(ids):
        raise EyeProcessValidationError("Generated perturbation IDs are not unique; use distinct values.")
    table = pd.DataFrame([
        {k: s[k] for k in ("perturbation_id", "operation", "margin_x", "margin_y", "translation_x", "translation_y", "unit", "seed", "boundary_policy")}
        for s in specs
    ])
    return _result("eye_aoi_perturbation_grid", specifications=specs, table=table, software=_software_provenance())


def apply_aoi_perturbation_grid(aois: pd.DataFrame, grid: Any) -> EyeResult:
    specs = grid["specifications"] if isinstance(grid, Mapping) else getattr(grid, "specifications", None)
    if specs is None:
        try:
            specs = grid["specifications"]
        except Exception as exc:
            raise EyeProcessValidationError("`grid` must be created by `create_aoi_perturbation_grid()`.") from exc
    outputs: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for spec in specs:
        pid = str(_spec_value(spec, "perturbation_id"))
        try:
            out = _transform_geometry(aois, spec)
            outputs[pid] = out
            audits.append({"perturbation_id": pid, "status": "completed", "message": None, "geometry_hash": _stable_frame_hash(out)})
        except Exception as exc:
            audits.append({"perturbation_id": pid, "status": "failed", "message": str(exc), "geometry_hash": None})
    return _result(
        "eye_aoi_perturbation_grid_result",
        geometries=outputs,
        audit=pd.DataFrame(audits),
        grid=grid,
        nominal_hash=_stable_frame_hash(validate_aoi_geometry(aois)["geometry"]),
        software=_software_provenance(),
    )
