"""Unit conversion and perturbation-spec contracts for AOI uncertainty analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._aoi_geometry_primitives import _finite_scalar, _normalise_pair, _software_provenance
from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _screen_geometry(
    *,
    screen_width_px: int,
    screen_height_px: int,
    viewing_distance: float,
    physical_screen_size: Sequence[float],
) -> tuple[float, float, float, float]:
    sw = _finite_scalar(screen_width_px, "screen_width_px", nonnegative=True)
    sh = _finite_scalar(screen_height_px, "screen_height_px", nonnegative=True)
    if sw <= 0 or sh <= 0:
        raise EyeProcessValidationError("Screen pixel dimensions must be positive.")
    distance = _finite_scalar(viewing_distance, "viewing_distance", nonnegative=True)
    if distance <= 0:
        raise EyeProcessValidationError("`viewing_distance` must be positive.")
    physical = _normalise_pair(physical_screen_size, "physical_screen_size", nonnegative=True)
    if physical[0] <= 0 or physical[1] <= 0:
        raise EyeProcessValidationError("Physical screen dimensions must be positive.")
    deg_per_px_x = math.degrees(2 * math.atan((physical[0] / sw) / (2 * distance)))
    deg_per_px_y = math.degrees(2 * math.atan((physical[1] / sh) / (2 * distance)))
    return sw, sh, deg_per_px_x, deg_per_px_y


def convert_aoi_margin_to_degrees(
    margin_pixels: Any,
    *,
    screen_width_px: int,
    screen_height_px: int,
    viewing_distance: float,
    physical_screen_size: Sequence[float],
) -> tuple[float, float]:
    """Convert pixel margins to degrees of visual angle on x and y axes."""
    mx, my = _normalise_pair(margin_pixels, "margin_pixels", nonnegative=False)
    _, _, dx, dy = _screen_geometry(
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        viewing_distance=viewing_distance,
        physical_screen_size=physical_screen_size,
    )
    return (mx * dx, my * dy)


def convert_aoi_margin_to_pixels(
    margin_degrees: Any,
    *,
    screen_width_px: int,
    screen_height_px: int,
    viewing_distance: float,
    physical_screen_size: Sequence[float],
) -> tuple[float, float]:
    """Convert visual-angle margins to pixels on x and y axes."""
    mx, my = _normalise_pair(margin_degrees, "margin_degrees", nonnegative=False)
    _, _, dx, dy = _screen_geometry(
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        viewing_distance=viewing_distance,
        physical_screen_size=physical_screen_size,
    )
    return (mx / dx, my / dy)


def aoi_perturbation_spec(
    perturbation_id: str,
    operation: str = "baseline",
    *,
    margin_x: float = 0.0,
    margin_y: float | None = None,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    unit: str = "px",
    screen_width_px: int | None = None,
    screen_height_px: int | None = None,
    viewing_distance: float | None = None,
    physical_screen_size: Sequence[float] | None = None,
    degrees_per_pixel: Sequence[float] | None = None,
    seed: int | None = None,
    boundary_policy: str = "warn",
) -> EyeResult:
    """Create a validated, reproducible AOI perturbation specification."""
    if not isinstance(perturbation_id, str) or not perturbation_id.strip():
        raise EyeProcessValidationError("`perturbation_id` must be a non-empty string.")
    operation = str(operation).lower()
    operations = {"baseline", "dilation", "erosion", "translate", "jitter", "anisotropic_expansion"}
    if operation not in operations:
        raise EyeProcessValidationError(f"Unsupported perturbation operation `{operation}`.")
    unit = str(unit).lower()
    if unit not in {"px", "deg"}:
        raise EyeProcessValidationError("`unit` must be 'px' or 'deg'.")
    boundary_policy = str(boundary_policy).lower()
    if boundary_policy not in {"warn", "clip", "error", "allow"}:
        raise EyeProcessValidationError("`boundary_policy` must be warn, clip, error, or allow.")
    mx = _finite_scalar(margin_x, "margin_x")
    my = mx if margin_y is None else _finite_scalar(margin_y, "margin_y")
    tx = _finite_scalar(translation_x, "translation_x")
    ty = _finite_scalar(translation_y, "translation_y")
    if operation == "dilation" and (mx < 0 or my < 0):
        raise EyeProcessValidationError("Dilation margins must be non-negative.")
    if operation == "erosion" and (mx < 0 or my < 0):
        raise EyeProcessValidationError(
            "Erosion margins must be non-negative; erosion direction is implied by the operation."
        )
    if operation in {"dilation", "erosion"} and not math.isclose(mx, my, rel_tol=0, abs_tol=1e-12):
        raise EyeProcessValidationError(
            "Polygon-safe dilation/erosion uses a uniform margin; use anisotropic_expansion for different x/y margins."
        )
    resolved_dpp: tuple[float, float] | None = None
    if degrees_per_pixel is not None:
        resolved_dpp = _normalise_pair(degrees_per_pixel, "degrees_per_pixel", nonnegative=True)
        if min(resolved_dpp) <= 0:
            raise EyeProcessValidationError("`degrees_per_pixel` values must be positive.")
    elif all(
        v is not None
        for v in (screen_width_px, screen_height_px, viewing_distance, physical_screen_size)
    ):
        assert screen_width_px is not None
        assert screen_height_px is not None
        assert viewing_distance is not None
        assert physical_screen_size is not None

        _, _, dx, dy = _screen_geometry(
            screen_width_px=int(screen_width_px),
            screen_height_px=int(screen_height_px),
            viewing_distance=float(viewing_distance),
            physical_screen_size=physical_screen_size,
        )
        resolved_dpp = (dx, dy)
    if unit == "deg" and resolved_dpp is None:
        raise EyeProcessValidationError(
            "Degree-based perturbations require `degrees_per_pixel` or complete screen geometry and viewing distance."
        )
    if seed is not None:
        if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
            raise EyeProcessValidationError("`seed` must be a non-negative integer or None.")
        seed = int(seed)
    return _result(
        "eye_aoi_perturbation_spec",
        perturbation_id=perturbation_id,
        operation=operation,
        margin_x=mx,
        margin_y=my,
        translation_x=tx,
        translation_y=ty,
        unit=unit,
        screen_width_px=None if screen_width_px is None else int(screen_width_px),
        screen_height_px=None if screen_height_px is None else int(screen_height_px),
        viewing_distance=viewing_distance,
        physical_screen_size=None
        if physical_screen_size is None
        else tuple(float(v) for v in physical_screen_size),
        degrees_per_pixel=resolved_dpp,
        seed=seed,
        boundary_policy=boundary_policy,
        software=_software_provenance(),
    )


def _spec_value(spec: Any, key: str) -> Any:
    if isinstance(spec, Mapping):
        return spec[key]
    try:
        return spec[key]
    except Exception as exc:
        raise EyeProcessValidationError(
            "Perturbation specifications must be mapping-like objects created by `aoi_perturbation_spec()`."
        ) from exc


def _spec_to_px(spec: Any) -> dict[str, Any]:
    unit = str(_spec_value(spec, "unit"))
    mx = float(_spec_value(spec, "margin_x"))
    my = float(_spec_value(spec, "margin_y"))
    tx = float(_spec_value(spec, "translation_x"))
    ty = float(_spec_value(spec, "translation_y"))
    if unit == "deg":
        dpp = _spec_value(spec, "degrees_per_pixel")
        if dpp is None:
            raise EyeProcessValidationError(
                "Degree perturbation lacks `degrees_per_pixel` provenance."
            )
        dx, dy = dpp
        mx, my, tx, ty = mx / dx, my / dy, tx / dx, ty / dy
    return {
        "perturbation_id": _spec_value(spec, "perturbation_id"),
        "operation": _spec_value(spec, "operation"),
        "margin_x": mx,
        "margin_y": my,
        "translation_x": tx,
        "translation_y": ty,
        "seed": _spec_value(spec, "seed"),
        "screen_width_px": _spec_value(spec, "screen_width_px"),
        "screen_height_px": _spec_value(spec, "screen_height_px"),
        "boundary_policy": _spec_value(spec, "boundary_policy"),
        "source_unit": unit,
    }
