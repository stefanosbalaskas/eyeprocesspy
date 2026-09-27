"""Naturalistic/mobile coordinate transforms with explicit geometry provenance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _frame(value: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _require(data: pd.DataFrame, columns: list[str], name: str = "data") -> None:
    missing = [c for c in columns if c not in data]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required column(s): {', '.join(missing)}.")


def pixels_to_visual_angle(
    pixels: Any,
    *,
    pixels_per_mm: float,
    viewing_distance_mm: float,
) -> np.ndarray:
    """Convert signed pixel displacement to visual angle in degrees."""
    ppm, distance = float(pixels_per_mm), float(viewing_distance_mm)
    if ppm <= 0 or distance <= 0 or not np.isfinite([ppm, distance]).all():
        raise EyeProcessValidationError("pixels_per_mm and viewing_distance_mm must be finite and positive.")
    millimetres = np.asarray(pixels, dtype=float) / ppm
    return np.degrees(2 * np.arctan2(millimetres / 2, distance))


def visual_angle_to_pixels(
    degrees: Any,
    *,
    pixels_per_mm: float,
    viewing_distance_mm: float,
) -> np.ndarray:
    """Convert signed visual angle to pixel displacement."""
    ppm, distance = float(pixels_per_mm), float(viewing_distance_mm)
    if ppm <= 0 or distance <= 0 or not np.isfinite([ppm, distance]).all():
        raise EyeProcessValidationError("pixels_per_mm and viewing_distance_mm must be finite and positive.")
    mm = 2 * distance * np.tan(np.radians(np.asarray(degrees, dtype=float)) / 2)
    return mm * ppm


def screen_gaze_to_head_vectors(
    data: Any,
    *,
    x: str = "gaze_x_px",
    y: str = "gaze_y_px",
    screen_width_px: float,
    screen_height_px: float,
    screen_width_mm: float,
    screen_height_mm: float,
    viewing_distance_mm: float,
    origin: str = "top_left",
) -> pd.DataFrame:
    """Map 2D screen gaze into unit direction vectors in a head-centred frame."""
    frame = _frame(data)
    _require(frame, [x, y])
    geometry = np.asarray(
        [screen_width_px, screen_height_px, screen_width_mm, screen_height_mm, viewing_distance_mm],
        float,
    )
    if not np.isfinite(geometry).all() or np.any(geometry <= 0):
        raise EyeProcessValidationError("Screen geometry and viewing distance must be finite and positive.")
    origin = str(origin).lower()
    if origin not in {"top_left", "center"}:
        raise EyeProcessValidationError("origin must be 'top_left' or 'center'.")
    gx = pd.to_numeric(frame[x], errors="coerce").to_numpy(float)
    gy = pd.to_numeric(frame[y], errors="coerce").to_numpy(float)
    if origin == "top_left":
        gx = gx - float(screen_width_px) / 2
        gy = float(screen_height_px) / 2 - gy
    mm_x = gx * float(screen_width_mm) / float(screen_width_px)
    mm_y = gy * float(screen_height_mm) / float(screen_height_px)
    vectors = np.column_stack([mm_x, mm_y, np.full(len(frame), float(viewing_distance_mm))])
    norms = np.linalg.norm(vectors, axis=1)
    vectors = np.divide(vectors, norms[:, None], out=np.full_like(vectors, np.nan), where=norms[:, None] > 0)
    out = frame.copy()
    out[["head_gaze_x", "head_gaze_y", "head_gaze_z"]] = vectors
    out.attrs["eyeprocess_class"] = "eye_head_gaze_vectors"
    out.attrs["coordinate_provenance"] = {
        "source": "screen_pixels",
        "target": "head_unit_vector",
        "origin": origin,
        "screen_width_px": float(screen_width_px),
        "screen_height_px": float(screen_height_px),
        "screen_width_mm": float(screen_width_mm),
        "screen_height_mm": float(screen_height_mm),
        "viewing_distance_mm": float(viewing_distance_mm),
    }
    return out


def align_head_pose_to_gaze(
    gaze: Any,
    pose: Any,
    *,
    gaze_time: str = "timestamp",
    pose_time: str = "timestamp",
    quaternion: tuple[str, str, str, str] = ("qx", "qy", "qz", "qw"),
    position: tuple[str, str, str] = ("px", "py", "pz"),
) -> pd.DataFrame:
    """Interpolate pose to gaze timestamps using Slerp for rotation and linear position."""
    g = _frame(gaze, "gaze")
    p = _frame(pose, "pose")
    _require(g, [gaze_time], "gaze")
    _require(p, [pose_time, *quaternion, *position], "pose")
    if g.empty:
        raise EyeProcessValidationError("gaze must contain at least one timestamp.")
    gt = pd.to_numeric(g[gaze_time], errors="coerce").to_numpy(float)
    pt = pd.to_numeric(p[pose_time], errors="coerce").to_numpy(float)
    if not np.isfinite(gt).all() or not np.isfinite(pt).all() or len(pt) < 2 or np.any(np.diff(pt) <= 0):
        raise EyeProcessValidationError("Pose requires at least two finite, strictly increasing timestamps.")
    if gt.min() < pt.min() or gt.max() > pt.max():
        raise EyeProcessValidationError("Pose does not cover all gaze timestamps; extrapolation is not performed.")
    q = p[list(quaternion)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pos = p[list(position)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(q).all() or not np.isfinite(pos).all():
        raise EyeProcessValidationError("Pose quaternions and positions must be finite.")
    rotations = Slerp(pt, Rotation.from_quat(q))(gt).as_quat()
    interpolated_pos = np.column_stack([np.interp(gt, pt, pos[:, i]) for i in range(3)])
    out = g.copy()
    out[["qx", "qy", "qz", "qw"]] = rotations
    out[["px", "py", "pz"]] = interpolated_pos
    out.attrs["eyeprocess_class"] = "eye_pose_aligned_gaze"
    out.attrs["alignment"] = "slerp_rotation_linear_position_no_extrapolation"
    return out


def transform_head_gaze_to_world(
    data: Any,
    *,
    vector: tuple[str, str, str] = ("head_gaze_x", "head_gaze_y", "head_gaze_z"),
    quaternion: tuple[str, str, str, str] = ("qx", "qy", "qz", "qw"),
    position: tuple[str, str, str] = ("px", "py", "pz"),
) -> pd.DataFrame:
    """Rotate head-centred gaze into world space and retain ray origins."""
    frame = _frame(data)
    _require(frame, [*vector, *quaternion, *position])
    vectors = frame[list(vector)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    quats = frame[list(quaternion)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    positions = frame[list(position)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(vectors).all() or not np.isfinite(quats).all() or not np.isfinite(positions).all():
        raise EyeProcessValidationError("Gaze vectors and aligned pose must be finite.")
    world = Rotation.from_quat(quats).apply(vectors)
    out = frame.copy()
    out[["world_gaze_x", "world_gaze_y", "world_gaze_z"]] = world
    out[["world_origin_x", "world_origin_y", "world_origin_z"]] = positions
    out.attrs["eyeprocess_class"] = "eye_world_gaze_rays"
    return out


def transform_screen_gaze_to_world(
    gaze: Any,
    pose: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience pipeline: screen pixels -> head vector -> pose alignment -> world ray."""
    screen_keys = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k in {"x", "y", "screen_width_px", "screen_height_px", "screen_width_mm", "screen_height_mm", "viewing_distance_mm", "origin"}
    }
    head = screen_gaze_to_head_vectors(gaze, **screen_keys)
    aligned = align_head_pose_to_gaze(head, pose, **kwargs)
    return transform_head_gaze_to_world(aligned)


def propagate_coordinate_uncertainty(
    data: Any,
    *,
    x: str = "gaze_x_px",
    y: str = "gaze_y_px",
    viewing_distance_mm: float,
    viewing_distance_sd_mm: float,
    draws: int = 500,
    seed: int = 1,
    **screen_geometry: Any,
) -> EyeResult:
    """Propagate declared viewing-distance uncertainty into head-centred gaze vectors."""
    draws, seed = int(draws), int(seed)
    if draws < 1 or seed < 0 or float(viewing_distance_sd_mm) < 0:
        raise EyeProcessValidationError("draws must be positive, seed non-negative, and distance SD non-negative.")
    rng = np.random.default_rng(seed)
    distances = rng.normal(float(viewing_distance_mm), float(viewing_distance_sd_mm), draws)
    if np.any(distances <= 0):
        raise EyeProcessValidationError("Viewing-distance uncertainty generated non-positive distances.")
    rows = []
    for draw_id, distance in enumerate(distances, start=1):
        converted = screen_gaze_to_head_vectors(
            data,
            x=x,
            y=y,
            viewing_distance_mm=float(distance),
            **screen_geometry,
        )
        block = converted[["head_gaze_x", "head_gaze_y", "head_gaze_z"]].copy()
        block["sample_id"] = np.arange(1, len(block) + 1)
        block["draw_id"] = draw_id
        block["viewing_distance_mm"] = float(distance)
        rows.append(block)
    return EyeResult(
        {"draws": pd.concat(rows, ignore_index=True), "seed": seed, "n_draws": draws},
        eyeprocess_class="eye_coordinate_uncertainty",
    )


def plot_world_gaze_vectors(data: Any, ax: Any = None) -> Any:
    """Plot world-space gaze directions projected on x/z and y/z."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    frame = _frame(data)
    _require(frame, ["world_gaze_x", "world_gaze_y", "world_gaze_z"])
    axis = plt.subplots()[1] if ax is None else ax
    z = pd.to_numeric(frame.world_gaze_z, errors="coerce").to_numpy(float)
    axis.scatter(pd.to_numeric(frame.world_gaze_x, errors="coerce") / z, pd.to_numeric(frame.world_gaze_y, errors="coerce") / z)
    axis.set_xlabel("World gaze x/z")
    axis.set_ylabel("World gaze y/z")
    axis.set_title("World-space gaze directions")
    axis.eyeprocess_plot_data = frame.copy()
    return axis


__all__ = [
    "align_head_pose_to_gaze",
    "pixels_to_visual_angle",
    "plot_world_gaze_vectors",
    "propagate_coordinate_uncertainty",
    "screen_gaze_to_head_vectors",
    "transform_head_gaze_to_world",
    "transform_screen_gaze_to_world",
    "visual_angle_to_pixels",
]
