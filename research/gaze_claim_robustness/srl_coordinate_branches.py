#!/usr/bin/env python3
"""Explicit SRL pixel-to-visual-degree coordinate branches.

The SRL descriptor reports:
- 22-inch display;
- 1600 x 900 resolution;
- participant viewing distance approximately 60-70 cm.

It does not report a participant-specific viewing distance or physical panel
width/height. This module therefore makes the geometric assumptions explicit
and treats viewing distance as a robustness decision rather than silently
choosing one value.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import eyeprocesspy as ep

DEFAULT_DIAGONAL_INCHES = 22.0
DEFAULT_WIDTH_PX = 1600
DEFAULT_HEIGHT_PX = 900
DEFAULT_DISTANCE_BRANCHES_CM = (60.0, 65.0, 70.0)


def display_geometry(
    *,
    diagonal_inches: float = DEFAULT_DIAGONAL_INCHES,
    width_px: int = DEFAULT_WIDTH_PX,
    height_px: int = DEFAULT_HEIGHT_PX,
) -> dict[str, float]:
    """Derive physical display geometry from explicit diagonal/aspect assumptions."""
    diagonal = float(diagonal_inches)
    width = int(width_px)
    height = int(height_px)
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("diagonal_inches must be finite and > 0.")
    if width <= 0 or height <= 0:
        raise ValueError("width_px and height_px must be positive.")

    aspect_norm = math.hypot(width, height)
    width_inches = diagonal * width / aspect_norm
    height_inches = diagonal * height / aspect_norm
    width_cm = width_inches * 2.54
    height_cm = height_inches * 2.54
    return {
        "diagonal_inches": diagonal,
        "width_px": float(width),
        "height_px": float(height),
        "width_cm": width_cm,
        "height_cm": height_cm,
        "pixels_per_cm_x": width / width_cm,
        "pixels_per_cm_y": height / height_cm,
    }


def pixels_per_degree(
    viewing_distance_cm: float,
    *,
    diagonal_inches: float = DEFAULT_DIAGONAL_INCHES,
    width_px: int = DEFAULT_WIDTH_PX,
    height_px: int = DEFAULT_HEIGHT_PX,
) -> float:
    """Return px/degree under the declared display and viewing-distance geometry."""
    distance = float(viewing_distance_cm)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("viewing_distance_cm must be finite and > 0.")

    geometry = display_geometry(
        diagonal_inches=diagonal_inches,
        width_px=width_px,
        height_px=height_px,
    )
    if not np.isclose(
        geometry["pixels_per_cm_x"],
        geometry["pixels_per_cm_y"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Pixel density differs across axes under the declared geometry; "
            "use an axis-specific angular conversion instead."
        )

    cm_per_degree = 2.0 * distance * math.tan(math.radians(0.5))
    return float(geometry["pixels_per_cm_x"] * cm_per_degree)


def convert_srl_pixels_to_degrees(
    dataset: ep.EyeDataset,
    *,
    viewing_distance_cm: float,
    diagonal_inches: float = DEFAULT_DIAGONAL_INCHES,
) -> ep.EyeDataset:
    """Create a reversible angular-coordinate branch without altering validity."""
    if not ep.is_eye_dataset(dataset):
        raise TypeError("dataset must be an EyeDataset.")
    gaze = dataset["gaze_samples"].copy()
    if gaze.empty:
        raise ValueError("dataset contains no gaze samples.")

    recording = dataset["recordings"]
    if recording.empty:
        raise ValueError("dataset contains no recording metadata.")
    width_values = (
        pd.to_numeric(recording["screen_width_px"], errors="coerce")
        .dropna()
        .unique()
    )
    height_values = (
        pd.to_numeric(recording["screen_height_px"], errors="coerce")
        .dropna()
        .unique()
    )
    if len(width_values) != 1 or len(height_values) != 1:
        raise ValueError(
            "All recordings must share one screen width and one screen height."
        )
    width_px = int(width_values[0])
    height_px = int(height_values[0])

    ppd = pixels_per_degree(
        viewing_distance_cm,
        diagonal_inches=diagonal_inches,
        width_px=width_px,
        height_px=height_px,
    )

    out = dataset.copy()
    gaze = out["gaze_samples"].copy()
    gaze["source_gaze_x_px"] = pd.to_numeric(gaze["gaze_x"], errors="coerce")
    gaze["source_gaze_y_px"] = pd.to_numeric(gaze["gaze_y"], errors="coerce")
    gaze["gaze_x"] = (gaze["source_gaze_x_px"] - width_px / 2.0) / ppd
    gaze["gaze_y"] = (gaze["source_gaze_y_px"] - height_px / 2.0) / ppd

    distance = float(viewing_distance_cm)
    coordinate_space_id = f"srl_display_degrees_center_{distance:g}cm"
    gaze["coordinate_space_id"] = coordinate_space_id
    out["gaze_samples"] = gaze

    width_deg = width_px / ppd
    height_deg = height_px / ppd
    spaces = out["coordinate_spaces"].copy()
    new_space = ep.new_coordinate_space(
        coordinate_space_id,
        "custom",
        origin="center",
        x_unit="degrees",
        y_unit="degrees",
        width=width_deg,
        height=height_deg,
        reference_object="SRL display geometry branch",
    )
    out["coordinate_spaces"] = pd.concat(
        [spaces, new_space],
        ignore_index=True,
        sort=False,
    )

    out.vendor_metadata = dict(out.vendor_metadata)
    out.vendor_metadata["angular_coordinate_branch"] = {
        "viewing_distance_cm": distance,
        "display_diagonal_inches": float(diagonal_inches),
        "screen_width_px": width_px,
        "screen_height_px": height_px,
        "physical_aspect_assumption": "same as pixel aspect ratio",
        "pixels_per_degree": ppd,
        "source_pixel_coordinates_retained": True,
    }

    return ep.add_provenance(
        out,
        "convert_srl_pixels_to_degrees",
        "gaze_samples",
        (
            f"viewing_distance_cm={distance:g};"
            f"display_diagonal_inches={float(diagonal_inches):g};"
            f"width_px={width_px};height_px={height_px};"
            f"pixels_per_degree={ppd:.12g};"
            "physical_aspect=pixel_aspect;"
            "source_pixels_retained=true"
        ),
        software="eyeprocesspy research adapter",
        reversible=True,
    )


def coordinate_branch_manifest() -> pd.DataFrame:
    """Return the prespecified viewing-distance sensitivity branches."""
    rows = []
    for distance in DEFAULT_DISTANCE_BRANCHES_CM:
        ppd = pixels_per_degree(distance)
        rows.append(
            {
                "viewing_distance_cm": distance,
                "pixels_per_degree": ppd,
                "degrees_per_pixel": 1.0 / ppd,
                "source_status": (
                    "lower_bound_reported"
                    if distance == 60.0
                    else "midpoint_sensitivity"
                    if distance == 65.0
                    else "upper_bound_reported"
                ),
                "branch_role": "measurement_geometry_sensitivity",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args()
    manifest = coordinate_branch_manifest()
    if args.print_manifest:
        print(manifest.to_string(index=False))
    else:
        print(manifest.to_csv(index=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
