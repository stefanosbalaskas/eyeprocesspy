#!/usr/bin/env python3
"""Frozen MCFW normalized-display to visual-degree conversion.

The public MCFW-Gaze descriptor reports a 1920 x 1080 display with a
31.0 x 17.5 cm physical extent and an approximately 65 cm viewing distance.
The primary validation uses that single nominal geometry; it does not invent
participant-specific distances or a distance sensitivity range.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import eyeprocesspy as ep

SCREEN_WIDTH_PX = 1920
SCREEN_HEIGHT_PX = 1080
SCREEN_WIDTH_MM = 310.0
SCREEN_HEIGHT_MM = 175.0
VIEWING_DISTANCE_MM = 650.0


def geometry_manifest() -> pd.DataFrame:
    """Return the frozen source-reported nominal display geometry."""
    x_ppm = SCREEN_WIDTH_PX / SCREEN_WIDTH_MM
    y_ppm = SCREEN_HEIGHT_PX / SCREEN_HEIGHT_MM
    half_width_deg = float(
        ep.pixels_to_visual_angle(
            SCREEN_WIDTH_PX / 2.0,
            pixels_per_mm=x_ppm,
            viewing_distance_mm=VIEWING_DISTANCE_MM,
        )
    )
    half_height_deg = float(
        ep.pixels_to_visual_angle(
            SCREEN_HEIGHT_PX / 2.0,
            pixels_per_mm=y_ppm,
            viewing_distance_mm=VIEWING_DISTANCE_MM,
        )
    )
    return pd.DataFrame(
        [
            {
                "screen_width_px": SCREEN_WIDTH_PX,
                "screen_height_px": SCREEN_HEIGHT_PX,
                "screen_width_mm": SCREEN_WIDTH_MM,
                "screen_height_mm": SCREEN_HEIGHT_MM,
                "viewing_distance_mm": VIEWING_DISTANCE_MM,
                "pixels_per_mm_x": x_ppm,
                "pixels_per_mm_y": y_ppm,
                "horizontal_extent_deg": 2.0 * half_width_deg,
                "vertical_extent_deg": 2.0 * half_height_deg,
                "viewing_distance_status": "source_reported_approximate",
                "primary_distance_branches": 1,
            }
        ]
    )


def convert_mcfw_normalized_to_degrees(dataset: ep.EyeDataset) -> ep.EyeDataset:
    """Map normalized display coordinates to signed visual degrees.

    The conversion is axis-specific because the source-reported horizontal and
    vertical physical pixel densities are not numerically identical. Source
    normalized coordinates are retained. Sample validity is unchanged.
    """
    if not ep.is_eye_dataset(dataset):
        raise TypeError("dataset must be an EyeDataset.")
    gaze = dataset["gaze_samples"]
    if gaze.empty:
        raise ValueError("dataset contains no gaze samples.")

    recording = dataset["recordings"]
    if recording.empty:
        raise ValueError("dataset contains no recording metadata.")
    widths = (
        pd.to_numeric(recording["screen_width_px"], errors="coerce")
        .dropna()
        .unique()
    )
    heights = (
        pd.to_numeric(recording["screen_height_px"], errors="coerce")
        .dropna()
        .unique()
    )
    if len(widths) != 1 or int(widths[0]) != SCREEN_WIDTH_PX:
        raise ValueError(
            f"MCFW primary geometry requires screen_width_px={SCREEN_WIDTH_PX}."
        )
    if len(heights) != 1 or int(heights[0]) != SCREEN_HEIGHT_PX:
        raise ValueError(
            f"MCFW primary geometry requires screen_height_px={SCREEN_HEIGHT_PX}."
        )

    out = dataset.copy()
    samples = out["gaze_samples"].copy()
    source_x = pd.to_numeric(samples["gaze_x"], errors="coerce").to_numpy(float)
    source_y = pd.to_numeric(samples["gaze_y"], errors="coerce").to_numpy(float)
    samples["source_gaze_x_normalized"] = source_x
    samples["source_gaze_y_normalized"] = source_y

    x_px_from_center = (source_x - 0.5) * SCREEN_WIDTH_PX
    y_px_from_center = (source_y - 0.5) * SCREEN_HEIGHT_PX
    x_ppm = SCREEN_WIDTH_PX / SCREEN_WIDTH_MM
    y_ppm = SCREEN_HEIGHT_PX / SCREEN_HEIGHT_MM

    samples["gaze_x"] = ep.pixels_to_visual_angle(
        x_px_from_center,
        pixels_per_mm=x_ppm,
        viewing_distance_mm=VIEWING_DISTANCE_MM,
    )
    samples["gaze_y"] = ep.pixels_to_visual_angle(
        y_px_from_center,
        pixels_per_mm=y_ppm,
        viewing_distance_mm=VIEWING_DISTANCE_MM,
    )

    manifest = geometry_manifest().iloc[0]
    coordinate_space_id = "mcfw_display_degrees_center_65cm_nominal"
    samples["coordinate_space_id"] = coordinate_space_id
    out["gaze_samples"] = samples

    spaces = out["coordinate_spaces"].copy()
    new_space = ep.new_coordinate_space(
        coordinate_space_id,
        "custom",
        origin="center",
        x_unit="degrees",
        y_unit="degrees",
        width=float(manifest["horizontal_extent_deg"]),
        height=float(manifest["vertical_extent_deg"]),
        reference_object="MCFW source-reported nominal display geometry",
    )
    out["coordinate_spaces"] = pd.concat(
        [spaces, new_space],
        ignore_index=True,
        sort=False,
    )

    out.vendor_metadata = dict(out.vendor_metadata)
    out.vendor_metadata["angular_coordinate_branch"] = {
        "screen_width_px": SCREEN_WIDTH_PX,
        "screen_height_px": SCREEN_HEIGHT_PX,
        "screen_width_mm": SCREEN_WIDTH_MM,
        "screen_height_mm": SCREEN_HEIGHT_MM,
        "viewing_distance_mm": VIEWING_DISTANCE_MM,
        "viewing_distance_status": "source_reported_approximate",
        "pixels_per_mm_x": x_ppm,
        "pixels_per_mm_y": y_ppm,
        "source_normalized_coordinates_retained": True,
        "distance_sensitivity_branches": 1,
    }

    return ep.add_provenance(
        out,
        "convert_mcfw_normalized_to_degrees",
        "gaze_samples",
        (
            f"screen={SCREEN_WIDTH_PX}x{SCREEN_HEIGHT_PX}px;"
            f"physical={SCREEN_WIDTH_MM:g}x{SCREEN_HEIGHT_MM:g}mm;"
            f"viewing_distance_mm={VIEWING_DISTANCE_MM:g};"
            f"pixels_per_mm_x={x_ppm:.12g};pixels_per_mm_y={y_ppm:.12g};"
            "source_normalized_coordinates_retained=true;"
            "distance_status=source_reported_approximate"
        ),
        software="eyeprocesspy research adapter",
        reversible=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args()
    manifest = geometry_manifest()
    if args.print_manifest:
        print(manifest.to_string(index=False))
    else:
        print(manifest.to_csv(index=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
