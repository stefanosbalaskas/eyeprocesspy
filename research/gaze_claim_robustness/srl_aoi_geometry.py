#!/usr/bin/env python3
"""Independent SRL quartile AOI geometry branches.

The source reports four AOIs per learning slide, a 1600 x 900 display, and an
AOI area of 359,550 px. It also describes transitions between different
quartiles of the slide.

Two pre-results geometry branches are represented here:

1. exact_quarters: mathematical 800 x 450 quarters (360,000 px each).
2. reported_area_center_seam: 799 x 450 rectangles (359,550 px each),
   symmetrically leaving a two-pixel-wide vertical seam around the screen
   midpoint.

The second branch is an explicit inference from the published area, not a claim
that these were the exact original BeGaze coordinates. Released vendor AOI
labels remain reference evidence only and are never used to choose the geometry
that produces a preferred focal effect.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

import eyeprocesspy as ep

SCREEN_WIDTH_PX = 1600.0
SCREEN_HEIGHT_PX = 900.0

PIXEL_CONVENTIONS: dict[str, tuple[tuple[str, float, float, float, float], ...]] = {
    "exact_quarters": (
        ("Q1_top_left", 0.0, 0.0, 800.0, 450.0),
        ("Q2_top_right", 800.0, 0.0, 800.0, 450.0),
        ("Q3_bottom_left", 0.0, 450.0, 800.0, 450.0),
        ("Q4_bottom_right", 800.0, 450.0, 800.0, 450.0),
    ),
    "reported_area_center_seam": (
        ("Q1_top_left", 0.0, 0.0, 799.0, 450.0),
        ("Q2_top_right", 801.0, 0.0, 799.0, 450.0),
        ("Q3_bottom_left", 0.0, 450.0, 799.0, 450.0),
        ("Q4_bottom_right", 801.0, 450.0, 799.0, 450.0),
    ),
}


def geometry_manifest() -> pd.DataFrame:
    """Describe the frozen pre-results AOI geometry branches."""
    rows: list[dict[str, object]] = []
    for convention, boxes in PIXEL_CONVENTIONS.items():
        for label, x, y, width, height in boxes:
            rows.append(
                {
                    "convention": convention,
                    "quartile": label,
                    "x_px": x,
                    "y_px": y,
                    "width_px": width,
                    "height_px": height,
                    "area_px": width * height,
                    "source_role": (
                        "mathematical_quarter_sensitivity"
                        if convention == "exact_quarters"
                        else "published_area_inferred_reference"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _coordinate_context(dataset: ep.EyeDataset) -> tuple[str, float | None]:
    samples = dataset["gaze_samples"]
    if samples.empty:
        raise ValueError("dataset contains no gaze samples.")
    spaces = sorted(
        {
            str(value)
            for value in samples["coordinate_space_id"].dropna().unique()
        }
    )
    if len(spaces) != 1:
        raise ValueError(
            "Exactly one active gaze coordinate space is required; "
            f"found {spaces}."
        )
    coordinate_space_id = spaces[0]

    if coordinate_space_id == "srl_display_pixels_top_left":
        return coordinate_space_id, None

    branch = dataset.vendor_metadata.get("angular_coordinate_branch")
    if not isinstance(branch, dict):
        raise ValueError(
            "Angular SRL AOI registration requires angular_coordinate_branch provenance."
        )
    ppd = float(branch.get("pixels_per_degree", np.nan))
    if not np.isfinite(ppd) or ppd <= 0:
        raise ValueError("pixels_per_degree must be finite and > 0.")
    if int(branch.get("screen_width_px", -1)) != int(SCREEN_WIDTH_PX):
        raise ValueError("Angular branch screen width disagrees with SRL source geometry.")
    if int(branch.get("screen_height_px", -1)) != int(SCREEN_HEIGHT_PX):
        raise ValueError("Angular branch screen height disagrees with SRL source geometry.")
    return coordinate_space_id, ppd


def _box_in_active_space(
    box: tuple[str, float, float, float, float],
    *,
    ppd: float | None,
) -> tuple[str, float, float, float, float]:
    label, x, y, width, height = box
    if ppd is None:
        return box
    return (
        label,
        (x - SCREEN_WIDTH_PX / 2.0) / ppd,
        (y - SCREEN_HEIGHT_PX / 2.0) / ppd,
        width / ppd,
        height / ppd,
    )


def register_srl_quartile_aois(
    dataset: ep.EyeDataset,
    *,
    stimulus_id: str,
    convention: str,
    overwrite: bool = False,
) -> ep.EyeDataset:
    """Register four independent quartile AOIs for one SRL learning slide."""
    if not ep.is_eye_dataset(dataset):
        raise TypeError("dataset must be an EyeDataset.")
    if convention not in PIXEL_CONVENTIONS:
        raise ValueError(
            "convention must be one of: " + ", ".join(sorted(PIXEL_CONVENTIONS))
        )
    stimulus = str(stimulus_id).strip()
    if not stimulus:
        raise ValueError("stimulus_id must be non-empty.")

    coordinate_space_id, ppd = _coordinate_context(dataset)
    aois = []
    for box in PIXEL_CONVENTIONS[convention]:
        label, x, y, width, height = _box_in_active_space(box, ppd=ppd)
        aois.append(
            ep.new_aoi(
                f"{stimulus}_{convention}_{label}",
                aoi_name=label,
                stimulus_id=stimulus,
                shape="rectangle",
                x=x,
                y=y,
                width=width,
                height=height,
                coordinate_space_id=coordinate_space_id,
                source=f"SRL independent geometry: {convention}",
            )
        )

    out = ep.register_aois(dataset, *aois, overwrite=overwrite)
    return ep.add_provenance(
        out,
        "register_srl_quartile_aois",
        "aoi_definitions",
        (
            f"stimulus_id={stimulus};convention={convention};"
            f"coordinate_space_id={coordinate_space_id};"
            f"pixels_per_degree={ppd if ppd is not None else 'not_applicable'};"
            "vendor_aoi_labels_used=false"
        ),
        software="eyeprocesspy research adapter",
        reversible=True,
    )


def register_srl_quartile_aois_many(
    dataset: ep.EyeDataset,
    *,
    stimulus_ids: Iterable[str],
    convention: str,
) -> ep.EyeDataset:
    """Register the same geometry convention independently for multiple tasks."""
    out = dataset
    seen: set[str] = set()
    for value in stimulus_ids:
        stimulus = str(value).strip()
        if not stimulus or stimulus in seen:
            continue
        out = register_srl_quartile_aois(
            out,
            stimulus_id=stimulus,
            convention=convention,
            overwrite=False,
        )
        seen.add(stimulus)
    if not seen:
        raise ValueError("Supply at least one non-empty stimulus_id.")
    return out


__all__ = [
    "PIXEL_CONVENTIONS",
    "geometry_manifest",
    "register_srl_quartile_aois",
    "register_srl_quartile_aois_many",
]
