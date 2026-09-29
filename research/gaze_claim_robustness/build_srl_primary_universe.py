#!/usr/bin/env python3
"""Build the pre-results primary SRL measurement universe.

The primary denominator contains only open/reproducible raw-sample branches
whose decisions are structurally compatible:

4 detector specifications
x 2 eyes
x 3 viewing-distance assumptions
x 2 AOI geometry conventions
x 2 quality rules
= 96 planned specifications.

The released proprietary BeGaze event catalogue is retained as a historical
reference outside this denominator. It is not duplicated across viewing
distance assumptions that do not affect the already-detected vendor events.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PRIMARY_DETECTORS = (
    "ivt_30_100_simple",
    "ivt_40_50_simple",
    "idt_1_100",
    "remodnav_defaults",
)
EYES = ("left", "right")
VIEWING_DISTANCES_CM = (60.0, 65.0, 70.0)
AOI_CONVENTIONS = (
    "exact_quarters",
    "reported_area_center_seam",
)
QUALITY_RULES = (
    "released_sample",
    "trial_80_sensitivity",
)
PRIMARY_MODEL_ID = "primary_nb_glmm"
ESTIMAND_ID = "prompt_transition_rate_ratio"


def _hash(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_primary_universe(research_dir: Path) -> pd.DataFrame:
    """Return the deterministic 96-row planned primary universe."""
    detector_plan = pd.read_csv(research_dir / "srl_detector_plan.csv")
    quality_plan = pd.read_csv(research_dir / "srl_quality_plan.csv")
    model_plan = pd.read_csv(research_dir / "srl_model_plan.csv")

    detector_rows = detector_plan.loc[
        detector_plan["detector_id"].astype(str).isin(PRIMARY_DETECTORS)
    ].copy()
    if set(detector_rows["detector_id"].astype(str)) != set(PRIMARY_DETECTORS):
        missing = sorted(
            set(PRIMARY_DETECTORS) - set(detector_rows["detector_id"].astype(str))
        )
        raise ValueError(f"Primary detector plan is incomplete: {missing}.")

    blocked_detector_status = detector_rows["status"].astype(str).str.contains(
        "parameters_pending|thresholds_frozen_gap_pending",
        regex=True,
    )
    if blocked_detector_status.any():
        bad = detector_rows.loc[
            blocked_detector_status,
            ["detector_id", "status"],
        ].to_dict(orient="records")
        raise ValueError(f"Primary detector definitions are not frozen: {bad}.")

    quality_rows = quality_plan.loc[
        quality_plan["quality_id"].astype(str).isin(QUALITY_RULES)
    ]
    if set(quality_rows["quality_id"].astype(str)) != set(QUALITY_RULES):
        missing = sorted(
            set(QUALITY_RULES) - set(quality_rows["quality_id"].astype(str))
        )
        raise ValueError(f"Primary quality plan is incomplete: {missing}.")

    model = model_plan.loc[
        model_plan["model_id"].astype(str).eq(PRIMARY_MODEL_ID)
    ]
    if len(model) != 1 or str(model.iloc[0]["status"]) != "engine_frozen":
        raise ValueError("Primary glmmTMB model engine is not frozen.")

    rows: list[dict[str, object]] = []
    index = 0
    for detector in PRIMARY_DETECTORS:
        detector_status = str(
            detector_rows.loc[
                detector_rows["detector_id"].astype(str).eq(detector),
                "status",
            ].iloc[0]
        )
        for eye in EYES:
            for distance in VIEWING_DISTANCES_CM:
                for aoi in AOI_CONVENTIONS:
                    for quality in QUALITY_RULES:
                        index += 1
                        payload: dict[str, object] = {
                            "detector_id": detector,
                            "eye": eye,
                            "viewing_distance_cm": distance,
                            "aoi_convention": aoi,
                            "quality_rule": quality,
                            "model_id": PRIMARY_MODEL_ID,
                            "estimand_id": ESTIMAND_ID,
                        }
                        rows.append(
                            {
                                "universe_id": f"srl_u{index:03d}",
                                **payload,
                                "detector_status": detector_status,
                                "requires_dense_regular_timebase": (
                                    detector == "remodnav_defaults"
                                ),
                                "effect_scale": "log_rate_ratio",
                                "planned": True,
                                "specification_hash": _hash(payload),
                            }
                        )

    out = pd.DataFrame(rows)
    if len(out) != 96:
        raise RuntimeError(f"Expected 96 primary specifications; generated {len(out)}.")
    if out["specification_hash"].duplicated().any():
        raise RuntimeError("Primary universe contains duplicate specification hashes.")
    return out


def historical_reference_manifest() -> pd.DataFrame:
    """Return non-denominator historical/reference branches."""
    rows = [
        {
            "reference_id": "vendor_begaze_released",
            "role": "historical_reference_outside_primary_denominator",
            "event_source": "released BeGaze event catalogue",
            "viewing_distance_branch": "not_applicable",
            "eye_semantics": "pending_actual_archive_audit",
            "aoi_reference": "released vendor AOI labels / transition columns",
            "quality_reference": "released retained participant set",
            "reason_outside_denominator": (
                "Already-detected proprietary events are not meaningfully crossed "
                "with raw-sample viewing-distance assumptions; duplicating them would "
                "overweight one historical pipeline."
            ),
        },
        {
            "reference_id": "adaptive_mad_eyeprocesspy",
            "role": "implementation_diagnostic_outside_primary_denominator",
            "event_source": "raw sample stream",
            "viewing_distance_branch": "60/65/70 cm if diagnostic is run",
            "eye_semantics": "left/right explicit",
            "aoi_reference": "same frozen AOI multiverse",
            "quality_reference": "same frozen quality multiverse",
            "reason_outside_denominator": (
                "Algorithm family is available, but its numerical adaptive parameters "
                "were not frozen as an externally validated named detector."
            ),
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--research-dir",
        type=Path,
        default=Path("research/gaze_claim_robustness"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_primary_universe"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe = build_primary_universe(args.research_dir)
    references = historical_reference_manifest()
    universe.to_csv(args.output_dir / "srl_primary_universe.csv", index=False)
    references.to_csv(
        args.output_dir / "srl_reference_branches.csv",
        index=False,
    )

    print(
        f"Frozen primary universe: {len(universe)} specifications; "
        f"unique hashes={universe['specification_hash'].nunique()}."
    )
    print(
        universe.groupby(
            ["detector_id", "eye"],
            sort=True,
        ).size().rename("specifications").reset_index().to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
