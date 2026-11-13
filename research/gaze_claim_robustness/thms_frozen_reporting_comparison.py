#!/usr/bin/env python3
"""Post-freeze descriptive comparison of frozen SRL report views.

No model fitting and no alteration of the original 144-specification universe.
Reference settings are illustrative, selected for transparent reporting rather
than a retrospectively designated 'primary pipeline'. The single-pipeline and
detector-only views are nested subsets of the same frozen result ledger.

Usage (from repository root):
  python research/gaze_claim_robustness/thms_frozen_reporting_comparison.py

Requires pandas; reads only the frozen published result snapshot.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "srl_paper_figure_snapshot.csv"
DEST = ROOT / "results" / "thms_reporting_comparison.csv"
DETAIL = ROOT / "results" / "thms_detector_only_snapshot.csv"

# Exact deterministic order from build_srl_primary_universe.py.
DETECTORS = ("ivt_30_100_simple", "ivt_40_50_simple", "idt_1_100")
EYES = ("left", "right")
DISTANCES = (60.0, 65.0, 70.0)
AOIS = ("exact_quarters", "reported_area_center_seam")
QUALITY = ("released_sample", "trial_80_sensitivity")
COHORTS = ("exact_raw82", "nominal250_77")

# Post-freeze illustration, *not* an originally preferred pipeline.
REFERENCE = {
    "eye": "left",
    "viewing_distance_cm": 65.0,
    "aoi_convention": "exact_quarters",
    "quality_rule": "released_sample",
    "cohort_id": "exact_raw82",
}
REFERENCE_DETECTOR = "ivt_40_50_simple"

def build_views() -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = pd.read_csv(SOURCE)
    assert len(frozen) == 144
    rows = [
        dict(
            universe_id=f"srl_u{i:03d}",
            detector_id=d, eye=e, viewing_distance_cm=v,
            aoi_convention=a, quality_rule=q, cohort_id=c,
        )
        for i, (d, e, v, a, q, c) in enumerate(
            product(DETECTORS, EYES, DISTANCES, AOIS, QUALITY, COHORTS), 1
        )
    ]
    design = pd.DataFrame(rows)
    full = design.merge(
        frozen, on=["universe_id", "detector_id", "viewing_distance_cm"],
        how="left", validate="one_to_one",
    )
    assert len(full) == 144 and full["estimate"].notna().all()
    assert (full["CI_lower"] <= 0).all() and (full["CI_upper"] >= 0).all()
    subset = full.loc[
        (full[list(REFERENCE)] == pd.Series(REFERENCE)).all(axis=1)
    ].sort_values("detector_id")
    assert len(subset) == 3 and set(subset["detector_id"]) == set(DETECTORS)
    single = subset.loc[subset["detector_id"] == REFERENCE_DETECTOR]
    assert len(single) == 1 and single.iloc[0]["universe_id"] == "srl_u057"

    outputs = []
    for label, data in [
        ("Illustrative single pipeline", single),
        ("Detector-only alternatives", subset),
        ("Full frozen cross-stage universe", full),
    ]:
        outputs.append({
            "view": label,
            "n_specifications": len(data),
            "positive": int((data["estimate"] > 0).sum()),
            "negative": int((data["estimate"] < 0).sum()),
            "median_rate_ratio": float(data["rate_ratio"].median()),
            "min_rate_ratio": float(data["rate_ratio"].min()),
            "max_rate_ratio": float(data["rate_ratio"].max()),
            "intervals_including_null": int(
                ((data["CI_lower"] <= 0) & (data["CI_upper"] >= 0)).sum()
            ),
        })
    return pd.DataFrame(outputs), subset

def main() -> int:
    summary, detail = build_views()
    summary.to_csv(DEST, index=False, float_format="%.12g")
    detail.to_csv(DETAIL, index=False, float_format="%.12g")
    print(summary.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
