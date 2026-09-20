"""Reproducible standardized gaze-quality example with synthetic validation data."""

from __future__ import annotations

import eyeprocesspy as ep


def main() -> None:
    data = ep.simulate_gaze_quality_calibration(seed=20260918, samples_per_target=12)
    report = ep.create_gaze_quality_report(
        data,
        by=["profile", "target_id"],
        valid="valid",
        missing_reason="missing_reason",
        nominal_sampling_hz=60,
        bcea_probability=0.68,
        thresholds={
            "accuracy_mean": {"max": 1.0},
            "valid_sample_fraction": {"min": 0.80},
        },
        preprocessing_spec="synthetic raw validation samples; no interpolation",
        quality_rules="study-defined review thresholds; no automatic exclusion",
    )

    columns = [
        "profile",
        "target_id",
        "accuracy_mean",
        "precision_rms_s2s",
        "precision_sd",
        "bcea",
        "effective_sampling_hz",
        "data_loss_fraction",
        "quality_flags",
        "review_required",
    ]
    print(report[columns].to_string(index=False))
    print()
    print(ep.report_gaze_quality(report))


if __name__ == "__main__":
    main()
