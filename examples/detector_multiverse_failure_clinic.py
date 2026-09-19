"""Deterministic failure clinic for event-detector sensitivity analysis.

The script deliberately triggers an external-detector failure, one quality
exclusion, one missing outcome, and one invalid duplicate-term model callback.
Every state is retained in an audit table rather than repaired silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import eyeprocesspy as ep


def _ivt():
    return ep.define_event_detector_spec(
        "ivt30",
        "ivt",
        velocity_threshold=30,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
    )


def _failing_external(*, data, spec):
    raise RuntimeError("deliberate external detector failure")


def _tidy_callback(data, spec):
    return pd.DataFrame([
        {
            "term": "condition",
            "estimate": 1.0,
            "SE": 0.2,
            "CI_lower": 0.6,
            "CI_upper": 1.4,
            "p": 0.01,
            "converged": True,
            "N": len(data),
        }
    ])


def _duplicate_callback(data, spec):
    row = _tidy_callback(data, spec)
    return pd.concat([row, row], ignore_index=True)


def main():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=20260919)
    external = ep.define_event_detector_spec(
        "external_fail",
        "external",
        callback=_failing_external,
        implementation="deliberate_failure_fixture",
    )
    detected = ep.run_detector_multiverse(data, [_ivt(), external], continue_on_error=True)
    assert set(detected.status.status) == {"ok", "failed"}
    assert "deliberate external detector failure" in detected.failures.iloc[0].error

    features = ep.propagate_detector_to_aoi(detected)
    features = ep.propagate_detector_to_features(features)
    target = features.features[
        features.features.detector_id.eq("ivt30")
        & features.features.aoi_id.astype(str).eq("disclosure")
    ].index
    assert len(target) >= 3

    # Normalize the synthetic target rows first so the deliberate failures below
    # are the only attrition events in this clinic.
    features.features.loc[target, "valid_data_fraction"] = 1.0
    finite = np.isfinite(
        pd.to_numeric(features.features.loc[target, "dwell_time_ms"], errors="coerce")
    )
    if not finite.all():
        features.features.loc[target[~finite], "dwell_time_ms"] = 100.0
    features.features.loc[target[0], "valid_data_fraction"] = 0.1
    features.features.loc[target[1], "dwell_time_ms"] = np.nan

    inference = ep.run_detector_inference_multiverse(
        features,
        {"engine": "callback", "outcome": "dwell_time_ms", "aoi_id": "disclosure"},
        model_callback=_tidy_callback,
        minimum_valid_fraction=0.5,
    )
    audit = inference.input_audit[inference.input_audit.detector_id.eq("ivt30")].iloc[0]
    assert audit.quality_excluded_rows == 1
    assert audit.outcome_missing_rows == 1
    assert audit.status == "modelled"

    invalid = ep.run_detector_inference_multiverse(
        features,
        {"engine": "callback", "outcome": "dwell_time_ms", "aoi_id": "disclosure"},
        model_callback=_duplicate_callback,
    )
    assert invalid.coefficients.empty
    invalid_failure = invalid.failures[invalid.failures.detector_id.eq("ivt30")].iloc[0]
    assert "at most one row per coefficient term" in invalid_failure.error
    invalid_audit = invalid.input_audit[invalid.input_audit.detector_id.eq("ivt30")].iloc[0]
    assert invalid_audit.status == "failed"

    return {
        "detection_status": detected.status,
        "detection_failures": detected.failures,
        "model_input_audit": inference.input_audit,
        "model_input_warnings": inference.warnings,
        "invalid_callback_failures": invalid.failures,
    }


if __name__ == "__main__":
    result = main()
    for name, table in result.items():
        print(f"\n## {name}\n")
        print(table.to_string(index=False))
