import math

from eyeprocesspy.measurement_accountability_11 import (
    event_marker_qc,
    pupil_latency_sensitivity,
    validation_ladder,
)


def test_pupil_latency_sensitivity_recovers_synthetic_onset():
    time = [-0.5 + i / 100.0 for i in range(251)]
    pupil = [
        4.0 if t < 0.30 else 4.0 - 0.8 * (1.0 - math.exp(-(t - 0.30) / 0.20))
        for t in time
    ]
    out = pupil_latency_sensitivity(time, pupil, simulations=20, seed=7)
    assert out["sampling_hz"] == pytest.approx(100.0, rel=1e-6)
    assert 0.20 <= out["estimates_s"]["sustained_threshold"] <= 0.45
    assert out["latency_resolvability"] in {"high", "moderate"}
    assert out["simulation"]["n"] == 20


def test_event_marker_qc_separates_plausibility_from_clock_sync():
    out = event_marker_qc([0.010, 0.015, 0.020], tolerance=0.050)
    assert out["status"] == "confirmed"
    assert "no clock-drift correction" in out["note"]


def test_generalization_requires_held_out_person_evidence():
    out = validation_ladder(
        {
            "acquisition_qc": True,
            "analytical_qc": True,
            "construct_check": True,
            "within_person": True,
            "held_out_person": None,
        },
        claim="generalizable",
    )
    assert out["claim_status"] == "not_supported"
    assert out["held_out_person_generalization"] is False


import pytest
