import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.october_2026.event_reconstruction import (  # noqa: E402
    confidence_threshold_audit,
    event_reconstruction_audit,
)


def _frames():
    ref = pd.DataFrame({
        "participant_id": ["p"] * 6, "trial_id": ["t"] * 6,
        "time_s": list(range(6)), "x_deg": [0., 0., 0., 1., 2., 3.],
        "y_deg": [0.] * 6,
        "event": ["fixation"] * 3 + ["saccade"] * 3,
    })
    est = ref.drop(columns=["event"]).copy()
    return ref, est


def test_perfect_event_geometry():
    r, p = _frames()
    result = event_reconstruction_audit(r, p)
    assert result["angular_error_mean_deg"] == pytest.approx(0)
    assert result["event_geometry"].iloc[0].fixation_centroid_error_deg == pytest.approx(0)
    assert result["event_geometry"].iloc[1].saccade_amplitude_error_deg == pytest.approx(0)
    assert result["event_geometry"].iloc[1].saccade_direction_error_deg == pytest.approx(0)


def test_stationary_predictor_fails_saccade_reconstruction():
    r, p = _frames()
    p["x_deg"] = 0.
    result = event_reconstruction_audit(r, p)
    assert result["event_geometry"].iloc[1].saccade_amplitude_error_deg == pytest.approx(2)
    assert np.isnan(result["event_geometry"].iloc[1].saccade_direction_error_deg)


def test_rejects_unmatched_timestamps():
    r, p = _frames()
    p.loc[0, "time_s"] = 0.1
    with pytest.raises(ValueError, match="exact row order"):
        event_reconstruction_audit(r, p)


def test_vendor_confidence_is_participant_scoped():
    d = pd.DataFrame({
        "participant_id": ["a", "a", "b", "b"],
        "vendor_confidence": [0.8, 0.9, 0.8, 0.9],
        "reference_error_deg": [0.2, 0.3, 8., 9.],
    })
    s = confidence_threshold_audit(d)
    assert s.iloc[0].acceptable_fraction_selected == 1
    assert s.iloc[1].acceptable_fraction_selected == 0
    assert (s.calibration_scope == "within_participant_descriptive_only").all()
