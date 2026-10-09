import pytest

from research.methods_briefing_2026.gaze_reference_provenance import (
    declare_gaze_validation_provenance,
)


def test_declared_reference_is_hashed_and_accuracy_requires_targets():
    kw = dict(reference_type="human_validation_target", coordinate_space="absolute_gaze",
              reference_identifier="9-point reference",
              accuracy_deg=0.6, precision_deg=0.2,
              target_measurements_available=True)
    result = declare_gaze_validation_provenance(**kw)
    assert result["sha256"] == declare_gaze_validation_provenance(**kw)["sha256"]
    assert result["accuracy_deg"] == .6
    with pytest.raises(ValueError, match="target measurement"):
        declare_gaze_validation_provenance(**dict(kw, target_measurements_available=False))


def test_relative_accuracy_is_not_implicitly_absolute():
    with pytest.raises(ValueError, match="not absolute"):
        declare_gaze_validation_provenance(
            reference_type="trajectory_relative", coordinate_space="trajectory_relative",
            reference_identifier="offset-trajectory", accuracy_deg=.1,
        )
    result = declare_gaze_validation_provenance(
        reference_type="trajectory_relative", coordinate_space="trajectory_relative",
        reference_identifier="offset-trajectory", precision_deg=.1,
    )
    assert result["accuracy_deg"] is None
