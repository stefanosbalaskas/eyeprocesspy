from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.irt as irt
import eyeprocesspy.validation_atlas_09 as va
import eyeprocesspy.validation_extras_09 as validation_extras


def test_private_class_and_frame_coercion_residuals():
    tagged_frame = pd.DataFrame({"x": [1]})
    tagged_frame.attrs["eyeprocess_class"] = "eye_tagged_frame"
    assert va._class_name(tagged_frame) == "eye_tagged_frame"

    class TaggedObject:
        eyeprocess_class = "eye_tagged_object"

    assert va._class_name(TaggedObject()) == "eye_tagged_object"
    assert va._class_name(object()) is None

    series = pd.Series([1.0, 2.0], name="value")
    series_frame = va._as_frame(series, name="series")
    assert series_frame.columns.tolist() == ["value"]

    mapping_frame = va._as_frame({"value": [1, 2]}, name="mapping")
    assert mapping_frame["value"].tolist() == [1, 2]

    scalar_mapping = va._as_frame(
        {"value": 1, "label": "one"},
        name="scalar mapping",
    )
    assert scalar_mapping.to_dict("records") == [{"value": 1, "label": "one"}]

    sequence_frame = va._as_frame([[1, 2]], name="sequence")
    assert sequence_frame.shape == (1, 2)

    with pytest.raises(ep.EyeProcessValidationError, match="coercible to a data frame"):
        va._as_frame(object(), name="bad")


def test_round_numeric_validation_and_non_numeric_false_branch():
    with pytest.raises(ep.EyeProcessValidationError, match="digits must be an integer"):
        va._round_numeric(pd.DataFrame({"x": [1.234]}), digits="not-an-integer")

    text = pd.DataFrame({"label": ["a", "b"]})
    rounded = va._round_numeric(text, digits=2)
    pd.testing.assert_frame_equal(rounded, text)


def test_specialized_recovery_sbc_and_stress_paths(monkeypatch):
    monkeypatch.setattr(
        irt,
        "eyeprocess_irt_recovery_summary",
        lambda x: pd.DataFrame(
            {
                "scenario_id": ["S1"],
                "parameter": ["a"],
                "bias": [0.123456],
            }
        ),
    )
    recovery = ep.eyeprocess_recovery_evidence_table(
        {"eyeprocess_class": "eye_irt_recovery_result"},
        digits=3,
    )
    assert recovery.columns.tolist() == ["scenario_id", "parameter", "bias"]
    assert recovery.loc[0, "bias"] == pytest.approx(0.123)

    monkeypatch.setattr(
        validation_extras,
        "sbc_ecdf_deviation",
        lambda x: 0.125,
    )
    diagnostics = ep.eyeprocess_sbc_evidence_table(
        {
            "eyeprocess_class": "eye_sbc_diagnostics",
            "ranks": [0, 1, 2],
            "n_draws": 10,
        }
    )
    assert diagnostics.loc[0, "n_ranks"] == 3
    assert diagnostics.loc[0, "n_draws"] == 10
    assert diagnostics.loc[0, "ecdf_max_deviation"] == pytest.approx(0.125)

    irt_sbc = ep.eyeprocess_sbc_evidence_table(
        {
            "eyeprocess_class": "eye_irt_sbc_evidence",
            "n": 20,
            "n_draws": 100,
            "ecdf_deviation": 0.04,
            "coverage": 0.95,
            "nominal_coverage": None,
            "coverage_error": 0.01,
        }
    )
    assert irt_sbc.loc[0, "coverage"] == pytest.approx(0.95)
    assert "nominal_coverage" not in irt_sbc.columns
    assert irt_sbc.loc[0, "coverage_error"] == pytest.approx(0.01)

    stress = ep.eyeprocess_stress_evidence_table(
        {
            "eyeprocess_class": "eye_stress_test_summary",
            "table": pd.DataFrame({"severity": [0.333333]}),
        },
        digits=2,
    )
    assert stress.loc[0, "severity"] == pytest.approx(0.33)


def test_specialized_reliability_and_negative_control_paths(monkeypatch):
    mapping_icc = ep.eyeprocess_reliability_evidence_table(
        {
            "eyeprocess_class": "eye_process_reliability_profile",
            "measure": "dwell",
            "icc": {"icc_a1": 0.87654},
            "temporal": pd.DataFrame({"session": [1, 2, 3]}),
        },
        digits=3,
    )
    assert mapping_icc.loc[0, "icc_a1"] == pytest.approx(0.877)
    assert mapping_icc.loc[0, "temporal_pairs"] == 3

    object_icc = ep.eyeprocess_reliability_evidence_table(
        {
            "eyeprocess_class": "eye_process_reliability_profile",
            "measure": "pupil",
            "icc": SimpleNamespace(icc_a1=0.76543),
            "temporal": None,
        },
        digits=2,
    )
    assert object_icc.loc[0, "icc_a1"] == pytest.approx(0.77)
    assert "temporal_pairs" not in object_icc.columns

    monkeypatch.setattr(
        va,
        "summarise_process_negative_controls",
        lambda x: pd.DataFrame({"control": ["shuffle"], "effect": [0.012345]}),
    )
    negative = ep.eyeprocess_negative_control_evidence_table(
        {"eyeprocess_class": "eye_process_negative_controls"},
        digits=4,
    )
    assert negative.loc[0, "control"] == "shuffle"
    assert negative.loc[0, "effect"] == pytest.approx(0.0123)


def test_default_irt_theta_and_index_missing_root(monkeypatch, tmp_path):
    captured: dict[str, np.ndarray] = {}

    def fake_information(theta, items):
        captured["theta"] = np.asarray(theta, dtype=float)
        return pd.DataFrame(
            {
                "theta": [-3.0],
                "information": [1.25],
                "conditional_sem": [1.0 / np.sqrt(1.25)],
            }
        )

    monkeypatch.setattr(irt, "eyeprocess_irt_test_information", fake_information)
    table = ep.eyeprocess_irt_precision_evidence_table(
        pd.DataFrame({"item_id": ["I1"]}),
        theta=None,
    )
    assert table.loc[0, "information"] == pytest.approx(1.25)
    np.testing.assert_allclose(captured["theta"], np.arange(-3.0, 3.25, 0.5))

    with pytest.raises(ep.EyeProcessValidationError, match="existing path"):
        ep.eyeprocess_validation_evidence_index(tmp_path / "does-not-exist")


def test_freeze_default_metadata_and_report_invalid_class(tmp_path):
    claims = pd.DataFrame(
        {
            "claim_id": ["C1"],
            "claim": ["deterministic"],
            "status": ["supported"],
        }
    )
    atlas = ep.eyeprocess_validation_evidence_atlas(claims)
    frozen = ep.freeze_eyeprocess_validation_atlas(atlas)
    assert frozen["payload"]["metadata"] == {}
    assert ep.verify_eyeprocess_validation_atlas(frozen)

    with pytest.raises(ep.EyeProcessValidationError, match="eye_validation_evidence_atlas"):
        ep.write_eyeprocess_validation_report(
            {},
            tmp_path / "invalid-report.md",
        )
