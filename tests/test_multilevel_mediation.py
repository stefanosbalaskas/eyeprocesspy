import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyeprocesspy.multilevel_mediation import (
    MultilevelMediationData,
    audit_mediation_missingness,
    center_within_participant,
    check_mediation_trial_counts,
    decompose_within_between,
    identify_mediation_levels,
    mediation_provenance_json,
    prepare_multilevel_mediation_data,
    summarise_within_between_variance,
    validate_multilevel_mediation_data,
)


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": np.repeat(["p1", "p2", "p3"], 4),
            "trial_id": list(range(1, 5)) * 3,
            "condition": [0, 1, 0, 1] * 3,
            "dwell_ms": [100.0, 0.0, 120.0, np.nan, 90.0, 140.0, 80.0, 160.0, 110.0, 130.0, 100.0, 150.0],
            "override": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "valid_fraction": [0.95, 0.99, 0.92, 0.96, 0.94, 0.40, 0.97, 0.93, 0.96, 0.95, 0.98, 0.92],
        }
    )


def test_decompose_within_between_preserves_rows_and_group_means():
    data = make_data()
    out = decompose_within_between(data, ["condition", "dwell_ms"])
    assert len(out) == len(data)
    assert out.groupby("participant_id")["condition_within"].mean().abs().max() < 1e-12
    assert out.loc[data["dwell_ms"].isna(), "dwell_ms_within"].isna().all()
    assert set(out["condition_between"].unique()) == {0.5}


def test_center_within_participant():
    out = center_within_participant(make_data(), "condition")
    assert out.groupby("participant_id")["condition_within"].mean().abs().max() < 1e-12


def test_prepare_distinguishes_zero_missing_and_poor_quality_without_dropping():
    data = make_data()
    with pytest.warns(UserWarning):
        prepared = prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
            minimum_quality=0.80,
            source_id="synthetic-ai-advice",
        )
    assert isinstance(prepared, MultilevelMediationData)
    assert prepared.n_rows == 12
    assert prepared.data["mediation_mediator_true_zero"].sum() == 1
    assert prepared.data["mediation_mediator_state"].eq("not_observed").sum() == 1
    assert prepared.data["mediation_mediator_state"].eq("poor_quality").sum() == 1
    assert prepared.data["dwell_ms"].isna().sum() == 1
    assert prepared.data.loc[prepared.data["valid_fraction"].eq(0.40), "dwell_ms"].iloc[0] == 140.0
    assert prepared.n_analysis_eligible == 10


def test_explicit_quality_mask_masks_but_does_not_drop():
    data = make_data()
    with pytest.warns(UserWarning):
        prepared = prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
            minimum_quality=0.80,
            quality_action="mask_mediator",
        )
    assert len(prepared.data) == len(data)
    assert prepared.data["dwell_ms"].isna().sum() == 2
    assert prepared.data.loc[prepared.data["valid_fraction"].eq(0.40), "dwell_ms"].isna().all()


def test_invalid_quality_policy_is_explicit_error():
    with pytest.raises(ValueError, match="quality_action"):
        prepare_multilevel_mediation_data(
            make_data(),
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_action="drop",
        )


def test_no_within_participant_x_variation_fails_when_required():
    data = make_data()
    data["condition"] = data["participant_id"].map({"p1": 0, "p2": 1, "p3": 0})
    audit = validate_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
    )
    assert not audit["valid"]
    assert "X has no within-participant variation" in audit["issues"]
    with pytest.raises(ValueError, match="no within-participant variation"):
        prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
        )


def test_between_only_x_allowed_when_explicitly_requested():
    data = make_data()
    data["condition"] = data["participant_id"].map({"p1": 0, "p2": 1, "p3": 0})
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        require_within_x=False,
        warn=False,
    )
    level = prepared.levels.set_index("variable").loc["condition", "level"]
    assert level == "between_only"


def test_singleton_participant_is_warning_not_silent_exclusion():
    data = make_data()
    singleton = pd.DataFrame(
        {
            "participant_id": ["p4"],
            "trial_id": [1],
            "condition": [1],
            "dwell_ms": [100.0],
            "override": [1],
            "valid_fraction": [0.9],
        }
    )
    data = pd.concat([data, singleton], ignore_index=True)
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        warn=False,
    )
    assert prepared.trial_counts.loc[prepared.trial_counts["participant_id"].eq("p4"), "singleton"].item()
    assert len(prepared.data) == len(data)
    assert any("only one observed trial" in x for x in prepared.warnings)


def test_variance_and_level_audits():
    data = make_data()
    variance = summarise_within_between_variance(data, ["condition", "dwell_ms"])
    assert set(variance["variable"]) == {"condition", "dwell_ms"}
    levels = identify_mediation_levels(data, ["condition", "dwell_ms"])
    assert levels.set_index("variable").loc["condition", "level"] == "within_only"


def test_missingness_audit_counts_true_zero_separately():
    audit = audit_mediation_missingness(
        make_data(),
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        quality_col="valid_fraction",
        minimum_quality=0.80,
    ).set_index("issue")
    assert audit.loc["mediator_not_observed", "n"] == 1
    assert audit.loc["mediator_observed_zero", "n"] == 1
    assert audit.loc["poor_quality_trial", "n"] == 1


def test_trial_count_audit_handles_unbalanced_trials():
    data = make_data().drop(index=[7, 10])
    counts = check_mediation_trial_counts(data)
    assert set(counts["n_trials"]) == {3, 4}


def test_provenance_is_deterministic_and_serializable():
    kwargs = dict(
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        source_id="fixture-v1",
        preprocessing_spec={"blink": "flag"},
        event_detector={"algorithm": "I-VT", "threshold": 30},
        aoi_specification={"source": "synthetic"},
        warn=False,
    )
    a = prepare_multilevel_mediation_data(make_data(), **kwargs)
    b = prepare_multilevel_mediation_data(make_data(), **kwargs)
    assert a.provenance["source_fingerprint_sha256"] == b.provenance["source_fingerprint_sha256"]
    decoded = json.loads(mediation_provenance_json(a))
    assert decoded["event_detector"]["algorithm"] == "I-VT"
    assert decoded["software"]["package"] == "eyeprocesspy"


def test_duplicate_participant_trial_key_fails():
    data = pd.concat([make_data(), make_data().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
        )


def test_additional_component_supports_serial_or_moderator_without_backend_decomposition():
    from eyeprocesspy.multilevel_mediation import add_multilevel_mediation_component

    data = make_data()
    data["trust"] = [3.0, 4.0, 3.2, 4.4, 2.8, 4.1, 3.1, 4.5, 3.3, 4.2, 3.0, 4.6]
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        warn=False,
    )
    augmented = add_multilevel_mediation_component(
        prepared,
        value_col="trust",
        semantic="mediator2",
        within_col="M2_within",
        between_col="M2_between",
    )
    assert "M2_within" in augmented.data
    assert "M2_between" in augmented.data
    assert augmented.columns["mediator2"] == "trust"
    assert augmented.data.groupby("participant_id")["M2_within"].mean().abs().max() < 1e-12
    assert augmented.provenance["additional_mediation_components"][0]["semantic"] == "mediator2"


def test_observation_flags_are_strict_not_truthy_strings():
    data = make_data()
    data["mediator_seen"] = pd.Series([True] * len(data), dtype=object)
    data.loc[0, "mediator_seen"] = "false"
    with pytest.raises(ValueError, match="TRUE/FALSE or 0/1"):
        prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            mediator_observed_col="mediator_seen",
            warn=False,
        )


def test_numeric_zero_one_observation_flags_are_supported():
    data = make_data()
    data["response_seen"] = 1
    data.loc[0, "response_seen"] = 0
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        response_observed_col="response_seen",
        warn=False,
    )
    assert not prepared.data.loc[0, "mediation_response_observed"]
    assert not prepared.data.loc[0, "mediation_analysis_eligible"]


def test_nonnumeric_quality_fails_instead_of_becoming_missing_quality():
    data = make_data()
    data["valid_fraction"] = data["valid_fraction"].astype(object)
    data.loc[0, "valid_fraction"] = "bad"
    with pytest.raises(ValueError, match="must be numeric"):
        prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
            minimum_quality=0.8,
            warn=False,
        )


def test_nonfinite_minimum_quality_fails_explicitly():
    with pytest.raises(ValueError, match="finite numeric"):
        prepare_multilevel_mediation_data(
            make_data(),
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
            minimum_quality=float("nan"),
            warn=False,
        )


def test_existing_derived_columns_are_not_silently_overwritten():
    data = make_data()
    data["X_within"] = 999.0
    with pytest.raises(ValueError, match="already contains derived mediation column"):
        prepare_multilevel_mediation_data(
            data,
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            warn=False,
        )


def test_additional_component_refuses_output_collision():
    from eyeprocesspy.multilevel_mediation import add_multilevel_mediation_component

    data = make_data()
    data["trust"] = np.linspace(1, 2, len(data))
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        warn=False,
    )
    prepared.data["M2_within"] = 0.0
    with pytest.raises(ValueError, match="already contains derived mediation column"):
        add_multilevel_mediation_component(
            prepared,
            value_col="trust",
            semantic="mediator2",
            within_col="M2_within",
            between_col="M2_between",
        )


def test_provenance_records_python_row_position_convention():
    prepared = prepare_multilevel_mediation_data(
        make_data(),
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        warn=False,
    )
    assert prepared.provenance["row_position_convention"]["base"] == 0


def test_cross_language_expected_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "multilevel_mediation_expected.csv"
    expected = pd.read_csv(fixture_path)
    prepared = prepare_multilevel_mediation_data(
        make_data(),
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        quality_col="valid_fraction",
        minimum_quality=0.80,
        warn=False,
    )
    cols = list(expected.columns)
    actual = prepared.data.rename(columns={"condition": "unused"}).copy()
    actual["participant_id"] = prepared.data["participant_id"]
    actual["trial_id"] = prepared.data["trial_id"]
    for col in cols:
        if col in {"participant_id", "trial_id"}:
            continue
        assert col in prepared.data.columns
    pd.testing.assert_frame_equal(
        prepared.data[cols].reset_index(drop=True),
        expected[cols].reset_index(drop=True),
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )
