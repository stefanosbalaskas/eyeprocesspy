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
            "dwell_ms": [
                100.0,
                0.0,
                120.0,
                np.nan,
                90.0,
                140.0,
                80.0,
                160.0,
                110.0,
                130.0,
                100.0,
                150.0,
            ],
            "override": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "valid_fraction": [
                0.95,
                0.99,
                0.92,
                0.96,
                0.94,
                0.40,
                0.97,
                0.93,
                0.96,
                0.95,
                0.98,
                0.92,
            ],
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
    assert prepared.trial_counts.loc[
        prepared.trial_counts["participant_id"].eq("p4"), "singleton"
    ].item()
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


def test_helper_validation_and_repr_provenance_branches():
    from eyeprocesspy import multilevel_mediation as mm

    class Custom:
        def __repr__(self):
            return "CUSTOM"

    assert mm._json_safe(Custom()) == "CUSTOM"
    with pytest.raises(TypeError, match="pandas DataFrame"):
        mm.center_within_participant([], "condition")
    with pytest.raises(ValueError, match="non-empty column name"):
        mm.prepare_multilevel_mediation_data(
            make_data(), x_col="", mediator_col="dwell_ms", outcome_col="override", warn=False
        )
    with pytest.raises(ValueError, match="Missing required columns"):
        mm.center_within_participant(make_data(), "does_not_exist")
    bad = make_data().astype({"dwell_ms": object})
    bad.loc[0, "dwell_ms"] = "oops"
    with pytest.raises(ValueError, match="must be numeric"):
        mm.center_within_participant(bad, "dwell_ms")


def test_boolean_observation_indicator_and_participant_count_property():
    data = make_data()
    data["seen"] = pd.Series([True] * len(data), dtype=bool)
    data.loc[0, "seen"] = False
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        mediator_observed_col="seen",
        warn=False,
    )
    assert prepared.n_participants == 3
    assert prepared.data.loc[0, "mediation_mediator_state"] == "not_observed"


def test_decomposition_empty_and_grand_mean_centering():
    with pytest.raises(ValueError, match="at least one variable"):
        decompose_within_between(make_data(), [])
    out = decompose_within_between(make_data(), "condition", grand_mean_center_between=True)
    assert np.isclose(out["condition_between"].mean(), 0.0)


def test_variance_empty_singleton_and_constant_level_branches():
    with pytest.raises(ValueError, match="at least one variable"):
        summarise_within_between_variance(make_data(), [])
    one = pd.DataFrame({"participant_id": ["p1"], "trial_id": [1], "v": [2.0]})
    summary = summarise_within_between_variance(one, "v")
    assert np.isnan(summary.loc[0, "total_variance"])
    assert np.isnan(summary.loc[0, "within_variance"])
    assert np.isnan(summary.loc[0, "between_variance"])
    constant = make_data()
    constant["constant"] = 1.0
    levels = identify_mediation_levels(constant, ["condition", "dwell_ms", "constant"])
    lookup = levels.set_index("variable")["level"]
    assert lookup["constant"] == "constant_or_unidentified"
    both = make_data()
    both.loc[both["participant_id"].eq("p2"), "condition"] += 1
    both_level = identify_mediation_levels(both, "condition").loc[0, "level"]
    assert both_level == "within_and_between"
    with pytest.raises(ValueError, match="finite non-negative"):
        identify_mediation_levels(make_data(), "condition", tolerance=-1)


def test_missingness_audit_validation_branches_and_empty_proportion():
    with pytest.raises(ValueError, match="supplied together"):
        audit_mediation_missingness(
            make_data(),
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
        )
    with pytest.raises(ValueError, match="finite numeric"):
        audit_mediation_missingness(
            make_data(),
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
            minimum_quality=True,
        )
    data = make_data()
    data["m_seen"] = 1
    data["y_seen"] = 1
    data.loc[0, "m_seen"] = 0
    data.loc[1, "y_seen"] = 0
    audit = audit_mediation_missingness(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        mediator_observed_col="m_seen",
        response_observed_col="y_seen",
    ).set_index("issue")
    assert audit.loc["mediator_not_observed", "n"] >= 2
    assert audit.loc["response_missing", "n"] == 1
    empty = data.iloc[0:0]
    audit0 = audit_mediation_missingness(
        empty, x_col="condition", mediator_col="dwell_ms", outcome_col="override"
    )
    assert audit0["proportion"].isna().all()


def test_trial_count_minimum_validation():
    for bad in [0, True, 1.5]:
        with pytest.raises(ValueError, match="integer of at least 1"):
            check_mediation_trial_counts(make_data(), minimum_trials=bad)


def test_validation_reports_identifier_x_mediator_and_outcome_issues():
    empty = make_data().iloc[0:0]
    out = validate_multilevel_mediation_data(
        empty, x_col="condition", mediator_col="dwell_ms", outcome_col="override"
    )
    assert "data has no rows" in out["issues"]

    ids = make_data()
    ids.loc[0, "participant_id"] = np.nan
    ids.loc[1, "trial_id"] = np.nan
    out = validate_multilevel_mediation_data(
        ids, x_col="condition", mediator_col="dwell_ms", outcome_col="override"
    )
    assert "participant identifiers contain missing values" in out["issues"]
    assert "trial identifiers contain missing values" in out["issues"]

    badx = make_data().astype({"condition": object})
    badx.loc[0, "condition"] = "bad"
    out = validate_multilevel_mediation_data(
        badx,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        require_within_x=False,
    )
    assert "X must be numeric or explicitly coded before decomposition" in out["issues"]

    flatm = make_data()
    flatm["dwell_ms"] = 1.0
    flatm.loc[0, "override"] = np.nan
    out = validate_multilevel_mediation_data(
        flatm, x_col="condition", mediator_col="dwell_ms", outcome_col="override"
    )
    assert "mediator has no detectable within-participant variation" in out["warnings"]
    assert "outcome contains missing values; rows are preserved and flagged" in out["warnings"]


def test_prepare_quality_pair_validation_and_explicit_observation_columns():
    with pytest.raises(ValueError, match="supplied together"):
        prepare_multilevel_mediation_data(
            make_data(),
            x_col="condition",
            mediator_col="dwell_ms",
            outcome_col="override",
            quality_col="valid_fraction",
            warn=False,
        )
    data = make_data()
    data["m_seen"] = 1
    data["y_seen"] = 1
    data.loc[0, "m_seen"] = 0
    data.loc[1, "y_seen"] = 0
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="condition",
        mediator_col="dwell_ms",
        outcome_col="override",
        mediator_observed_col="m_seen",
        response_observed_col="y_seen",
        warn=False,
    )
    assert not prepared.data.loc[0, "mediation_mediator_observed"]
    assert not prepared.data.loc[1, "mediation_response_observed"]


def test_bad_provenance_object_and_component_validation_branches():
    with pytest.raises(TypeError, match="MultilevelMediationData"):
        mediation_provenance_json({})
    with pytest.raises(TypeError, match="MultilevelMediationData"):
        from eyeprocesspy.multilevel_mediation import add_multilevel_mediation_component

        add_multilevel_mediation_component(
            {}, value_col="trust", semantic="m2", within_col="M2w", between_col="M2b"
        )

    from eyeprocesspy.multilevel_mediation import add_multilevel_mediation_component

    data = make_data()
    data["trust"] = np.linspace(1, 2, len(data))
    prepared = prepare_multilevel_mediation_data(
        data, x_col="condition", mediator_col="dwell_ms", outcome_col="override", warn=False
    )
    with pytest.raises(ValueError, match="must be different"):
        add_multilevel_mediation_component(
            prepared, value_col="trust", semantic="m2", within_col="same", between_col="same"
        )
    explicit = add_multilevel_mediation_component(
        prepared,
        value_col="trust",
        semantic="m2",
        within_col="M2w",
        between_col="M2b",
        grand_mean_center_between=True,
    )
    assert np.isclose(explicit.data["M2b"].mean(), 0.0)


def test_validator_reports_nonnumeric_mediator_without_accidental_classifier_error():
    data = make_data().astype({"dwell_ms": object})
    data.loc[0, "dwell_ms"] = "bad"
    audit = validate_multilevel_mediation_data(
        data, x_col="condition", mediator_col="dwell_ms", outcome_col="override"
    )
    assert "mediator must be numeric or explicitly coded before decomposition" in audit["issues"]
    with pytest.raises(ValueError, match="mediator must be numeric"):
        prepare_multilevel_mediation_data(
            data, x_col="condition", mediator_col="dwell_ms", outcome_col="override", warn=False
        )


def test_warning_note_branches_independently_cover_missing_and_zero_false_paths():
    clean = make_data().copy()
    clean["dwell_ms"] = clean["dwell_ms"].fillna(125.0).replace(0.0, 105.0)
    prepared_clean = prepare_multilevel_mediation_data(
        clean, x_col="condition", mediator_col="dwell_ms", outcome_col="override", warn=False
    )
    assert not any("missing mediator" in x for x in prepared_clean.warnings)
    assert not any("observed zero" in x for x in prepared_clean.warnings)

    zero_only = clean.copy()
    zero_only.loc[0, "dwell_ms"] = 0.0
    prepared_zero = prepare_multilevel_mediation_data(
        zero_only, x_col="condition", mediator_col="dwell_ms", outcome_col="override", warn=False
    )
    assert any("observed zero" in x for x in prepared_zero.warnings)
    assert not any("missing mediator" in x for x in prepared_zero.warnings)
