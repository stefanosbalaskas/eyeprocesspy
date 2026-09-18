# ruff: noqa: I001
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


FIXTURE = Path(__file__).parent / "fixtures" / "detector_multiverse_contract.json"


def ivt(detector_id="ivt30", threshold=30):
    return ep.define_event_detector_spec(
        detector_id,
        "ivt",
        velocity_threshold=threshold,
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
    )


def idt(detector_id="idtA"):
    return ep.define_event_detector_spec(
        detector_id,
        "idt",
        dispersion_threshold=1.2,
        minimum_duration_ms=80,
        sampling_rate=60,
        coordinate_unit="degrees",
    )


def adaptive(detector_id="adaptive"):
    return ep.define_event_detector_spec(
        detector_id,
        "adaptive_velocity",
        minimum_duration_ms=60,
        maximum_gap_ms=75,
        sampling_rate=60,
        coordinate_unit="degrees",
        parameters={"noise_factor": 4, "minimum_velocity_threshold": 20},
    )


def test_contract_fixture_builds_same_semantic_specs():
    payload = json.loads(FIXTURE.read_text())
    built = []
    for row in payload["specs"]:
        params = row.copy()
        params.pop("implementation_version")
        params["parameters"] = row["parameters"]
        built.append(ep.define_event_detector_spec(**params))
    manifest = ep.create_detector_multiverse(built).manifest
    assert set(manifest.detector_id) == {"ivt30", "idtA"}
    assert manifest.set_index("detector_id").loc["ivt30", "velocity_threshold"] == 30
    assert manifest.set_index("detector_id").loc["idtA", "dispersion_threshold"] == pytest.approx(1.2)


def test_spec_validation_requires_scientific_inputs():
    with pytest.raises(ep.EyeProcessValidationError, match="sampling_rate"):
        ep.define_event_detector_spec("bad", "ivt", velocity_threshold=30, minimum_duration_ms=60)
    with pytest.raises(ep.EyeProcessValidationError, match="velocity_threshold"):
        ep.define_event_detector_spec("bad", "ivt", velocity_threshold=-1, minimum_duration_ms=60, sampling_rate=60)
    with pytest.raises(ep.EyeProcessValidationError, match="dispersion_threshold"):
        ep.define_event_detector_spec("bad", "idt", dispersion_threshold=0, minimum_duration_ms=80, sampling_rate=60)
    with pytest.raises(ep.EyeProcessValidationError, match="callback"):
        ep.define_event_detector_spec("bad", "external")


def test_parameter_grid_is_explicit_and_deterministic():
    base = ivt("ivt")
    grid = ep.create_detector_multiverse(
        base_spec=base,
        parameter_grid={"velocity_threshold": [20, 25, 30, 35, 40], "minimum_duration_ms": [60, 80]},
    )
    assert len(grid.specs) == 10
    assert [s.detector_id for s in grid.specs] == sorted(s.detector_id for s in grid.specs)
    assert {s.velocity_threshold for s in grid.specs} == {20, 25, 30, 35, 40}


def test_specification_order_does_not_change_results():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=11)
    specs = [ivt("b", 30), idt("a"), adaptive("c")]
    one = ep.run_detector_multiverse(data, ep.create_detector_multiverse(specs))
    two = ep.run_detector_multiverse(data, ep.create_detector_multiverse(list(reversed(specs))))
    pd.testing.assert_frame_equal(one.status.reset_index(drop=True), two.status.reset_index(drop=True))
    cols = ["detector_id", "episode_type", "recording_id", "trial_id", "start_time", "end_time", "duration_ms"]
    pd.testing.assert_frame_equal(
        one.events[cols].sort_values(cols[:4] + ["start_time"]).reset_index(drop=True),
        two.events[cols].sort_values(cols[:4] + ["start_time"]).reset_index(drop=True),
    )


def test_identical_detector_specs_give_identical_scientific_events():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=12)
    result = ep.run_detector_multiverse(data, [ivt("x", 30), ivt("y", 30)])
    x = result.events[result.events.detector_id.eq("x")]
    y = result.events[result.events.detector_id.eq("y")]
    cols = ["recording_id", "trial_id", "episode_type", "start_time", "end_time", "duration_ms", "centroid_x", "centroid_y"]
    pd.testing.assert_frame_equal(x[cols].reset_index(drop=True), y[cols].reset_index(drop=True))


def test_event_matching_fixture_uses_one_to_one_temporal_matching():
    payload = json.loads(FIXTURE.read_text())["event_matching_fixture"]
    reference = pd.DataFrame(payload["reference"])
    candidate = pd.DataFrame(payload["candidate"])
    matches = ep.match_detected_events(reference, candidate, onset_tolerance_ms=75, minimum_overlap=.1)
    assert len(matches) == 2
    summary = ep.compare_event_catalogues(reference, candidate).iloc[0]
    for key, value in payload["expected"].items():
        assert int(summary[key]) == value
    assert summary.matched_event_precision == pytest.approx(2 / 3)
    assert summary.matched_event_recall == pytest.approx(1.0)


def test_short_and_zero_event_trials_are_retained_in_features():
    data = ep.simulate_detector_multiverse_data(n_participants=4, trial_duration_s=.08, seed=13)
    result = ep.run_detector_multiverse(data, [ivt()])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    n_trials = len(data["intervals"][data["intervals"].interval_type.eq("trial")])
    n_aois = len(data["aoi_definitions"])
    assert len(result.features) == n_trials * n_aois
    assert (result.features.fixation_count.fillna(0) == 0).any()


def test_extreme_missingness_is_not_converted_to_zero_events():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=14)
    data = data.copy()
    data["gaze_samples"]["valid"] = False
    result = ep.run_detector_multiverse(data, [ivt()])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    assert result.features.valid_data_fraction.eq(0).all()
    assert result.features.fixation_count.isna().all()
    assert result.features.dwell_time_ms.isna().all()
    assert result.features.feature_review_required.all()


def test_constant_gaze_yields_fixation_without_crashing():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=15)
    data = data.copy()
    data["gaze_samples"]["gaze_x"] = 6.0
    data["gaze_samples"]["gaze_y"] = 2.4
    data["gaze_samples"]["valid"] = True
    result = ep.run_detector_multiverse(data, [ivt()])
    assert result.status.loc[0, "status"] == "ok"
    assert result.status.loc[0, "n_fixations"] > 0


def test_dense_motion_trial_is_preserved_even_when_fixations_are_sparse():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=16)
    data = data.copy()
    idx = np.arange(len(data["gaze_samples"]))
    data["gaze_samples"]["gaze_x"] = np.where(idx % 2, 10.0, 0.0)
    data["gaze_samples"]["gaze_y"] = np.where(idx % 2, 0.0, 8.0)
    data["gaze_samples"]["valid"] = True
    result = ep.run_detector_multiverse(data, [ivt("strict", 20)])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    assert len(result.features) == len(data["intervals"]) * len(data["aoi_definitions"])
    assert (result.features.fixation_count == 0).all()


def test_external_detector_import_and_callback_failure_are_explicit():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=17)

    def external(data, spec):
        trial = data["intervals"].iloc[0]
        return pd.DataFrame([{
            "recording_id": trial.recording_id,
            "trial_id": trial.trial_id,
            "episode_type": "fixation",
            "start_time": trial.start_time + .1,
            "end_time": trial.start_time + .2,
            "centroid_x": 6.0,
            "centroid_y": 2.4,
            "coordinate_space_id": "deg_display",
            "stimulus_id": "stim_01",
        }])

    good = ep.define_event_detector_spec("external_good", "external", callback=external, implementation="test_callback")
    result = ep.run_detector_multiverse(data, [good])
    assert result.failures.empty
    assert len(result.events) == 1
    assert result.events.iloc[0].detector_id == "external_good"

    def broken(data, spec):
        raise RuntimeError("detector exploded")

    bad = ep.define_event_detector_spec("external_bad", "external", callback=broken, implementation="test_callback")
    failed = ep.run_detector_multiverse(data, [bad])
    assert failed.status.iloc[0].status == "failed"
    assert "detector exploded" in failed.failures.iloc[0].error


def test_aoi_ambiguity_requires_explicit_resolution():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=18)
    overlap = ep.new_aoi(
        "overlap", "Overlap", "stim_01", "rectangle",
        x=4.5, y=1.5, width=3.0, height=2.0, coordinate_space_id="deg_display",
    )
    data = ep.register_aois(data, overlap)
    result = ep.run_detector_multiverse(data, [ivt()])
    with pytest.raises(ep.EyeProcessValidationError, match="Ambiguous AOI"):
        ep.propagate_detector_to_aoi(result, overlap="error", continue_on_error=False)
    resolved = ep.propagate_detector_to_aoi(result, overlap="smallest", continue_on_error=False)
    assert resolved.failures.empty


def test_provenance_retained_through_events_and_features():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=19)
    result = ep.run_detector_multiverse(data, [ivt()])
    required = {"source_data_hash", "preprocessing_provenance_hash", "aoi_spec_hash", "detector_spec_hash", "software", "software_version"}
    assert required.issubset(result.events.columns)
    assert result.events.detector_spec_hash.notna().all()
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    assert required.issubset(result.features.columns)
    assert result.features.source_data_hash.notna().all()


def test_synthetic_truth_propagates_known_disclosure_dwell_effect():
    data = ep.simulate_detector_multiverse_data(n_participants=8, seed=20)
    result = ep.run_detector_multiverse(data, [ivt("ivt25", 25), ivt("ivt35", 35), idt(), adaptive()])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    target = result.features[result.features.aoi_id.eq("disclosure")]
    means = target.groupby(["detector_id", "condition_id"]).dwell_time_ms.mean().unstack()
    assert (means["disclosure"] > means["control"]).all()


def test_inference_multiverse_records_coefficients_and_nonconvergence():
    data = ep.simulate_detector_multiverse_data(n_participants=6, seed=21)
    result = ep.run_detector_multiverse(data, [ivt("ivt25", 25), ivt("ivt35", 35), idt()])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    model_spec = {
        "engine": "statsmodels_ols",
        "formula": "dwell_time_ms ~ C(condition_id) + C(participant_id)",
        "outcome": "dwell_time_ms",
        "aoi_id": "disclosure",
    }
    inference = ep.run_detector_inference_multiverse(result, model_spec)
    term = inference.coefficients[inference.coefficients.term.str.contains("condition_id", na=False)].term.iloc[0]
    stability = ep.assess_detector_inference_stability(inference, term=term, substantive_threshold=100)
    assert stability.iloc[0].convergence_rate == 1
    assert stability.iloc[0].same_sign_proportion == 1
    assert stability.iloc[0].substantive_conclusion_stability == 1
    assert {"model_spec_hash", "feature_fingerprint"}.issubset(inference.coefficients.columns)

    def nonconverged(data, spec):
        return pd.DataFrame([{
            "term": "condition", "estimate": 1.0, "SE": 1.0, "CI_lower": -1.0, "CI_upper": 3.0,
            "p": .5, "converged": False, "N": len(data),
        }])

    callback_spec = dict(model_spec, engine="callback")
    bad = ep.run_detector_inference_multiverse(result, callback_spec, model_callback=nonconverged)
    summary = ep.assess_detector_inference_stability(bad, term="condition")
    assert summary.iloc[0].converged_specifications == 0
    assert summary.iloc[0].convergence_rate == 0


def test_statsmodels_missing_predictors_fail_instead_of_silent_row_drop():
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=22)
    result = ep.run_detector_multiverse(data, [ivt()])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    result.features.loc[result.features.index[0], "condition_id"] = pd.NA
    model = {"engine":"statsmodels_ols","formula":"dwell_time_ms ~ C(condition_id)","outcome":"dwell_time_ms","aoi_id":"disclosure"}
    inference = ep.run_detector_inference_multiverse(result, model)
    assert not inference.failures.empty
    assert inference.coefficients.empty


def test_report_and_plot_surfaces_smoke(tmp_path):
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=23)
    result = ep.run_detector_multiverse(data, [ivt("ivt25", 25), ivt("ivt35", 35), idt()])
    result = ep.propagate_detector_to_aoi(result)
    result = ep.propagate_detector_to_features(result)
    report_path = tmp_path / "detector-report.md"
    text = ep.report_detector_multiverse(result, path=str(report_path))
    assert "Do not summarize robustness by counting p-values alone" in text
    assert report_path.exists()
    ax1 = ep.plot_detector_event_timeline(result)
    ax2 = ep.plot_detector_agreement(result)
    ax3 = ep.plot_detector_feature_distributions(result, aoi_id="disclosure")
    assert ax1.figure and ax2.figure and ax3.figure


def test_remodnav_bridge_fails_explicitly_when_dependency_unavailable():
    spec = ep.define_event_detector_spec(
        "remodnav", "remodnav", minimum_duration_ms=60, sampling_rate=60,
        coordinate_unit="degrees", parameters={"noise_factor":5},
    )
    data = ep.simulate_detector_multiverse_data(n_participants=4, seed=24)
    try:
        import remodnav  # noqa: F401
    except ImportError:
        result = ep.run_detector_multiverse(data, [spec])
        assert result.status.iloc[0].status == "failed"
        assert "REMoDNaV is not installed" in result.failures.iloc[0].error
    else:
        result = ep.run_detector_multiverse(data, [spec])
        assert result.status.iloc[0].status == "ok"


def test_public_api_exports_are_available():
    names = [
        "define_event_detector_spec", "validate_event_detector_spec", "create_detector_multiverse",
        "run_detector_multiverse", "detect_events_with_spec", "import_external_detector_events",
        "compare_event_catalogues", "match_detected_events", "estimate_detector_agreement",
        "summarise_detector_events", "summarise_detector_disagreement", "propagate_detector_to_aoi",
        "propagate_detector_to_features", "run_detector_inference_multiverse",
        "assess_detector_inference_stability", "summarise_detector_robustness",
        "plot_detector_event_timeline", "plot_detector_agreement", "plot_detector_feature_distributions",
        "plot_detector_coefficient_stability", "plot_detector_multiverse", "report_detector_multiverse",
        "simulate_detector_multiverse_data",
    ]
    assert all(callable(getattr(ep, name, None)) for name in names)
