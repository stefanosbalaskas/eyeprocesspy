from __future__ import annotations

import pandas as pd
import pytest

import eyeprocesspy as ep


def _dataset():
    recordings = pd.DataFrame(
        [
            {
                "recording_id": "R1",
                "participant_id": "P1",
                "nominal_sampling_rate": 2.0,
            }
        ]
    )
    coordinate_spaces = ep.new_coordinate_space("coord_display_normalized_top_left")
    gaze = pd.DataFrame(
        {
            "recording_id": ["R1"] * 5,
            "stream_id": ["G1"] * 5,
            "sample_id": [f"S{i}" for i in range(5)],
            "timestamp_seconds": [0.0, 0.5, 1.0, 1.5, 2.0],
            "gaze_x": [0.1, 0.2, 0.8, 0.8, 0.2],
            "gaze_y": [0.1, 0.2, 0.8, 0.7, 0.2],
            "valid": [True] * 5,
            "trial_id": [pd.NA] * 5,
            "stimulus_id": ["stim"] * 5,
            "coordinate_space_id": ["coord_display_normalized_top_left"] * 5,
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "recording_id": "R1",
                "timestamp_seconds": 0.0,
                "event_type": "trial",
                "event_name": "TRIAL_START",
                "event_value": "T1",
                "trial_id": pd.NA,
                "stimulus_id": "stim",
            },
            {
                "event_id": "E2",
                "recording_id": "R1",
                "timestamp_seconds": 2.0,
                "event_type": "trial",
                "event_name": "TRIAL_END",
                "event_value": "T1",
                "trial_id": pd.NA,
                "stimulus_id": "stim",
            },
        ]
    )
    eyes = pd.DataFrame(
        {
            "recording_id": ["R1", "R1"],
            "sample_id": ["P1", "P2"],
            "timestamp_seconds": [0.0, 1.0],
            "eye": ["left", "left"],
            "pupil_diameter": [3.0, 3.2],
            "pupil_valid": [True, True],
            "trial_id": [pd.NA, pd.NA],
            "stimulus_id": ["stim", "stim"],
        }
    )
    bio = pd.DataFrame(
        {
            "recording_id": ["R1", "R1"],
            "stream_id": ["B1", "B1"],
            "timestamp_seconds": [0.25, 1.25],
            "channel": ["eda", "eda"],
            "value": [1.0, 2.0],
            "valid": [True, True],
            "trial_id": [pd.NA, pd.NA],
            "stimulus_id": ["stim", "stim"],
        }
    )
    return ep.new_eye_dataset(
        recordings=recordings,
        coordinate_spaces=coordinate_spaces,
        gaze_samples=gaze,
        events=events,
        eye_samples=eyes,
        biometrics=bio,
    )


def test_as_eye_dataset_identity():
    x = _dataset()
    assert ep.as_eye_dataset(x) is x


def test_convert_xy_public_contract():
    out = ep.convert_xy(
        [0.5],
        [0.25],
        "display_normalized_top_left",
        "display_pixels_top_left",
        to_width=100,
        to_height=200,
    )
    assert out.loc[0, "x"] == pytest.approx(50.0)
    assert out.loc[0, "y"] == pytest.approx(50.0)


def test_audit_clock_sync():
    report = ep.audit_clock_sync(_dataset())
    assert report.loc[0, "recording_id"] == "R1"
    assert report.loc[0, "status"] == "overlap"


def test_build_trials_and_assign_trials():
    out = ep.build_trials(_dataset())
    trials = out["intervals"][out["intervals"]["interval_type"].eq("trial")]
    assert len(trials) == 1
    assert trials.iloc[0]["trial_id"] == "T1"
    assert out["gaze_samples"]["trial_id"].eq("T1").all()


def test_build_stimulus_intervals():
    out = ep.build_stimulus_intervals(_dataset())
    stimuli = out["intervals"][out["intervals"]["interval_type"].eq("stimulus")]
    assert len(stimuli) == 1
    assert stimuli.iloc[0]["stimulus_id"] == "stim"


def test_add_responses_and_build_item_responses():
    x = ep.build_trials(_dataset())
    response = pd.DataFrame([{"participant_id": "P1", "item_id": "I1", "response": "yes"}])
    out = ep.add_responses(x, response)
    assert len(out["responses"]) == 1

    x2 = x.copy()
    x2["intervals"].loc[x2["intervals"]["interval_type"].eq("trial"), "item_id"] = "I1"
    generated = ep.build_item_responses(x2)
    assert len(generated["responses"]) == 1


def test_new_register_assign_aoi_and_visits():
    x = ep.build_trials(_dataset())
    aoi = ep.new_aoi("A1", x=0, y=0, width=0.5, height=0.5)
    x = ep.register_aois(x, aoi)
    x = ep.assign_aois(x)
    assert x["gaze_samples"]["aoi_id"].notna().sum() == 3
    x = ep.build_aoi_visits(x, minimum_duration_ms=0)
    visits = x["episodes"][x["episodes"]["episode_type"].eq("aoi_visit")]
    assert len(visits) >= 1


def test_quality_sampling_and_store():
    x = _dataset()
    report = ep.audit_sampling_rate(x, expected_hz=2)
    assert report.loc[0, "status"] == "ok"
    stored = ep.store_quality(x, report)
    assert len(stored["quality"]) == 1


def test_signal_and_pupil_quality():
    x = _dataset()
    signal = ep.audit_signal_quality(x)
    assert "valid_gaze_fraction" in set(signal["metric"])
    pupil = ep.audit_pupil_quality(x, plausible_range=(2.0, 4.0))
    assert pupil["status"].eq("ok").all()


def test_episode_event_trial_aoi_audits():
    x = ep.build_trials(_dataset())
    aoi = ep.new_aoi("A1", x=0, y=0, width=1, height=1)
    x = ep.register_aois(x, aoi)
    assert ep.audit_event_order(x).loc[0, "status"] == "ok"
    assert ep.audit_trial_coverage(x).loc[0, "status"] == "ok"
    assert ep.audit_aois(x).loc[0, "status"] == "ok"
    assert ep.audit_episodes(x).empty


def test_missingness_report():
    report = ep.audit_missingness(_dataset(), component="gaze_samples")
    hit = report[report["field"].eq("gaze_x")]
    assert hit.iloc[0]["missing_fraction"] == pytest.approx(0.0)


def test_process_leakage():
    x = ep.build_trials(_dataset())
    x = ep.add_responses(
        x,
        pd.DataFrame(
            [
                {
                    "recording_id": "R1",
                    "participant_id": "P1",
                    "trial_id": "T1",
                    "item_id": "I1",
                    "response": "yes",
                    "response_timestamp": 1.0,
                }
            ]
        ),
    )
    feature = pd.DataFrame(
        [
            {
                "feature_id": "F1",
                "recording_id": "R1",
                "trial_id": "T1",
                "feature_name": "late",
                "window_end": 1.5,
                "level": "trial",
            }
        ]
    )
    x["features"] = ep.standardize_eye_table(feature, "features")
    report = ep.check_process_leakage(x)
    assert report.loc[0, "status"] == "error"


def test_feature_level():
    x = _dataset()
    feature = pd.DataFrame(
        [
            {
                "feature_id": "F1",
                "recording_id": "R1",
                "trial_id": pd.NA,
                "feature_name": "test",
                "level": "trial",
            }
        ]
    )
    x["features"] = ep.standardize_eye_table(feature, "features")
    report = ep.check_feature_level(x)
    assert report.loc[0, "status"] == "warning"
    assert report.loc[0, "missing_keys"] == "trial_id"


def test_interpretive_warnings_frozen_contract():
    table = ep.interpretive_warnings()
    assert len(table) == 7
    assert table.loc[0, "observation"] == "fixation"


def test_analysis_readiness_and_preprocessing_comparison():
    x = ep.build_trials(_dataset())
    readiness = ep.analysis_readiness(x)
    assert set(readiness["domain"]) == {
        "schema",
        "recordings",
        "timestamps",
        "coordinates",
        "trials",
        "responses",
        "gaze_quality",
        "provenance",
    }
    comparison = ep.compare_preprocessing(x, x)
    assert len(comparison) == 2


def test_compare_aoi_definitions():
    x = ep.build_trials(_dataset())
    x = ep.register_aois(x, ep.new_aoi("A1", x=0, y=0, width=1, height=1))
    x = ep.assign_aois(x)
    result = ep.compare_aoi_definitions({"candidate": x})
    assert result.loc[0, "definition_set"] == "candidate"


def test_sensitivity_process():
    x = _dataset()
    result = ep.sensitivity_process(x, label=["default"])
    assert list(result) == ["summary", "compared_at"]
    assert result["summary"].loc[0, "pipeline"] == "default"


def test_synchronize_eye_biometrics_none_smoke():
    gaze = _dataset()
    bio = ep.new_eye_dataset(
        recordings=pd.DataFrame([{"recording_id": "R2", "participant_id": "P2"}]),
        biometrics=pd.DataFrame(
            {
                "recording_id": ["R2"],
                "stream_id": ["B2"],
                "timestamp_seconds": [0.0],
                "channel": ["eda"],
                "value": [1.0],
            }
        ),
    )
    out = ep.synchronize_eye_biometrics(gaze, bio, method="none", resolve_ids=False)
    assert len(out["recordings"]) == 2
