from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


def _dataset():
    recordings = pd.DataFrame(
        [{"recording_id": "R1", "participant_id": "P1", "nominal_sampling_rate": 2.0}]
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
            "pupil_diameter": [3.0, 4.5],
            "pupil_valid": [True, False],
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


def test_foundation_coercion_and_sync_guards():
    with pytest.raises(TypeError):
        ep.as_eye_dataset(object())
    x = _dataset()
    with pytest.raises(ValueError, match="method"):
        ep.synchronize_eye_biometrics(x, x, method="warp")
    with pytest.warns(RuntimeWarning, match="Marker pairs"):
        out = ep.synchronize_eye_biometrics(x, x, method="linear", resolve_ids=True)
    assert len(out["recordings"]) >= 1


def test_clock_sync_unavailable_and_channel_filter():
    x = _dataset()
    empty = x.copy()
    empty["biometrics"] = empty["biometrics"].iloc[0:0].copy()
    report = ep.audit_clock_sync(empty)
    assert report.loc[0, "status"] == "unavailable"
    filtered = ep.audit_clock_sync(x, channel="missing")
    assert filtered.empty


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"close_open": "bad"}, "close_open"),
        ({"event_field": "bad"}, "event_field"),
    ],
)
def test_build_trials_argument_guards(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ep.build_trials(_dataset(), **kwargs)


def test_build_trials_requires_events_and_rejects_invalid_regex():
    x = _dataset()
    no_events = x.copy()
    no_events["events"] = no_events["events"].iloc[0:0].copy()
    with pytest.raises(ValueError, match="No events"):
        ep.build_trials(no_events)
    with pytest.raises(ValueError, match="Invalid regular expression"):
        ep.build_trials(x, start_events=("[",))


def test_build_trials_next_start_and_drop_open_intervals():
    x = _dataset()
    starts = x["events"].iloc[[0]].copy()
    starts = pd.concat(
        [
            starts,
            starts.assign(
                event_id="E3",
                timestamp_seconds=1.0,
                event_value="T2",
                stimulus_id="stim",
            ),
        ],
        ignore_index=True,
    )
    x["events"] = ep.standardize_eye_table(starts, "events")
    next_start = ep.build_trials(x, close_open="next_start")
    trials = next_start["intervals"].query("interval_type == 'trial'")
    assert len(trials) == 1
    assert trials.iloc[0]["end_time"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="No trial intervals"):
        ep.build_trials(x, close_open="drop")


def test_build_trials_pattern_extracts_identifier():
    x = _dataset()
    x["events"].loc[x["events"]["event_name"].eq("TRIAL_START"), "event_value"] = "trial=ABC42"
    out = ep.build_trials(x, trial_id_pattern=r"trial=(\w+)")
    trial = out["intervals"].query("interval_type == 'trial'").iloc[0]
    assert trial["trial_id"] == "ABC42"


def test_stimulus_interval_guards_and_event_source():
    with pytest.raises(ValueError, match="source"):
        ep.build_stimulus_intervals(_dataset(), source="bad")
    x = _dataset()
    no_stim = x.copy()
    no_stim["gaze_samples"]["stimulus_id"] = pd.NA
    with pytest.raises(ValueError, match="No stimulus ids"):
        ep.build_stimulus_intervals(no_stim)
    x["events"] = ep.standardize_eye_table(
        pd.DataFrame(
            [
                {
                    "event_id": "M1",
                    "recording_id": "R1",
                    "timestamp_seconds": 0.0,
                    "event_type": "media_change",
                    "event_name": "MEDIA_START",
                    "event_value": "stimA",
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                },
                {
                    "event_id": "M2",
                    "recording_id": "R1",
                    "timestamp_seconds": 1.0,
                    "event_type": "media_change",
                    "event_name": "MEDIA_START",
                    "event_value": "stimB",
                    "trial_id": pd.NA,
                    "stimulus_id": pd.NA,
                },
            ]
        ),
        "events",
    )
    out = ep.build_stimulus_intervals(x, source="events")
    assert len(out["intervals"].query("interval_type == 'stimulus'")) == 2


def test_assign_trials_noop_and_overwrite():
    x = _dataset()
    assert ep.assign_trials(x)["gaze_samples"]["trial_id"].isna().all()
    built = ep.build_trials(x)
    built["gaze_samples"]["trial_id"] = "old"
    out = ep.assign_trials(built, overwrite=True)
    assert out["gaze_samples"]["trial_id"].eq("T1").all()


def test_add_responses_guards_and_overwrite():
    x = ep.build_trials(_dataset())
    with pytest.raises(TypeError, match="DataFrame"):
        ep.add_responses(x, [])
    with pytest.raises(ValueError, match="missing required"):
        ep.add_responses(x, pd.DataFrame({"participant_id": ["P1"]}))
    first = pd.DataFrame(
        [{"participant_id": "P1", "item_id": "I1", "trial_id": "T1", "response": "yes"}]
    )
    second = pd.DataFrame(
        [{"participant_id": "P1", "item_id": "I1", "trial_id": "T1", "response": "no"}]
    )
    x = ep.add_responses(x, first)
    out = ep.add_responses(x, second, overwrite=True)
    assert len(out["responses"]) == 1
    assert out["responses"].iloc[0]["response"] == "no"


def test_build_item_responses_guards_and_existing_response_noop():
    x = _dataset()
    with pytest.raises(ValueError, match="Trial intervals"):
        ep.build_item_responses(x)
    x = ep.build_trials(x)
    with pytest.raises(ValueError, match="score_key"):
        ep.build_item_responses(x, score_key=["bad"])
    existing = ep.add_responses(
        x,
        pd.DataFrame([{"participant_id": "P1", "item_id": "I1", "response": "yes"}]),
    )
    assert ep.build_item_responses(existing) is existing


@pytest.mark.parametrize("shape", ["triangle", "ellipse"])
def test_new_aoi_rejects_unknown_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        ep.new_aoi("A", shape=shape)


def test_polygon_aoi_validation_and_assignment():
    with pytest.raises(ValueError, match="two-column"):
        ep.new_aoi("P", shape="polygon", polygon=[0, 1, 2])
    poly = ep.new_aoi(
        "P",
        shape="polygon",
        polygon=np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),
    )
    x = ep.register_aois(_dataset(), poly)
    assigned = ep.assign_aois(x)
    assert assigned["gaze_samples"]["aoi_id"].notna().sum() == 3


def test_circle_and_overlap_modes():
    x = _dataset()
    large = ep.new_aoi("L", x=0, y=0, width=1, height=1)
    small = ep.new_aoi("S", x=0, y=0, width=0.4, height=0.4)
    circle = ep.new_aoi("C", shape="circle", x=0.5, y=0.5, width=1, height=1)
    x = ep.register_aois(x, [large, small, circle])
    all_hits = ep.assign_aois(x, overlap="all")
    assert all_hits["gaze_samples"]["aoi_id"].astype("string").str.contains("L").any()
    smallest = ep.assign_aois(x, overlap="smallest")
    assert "S" in set(smallest["gaze_samples"]["aoi_id"].dropna().astype(str))


def test_register_aoi_guards_duplicate_and_overwrite():
    x = _dataset()
    with pytest.raises(ValueError, match="Supply"):
        ep.register_aois(x)
    aoi = ep.new_aoi("A", x=0, y=0, width=1, height=1)
    x = ep.register_aois(x, aoi)
    with pytest.raises(ValueError, match="already exists"):
        ep.register_aois(x, aoi)
    replaced = ep.register_aois(x, ep.new_aoi("A", x=0, y=0, width=0.2, height=0.2), overwrite=True)
    assert replaced["aoi_definitions"]["aoi_id"].eq("A").sum() == 1


def test_assign_aoi_guards_and_preserve_existing_when_not_overwriting():
    x = _dataset()
    with pytest.raises(ValueError, match="component"):
        ep.assign_aois(x, component="bad")
    with pytest.raises(ValueError, match="overlap"):
        ep.assign_aois(x, overlap="bad")
    with pytest.raises(ValueError, match="No AOIs"):
        ep.assign_aois(x)
    x = ep.register_aois(x, ep.new_aoi("A", x=0, y=0, width=1, height=1))
    x["gaze_samples"]["aoi_id"] = pd.Series(["KEEP", pd.NA, pd.NA, pd.NA, pd.NA], dtype="string")
    out = ep.assign_aois(x, overwrite=False)
    assert out["gaze_samples"].iloc[0]["aoi_id"] == "KEEP"


def test_aoi_visit_guards_and_minimum_duration_filter():
    x = _dataset()
    with pytest.raises(ValueError, match="source"):
        ep.build_aoi_visits(x, source="bad")
    with pytest.raises(ValueError, match="Assign AOIs"):
        ep.build_aoi_visits(x)
    x = ep.register_aois(x, ep.new_aoi("A", x=0, y=0, width=1, height=1))
    x = ep.assign_aois(x)
    out = ep.build_aoi_visits(x, minimum_duration_ms=10_000)
    assert out["episodes"].empty


def test_quality_and_event_audit_edge_paths():
    x = _dataset()
    pupil = ep.audit_pupil_quality(x, maximum_interpolated_fraction=0, plausible_range=(2.0, 4.0))
    assert pupil["status"].eq("warning").any()
    empty = x.copy()
    empty["eye_samples"] = empty["eye_samples"].iloc[0:0].copy()
    assert ep.audit_pupil_quality(empty).empty
    x["events"] = x["events"].iloc[::-1].reset_index(drop=True)
    order = ep.audit_event_order(x)
    assert order.loc[0, "status"] == "warning"


def test_trial_aoi_and_missingness_edge_paths():
    x = _dataset()
    trial = ep.audit_trial_coverage(x)
    assert trial.loc[0, "status"] == "error"
    assert ep.audit_aois(x).loc[0, "status"] == "unavailable"
    with pytest.raises(ValueError, match="component"):
        ep.audit_missingness(x, component="events")
    ungrouped = ep.audit_missingness(x, by="does_not_exist")
    assert set(ungrouped["group"]) == {"all"}


def test_leakage_and_feature_checks_empty_paths():
    x = _dataset()
    assert ep.check_process_leakage(x).empty
    assert ep.check_feature_level(x).empty


def test_analysis_readiness_unprepared_dataset_marks_expected_domains_false():
    ready = ep.analysis_readiness(_dataset()).set_index("domain")["ready"]
    assert not bool(ready["trials"])
    assert not bool(ready["responses"])
    assert bool(ready["recordings"])
