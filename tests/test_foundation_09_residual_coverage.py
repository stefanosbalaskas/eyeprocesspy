from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.foundation_09 as fd


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
            "valid": [True, True, True, True, True],
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


def _fixation_rows():
    return pd.DataFrame(
        {
            "episode_id": ["F1", "F2", "F3"],
            "recording_id": ["R1"] * 3,
            "episode_type": ["fixation"] * 3,
            "eye": ["combined"] * 3,
            "start_time": [0.0, 0.15, 0.6],
            "end_time": [0.1, 0.25, 0.7],
            "duration_ms": [100.0] * 3,
            "centroid_x": [0.1, 0.2, 0.8],
            "centroid_y": [0.1, 0.2, 0.8],
            "coordinate_space_id": ["coord_display_normalized_top_left"] * 3,
            "trial_id": ["T1"] * 3,
            "stimulus_id": ["stim"] * 3,
            "aoi_id": ["A", "A", "B"],
            "derived_by": ["vendor", "eyeprocess", "vendor"],
        }
    )


def test_private_helper_fallbacks_and_event_pattern_guards():
    assert fd._first_nonmissing(pd.Series([pd.NA, "", "value"])) == "value"
    assert fd._first_nonmissing((np.nan, " ", 4)) == 4
    assert fd._first_nonmissing("single") == "single"
    assert fd._first_nonmissing(pd.NA, default="fallback") == "fallback"
    assert pd.isna(fd._mode_value([]))
    assert fd._mode_value(["b", "a", "b"]) == "b"
    assert fd._as_components("gaze_samples") == ["gaze_samples"]
    assert fd._as_components(("events", "biometrics")) == ["events", "biometrics"]
    assert np.isnan(fd._safe_min_finite([np.nan, np.inf]))
    assert np.isnan(fd._safe_max_finite([np.nan, -np.inf]))
    assert fd._safe_min_finite([2, 1, np.nan]) == 1
    assert fd._safe_max_finite([2, 1, np.nan]) == 2

    values = pd.Series(["TRIAL_START", "other", pd.NA])
    matched = fd._event_matches(("TRIAL_START", "other"), values)
    assert matched.tolist() == [True, True, False]
    with pytest.raises(ValueError, match="Invalid event pattern"):
        fd._event_matches(("[",), values)
    assert pd.isna(fd._participant_for_recording(_dataset(), "missing"))


def test_clock_sync_no_overlap_invalid_times_and_channel_sequence():
    x = _dataset()
    far = x.copy()
    far["biometrics"]["timestamp_seconds"] = [10.0, 11.0]
    report = ep.audit_clock_sync(far, channel=["eda"])
    assert report.loc[0, "status"] == "no_overlap"
    assert report.loc[0, "overlap_fraction"] == 0

    invalid = x.copy()
    invalid["biometrics"]["timestamp_seconds"] = np.nan
    assert ep.audit_clock_sync(invalid).empty


def test_build_trials_auto_id_recording_end_overwrite_and_no_match():
    x = _dataset()
    x["events"].loc[x["events"]["event_name"].eq("TRIAL_START"), "event_value"] = ""
    built = ep.build_trials(x, close_open="recording_end")
    trial = built["intervals"].query("interval_type == 'trial'").iloc[0]
    assert trial.trial_id.startswith("R1_trial_")

    rebuilt = ep.build_trials(built, overwrite=True)
    assert rebuilt["intervals"]["interval_type"].eq("trial").sum() == 1

    with pytest.raises(ValueError, match="No trial intervals"):
        ep.build_trials(_dataset(), start_events=("NEVER",), end_events=("NEVER_END",))


def test_build_trials_pattern_without_capture_and_invalid_interval_time():
    x = _dataset()
    x["events"].loc[0, "event_value"] = "trial=ABC"
    out = ep.build_trials(x, trial_id_pattern=r"trial=\w+")
    assert out["intervals"].query("interval_type == 'trial'").iloc[0].trial_id == "trial=ABC"

    broken = _dataset()
    broken["events"].loc[0, "timestamp_seconds"] = np.nan
    out = ep.build_trials(broken, close_open="recording_end")
    assert not bool(out["intervals"].query("interval_type == 'trial'").iloc[0].valid_interval)


def test_stimulus_runs_event_guard_and_overwrite():
    x = _dataset()
    x["gaze_samples"]["stimulus_id"] = ["A", "A", "B", "B", "A"]
    built = ep.build_stimulus_intervals(x)
    assert built["intervals"]["interval_type"].eq("stimulus").sum() == 3
    rebuilt = ep.build_stimulus_intervals(built, overwrite=True)
    assert rebuilt["intervals"]["interval_type"].eq("stimulus").sum() == 3

    no_media = _dataset()
    no_media["events"]["event_type"] = "other"
    no_media["events"]["event_name"] = "OTHER"
    with pytest.raises(ValueError, match="No stimulus/media events"):
        ep.build_stimulus_intervals(no_media, source="events")


def test_assign_trials_fills_episode_ids_and_preserves_existing_values():
    x = ep.build_trials(_dataset())
    episodes = _fixation_rows()
    episodes["trial_id"] = [pd.NA, "KEEP", pd.NA]
    x["episodes"] = ep.standardize_eye_table(episodes, "episodes")
    filled = ep.assign_trials(x, overwrite=False)
    assert filled["episodes"].trial_id.tolist() == ["T1", "KEEP", "T1"]
    overwritten = ep.assign_trials(filled, overwrite=True)
    assert overwritten["episodes"].trial_id.eq("T1").all()


def test_item_response_missing_times_and_response_overwrite_empty_input():
    x = ep.build_trials(_dataset())
    mask = x["intervals"]["interval_type"].eq("trial")
    x["intervals"].loc[mask, "item_id"] = "I1"
    x["intervals"].loc[mask, "end_time"] = np.nan
    generated = ep.build_item_responses(x)
    assert np.isnan(generated["responses"].response_time.iloc[0])

    empty = pd.DataFrame(columns=["participant_id", "item_id", "response"])
    out = ep.add_responses(_dataset(), empty, overwrite=True)
    assert out["responses"].empty


def test_polygon_and_aoi_contains_private_residuals():
    assert not fd._point_in_polygon([0.1], [0.1], None).any()
    assert not fd._point_in_polygon([0.1], [0.1], [[0, 0], [1, 1]]).any()
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert fd._point_in_polygon([0.5, 2.0], [0.5, 2.0], polygon).tolist() == [True, False]

    geometry = pd.Series(
        {
            "valid_from": 0.0,
            "valid_to": 1.0,
            "visible": False,
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
            "polygon": polygon,
        }
    )
    definition = pd.Series({"shape_type": "rectangle"})
    assert not fd._aoi_contains([0.5], [0.5], [0.5], definition, geometry).any()

    geometry["visible"] = True
    definition["shape_type"] = "circle"
    assert fd._aoi_contains([0.5], [0.5], [0.5], definition, geometry).all()
    definition["shape_type"] = "polygon"
    assert fd._aoi_contains([0.5], [0.5], [0.5], definition, geometry).all()
    definition["shape_type"] = "unknown"
    assert not fd._aoi_contains([0.5], [0.5], [0.5], definition, geometry).any()


def test_assign_aois_empty_stimulus_specific_infinite_area_and_episode_paths():
    x = _dataset()
    aoi = ep.new_aoi("A", stimulus_id="different", x=0, y=0, width=1, height=1)
    x = ep.register_aois(x, aoi)
    unmatched = ep.assign_aois(x)
    assert unmatched["gaze_samples"].aoi_id.isna().all()

    empty = x.copy()
    empty["gaze_samples"] = empty["gaze_samples"].iloc[0:0].copy()
    assert ep.assign_aois(empty)["gaze_samples"].empty

    area = ep.new_aoi("NAN", x=0, y=0, width=np.nan, height=np.nan)
    area_ds = ep.register_aois(_dataset(), area)
    smallest = ep.assign_aois(area_ds, overlap="smallest")
    assert smallest["gaze_samples"].aoi_id.isna().all()

    episode_ds = ep.register_aois(_dataset(), ep.new_aoi("A", x=0, y=0, width=0.5, height=0.5))
    episode_ds["episodes"] = ep.standardize_eye_table(_fixation_rows(), "episodes")
    episode_ds["episodes"]["aoi_id"] = ["KEEP", pd.NA, pd.NA]
    assigned = ep.assign_aois(episode_ds, component="episodes", overwrite=False)
    assert assigned["episodes"].aoi_id.iloc[0] == "KEEP"
    assert assigned["episodes"].aoi_id.notna().sum() >= 2

    empty_ep = episode_ds.copy()
    empty_ep["episodes"] = empty_ep["episodes"].iloc[0:0].copy()
    assert ep.assign_aois(empty_ep, component="episodes")["episodes"].empty


def test_build_aoi_visits_from_fixations_and_empty_episode_source():
    x = _dataset()
    x["episodes"] = ep.standardize_eye_table(_fixation_rows(), "episodes")
    visits = ep.build_aoi_visits(x, source="episodes", gap_tolerance_ms=100)
    made = visits["episodes"]["episode_type"].eq("aoi_visit")
    assert made.sum() == 2

    filtered = ep.build_aoi_visits(x, source="episodes", minimum_duration_ms=10_000)
    assert not filtered["episodes"]["episode_type"].eq("aoi_visit").any()

    empty = x.copy()
    empty["episodes"] = empty["episodes"].iloc[0:0].copy()
    assert ep.build_aoi_visits(empty, source="episodes")["episodes"].empty


def test_quality_row_store_standardization_replace_and_sampling_statuses():
    row = fd._quality_row(
        "R1",
        metric="m",
        value=pd.NA,
        threshold=pd.NA,
        status="unknown",
        message="x",
    )
    assert np.isnan(row.value.iloc[0])
    assert np.isnan(row.threshold.iloc[0])

    x = _dataset()
    minimal = pd.DataFrame({"recording_id": ["R1"], "metric": ["m"], "value": [1.0]})
    stored = ep.store_quality(x, minimal)
    assert len(stored["quality"]) == 1
    replacement = minimal.assign(value=2.0)
    stored = ep.store_quality(stored, replacement, replace_metric=True)
    assert len(stored["quality"]) == 1
    assert stored["quality"].value.iloc[0] == 2

    unknown = _dataset()
    unknown["recordings"]["nominal_sampling_rate"] = np.nan
    report = ep.audit_sampling_rate(unknown)
    assert report.status.iloc[0] == "unknown"
    warning = ep.audit_sampling_rate(_dataset(), expected_hz=10, tolerance_hz=0.1)
    assert warning.status.iloc[0] == "warning"
    stored_report = ep.audit_sampling_rate(_dataset(), expected_hz=2, store=True)
    assert len(stored_report["quality"]) == 1


def test_nullable_signal_quality_no_trial_and_store_paths():
    assert np.isnan(fd._nullable_valid_fraction([pd.NA], [False]))
    x = _dataset()
    x["gaze_samples"]["trial_id"] = pd.NA
    x["eye_samples"]["trial_id"] = pd.NA
    x["gaze_samples"]["valid"] = False
    report = ep.audit_signal_quality(x, by_trial=True)
    assert report.status.eq("warning").any()
    assert report.trial_id.isna().all()
    stored = ep.audit_signal_quality(x, store=True)
    assert not stored["quality"].empty


def test_pupil_quality_without_interpolated_and_store_branch():
    x = _dataset()
    assert "interpolated" not in x["eye_samples"].columns
    report = ep.audit_pupil_quality(x, maximum_interpolated_fraction=0)
    assert report.value.iloc[0] == 0
    stored = ep.audit_pupil_quality(x, plausible_range=(3.05, 3.1), store=True)
    assert stored["quality"].status.eq("warning").any()


def test_episode_event_trial_and_aoi_filtered_error_branches():
    x = _dataset()
    x["episodes"] = ep.standardize_eye_table(_fixation_rows(), "episodes")
    episodes = ep.audit_episodes(x, type=["fixation"])
    assert episodes.vendor_derived.iloc[0] == 2
    assert episodes.package_derived.iloc[0] == 1
    assert ep.audit_episodes(x, type="saccade").empty

    x["events"].loc[1, "timestamp_seconds"] = 0.0
    order = ep.audit_event_order(x, event_type=["trial"])
    assert order.n_duplicate_timestamps.iloc[0] == 1
    assert ep.audit_event_order(x, event_type="missing").empty

    built = ep.build_trials(_dataset())
    built["gaze_samples"] = built["gaze_samples"].iloc[0:0].copy()
    coverage = ep.audit_trial_coverage(built)
    assert coverage.n_gaze_samples.iloc[0] == 0

    bad_aoi = ep.register_aois(_dataset(), ep.new_aoi("BAD", coordinate_space_id="unknown", x=0, y=0, width=1, height=1))
    bad_aoi["aoi_geometry"] = bad_aoi["aoi_geometry"].iloc[0:0].copy()
    audit = ep.audit_aois(bad_aoi)
    assert audit.status.iloc[0] == "error"
    assert not bool(audit.has_geometry.iloc[0])
    assert not bool(audit.coordinate_registered.iloc[0])


def test_missingness_infinite_grouping_leakage_and_feature_level_residuals():
    x = _dataset()
    x["gaze_samples"].loc[0, "gaze_x"] = np.inf
    report = ep.audit_missingness(x, by=["recording_id", "trial_id"])
    gaze_x = report[report.field.eq("gaze_x")]
    assert gaze_x.missing_fraction.max() > 0

    x = ep.build_trials(_dataset())
    feature = pd.DataFrame(
        [
            {
                "feature_id": "F1",
                "recording_id": "R1",
                "trial_id": "T1",
                "feature_name": "early",
                "window_end": 1.0,
                "level": "unknown",
            }
        ]
    )
    x["features"] = ep.standardize_eye_table(feature, "features")
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
    leakage = ep.check_process_leakage(x, response_time_tolerance=0.1)
    assert leakage.status.iloc[0] == "ok"
    levels = ep.check_feature_level(x)
    assert levels.status.iloc[0] == "ok"

    orphan = x.copy()
    orphan["responses"] = orphan["responses"].iloc[0:0].copy()
    assert ep.check_process_leakage(orphan).empty


def test_analysis_readiness_forced_audit_edges(monkeypatch):
    x = _dataset()
    monkeypatch.setattr(fd, "validate_eye_dataset", lambda value: pd.DataFrame())
    monkeypatch.setattr(
        fd,
        "audit_signal_quality",
        lambda value: pd.DataFrame({"status": ["warning"]}),
    )
    monkeypatch.setattr(fd, "audit_coordinate_spaces", lambda value: pd.DataFrame())
    monkeypatch.setattr(
        fd,
        "audit_timebase",
        lambda value: pd.DataFrame({"status": ["warning"]}),
    )
    monkeypatch.setattr(
        fd,
        "audit_trial_coverage",
        lambda value: pd.DataFrame({"status": ["error"]}),
    )
    ready = ep.analysis_readiness(x).set_index("domain").ready
    assert bool(ready.schema)
    assert not bool(ready.timestamps)
    assert bool(ready.coordinates)
    assert not bool(ready.trials)
    assert not bool(ready.gaze_quality)


def test_input_normalisation_comparison_guards_and_episode_aoi_counts():
    x = _dataset()
    assert [name for name, _ in fd._normalise_dataset_inputs(({"named": x},))] == ["named"]
    assert [name for name, _ in fd._normalise_dataset_inputs(([x, x],))] == ["pipeline_1", "pipeline_2"]
    assert [name for name, _ in fd._normalise_dataset_inputs((x, x))] == ["pipeline_1", "pipeline_2"]

    with pytest.raises(TypeError, match="eye_dataset"):
        ep.compare_preprocessing(x, object())

    x["episodes"] = ep.standardize_eye_table(_fixation_rows(), "episodes")
    counts = ep.compare_aoi_definitions({"episodes": x}, source="episodes")
    assert set(counts.aoi_id.dropna()) == {"A", "B"}
    assert counts.definition_set.eq("episodes").all()

    no_label_change = ep.sensitivity_process(x, label=["a", "b"])
    assert no_label_change["summary"].pipeline.iloc[0] == "pipeline_1"
