from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGET_EXPORTS = {
    "baseline_pupil",
    "derive_all_features",
    "derive_biometric_features",
    "derive_gaze_features",
    "derive_pupil_features",
    "derive_rt_features",
    "detect_blinks",
    "detect_fixations_idt",
    "detect_fixations_ivt",
    "detect_saccades",
    "feature_dictionary",
    "feature_spec",
    "features_wide",
    "filter_gaze",
    "filter_pupil",
    "flag_gaze_outliers",
    "gaze_entropy",
    "gaze_velocity",
    "interpolate_pupil",
    "preprocess_eye",
    "preprocess_spec",
    "pupil_deconvolve",
    "rolling_apply",
    "scanpath_sequence",
    "summarize_fixations",
    "transition_entropy",
    "transition_matrix",
    "trial_table",
}


def _dataset():
    recordings = pd.DataFrame(
        [
            {
                "recording_id": "R1",
                "participant_id": "P1",
                "nominal_sampling_rate": 10.0,
            }
        ]
    )
    streams = pd.DataFrame(
        [
            {
                "stream_id": "G1",
                "recording_id": "R1",
                "stream_type": "gaze_combined",
                "observed_rate_hz": 10.0,
            },
            {
                "stream_id": "B1",
                "recording_id": "R1",
                "stream_type": "biometric",
                "observed_rate_hz": 10.0,
            },
        ]
    )
    spaces = ep.new_coordinate_space("C1")
    t = np.arange(0.0, 1.1, 0.1)
    gaze = pd.DataFrame(
        {
            "recording_id": "R1",
            "stream_id": "G1",
            "sample_id": [f"G{i}" for i in range(len(t))],
            "timestamp_seconds": t,
            "gaze_x": [0.10, 0.11, 0.12, 0.13, 0.70, 0.72, 0.74, 0.20, 0.21, 0.22, 0.23],
            "gaze_y": [0.10, 0.11, 0.12, 0.13, 0.70, 0.71, 0.72, 0.20, 0.21, 0.22, 0.23],
            "valid": True,
            "trial_id": "T1",
            "stimulus_id": "S1",
            "coordinate_space_id": "C1",
            "aoi_id": ["A", "A", "A", "A", "B", "B", "B", "A", "A", "A", "A"],
        }
    )
    pupil = np.array([3.0, 3.0, np.nan, 3.2, 3.4, 3.5, 3.6, 3.5, 3.4, 3.3, 3.2])
    eyes = pd.DataFrame(
        {
            "recording_id": "R1",
            "sample_id": [f"P{i}" for i in range(len(t))],
            "timestamp_seconds": t,
            "eye": "left",
            "pupil_diameter": pupil,
            "pupil_unit": "mm",
            "pupil_valid": np.isfinite(pupil),
            "trial_id": "T1",
            "stimulus_id": "S1",
        }
    )
    intervals = pd.DataFrame(
        [
            {
                "interval_id": "I1",
                "recording_id": "R1",
                "interval_type": "trial",
                "start_time": 0.0,
                "end_time": 1.0,
                "trial_id": "T1",
                "participant_id": "P1",
                "item_id": "ITEM1",
                "stimulus_id": "S1",
                "valid_interval": True,
            }
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "response_id": "RESP1",
                "recording_id": "R1",
                "participant_id": "P1",
                "trial_id": "T1",
                "item_id": "ITEM1",
                "response": "yes",
                "score": 1.0,
                "response_time": 0.8,
                "response_timestamp": 0.8,
                "response_type": "observed",
                "valid_response": True,
            }
        ]
    )
    biometrics = pd.DataFrame(
        {
            "recording_id": "R1",
            "stream_id": "B1",
            "timestamp_seconds": t,
            "channel": "eda",
            "value": np.linspace(1.0, 2.0, len(t)),
            "unit": "uS",
            "valid": True,
            "trial_id": "T1",
            "stimulus_id": "S1",
        }
    )
    return ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        coordinate_spaces=spaces,
        gaze_samples=gaze,
        eye_samples=eyes,
        intervals=intervals,
        responses=responses,
        biometrics=biometrics,
    )


def _with_fixations():
    x = _dataset()
    episodes = pd.DataFrame(
        [
            {
                "episode_id": "F1",
                "recording_id": "R1",
                "episode_type": "fixation",
                "eye": "combined",
                "start_time": 0.0,
                "end_time": 0.3,
                "duration_ms": 300.0,
                "centroid_x": 0.115,
                "centroid_y": 0.115,
                "coordinate_space_id": "C1",
                "source_algorithm": "vendor",
                "derived_by": "vendor",
                "trial_id": "T1",
                "stimulus_id": "S1",
                "aoi_id": "A",
            },
            {
                "episode_id": "F2",
                "recording_id": "R1",
                "episode_type": "fixation",
                "eye": "combined",
                "start_time": 0.4,
                "end_time": 0.6,
                "duration_ms": 200.0,
                "centroid_x": 0.72,
                "centroid_y": 0.71,
                "coordinate_space_id": "C1",
                "source_algorithm": "vendor",
                "derived_by": "vendor",
                "trial_id": "T1",
                "stimulus_id": "S1",
                "aoi_id": "B",
            },
            {
                "episode_id": "F3",
                "recording_id": "R1",
                "episode_type": "fixation",
                "eye": "combined",
                "start_time": 0.7,
                "end_time": 1.0,
                "duration_ms": 300.0,
                "centroid_x": 0.215,
                "centroid_y": 0.215,
                "coordinate_space_id": "C1",
                "source_algorithm": "vendor",
                "derived_by": "vendor",
                "trial_id": "T1",
                "stimulus_id": "S1",
                "aoi_id": "A",
            },
        ]
    )
    x["episodes"] = ep.standardize_eye_table(episodes, "episodes")
    return x


def test_all_frozen_exports_are_public():
    assert all(callable(getattr(ep, name)) for name in TARGET_EXPORTS)


def test_specs_are_concrete_and_validated():
    p = ep.preprocess_spec(fixation_algorithm="ivt", fixation_parameters={"velocity_threshold": 5})
    assert p.fixation_algorithm == "ivt"
    f = ep.feature_spec(level="trial_aoi")
    assert f.level == "trial_aoi"
    with pytest.raises(ep.EyeProcessValidationError):
        ep.feature_spec(level="bad")


def test_rolling_and_gaze_filter():
    y = ep.rolling_apply([1, 9, 1], width=3)
    assert y.tolist() == [5.0, 1.0, 5.0]
    x = ep.filter_gaze(_dataset(), method="mean", window=3)
    assert "filter_gaze" in set(x["provenance"]["action"])
    assert len(x["gaze_samples"]) == 11


def test_gaze_velocity_and_outlier_flagging():
    v = ep.gaze_velocity(_dataset())
    assert len(v) == 11
    assert np.isnan(v.iloc[0]["velocity"])
    x = _dataset()
    x["gaze_samples"].loc[0, "gaze_x"] = 1.5
    out = ep.flag_gaze_outliers(x, method="bounds")
    assert bool(out["gaze_samples"].iloc[0]["outlier_flag"])


def test_pupil_interpolation_filter_baseline_and_deconvolution():
    x = ep.interpolate_pupil(_dataset(), max_gap_ms=300)
    assert np.isfinite(float(x["eye_samples"].iloc[2]["pupil_diameter"]))
    assert bool(x["eye_samples"].iloc[2]["interpolated"])
    x = ep.filter_pupil(x, method="median", window=3)
    assert "pupil_raw" in x["eye_samples"]
    x = ep.baseline_pupil(
        x,
        method="subtract",
        baseline_window=(0.0, 0.3),
        minimum_samples=2,
    )
    assert "pupil_baseline" in x["eye_samples"]
    x = ep.pupil_deconvolve(x)
    assert "pupil_phasic" in x["eye_samples"]


def test_detect_blinks_from_missing_pupil():
    x = _dataset()
    # Make a 200 ms internal missing run.
    x["eye_samples"].loc[2:4, "pupil_diameter"] = np.nan
    out = ep.detect_blinks(x, min_duration_ms=100, max_duration_ms=500)
    blinks = out["episodes"][out["episodes"]["episode_type"].eq("blink")]
    assert len(blinks) == 1
    assert blinks.iloc[0]["source_algorithm"] == "eyeprocess_pupil_missing"


def test_ivt_and_idt_detectors_create_eyeprocess_fixations():
    x = _dataset()
    ivt = ep.detect_fixations_ivt(
        x,
        velocity_threshold=2.0,
        minimum_duration_ms=100,
        maximum_gap_ms=150,
        coordinate_units="normalized",
    )
    assert (ivt["episodes"]["episode_type"] == "fixation").any()
    idt = ep.detect_fixations_idt(
        x,
        dispersion_threshold=0.08,
        minimum_duration_ms=100,
        coordinate_units="normalized",
    )
    assert (idt["episodes"]["episode_type"] == "fixation").any()


def test_saccade_detection():
    out = ep.detect_saccades(_dataset(), velocity_threshold=2.0, minimum_duration_ms=0)
    assert (out["episodes"]["episode_type"] == "saccade").any()


def test_preprocess_eye_pipeline():
    spec = ep.preprocess_spec(
        gaze_filter="median",
        pupil_interpolation="linear",
        pupil_max_gap_ms=300,
        pupil_filter="median",
        pupil_baseline="none",
        blink_detection=False,
        fixation_algorithm="none",
    )
    out = ep.preprocess_eye(_dataset(), spec)
    assert out["provenance"]["action"].iloc[-1] == "preprocess_eye"


def test_trial_table_and_fixation_summary():
    x = _with_fixations()
    assert len(ep.trial_table(x)) == 1
    s = ep.summarize_fixations(x, by=("recording_id", "trial_id"), source="vendor")
    assert s.iloc[0]["fixation_count"] == 3
    assert s.iloc[0]["fixation_duration_total_ms"] == pytest.approx(800)


def test_scanpath_transition_and_entropy_contracts():
    x = _with_fixations()
    seq = ep.scanpath_sequence(x, source="fixations")
    assert seq.iloc[0]["sequence"] == "A > B > A"
    mat = ep.transition_matrix(x, source="fixations")
    assert mat.loc["A", "B"] == 1
    assert mat.loc["B", "A"] == 1
    ent = ep.gaze_entropy(x, source="fixations")
    assert ent.iloc[0]["n_states"] == 2
    tent = ep.transition_entropy(x, source="fixations")
    assert set(tent["aoi_id"]) == {"A", "B"}


def test_derive_gaze_features():
    out = ep.derive_gaze_features(_with_fixations(), level="trial", source="fixations")
    names = set(out["features"]["feature_name"])
    assert {"fixation_count", "dwell_time_ms", "gaze_entropy"} <= names
    dwell = out["features"].loc[out["features"]["feature_name"].eq("dwell_time_ms"), "value"]
    assert float(dwell.iloc[0]) == pytest.approx(800.0)


def test_derive_pupil_features():
    out = ep.derive_pupil_features(_dataset())
    names = set(out["features"]["feature_name"])
    assert {"pupil_mean", "pupil_auc", "pupil_latency_peak_ms"} <= names


def test_derive_rt_features():
    out = ep.derive_rt_features(_dataset())
    assert set(out["features"]["feature_name"]) == {"response_time", "score"}


def test_derive_biometric_features():
    out = ep.derive_biometric_features(_dataset())
    names = set(out["features"]["feature_name"])
    assert {"eda_mean", "eda_auc", "eda_observed_fraction"} <= names


def test_derive_all_features_and_dictionary():
    out = ep.derive_all_features(_with_fixations(), reset=True)
    assert not out["features"].empty
    dictionary = ep.feature_dictionary(out)
    assert dictionary["feature_name"].is_unique


def test_features_wide():
    out = ep.derive_all_features(_with_fixations(), reset=True)
    wide = ep.features_wide(out)
    assert len(wide) >= 1
    assert "pupil_mean" in wide.columns
    assert "response_time" in wide.columns


def test_empty_contracts_do_not_create_spurious_rows():
    x = _dataset()
    x["features"] = ep.empty_eye_table("features")
    assert ep.features_wide(x).empty
    assert ep.feature_dictionary(x).empty
