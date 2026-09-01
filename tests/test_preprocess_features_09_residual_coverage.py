from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.preprocess_features_09 as pf


def _dataset(*, with_episodes: bool = True):
    recordings = pd.DataFrame(
        [{"recording_id": "R1", "participant_id": "P1", "vendor": "Generic", "nominal_sampling_rate": 10.0}]
    )
    streams = pd.DataFrame(
        [{
            "stream_id": "G1",
            "recording_id": "R1",
            "stream_type": "gaze_combined",
            "timestamp_unit": "seconds",
            "observed_rate_hz": 10.0,
        }]
    )
    spaces = ep.new_coordinate_space("C1")
    t = np.arange(0.0, 0.7, 0.1)
    gaze = pd.DataFrame(
        {
            "recording_id": ["R1"] * len(t),
            "stream_id": ["G1"] * len(t),
            "sample_id": [f"G{i}" for i in range(len(t))],
            "timestamp_native": t,
            "timestamp_seconds": t,
            "gaze_x": [0.10, 0.11, 0.12, 0.70, 0.71, 0.20, 0.21],
            "gaze_y": [0.10, 0.11, 0.12, 0.70, 0.71, 0.20, 0.21],
            "valid": [True] * len(t),
            "trial_id": ["T1"] * len(t),
            "stimulus_id": ["S1"] * len(t),
            "coordinate_space_id": ["C1"] * len(t),
            "aoi_id": ["A", "A", "B", "B", "A", pd.NA, "C"],
        }
    )
    pupil = np.array([3.0, 3.1, np.nan, 3.3, 3.4, 3.5, 3.6])
    eyes = pd.DataFrame(
        {
            "recording_id": ["R1"] * len(t),
            "sample_id": [f"P{i}" for i in range(len(t))],
            "timestamp_native": t,
            "timestamp_seconds": t,
            "eye": ["left"] * len(t),
            "pupil_diameter": pupil,
            "pupil_unit": ["mm"] * len(t),
            "pupil_valid": np.isfinite(pupil),
            "trial_id": ["T1"] * len(t),
            "stimulus_id": ["S1"] * len(t),
        }
    )
    intervals = pd.DataFrame(
        [
            {
                "interval_id": "I1",
                "recording_id": "R1",
                "interval_type": "trial",
                "start_time": 0.0,
                "end_time": 0.6,
                "trial_id": "T1",
                "participant_id": "P1",
                "item_id": "ITEM1",
                "stimulus_id": "S1",
                "valid_interval": True,
            }
        ]
    )
    episodes = pd.DataFrame()
    if with_episodes:
        episodes = pd.DataFrame(
            [
                {
                    "episode_id": "F1",
                    "recording_id": "R1",
                    "episode_type": "fixation",
                    "eye": "combined",
                    "start_time": 0.05,
                    "end_time": 0.15,
                    "duration_ms": 100.0,
                    "coordinate_space_id": "C1",
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
                    "start_time": 0.20,
                    "end_time": 0.32,
                    "duration_ms": 120.0,
                    "coordinate_space_id": "C1",
                    "derived_by": "eyeprocess",
                    "trial_id": "T1",
                    "stimulus_id": "S1",
                    "aoi_id": "B",
                },
                {
                    "episode_id": "V1",
                    "recording_id": "R1",
                    "episode_type": "aoi_visit",
                    "eye": "combined",
                    "start_time": 0.05,
                    "end_time": 0.18,
                    "duration_ms": 130.0,
                    "coordinate_space_id": "C1",
                    "derived_by": "eyeprocess",
                    "trial_id": "T1",
                    "stimulus_id": "S1",
                    "aoi_id": "A",
                },
                {
                    "episode_id": "V2",
                    "recording_id": "R1",
                    "episode_type": "aoi_visit",
                    "eye": "combined",
                    "start_time": 0.22,
                    "end_time": 0.35,
                    "duration_ms": 130.0,
                    "coordinate_space_id": "C1",
                    "derived_by": "eyeprocess",
                    "trial_id": "T1",
                    "stimulus_id": "S1",
                    "aoi_id": "B",
                },
            ]
        )
    return ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        coordinate_spaces=spaces,
        gaze_samples=gaze,
        eye_samples=eyes,
        intervals=intervals,
        episodes=episodes,
        validate=False,
    )


def test_private_numeric_and_group_helpers_cover_empty_and_finite_contracts():
    assert pd.isna(pf._mode_value([pd.NA, np.nan]))
    assert pf._mode_value(["B", "A", "B"]) == "B"
    assert pf._first_nonmissing([None, " ", np.nan], "fallback") == "fallback"
    assert pf._first_nonmissing([None, " value ", "later"]) == " value "
    assert np.isnan(pf._safe_span([np.nan]))
    assert np.isnan(pf._safe_max([np.nan]))
    assert np.isnan(pf._mad([np.nan]))
    assert pf._mad([1.0, 2.0, 3.0]) > 0
    assert np.isnan(pf._trapz([], []))
    assert pf._trapz([2.0], [3.0]) == 0.0
    assert pf._trapz([2.0, 0.0, 1.0], [2.0, 0.0, 1.0]) == pytest.approx(2.0)

    d = pd.DataFrame({"x": [1, 2], "g": [pd.NA, pd.NA]})
    assert pf._group_frames(d.iloc[0:0], ["g"]) == []
    frames = pf._group_frames(d, ["missing"])
    assert len(frames) == 1 and len(frames[0]) == 2
    assert pf._group_frames(d, ["g"], dropna=True) == []
    assert len(pf._group_frames(d, ["g"], dropna=False)) == 1


def test_feature_rows_empty_scalar_and_single_unit_expansion():
    assert pf._feature_rows({}, {}, "unit", "trial", "test").empty
    one = pf._feature_rows({"recording_id": "R1"}, {"a": 1.0, "b": 2.0}, "unit", "trial", "test")
    assert one["unit"].tolist() == ["unit", "unit"]
    expanded = pf._feature_rows({"recording_id": "R1"}, {"a": 1.0, "b": 2.0}, ["u"], "trial", "test")
    assert expanded["unit"].tolist() == ["u", "u"]


def test_flag_gaze_outliers_mad_duplicate_velocity_and_pixel_bounds():
    x = _dataset()
    mad = x.copy()
    mad["gaze_samples"].loc[:, "gaze_x"] = [0.10, 0.11, 0.10, 0.11, 0.10, 5.0, np.nan]
    mad["gaze_samples"].loc[:, "gaze_y"] = [0.10, 0.10, 0.11, 0.10, 0.11, 0.10, np.nan]
    flagged = ep.flag_gaze_outliers(mad, method="mad", threshold=3)
    assert bool(flagged["gaze_samples"].loc[5, "outlier_flag"])
    assert not bool(flagged["gaze_samples"].loc[6, "outlier_flag"])

    duplicate = x.copy()
    duplicate["gaze_samples"].loc[1, "sample_id"] = duplicate["gaze_samples"].loc[0, "sample_id"]
    duplicate["gaze_samples"].loc[1, "timestamp_seconds"] = duplicate["gaze_samples"].loc[0, "timestamp_seconds"]
    velocity = ep.flag_gaze_outliers(duplicate, method="velocity", threshold=0.01)
    assert "outlier_flag" in velocity["gaze_samples"]

    pixels = x.copy()
    spaces = pixels["coordinate_spaces"].copy()
    spaces.loc[0, "x_unit"] = "pixels"
    spaces.loc[0, "y_unit"] = "pixels"
    spaces.loc[0, "width"] = 100.0
    spaces.loc[0, "height"] = 100.0
    pixels["coordinate_spaces"] = spaces
    pixels["gaze_samples"].loc[0, ["gaze_x", "gaze_y"]] = [101.0, 50.0]
    pixels["gaze_samples"].loc[1, "coordinate_space_id"] = pd.NA
    bounded = ep.flag_gaze_outliers(pixels, method="bounds")
    assert bool(bounded["gaze_samples"].loc[0, "outlier_flag"])
    assert not bool(bounded["gaze_samples"].loc[1, "outlier_flag"])


def test_interpolation_too_wide_gap_and_filter_pupil_mean_path():
    x = _dataset()
    too_wide = ep.interpolate_pupil(x, method="linear", max_gap_ms=50)
    assert np.isnan(too_wide["eye_samples"].loc[2, "pupil_diameter"])
    filtered = ep.filter_pupil(x, method="moving_average", window=3)
    assert "pupil_raw" in filtered["eye_samples"]
    assert np.isfinite(filtered["eye_samples"]["pupil_diameter"]).any()


def test_blink_missing_source_duration_rejection_and_append_path():
    x = _dataset()
    short = ep.detect_blinks(x, source="pupil_missing", min_duration_ms=200, max_duration_ms=500)
    assert not short["episodes"]["episode_type"].eq("blink").any()

    y = x.copy()
    y["eye_samples"].loc[2:4, "pupil_diameter"] = np.nan
    appended = ep.detect_blinks(y, source="pupil_missing", min_duration_ms=100, max_duration_ms=500)
    assert appended["episodes"]["episode_type"].eq("blink").any()
    assert appended["episodes"]["episode_type"].eq("fixation").any()


def test_idt_guard_empty_non_degree_and_overwrite_paths():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid coordinate units"):
        ep.detect_fixations_idt(x, coordinate_units="meters")
    empty = x.copy()
    empty["gaze_samples"] = empty["gaze_samples"].iloc[0:0].copy()
    assert ep.detect_fixations_idt(empty) is empty

    found = ep.detect_fixations_idt(
        x,
        dispersion_threshold=2.0,
        minimum_duration_ms=0,
        coordinate_units="normalized",
        overwrite=True,
    )
    assert found["episodes"]["episode_type"].eq("fixation").any()
    prov = found["provenance"].iloc[-1]
    assert "not visual degrees" in str(prov.get("warnings", ""))


def test_saccade_empty_detection_and_overwrite_paths():
    x = _dataset()
    empty = x.copy()
    empty["gaze_samples"] = empty["gaze_samples"].iloc[0:0].copy()
    assert ep.detect_saccades(empty) is empty
    detected = ep.detect_saccades(x, velocity_threshold=0.5, minimum_duration_ms=0, overwrite=True)
    assert detected["episodes"]["episode_type"].eq("saccade").any()


def test_preprocess_eye_spec_guard_idt_and_unknown_algorithm():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="preprocess_spec"):
        ep.preprocess_eye(x, spec=object())

    spec = ep.preprocess_spec(
        gaze_filter="none",
        pupil_interpolation="none",
        pupil_filter="none",
        pupil_baseline="none",
        blink_detection=False,
        fixation_algorithm="idt",
        fixation_parameters={
            "dispersion_threshold": 2.0,
            "minimum_duration_ms": 0,
            "coordinate_units": "normalized",
        },
    )
    out = ep.preprocess_eye(x, spec)
    assert out["episodes"]["episode_type"].eq("fixation").any()

    bad = ep.preprocess_spec(
        gaze_filter="none",
        pupil_interpolation="none",
        pupil_filter="none",
        pupil_baseline="none",
        blink_detection=False,
        fixation_algorithm="vendor_magic",
    )
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown fixation algorithm"):
        ep.preprocess_eye(x, bad)


def test_summarize_fixations_source_filters_and_empty_grouping():
    x = _dataset()
    vendor = ep.summarize_fixations(x, source="vendor")
    derived = ep.summarize_fixations(x, source="eyeprocess")
    assert vendor["fixation_count"].sum() == 1
    assert derived["fixation_count"].sum() == 1
    assert ep.summarize_fixations(x, by=("does_not_exist",)).empty
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid fixation source"):
        ep.summarize_fixations(x, source="bad")


def test_scanpath_samples_filters_collapse_and_invalid_paths():
    x = _dataset()
    collapsed = ep.scanpath_sequence(x, source="samples")
    assert collapsed.loc[0, "sequence"] == "A > B > A > C"
    uncollapsed = ep.scanpath_sequence(x, source="samples", collapse_consecutive=False)
    assert uncollapsed.loc[0, "length"] > collapsed.loc[0, "length"]
    assert ep.scanpath_sequence(x, source="samples", trial_id=["missing"]).empty
    assert ep.scanpath_sequence(x, source="samples", recording_id=["missing"]).empty
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid scanpath source"):
        ep.scanpath_sequence(x, source="bad")

    missing = x.copy()
    missing["gaze_samples"] = missing["gaze_samples"].drop(columns=["aoi_id"])
    with pytest.raises(ep.EyeProcessValidationError, match="AOIs have not been assigned"):
        ep.scanpath_sequence(missing, source="samples")


def test_transition_matrix_normalizations_self_and_empty_contracts():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid transition normalization"):
        ep.transition_matrix(x, normalize="column", source="samples")
    raw = ep.transition_matrix(x, normalize="none", source="samples")
    row = ep.transition_matrix(x, normalize="row", source="samples")
    all_norm = ep.transition_matrix(x, normalize="all", source="samples")
    assert raw.to_numpy().sum() > 0
    assert np.all(row.sum(axis=1).to_numpy() <= 1.0 + 1e-12)
    assert all_norm.to_numpy().sum() == pytest.approx(1.0)
    self_matrix = ep.transition_matrix(x, normalize="none", source="samples", include_self=True)
    assert self_matrix.loc["A", "A"] >= 1

    no_aoi = x.copy()
    no_aoi["gaze_samples"]["aoi_id"] = pd.NA
    assert ep.transition_matrix(no_aoi, source="samples").empty


def test_gaze_and_transition_entropy_sources_and_recording_level():
    x = _dataset()
    samples = ep.gaze_entropy(x, source="samples", level="recording")
    visits = ep.gaze_entropy(x, source="visits", level="trial")
    assert samples.loc[0, "n_states"] == 3
    assert visits.loc[0, "n_states"] == 2
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid entropy source"):
        ep.gaze_entropy(x, source="bad")

    missing = x.copy()
    missing["gaze_samples"] = missing["gaze_samples"].drop(columns=["aoi_id"])
    with pytest.raises(ep.EyeProcessValidationError, match="AOIs have not been assigned"):
        ep.gaze_entropy(missing, source="samples")

    transition = ep.transition_entropy(x, source="samples")
    assert {"aoi_id", "transition_entropy"} <= set(transition.columns)
    no_aoi = x.copy()
    no_aoi["gaze_samples"]["aoi_id"] = pd.NA
    assert ep.transition_entropy(no_aoi, source="samples").empty


def test_derive_gaze_features_samples_invalid_empty_append_and_trial_mismatch():
    x = _dataset()
    samples = ep.derive_gaze_features(x, source="samples", level="trial")
    assert samples["features"]["feature_name"].eq("fixation_count").any()
    assert samples["features"]["feature_name"].eq("dwell_time_ms").any()
    n = len(samples["features"])
    twice = ep.derive_gaze_features(samples, source="samples", level="trial", append=True)
    assert len(twice["features"]) > n

    with pytest.raises(ep.EyeProcessValidationError, match="Invalid gaze-feature source"):
        ep.derive_gaze_features(x, source="bad")
    no_trials = x.copy()
    no_trials["intervals"] = no_trials["intervals"].iloc[0:0].copy()
    with pytest.raises(ep.EyeProcessValidationError, match="Trial intervals"):
        ep.derive_gaze_features(no_trials, source="samples")

    no_visits = _dataset(with_episodes=False)
    assert ep.derive_gaze_features(no_visits, source="visits") is no_visits

    mismatch = x.copy()
    mismatch["gaze_samples"]["trial_id"] = "OTHER"
    derived = ep.derive_gaze_features(mismatch, source="samples", level="trial", append=False)
    assert derived["features"].empty


def test_derive_pupil_feature_empty_missing_warning_and_no_finite_paths():
    x = _dataset()
    with pytest.warns(RuntimeWarning, match="not directly AOI-labelled"):
        out = ep.derive_pupil_features(x, level="trial_aoi")
    assert out["features"]["feature_name"].eq("pupil_mean").any()

    missing = x.copy()
    missing["eye_samples"] = missing["eye_samples"].drop(columns=["pupil_diameter"])
    with pytest.raises(ep.EyeProcessValidationError, match="Pupil column"):
        ep.derive_pupil_features(missing)

    empty = x.copy()
    empty["eye_samples"] = empty["eye_samples"].iloc[0:0].copy()
    assert ep.derive_pupil_features(empty) is empty

    no_finite = x.copy()
    no_finite["eye_samples"]["pupil_diameter"] = np.nan
    derived = ep.derive_pupil_features(no_finite, append=False)
    assert derived["features"].empty


def test_derive_all_features_spec_guard_and_reset_samples_path():
    x = _dataset(with_episodes=False)
    with pytest.raises(ep.EyeProcessValidationError, match="feature_spec"):
        ep.derive_all_features(x, spec=object())
    spec = ep.feature_spec(level="trial", response_time=False, biometrics=False)
    out = ep.derive_all_features(x, spec=spec, reset=True)
    assert not out["features"].empty


def test_rt_biometric_empty_and_features_wide_no_id_contracts():
    x = _dataset()
    assert ep.derive_rt_features(x) is x
    assert ep.derive_biometric_features(x) is x
    assert ep.features_wide(x).empty

    featured = ep.derive_gaze_features(x, source="samples", level="trial")
    wide = ep.features_wide(featured, id_cols=())
    assert len(wide) == 1
    dictionary = ep.feature_dictionary(featured)
    assert not dictionary.empty
