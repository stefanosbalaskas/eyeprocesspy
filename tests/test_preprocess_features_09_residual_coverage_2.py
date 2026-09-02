from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.preprocess_features_09 as pf


def _dataset():
    recordings = pd.DataFrame(
        [{"recording_id": "R1", "participant_id": "P1", "nominal_sampling_rate": 10.0}]
    )
    streams = pd.DataFrame(
        [{"stream_id": "G1", "recording_id": "R1", "stream_type": "gaze_combined"}]
    )
    spaces = ep.new_coordinate_space("C1")
    t = np.arange(0.0, 0.7, 0.1)
    gaze = pd.DataFrame(
        {
            "recording_id": ["R1"] * len(t),
            "stream_id": ["G1"] * len(t),
            "sample_id": [f"G{i}" for i in range(len(t))],
            "timestamp_seconds": t,
            "gaze_x": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
            "gaze_y": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
            "valid": [True] * len(t),
            "trial_id": ["T1"] * len(t),
            "stimulus_id": ["S1"] * len(t),
            "coordinate_space_id": ["C1"] * len(t),
        }
    )
    pupil = np.array([3.0, 3.1, np.nan, 3.3, 3.4, 3.5, 3.6])
    eyes = pd.DataFrame(
        {
            "recording_id": ["R1"] * len(t),
            "sample_id": [f"P{i}" for i in range(len(t))],
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
    return ep.new_eye_dataset(
        recordings=recordings,
        streams=streams,
        coordinate_spaces=spaces,
        gaze_samples=gaze,
        eye_samples=eyes,
        intervals=intervals,
    )


def _empty_component(x, name):
    x[name] = x[name].iloc[0:0].copy()


def test_private_run_and_filter_baseline_residual_paths():
    empty_episode = pf._episode_frame([])
    assert empty_episode.empty

    runs = pf._missing_runs(np.array([False, True, True]))
    assert len(runs) == 1 and runs[0].tolist() == [1, 2]
    assert pf._missing_runs(np.array([True]))[0].tolist() == [0]

    x = _dataset()
    assert ep.filter_pupil(x, method=["none"]) is x
    assert ep.baseline_pupil(x, method=["none"], anchor=["trial_start"]) is x

    existing = x.copy()
    existing["eye_samples"]["pupil_raw"] = existing["eye_samples"]["pupil_diameter"]
    filtered = ep.filter_pupil(existing, method="mean", window=3)
    assert "pupil_raw" in filtered["eye_samples"]

    existing2 = x.copy()
    existing2["eye_samples"]["pupil_uncorrected"] = existing2["eye_samples"]["pupil_diameter"]
    based = ep.baseline_pupil(
        existing2,
        method="subtract",
        anchor="recording_start",
        baseline_window=(0.0, 0.3),
        minimum_samples=2,
    )
    assert "pupil_baseline" in based["eye_samples"]


def test_bounds_registered_normalized_and_pixels_loop_paths():
    x = _dataset()
    x["gaze_samples"].loc[0, ["gaze_x", "gaze_y"]] = [1.2, -0.1]
    normalized = ep.flag_gaze_outliers(x, method="bounds")
    assert bool(normalized["gaze_samples"]["outlier_flag"].iloc[0])

    px = _dataset()
    px["coordinate_spaces"].loc[:, "x_unit"] = "pixels"
    px["coordinate_spaces"].loc[:, "y_unit"] = "pixels"
    px["coordinate_spaces"].loc[:, "width"] = 100.0
    px["coordinate_spaces"].loc[:, "height"] = 100.0
    px["gaze_samples"].loc[:, "gaze_x"] = [10, 20, 30, 40, 50, 60, 120]
    px["gaze_samples"].loc[:, "gaze_y"] = [10, 20, 30, 40, 50, 60, 20]
    pixels = ep.flag_gaze_outliers(px, method="bounds")
    assert bool(pixels["gaze_samples"]["outlier_flag"].iloc[-1])


def test_blink_nonfinite_duration_defensive_continue():
    x = _dataset()
    x["eye_samples"].loc[:, "pupil_diameter"] = [3.0, 3.1, 3.2, 3.3, 3.4, np.nan, np.nan]
    x["eye_samples"].loc[:, "pupil_valid"] = x["eye_samples"]["pupil_diameter"].notna()
    x["eye_samples"].loc[6, "timestamp_seconds"] = np.nan
    out = ep.detect_blinks(x, min_duration_ms=0, max_duration_ms=10000)
    assert not bool(out["episodes"]["episode_type"].eq("blink").any())


def test_ivt_list_empty_short_no_rows_overwrite_and_defensive_pos(monkeypatch):
    x = _dataset()

    empty = x.copy()
    _empty_component(empty, "gaze_samples")
    assert ep.detect_fixations_ivt(empty) is empty

    short = ep.detect_fixations_ivt(
        x,
        coordinate_units=["degrees"],
        velocity_threshold=1000,
        minimum_duration_ms=10000,
    )
    assert not bool(short["episodes"]["episode_type"].eq("fixation").any())

    invalid = x.copy()
    invalid["gaze_samples"].loc[:, "valid"] = False
    none = ep.detect_fixations_ivt(invalid, coordinate_units="degrees")
    assert not bool(none["episodes"]["episode_type"].eq("fixation").any())

    first = ep.detect_fixations_ivt(
        x,
        coordinate_units="degrees",
        velocity_threshold=1000,
        minimum_duration_ms=0,
    )
    assert bool(first["episodes"]["episode_type"].eq("fixation").any())
    second = ep.detect_fixations_ivt(
        first,
        coordinate_units="degrees",
        velocity_threshold=1000,
        minimum_duration_ms=0,
        overwrite=True,
    )
    assert bool(second["episodes"]["episode_type"].eq("fixation").any())

    with monkeypatch.context() as mp:
        real_unique = pf.pd.unique

        def bogus_unique(values):
            arr = np.asarray(values)
            if arr.dtype.kind in "iu" and arr.size:
                return np.array([999999])
            return real_unique(values)

        mp.setattr(pf.pd, "unique", bogus_unique)
        defensive = ep.detect_fixations_ivt(
            x,
            coordinate_units="degrees",
            velocity_threshold=1000,
            minimum_duration_ms=0,
        )
        assert not bool(defensive["episodes"]["episode_type"].eq("fixation").any())


def test_idt_break_dispersion_continue_and_no_rows():
    x = _dataset()

    too_short = ep.detect_fixations_idt(
        x,
        dispersion_threshold=10,
        minimum_duration_ms=10000,
        coordinate_units="degrees",
    )
    assert not bool(too_short["episodes"]["episode_type"].eq("fixation").any())

    rejected = ep.detect_fixations_idt(
        x,
        dispersion_threshold=-1,
        minimum_duration_ms=0,
        coordinate_units="degrees",
    )
    assert not bool(rejected["episodes"]["episode_type"].eq("fixation").any())


def test_saccade_short_and_missing_original_defensive_paths(monkeypatch):
    x = _dataset()

    with monkeypatch.context() as mp:
        def short_velocity(_):
            return pd.DataFrame(
                {
                    "recording_id": ["R1", "R1"],
                    "sample_id": ["G0", "G1"],
                    "timestamp_seconds": [0.0, 0.1],
                    "velocity": [100.0, 100.0],
                }
            )

        mp.setattr(pf, "gaze_velocity", short_velocity)
        out = ep.detect_saccades(x, velocity_threshold=10, minimum_duration_ms=1000)
        assert not bool(out["episodes"]["episode_type"].eq("saccade").any())

    with monkeypatch.context() as mp:
        def missing_velocity(_):
            return pd.DataFrame(
                {
                    "recording_id": ["R1", "R1"],
                    "sample_id": ["MISSING_A", "MISSING_B"],
                    "timestamp_seconds": [0.0, 0.1],
                    "velocity": [100.0, 100.0],
                }
            )

        mp.setattr(pf, "gaze_velocity", missing_velocity)
        out = ep.detect_saccades(x, velocity_threshold=10, minimum_duration_ms=0)
        assert not bool(out["episodes"]["episode_type"].eq("saccade").any())


def test_preprocess_baseline_blink_and_ivt_dispatch_paths():
    x = _dataset()
    defaultish = ep.preprocess_spec(
        gaze_filter="none",
        pupil_interpolation="none",
        pupil_filter="none",
        pupil_baseline="subtract",
        pupil_baseline_window=(0.0, 0.3),
        blink_detection=True,
        fixation_algorithm="none",
    )
    out = ep.preprocess_eye(x, defaultish)
    assert out is not None

    ivt = ep.preprocess_spec(
        gaze_filter="none",
        pupil_interpolation="none",
        pupil_filter="none",
        pupil_baseline="none",
        blink_detection=False,
        fixation_algorithm="ivt",
        fixation_parameters={
            "coordinate_units": "degrees",
            "velocity_threshold": 1000,
            "minimum_duration_ms": 10000,
        },
    )
    out2 = ep.preprocess_eye(x, ivt)
    assert out2 is not None


def test_summary_fixations_groups_empty_after_dropna():
    x = _dataset()
    x = ep.detect_fixations_ivt(
        x,
        coordinate_units="degrees",
        velocity_threshold=1000,
        minimum_duration_ms=0,
    )
    x["episodes"].loc[x["episodes"]["episode_type"].eq("fixation"), "aoi_id"] = pd.NA
    out = ep.summarize_fixations(x, by=("aoi_id",), source="eyeprocess")
    assert out.empty


def test_sequence_transition_entropy_and_gaze_tuple_coercion_paths():
    x = _dataset()
    x["gaze_samples"]["aoi_id"] = ["A", "A", "B", "B", "A", "C", "C"]

    seq = ep.scanpath_sequence(
        x,
        source=["samples"],
        trial_id=["T1"],
        recording_id=["R1"],
    )
    assert len(seq) == 1

    matrix = ep.transition_matrix(
        x,
        normalize=["all"],
        source=["samples"],
        include_self=True,
    )
    assert matrix.to_numpy().sum() == pytest.approx(1.0)

    entropy = ep.gaze_entropy(x, level=["recording"], source=["samples"])
    assert len(entropy) == 1

    te = ep.transition_entropy(x, source=["samples"])
    assert not te.empty

    derived = ep.derive_gaze_features(
        x,
        level=["trial"],
        source=["samples"],
        append=False,
    )
    assert not derived["features"].empty


def test_pupil_feature_no_trials_and_trial_mismatch_paths():
    x = _dataset()

    no_trials = x.copy()
    _empty_component(no_trials, "intervals")
    with pytest.raises(ep.EyeProcessValidationError, match="Trial intervals"):
        ep.derive_pupil_features(no_trials)

    mismatch = x.copy()
    mismatch["eye_samples"].loc[:, "trial_id"] = "NO_MATCH"
    out = ep.derive_pupil_features(mismatch, append=False)
    assert out["features"].empty


def test_biometric_all_nonfinite_group_is_skipped():
    x = _dataset()
    template = x["biometrics"].copy()
    row = {column: pd.NA for column in template.columns}
    row.update(
        {
            "recording_id": "R1",
            "sample_id": "B1",
            "timestamp_seconds": 0.1,
            "channel": "eda",
            "value": np.nan,
            "unit": "uS",
            "trial_id": "T1",
            "stimulus_id": "S1",
        }
    )
    x["biometrics"] = pd.DataFrame([row], columns=template.columns)
    out = ep.derive_biometric_features(x, append=False)
    assert out["features"].empty


def test_derive_all_empty_components_and_features_wide_scalar_group_key():
    x = _dataset()
    empty = x.copy()
    for name in ("gaze_samples", "eye_samples", "episodes", "responses", "biometrics"):
        _empty_component(empty, name)
    out = ep.derive_all_features(empty, reset=False)
    assert out["features"].empty

    with_features = ep.derive_pupil_features(x, append=False)
    wide = ep.features_wide(with_features, id_cols=("recording_id",))
    assert len(wide) == 1
