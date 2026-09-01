from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


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
            "gaze_x": [0.1, 0.11, 0.12, 0.7, 0.71, 0.2, 0.21],
            "gaze_y": [0.1, 0.11, 0.12, 0.7, 0.71, 0.2, 0.21],
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


def test_rolling_apply_guards_even_width_and_na_rm_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="width"):
        ep.rolling_apply([1, 2], width=0)
    even = ep.rolling_apply([1, 9, 1], width=2)
    odd = ep.rolling_apply([1, 9, 1], width=3)
    np.testing.assert_allclose(even, odd)
    out = ep.rolling_apply([1.0, np.nan, 3.0], width=3, FUN=np.mean, na_rm=False)
    assert np.isnan(out[1])


def test_filter_gaze_guards_noop_empty_and_missing_fields():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid gaze filter"):
        ep.filter_gaze(x, method="savgol")
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown component"):
        ep.filter_gaze(x, component="missing")
    assert ep.filter_gaze(x, method="none") is x

    empty = x.copy()
    empty["gaze_samples"] = empty["gaze_samples"].iloc[0:0].copy()
    assert ep.filter_gaze(empty, method="mean") is empty

    bad = x.copy()
    bad["gaze_samples"] = bad["gaze_samples"].drop(columns=["gaze_x"])
    with pytest.raises(ep.EyeProcessValidationError, match="Missing gaze field"):
        ep.filter_gaze(bad, method="mean")


def test_gaze_velocity_guards_and_empty_contract():
    with pytest.raises(TypeError, match="DataFrame"):
        ep.gaze_velocity([1, 2, 3])
    bad = pd.DataFrame({"recording_id": ["R1"]})
    with pytest.raises(ep.EyeProcessValidationError, match="Missing gaze field"):
        ep.gaze_velocity(bad)
    empty = _dataset()["gaze_samples"].iloc[0:0].copy()
    out = ep.gaze_velocity(empty)
    assert out.empty
    assert "velocity" in out.columns


def test_flag_gaze_outliers_invalid_empty_velocity_and_unregistered_space():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid gaze-outlier"):
        ep.flag_gaze_outliers(x, method="bad")
    empty = x.copy()
    empty["gaze_samples"] = empty["gaze_samples"].iloc[0:0].copy()
    assert ep.flag_gaze_outliers(empty) is empty

    velocity = ep.flag_gaze_outliers(x, method="velocity", max_velocity=0.01)
    assert velocity["gaze_samples"]["outlier_flag"].any()

    unregistered = x.copy()
    unregistered["gaze_samples"]["coordinate_space_id"] = "UNKNOWN"
    bounds = ep.flag_gaze_outliers(unregistered, method="bounds")
    assert not bounds["gaze_samples"]["outlier_flag"].any()


def test_interpolate_pupil_guards_noop_edges_all_missing_constant_and_mark_false():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid pupil interpolation"):
        ep.interpolate_pupil(x, method="spline")
    assert ep.interpolate_pupil(x, method="none") is x

    edge = x.copy()
    edge["eye_samples"].loc[0, "pupil_diameter"] = np.nan
    out = ep.interpolate_pupil(edge, max_gap_ms=1000)
    assert np.isnan(out["eye_samples"].loc[0, "pupil_diameter"])

    all_missing = x.copy()
    all_missing["eye_samples"]["pupil_diameter"] = np.nan
    out = ep.interpolate_pupil(all_missing, max_gap_ms=1000)
    assert out["eye_samples"]["pupil_diameter"].isna().all()

    constant = ep.interpolate_pupil(x, method="constant", max_gap_ms=500)
    assert constant["eye_samples"].loc[2, "pupil_diameter"] == pytest.approx(3.1)
    unmarked = ep.interpolate_pupil(x, method="linear", max_gap_ms=500, mark=False)
    assert not bool(unmarked["eye_samples"].loc[2, "interpolated"])


def test_filter_pupil_guards_and_noop():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid pupil filter"):
        ep.filter_pupil(x, method="loess")
    assert ep.filter_pupil(x, method="none") is x


@pytest.mark.parametrize("method", ["divide", "percent", "zscore"])
def test_baseline_pupil_alternate_methods(method):
    x = _dataset()
    out = ep.baseline_pupil(
        x,
        method=method,
        baseline_window=(0.0, 0.2),
        minimum_samples=2,
    )
    assert out["eye_samples"]["pupil_baseline"].notna().any()


def test_baseline_pupil_guards_recording_anchor_and_insufficient_baseline():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid pupil baseline method"):
        ep.baseline_pupil(x, method="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid pupil baseline anchor"):
        ep.baseline_pupil(x, anchor="event")

    no_trials = x.copy()
    no_trials["intervals"] = no_trials["intervals"].iloc[0:0].copy()
    with pytest.raises(ep.EyeProcessValidationError, match="Trial intervals"):
        ep.baseline_pupil(no_trials)

    recording = ep.baseline_pupil(
        x,
        anchor="recording_start",
        baseline_window=(0.0, 0.2),
        minimum_samples=2,
    )
    assert recording["eye_samples"]["pupil_baseline"].notna().any()

    insufficient = ep.baseline_pupil(
        x,
        baseline_window=(0.0, 0.0),
        minimum_samples=3,
    )
    assert insufficient["eye_samples"]["pupil_baseline"].isna().all()


def test_pupil_deconvolution_empty_sparse_and_regular_paths():
    x = _dataset()
    empty = x.copy()
    empty["eye_samples"] = empty["eye_samples"].iloc[0:0].copy()
    assert ep.pupil_deconvolve(empty) is empty

    sparse = x.copy()
    sparse["eye_samples"].loc[:, "pupil_diameter"] = np.nan
    sparse["eye_samples"].loc[:1, "pupil_diameter"] = [3.0, 3.1]
    out = ep.pupil_deconvolve(sparse)
    assert out["eye_samples"]["pupil_phasic"].isna().all()

    regular = ep.pupil_deconvolve(x, tau=1.2, regularization=0.2, output_column="phasic")
    assert regular["eye_samples"]["phasic"].notna().sum() >= 3


def test_detect_blinks_guards_empty_validity_and_overwrite():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid blink source"):
        ep.detect_blinks(x, source="bad")
    empty = x.copy()
    empty["eye_samples"] = empty["eye_samples"].iloc[0:0].copy()
    assert ep.detect_blinks(empty) is empty

    validity = x.copy()
    validity["eye_samples"].loc[2:4, "pupil_valid"] = False
    first = ep.detect_blinks(
        validity,
        source="validity",
        min_duration_ms=100,
        max_duration_ms=500,
    )
    n_first = int(first["episodes"]["episode_type"].eq("blink").sum())
    assert n_first >= 1
    second = ep.detect_blinks(
        first,
        source="validity",
        min_duration_ms=100,
        max_duration_ms=500,
        overwrite=True,
    )
    assert int(second["episodes"]["episode_type"].eq("blink").sum()) == n_first


def test_ivt_coordinate_unit_guard_and_warning_contract():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="Invalid coordinate units"):
        ep.detect_fixations_ivt(x, coordinate_units="meters")
    with pytest.warns(RuntimeWarning, match="not visual degrees"):
        ep.detect_fixations_ivt(
            x,
            coordinate_units="pixels",
            velocity_threshold=100,
            minimum_duration_ms=0,
        )
