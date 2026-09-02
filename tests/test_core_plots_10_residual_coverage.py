from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.core_plots_10 as cp


@pytest.fixture()
def dataset():
    return ep.simulate_eye_dataset(
        n_person=3,
        n_item=3,
        samples_per_trial=20,
        include_pupil=True,
        include_biometrics=True,
        seed=13,
    )


def _close(axis):
    import matplotlib.pyplot as plt

    if axis is not None:
        plt.close(axis.figure)


def _copy_with(dataset, **tables):
    out = dataset.copy()
    for name, value in tables.items():
        out[name] = value.copy()
    return out


def test_private_plot_helpers_cover_backend_axis_and_argument_edges(monkeypatch):
    import matplotlib.pyplot as plt

    axis = plt.subplots()[1]
    try:
        assert cp._ax(axis) is axis
        assert cp._as_values(None) == []
        assert cp._as_values("x") == ["x"]
        assert cp._as_values(("a", "b")) == ["a", "b"]
        assert cp._as_values(7) == [7]

        frame = pd.DataFrame(
            {
                "recording_id": ["r1", "r2"],
                "trial_id": ["t1", "t2"],
                "x": [1, 2],
            }
        )
        selected = cp._select_trial_data(frame, trial_id="t2", recording_id="r2")
        assert selected.x.tolist() == [2]
        assert cp._select_trial_data(pd.DataFrame({"x": [1]}), "t", "r").x.tolist() == [1]

        finite = cp._finite(pd.Series([1, np.inf, -np.inf, "bad"]))
        assert finite.notna().sum() == 1

        assert cp._set_axis_data(axis, {"x": 1}) is axis
        assert axis.eyeprocess_plot_data == {"x": 1}
        cp._set_axis_data(axis, pd.DataFrame(), np.eye(2))
        assert axis.eyeprocess_plot_matrix.shape == (2, 2)

        assert cp._validate_choice(("row", "none"), ("row", "none"), "mode") == "row"
        with pytest.raises(ep.EyeProcessValidationError, match="mode"):
            cp._validate_choice("bad", ("row", "none"), "mode")

        empty = cp._empty("nothing", main=None, ax=axis)
        assert empty.get_title() == ""
        assert empty.eyeprocess_plot_data.empty
    finally:
        plt.close(axis.figure)

    real = sys.modules.get("matplotlib.pyplot")
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    try:
        with pytest.raises(cp.EyeProcessBackendError, match="requires matplotlib"):
            cp._get_plt()
    finally:
        if real is not None:
            monkeypatch.setitem(sys.modules, "matplotlib.pyplot", real)


def test_overview_trace_fixation_and_scanpath_residuals(dataset):
    no_intervals = _copy_with(dataset, intervals=pd.DataFrame())
    axis = ep.plot_eye_overview(no_intervals)
    assert axis.eyeprocess_plot_data["trials"] == 0
    _close(axis)

    no_interval_type = _copy_with(dataset, intervals=pd.DataFrame({"other": [1]}))
    axis = ep.plot_eye_overview(no_interval_type)
    assert axis.eyeprocess_plot_data["trials"] == 0
    _close(axis)

    gaze = dataset["gaze_samples"].copy()
    gaze = gaze.drop(columns=["valid"], errors="ignore")
    trace_ds = _copy_with(dataset, gaze_samples=gaze)
    axis = ep.plot_eye_trace(trace_ds, valid_only=True, reverse_y=False)
    assert not axis.yaxis_inverted()
    _close(axis)

    axis = ep.plot_eye_trace(dataset, recording_id="does-not-exist")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    episodes = dataset["episodes"].copy()
    if "derived_by" not in episodes:
        episodes["derived_by"] = "eyeprocess"
    episodes.loc[:, "derived_by"] = "vendor"
    fix_ds = _copy_with(dataset, episodes=episodes)
    axis = ep.plot_fixations(fix_ds, source="vendor", reverse_y=False)
    assert not axis.yaxis_inverted()
    _close(axis)

    axis = ep.plot_fixations(fix_ds, source="eyeprocess")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    no_derived = episodes.drop(columns=["derived_by"], errors="ignore")
    axis = ep.plot_fixations(_copy_with(dataset, episodes=no_derived), source="vendor")
    assert not axis.eyeprocess_plot_data.empty
    _close(axis)

    no_type = episodes.drop(columns=["episode_type"], errors="ignore")
    if not no_type.empty:
        no_type.loc[:, "centroid_x"] = np.nan
    axis = ep.plot_fixations(_copy_with(dataset, episodes=no_type))
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    with pytest.raises(ep.EyeProcessValidationError, match="source"):
        ep.plot_fixations(dataset, source="bad")

    empty_ds = _copy_with(dataset, episodes=dataset["episodes"].iloc[0:0])
    axis = ep.plot_scanpath(empty_ds)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    visits = pd.DataFrame(
        {
            "episode_type": ["aoi_visit", "aoi_visit"],
            "start_time": [0.2, 0.1],
            "centroid_x": [0.2, 0.1],
            "centroid_y": [0.3, 0.2],
            "duration_ms": [120.0, 80.0],
            "recording_id": ["r1", "r1"],
            "trial_id": ["t1", "t1"],
        }
    )
    axis = ep.plot_scanpath(
        _copy_with(dataset, episodes=visits),
        label=False,
        reverse_y=False,
    )
    assert len(axis.texts) == 0
    assert not axis.yaxis_inverted()
    _close(axis)

    bad_visits = visits.copy()
    bad_visits["centroid_x"] = np.nan
    axis = ep.plot_scanpath(_copy_with(dataset, episodes=bad_visits))
    assert axis.eyeprocess_plot_data.empty
    _close(axis)


def test_heatmap_aoi_transition_and_pupil_residuals(dataset, monkeypatch):
    gaze = dataset["gaze_samples"].copy().drop(columns=["valid"], errors="ignore")
    heat_ds = _copy_with(dataset, gaze_samples=gaze)
    axis = ep.plot_gaze_heatmap(heat_ds, valid_only=True, bins=(4, 3))
    assert axis.eyeprocess_plot_matrix.shape == (4, 3)
    _close(axis)

    axis = ep.plot_gaze_heatmap(dataset, valid_only=False, recording_id="missing")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    with pytest.raises(ep.EyeProcessValidationError, match="bins"):
        ep.plot_gaze_heatmap(dataset, bins=(4,))

    with pytest.raises(ep.EyeProcessValidationError, match="feature"):
        ep.plot_aoi_dwell(dataset, feature="bad")

    axis = ep.plot_aoi_dwell(_copy_with(dataset, features=dataset["features"].iloc[0:0]))
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    unmatched = dataset["features"].copy()
    unmatched["feature_name"] = "other"
    axis = ep.plot_aoi_dwell(_copy_with(dataset, features=unmatched))
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(cp, "transition_matrix", lambda *args, **kwargs: None)
    axis = ep.plot_transition_matrix(dataset, normalize="none", source="samples")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(cp, "transition_matrix", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_transition_matrix(dataset, normalize="none", source="samples")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    with pytest.raises(ep.EyeProcessValidationError, match="source"):
        ep.plot_transition_matrix(dataset, source="bad")

    with pytest.raises(ep.EyeProcessValidationError, match="Pupil column"):
        ep.plot_pupil_timeseries(dataset, column="not_there")

    eye = dataset["eye_samples"].copy()
    if not eye.empty:
        eye.loc[:, "pupil_diameter"] = np.nan
    axis = ep.plot_pupil_timeseries(_copy_with(dataset, eye_samples=eye))
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    axis = ep.plot_pupil_timeseries(dataset, eye="not-an-eye")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    eye_data = dataset["eye_samples"]
    if not eye_data.empty:
        recording = eye_data["recording_id"].iloc[0]
        eye_name = eye_data["eye"].iloc[0]
        axis = ep.plot_pupil_timeseries(
            dataset,
            recording_id=recording,
            eye=eye_name,
        )
        assert len(axis.lines) == 1
        _close(axis)


def test_biometrics_quality_sampling_missingness_and_timeline_edges(dataset, monkeypatch):
    axis = ep.plot_biometrics(dataset, channels="not-a-channel")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    one_channel = dataset["biometrics"].copy()
    if not one_channel.empty:
        keep = one_channel["channel"].iloc[0]
        one_channel = one_channel[one_channel["channel"].eq(keep)].copy()
    axis = ep.plot_biometrics(_copy_with(dataset, biometrics=one_channel))
    assert len(axis.lines) <= 1
    _close(axis)

    monkeypatch.setattr(cp, "audit_signal_quality", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_signal_quality(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    quality = pd.DataFrame(
        {
            "recording_id": ["r1"],
            "trial_id": ["t1"],
            "metric": ["valid_fraction"],
            "value": [np.nan],
        }
    )
    monkeypatch.setattr(cp, "audit_signal_quality", lambda *args, **kwargs: quality.copy())
    axis = ep.plot_signal_quality(dataset, by_trial=True)
    assert axis.get_ylim()[1] == pytest.approx(1.0)
    _close(axis)

    monkeypatch.setattr(cp, "audit_sampling_rate", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_sampling_rate(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    rate = pd.DataFrame({"value": [59.0, 61.0]})
    monkeypatch.setattr(cp, "audit_sampling_rate", lambda *args, **kwargs: rate.copy())
    axis = ep.plot_sampling_rate(dataset, expected_hz=None)
    assert len(axis.lines) == 0
    _close(axis)

    monkeypatch.setattr(cp, "audit_missingness", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_missingness(dataset, component="eye_samples")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(cp, "trial_table", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_trial_timeline(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    trials = pd.DataFrame(
        {
            "recording_id": ["r1", "r2"],
            "trial_id": ["t1", "t2"],
            "start_time": [0.0, 1.0],
            "end_time": [0.5, 1.5],
        }
    )
    monkeypatch.setattr(cp, "trial_table", lambda *args, **kwargs: trials.copy())
    axis = ep.plot_trial_timeline(dataset, recording_id="r2")
    assert axis.eyeprocess_plot_data.trial_id.tolist() == ["t2"]
    _close(axis)


def test_feature_coordinate_clock_and_item_residuals(dataset, monkeypatch):
    empty_features = _copy_with(dataset, features=dataset["features"].iloc[0:0])
    axis = ep.plot_feature_distribution(empty_features, "x")
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    features = pd.DataFrame(
        {
            "feature_name": ["f", "f", "f", "f"],
            "value": [1.0, 2.0, 3.0, np.nan],
            "group": ["a", "a", "b", "b"],
            "recording_id": ["r1"] * 4,
        }
    )
    feat_ds = _copy_with(dataset, features=features)
    axis = ep.plot_feature_distribution(feat_ds, "f")
    assert len(axis.patches) > 0
    _close(axis)

    axis = ep.plot_feature_distribution(feat_ds, "f", group="group")
    assert axis.get_xlabel() == "group"
    _close(axis)

    monkeypatch.setattr(
        cp,
        "features_wide",
        lambda *args, **kwargs: pd.DataFrame({"recording_id": ["r1"], "f": [1.0]}),
    )
    axis = ep.plot_feature_correlation(dataset, features=["missing"])
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "features_wide",
        lambda *args, **kwargs: pd.DataFrame(
            {"recording_id": ["r1", "r2"], "f": [1.0, 2.0], "g": [2.0, 4.0]}
        ),
    )
    axis = ep.plot_feature_correlation(dataset, features=["f", "g"])
    assert axis.eyeprocess_plot_matrix.shape == (2, 2)
    _close(axis)

    monkeypatch.setattr(cp, "audit_coordinate_spaces", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_coordinate_spaces(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "audit_coordinate_spaces",
        lambda *args, **kwargs: pd.DataFrame({"coordinate_space_id": ["s"], "other": [1]}),
    )
    axis = ep.plot_coordinate_spaces(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "audit_coordinate_spaces",
        lambda *args, **kwargs: pd.DataFrame({"n_gaze": [2.0]}),
    )
    axis = ep.plot_coordinate_spaces(dataset)
    assert len(axis.patches) == 1
    _close(axis)

    monkeypatch.setattr(cp, "audit_clock_sync", lambda *args, **kwargs: pd.DataFrame())
    axis = ep.plot_clock_alignment(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "audit_clock_sync",
        lambda *args, **kwargs: pd.DataFrame({"gaze_start": [0.0]}),
    )
    axis = ep.plot_clock_alignment(dataset)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "audit_clock_sync",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "gaze_start": [0.0, 1.0],
                "gaze_end": [0.5, 1.5],
                "biometric_start": [0.1, 1.1],
                "biometric_end": [0.6, 1.6],
            }
        ),
    )
    axis = ep.plot_clock_alignment(dataset)
    assert len(axis.collections) >= 4
    _close(axis)

    monkeypatch.setattr(cp, "item_parameters", lambda model: pd.DataFrame())
    axis = ep.plot_item_difficulty(object())
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "item_parameters",
        lambda model: pd.DataFrame({"item_id": ["i1"], "slope": [1.0]}),
    )
    axis = ep.plot_item_difficulty(object())
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    monkeypatch.setattr(
        cp,
        "item_parameters",
        lambda model: pd.DataFrame({"b": [-0.5, 0.5]}),
    )
    axis = ep.plot_item_difficulty(object())
    assert len(axis.collections) == 1
    _close(axis)


def test_model_diagnostic_empty_and_residual_fallback_paths():
    class BrokenModel:
        eyeprocess_class = "eyeprocess_model"

        @property
        def fit(self):
            raise RuntimeError("broken")

    axis = ep.plot_model_diagnostics(object())
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    axis = ep.plot_model_diagnostics(BrokenModel())
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    no_fitted = SimpleNamespace(
        eyeprocess_class="eyeprocess_model",
        fit=SimpleNamespace(resid=np.array([1.0])),
    )
    axis = ep.plot_model_diagnostics(no_fitted)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    no_resid = SimpleNamespace(
        eyeprocess_class="eyeprocess_model",
        fit=SimpleNamespace(fittedvalues=np.array([1.0])),
    )
    axis = ep.plot_model_diagnostics(no_resid)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    nonfinite = SimpleNamespace(
        eyeprocess_class="eyeprocess_model",
        fit=SimpleNamespace(
            fittedvalues=np.array([np.inf]),
            resid_pearson=np.array([np.nan]),
        ),
    )
    axis = ep.plot_model_diagnostics(nonfinite)
    assert axis.eyeprocess_plot_data.empty
    _close(axis)

    fallback = SimpleNamespace(
        eyeprocess_class="eyeprocess_model",
        fit=SimpleNamespace(
            fittedvalues=np.array([1.0, 2.0]),
            resid=np.array([0.2, -0.2]),
        ),
    )
    axis = ep.plot_model_diagnostics(fallback)
    assert len(axis.eyeprocess_plot_data) == 2
    _close(axis)
