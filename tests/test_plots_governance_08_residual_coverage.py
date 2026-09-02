from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.plots_governance_08 as plots
import eyeprocesspy.process_governance_08 as governance


def _obj(class_name: str, **kwargs):
    return SimpleNamespace(eyeprocess_class=class_name, **kwargs)


def _close(ax):
    if ax is not None and hasattr(ax, "figure"):
        plt.close(ax.figure)


def test_axis_and_class_guards_and_biometric_preflight_modes():
    fig, supplied = plt.subplots()
    assert plots._ax(supplied) is supplied
    plt.close(fig)

    with pytest.raises(ep.EyeProcessValidationError, match="eye_biometric_preflight"):
        plots.plot_eye_biometric_preflight({})

    table = pd.DataFrame(
        {
            "preflight_decision": ["pass_preflight", "use_with_caution", "pass_preflight"],
            "preflight_flag_count": [0, 1, 0],
            "low_valid_gaze_flag": [False, True, False],
            "low_valid_pupil_flag": [False, False, True],
        }
    )
    obj = _obj(
        "eye_biometric_preflight",
        table=table,
        flag_columns=["low_valid_gaze_flag", "low_valid_pupil_flag"],
    )
    for mode in ("decision_counts", "heatmap"):
        ax = plots.plot_eye_biometric_preflight(obj, type=mode)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    with pytest.raises(ep.EyeProcessValidationError, match="heatmap or decision_counts"):
        plots.plot_eye_biometric_preflight(obj, type="bad")


def test_anomaly_and_accessibility_plots():
    anomaly = _obj(
        "eye_process_anomaly_audit",
        table=pd.DataFrame({"mahalanobis_process_distance": [0.4, 1.2, 2.0]}),
        threshold=1.5,
    )
    ax = plots.plot_eye_process_anomaly_audit(anomaly)
    assert ax.eyeprocess_plot_data.shape[0] == 3
    _close(ax)

    accessibility = _obj(
        "eye_presentation_accessibility",
        table=pd.DataFrame({"presentation_sensitivity_index": [0.1, np.nan, 0.5, 0.9]}),
        threshold=0.6,
    )
    ax = plots.plot_eye_presentation_accessibility(accessibility)
    assert ax.eyeprocess_plot_data.shape[0] == 4
    _close(ax)


def _drift_obj():
    trajectories = pd.DataFrame(
        {
            "item_id": ["I1", "I1", "I2", "I2"],
            "batch_order": [1, 2, 1, 2],
            "accuracy": [0.70, 0.74, 0.60, 0.66],
        }
    )
    table = pd.DataFrame(
        {
            "item_id": ["I1", "I2", "I3"],
            "accuracy_delta": [-0.10, 0.00, 0.12],
            "latency_delta": [2.0, 1.0, 3.0],
        }
    )
    return _obj(
        "eye_process_drift_audit",
        metrics=["accuracy"],
        item="item_id",
        trajectories=trajectories,
        table=table,
    )


def test_process_drift_all_modes_and_guards():
    obj = _drift_obj()
    for mode in ("trajectory", "delta", "control", "heatmap"):
        ax = plots.plot_eye_process_drift_audit(obj, type=mode)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)

    ax = plots.plot_eye_process_drift_audit(obj, type="trajectory", item=["I1", "I2"])
    assert len(ax.eyeprocess_plot_data) == 4
    _close(ax)

    with pytest.raises(ep.EyeProcessValidationError, match="Unknown metric"):
        plots.plot_eye_process_drift_audit(obj, metric="missing")
    with pytest.raises(ep.EyeProcessValidationError, match="trajectory, delta, heatmap, or control"):
        plots.plot_eye_process_drift_audit(obj, type="bad")


def test_process_window_sensitivity_and_frequency_feature_modes():
    window = _obj(
        "eye_process_window_sensitivity",
        metric="pupil",
        table=pd.DataFrame(
            {
                "step_ms": [100, 100, 200, 200],
                "width_ms": [300, 500, 300, 500],
                "mean_value": [1.0, 1.2, 0.9, 1.1],
            }
        ),
    )
    ax = plots.plot_eye_process_window_sensitivity(window)
    assert ax.get_legend() is not None
    _close(ax)

    features = _obj(
        "eye_pupil_frequency_features",
        features=pd.DataFrame(
            {
                "pupil_low_frequency_power": [1.0, 2.0],
                "pupil_high_frequency_power": [0.5, 1.5],
                "pupil_frequency_contrast": [0.1, 0.2],
                "pupil_velocity_activity": [2.0, 3.0],
                "pupil_ripa_proxy": [-0.1, 0.1],
            }
        ),
    )
    for mode in ("power_relationship", "features"):
        ax = plots.plot_eye_pupil_frequency_features(features, type=mode)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    with pytest.raises(ep.EyeProcessValidationError, match="features or power_relationship"):
        plots.plot_eye_pupil_frequency_features(features, type="bad")

    stability = _obj(
        "eye_pupil_frequency_stability",
        table=pd.DataFrame(
            {
                "window_ms": [500, 500, 1000, 1000],
                "pupil_frequency_contrast": [0.1, 0.2, 0.15, 0.25],
            }
        ),
    )
    ax = plots.plot_eye_pupil_frequency_stability(stability)
    assert len(ax.eyeprocess_plot_data) == 2
    _close(ax)


def test_pupil_deconvolution_all_modes_and_invalid_type():
    obj = _obj(
        "eye_pupil_deconvolution",
        tmax_ms=930,
        shape=10.1,
        effects=pd.DataFrame({"beta__stimulus": [0.2, 0.4], "beta__response": [-0.1, 0.1]}),
        fitted=pd.DataFrame(
            {
                "time": [0.0, 100.0, 200.0],
                "observed": [3.0, 3.2, 3.1],
                "fitted": [3.0, 3.1, 3.15],
                "residual": [0.0, 0.1, -0.05],
            }
        ),
    )
    for mode in ("kernels", "effects", "residuals", "observed_fitted"):
        ax = plots.plot_eye_pupil_deconvolution(obj, type=mode)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    with pytest.raises(ep.EyeProcessValidationError, match="unknown deconvolution"):
        plots.plot_eye_pupil_deconvolution(obj, type="bad")


def test_pupil_confound_all_modes_including_theta_unavailable():
    data = pd.DataFrame(
        {
            ".luminance": [0.1, 0.3, 0.5],
            ".pupil": [3.0, 3.2, 3.4],
            ".trial": [1.0, 2.0, 3.0],
            ".theta": [-1.0, 0.0, 1.0],
            "pupil_confound_adjusted": [3.05, 3.15, 3.35],
        }
    )
    obj = _obj("eye_pupil_confound_model", data=data)
    for mode in ("luminance", "trial_order", "raw_adjusted", "theta_luminance_surface"):
        ax = plots.plot_eye_pupil_confound_model(obj, type=mode)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)

    constant = data.copy()
    constant[".theta"] = 0.0
    ax = plots.plot_eye_pupil_confound_model(
        _obj("eye_pupil_confound_model", data=constant),
        type="theta_luminance_surface",
    )
    assert any(text.get_text() == "Theta unavailable" for text in ax.texts)
    _close(ax)

    with pytest.raises(ep.EyeProcessValidationError, match="unknown pupil confound"):
        plots.plot_eye_pupil_confound_model(obj, type="bad")


def test_aoi_trajectory_modes_and_growth_curve(monkeypatch):
    trajectory = _obj(
        "eye_aoi_trajectory",
        features=pd.DataFrame(
            {
                "target_gca_degree1": [0.1, 0.2],
                "target_gca_degree2": [-0.1, 0.3],
                "other": [1.0, 2.0],
            }
        ),
    )
    for mode in ("coefficients", "profiles"):
        ax = plots.plot_eye_aoi_trajectory(trajectory, type=mode)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    with pytest.raises(ep.EyeProcessValidationError, match="coefficients or profiles"):
        plots.plot_eye_aoi_trajectory(trajectory, type="bad")

    growth = _obj(
        "eye_aoi_growth_curve",
        time=pd.Series([0.0, 1.0, 2.0]),
        outcome=pd.Series([0.1, 0.4, 0.7]),
    )
    predicted = pd.DataFrame({"time": [0.0, 1.0, 2.0], "predicted": [0.2, 0.4, 0.6]})
    monkeypatch.setattr(governance, "predict_aoi_trajectory", lambda x: predicted)
    ax = plots.plot_eye_aoi_growth_curve(growth)
    pd.testing.assert_frame_equal(ax.eyeprocess_plot_data, predicted)
    _close(ax)


def test_signal_filter_process_windows_fatigue_and_fairness():
    filt = _obj(
        "eye_signal_filter_audit",
        method="median",
        data=pd.DataFrame(
            {
                "sample_index": [1, 2, 3],
                "raw": [1.0, 2.0, 1.5],
                "filtered": [1.1, 1.8, 1.6],
            }
        ),
    )
    ax = plots.plot_eye_signal_filter_audit(filt)
    _close(ax)

    windows = _obj(
        "eye_process_windows",
        data=pd.DataFrame(
            {
                "window_mid": [100, 200, 100, 200],
                "group": ["A", "A", "B", "B"],
                "pupil_mean": [1.0, 1.2, 0.8, 1.1],
            }
        ),
    )
    for group in (None, "group"):
        ax = plots.plot_eye_process_windows(windows, group=group)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    ax = plots.plot_eye_process_windows(windows, group="missing")
    _close(ax)

    fatigue = _obj(
        "eye_pupil_fatigue_drift",
        engine="lm_fixed_effects",
        data=pd.DataFrame({"trial_order": [1.0, 2.0, 3.0], "pupil": [3.0, 2.9, 2.8]}),
    )
    ax = plots.plot_eye_pupil_fatigue_drift(fatigue)
    _close(ax)

    fairness = _obj(
        "eye_presentation_fairness_comparison",
        summary=pd.DataFrame({"variant": ["A", "B"], "mean": [0.5, 0.7]}),
    )
    ax = plots.plot_eye_presentation_fairness_comparison(fairness)
    _close(ax)


def test_pupil_spectrum_band_activity_and_aliases():
    with pytest.raises(ep.EyeProcessValidationError, match="positive"):
        plots.plot_pupil_spectrum(np.arange(10.0), 0)
    with pytest.raises(ep.EyeProcessValidationError, match="At least eight"):
        plots.plot_pupil_spectrum([1.0, np.nan, 2.0], 60)

    signal = np.sin(np.linspace(0, 4 * np.pi, 32))
    ax = plots.plot_pupil_spectrum(signal, 60)
    assert len(ax.eyeprocess_plot_data) > 0
    _close(ax)
    ax = plots.plot_pupil_spectrum(signal, 60, max_hz=4.0)
    assert ax.eyeprocess_plot_data.frequency_hz.max() <= 4.0
    _close(ax)

    features = _obj(
        "eye_pupil_frequency_features",
        features=pd.DataFrame(
            {
                "pupil_low_frequency_power": [1.0, 2.0],
                "pupil_high_frequency_power": [0.5, 1.0],
                "pupil_frequency_contrast": [0.1, 0.2],
            }
        ),
    )
    ax = plots.plot_pupil_band_power(features)
    _close(ax)
    ax = plots.plot_pupil_activity_windows(features)
    _close(ax)

    stability = _obj(
        "eye_pupil_frequency_stability",
        table=pd.DataFrame({"pupil_frequency_contrast": [0.1, 0.2], "window_ms": [500, 1000]}),
    )
    ax = plots.plot_pupil_activity_windows(stability)
    _close(ax)
    ax = plots.plot_pupil_activity_sensitivity(stability)
    _close(ax)

    with pytest.raises(ep.EyeProcessValidationError, match="feature/stability"):
        plots.plot_pupil_activity_windows({})

    window = _obj(
        "eye_process_window_sensitivity",
        metric="pupil",
        table=pd.DataFrame(
            {"step_ms": [100, 100], "width_ms": [300, 500], "mean_value": [1.0, 1.1]}
        ),
    )
    ax = plots.plot_process_window_sensitivity(window)
    _close(ax)
