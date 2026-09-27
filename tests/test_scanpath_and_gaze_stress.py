from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


def test_scanpath_metrics_sensitivity_and_plot():
    reference = ["A", "B", "C", "B"]
    same = ["A", "B", "C", "B"]
    other = ["A", "C", "C", "D"]

    assert ep.levenshtein_scanpath(reference, same) == 0
    assert ep.levenshtein_scanpath([], [], normalize=False) == 0
    assert 0 <= ep.transition_js_distance(reference, other) <= 1
    assert ep.transition_js_distance(["A"], ["B"]) == 0
    assert ep.ngram_jaccard_similarity(reference, same) == 1
    assert ep.ngram_jaccard_similarity(["A"], ["A"], n=2) == 1

    traj_a = pd.DataFrame({"x": [0, 1, 2], "y": [0, 0, 0]})
    traj_b = pd.DataFrame({"x": [0, 1.2, 2], "y": [0, 0.1, 0]})
    dtw = ep.dynamic_time_warping_distance(traj_a, traj_b)
    assert dtw >= 0
    assert ep.dynamic_time_warping_distance(traj_a, traj_a, normalize=False) == 0

    table = ep.compare_scanpath_metrics(
        reference,
        other,
        reference_trajectory=traj_a,
        candidate_trajectory=traj_b,
    )
    assert set(table.metric) == {
        "normalized_levenshtein",
        "transition_js_distance",
        "bigram_jaccard",
        "trajectory_dtw",
    }
    basic = ep.compare_scanpath_metrics(reference, other)
    assert len(basic) == 3

    sensitivity = ep.scanpath_metric_sensitivity(
        reference,
        {"same": same, "other": other},
    )
    assert sensitivity.eyeprocess_class == "eye_scanpath_metric_sensitivity"
    assert sensitivity.table.groupby("metric")["rank"].min().eq(1).all()
    ax = ep.plot_scanpath_sensitivity(sensitivity)
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(ep.EyeProcessValidationError):
        ep.ngram_jaccard_similarity(reference, other, n=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.dynamic_time_warping_distance(pd.DataFrame({"x": [1]}), traj_b)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.dynamic_time_warping_distance(pd.DataFrame({"x": [np.nan], "y": [0]}), traj_b)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.compare_scanpath_metrics(reference, other, reference_trajectory=traj_a)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.scanpath_metric_sensitivity(reference, {})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_scanpath_sensitivity({})


def test_known_gaze_simulation_and_corruptions():
    data = ep.simulate_known_gaze_process(n_cycles=2, sampling_hz=300, seed=4)
    assert set(data.event_type.unique()) == {"fixation", "saccade", "pursuit"}
    assert data.attrs["eyeprocess_class"] == "eye_known_gaze_simulation"

    drift = ep.inject_spatial_drift(data, max_offset_x=1, max_offset_y=0.5)
    assert drift.gaze_x_deg.iloc[-1] > drift.gaze_x_deg_clean.iloc[-1]
    noise = ep.inject_spatial_noise(data, sd=0.1, seed=2)
    assert not np.allclose(noise.gaze_x_deg, data.gaze_x_deg)
    gaps = ep.inject_blink_gaps(data, fraction=0.1, mean_gap_samples=3, seed=2)
    assert gaps.synthetic_blink_gap.any()
    mislabeled = ep.inject_event_misclassification(data, probability=0.2, seed=3)
    assert "event_type_clean" in mislabeled

    with pytest.raises(ep.EyeProcessValidationError):
        ep.simulate_known_gaze_process(n_cycles=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.inject_spatial_drift(data.drop(columns="gaze_x_deg"))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.inject_spatial_noise(data, sd=-1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.inject_blink_gaps(data, fraction=1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.inject_event_misclassification(data, probability=1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.inject_event_misclassification(data.assign(event_type="fixation"), probability=0.1)


def test_gaze_stress_suite_all_corruptions_failure_accounting_and_plot():
    data = ep.simulate_known_gaze_process(n_cycles=1, sampling_hz=300)

    def evaluator(frame):
        return {
            "mean_x": float(pd.to_numeric(frame.gaze_x_deg, errors="coerce").mean()),
            "valid_fraction": float(frame.gaze_x_deg.notna().mean()),
        }

    for corruption in ["drift", "noise", "missingness", "event_misclassification"]:
        result = ep.run_gaze_stress_suite(
            data,
            evaluator,
            corruption=corruption,
            severities=(0, 0.1),
        )
        assert result.table.status.eq("success").all()

    failed = ep.run_gaze_stress_suite(data, evaluator, corruption="bad", severities=(0,))
    assert failed.table.status.iloc[0] == "failed"

    plotted = ep.run_gaze_stress_suite(data, evaluator, corruption="noise", severities=(0, 0.1))
    ax = ep.plot_gaze_stress(plotted, metric="mean_x")
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(ep.EyeProcessValidationError):
        ep.run_gaze_stress_suite(data, "not callable", corruption="noise")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_gaze_stress({}, metric="mean_x")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_gaze_stress(plotted, metric="missing")
