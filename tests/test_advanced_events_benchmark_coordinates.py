from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import eyeprocesspy as ep


def _trajectory(hz: float = 500.0, n: int = 500) -> pd.DataFrame:
    t = np.arange(n) * 1000.0 / hz
    x = np.zeros(n)
    x[100:120] = np.linspace(0, 1.5, 20)
    x[120:300] = np.linspace(1.5, 4.0, 180)
    x[300:] = 4.0
    return pd.DataFrame({"timestamp_ms": t, "gaze_x_deg": x, "gaze_y_deg": np.zeros(n)})


def test_pursuit_detectors_gain_summary_and_plots():
    data = _trajectory()
    ivvt = ep.detect_events_ivvt(
        data,
        fixation_velocity_threshold=1,
        saccade_velocity_threshold=80,
        minimum_duration_ms=2,
    )
    directional = ep.detect_events_directional(
        data,
        minimum_velocity=1,
        maximum_velocity=80,
        maximum_direction_change_deg=90,
        minimum_duration_ms=2,
    )
    assert ivvt.sampling_hz == pytest.approx(500, rel=0.02)
    assert directional.method == "directional"
    assert ep.detect_smooth_pursuits(data, method="ivvt", fixation_velocity_threshold=1, saccade_velocity_threshold=80, minimum_duration_ms=2).method == "ivvt"
    assert "status" in ep.validate_pursuit_detection(ivvt)
    assert ep.summarise_pursuits(ivvt).n_pursuits.iloc[0] >= 0

    target = data.copy()
    target["target_x_deg"] = np.linspace(0, 4, len(target))
    target["target_y_deg"] = 0.0
    gain = ep.compute_pursuit_gain(target)
    error = ep.compute_pursuit_velocity_error(target)
    assert "pursuit_gain" in gain
    assert "absolute_velocity_error" in error

    ax = ep.plot_pursuit_velocity(directional)
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_events_ivvt(data, fixation_velocity_threshold=10, saccade_velocity_threshold=5)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_events_directional(data, minimum_velocity=10, maximum_velocity=5)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_events_ivvt(data.iloc[::20].reset_index(drop=True), minimum_sampling_hz=100)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_smooth_pursuits(data, method="bad")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.summarise_pursuits({})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_pursuit_velocity({})
    bad_time = data.copy()
    bad_time.loc[2, "timestamp_ms"] = bad_time.loc[1, "timestamp_ms"]
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_events_ivvt(bad_time)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_events_ivvt(data, time_unit="minutes")


def test_microsaccades_and_main_sequence():
    data = _trajectory()
    result = ep.detect_microsaccades(
        data,
        lambda_threshold=3,
        minimum_duration_ms=0,
        maximum_amplitude_deg=5,
        minimum_sampling_hz=200,
    )
    assert result.eyeprocess_class == "eye_microsaccades"
    summary = ep.summarise_microsaccades(result)
    assert summary.n_microsaccades.iloc[0] >= 0
    main = ep.microsaccade_main_sequence(result)
    assert set(main.columns) == {"event_id", "amplitude_deg", "peak_velocity"}
    ax = ep.plot_microsaccade_main_sequence(result)
    assert hasattr(ax, "eyeprocess_plot_data")

    binocular = data.copy()
    binocular["right_x"] = binocular.gaze_x_deg
    binocular["right_y"] = binocular.gaze_y_deg
    right = ep.detect_microsaccades(
        binocular,
        lambda_threshold=3,
        minimum_duration_ms=0,
        maximum_amplitude_deg=5,
        right_x="right_x",
        right_y="right_y",
    )
    assert right.binocular

    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_microsaccades(data, right_x="gaze_x_deg")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_microsaccades(data, lambda_threshold=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_microsaccades(data.iloc[::10].reset_index(drop=True), minimum_sampling_hz=200)
    flat = data.assign(gaze_x_deg=0.0, gaze_y_deg=0.0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detect_microsaccades(flat)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.summarise_microsaccades({})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.microsaccade_main_sequence({})


def _events() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "start_time": [0.0, 100.0],
            "end_time": [50.0, 160.0],
            "event_type": ["fixation", "saccade"],
        }
    )
    detected = pd.DataFrame(
        {
            "start_time": [2.0, 105.0, 200.0],
            "end_time": [48.0, 155.0, 220.0],
            "event_type": ["fixation", "saccade", "fixation"],
        }
    )
    return truth, detected


def test_event_benchmark_confusion_sensitivity_and_plots():
    truth, detected = _events()
    result = ep.benchmark_event_detector(truth, detected, minimum_iou=0.3)
    assert result.summary.tp.iloc[0] == 2
    boundaries = ep.event_boundary_error(result)
    assert len(boundaries) == 2
    confusion = ep.event_confusion_matrix(truth, detected, sample_step=10)
    assert not confusion.empty
    comparison = ep.compare_event_detectors(truth, {"a": detected, "b": truth}, minimum_iou=0.3)
    assert len(comparison) == 2

    def detector(data, offset=0, fail=False):
        if fail:
            raise RuntimeError("planned failure")
        out = detected.copy()
        out["start_time"] += offset
        return {"events": out}

    sensitivity = ep.detector_parameter_sensitivity(
        pd.DataFrame({"x": [1]}),
        truth,
        detector,
        {"offset": [0, 2], "fail": [False, True]},
        benchmark_kwargs={"minimum_iou": 0.3},
    )
    assert sensitivity.planned_branches == 4
    assert sensitivity.failed_branches == 2
    ax = ep.plot_detector_benchmark(comparison)
    assert hasattr(ax, "eyeprocess_plot_data")

    empty = ep.benchmark_event_detector(truth, detected.assign(event_type="other"), minimum_iou=0.3)
    assert ep.event_boundary_error(empty).empty
    assert ep.event_confusion_matrix([], []).empty

    with pytest.raises(ep.EyeProcessValidationError):
        ep.benchmark_event_detector(truth, detected, minimum_iou=2)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.event_confusion_matrix(truth, detected, sample_step=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.compare_event_detectors(truth, {})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detector_parameter_sensitivity([], truth, "not callable", {"x": [1]})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.detector_parameter_sensitivity([], truth, detector, {})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_detector_benchmark(pd.DataFrame({"detector": ["a"]}), metric="f1")


def test_naturalistic_coordinate_pipeline_uncertainty_and_guards():
    px = np.array([-100.0, 0.0, 100.0])
    deg = ep.pixels_to_visual_angle(px, pixels_per_mm=4, viewing_distance_mm=600)
    back = ep.visual_angle_to_pixels(deg, pixels_per_mm=4, viewing_distance_mm=600)
    np.testing.assert_allclose(px, back)

    gaze = pd.DataFrame(
        {
            "timestamp": [0.0, 0.5, 1.0],
            "gaze_x_px": [960.0, 970.0, 950.0],
            "gaze_y_px": [540.0, 535.0, 545.0],
        }
    )
    head = ep.screen_gaze_to_head_vectors(
        gaze,
        screen_width_px=1920,
        screen_height_px=1080,
        screen_width_mm=530,
        screen_height_mm=300,
        viewing_distance_mm=600,
    )
    assert np.allclose(np.linalg.norm(head[["head_gaze_x","head_gaze_y","head_gaze_z"]], axis=1), 1)

    q = Rotation.from_euler("z", [0, 10], degrees=True).as_quat()
    pose = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
            "px": [0.0, 1.0], "py": [0.0, 0.0], "pz": [0.0, 0.0],
        }
    )
    aligned = ep.align_head_pose_to_gaze(head, pose)
    world = ep.transform_head_gaze_to_world(aligned)
    assert "world_gaze_z" in world
    full = ep.transform_screen_gaze_to_world(
        gaze,
        pose,
        screen_width_px=1920,
        screen_height_px=1080,
        screen_width_mm=530,
        screen_height_mm=300,
        viewing_distance_mm=600,
    )
    assert len(full) == len(gaze)
    uncertainty = ep.propagate_coordinate_uncertainty(
        gaze,
        screen_width_px=1920,
        screen_height_px=1080,
        screen_width_mm=530,
        screen_height_mm=300,
        viewing_distance_mm=600,
        viewing_distance_sd_mm=5,
        draws=4,
    )
    assert len(uncertainty.draws) == 12
    ax = ep.plot_world_gaze_vectors(world)
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(ep.EyeProcessValidationError):
        ep.pixels_to_visual_angle([1], pixels_per_mm=0, viewing_distance_mm=600)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.visual_angle_to_pixels([1], pixels_per_mm=4, viewing_distance_mm=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.screen_gaze_to_head_vectors(gaze, screen_width_px=0, screen_height_px=1080, screen_width_mm=530, screen_height_mm=300, viewing_distance_mm=600)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.screen_gaze_to_head_vectors(gaze, screen_width_px=1920, screen_height_px=1080, screen_width_mm=530, screen_height_mm=300, viewing_distance_mm=600, origin="bad")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.align_head_pose_to_gaze(gaze.iloc[:0], pose)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.align_head_pose_to_gaze(gaze.assign(timestamp=[0, 1, 2]), pose)
    bad_pose = pose.copy()
    bad_pose.loc[1, "timestamp"] = 0
    with pytest.raises(ep.EyeProcessValidationError):
        ep.align_head_pose_to_gaze(gaze, bad_pose)
    bad_world = aligned.copy()
    bad_world.loc[0, "head_gaze_x"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError):
        ep.transform_head_gaze_to_world(bad_world)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.propagate_coordinate_uncertainty(gaze, screen_width_px=1920, screen_height_px=1080, screen_width_mm=530, screen_height_mm=300, viewing_distance_mm=600, viewing_distance_sd_mm=-1)
