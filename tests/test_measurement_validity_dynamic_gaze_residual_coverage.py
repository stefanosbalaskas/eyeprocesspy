from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.advanced_events as ae
import eyeprocesspy.bayesian_networks.robustness as br
import eyeprocesspy.dynamic_aoi as da
import eyeprocesspy.event_benchmark as eb
import eyeprocesspy.measurement_validity as mv
import eyeprocesspy.naturalistic_coordinates as nc
import eyeprocesspy.scanpath_sensitivity as ss
from eyeprocesspy.bayesian_networks import fit_bayesian_network, prepare_bayesian_network_data
from eyeprocesspy.irt import EyeResult


class _BadFrame:
    def __iter__(self):
        raise RuntimeError("cannot iterate")


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": ["A", "A", "A", "B", "B", "B"],
            "target_x": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "target_y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "gaze_x": [0.1, 0.11, 0.09, 1.1, 1.11, 1.09],
            "gaze_y": [-0.05, -0.04, -0.06, 0.95, 0.96, 0.94],
            "pupil": [3.0, 3.1, 3.2, 3.0, 3.1, 3.2],
            "group": ["g1", "g1", "g1", "g2", "g2", "g2"],
        }
    )


def test_measurement_validity_residual_guards_and_exact_targets():
    with pytest.raises(ep.EyeProcessValidationError):
        mv.compute_spatial_error_field(_BadFrame())
    with pytest.raises(ep.EyeProcessValidationError):
        mv.compute_spatial_error_field(pd.DataFrame({"gaze_x": [0.0]}))

    data = _validation()
    field = mv.compute_spatial_error_field(data, target_id="target_id")
    exact = mv.correct_gaze_with_spatial_error(
        pd.DataFrame({"gaze_x": [0.0], "gaze_y": [0.0]}),
        field,
    )
    assert exact.gaze_x_corrected.iloc[0] == pytest.approx(-0.1)

    grouped = mv.fit_pupil_size_artifact(data, by="group", minimum_samples=3)
    grouped_corrected = mv.correct_pupil_size_artifact(data, grouped, by="group")
    assert grouped_corrected.gaze_x_pupil_artifact.notna().all()

    missing_pupil = data.iloc[:1].copy()
    missing_pupil["pupil"] = np.nan
    model = mv.fit_pupil_size_artifact(data, minimum_samples=3)
    corrected = mv.correct_pupil_size_artifact(missing_pupil, model)
    assert corrected.gaze_x_pupil_artifact.isna().all()


def _rectangle_spec() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"aoi": "A", "timestamp": 0.0, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            {"aoi": "A", "timestamp": 1.0, "x_min": 1.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0},
        ]
    )


def test_dynamic_aoi_residual_geometry_and_validation_paths():
    spec = da.validate_dynamic_aoi_spec(_rectangle_spec().to_dict("list"))
    assert spec.keyframes["shape"].eq("rectangle").all()

    with pytest.raises(ep.EyeProcessValidationError):
        da.validate_dynamic_aoi_spec(_BadFrame())
    with pytest.raises(ep.EyeProcessValidationError):
        da.validate_dynamic_aoi_spec(pd.DataFrame({"aoi": ["A"]}))
    with pytest.raises(ep.EyeProcessValidationError):
        da.validate_dynamic_aoi_spec(
            pd.DataFrame(
                [{"aoi": "P", "timestamp": 0.0, "shape": "polygon", "vertices": [(0, 0), (1, 0)]}]
            )
        )
    with pytest.raises(ep.EyeProcessValidationError):
        da.validate_dynamic_aoi_spec(
            pd.DataFrame([{"aoi": "P", "timestamp": 0.0, "shape": "polygon"}])
        )
    with pytest.raises(ep.EyeProcessValidationError):
        da.validate_dynamic_aoi_spec(
            pd.DataFrame(
                [
                    {
                        "aoi": "M",
                        "timestamp": 0.0,
                        "shape": "mask",
                        "mask": np.ones((2, 2), bool),
                        "x_min": 0.0,
                        "x_max": 0.0,
                        "y_min": 0.0,
                        "y_max": 1.0,
                    }
                ]
            )
        )

    endpoint = da.assign_dynamic_aoi(
        pd.DataFrame({"timestamp": [1.0], "gaze_x": [1.5], "gaze_y": [0.5]}),
        spec,
    )
    assert endpoint.assignments.aoi.iloc[0] == "A"

    shape_change = pd.DataFrame(
        [
            {"aoi": "X", "timestamp": 0.0, "shape": "rectangle", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "vertices": pd.NA},
            {"aoi": "X", "timestamp": 1.0, "shape": "polygon", "x_min": np.nan, "x_max": np.nan, "y_min": np.nan, "y_max": np.nan, "vertices": [(0, 0), (1, 0), (0, 1)]},
        ]
    )
    with pytest.raises(ep.EyeProcessValidationError, match="Shape type"):
        da.assign_dynamic_aoi(
            pd.DataFrame({"timestamp": [0.5], "gaze_x": [0.5], "gaze_y": [0.5]}),
            shape_change,
        )

    polygon_mismatch = pd.DataFrame(
        [
            {"aoi": "P", "timestamp": 0.0, "shape": "polygon", "vertices": [(0, 0), (1, 0), (0, 1)]},
            {"aoi": "P", "timestamp": 1.0, "shape": "polygon", "vertices": [(0, 0), (1, 0), (1, 1), (0, 1)]},
        ]
    )
    with pytest.raises(ep.EyeProcessValidationError, match="matching vertices"):
        da.assign_dynamic_aoi(
            pd.DataFrame({"timestamp": [0.5], "gaze_x": [0.5], "gaze_y": [0.5]}),
            polygon_mismatch,
        )

    mask = pd.DataFrame(
        [
            {"aoi": "M", "timestamp": 0.0, "shape": "mask", "mask": np.ones((2, 2), bool), "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        ]
    )
    outside = da.assign_dynamic_aoi(
        pd.DataFrame({"timestamp": [0.0], "gaze_x": [2.0], "gaze_y": [2.0]}),
        mask,
        interpolation="step",
    )
    assert outside.assignments.status.iloc[0] == "outside"


def _events() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {"start_time": [0.0, 100.0], "end_time": [50.0, 150.0], "event_type": ["fixation", "saccade"]}
    )
    detected = pd.DataFrame(
        {"start_time": [0.0, 105.0], "end_time": [50.0, 148.0], "event_type": ["fixation", "saccade"]}
    )
    return truth, detected


def test_event_benchmark_residual_input_matching_and_boundary_paths():
    truth, detected = _events()
    mapped = eb.benchmark_event_detector(truth.to_dict("list"), detected.to_dict("list"))
    assert mapped.summary.tp.iloc[0] == 2

    with pytest.raises(ep.EyeProcessValidationError):
        eb.benchmark_event_detector(_BadFrame(), detected)
    with pytest.raises(ep.EyeProcessValidationError):
        eb.benchmark_event_detector(pd.DataFrame({"start_time": [0]}), detected)

    invalid = truth.copy()
    invalid.loc[0, "end_time"] = -1
    with pytest.raises(ep.EyeProcessValidationError):
        eb.benchmark_event_detector(invalid, detected)

    group_truth = truth.assign(group=["g1", "g2"])
    group_detected = detected.assign(group=["g2", "g2"])
    grouped = eb.benchmark_event_detector(group_truth, group_detected, group="group")
    assert grouped.summary.tp.iloc[0] == 1

    duplicate_truth = pd.DataFrame(
        {
            "start_time": [0.0, 0.0],
            "end_time": [10.0, 10.0],
            "event_type": ["fixation", "fixation"],
        }
    )
    one_detected = duplicate_truth.iloc[[0]].copy()
    duplicate = eb.benchmark_event_detector(duplicate_truth, one_detected, minimum_iou=0.1)
    assert duplicate.summary.tp.iloc[0] == 1

    instantaneous = eb.benchmark_event_detector(
        pd.DataFrame({"start_time": [1.0], "end_time": [1.0], "event_type": ["fixation"]}),
        pd.DataFrame({"start_time": [1.0], "end_time": [1.0], "event_type": ["fixation"]}),
    )
    assert instantaneous.summary.tp.iloc[0] == 1

    with pytest.raises(ep.EyeProcessValidationError):
        eb.event_boundary_error({})


def _event_trace(n: int = 120) -> pd.DataFrame:
    t = np.arange(n) * 2.0
    x = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.02
    y = np.cos(np.linspace(0, 4 * np.pi, n)) * 0.02
    x[40:50] += np.linspace(0, 1.0, 10)
    x[50:] += 1.0
    return pd.DataFrame({"timestamp_ms": t, "gaze_x_deg": x, "gaze_y_deg": y})


def test_advanced_events_residual_helpers_empty_events_and_binocular(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError):
        ae.detect_events_ivvt(_BadFrame())
    with pytest.raises(ep.EyeProcessValidationError):
        ae.detect_events_ivvt(pd.DataFrame({"timestamp_ms": [0.0]}))

    labels = np.asarray(["fixation", "fixation", "unclassified"], dtype=object)
    empty_episode = ae._episodes(
        labels,
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.0, 1.0, 1.0]),
        minimum_duration_ms=10.0,
    )
    assert empty_episode.empty

    low_hz = _event_trace().iloc[::20].reset_index(drop=True)
    with pytest.raises(ep.EyeProcessValidationError):
        ae.detect_events_directional(low_hz, minimum_sampling_hz=100)

    direct = ae.detect_smooth_pursuits(
        _event_trace(),
        method="directional",
        minimum_velocity=0.001,
        maximum_velocity=100,
        maximum_direction_change_deg=180,
        minimum_duration_ms=0,
    )
    assert direct.method == "directional"

    no_pursuit = EyeResult(
        {
            "events": pd.DataFrame(columns=["event_type", "duration_ms", "mean_velocity"]),
            "sampling_hz": 500.0,
            "method": "ivvt",
        },
        eyeprocess_class="eye_ivvt_events",
    )
    assert ae.summarise_pursuits(no_pursuit).n_pursuits.iloc[0] == 0
    assert np.isnan(ae._robust_sigma(np.asarray([1.0, 2.0])))

    noisy = _event_trace()
    empty_micro = ae.detect_microsaccades(
        noisy,
        lambda_threshold=1e9,
        minimum_duration_ms=0,
        maximum_amplitude_deg=5,
    )
    assert empty_micro.events.empty
    assert ae.summarise_microsaccades(empty_micro).n_microsaccades.iloc[0] == 0

    original = ae.detect_microsaccades

    def fake_recursive(*args, **kwargs):
        return EyeResult({"events": pd.DataFrame()}, eyeprocess_class="eye_microsaccades")

    def fake_episodes(*args, **kwargs):
        return pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "event_type": "microsaccade",
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "duration_ms": 2.0,
                    "mean_velocity": 1.0,
                    "peak_velocity": 2.0,
                }
            ]
        )

    binocular = noisy.assign(right_x=noisy.gaze_x_deg, right_y=noisy.gaze_y_deg)
    monkeypatch.setattr(ae, "_episodes", fake_episodes)
    monkeypatch.setattr(ae, "detect_microsaccades", fake_recursive)
    filtered = original(
        binocular,
        lambda_threshold=3,
        minimum_duration_ms=0,
        maximum_amplitude_deg=5,
        right_x="right_x",
        right_y="right_y",
    )
    assert filtered.events.empty


def _pose_and_gaze() -> tuple[pd.DataFrame, pd.DataFrame]:
    gaze = pd.DataFrame({"timestamp": [0.0, 0.5, 1.0], "gaze_x_px": [0.0, 5.0, -5.0], "gaze_y_px": [0.0, 3.0, -3.0]})
    pose = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "qx": [0.0, 0.0],
            "qy": [0.0, 0.0],
            "qz": [0.0, 0.0],
            "qw": [1.0, 1.0],
            "px": [0.0, 0.0],
            "py": [0.0, 0.0],
            "pz": [0.0, 0.0],
        }
    )
    return gaze, pose


def test_naturalistic_coordinate_residual_conversion_and_guards():
    gaze, pose = _pose_and_gaze()
    center = nc.screen_gaze_to_head_vectors(
        gaze.to_dict("list"),
        screen_width_px=100,
        screen_height_px=100,
        screen_width_mm=100,
        screen_height_mm=100,
        viewing_distance_mm=100,
        origin="center",
    )
    assert center.attrs["coordinate_provenance"]["origin"] == "center"
    assert nc.pixels_to_visual_angle([4.0], pixels_per_mm=4.0, viewing_distance_mm=1.0)[0] == pytest.approx(45.0)

    with pytest.raises(ep.EyeProcessValidationError):
        nc.screen_gaze_to_head_vectors(
            _BadFrame(),
            screen_width_px=100,
            screen_height_px=100,
            screen_width_mm=100,
            screen_height_mm=100,
            viewing_distance_mm=100,
        )
    with pytest.raises(ep.EyeProcessValidationError):
        nc.screen_gaze_to_head_vectors(
            pd.DataFrame({"gaze_x_px": [0.0]}),
            screen_width_px=100,
            screen_height_px=100,
            screen_width_mm=100,
            screen_height_mm=100,
            viewing_distance_mm=100,
        )

    nonfinite_pose = pose.copy()
    nonfinite_pose.loc[0, "px"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError):
        nc.align_head_pose_to_gaze(gaze, nonfinite_pose)

    with pytest.raises(ep.EyeProcessValidationError):
        nc.propagate_coordinate_uncertainty(
            gaze,
            screen_width_px=100,
            screen_height_px=100,
            screen_width_mm=100,
            screen_height_mm=100,
            viewing_distance_mm=-1,
            viewing_distance_sd_mm=0,
            draws=2,
        )


def test_scanpath_residual_string_and_invalid_scalar():
    assert ss.levenshtein_scanpath("A > B", "A > B") == 0
    with pytest.raises(ep.EyeProcessValidationError):
        ss.levenshtein_scanpath(1, ["A"])


def _discrete_spec(n: int = 60):
    rng = np.random.default_rng(2026)
    condition = rng.choice(["A", "B"], n)
    trust = np.where(rng.random(n) < 0.5, "H", "L")
    choice = np.where(rng.random(n) < np.where(trust == "H", 0.8, 0.2), "Y", "N")
    data = pd.DataFrame(
        {
            "participant_id": [f"p{i // 2}" for i in range(n)],
            "condition": condition,
            "trust": trust,
            "choice": choice,
        }
    )
    return prepare_bayesian_network_data(
        data,
        nodes={
            "condition": {"type": "categorical"},
            "trust": {"type": "categorical"},
            "choice": {"type": "categorical"},
        },
        participant_id="participant_id",
        trial_id=None,
        stimulus_id=None,
    )


def _fitted_discrete():
    spec = _discrete_spec()
    return fit_bayesian_network(
        [("condition", "trust"), ("trust", "choice")],
        data=spec,
        estimator="bayesian",
        equivalent_sample_size=5,
    )


@pytest.mark.filterwarnings("ignore:Unconstrained")
def test_bn_robustness_residual_guards_sampling_and_calibration(monkeypatch):
    pytest.importorskip("pgmpy")
    fitted = _fitted_discrete()

    with pytest.raises(ep.EyeProcessValidationError):
        br._require_result("bad")  # type: ignore[arg-type]
    with pytest.raises(ep.EyeProcessValidationError):
        br._require_result(replace(fitted, fitted=False), fitted=True)
    assert not br._acyclic(list(fitted.nodes), [("condition", "condition")])

    no_cpd = replace(
        fitted,
        backend_model=SimpleNamespace(get_cpds=lambda node: None),
    )
    with pytest.raises(ep.EyeProcessValidationError):
        br.cpt_sensitivity_analysis(
            no_cpd,
            node="trust",
            state="H",
            values=(0.5,),
            target="choice",
        )
    with pytest.raises(ep.EyeProcessValidationError):
        br.cpt_sensitivity_analysis(
            fitted,
            node="trust",
            state="H",
            values=(0.5,),
            target="choice",
            parent_configuration={"condition": "missing"},
        )

    no_changes = br.structural_perturbation_sensitivity(
        fitted,
        include_delete=False,
        include_reverse=False,
    )
    assert no_changes.provenance["planned_perturbations"] == 0

    continuous_data = pd.DataFrame(
        {"x": np.linspace(-1, 1, 20), "y": np.linspace(-1, 1, 20) ** 2}
    )
    continuous = prepare_bayesian_network_data(
        continuous_data,
        nodes={"x": {"type": "continuous"}, "y": {"type": "continuous"}},
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
    )
    with pytest.raises(ep.EyeProcessValidationError):
        br.discretization_sensitivity(
            continuous,
            columns=("x",),
            schemes={"badcuts": {"method": "cuts", "bins": [0, 0, 1]}},
        )
    with pytest.raises(ep.EyeProcessValidationError):
        br.discretization_sensitivity(
            continuous,
            columns=("x",),
            schemes={"outside": {"method": "cuts", "bins": [-0.1, 0, 0.1]}},
        )
    with pytest.raises(ep.EyeProcessValidationError):
        br.measurement_noise_sensitivity(_discrete_spec(), columns=("condition",))

    incomplete = continuous
    incomplete.data.loc[0, "x"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError):
        br.measurement_noise_sensitivity(incomplete, columns=("x",), repeats=1)

    ungrouped_curve = br.sample_size_stability_curve(
        _discrete_spec(),
        fractions=(0.8,),
        repeats=1,
        resample_by=None,
    )
    assert len(ungrouped_curve) == 1

    calls = {"n": 0}
    original_learn = br.learn_bayesian_network

    def fail_after_baseline(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(edges=(("condition", "trust"),))
        raise RuntimeError("planned sampled fit failure")

    monkeypatch.setattr(br, "learn_bayesian_network", fail_after_baseline)
    failed_curve = br.sample_size_stability_curve(
        _discrete_spec(),
        fractions=(0.8,),
        repeats=1,
    )
    assert failed_curve.status.iloc[0] == "failed"
    monkeypatch.setattr(br, "learn_bayesian_network", original_learn)

    with pytest.raises(ep.EyeProcessValidationError):
        br.predictive_calibration(
            replace(fitted, model_family="gaussian"),
            target="choice",
        )
    explicit = br.predictive_calibration(
        fitted,
        target="choice",
        evidence_nodes=("trust",),
    )
    assert explicit.evidence_nodes == ("trust",)
    with pytest.raises(ep.EyeProcessValidationError):
        br.predictive_calibration(
            fitted,
            target="choice",
            evidence_nodes=("choice",),
        )
    with pytest.raises(ep.EyeProcessValidationError):
        br.predictive_calibration(
            fitted,
            target="choice",
            evidence_nodes=("missing",),
        )

    missing_target = fitted.data_spec.data.copy()
    missing_target.loc[0, "choice"] = pd.NA
    one_missing = br.predictive_calibration(
        fitted,
        target="choice",
        data=missing_target,
    )
    assert one_missing.summary["n"] == len(missing_target) - 1

    unknown_state = fitted.data_spec.data.copy()
    unknown_state.loc[0, "choice"] = "UNKNOWN"
    with pytest.raises(ep.EyeProcessValidationError):
        br.predictive_calibration(
            fitted,
            target="choice",
            data=unknown_state,
        )

    no_targets = fitted.data_spec.data.copy()
    no_targets["choice"] = pd.NA
    with pytest.raises(ep.EyeProcessValidationError):
        br.predictive_calibration(
            fitted,
            target="choice",
            data=no_targets,
        )
