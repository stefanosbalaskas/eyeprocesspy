from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep


def _assert_plot(ax) -> None:
    assert hasattr(ax, "eyeprocess_plot_data")
    assert ax.eyeprocess_plot_data is not None
    plt.close(ax.figure)


def _trial_data(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "person_id": np.repeat([f"P{i}" for i in range(8)], 10)[:n],
            "item_id": np.tile([f"I{i}" for i in range(10)], 8)[:n],
            "person": np.repeat([f"P{i}" for i in range(8)], 10)[:n],
            "item": np.tile([f"I{i}" for i in range(10)], 8)[:n],
            "session": np.tile(np.repeat([1, 2], 5), 8)[:n],
            "device": np.tile(["A", "B"], 40)[:n],
            "dwell_ms": rng.normal(1000, 120, n),
            "pupil": rng.normal(0.1, 0.03, n),
            "observed": rng.binomial(1, 0.8, n),
        }
    )


def _gaze_data(n: int = 100, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(0, 0.03, n)) + 0.5,
            "y": np.cumsum(rng.normal(0, 0.03, n)) + 0.5,
            "time": np.arange(n),
            "pupil": rng.normal(1, 0.1, n),
        }
    )


def test_p4_uncertainty_plot_data_contracts() -> None:
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        {
            "dwell": rng.normal(1000, 100, 60),
            "pupil": rng.normal(0.1, 0.03, 60),
        }
    )
    spec = ep.process_uncertainty_spec(
        source_sd={"calibration": 3.0, "preprocessing": 2.0}, draws=30
    )
    fit = ep.estimate_process_uncertainty(data, spec, metrics=["dwell", "pupil"])

    _assert_plot(ep.plot_uncertainty_by_item(fit, metric="dwell"))
    _assert_plot(ep.plot_uncertainty_by_stage(fit, metric="dwell"))
    _assert_plot(ep.plot_uncertainty_tornado(fit, metric="dwell"))


def test_p4_calibration_plot_data_contracts() -> None:
    rng = np.random.default_rng(12)
    reference = pd.DataFrame(
        {"target_x": rng.random(60), "target_y": rng.random(60)}
    )
    observed = reference.assign(
        x=lambda z: z.target_x + 0.05 + rng.normal(0, 0.005, 60),
        y=lambda z: z.target_y - 0.03 + rng.normal(0, 0.005, 60),
        time=np.arange(60),
    )
    drift = ep.detect_calibration_drift(
        observed, window=10, x_col="x", y_col="y", time_col="time"
    )

    _assert_plot(ep.plot_calibration_error_ellipses(drift))
    _assert_plot(ep.plot_drift_over_time(drift))
    _assert_plot(ep.plot_screen_coverage(drift))


def test_p4_reliability_components_and_plot_contracts() -> None:
    data = _trial_data(seed=13)
    gstudy = ep.fit_process_gstudy(
        data, "dwell_ms", facets=["person", "item", "session", "device"]
    )
    components = ep.process_variance_components(gstudy)
    assert set(components.columns) >= {"component", "variance", "proportion"}
    assert np.isclose(components.proportion.sum(), 1.0)

    _assert_plot(ep.plot_variance_components(gstudy))
    _assert_plot(ep.plot_reliability_by_metric(gstudy))
    _assert_plot(ep.plot_session_stability(gstudy))
    _assert_plot(ep.plot_item_sampling_reliability(gstudy))


def test_p4_pupil_registration_audit_and_plot_contracts() -> None:
    rng = np.random.default_rng(14)
    rows = []
    for i in range(12):
        time = np.linspace(0, 2, 50)
        rows.append(
            pd.DataFrame(
                {
                    "person_id": f"P{i + 1}",
                    "time": time,
                    "pupil": np.exp(-((time - (0.8 + (i + 1) / 100)) ** 2) / 0.08)
                    + rng.normal(0, 0.02, 50),
                }
            )
        )
    data = pd.concat(rows, ignore_index=True)
    registration = ep.register_pupil_curves(data, "time", "pupil")
    decomposition = ep.decompose_pupil_phase_amplitude(registration, components=2)
    responses = pd.DataFrame(
        rng.binomial(1, 0.6, (12, 5)), index=[f"P{i + 1}" for i in range(12)]
    )
    fitted = ep.fit_phase_amplitude_irt(responses, decomposition)

    audit = ep.audit_pupil_registration(registration)
    assert audit.eyeprocess_class == "eye_pupil_registration_audit"
    assert np.isfinite(audit.summary.peak_variance_before.iloc[0])

    _assert_plot(ep.plot_warping_functions(registration))
    _assert_plot(ep.plot_phase_amplitude_scores(decomposition))
    _assert_plot(ep.plot_item_phase_delay(registration))
    _assert_plot(ep.plot_registered_pupil_effects(fitted))


def test_p4_missingness_plot_contracts() -> None:
    data = _trial_data(seed=15)
    observation = ep.fit_process_observation_model(
        data, "observed", ["dwell_ms", "pupil"]
    )
    values = data.pupil.to_numpy().copy()
    values[data.observed.eq(0)] = np.nan
    sensitivity = ep.process_pattern_mixture(values, delta=[-1, 0, 1])
    tipping = ep.sensitivity_mnar_process(sensitivity)

    _assert_plot(ep.plot_observation_probability(observation))
    _assert_plot(ep.plot_missingness_by_time(observation, time=np.arange(len(data))))
    _assert_plot(ep.plot_missingness_by_aoi(observation, aoi=data.item.to_numpy()))
    _assert_plot(ep.plot_complete_case_sensitivity(tipping))


def test_p4_recurrence_features_and_plot_contracts() -> None:
    gaze = _gaze_data(seed=16)
    recurrence = ep.gaze_recurrence(gaze)
    windowed = ep.windowed_recurrence(recurrence, window=20, step=10)
    cross = ep.cross_recurrence(gaze.x, gaze.pupil)

    features = ep.recurrence_features(recurrence)
    assert set(features.columns) == {
        "recurrence_rate",
        "determinism",
        "laminarity",
        "trapping_time",
        "diagonal_entropy",
    }
    assert features.recurrence_rate.between(0, 1).all()

    _assert_plot(ep.plot_windowed_recurrence(windowed))
    _assert_plot(ep.plot_diagonal_recurrence_profile(recurrence))
    _assert_plot(ep.plot_crossmodal_recurrence(cross))
    _assert_plot(ep.plot_recurrence_network(recurrence))


def test_p4_fixation_point_process_plot_contracts() -> None:
    gaze = _gaze_data(seed=17)
    gaze["salience"] = np.linspace(0, 1, len(gaze))
    model = ep.fit_fixation_point_process(
        gaze,
        spatial_covariates=["salience"],
        interaction="self_exciting",
        grid_size=8,
    )

    _assert_plot(ep.plot_covariate_effect_surface(model))
    _assert_plot(ep.plot_observed_expected_fixations(model))
    _assert_plot(ep.plot_spatial_residuals(model))
    _assert_plot(ep.plot_temporal_excitation_kernel(model))


def test_p4_scanpath_plot_contracts() -> None:
    paths = {
        "A": ["stem", "evidence", "options"],
        "B": ["stem", "options", "evidence"],
        "C": ["stem", "evidence", "options", "evidence"],
        "D": ["options", "stem", "evidence"],
    }
    representative = ep.representative_scanpath(paths, method="consensus", distance="edit")

    _assert_plot(ep.plot_group_scanpath_transport(representative))
    _assert_plot(ep.plot_scanpath_atlas(representative))
    _assert_plot(ep.plot_scanpath_dispersion(representative))
    _assert_plot(ep.plot_scanpath_similarity_matrix(representative))


def test_p4_episode_plot_contracts() -> None:
    rng = np.random.default_rng(18)
    data = pd.DataFrame(
        {
            "time": np.arange(120),
            "pupil": np.r_[
                rng.normal(0, 1, 40),
                rng.normal(2, 1, 40),
                rng.normal(-1, 1, 40),
            ],
            "gaze_velocity": np.r_[
                rng.normal(1, 1, 40),
                rng.normal(3, 1, 40),
                rng.normal(0.5, 1, 40),
            ],
        }
    )
    changepoints = ep.detect_process_changepoints(
        data, ["pupil", "gaze_velocity"], time_col="time", window=8
    )
    episodes = ep.label_process_episodes(ep.segment_process_episodes(changepoints))

    _assert_plot(ep.plot_changepoint_ribbons(episodes))
    _assert_plot(ep.plot_episode_duration_distribution(episodes))
    _assert_plot(ep.plot_episode_transition_graph(episodes))
    _assert_plot(ep.plot_episode_waterfall(episodes))


def test_p4_evidence_graph_plot_contracts() -> None:
    graph = ep.build_evidence_graph(
        raw_data=["pupil_samples", "gaze_samples"],
        transformations=["blink_removal", "baseline_correction"],
        metrics=["pupil_auc", "aoi_entropy"],
        models=["functional_pupil_model"],
        diagnostics=["high_effort_evidence"],
        decisions=["item_I01_revise"],
    )
    trace = ep.trace_item_decision(graph, "I01")

    _assert_plot(ep.plot_item_decision_path(trace))
    _assert_plot(ep.plot_metric_dependency_graph(graph))
    _assert_plot(ep.plot_model_decision_impact(graph))
