from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy.detector_multiverse as dm
import eyeprocesspy.dynamic_irt as di
import eyeprocesspy.evidence_graph as eg
import eyeprocesspy.importers as imp
import eyeprocesspy.irt as irt
import eyeprocesspy.irt_validation_07 as iv
import eyeprocesspy.measurement_accountability_11 as ma
import eyeprocesspy.measurement_intelligence as mi
import eyeprocesspy.multimodal_staged as mm
import eyeprocesspy.plots_aoi_perturbation as pap
import eyeprocesspy.plots_functional_pupil as pfp
import eyeprocesspy.plots_irt as pirt
import eyeprocesspy.plots_operational_08 as pop
import eyeprocesspy.plots_process_irt_07 as ppirt
import eyeprocesspy.process_dynamics as pdyn
import eyeprocesspy.process_quality_09 as pq
import eyeprocesspy.pupil_missingness as pm
import eyeprocesspy.spatial_quality as sq
import eyeprocesspy.survival as surv
from eyeprocesspy.irt import EyeResult


def _close_all() -> None:
    plt.close("all")


def _survival_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "trial_id": ["1", "2", "1", "2", "1", "2"],
            "stimulus_id": ["S"] * 6,
            "condition": ["A", "B"] * 3,
            "target_aoi": ["x"] * 6,
            "time_origin": ["trial_start"] * 6,
            "event_time": [1.0, np.nan, 2.0, 2.5, np.nan, 4.0],
            "censor_time": [5.0] * 6,
            "analysis_time": [1.0, 5.0, 2.0, 2.5, 5.0, 4.0],
            "event_observed": [1, 0, 1, 1, 0, 1],
            "event_type": ["first_fixation"] * 6,
            "n_valid_samples": [300] * 6,
            "valid_data_fraction": [0.98] * 6,
            "trial_duration": [5.0] * 6,
            "analysis_eligible": [True] * 6,
        }
    )


def test_dynamic_defaults_availability_and_default_simulation_paths():
    availability = pd.DataFrame(
        {
            "item_id": [
                "I1",
                "I2",
                "not_an_item",
                "I1",
            ],
            "strategy": [
                "visual",
                "analytic",
                "visual",
                "not_a_strategy",
            ],
            "available": [
                True,
                True,
                False,
                False,
            ],
        }
    )

    spec = di.theory_strategy_spec(
        {
            "visual": {"f1": 1.0, "f2": 0.2},
            "analytic": {"f1": -0.5, "f2": 1.0},
        },
        item_availability=availability,
        chains=3,
        parallel_chains=None,
    )

    assert spec.parallel_chains == 3

    data = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "score": [1, 0, 0, 1],
            "f1": [0.8, -0.2, 0.6, -0.4],
            "f2": [0.1, 0.9, 0.2, 1.1],
        }
    )

    prepared = di.prepare_strategy_mixture_data(
        data,
        spec,
        standardize=False,
    )

    assert prepared.availability.shape == (4, 2)

    simulated = di.simulate_strategy_mixture_data(
        n_person=2,
        n_item=2,
        signatures=None,
        seed=20260920,
    )

    assert len(simulated) == 4
    assert {"feature_1", "feature_2"}.issubset(simulated.columns)

    diffusion = di.gaze_diffusion_spec(
        chains=2,
        parallel_chains=None,
    )

    assert diffusion.parallel_chains == 2


def test_evidence_graph_existing_stage_and_missing_endpoint_branches():
    nodes = pd.DataFrame(
        {
            "node_id": ["n1", "n2"],
            "label": ["raw", "decision"],
            "stage": ["raw_data", "decisions"],
        }
    )

    preserved = eg._nodes(nodes, "different_stage")
    assert preserved["stage"].tolist() == ["raw_data", "decisions"]

    graph = eg.build_evidence_graph(
        raw_data=nodes,
        edges=[
            {
                "from": "missing_source",
                "to": "n1",
                "relation": "supports",
            },
            {
                "from": "n2",
                "to": "missing_target",
                "relation": "supports",
            },
        ],
    )

    audit = eg.audit_evidence_dependencies(graph)

    assert audit.summary.loc[0, "missing_source_nodes"] == 1
    assert audit.summary.loc[0, "missing_target_nodes"] == 1
    assert not bool(audit.summary.loc[0, "passed"])


def test_generic_import_session_none_and_all_missing_validity_fallback():
    frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "x": [0.1, 0.2, 0.3],
            "y": [0.2, 0.3, 0.4],
            "validity": [pd.NA, pd.NA, pd.NA],
        }
    )

    out = imp.read_eye_generic(
        frame,
        mapping={
            "timestamp": "time",
            "x": "x",
            "y": "y",
            "gaze_valid": "validity",
        },
        recording_id="R1",
        participant_id="P1",
        session_id=None,
        keep_raw=False,
        quiet=True,
    )

    assert out["gaze_samples"]["valid"].astype("boolean").fillna(False).all()


def test_process_aware_penalty_broadcasts_scalar_quality_risk():
    value = irt.eyeprocess_irt_process_aware_selection_penalty(
        information=[2.0, 3.0, 4.0],
        burden=[0.1, 0.2, 0.3],
        burden_weight=1.0,
        quality_risk=0.5,
        quality_weight=2.0,
    )

    np.testing.assert_allclose(
        value,
        [0.9, 1.8, 2.7],
    )


def test_transportability_without_max_range_and_scalar_device_group():
    transport = iv.audit_measurement_transportability(
        pd.DataFrame({"metric": [0.80, 0.82, 0.81]}),
        metric="metric",
        max_range=None,
        minimum=0.70,
    )

    assert bool(transport.loc[0, "pass"])

    linking = EyeResult(
        {
            "paired": pd.DataFrame(
                {
                    "device": ["D1", "D1", "D1"],
                    "difference": [0.01, -0.01, 0.00],
                }
            ),
            "device_col": "device",
        },
        eyeprocess_class="eye_device_linking",
    )

    equivalence = mi.audit_device_equivalence(
        linking,
        equivalence_margin=0.5,
        by=(),
    )

    assert equivalence.summary.loc[0, "device"] == "D1"
    assert bool(equivalence.summary.loc[0, "equivalent"])


def test_pupil_latency_high_resolvability_contract():
    time = np.arange(-0.50, 1.51, 0.01)

    response = np.where(
        time >= 0.20,
        0.30 * (time - 0.20),
        0.0,
    )

    pupil = 3.0 - response

    result = ma.pupil_latency_sensitivity(
        time,
        pupil,
        simulations=0,
        seed=20260920,
    )

    assert result["latency_resolvability"] == "high"
    assert np.isfinite(result["estimator_spread_ms"])
    assert result["estimator_spread_ms"] <= 50


def test_process_episode_two_episode_default_branch_and_process_quality_no_time():
    episodes = EyeResult(
        {
            "summary": pd.DataFrame(
                {
                    "episode_id": [1, 2],
                    "n_observations": [2, 2],
                }
            ),
            "data": pd.DataFrame(
                {
                    "episode_id": [1, 1, 2, 2],
                    "value": [1.0, 1.1, 2.0, 2.1],
                }
            ),
        },
        eyeprocess_class="eye_process_episodes",
    )

    labelled = pdyn.label_process_episodes(episodes)

    assert labelled.summary["episode_label"].tolist() == [
        "orientation",
        "commitment",
    ]

    precision = pq.gaze_precision_rms_s2s(
        pd.DataFrame(
            {
                "gaze_x": [0.0, 1.0, 2.0],
                "gaze_y": [0.0, 0.5, 1.0],
            }
        ),
        time=None,
        by=None,
    )

    assert precision.loc[0, "n_steps"] == 2
    assert np.isfinite(precision.loc[0, "rms_s2s"])


def test_pupil_registration_nonelastic_and_explicit_pattern_metric():
    curves = pd.DataFrame(
        {
            "person_id": ["P1"] * 4 + ["P2"] * 4,
            "time": [0.0, 1.0, 2.0, 3.0] * 2,
            "pupil": [
                1.0,
                2.0,
                3.0,
                2.0,
                1.0,
                1.5,
                2.5,
                2.0,
            ],
        }
    )

    registered = pm.register_pupil_curves(
        curves,
        "time",
        "pupil",
        method="shift",
        id_col="person_id",
        grid_size=11,
    )

    assert registered.method == "shift"

    mixture = pm.process_pattern_mixture(
        pd.DataFrame(
            {
                "metric": [1.0, np.nan, 3.0],
                "other": [9.0, 8.0, 7.0],
            }
        ),
        delta=[-1.0, 0.0, 1.0],
        metric="metric",
    )

    assert len(mixture.summary) == 3


def test_multimodal_ppc_absent_channel_branch():
    out = mm.multimodal_m2_ppc(
        pd.DataFrame(
            {
                "response": [0, 1, 1],
                "rt": [1.0, 1.2, 0.9],
            }
        )
    )

    assert set(out.summary["channel"]) == {
        "response",
        "rt",
    }


def test_aoi_plot_data_none_and_untracked_assignment_branches():
    geometry = pd.DataFrame(
        [
            {
                "aoi_id": "A",
                "shape_type": "rectangle",
                "xmin": 0.0,
                "xmax": 1.0,
                "ymin": 0.0,
                "ymax": 1.0,
            }
        ]
    )

    obj = {
        "nominal_geometry": geometry,
        "perturbed_geometry": geometry.copy(),
        "perturbation_id": "translated",
    }

    fig1, ax1 = plt.subplots()
    pap.plot_aoi_perturbations(
        obj,
        ax=ax1,
    )

    fig2, ax2 = plt.subplots()
    pap.plot_aoi_perturbations(
        obj,
        data=pd.DataFrame(
            {
                "x": [0.25, 0.75],
                "y": [0.25, 0.75],
            }
        ),
        x_col="x",
        y_col="y",
        ax=ax2,
    )

    assert len(ax2.collections) >= 1
    _close_all()


def test_functional_pupil_plot_existing_axis_and_empty_data_branches():
    fig1, ax1 = plt.subplots()

    legacy = SimpleNamespace(
        legacy=True,
        data={"features": pd.DataFrame()},
        feature_names=[],
    )

    returned = pfp.plot_eye_functional_pupil_irt(
        legacy,
        ax=ax1,
    )

    assert returned is ax1

    fig2, ax2 = plt.subplots()
    diagnostics = SimpleNamespace(
        residual_acf=pd.DataFrame(
            {
                "lag1": pd.Series(dtype=float),
            }
        )
    )

    returned = pfp.plot_eye_functional_pupil_diagnostics(
        diagnostics,
        ax=ax2,
    )

    assert returned is ax2

    fig3, ax3 = plt.subplots()
    sensitivity = SimpleNamespace(
        results=pd.DataFrame(
            {
                "parameter": pd.Series(dtype=str),
                "specification": pd.Series(dtype=str),
                "estimate": pd.Series(dtype=float),
            }
        )
    )

    returned = pfp.plot_eye_functional_pupil_sensitivity(
        sensitivity,
        ax=ax3,
    )

    assert returned is ax3
    _close_all()


def test_irt_plot_empty_finite_recovery_and_missing_expected_sbc_path():
    fig1, ax1 = plt.subplots()

    recovery = {
        "estimates": pd.DataFrame(
            {
                "b_truth": [np.nan],
                "b_estimate": [np.nan],
            }
        )
    }

    returned = pirt.plot_eye_irt_recovery_result(
        recovery,
        parameter="b",
        ax=ax1,
    )

    assert returned is ax1

    fig2, ax2 = plt.subplots()

    returned = pirt.plot_eye_irt_sbc_evidence(
        {
            "diagnostics": {
                "counts": [2, 3, 1],
            }
        },
        ax=ax2,
    )

    assert returned is ax2
    _close_all()


def test_operational_plot_finite_theta_with_no_finite_se():
    history = pd.DataFrame(
        {
            "step": [1, 2],
            "theta": [0.1, 0.2],
            "theta_se": [np.nan, np.nan],
        }
    )

    result = EyeResult(
        {
            "history": history,
            "method": "test",
        },
        eyeprocess_class="eye_streaming_score",
    )

    fig, ax = plt.subplots()
    returned = pop.plot_eye_streaming_score(
        result,
        ax=ax,
    )

    assert returned is ax
    _close_all()


def test_process_cat_plot_without_information_metric():
    data = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "item_id": ["I1", "I2", "I3"],
        }
    )

    fig, ax = plt.subplots()
    returned = ppirt.plot_eye_process_cat_simulation(
        data,
        ax=ax,
    )

    assert returned is ax
    _close_all()


def test_detector_plot_existing_axes_and_dispatch_coefficient_branch(monkeypatch):
    feature_holder = SimpleNamespace(
        features=pd.DataFrame(
            {
                "detector_id": ["d1", "d1", "d2", "d2"],
                "aoi_id": ["A", "A", "A", "A"],
                "dwell_time_ms": [100.0, 120.0, 90.0, 110.0],
            }
        )
    )

    fig1, ax1 = plt.subplots()
    returned = dm.plot_detector_feature_distributions(
        feature_holder,
        ax=ax1,
    )

    assert returned is ax1

    inference = SimpleNamespace(
        coefficients=pd.DataFrame(
            {
                "term": ["condition", "condition"],
                "detector_id": ["d1", "d2"],
                "estimate": [0.20, 0.15],
                "CI_lower": [0.05, 0.01],
                "CI_upper": [0.35, 0.29],
            }
        )
    )

    fig2, ax2 = plt.subplots()
    returned = dm.plot_detector_coefficient_stability(
        inference,
        term="condition",
        ax=ax2,
    )

    assert returned is ax2

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()

    monkeypatch.setattr(
        dm,
        "plot_detector_agreement",
        lambda *args, **kwargs: ax3,
    )
    monkeypatch.setattr(
        dm,
        "plot_detector_feature_distributions",
        lambda *args, **kwargs: ax4,
    )
    monkeypatch.setattr(
        dm,
        "plot_detector_coefficient_stability",
        lambda *args, **kwargs: ax5,
    )

    plots = dm.plot_detector_multiverse(
        object(),
        inference=inference,
        term="condition",
    )

    assert set(plots) == {
        "agreement",
        "feature",
        "coefficient",
    }

    _close_all()


def test_survival_scalar_censor_group_and_existing_axes():
    data = _survival_fixture()

    summary = surv.summarise_gaze_censoring(
        data,
        by="condition",
    )

    assert set(summary["condition"]) == {
        "A",
        "B",
    }

    fig1, ax1 = plt.subplots()
    returned = surv.plot_gaze_survival_curve(
        data,
        group=None,
        ax=ax1,
    )
    assert returned is ax1

    fig2, ax2 = plt.subplots()
    returned = surv.plot_gaze_cumulative_incidence(
        data,
        group=None,
        ax=ax2,
    )
    assert returned is ax2

    fig3, ax3 = plt.subplots()
    returned = surv.plot_gaze_hazard(
        data,
        group=None,
        ax=ax3,
    )
    assert returned is ax3

    _close_all()


def test_spatial_loss_run_with_nonfinite_endpoints_is_preserved():
    data = pd.DataFrame(
        {
            "gaze_x": [np.nan, np.nan, np.nan],
            "gaze_y": [np.nan, np.nan, np.nan],
            "timestamp_ms": [np.nan, 10.0, np.nan],
        }
    )

    out = sq.compute_gaze_data_loss(
        data,
        time="timestamp_ms",
        time_unit="ms",
    )

    assert out.loc[0, "longest_missing_run_samples"] == 3
    assert np.isnan(out.loc[0, "longest_missing_run_ms"])
