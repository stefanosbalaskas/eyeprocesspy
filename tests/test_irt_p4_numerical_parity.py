from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult


def _items(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [f"I{i + 1}" for i in range(n)],
            "a": np.linspace(0.8, 1.4, n),
            "b": np.linspace(-1.5, 1.5, n),
            "c": np.zeros(n),
            "d": np.ones(n),
        }
    )


def _responses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        columns=[f"I{i + 1}" for i in range(6)],
        index=["P1", "P2", "P3", "P4"],
        dtype=float,
    )


def test_p4_foundation_probability_information_contracts_are_numerically_coherent() -> None:
    theta = np.linspace(-3, 3, 31)
    p3 = ep.eyeprocess_irt_3pl_probability(theta, a=1.2, b=0.2, c=0.15)
    p4 = ep.eyeprocess_irt_4pl_probability(theta, a=1.2, b=0.2, c=0.1, d=0.9)
    assert np.all((p3 >= 0.15) & (p3 < 1))
    assert np.all((p4 >= 0.1) & (p4 <= 0.9))
    assert np.all(np.diff(p3) > 0)
    assert np.all(np.diff(p4) > 0)

    sem = ep.eyeprocess_irt_conditional_sem([1.0, 4.0, 0.0])
    np.testing.assert_allclose(sem[:2], [1.0, 0.5])
    assert np.isinf(sem[2])

    score3 = ep.eyeprocess_irt_expected_score(theta, family="3pl", a=1.2, b=0.2, c=0.15)
    np.testing.assert_allclose(score3, p3)
    grm = ep.eyeprocess_irt_expected_score(theta, family="grm", a=1.1, thresholds=[-1, 0, 1])
    assert np.all((grm >= 0) & (grm <= 3))
    assert np.all(np.diff(grm) > 0)

    tcc = ep.eyeprocess_irt_test_characteristic_curve(theta, _items())
    assert tcc.attrs["eyeprocess_class"] == "eye_irt_test_characteristic_curve"
    assert np.all(np.diff(tcc.expected_score) > 0)
    assert np.all(tcc.max_score == 6)

    area = ep.eyeprocess_irt_information_area([-1, 0, 1], [1, 2, 1])
    assert area == pytest.approx(3.0)


def test_p4_diagnostic_audits_detect_expected_structure_and_ppc_contracts() -> None:
    y = _responses()
    extreme = ep.eyeprocess_irt_extreme_score_audit(y, lower_fraction=0.2, upper_fraction=0.8)
    assert len(extreme) == len(y)
    assert {"lower_extreme", "upper_extreme"}.issubset(extreme.columns)

    ordered = ep.eyeprocess_irt_threshold_order_audit("I1", [-1.0, 0.0, 1.0])
    reversed_ = ep.eyeprocess_irt_threshold_order_audit("I1", [-1.0, 0.5, 0.25])
    assert ordered.ordered is True
    assert reversed_.ordered is False and reversed_.reversals == [2]

    theta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    probability = ep.eyeprocess_irt_2pl_probability(theta)
    monotone = ep.eyeprocess_irt_monotonicity_audit(theta, probability)
    assert monotone.monotone_non_decreasing is True
    assert monotone.n_decreases == 0

    cat = ep.eyeprocess_irt_category_function_audit(
        ep.eyeprocess_irt_grm_probability(theta, thresholds=[-1, 0, 1])
    )
    assert cat.valid_bounds is True
    assert cat.rows_sum_to_one is True

    plaus = ep.eyeprocess_irt_parameter_plausibility_audit(_items())
    assert not plaus.any_flag.any()

    observed = y.iloc[:, :3].to_numpy()
    replicated = np.stack([observed, 1 - observed, observed], axis=0)
    ppc = ep.eyeprocess_irt_ppc_discrepancy(observed, replicated, statistic="mean_score")
    assert ppc.eyeprocess_class == "eye_irt_ppc_discrepancy"
    assert 0 <= ppc.posterior_predictive_p <= 1

    dashboard = ep.eyeprocess_irt_fit_dashboard(
        item_fit=pd.DataFrame({"item": ["I1"]}),
        parameter_audit=plaus,
    )
    assert dashboard.n_components == 2
    assert set(dashboard.present) == {"item_fit", "parameter_audit"}


def test_p4_scoring_precision_targeting_and_exposure_contracts() -> None:
    items = _items()
    y = _responses()
    scores = ep.eyeprocess_irt_score_table(y, items, method="EAP")
    assert len(scores) == len(y)
    assert np.all(np.isfinite(scores.estimate))
    assert np.all(scores.se > 0)

    reliability = ep.eyeprocess_irt_marginal_reliability(scores.estimate, scores.se)
    assert np.isnan(reliability) or 0 <= reliability <= 1

    uncertainty = ep.eyeprocess_irt_score_uncertainty(scores)
    assert uncertainty.n == len(y)
    assert uncertainty.mean_se > 0

    targeting = ep.eyeprocess_irt_information_targeting(items, [-1, 0, 1], weights=[1, 2, 1])
    assert targeting.weighted_information > 0
    assert targeting.weighted_sem > 0
    np.testing.assert_allclose(targeting.weights.sum(), 1.0)

    gain = ep.eyeprocess_irt_information_gain([1.0, 0.8], [0.8, 0.5])
    assert np.all(gain > 0)

    exposure = ep.eyeprocess_irt_exposure_summary(
        ["I1", "I1", "I2", "I3"], item_bank_ids=["I1", "I2", "I3", "I4"]
    )
    assert exposure.set_index("item_id").loc["I1", "rate"] == pytest.approx(0.5)
    assert exposure.set_index("item_id").loc["I4", "rate"] == 0

    bank = ep.eyeprocess_irt_item_bank(items, content=["A", "A", "A", "B", "B", "B"])
    assert ep.validate_eyeprocess_irt_item_bank(bank) is True


def test_p4_linking_anchor_invariance_and_device_contracts() -> None:
    reference = _items()
    focal = reference.copy()
    A = 1.2
    B = -0.3
    focal["a"] = reference.a * A
    focal["b"] = (reference.b - B) / A

    mean_mean = ep.eyeprocess_irt_mean_mean_link(reference, focal)
    assert mean_mean.A == pytest.approx(A, rel=1e-12, abs=1e-12)
    assert mean_mean.B == pytest.approx(B, rel=1e-12, abs=1e-12)

    stability = ep.eyeprocess_irt_link_stability(
        reference,
        focal,
        anchor_sets={"low": ["I1", "I2", "I3"], "high": ["I4", "I5", "I6"]},
        method="mean-sigma",
    )
    assert len(stability.table) == 2
    assert np.all(stability.table.n_anchors == 3)

    dif = pd.DataFrame({"item_id": reference.item_id, "effect": [0.01, 0.02, 0.03, 0.2, 0.01, 0.02]})
    anchor_audit = ep.eyeprocess_irt_anchor_audit(reference, dif=dif, max_abs_effect=0.1)
    assert int((~anchor_audit.eligible).sum()) == 1

    purified = ep.eyeprocess_irt_anchor_purification(
        reference,
        effect_fun=lambda anchors: pd.DataFrame({"item_id": anchors, "effect": np.zeros(len(anchors))}),
        threshold=0.1,
    )
    assert purified.anchors == reference.item_id.tolist()
    assert len(purified.history) == 1

    dtf = ep.eyeprocess_irt_dtf_curve(reference, focal, theta=[-2, -1, 0, 1, 2])
    assert dtf.attrs["eyeprocess_class"] == "eye_irt_dtf_curve"
    assert np.all(dtf.absolute_difference >= 0)

    process = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "effect": [0.1, 0.2, 0.4]})
    dif_small = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "effect": [0.2, 0.3, 0.5]})
    concordance = ep.eyeprocess_irt_process_dif_concordance(dif_small, process)
    assert concordance.n == 3
    assert np.isfinite(concordance.correlation)

    device = ep.eyeprocess_irt_device_drift(
        pd.DataFrame(
            {
                "item_id": ["I1", "I1", "I2", "I2"],
                "device": ["A", "B", "A", "B"],
                "b": [-0.2, -0.1, 0.3, 0.5],
            }
        )
    )
    assert np.all(device.n_devices == 2)

    evidence = ep.eyeprocess_irt_invariance_evidence(
        anchor_audit=anchor_audit,
        dif=dif_small,
        dtf=dtf,
        linking=mean_mean,
        process_concordance=concordance,
    )
    assert evidence.completeness == 1.0


def test_p4_process_profiles_bundle_and_alignment_contracts() -> None:
    data = pd.DataFrame(
        {
            "person": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "item": ["I1", "I2", "I1", "I2", "I1", "I2"],
            "response": [1, 0, 1, 1, 0, 1],
            "rt": [1.0, 1.5, 0.9, 1.2, 1.8, 1.1],
            "pupil": [3.1, 3.4, 3.0, 3.5, 3.2, 3.6],
            "gaze": [0.2, 0.4, 0.3, 0.5, 0.25, 0.55],
        }
    )
    bundle = ep.eyeprocess_process_irt_data_bundle(
        data,
        person="person",
        item="item",
        response="response",
        response_time="rt",
        process=["pupil", "gaze"],
    )
    assert bundle.n_persons == 3 and bundle.n_items == 2

    rt = ep.eyeprocess_response_time_profile(data, "person", "item", "rt")
    assert rt.n == len(data)
    assert len(rt.item) == 2 and len(rt.person) == 3

    speed = ep.eyeprocess_speed_accuracy_profile(data, "person", "response", "rt")
    assert len(speed.person) == 3

    item_profile = ep.eyeprocess_process_item_profile(data, "item", ["pupil", "gaze"])
    person_profile = ep.eyeprocess_process_person_profile(data, "person", ["pupil", "gaze"])
    assert len(item_profile) == 2
    assert len(person_profile) == 3

    alignment_items = pd.DataFrame(
        {
            "item_id": ["I1", "I2", "I3"],
            "a": [0.8, 1.0, 1.3],
            "b": [-0.5, 0.0, 0.7],
            "c": 0.0,
            "d": 1.0,
        }
    )
    profile = pd.DataFrame(
        {
            "item_id": ["I1", "I2", "I3"],
            "mean_pupil": [3.0, 3.3, 3.8],
            "mean_gaze": [0.2, 0.35, 0.6],
        }
    )
    alignment = ep.eyeprocess_irt_process_alignment(alignment_items, profile)
    assert alignment.eyeprocess_class == "eye_irt_process_alignment"
    assert set(alignment.correlations.channel) == {"mean_pupil", "mean_gaze"}


def test_p4_external_engine_adapters_remain_explicitly_gated() -> None:
    status = ep.eyeprocess_irt_engine_status("mirt")
    assert len(status) == 1
    assert bool(status.available.iloc[0]) is False

    tam = ep.fit_eyeprocess_tam(np.array([[1, 0], [0, 1]]), model="rasch")
    lnirt = ep.fit_eyeprocess_lnirt(np.array([[1, 0], [0, 1]]), np.ones((2, 2)))
    erm = ep.fit_eyeprocess_erm(np.array([[1, 0], [0, 1]]), model="RM")
    catr = ep.simulate_eyeprocess_catr(_items(4), trueTheta=0.25)
    mirtcat = ep.run_eyeprocess_mirtcat()

    for result, engine in [
        (tam, "TAM"),
        (lnirt, "LNIRT"),
        (erm, "eRm"),
        (catr, "catR"),
        (mirtcat, "mirtCAT"),
    ]:
        assert result.eyeprocess_class == "eye_gated_irt_engine"
        assert result.engine == engine
        assert result.fit is None


def test_p4_recovery_failure_and_misspecification_summaries() -> None:
    estimates = pd.DataFrame(
        {
            "scenario_id": ["S1", "S1", "S1"],
            "a_truth": [0.8, 1.0, 1.2],
            "a_estimate": [0.82, 0.98, 1.25],
            "b_truth": [-0.5, 0.0, 0.5],
            "b_estimate": [-0.45, 0.02, 0.48],
        }
    )
    design = pd.DataFrame({"scenario_id": ["S1"], "replications": [4]})
    failures = pd.DataFrame({"scenario_id": ["S1"], "replication": [2], "error": ["fit failed"]})
    recovery = EyeResult(
        {"estimates": estimates, "design": design, "failures": failures},
        eyeprocess_class="eye_irt_recovery_result",
    )
    summary = ep.eyeprocess_irt_recovery_summary(recovery)
    assert set(summary.parameter) == {"a", "b"}
    assert np.all(summary.rmse >= 0)

    failure_summary = ep.eyeprocess_irt_recovery_failures(recovery)
    assert failure_summary.failures.iloc[0] == 1
    assert failure_summary.failure_rate.iloc[0] == pytest.approx(0.25)

    ref = pd.DataFrame({"parameter": ["a", "b"], "bias": [0.01, 0.02], "rmse": [0.1, 0.2]})
    miss = pd.DataFrame({"parameter": ["a", "b"], "bias": [0.03, 0.05], "rmse": [0.15, 0.3]})
    metrics = ep.eyeprocess_irt_misspecification_metrics(ref, miss)
    assert len(metrics) == 2
    assert np.all(metrics.rmse_misspecified >= metrics.rmse_reference)


def test_p4_multidimensional_testlet_and_latent_regression_contracts() -> None:
    directional = ep.eyeprocess_mirt_directional_information(
        theta=[0.0, 0.0], discrimination=[1.0, 0.5], difficulty=0.0, direction=[1.0, 0.0]
    )
    assert directional > 0

    regression = ep.eyeprocess_irt_latent_regression_design(
        pd.DataFrame({"age": [20.0, 30.0, 40.0], "group": [0.0, 1.0, 1.0]}),
        formula="~ age + group",
    )
    assert regression.eyeprocess_class == "eye_irt_latent_regression_design"
    assert regression.matrix.shape == (3, 3)
    assert regression.complete.all()

    spec = ep.eyeprocess_irt_testlet_spec(
        ["I1", "I2", "I3", "I4"], ["T1", "T1", "T2", "T2"]
    )
    assert spec.attrs["eyeprocess_class"] == "eye_irt_testlet_spec"
    audit = ep.eyeprocess_irt_testlet_audit(spec, min_items=2)
    assert len(audit) == 2
    assert audit.meets_minimum.all()


def test_p4_prior_specification_and_sensitivity_summary_contracts() -> None:
    prior = ep.eyeprocess_irt_prior_spec(
        discrimination="lognormal",
        difficulty="normal",
        guessing="beta",
        location=0,
        scale=1.5,
        guessing_shape=[4, 16],
        label="robust",
    )
    assert prior.eyeprocess_class == "eye_irt_prior_spec"
    assert prior.scale == pytest.approx(1.5)

    sensitivity = ep.eyeprocess_irt_prior_sensitivity_summary(
        pd.DataFrame({"prior_id": ["P1", "P2", "P3"], "estimate": [0.1, 0.2, 0.4]})
    )
    assert sensitivity.n_specifications == 3
    assert sensitivity.n_finite == 3
    assert sensitivity.median == pytest.approx(0.2)
    assert sensitivity.range == pytest.approx(0.3)
