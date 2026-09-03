from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.process_irt_07 as pi
from eyeprocesspy.exceptions import (
    EyeProcessBackendError,
    EyeProcessModelError,
    EyeProcessValidationError,
)
from eyeprocesspy.irt import EyeResult


def _binary_data(n: int = 24) -> pd.DataFrame:
    i = np.arange(n)
    return pd.DataFrame(
        {
            "participant_id": [f"p{x % 6}" for x in i],
            "item_id": [f"i{x % 4}" for x in i],
            "response": (i % 3 != 0).astype(int),
            "rt": 1.0 + (i % 5) * 0.2,
            "fixation_count": (i % 7).astype(float),
            "theta": np.linspace(-1.5, 1.5, n),
            "group": np.where(i % 2 == 0, "A", "B"),
            "device": np.where(i % 2 == 0, "d1", "d2"),
            "item_order": i % 8 + 1,
            "reached": True,
        }
    )


def _nominal_data(n: int = 36) -> pd.DataFrame:
    i = np.arange(n)
    cats = np.array(["A", "B", "C"])
    response = cats[i % 3]
    return pd.DataFrame(
        {
            "participant_id": [f"p{x % 9}" for x in i],
            "item_id": [f"i{x % 4}" for x in i],
            "response_option": response,
            "correct_col": cats[(i + 1) % 3],
            "ability": np.linspace(-1, 1, n),
            "gA": 1.0 + (i % 5),
            "gB": 1.5 + ((i + 1) % 5),
            "gC": 2.0 + ((i + 2) % 5),
        }
    )


def _manyfacet_data() -> pd.DataFrame:
    d = _binary_data(24)
    d["process"] = 1.0 + np.arange(len(d)) / 10
    d["constant_device"] = "only"
    return d


def test_scalar_choice_channel_and_model_spec_guards():
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        pi._scalar_chr("", "id")
    with pytest.raises(EyeProcessValidationError, match="must be one of"):
        pi._choice("bad", ("a", "b"), "mode")

    with pytest.raises(EyeProcessValidationError, match="at least two"):
        ep.irt_compositional_channel(["a"])
    with pytest.raises(EyeProcessValidationError, match="lower < upper"):
        ep.irt_continuous_channel(lower=1, upper=1)
    with pytest.raises(EyeProcessValidationError, match="lower < upper"):
        ep.irt_continuous_channel(lower=np.nan, upper=1)

    response = ep.irt_response_channel()
    seq_spec = ep.irt_model_spec("resid_seq_spec", "ability", [response])
    assert list(seq_spec.channels) == ["channel_1"]

    with pytest.raises(EyeProcessValidationError, match="channels"):
        ep.irt_model_spec("resid_bad_channels", "ability", [])
    with pytest.raises(EyeProcessValidationError, match="channels"):
        ep.irt_model_spec("resid_bad_channels_2", "ability", object())
    with pytest.raises(EyeProcessValidationError, match="latent"):
        ep.irt_model_spec("resid_bad_latent", [], {"response": response})
    with pytest.raises(EyeProcessValidationError, match="callbacks"):
        ep.irt_model_spec(
            "resid_bad_callback",
            "ability",
            {"response": response},
            fit_fun="not callable",
        )


def test_registry_duplicate_unavailable_and_unknown_model_guards():
    with pytest.raises(EyeProcessValidationError, match="spec"):
        ep.register_irt_model({})

    spec = ep.irt_model_spec(
        "resid_duplicate_model",
        "ability",
        {"response": ep.irt_response_channel()},
    )
    ep.register_irt_model(spec, overwrite=True)
    with pytest.raises(EyeProcessValidationError, match="already registered"):
        ep.register_irt_model(spec)

    unavailable = pi._unavailable("future_fit")
    assert unavailable.__name__ == "future_fit"
    with pytest.raises(EyeProcessBackendError, match="not yet parity-validated"):
        unavailable(None)

    with pytest.raises(EyeProcessValidationError, match="Unknown IRT model"):
        ep.get_irt_model("definitely_missing_model")


def test_fit_simulate_validate_interfaces_cover_residual_paths():
    with pytest.raises(EyeProcessValidationError, match="model id"):
        ep.fit_irt_model({}, pd.DataFrame())
    with pytest.raises(EyeProcessValidationError, match="model id"):
        ep.simulate_irt_model({}, n=2)
    with pytest.raises(EyeProcessValidationError, match="model id"):
        ep.validate_irt_model({})

    nofit = ep.irt_model_spec(
        "resid_nofit",
        "ability",
        {"response": ep.irt_response_channel()},
        status="reference",
    )
    with pytest.raises(EyeProcessBackendError, match="no bundled fitter"):
        ep.fit_irt_model(nofit, pd.DataFrame())

    fit_spec = ep.irt_model_spec(
        "resid_fit_attach",
        "ability",
        {"response": ep.irt_response_channel()},
        status="reference",
        fit_fun=lambda data: EyeResult({"n": len(data)}, eyeprocess_class="resid_fit"),
    )
    fit = ep.fit_irt_model(fit_spec, pd.DataFrame({"x": [1, 2]}))
    assert fit.eye_irt_model_spec.id == "resid_fit_attach"

    no_sim = ep.irt_model_spec(
        "resid_no_sim",
        "ability",
        {"response": ep.irt_response_channel()},
    )
    with pytest.raises(EyeProcessBackendError, match="does not register a simulator"):
        ep.simulate_irt_model(no_sim)

    sim_spec = ep.irt_model_spec(
        "resid_sim",
        "ability",
        {"response": ep.irt_response_channel()},
        simulate_fun=lambda n=2: list(range(n)),
    )
    with pytest.raises(EyeProcessModelError, match="disabled"):
        ep.simulate_irt_model(sim_spec, n=2, allow_experimental=False)
    assert ep.simulate_irt_model(sim_spec, n=3) == [0, 1, 2]

    with pytest.raises(EyeProcessValidationError, match="validation object"):
        ep.validate_irt_model(no_sim)

    custom_validation = ep.irt_model_spec(
        "resid_custom_validator",
        "ability",
        {"response": ep.irt_response_channel()},
        validate_fun=lambda validation=None, marker=None: (validation, marker),
    )
    assert ep.validate_irt_model(custom_validation, {"x": 1}, marker="ok") == ({"x": 1}, "ok")


def test_evidence_grade_model_metric_compare_and_promotion_residuals():
    assert pi._grade_evidence({"pass": False}).pass_value is False
    assert pi._grade_evidence({"grade": "green"}).pass_value is True
    assert pi._grade_evidence({"other": 1}).grade == "review"
    assert pi._grade_evidence(True).grade == "pass"

    class MetricObject:
        AIC = 12.0

        def BIC(self):
            return 13.0

    class Nested:
        logLik = -4.0

    assert pi._model_metric({"AIC": "not-a-number", "model": MetricObject()}, "AIC") == 12.0
    assert pi._model_metric({"model": Nested()}, "logLik") == -4.0
    assert pi._model_metric(MetricObject(), "BIC") == 13.0
    assert math.isnan(pi._model_metric({}, "missing"))

    with pytest.raises(EyeProcessValidationError, match="at least one"):
        ep.compare_irt_models()
    with pytest.raises(EyeProcessValidationError, match="names"):
        ep.compare_irt_models({"AIC": 1}, {"AIC": 2}, names=["one"])
    default = ep.compare_irt_models({"AIC": 1})
    assert default.model.tolist() == ["model_1"]

    spec = ep.irt_model_spec(
        "resid_promote",
        "ability",
        {"response": ep.irt_response_channel()},
    )
    with pytest.raises(EyeProcessValidationError, match="target"):
        ep.promote_irt_model(spec, True, target="bad")
    with pytest.raises(EyeProcessValidationError, match="registered model"):
        ep.promote_irt_model({}, True)
    with pytest.raises(EyeProcessModelError, match="cannot be promoted"):
        ep.promote_irt_model(spec, False, target="reference")

    promoted = ep.promote_irt_model(spec, True, target="reference", update_registry=True)
    assert promoted.to == "reference"
    assert ep.get_irt_model(spec.id).status == "reference"


def test_design_and_reference_fit_helper_fallbacks():
    d = pd.DataFrame({"x": [1.0, 2.0], "cat": ["only", "only"]})
    X, names = pi._dummy_design(d, intercept=False)
    assert X.shape == (2, 0)
    assert names == []

    X, names = pi._dummy_design(d, continuous=["x"], categorical=["cat"], intercept=False)
    assert X.shape == (2, 1)
    assert names == ["x"]

    empty_logit = pi._logistic_fit(np.ones((0, 1)), np.array([], dtype=float))
    assert empty_logit.convergence == 1
    assert math.isnan(empty_logit.logLik)

    one_class = pi._logistic_fit(np.ones((3, 1)), np.ones(3))
    assert one_class.convergence == 1
    assert np.allclose(one_class.fitted, 1)

    no_ols = pi._ols_fit(np.array([[np.nan], [np.nan]]), np.array([1.0, 2.0]))
    assert np.isnan(no_ols.coefficients).all()
    assert math.isnan(no_ols.logLik)

    with pytest.raises(EyeProcessValidationError, match="two response categories"):
        pi._multinomial_fit(np.ones((3, 1)), ["A", "A", "A"])


def test_joint_and_graded_model_validation_gates():
    d = _binary_data()
    with pytest.raises(EyeProcessBackendError, match="brms"):
        ep.fit_joint_gaze_rt_irt(d, engine="brms")

    bad = d.copy()
    bad.loc[0, "response"] = 2
    with pytest.raises(EyeProcessValidationError, match="0/1"):
        ep.fit_joint_gaze_rt_irt(bad)

    bad = d.copy()
    bad.loc[0, "rt"] = 0
    with pytest.raises(EyeProcessValidationError, match="strictly positive"):
        ep.fit_joint_gaze_rt_irt(bad)

    bad = d.copy()
    bad.loc[0, "fixation_count"] = -1
    with pytest.raises(EyeProcessValidationError, match="non-negative"):
        ep.fit_joint_gaze_rt_irt(bad)

    with pytest.raises(EyeProcessBackendError, match="brms"):
        ep.fit_joint_graded_rt_process_irt(d, engine="brms")
    bad = d.copy()
    bad.loc[0, "rt"] = 0
    with pytest.raises(EyeProcessValidationError, match="positive"):
        ep.fit_joint_graded_rt_process_irt(bad)

    gaussian = ep.fit_joint_graded_rt_process_irt(
        d,
        process="fixation_count",
        process_family="gaussian",
    )
    assert gaussian.eyeprocess_class == "eye_joint_graded_rt_process_irt"


@pytest.mark.parametrize(
    ("correct_option", "ability"),
    [
        ("correct_col", None),
        ("A", None),
        (np.array(["A", "B", "C"] * 12), None),
        (None, "ability"),
    ],
)
def test_nominal_gaze_alternate_correctness_and_ability_paths(correct_option, ability):
    d = _nominal_data()
    fit = ep.fit_nominal_gaze_irt(
        d,
        option_gaze=["gA", "gB", "gC"],
        correct_option=correct_option,
        ability=ability,
        add_item_effects=False,
    )
    assert fit.eyeprocess_class == "eye_nominal_gaze_irt"
    assert all(not name.startswith("item_id_") for name in fit.design_names)


def test_nominal_and_distractor_guards_and_missing_category_path():
    with pytest.raises(EyeProcessValidationError, match="at least two"):
        ep.fit_nominal_gaze_irt(_nominal_data(), option_gaze=["gA"])
    with pytest.raises(EyeProcessValidationError, match="eye_nominal_gaze_irt"):
        ep.option_process_information({})
    with pytest.raises(EyeProcessValidationError, match="eye_nominal_gaze_irt"):
        ep.distractor_process_map({})

    d = pd.DataFrame(
        {
            "response_option": ["A", "missing"],
            "gA": [3.0, 1.0],
            "gB": [1.0, 2.0],
        }
    )
    out = ep.audit_distractor_attention(d, option_gaze=["gA", "gB"])
    assert out.n.iloc[0] == 0
    assert np.isnan(out.mean_difference.iloc[0])


def test_missingness_omitted_after_reach_and_exposure_single_class_fallback():
    d = pd.DataFrame({"response": [np.nan, 1.0], "reached": [True, True]})
    states = ep.classify_item_missingness(d).astype(str).tolist()
    assert states == ["omitted_after_reach", "answered"]

    exposure = pd.DataFrame({"reached": [True, True, True], "x": [1.0, 2.0, 3.0]})
    fit = ep.estimate_visual_exposure_probability(exposure, predictors=["x"])
    assert fit.model.convergence == 1
    assert np.allclose(fit.fitted_probability, 1)


def test_omission_survival_omission_time_fallback_and_no_response_model():
    d = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2"],
            "item_id": ["i1", "i2", "i1"],
            "response": [np.nan, np.nan, np.nan],
            "response_time": [np.nan, 0.0, np.nan],
            "omission_time": [2.0, np.nan, np.nan],
            "reached": [True, False, True],
            "gaze": [True, False, True],
        }
    )
    fit = ep.fit_omission_survival_irt(
        d,
        omission_time="omission_time",
        gaze_exposure="gaze",
    )
    assert fit.response_model is None
    assert np.isfinite(fit.classified_data[".time"]).all()
    assert fit.classified_data[".time"].min() > 0

    none_finite = d.copy()
    none_finite["omission_time"] = np.nan
    fit2 = ep.fit_omission_survival_irt(none_finite, omission_time="omission_time")
    assert np.allclose(fit2.classified_data[".time"], 1.0)


def test_manyfacet_single_level_no_process_and_facet_guards():
    d = _manyfacet_data()
    no_process = ep.fit_manyfacet_process_irt(
        d,
        response="response",
        process=None,
        device="constant_device",
    )
    assert no_process.process_model is None
    vc = no_process.variance_components["response"]
    device = vc[vc.grp.eq("device")]
    assert device.vcov.iloc[0] == 0

    with pytest.raises(EyeProcessValidationError, match="eye_manyfacet_process_irt"):
        ep.facet_effects({})
    with pytest.raises(EyeProcessValidationError, match="No process model"):
        ep.facet_effects(no_process, "process")

    inv = ep.audit_process_measurement_invariance(no_process, "response", relative_sd_threshold=0)
    assert inv.eyeprocess_class == "eye_process_measurement_invariance"


def test_segment_and_changepoint_edge_paths():
    ll, k = pi._segment_loglik(y=np.array([np.nan]), rt=np.array([1.0]), gaze=np.array([np.nan]))
    assert k == 0
    assert ll == 0

    no_split = pi._best_split(np.array([0.0, 1.0, 0.0]), None, None, min_segment=2)
    assert no_split["index"] is None
    assert no_split["delta_sic"] == 0

    with pytest.raises(EyeProcessValidationError, match="min_segment"):
        ep.detect_irt_changepoints(_binary_data(), min_segment=1)

    flat = _binary_data(8)
    flat["participant_id"] = "p1"
    flat["fixation_count"] = 3.0
    cp = ep.detect_irt_changepoints(flat, gaze="fixation_count", min_segment=4)
    assert len(cp.results) == 1

    with pytest.raises(EyeProcessValidationError, match="policy"):
        ep.recalibrate_after_changepoint(flat, lambda x: x, min_segment=4, policy="bad")
    with pytest.raises(EyeProcessValidationError, match="fitter"):
        ep.recalibrate_after_changepoint(flat, "not callable", min_segment=4)

    out = ep.recalibrate_after_changepoint(
        flat,
        lambda x: {"n": len(x)},
        min_segment=4,
        min_delta_sic=-1,
        policy="exclude_post_change",
    )
    assert out.fit["n"] <= len(flat)


def test_censored_normal_validation_sparse_item_and_prediction_edges():
    theta = np.linspace(-1, 1, 8)
    with pytest.raises(EyeProcessValidationError, match="two-dimensional"):
        ep.fit_censored_normal_process_irt(np.array([0.1, 0.2]), theta[:2])
    with pytest.raises(EyeProcessValidationError, match=r"length\(theta\)"):
        ep.fit_censored_normal_process_irt(np.ones((3, 2)) * 0.5, [0, 1])
    with pytest.raises(EyeProcessValidationError, match="lower"):
        ep.fit_censored_normal_process_irt(np.ones((3, 2)) * 0.5, [0, 1, 2], lower=1, upper=0)
    with pytest.raises(EyeProcessValidationError, match="within"):
        ep.fit_censored_normal_process_irt(np.array([[0.2, 1.2], [0.3, 0.4]]), [0, 1])

    sparse = pd.DataFrame({"constant": np.repeat(0.5, 8), "varying": np.linspace(0.2, 0.8, 8)})
    fit = ep.fit_censored_normal_process_irt(sparse, theta)
    assert fit.coefficients.convergence.eq(99).all()

    with pytest.raises(EyeProcessValidationError, match="eye_censored_normal_process_irt"):
        ep.predict_eye_censored_normal_process_irt({})
    pred = ep.predict_eye_censored_normal_process_irt(fit, theta=[0, 1], items=["missing"])
    assert pred.shape == (2, 0)


def test_censored_nll_exercises_left_right_and_middle_observations():
    y = np.array([0.0, 0.5, 1.0])
    theta = np.array([-1.0, 0.0, 1.0])
    value = pi._cn_nll(np.array([0.5, 0.5, math.log(0.2)]), y, theta, 0.0, 1.0)
    assert np.isfinite(value)


def test_process_discrimination_and_ablation_validation_paths():
    bad = _binary_data()
    bad.loc[0, "response"] = 2
    with pytest.raises(EyeProcessValidationError, match="binary"):
        ep.process_dependent_discrimination_audit(
            bad,
            "response",
            "theta",
            "fixation_count",
            "participant_id",
            "item_id",
        )

    d = pd.DataFrame({"y": [1, 2], "a": [2, 3]})
    with pytest.raises(EyeProcessValidationError, match="channels"):
        ep.process_channel_ablation(d, {}, lambda *args: 0)
    with pytest.raises(EyeProcessValidationError, match="evaluator"):
        ep.process_channel_ablation(d, {"a": ["a"]}, "bad")
    out = ep.process_channel_ablation(
        d,
        {"a": ["a"]},
        lambda data, active, name: len(active),
        baseline=["y"],
        higher_is_better=False,
    )
    assert out.information_loss.iloc[0] == -1


def test_generalizability_zero_variance_and_cross_device_sparse_anchor():
    d = pd.DataFrame({"outcome": [1.0] * 4, "facet": ["a"] * 4})
    gs = ep.generalizability_process_study(d, "outcome", ["facet"])
    assert gs.variance_components.proportion.isna().all()

    eq = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0],
            "reference": [1.1, 2.1, 3.1, 4.1],
            "device": ["d1", "d1", "d2", "d2"],
            "anchor": [1, np.nan, 1, np.nan],
        }
    )
    out = ep.cross_device_process_equating_audit(
        eq,
        "value",
        "reference",
        "device",
        anchor="anchor",
    )
    assert out.n.tolist() == [1, 1]
    assert out.A.isna().all()


def test_response_combination_sort_empty_and_pairwise_degenerate_paths():
    d = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "item_id": ["i1"] * 4,
            "option_id": ["B", "A", "B", "A"],
            "selected": [True, True, pd.NA, False],
        }
    )
    unsorted = ep.encode_response_combinations(d, sort_options=False, empty_code="EMPTY")
    assert unsorted.loc[unsorted.participant_id.eq("p1"), "response_combination"].iloc[0] == "B|A"
    assert unsorted.loc[unsorted.participant_id.eq("p2"), "response_combination"].iloc[0] == "EMPTY"

    corr = pi._pairwise_corr(np.array([[1.0, 2.0], [1.0, 3.0]]))
    assert np.isnan(corr[0, 1])


def test_local_dependence_input_guards_ndarray_and_process_shape():
    with pytest.raises(EyeProcessValidationError, match="two residual columns"):
        ep.audit_process_local_dependence(np.array([1.0, 2.0]))

    R = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    audit = ep.audit_process_local_dependence(R)
    assert audit.pairs.response_residual_correlation.isna().all()
    assert np.isnan(audit.max_absolute_response)

    with pytest.raises(EyeProcessValidationError, match="same dimensions"):
        ep.audit_process_local_dependence(np.ones((3, 2)), np.ones((3, 3)))


def test_multiple_response_external_and_constant_gaze_paths():
    d = pd.DataFrame(
        {
            "participant_id": np.repeat(["p1", "p2"], 4),
            "item_id": ["i1", "i1", "i2", "i2"] * 2,
            "option_id": ["A", "B"] * 4,
            "theta": np.repeat([-0.5, 0.5], 4),
            "selected": [0, 1, 1, 0, 1, 0, 0, 1],
            "gaze": 2.0,
        }
    )
    with pytest.raises(EyeProcessBackendError, match="external_engine"):
        ep.fit_multiple_response_process_irt(d, engine="external")
    ext = ep.fit_multiple_response_process_irt(
        d,
        engine="external",
        external_engine=lambda data, **kwargs: {"n": len(data)},
    )
    assert ext.model["n"] == len(d)

    ref = ep.fit_multiple_response_process_irt(d, gaze="gaze")
    assert np.allclose(ref.data[".gaze"], 0)
    assert ".interaction" in ref.data


def test_revisit_process_validation_and_gaze_list_paths():
    R = np.array([[1, 0], [0, 1]], dtype=float)
    Q = np.array([[1], [0]], dtype=int)
    process = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "revisited": [0, 1, 1, 0],
            "response_time": [1.0, 1.2, 1.1, 1.3],
            "g1": [1.0, 2.0, 3.0, 4.0],
            "g2": [2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(EyeProcessValidationError, match="align"):
        ep.fit_revisit_process_cdm(np.ones((2, 3)), Q, process)
    with pytest.raises(EyeProcessValidationError, match="0/1"):
        ep.fit_revisit_process_cdm(R, np.array([[2], [0]]), process)

    fit = ep.fit_revisit_process_cdm(R, Q, process, gaze=["g1", "g2"])
    assert fit.revisit_process_features == ["revisited", "response_time", "g1", "g2"]


def test_plot_type_guards_and_provided_axis_paths():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with pytest.raises(EyeProcessValidationError, match="process_dependent"):
        ep.plot_eye_process_dependent_discrimination({})
    with pytest.raises(EyeProcessValidationError, match="process_g_study"):
        ep.plot_eye_process_g_study({})
    with pytest.raises(EyeProcessValidationError, match="local_dependence"):
        ep.plot_eye_process_local_dependence_audit({})

    fig, ax = plt.subplots()
    ab = ep.process_channel_ablation(
        pd.DataFrame({"y": [1, 2], "a": [2, 3]}),
        {"a": ["a"]},
        lambda data, active, name: len(active),
        baseline=["y"],
    )
    assert ep.plot_eye_process_channel_ablation(ab, ax=ax) is ax
    plt.close(fig)
