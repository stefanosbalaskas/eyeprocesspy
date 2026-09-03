from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.dynamic_irt as dynamic_mod
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessModelError, EyeProcessValidationError


def _transition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I1", "I1", "I1"],
            "trial_id": ["T1", "T1", "T2", "T2"],
            "from_state": ["a", "b", "a", "b"],
            "to_state": ["b", "a", "b", "a"],
            "step": [1, 2, 1, 2],
            "time_gap": [0.2, 0.3, 0.2, 0.3],
            "score": [1, 1, 0, 0],
            "condition": ["x", "y", "x", "y"],
            "numeric_predictor": [0.0, 1.0, 2.0, 3.0],
        }
    )


def _strategy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "score": [0, 1, 1, 0],
            "f1": [-1.0, 1.0, 0.8, -0.7],
            "f2": [0.2, 1.0, 0.9, 0.1],
            "condition": ["A", "A", "B", "B"],
        }
    )


def _strategy_spec(**kwargs):
    return ep.theory_strategy_spec(
        {"analytic": {"f1": 1, "f2": 1}, "heuristic": {"f1": -1, "f2": 0.2}},
        multiple_starts=1,
        **kwargs,
    )


def _diffusion_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "score": [0, 1, 1, 0],
            "response_time": [0.5, 0.7, 0.8, 0.6],
            "gaze_balance": [-1.0, 0.5, 1.0, -0.5],
        }
    )


def test_private_dataframe_and_resource_guards():
    class BadFrame:
        def __iter__(self):
            raise RuntimeError("no frame")

    with pytest.raises(EyeProcessValidationError, match="must be a data frame"):
        dynamic_mod._df(BadFrame())
    with pytest.raises(EyeProcessValidationError, match="missing required columns"):
        dynamic_mod._required(pd.DataFrame({"a": [1]}), ["a", "b"])
    with pytest.raises(EyeProcessBackendError, match="Bundled Stan program"):
        dynamic_mod._stan_path("definitely_missing_eyeprocess_model.stan")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"hidden_states": 1}, "hidden_states"),
        ({"hidden_states": -2}, "hidden_states"),
        ({"engine": "bad"}, "engine"),
        ({"source": "bad"}, "source"),
        ({"person_effect": "bad"}, "person_effect"),
        ({"item_effect": "bad"}, "person_effect"),
        ({"missing_state": "bad"}, "missing_state"),
        ({"ridge": -1}, "ridge"),
        ({"chains": 0}, "controls"),
        ({"chains": 1, "parallel_chains": 2}, "controls"),
        ({"adapt_delta": 0}, "adapt_delta"),
        ({"adapt_delta": 1}, "adapt_delta"),
        ({"condition_columns": [""]}, "Predictor"),
        ({"transition_predictors": [None]}, "Predictor"),
    ],
)
def test_dynamic_spec_validation_guards(kwargs, message):
    with pytest.raises(EyeProcessValidationError, match=message):
        ep.dynamic_irtree_spec(**kwargs)


def test_dynamic_spec_effect_defaults_and_hidden_coercion():
    stan = ep.dynamic_irtree_spec(engine="stan", include_person=True, include_item=True)
    assert stan.person_effect == "random"
    assert stan.item_effect == "random"
    hidden = ep.dynamic_irtree_spec(
        engine="stan", hidden_states=2, include_person=True, include_item=True,
        condition_columns=None, transition_predictors=None, interactions=None,
    )
    assert hidden.person_effect == "fixed"
    assert hidden.item_effect == "fixed"
    assert hidden.condition_columns == []


def test_long_transition_preparation_guards_and_time_path():
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        ep.prepare_dynamic_irtree_data(pd.DataFrame({"participant_id": ["P1"]}))
    bad_person = pd.DataFrame({"participant_id": [None, None], "item_id": ["I1", "I1"], "state": ["a", "b"]})
    with pytest.raises(EyeProcessValidationError, match="identifiers"):
        ep.prepare_dynamic_irtree_data(bad_person)
    one = pd.DataFrame({"participant_id": ["P1"], "item_id": ["I1"], "state": ["a"]})
    with pytest.raises(EyeProcessValidationError, match="No transitions"):
        ep.prepare_dynamic_irtree_data(one)
    long = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P1"],
            "item_id": ["I1"] * 3,
            "trial_id": ["T1"] * 3,
            "state": ["a", "b", "a"],
            "timestamp": [0.0, 0.25, 1.0],
        }
    )
    out = ep.prepare_dynamic_irtree_data(long, time="timestamp")
    np.testing.assert_allclose(out["time_gap"], [0.25, 0.75])


def test_transition_preparation_argument_missing_state_and_probability_guards():
    d = _transition_frame()
    with pytest.raises(EyeProcessValidationError, match="Unexpected arguments"):
        ep.prepare_dynamic_irtree_data(d, unexpected=True)
    with pytest.raises(EyeProcessValidationError, match="spec must"):
        ep.prepare_dynamic_irtree_data(d, dynamic_mod._result("wrong"))

    missing = d.copy()
    missing.loc[0, "to_state"] = None
    dropped = ep.prepare_dynamic_irtree_data(missing, ep.dynamic_irtree_spec(missing_state="drop"))
    assert len(dropped) == len(d) - 1
    with pytest.raises(EyeProcessValidationError, match="requires the hidden-state"):
        ep.prepare_dynamic_irtree_data(missing, ep.dynamic_irtree_spec(missing_state="marginalize"))

    with pytest.raises(EyeProcessValidationError, match="Unknown state labels"):
        ep.prepare_dynamic_irtree_data(d, states=["a", "c"])
    one_state = d.assign(from_state="a", to_state="a")
    with pytest.raises(EyeProcessValidationError, match="At least two states"):
        ep.prepare_dynamic_irtree_data(one_state)

    prob_spec = ep.dynamic_irtree_spec(uncertain_state_probability="state_prob")
    with pytest.raises(EyeProcessValidationError, match="probability column is absent"):
        ep.prepare_dynamic_irtree_data(d, prob_spec)
    bad_prob = d.assign(state_prob=[1.0, 0.0, 0.5, 1.0])
    with pytest.raises(EyeProcessValidationError, match="probabilities"):
        ep.prepare_dynamic_irtree_data(bad_prob, prob_spec)
    good_prob = d.assign(state_prob=[1.0, 0.8, 0.5, 1.0])
    prepared = ep.prepare_dynamic_irtree_data(good_prob, prob_spec)
    np.testing.assert_allclose(prepared["state_probability"], good_prob["state_prob"])


def test_transition_mask_allowed_forbidden_and_guard_paths():
    with pytest.raises(EyeProcessValidationError, match="At least two states"):
        ep.structural_transition_mask(["a"])
    with pytest.raises(EyeProcessValidationError, match="from and to"):
        ep.structural_transition_mask(["a", "b"], forbidden=pd.DataFrame({"from": ["a"]}))
    with pytest.raises(EyeProcessValidationError, match="Unknown state"):
        ep.structural_transition_mask(["a", "b"], forbidden=pd.DataFrame({"from": ["a"], "to": ["z"]}))
    with pytest.raises(EyeProcessValidationError, match="at least one destination"):
        ep.structural_transition_mask(
            ["a", "b"],
            allowed=pd.DataFrame({"from": ["b"], "to": ["a"]}),
        )
    mask = ep.structural_transition_mask(["a", "b"], allow_self=False)
    assert not bool(mask.loc["a", "a"])
    assert bool(mask.loc["a", "b"])


def test_dynamic_design_predictor_fixed_effect_and_standardization_paths():
    d = _transition_frame()
    spec = ep.dynamic_irtree_spec(
        engine="multinomial", person_effect="fixed", item_effect="fixed",
        condition_columns=["condition"], transition_predictors=["numeric_predictor"],
        include_time_gap=False, standardize=True,
    )
    design = ep.dynamic_transition_design(d, spec, formula="test")
    assert design.formula == "test"
    assert any(c.startswith("condition_") for c in design.X.columns)
    assert "numeric_predictor" in design.X.columns
    assert any(c.startswith("participant_id_") for c in design.X.columns)
    assert not design.scaling.empty


def test_multinomial_and_decoding_validation_paths():
    with pytest.raises(EyeProcessValidationError, match="eye_transition_design"):
        ep.fit_multinomial_transition(dynamic_mod._result("wrong"))
    design = ep.dynamic_transition_design(_transition_frame(), ep.dynamic_irtree_spec(engine="multinomial"))
    with pytest.raises(EyeProcessValidationError, match="Unknown reference"):
        ep.fit_multinomial_transition(design, reference_state="z")
    fit = ep.fit_dynamic_irtree(_transition_frame(), ep.dynamic_irtree_spec(engine="multinomial"), min_transitions=1)
    with pytest.raises(EyeProcessValidationError, match="method"):
        ep.decode_dynamic_states(fit, "bad")
    draw = ep.decode_dynamic_states(fit, "draw")
    assert len(draw) == len(fit.transitions)
    with pytest.raises(EyeProcessModelError, match="unavailable"):
        ep.decode_dynamic_states(dynamic_mod._result("eye_dynamic_irtree", model=dynamic_mod._result("other")))


@pytest.mark.parametrize("kind", ["deviance", "randomized"])
def test_transition_residual_alternate_types(kind):
    fit = ep.fit_dynamic_irtree(_transition_frame(), ep.dynamic_irtree_spec(engine="multinomial"), min_transitions=1)
    diag = ep.transition_residual_diagnostics(fit, type=kind)
    assert len(diag.residuals) == len(fit.transitions)


def test_transition_residual_and_comparison_guards():
    with pytest.raises(EyeProcessValidationError, match="type"):
        ep.transition_residual_diagnostics(dynamic_mod._result("eye_dynamic_irtree"), type="bad")
    with pytest.raises(EyeProcessValidationError, match="Expected"):
        ep.transition_residual_diagnostics(dynamic_mod._result("wrong"))
    with pytest.raises(EyeProcessValidationError, match="At least one"):
        ep.compare_dynamic_transition_models()
    with pytest.raises(EyeProcessValidationError, match="At least one"):
        ep.compare_dynamic_transition_models(dynamic_mod._result("wrong"))


def test_dynamic_simulation_validation_and_regular_time():
    with pytest.raises(EyeProcessValidationError, match="too small"):
        ep.simulate_dynamic_irtree_data(n_person=1, n_item=2)
    with pytest.raises(EyeProcessValidationError, match="rates"):
        ep.simulate_dynamic_irtree_data(n_person=2, n_item=2, state_misclassification=1)
    with pytest.raises(EyeProcessValidationError, match="heterogeneity"):
        ep.simulate_dynamic_irtree_data(n_person=2, n_item=2, person_sd=-1)
    sim = ep.simulate_dynamic_irtree_data(n_person=2, n_item=2, transitions_per_trial=2, irregular_time=False, seed=3)
    assert set(sim.transitions["time_gap"]) == {1.0}
    default_recovery = ep.dynamic_irtree_recovery(replications=1, base_seed=2)
    assert len(default_recovery.plan.jobs) == 6


def test_dynamic_baseline_engine_and_no_support_guard():
    d = _transition_frame()
    fit = ep.fit_dynamic_irtree(d, ep.dynamic_irtree_spec(engine="baseline"), min_transitions=1)
    assert fit.model.eyeprocess_class == "eye_multinomial_transition"
    assert fit.fits
    with pytest.raises(EyeProcessModelError, match="sufficient transition support"):
        ep.fit_dynamic_irtree(d, ep.dynamic_irtree_spec(engine="baseline"), min_transitions=100)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"engine": "bad"}, "engine"),
        ({"strategies": {"only": {"f1": 1}}}, "strategies"),
        ({"strategies": {"a": {"f1": 1}, "b": {"f2": 1}}, "feature_columns": ["f1"]}, "Unknown signature"),
        ({"strategies": {"a": {"f1": 1}, "b": {"f1": -1}}, "multiple_starts": 0}, "multiple_starts"),
        ({"strategies": {"a": {"f1": 1}, "b": {"f1": -1}}, "anchor_strength": -1}, "anchor_strength"),
    ],
)
def test_strategy_spec_guard_paths(kwargs, message):
    with pytest.raises(EyeProcessValidationError, match=message):
        ep.theory_strategy_spec(**kwargs)


def test_strategy_legacy_matrix_paths_and_prepare_guards():
    matrix = pd.DataFrame([[1.0, 0.0], [-1.0, 0.5]], index=["a", "b"], columns=["f1", "f2"])
    spec = ep.theory_strategy_spec(matrix, multiple_starts=1)
    assert spec.legacy_mode is True
    bad = _strategy_frame().drop(columns="f2")
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        ep.prepare_strategy_mixture_data(bad, _strategy_spec())
    nonbinary = _strategy_frame().assign(score=[0, 1, 2, 0])
    with pytest.raises(EyeProcessValidationError, match="coded 0/1"):
        ep.prepare_strategy_mixture_data(nonbinary, _strategy_spec())
    availability = pd.DataFrame(
        {
            "item_id": ["I1", "I1"],
            "strategy": ["analytic", "heuristic"],
            "available": [False, False],
        }
    )
    with pytest.raises(EyeProcessValidationError, match="at least one strategy"):
        ep.prepare_strategy_mixture_data(_strategy_frame(), _strategy_spec(item_availability=availability))


def test_strategy_fit_probability_and_sensitivity_guards():
    with pytest.raises(EyeProcessValidationError, match="prepared strategy"):
        ep.fit_strategy_mixture_em(dynamic_mod._result("wrong"))
    with pytest.raises(EyeProcessValidationError, match="eye_theory_strategy"):
        ep.strategy_posterior_probabilities(dynamic_mod._result("wrong"))
    fit = ep.fit_theory_strategy_irt(_strategy_frame(), _strategy_spec(), max_iter=5, seed=1)
    with pytest.raises(EyeProcessValidationError, match="threshold"):
        ep.strategy_classification_uncertainty(fit, threshold=1.1)
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        ep.strategy_aoi_sensitivity({}, _strategy_spec())
    with pytest.raises(EyeProcessValidationError, match="Condition column"):
        ep.validate_strategy_manipulation(fit, "missing", "analytic")
    with pytest.raises(EyeProcessValidationError, match="Unknown strategy"):
        ep.validate_strategy_manipulation(fit, "condition", "missing")


@pytest.mark.parametrize("engine", ["ez_regression", "diffIRT", "brms"])
def test_diffusion_legacy_engines_are_explicitly_rejected(engine):
    spec = ep.gaze_diffusion_spec(drift_features=["gaze_balance"], engine=engine)
    with pytest.raises(EyeProcessBackendError, match="not silently substituted"):
        ep.fit_gaze_diffusion_irt(_diffusion_frame(), spec)


def test_diffusion_spec_and_data_validation_paths():
    with pytest.raises(EyeProcessValidationError, match="Unknown diffusion"):
        ep.gaze_diffusion_spec(engine="bad")
    with pytest.raises(EyeProcessValidationError, match="only one"):
        ep.gaze_diffusion_spec(drift_features=["x"], boundary_features=["x"])
    alias = ep.gaze_diffusion_spec(gaze_features=["gaze_balance"])
    assert alias.drift_features == ["gaze_balance"]

    spec = ep.gaze_diffusion_spec(drift_features=["gaze_balance"])
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        ep.prepare_gaze_diffusion_data(_diffusion_frame().drop(columns="score"), spec)
    too_fast = _diffusion_frame().assign(response_time=[0.01, 0.7, 0.8, 0.6])
    with pytest.raises(EyeProcessValidationError, match="greater than minimum_rt"):
        ep.prepare_gaze_diffusion_data(too_fast, spec)
    millis = _diffusion_frame().assign(response_time=[500, 700, 800, 600])
    with pytest.raises(EyeProcessValidationError, match="milliseconds"):
        ep.prepare_gaze_diffusion_data(millis, spec)
    nonbinary = _diffusion_frame().assign(score=[0, 1, 2, 0])
    with pytest.raises(EyeProcessValidationError, match="coded 0/1"):
        ep.prepare_gaze_diffusion_data(nonbinary, spec)

    censor_spec = ep.gaze_diffusion_spec(drift_features=["gaze_balance"], censor_column="censor")
    bad_censor = _diffusion_frame().assign(censor=["observed", "right", "bad", "left"])
    with pytest.raises(EyeProcessValidationError, match="Censoring values"):
        ep.prepare_gaze_diffusion_data(bad_censor, censor_spec)
    good = _diffusion_frame().assign(censor=["observed", "right", "left", "observed"])
    prepared = ep.prepare_gaze_diffusion_data(good, censor_spec)
    assert prepared.censor.tolist() == [0, 1, 2, 0]


def test_diffusion_diagnostics_comparison_and_simulation_guards():
    fit = ep.fit_gaze_diffusion_irt(
        _diffusion_frame(), ep.gaze_diffusion_spec(drift_features=["gaze_balance"], engine="baseline")
    )
    assert ep.diffusion_parameter_diagnostics(fit).engine == "baseline"
    assert ep.diffusion_posterior_predictive(fit).method == "baseline descriptive"
    with pytest.raises(EyeProcessValidationError, match="gaze-diffusion fit"):
        ep.compare_diffusion_accuracy_rt(dynamic_mod._result("wrong"))
    with pytest.raises(EyeProcessValidationError, match="at least two persons"):
        ep.simulate_gaze_diffusion_data(n_person=1, n_item=2)
    with pytest.raises(EyeProcessValidationError, match="contaminant_fraction"):
        ep.simulate_gaze_diffusion_data(n_person=2, n_item=2, contaminant_fraction=1)
    sim = ep.simulate_gaze_diffusion_data(n_person=2, n_item=2, time_step=0.01, max_decision_time=0.05, seed=2)
    assert len(sim) == 4
    default_plan = ep.diffusion_identification_study(replications=1)
    assert len(default_plan.plan.jobs) == 16
    frame_plan = ep.diffusion_identification_study(
        pd.DataFrame({"n_person": [2], "n_item": [2], "gaze_effect": [0.1], "contaminant_fraction": [0.0]}),
        replications=2,
    )
    assert len(frame_plan.plan.jobs) == 2
