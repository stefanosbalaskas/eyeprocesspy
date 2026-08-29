import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


def test_dynamic_irtree_simulation_and_design_preserve_transitions():
    sim = ep.simulate_dynamic_irtree_data(n_person=12, n_item=6, transitions_per_trial=8, seed=11)
    assert {"participant_id", "item_id", "trial_id", "to_state", "time"}.issubset(sim.transitions.columns)
    spec = ep.dynamic_irtree_spec(engine="multinomial", condition_columns=None)
    prepared = ep.prepare_dynamic_irtree_data(sim.transitions, spec)
    design = ep.dynamic_transition_design(prepared, spec)
    assert design.eyeprocess_class == "eye_transition_design"
    assert design.X.shape[0] == len(prepared)
    assert design.allowed.shape[1] == len(design.states)


def test_structural_zeros_are_enforced():
    states = ["prompt", "option", "submit"]
    mask = ep.structural_transition_mask(states, structural_zeros=pd.DataFrame({"from": ["submit"], "to": ["prompt"]}))
    assert not bool(mask.loc["submit", "prompt"])
    assert bool(mask.loc["prompt", "option"])


def test_multinomial_transition_returns_probabilities_and_diagnostics():
    sim = ep.simulate_dynamic_irtree_data(n_person=15, n_item=5, transitions_per_trial=7, seed=4)
    fit = ep.fit_dynamic_irtree(sim.transitions, ep.dynamic_irtree_spec(engine="multinomial"), min_transitions=2)
    assert fit.eyeprocess_class == "eye_dynamic_irtree"
    probability = ep.decode_dynamic_states(fit, "probability")
    assert len(probability) == len(fit.transitions)
    pcols = [c for c in probability if c != "transition"]
    np.testing.assert_allclose(probability[pcols].sum(axis=1), 1, atol=1e-6)
    diagnostic = ep.transition_residual_diagnostics(fit)
    assert diagnostic.eyeprocess_class == "eye_transition_diagnostics"
    comparison = ep.compare_dynamic_transition_models({"multinomial": fit})
    assert {"model", "engine", "AIC", "BIC"}.issubset(comparison.columns)


def test_strategy_signatures_normalized_and_zero_rejected():
    spec = ep.theory_strategy_spec(
        {"analytic": {"prompt": 1, "evidence": 2}, "heuristic": {"prompt": -1, "evidence": .2}},
        engine="em", multiple_starts=2,
    )
    assert spec.eyeprocess_class == "eye_theory_strategy_spec"
    np.testing.assert_allclose(np.sqrt((spec.signatures.to_numpy() ** 2).sum(axis=1)), [1, 1], atol=1e-8)
    with pytest.raises(Exception, match="non-zero"):
        ep.theory_strategy_spec({"a": {"x": 0}, "b": {"x": 1}})


def test_strategy_em_returns_anchored_probabilities():
    signatures = pd.DataFrame([[1, 1], [-1, .2]], index=["analytic", "heuristic"], columns=["prompt", "evidence"])
    sim = ep.simulate_strategy_mixture_data(10, 4, signatures, seed=12)
    spec = ep.theory_strategy_spec(
        {"analytic": {"prompt": 1, "evidence": 1}, "heuristic": {"prompt": -1, "evidence": .2}},
        engine="em", multiple_starts=2,
    )
    fit = ep.fit_theory_strategy_irt(sim, spec, seed=5, max_iter=30)
    assert fit.eyeprocess_class == "eye_theory_strategy_irt"
    probability = ep.strategy_posterior_probabilities(fit)
    assert len(probability) == len(sim)
    np.testing.assert_allclose(probability[spec.strategies].sum(axis=1), 1, atol=1e-6)
    uncertainty = ep.strategy_classification_uncertainty(fit)
    assert float(uncertainty.summary.iloc[0].uncertain_fraction) >= 0


def test_gaze_diffusion_data_and_baseline_fit():
    sim = ep.simulate_gaze_diffusion_data(8, 4, seed=9, time_step=.01, max_decision_time=2)
    spec = ep.gaze_diffusion_spec(drift_features=["gaze_balance"], engine="baseline")
    prepared = ep.prepare_gaze_diffusion_data(sim, spec)
    assert prepared.eyeprocess_class == "eye_gaze_diffusion_data"
    assert len(prepared.y) == len(sim)
    with pytest.raises(Exception, match="only one"):
        ep.gaze_diffusion_spec(drift_features=["x"], boundary_features=["x"])

    sim2 = ep.simulate_gaze_diffusion_data(10, 5, seed=2, time_step=.01, max_decision_time=2)
    fit = ep.fit_gaze_diffusion_irt(sim2, ep.gaze_diffusion_spec(drift_features=["gaze_balance"], engine="baseline"))
    assert fit.eyeprocess_class == "eye_gaze_diffusion_irt"
    parameters = ep.extract_diffusion_parameters(fit)
    assert {"component", "term", "estimate"}.issubset(parameters.columns)
    diagnostic = ep.diffusion_parameter_diagnostics(fit)
    assert diagnostic.engine == "baseline"


def test_advanced_stan_resources_present_and_syntax_guarded():
    from importlib import resources
    for name in ["dynamic_irtree_observed.stan", "dynamic_irtree_hidden.stan", "theory_strategy_mixture.stan", "gaze_diffusion_irt.stan"]:
        assert resources.files("eyeprocesspy").joinpath("resources", "stan", name).is_file()
    code = resources.files("eyeprocesspy").joinpath("resources", "stan", "gaze_diffusion_irt.stan").read_text()
    assert "wiener_lcdf_unnorm(rt, boundary, nondecision, starting, drift)" in code
    assert "wiener_lccdf_unnorm(rt, boundary, nondecision, starting, drift)" in code
    assert "fabs(" not in code
    assert "fmax(abs(drift), 0.25)" in code
