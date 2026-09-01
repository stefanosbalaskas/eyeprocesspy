from __future__ import annotations

import builtins
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.dynamic_irt as dm
from eyeprocesspy.exceptions import (
    EyeProcessBackendError,
    EyeProcessModelError,
    EyeProcessValidationError,
)


def _transitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I1", "I1", "I1"],
            "trial_id": ["T1", "T1", "T2", "T2"],
            "from_state": ["a", "b", "a", "b"],
            "to_state": ["b", "a", "b", "a"],
            "step": [1, 2, 1, 2],
            "time_gap": [0.2, 0.3, 0.4, 0.5],
            "score": [1.0, np.nan, 0.0, 1.0],
            "condition": ["x", "y", "x", "y"],
            "numeric": [0.0, 1.0, 2.0, 3.0],
        }
    )


def _strategy_data(condition: bool = False) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "score": [0, 1, 1, 0],
            "f1": [-1.0, 1.0, 0.8, -0.7],
            "f2": [0.2, 1.0, 0.9, 0.1],
        }
    )
    if condition:
        data["condition"] = ["one"] * len(data)
    return data


def _strategy_spec(**kwargs):
    return ep.theory_strategy_spec(
        {"analytic": {"f1": 1.0, "f2": 1.0}, "heuristic": {"f1": -1.0, "f2": 0.2}},
        multiple_starts=1,
        **kwargs,
    )


class _FakeFit:
    def __init__(self, summary=None, draws=None):
        self._summary = pd.DataFrame({"Mean": [0.0]}, index=["beta[1]"]) if summary is None else summary
        self._draws = draws

    def summary(self):
        return self._summary.copy()

    def draws_pd(self, vars=None):
        if isinstance(self._draws, Exception):
            raise self._draws
        if self._draws is not None:
            return self._draws.copy()
        variable = vars[0]
        return pd.DataFrame(
            {
                f"{variable}[1]": [1, 1, 2, 1],
                f"{variable}[2]": [2, 2, 1, 2],
                f"{variable}[3]": [1, 2, 1, 1],
                f"{variable}[4]": [2, 1, 2, 2],
            }
        )


def _fake_cmdstan(holder, *, version=(2, 38), fit=None, fail=False):
    class Model:
        def __init__(self, stan_file):
            holder["stan_file"] = stan_file
            if fail == "construct":
                raise RuntimeError("construct failed")

        def sample(self, data, **kwargs):
            holder["data"] = data
            holder["sample_kwargs"] = kwargs
            if fail == "sample":
                raise RuntimeError("sample failed")
            return _FakeFit() if fit is None else fit

    def cmdstan_version():
        if isinstance(version, Exception):
            raise version
        return version

    return SimpleNamespace(CmdStanModel=Model, cmdstan_version=cmdstan_version)


def test_low_level_import_long_and_direct_transition_residuals(monkeypatch):
    p = dm._softmax(np.array([[0.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "cmdstanpy":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(EyeProcessBackendError, match="stan.*extra"):
        dm._cmdstanpy()

    long = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2"],
            "item_id": ["I1", "I1", "I2"],
            "state": ["a", "b", "a"],
        }
    )
    auto = dm._long_to_transitions(
        long, "participant_id", "item_id", "trial_id", "state", None
    )
    assert auto["trial_id"].iloc[0] == "P1::I1"
    assert auto["time_gap"].iloc[0] == 1.0

    bad_trial = pd.DataFrame(
        {
            "participant_id": ["P1", "P1"],
            "item_id": ["I1", "I1"],
            "trial_id": [None, None],
            "state": ["a", "b"],
        }
    )
    with pytest.raises(EyeProcessValidationError, match="Trial identifiers"):
        dm._long_to_transitions(
            bad_trial, "participant_id", "item_id", "trial_id", "state", None
        )

    direct = pd.DataFrame(
        {"from_state": ["a", "b"], "to_state": ["b", "a"]}
    )
    prepared = ep.prepare_dynamic_irtree_data(direct)
    assert prepared["participant_id"].eq("P1").all()
    assert prepared["item_id"].eq("I1").all()
    assert prepared["trial_id"].eq("P1::I1").all()
    assert prepared["step"].tolist() == [1, 2]
    assert prepared["time_gap"].eq(1.0).all()

    blank = direct.assign(
        participant_id=["", ""], item_id=["I1", "I1"], trial_id=["T", "T"]
    )
    with pytest.raises(EyeProcessValidationError, match="non-missing and non-empty"):
        ep.prepare_dynamic_irtree_data(blank)


def test_missing_unknown_marginalize_gap_and_transition_aliases():
    data = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P1"],
            "item_id": ["I1"] * 3,
            "trial_id": ["T1"] * 3,
            "from_state": ["a", "a", "b"],
            "to_state": [None, "b", "a"],
            "time_gap": [np.nan, -2.0, np.inf],
        }
    )
    unknown = ep.prepare_dynamic_irtree_data(
        data, ep.dynamic_irtree_spec(missing_state="unknown")
    )
    assert "<UNKNOWN>" in unknown.attrs["states"]
    assert unknown["time_gap"].eq(1.0).all()
    assert unknown["state_probability"].eq(1.0).all()

    marginal = ep.prepare_dynamic_irtree_data(
        data,
        ep.dynamic_irtree_spec(
            engine="stan", hidden_states=2, missing_state="marginalize"
        ),
    )
    unknown_rows = marginal["to_state"].astype(str).eq("<UNKNOWN>")
    assert unknown_rows.any()
    assert np.all(
        marginal.loc[unknown_rows, "state_probability"].to_numpy(float) < 1.0
    )

    aliases = ep.structural_transition_mask(
        ["a", "b"],
        structural_zeros=pd.DataFrame({"from": ["a"], "to": ["a"]}),
    )
    assert not bool(aliases.loc["a", "a"])

    allowed = ep.structural_transition_mask(
        ["a", "b"],
        allowed_transitions=pd.DataFrame(
            {"from": ["a", "b"], "to": ["b", "a"]}
        ),
    )
    assert bool(allowed.loc["a", "b"])
    assert not bool(allowed.loc["a", "a"])


def test_dynamic_design_hidden_score_drop_and_missing_predictor():
    spec = ep.dynamic_irtree_spec(
        engine="stan",
        hidden_states=2,
        condition_columns=["condition", "missing_column"],
        transition_predictors=["numeric"],
        include_person=True,
        include_item=True,
    )
    design = ep.dynamic_transition_design(_transitions(), spec)
    assert not any(str(c).startswith("from_state_") for c in design.X.columns)
    assert "score" in design.X.columns
    assert "numeric" in design.X.columns
    assert len(design.data) == 3
    assert "missing_column" not in design.X.columns
    assert design.spec.person_effect == "fixed"
    assert design.spec.item_effect == "fixed"


def test_multinomial_structural_control_nonfinite_and_hessian_fallback(monkeypatch):
    design = ep.dynamic_transition_design(
        _transitions().dropna(subset=["score"]),
        ep.dynamic_irtree_spec(engine="multinomial", standardize=False),
    )
    bad = dm.EyeResult(dict(design), eyeprocess_class="eye_transition_design")
    bad_allowed = np.asarray(design.allowed, bool).copy()
    bad_allowed[0, int(design.y[0]) - 1] = False
    bad["allowed"] = bad_allowed
    with pytest.raises(EyeProcessValidationError, match="structural-transition"):
        ep.fit_multinomial_transition(bad)

    nan_design = dm.EyeResult(dict(design), eyeprocess_class="eye_transition_design")
    X = design.X.copy()
    X.iloc[0, 0] = np.nan
    nan_design["X"] = X

    class BadHessian:
        def __array__(self, *args, **kwargs):
            raise TypeError("no array")

    def fake_minimize(fun, x0, method, options=None):
        assert method == "BFGS"
        assert options["maxiter"] == 7
        assert options["gtol"] == pytest.approx(1e-5)
        value = fun(np.asarray(x0, dtype=float))
        assert np.isfinite(value)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            fun=value,
            hess_inv=BadHessian(),
            success=False,
            message="forced",
            nfev=1,
        )

    monkeypatch.setattr(dm, "minimize", fake_minimize)
    fit = ep.fit_multinomial_transition(
        nan_design, control={"maxit": 7, "reltol": 1e-5}
    )
    assert fit.convergence == 1
    assert fit.standard_error_matrix.isna().to_numpy().any()


def test_dynamic_stan_observed_hidden_invalid_and_backend_failure(monkeypatch):
    holder = {}
    monkeypatch.setattr(dm, "_cmdstanpy", lambda: _fake_cmdstan(holder))
    observed_spec = ep.dynamic_irtree_spec(engine="stan", include_person=True)
    observed_design = ep.dynamic_transition_design(
        _transitions().dropna(subset=["score"]), observed_spec
    )
    observed = ep.fit_dynamic_irtree_stan(
        observed_design, observed_spec, seed=9, refresh=1
    )
    assert observed.hidden is False
    assert holder["data"]["N"] == len(observed_design.data)
    assert holder["sample_kwargs"]["seed"] == 9

    hidden_spec = ep.dynamic_irtree_spec(
        engine="stan", hidden_states=2, missing_state="unknown"
    )
    hidden_design = ep.dynamic_transition_design(
        _transitions().dropna(subset=["score"]), hidden_spec
    )
    hidden = ep.fit_dynamic_irtree_stan(hidden_design, hidden_spec)
    assert hidden.hidden is True
    assert holder["data"]["H"] == 2
    assert holder["data"]["emission_prior"].shape == (
        2,
        len(hidden_design.states),
    )

    bad_spec = ep.dynamic_irtree_spec(
        engine="stan",
        hidden_states=2,
        misclassification_matrix=np.ones((1, 1)),
    )
    with pytest.raises(EyeProcessValidationError, match="Misclassification matrix"):
        ep.fit_dynamic_irtree_stan(hidden_design, bad_spec)

    monkeypatch.setattr(
        dm, "_cmdstanpy", lambda: _fake_cmdstan({}, fail="sample")
    )
    with pytest.raises(EyeProcessBackendError, match="dynamic IRTree fitting failed"):
        ep.fit_dynamic_irtree_stan(observed_design, observed_spec)


def test_dynamic_ppc_success_subsample_hidden_and_failure():
    frame = pd.DataFrame(
        {
            "y_rep[1]": [1, 1, 2, 1],
            "y_rep[2]": [2, 2, 1, 2],
            "y_rep[3]": [1, 2, 1, 1],
        }
    )
    stan_model = dm._result(
        "eye_dynamic_irtree_stan", hidden=False, fit=_FakeFit(draws=frame)
    )
    obj = dm._result(
        "eye_dynamic_irtree",
        model=stan_model,
        design=dm._result("eye_transition_design", y=np.array([1, 2, 1])),
        states=["a", "b"],
    )
    ppc = ep.dynamic_posterior_predictive_check(obj, draws=2, seed=2)
    assert ppc.replicated_frequency.shape == (2, 2)
    assert ppc.hidden is False

    hidden_frame = frame.rename(columns=lambda c: c.replace("y_rep", "observed_rep"))
    hidden_model = dm._result(
        "eye_dynamic_irtree_stan",
        hidden=True,
        fit=_FakeFit(draws=hidden_frame),
    )
    hidden_obj = dm._result(
        "eye_dynamic_irtree",
        model=hidden_model,
        design=obj.design,
        states=obj.states,
    )
    assert ep.dynamic_posterior_predictive_check(hidden_obj).hidden is True

    broken = dm._result(
        "eye_dynamic_irtree_stan",
        hidden=False,
        fit=_FakeFit(draws=RuntimeError("missing")),
    )
    with pytest.raises(EyeProcessModelError, match="draws are unavailable"):
        ep.dynamic_posterior_predictive_check(
            dm._result(
                "eye_dynamic_irtree",
                model=broken,
                design=obj.design,
                states=obj.states,
            )
        )


def test_dynamic_stan_residual_guard_comparison_recovery_and_stan_dispatch(monkeypatch):
    real = ep.fit_dynamic_irtree(
        _transitions().dropna(subset=["score"]),
        ep.dynamic_irtree_spec(engine="multinomial"),
        min_transitions=1,
    )
    fake_stan = dm._result(
        "eye_dynamic_irtree",
        spec=ep.dynamic_irtree_spec(engine="stan"),
        transitions=real.transitions,
        model=dm._result("eye_dynamic_irtree_stan"),
    )
    with pytest.raises(EyeProcessModelError, match="Residual diagnostics"):
        ep.transition_residual_diagnostics(fake_stan)

    compared = ep.compare_dynamic_transition_models(
        {"first": real, "second": real}, criterion="accuracy"
    )
    assert compared["rank"].notna().all()

    mixed = ep.compare_dynamic_transition_models(real, stan=fake_stan, criterion="bad")
    assert len(mixed) == 2

    custom_spec = ep.dynamic_irtree_spec(engine="stan")
    recovery = ep.dynamic_irtree_recovery(
        pd.DataFrame(
            {"state_misclassification": [0.1], "missing_state": [0.2]}
        ),
        replications=2,
        spec=custom_spec,
        base_seed=10,
    )
    assert recovery.plan.jobs["seed"].tolist() == [10, 11]
    assert recovery.spec is custom_spec

    stub_model = dm._result(
        "eye_dynamic_irtree_stan",
        summary=pd.DataFrame({"Mean": [0.0]}),
        diagnostics=pd.DataFrame({"ok": [True]}),
    )
    monkeypatch.setattr(dm, "fit_dynamic_irtree_stan", lambda *a, **k: stub_model)
    dispatched = ep.fit_dynamic_irtree(
        _transitions().dropna(subset=["score"]),
        ep.dynamic_irtree_spec(engine="stan"),
        min_transitions=1,
    )
    assert dispatched.model is stub_model
    assert dispatched.fits == {}


def test_dynamic_simulation_misclassification_missing_and_structural_zero():
    sim = ep.simulate_dynamic_irtree_data(
        n_person=2,
        n_item=2,
        transitions_per_trial=3,
        states=["a", "b", "c"],
        state_misclassification=0.95,
        missing_state=0.95,
        structural_zeros=pd.DataFrame({"from": ["a"], "to": ["a"]}),
        seed=11,
    )
    assert len(sim.transitions) == 12
    assert sim.transitions["to_state"].isna().any()
    assert (
        sim.transitions["true_to_state"].astype(str)
        != sim.transitions["to_state"].astype(str)
    ).any()


def test_strategy_spec_legacy_alias_sequence_and_zero_anchor_guards():
    alias = ep.theory_strategy_spec(
        strategies=None,
        prototypes=np.array([[1.0, 0.0], [-1.0, 1.0]]),
        feature_columns=["f1", "f2"],
        feature_sd=[2.0, 3.0],
        prior=[0.25, 0.75],
        multiple_starts=1,
    )
    np.testing.assert_allclose(alias.feature_sd, [2.0, 3.0])
    np.testing.assert_allclose(alias.prior, [0.25, 0.75])

    named = ep.theory_strategy_spec(
        pd.DataFrame(
            [[1.0, 0.0], [-1.0, 1.0]],
            index=["left", "right"],
            columns=["f1", "f2"],
        ),
        multiple_starts=1,
    )
    assert named.strategies == ["left", "right"]

    sequence = ep.theory_strategy_spec(
        {"a": [1.0, 0.0], "b": [-1.0, 1.0]},
        feature_columns=["f1", "f2"],
        multiple_starts=1,
    )
    assert sequence.prototypes.shape == (2, 2)

    with pytest.raises(EyeProcessValidationError, match="prototypes"):
        ep.theory_strategy_spec(np.array([1.0, 2.0]), multiple_starts=1)
    with pytest.raises(EyeProcessValidationError, match="named numeric vector"):
        ep.theory_strategy_spec({"a": [1.0], "b": [-1.0]}, multiple_starts=1)
    with pytest.raises(EyeProcessValidationError, match="unique and non-empty"):
        ep.theory_strategy_spec(
            {"": {"f": 1.0}, "b": {"f": -1.0}}, multiple_starts=1
        )
    with pytest.raises(EyeProcessValidationError, match="non-zero anchor"):
        ep.theory_strategy_spec(
            {"a": {"f": 0.0}, "b": {"f": 1.0}}, multiple_starts=1
        )


def test_strategy_prepare_condition_nonstandard_empty_and_availability_skip():
    spec = _strategy_spec(
        condition="condition",
        item_availability=pd.DataFrame({"unexpected": [1]}),
    )
    prepared = ep.prepare_strategy_mixture_data(
        _strategy_data(condition=True), spec, standardize=False
    )
    assert prepared.condition_levels == ["one"]
    np.testing.assert_allclose(prepared.center, [0.0, 0.0])
    np.testing.assert_allclose(prepared.scale, [1.0, 1.0])

    empty = _strategy_data().assign(f1=np.nan)
    with pytest.raises(EyeProcessValidationError, match="No complete"):
        ep.prepare_strategy_mixture_data(empty, _strategy_spec())


def test_strategy_em_zero_weight_stan_success_failure_and_override_dispatch(monkeypatch):
    availability = pd.DataFrame(
        {
            "item_id": ["I1", "I2"],
            "strategy": ["heuristic", "heuristic"],
            "available": [False, False],
        }
    )
    spec = _strategy_spec(item_availability=availability)
    prepared = ep.prepare_strategy_mixture_data(_strategy_data(), spec)
    em = ep.fit_strategy_mixture_em(prepared, starts=1, max_iter=2, seed=2)
    assert em.mixing[1] == 0.0

    holder = {}
    monkeypatch.setattr(dm, "_cmdstanpy", lambda: _fake_cmdstan(holder))
    stan_spec = _strategy_spec(engine="stan")
    stan_prepared = ep.prepare_strategy_mixture_data(_strategy_data(), stan_spec)
    stan_fit = ep.fit_strategy_mixture_stan(stan_prepared, seed=3)
    assert stan_fit.eyeprocess_class == "eye_strategy_mixture_stan"
    assert holder["data"]["N"] == 4

    monkeypatch.setattr(dm, "_cmdstanpy", lambda: _fake_cmdstan({}, fail="sample"))
    with pytest.raises(EyeProcessBackendError, match="strategy-mixture fitting failed"):
        ep.fit_strategy_mixture_stan(stan_prepared)

    monkeypatch.setattr(
        dm,
        "fit_strategy_mixture_stan",
        lambda prep, seed=1, **kwargs: dm._result(
            "eye_strategy_mixture_stan",
            prepared=prep,
            fit=_FakeFit(),
            summary=pd.DataFrame(),
        ),
    )
    renamed = _strategy_data().rename(
        columns={
            "participant_id": "person",
            "item_id": "itemx",
            "score": "response",
        }
    )
    override_spec = _strategy_spec(engine="stan")
    fit = ep.fit_theory_strategy_irt(
        renamed,
        override_spec,
        response="response",
        participant="person",
        item="itemx",
    )
    assert fit.prepared.spec.response == "response"


def test_strategy_stan_posterior_nonem_diagnostics_sequence_and_single_condition():
    spec = _strategy_spec(engine="stan", condition="condition")
    prepared = ep.prepare_strategy_mixture_data(_strategy_data(condition=True), spec)
    draws = pd.DataFrame(
        {
            "p1": [0.8, 0.6],
            "p2": [0.2, 0.4],
            "p3": [0.3, 0.4],
            "p4": [0.7, 0.6],
            "p5": [0.9, 0.8],
            "p6": [0.1, 0.2],
            "p7": [0.4, 0.3],
            "p8": [0.6, 0.7],
        }
    )
    model = dm._result(
        "eye_strategy_mixture_stan", fit=_FakeFit(draws=draws)
    )
    obj = dm._result(
        "eye_theory_strategy_irt", model=model, prepared=prepared, spec=spec
    )
    posterior = ep.strategy_posterior_probabilities(obj)
    assert len(posterior) == 4
    diag = ep.strategy_label_switching_diagnostics(obj)
    assert diag.assessed.iloc[0] == False

    validation = ep.validate_strategy_manipulation(
        obj, "condition", "analytic", minimum_contrast=0
    )
    assert validation.contrast == 0.0

    broken = dm._result(
        "eye_theory_strategy_irt",
        model=dm._result(
            "eye_strategy_mixture_stan",
            fit=_FakeFit(draws=RuntimeError("no posterior")),
        ),
        prepared=prepared,
        spec=spec,
    )
    with pytest.raises(EyeProcessModelError, match="probabilities are unavailable"):
        ep.strategy_posterior_probabilities(broken)

    em_spec = _strategy_spec()
    sensitivity = ep.strategy_aoi_sensitivity(
        [_strategy_data(), _strategy_data()],
        em_spec,
        max_iter=2,
        seed=2,
    )
    assert set(sensitivity.summary["definition"]) == {"definition1", "definition2"}


def test_strategy_simulation_input_variants_and_invalid_size():
    frame = pd.DataFrame(
        [[1.0, 0.0], [-1.0, 1.0]],
        index=["a", "b"],
        columns=["f1", "f2"],
    )
    a = ep.simulate_strategy_mixture_data(
        n_person=2,
        n_item=2,
        signatures=frame,
        strategy_prevalence=[0.7, 0.3],
        seed=3,
    )
    b = ep.simulate_strategy_mixture_data(
        n_person=2,
        n_item=2,
        signatures={"a": {"f1": 1}, "b": {"f1": -1}},
        seed=3,
    )
    c = ep.simulate_strategy_mixture_data(
        n_person=2,
        n_item=2,
        signatures=np.array([[1.0], [-1.0]]),
        seed=3,
    )
    assert len(a) == len(b) == len(c) == 4
    with pytest.raises(EyeProcessValidationError, match="sizes/signatures"):
        ep.simulate_strategy_mixture_data(
            n_person=1, n_item=2, signatures=np.array([[1.0], [-1.0]])
        )


def _diffusion_data():
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "item_id": ["I1", "I2", "I1", "I2"],
            "score": [0, 1, 1, 0],
            "response_time": [0.5, 0.7, 0.8, 0.6],
            "gaze": [-1.0, 0.5, 1.0, -0.5],
            "boundary": [1.0, 1.0, 1.0, 1.0],
            "censor": ["observed", "right", "left", "observed"],
        }
    )


def test_gaze_spec_matrix_prepare_guards_and_censor_paths():
    spec = ep.gaze_diffusion_spec(gaze_features=["gaze"])
    assert spec.drift_features == ["gaze"]
    assert dm._std_matrix(_diffusion_data(), [], "x").shape == (4, 0)
    const = dm._std_matrix(_diffusion_data(), ["boundary"], "b")
    assert np.isfinite(const).all()

    with pytest.raises(EyeProcessValidationError, match="only one diffusion"):
        ep.gaze_diffusion_spec(
            drift_features=["gaze"], boundary_features=["gaze"]
        )

    with pytest.raises(EyeProcessValidationError, match="Response times"):
        ep.prepare_gaze_diffusion_data(
            _diffusion_data().assign(response_time=[0.01] * 4), spec
        )
    with pytest.raises(EyeProcessValidationError, match="milliseconds"):
        ep.prepare_gaze_diffusion_data(
            _diffusion_data().assign(response_time=[500.0] * 4), spec
        )
    with pytest.raises(EyeProcessValidationError, match="coded 0/1"):
        ep.prepare_gaze_diffusion_data(
            _diffusion_data().assign(score=[0, 1, 2, 0]), spec
        )

    censor_spec = ep.gaze_diffusion_spec(
        gaze_features=["gaze"], censor_column="censor"
    )
    prepared = ep.prepare_gaze_diffusion_data(_diffusion_data(), censor_spec)
    assert prepared.censor.tolist() == [0, 1, 2, 0]

    with pytest.raises(EyeProcessValidationError, match="Censoring values"):
        ep.prepare_gaze_diffusion_data(
            _diffusion_data().assign(censor=["observed", "bad", "left", "right"]),
            censor_spec,
        )


def test_gaze_stan_version_success_error_and_dispatch(monkeypatch):
    prepared = ep.prepare_gaze_diffusion_data(
        _diffusion_data(), ep.gaze_diffusion_spec(gaze_features=["gaze"], engine="stan")
    )

    monkeypatch.setattr(dm, "_cmdstanpy", lambda: _fake_cmdstan({}, version=None))
    with pytest.raises(EyeProcessBackendError, match="2.38"):
        ep.fit_gaze_diffusion_stan(prepared)

    monkeypatch.setattr(dm, "_cmdstanpy", lambda: _fake_cmdstan({}, version=(2, 37)))
    with pytest.raises(EyeProcessBackendError, match="2.38"):
        ep.fit_gaze_diffusion_stan(prepared)

    monkeypatch.setattr(
        dm, "_cmdstanpy", lambda: _fake_cmdstan({}, version=RuntimeError("missing"))
    )
    with pytest.raises(EyeProcessBackendError, match="not available"):
        ep.fit_gaze_diffusion_stan(prepared)

    holder = {}
    monkeypatch.setattr(dm, "_cmdstanpy", lambda: _fake_cmdstan(holder, version=(2, 38)))
    fitted = ep.fit_gaze_diffusion_stan(prepared, seed=4)
    assert fitted.eyeprocess_class == "eye_gaze_diffusion_stan"
    assert holder["data"]["N"] == 4

    monkeypatch.setattr(
        dm,
        "_cmdstanpy",
        lambda: _fake_cmdstan({}, version=(2, 38), fail="sample"),
    )
    with pytest.raises(EyeProcessBackendError, match="gaze-diffusion fitting failed"):
        ep.fit_gaze_diffusion_stan(prepared)

    with pytest.raises(EyeProcessBackendError, match="Legacy R diffusion engine"):
        ep.fit_gaze_diffusion_irt(
            _diffusion_data(),
            ep.gaze_diffusion_spec(gaze_features=["gaze"], engine="brms"),
        )

    stub = dm._result(
        "eye_gaze_diffusion_stan",
        summary=pd.DataFrame({"variable": ["beta_drift[1]"], "Mean": [0.2]}),
        diagnostics=pd.DataFrame(),
    )
    monkeypatch.setattr(dm, "fit_gaze_diffusion_stan", lambda *a, **k: stub)
    dispatched = ep.fit_gaze_diffusion_irt(
        _diffusion_data(),
        ep.gaze_diffusion_spec(gaze_features=["gaze"], engine="stan"),
    )
    assert dispatched.model is stub


def test_diffusion_extract_diagnostics_ppc_and_compare_paths():
    baseline = ep.fit_gaze_diffusion_irt(
        _diffusion_data(), ep.gaze_diffusion_spec(gaze_features=["gaze"])
    )
    extracted = ep.extract_diffusion_parameters(baseline)
    assert set(extracted["component"]) == {"accuracy", "log_response_time"}
    assert ep.diffusion_parameter_diagnostics(baseline).engine == "baseline"
    assert ep.diffusion_posterior_predictive(baseline).method == "baseline descriptive"

    stan_summary = pd.DataFrame(
        {
            "variable": ["beta_drift[1]", "other[1]"],
            "Mean": [0.2, 9.0],
        }
    )
    stan_obj = dm._result(
        "eye_gaze_diffusion_irt",
        prepared=baseline.prepared,
        model=dm._result(
            "eye_gaze_diffusion_stan",
            summary=stan_summary,
            diagnostics=pd.DataFrame({"rhat": [1.0]}),
        ),
    )
    stan_extract = ep.extract_diffusion_parameters(stan_obj)
    assert stan_extract["variable"].tolist() == ["beta_drift[1]"]
    assert ep.diffusion_parameter_diagnostics(stan_obj).engine == "stan"
    assert ep.diffusion_posterior_predictive(stan_obj, method="stan").method == "stan"

    indexed = stan_summary.drop(columns=["variable"])
    indexed.index = ["beta_drift[1]", "other[1]"]
    indexed_obj = dm._result(
        "eye_gaze_diffusion_irt",
        prepared=baseline.prepared,
        model=dm._result(
            "eye_gaze_diffusion_stan",
            summary=indexed,
            diagnostics=pd.DataFrame(),
        ),
    )
    assert len(ep.extract_diffusion_parameters(indexed_obj)) == 1

    with pytest.raises(EyeProcessValidationError, match="gaze-diffusion fit"):
        ep.compare_diffusion_accuracy_rt(dm._result("wrong"))
    comparison = ep.compare_diffusion_accuracy_rt(baseline)
    assert comparison["parameters"].iloc[1] == len(extracted)


def test_wiener_boundaries_timeout_simulation_contaminant_and_identification():
    class FixedRng:
        def __init__(self, value):
            self.value = value

        def normal(self):
            return self.value

    y, _ = dm._wiener_trial(
        FixedRng(100.0), drift=0, boundary=1, nondecision=0.1, dt=0.1
    )
    assert y == 1
    y, _ = dm._wiener_trial(
        FixedRng(-100.0), drift=0, boundary=1, nondecision=0.1, dt=0.1
    )
    assert y == 0
    y, rt = dm._wiener_trial(
        FixedRng(0.0),
        drift=0,
        boundary=100,
        nondecision=0.1,
        dt=0.01,
        max_time=0.01,
    )
    assert y == 0
    assert rt == pytest.approx(0.11)

    with pytest.raises(EyeProcessValidationError, match="contaminant_fraction"):
        ep.simulate_gaze_diffusion_data(
            n_person=2, n_item=2, contaminant_fraction=1
        )
    simulated = ep.simulate_gaze_diffusion_data(
        n_person=2,
        n_item=2,
        trials_per_item=2,
        contaminant_fraction=0.99,
        max_decision_time=0.02,
        time_step=0.01,
        seed=7,
    )
    assert simulated["contaminant"].any()

    custom = ep.diffusion_identification_study(
        pd.DataFrame(
            {
                "n_person": [5],
                "n_item": [3],
                "gaze_effect": [0.2],
                "contaminant_fraction": [0.01],
            }
        ),
        replications=2,
        base_seed=10,
        spec=ep.gaze_diffusion_spec(engine="baseline"),
    )
    assert custom.plan.jobs["seed"].tolist() == [10, 11]
    default = ep.diffusion_identification_study(replications=1)
    assert len(default.plan.jobs) == 16
