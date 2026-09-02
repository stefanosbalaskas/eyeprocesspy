from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.functional_pupil as fp
import eyeprocesspy.legacy_models as legacy


def _long_pupil(n_person: int = 4, n_item: int = 3, n_time: int = 6) -> pd.DataFrame:
    rows = []
    for pi in range(n_person):
        for ji in range(n_item):
            score = (pi + ji) % 2
            trial = f"P{pi + 1}-I{ji + 1}"
            for si in range(n_time):
                rows.append(
                    {
                        "participant_id": f"P{pi + 1}",
                        "item_id": f"I{ji + 1}",
                        "trial_id": trial,
                        "time_ms": float(si * 100),
                        "event_time": 300.0,
                        "pupil": 3.0 + 0.08 * si + 0.03 * score + 0.01 * ji,
                        "score": score,
                        "response_time": 0.8 + 0.02 * ji,
                        "luminance": 0.2 + 0.01 * si,
                    }
                )
    return pd.DataFrame(rows)


def _trial_spec(**kwargs):
    args = dict(
        df=2,
        alignment="trial",
        latency_ms=0,
        baseline_window=(0, 200),
        min_baseline_samples=3,
        pupil_column="pupil",
        time_column="time_ms",
        engine="two_stage_glm",
    )
    args.update(kwargs)
    return ep.functional_pupil_irt_spec(**args)


def _eye(samples: pd.DataFrame | None = None, responses: pd.DataFrame | None = None):
    x = ep.new_eye_dataset(validate=False)
    if samples is not None:
        x["eye_samples"] = samples
    if responses is not None:
        x["responses"] = responses
    return x


def test_numeric_fallthrough_and_eye_dataset_response_merge_branches():
    assert fp._binary_response(pd.Series([2, 3, 2, 3])).tolist() == [0, 1, 0, 1]

    spec = _trial_spec()
    with pytest.raises(ep.EyeProcessValidationError, match="No eye/pupil sample table"):
        fp._prepare_frame(_eye(), spec)

    samples = pd.DataFrame(
        {
            "recording_id": ["R1", "R1"],
            "trial_id": ["T1", "T1"],
            "pupil": [3.0, 3.1],
            "time_ms": [0.0, 100.0],
        }
    )
    responses = pd.DataFrame(
        {
            "recording_id": ["R1"],
            "trial_id": ["T1"],
            "participant_id": ["P1"],
            "item_id": ["I1"],
            "score": [1],
            "response_time": [0.8],
        }
    )
    merged = fp._prepare_frame(_eye(samples, responses), spec)
    assert merged["participant_id"].eq("P1").all()
    assert merged["item_id"].eq("I1").all()
    assert merged["score"].eq(1).all()

    no_keys = pd.DataFrame({"pupil": [3.0], "time_ms": [0.0]})
    with pytest.raises(ep.EyeProcessValidationError, match="missing identifiers"):
        fp._prepare_frame(_eye(no_keys, pd.DataFrame({"score": [1]})), spec)

    canonical = pd.DataFrame(
        {
            "participant_id": ["P1", "P1"],
            "item_id": ["I1", "I1"],
            "trial_id": ["T1", "T1"],
            "pupil": [3.0, 3.1],
            "time_ms": [0.0, 100.0],
        }
    )
    response_only = pd.DataFrame({"trial_id": ["T1"], "score": [1]})
    second_merge = fp._prepare_frame(_eye(canonical, response_only), spec)
    assert second_merge["score"].eq(1).all()

    with pytest.raises(ep.EyeProcessValidationError, match="Response field"):
        fp._prepare_frame(_eye(canonical, pd.DataFrame({"score": [1]})), spec)


def test_prepare_guards_valid_percent_zscore_nuisance_fallback_and_zero_time(monkeypatch):
    d = _long_pupil(n_person=2, n_item=2, n_time=6)

    with pytest.raises(ep.EyeProcessValidationError, match="functional_pupil_irt_spec"):
        ep.prepare_functional_pupil_data(d, spec=object())

    event_spec = ep.functional_pupil_irt_spec(
        df=2,
        alignment="event",
        event_time_column="event_time",
        latency_ms=0,
        baseline_window=(-300, 0),
        min_baseline_samples=3,
        pupil_column="pupil",
        time_column="time_ms",
    )
    with pytest.raises(ep.EyeProcessValidationError, match="Event-alignment column"):
        ep.prepare_functional_pupil_data(d.drop(columns=["event_time"]), event_spec)

    percent = ep.prepare_functional_pupil_data(
        d,
        ep.functional_pupil_irt_spec(
            df=2,
            alignment="event",
            event_time_column="event_time",
            latency_ms=0,
            baseline_window=(-300, 0),
            min_baseline_samples=3,
            pupil_column="pupil",
            time_column="time_ms",
            baseline_method="percent",
        ),
    )
    assert np.isfinite(percent.data["pupil_corrected"]).all()

    zscore = ep.prepare_functional_pupil_data(
        d,
        ep.functional_pupil_irt_spec(
            df=2,
            alignment="event",
            event_time_column="event_time",
            latency_ms=0,
            baseline_window=(-300, 0),
            min_baseline_samples=3,
            pupil_column="pupil",
            time_column="time_ms",
            baseline_method="zscore",
        ),
    )
    assert np.isfinite(zscore.data["pupil_corrected"]).all()

    import statsmodels.formula.api as smf

    def broken_ols(*args, **kwargs):
        raise RuntimeError("forced nuisance failure")

    monkeypatch.setattr(smf, "ols", broken_ols)
    nuisance = ep.prepare_functional_pupil_data(
        d,
        _trial_spec(luminance_column="luminance"),
    )
    assert nuisance.nuisance_model is None
    assert np.allclose(
        nuisance.data["pupil_adjusted"].to_numpy(float),
        nuisance.data["pupil_corrected"].to_numpy(float),
    )

    flat_time = pd.DataFrame(
        {
            "participant_id": ["P1"] * 4,
            "item_id": ["I1"] * 4,
            "trial_id": ["T1"] * 4,
            "time_ms": [0.0] * 4,
            "pupil": [3.0, 3.1, 3.2, 3.3],
            "score": [1] * 4,
        }
    )
    with pytest.raises(ep.EyeProcessValidationError, match="times must vary"):
        ep.prepare_functional_pupil_data(
            flat_time,
            _trial_spec(baseline_window=(-1, 1), min_baseline_samples=3),
        )


def test_basis_and_stan_backend_residual_guards(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="df"):
        ep.functional_pupil_basis(np.linspace(-1, 1, 8), df=1)

    original_import = builtins.__import__

    def deny_patsy(name, *args, **kwargs):
        if name == "patsy" or name.startswith("patsy."):
            raise ImportError("forced patsy absence")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", deny_patsy)
        with pytest.raises(ep.EyeProcessBackendError, match="requires patsy"):
            ep.functional_pupil_basis(np.linspace(-1, 1, 8), df=3)

    with pytest.raises(ep.EyeProcessValidationError, match="Expected prepared"):
        fp.fit_functional_pupil_stan(object())

    prepared = ep.prepare_functional_pupil_data(_long_pupil(2, 2, 6), _trial_spec(engine="stan"))
    basis = ep.functional_pupil_basis(prepared, df=2)

    class BrokenCmdStanModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("compile failed")

    monkeypatch.setitem(sys.modules, "cmdstanpy", SimpleNamespace(CmdStanModel=BrokenCmdStanModel))
    with pytest.raises(ep.EyeProcessBackendError, match="CmdStan is required"):
        fp.fit_functional_pupil_stan(prepared, basis_matrix=basis)


def test_legacy_bridge_all_engine_paths_and_non_eye_reraise(monkeypatch):
    x = _eye()

    def features_with_basis(*args, **kwargs):
        y = ep.new_eye_dataset(validate=False)
        y["features"] = pd.DataFrame({"feature_name": ["functional_pupil_1", "other"]})
        return y

    fake_model = SimpleNamespace(name="legacy-glm")
    monkeypatch.setattr(legacy, "functional_pupil_features", features_with_basis)
    monkeypatch.setattr(legacy, "fit_explanatory_irt", lambda *args, **kwargs: fake_model)

    glm = ep.fit_joint_functional_pupil_irt(x, _trial_spec(engine="two_stage_glm"))
    assert glm.legacy is True
    assert glm.model is fake_model
    assert glm.feature_names == ["functional_pupil_1"]

    with pytest.raises(ep.EyeProcessBackendError, match="brms"):
        ep.fit_joint_functional_pupil_irt(x, _trial_spec(engine="brms"))

    with pytest.raises(ep.EyeProcessValidationError, match="explicit time column"):
        ep.fit_joint_functional_pupil_irt(x, _trial_spec(engine="stan"))

    def features_without_basis(*args, **kwargs):
        y = ep.new_eye_dataset(validate=False)
        y["features"] = pd.DataFrame({"feature_name": ["other"]})
        return y

    monkeypatch.setattr(legacy, "functional_pupil_features", features_without_basis)
    with pytest.raises(ep.EyeProcessValidationError, match="No functional pupil coefficients"):
        ep.fit_joint_functional_pupil_irt(x, _trial_spec(engine="two_stage_glm"))

    no_time = _long_pupil(2, 2, 6).drop(columns=["time_ms"])
    with pytest.raises(ep.EyeProcessValidationError, match="time column"):
        ep.fit_joint_functional_pupil_irt(no_time, _trial_spec())


def test_statsmodels_backend_guards_fit_and_comparison_and_fit_object_branch(monkeypatch):
    d = _long_pupil(n_person=4, n_item=3, n_time=6)

    original_import = builtins.__import__

    def deny_statsmodels(name, *args, **kwargs):
        if name == "statsmodels" or name.startswith("statsmodels."):
            raise ImportError("forced statsmodels absence")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", deny_statsmodels)
        with pytest.raises(ep.EyeProcessBackendError, match="requires statsmodels"):
            ep.fit_joint_functional_pupil_irt(d, _trial_spec())

    fit_without_basis = fp._result(
        "eye_functional_pupil_irt",
        trial_coefficients=pd.DataFrame(
            {
                "participant_id": ["P1"] * 10,
                "response": [0, 1] * 5,
                "pupil_peak": np.linspace(1, 2, 10),
            }
        ),
    )
    with pytest.raises(ep.EyeProcessValidationError, match="No functional pupil basis"):
        ep.compare_functional_scalar_models(fit_without_basis)

    prepared = ep.prepare_functional_pupil_data(d, _trial_spec())
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", deny_statsmodels)
        with pytest.raises(ep.EyeProcessBackendError, match="Model comparison requires statsmodels"):
            ep.compare_functional_scalar_models(prepared)


def test_advanced_simulation_size_and_effect_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="Simulation sizes"):
        ep.simulate_advanced_process_data(n_person=1, n_item=2, n_time=4, n_states=2)

    with pytest.raises(ep.EyeProcessValidationError, match="Simulation effects"):
        ep.simulate_advanced_process_data(
            n_person=2,
            n_item=2,
            n_time=4,
            n_states=2,
            gaze_effect=np.inf,
        )
