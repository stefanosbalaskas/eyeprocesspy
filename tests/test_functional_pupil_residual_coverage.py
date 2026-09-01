from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.functional_pupil as fp


def _long_pupil(n_person=12, n_item=4, n_time=9, *, event=True, seed=19):
    rng = np.random.default_rng(seed)
    rows = []
    for pi in range(n_person):
        for ji in range(n_item):
            score = int((pi + 2 * ji + (pi // 3)) % 2)
            trial = f"P{pi + 1}-I{ji + 1}"
            for s in range(n_time):
                time = 800.0 + s * 100.0
                rows.append(
                    {
                        "participant_id": f"P{pi + 1}",
                        "item_id": f"I{ji + 1}",
                        "trial_id": trial,
                        "time_ms": time,
                        "event_time": 1200.0 if event else np.nan,
                        "pupil": 3.2 + 0.015 * s + 0.025 * score + rng.normal(0, 0.035),
                        "score": score,
                        "response_time": 0.9 + 0.04 * ji + rng.normal(0, 0.01),
                        "luminance": 0.25 + 0.02 * ji,
                        "gaze_x": 0.2 + 0.01 * s,
                        "gaze_y": 0.4 - 0.005 * s,
                        "blink": s == n_time - 1,
                        "interpolated": s in {2, 3} and ji == 0,
                    }
                )
    return pd.DataFrame(rows)


def _event_spec(**kwargs):
    args = dict(
        df=3,
        alignment="event",
        event_time_column="event_time",
        latency_ms=0,
        baseline_window=(-400, 0),
        min_baseline_samples=3,
        pupil_column="pupil",
        time_column="time_ms",
        engine="two_stage_glm",
    )
    args.update(kwargs)
    return ep.functional_pupil_irt_spec(**args)


def test_choice_name_and_spec_validation_residuals():
    with pytest.raises(ep.EyeProcessValidationError, match="must not be empty"):
        fp._choice([], ("a", "b"), "choice")
    with pytest.raises(ep.EyeProcessValidationError, match="must be one of"):
        fp._choice("c", ("a", "b"), "choice")
    with pytest.raises(ep.EyeProcessValidationError, match="non-empty column"):
        fp._name("", "column")

    invalid = [
        ({"df": 1}, "at least two"),
        ({"response": ""}, "non-empty column"),
        ({"latency_ms": np.inf}, "finite number"),
        ({"baseline_window": (0, -1)}, "baseline_window"),
        ({"min_baseline_samples": 0}, "positive integer"),
        ({"time_window": (1, 1)}, "time_window"),
        ({"max_interpolated_fraction": 1.1}, "between zero and one"),
        ({"chains": 0}, "positive integers"),
        ({"pupil_column": ""}, "non-empty column"),
    ]
    for kwargs, message in invalid:
        with pytest.raises(ep.EyeProcessValidationError, match=message):
            ep.functional_pupil_irt_spec(**kwargs)


def test_first_table_find_col_and_binary_response_variants():
    x = ep.new_eye_dataset(validate=False)
    x["biometrics"] = pd.DataFrame({"channel": ["eda"], "value": [1.0]})
    first = fp._first_table(x)
    assert first.loc[0, "channel"] == "eda"
    x["biometrics"] = x["biometrics"].iloc[0:0]
    x["gaze_samples"] = pd.DataFrame({"gaze_x": [0.2]})
    assert fp._first_table(x).loc[0, "gaze_x"] == pytest.approx(0.2)
    x["gaze_samples"] = x["gaze_samples"].iloc[0:0]
    assert fp._first_table(x).empty

    d = pd.DataFrame({"pupil": [1.0]})
    assert fp._find_col(d, None, ("missing", "pupil"), "pupil column") == "pupil"
    with pytest.raises(ep.EyeProcessValidationError, match="unavailable"):
        fp._find_col(d, "missing", ("pupil",), "pupil column")
    with pytest.raises(ep.EyeProcessValidationError, match="Could not identify"):
        fp._find_col(pd.DataFrame({"x": [1]}), None, ("a", "b"), "field")

    assert fp._binary_response(pd.Series([False, True], dtype=bool)).tolist() == [0, 1]
    assert fp._binary_response(pd.Series([1, 2, 1, 2])).tolist() == [0, 1, 0, 1]
    assert fp._binary_response(pd.Series([-1, 1, -1, 1])).tolist() == [0, 1, 0, 1]
    assert fp._binary_response(pd.Series(["no", "yes", "no"])).tolist() == [0, 1, 0]
    with pytest.raises(ep.EyeProcessValidationError, match="exactly two observed levels"):
        fp._binary_response(pd.Series(["a", "b", "c"]))


def test_prepare_frame_guards_timestamp_scaling_and_response_contracts():
    spec = _event_spec()
    with pytest.raises(ep.EyeProcessValidationError, match="eye dataset or a long pupil data frame"):
        fp._prepare_frame(object(), spec)
    with pytest.raises(ep.EyeProcessValidationError, match="Could not identify pupil column"):
        fp._prepare_frame(pd.DataFrame({"time_ms": [0.0]}), spec)

    base = _long_pupil(n_person=2, n_item=2, n_time=5)
    no_ids = base.drop(columns=["item_id"])
    with pytest.raises(ep.EyeProcessValidationError, match="missing identifiers"):
        fp._prepare_frame(no_ids, spec)
    missing_id = base.copy()
    missing_id.loc[0, "participant_id"] = pd.NA
    with pytest.raises(ep.EyeProcessValidationError, match="identifiers must be non-missing"):
        fp._prepare_frame(missing_id, spec)
    no_response = base.drop(columns=["score"])
    with pytest.raises(ep.EyeProcessValidationError, match="Response field"):
        fp._prepare_frame(no_response, spec)

    seconds = base.rename(columns={"time_ms": "timestamp_seconds"})
    second_spec = ep.functional_pupil_irt_spec(
        df=3,
        alignment="event",
        event_time_column="event_time",
        latency_ms=0,
        baseline_window=(-400, 0),
        pupil_column="pupil",
        time_column="timestamp_seconds",
    )
    seconds["timestamp_seconds"] /= 1000.0
    scaled = fp._prepare_frame(seconds, second_spec)
    assert scaled[".time"].iloc[1] - scaled[".time"].iloc[0] == pytest.approx(100.0)

    generic_ts = base.rename(columns={"time_ms": "timestamp"})
    generic_ts["timestamp"] /= 1000.0
    generic_spec = ep.functional_pupil_irt_spec(
        df=3,
        alignment="event",
        event_time_column="event_time",
        latency_ms=0,
        baseline_window=(-400, 0),
        pupil_column="pupil",
        time_column="timestamp",
    )
    generic = fp._prepare_frame(generic_ts, generic_spec)
    assert generic[".time"].iloc[1] - generic[".time"].iloc[0] == pytest.approx(100.0)


def test_prepare_data_trial_alignment_blink_time_and_interpolation_paths():
    d = _long_pupil(n_person=3, n_item=2, n_time=9)
    trial_spec = ep.functional_pupil_irt_spec(
        df=3,
        alignment="trial",
        latency_ms=0,
        baseline_window=(0, 300),
        pupil_column="pupil",
        time_column="time_ms",
        blink_column="blink",
        interpolated_column="interpolated",
        max_interpolated_fraction=0.5,
        engine="two_stage_glm",
    )
    prepared = ep.prepare_functional_pupil_data(d, trial_spec)
    assert not prepared.data["blink"].any()
    assert prepared.data[".time"].min() == pytest.approx(0.0)
    assert prepared.data.interpolated_fraction.max() <= 0.5

    clipped = _event_spec(time_window=(-50, 50))
    with pytest.raises(ep.EyeProcessValidationError, match="No pupil samples remain"):
        ep.prepare_functional_pupil_data(d, clipped)

    strict_interp = _event_spec(
        interpolated_column="interpolated",
        max_interpolated_fraction=0.0,
    )
    only_interpolated = d.loc[d["item_id"].eq("I1")].copy()
    with pytest.raises(ep.EyeProcessValidationError, match="interpolation-quality"):
        ep.prepare_functional_pupil_data(only_interpolated, strict_interp)


def test_prepare_data_event_baseline_percent_zscore_and_invariance_guards():
    d = _long_pupil(n_person=2, n_item=2, n_time=9)
    bad_event = d.copy()
    bad_event.loc[bad_event.index[0], "event_time"] = 1300.0
    with pytest.raises(ep.EyeProcessValidationError, match="one finite, invariant"):
        ep.prepare_functional_pupil_data(bad_event, _event_spec())

    insufficient = _event_spec(min_baseline_samples=8)
    with pytest.raises(ep.EyeProcessValidationError, match="baseline-quality"):
        ep.prepare_functional_pupil_data(d, insufficient)

    zero = d.copy()
    zero["pupil"] = 0.0
    with pytest.raises(ep.EyeProcessValidationError, match="baseline-quality"):
        ep.prepare_functional_pupil_data(zero, _event_spec(baseline_method="percent"))

    constant = d.copy()
    constant["pupil"] = 3.0
    with pytest.raises(ep.EyeProcessValidationError, match="baseline-quality"):
        ep.prepare_functional_pupil_data(constant, _event_spec(baseline_method="zscore"))


def test_prepare_data_nuisance_residualization_and_varying_response_guard():
    d = _long_pupil(n_person=5, n_item=3, n_time=9)
    spec = _event_spec(
        luminance_column="luminance",
        gaze_x_column="gaze_x",
        gaze_y_column="gaze_y",
        nuisance_by_participant=True,
    )
    prepared = ep.prepare_functional_pupil_data(d, spec)
    assert prepared.nuisance_model is not None
    assert np.isfinite(prepared.data["pupil_adjusted"]).all()

    varying = d.copy()
    trial = varying["trial_id"].iloc[0]
    idx = varying.index[varying["trial_id"].eq(trial)]
    varying.loc[idx[-1], "score"] = 1 - int(varying.loc[idx[0], "score"])
    with pytest.raises(ep.EyeProcessValidationError, match="invariant response"):
        ep.prepare_functional_pupil_data(varying, _event_spec())


def test_functional_basis_bspline_and_guard_paths():
    time = np.linspace(-1, 1, 20)
    bs = ep.functional_pupil_basis(time, df=4, basis="bspline", degree=2, boundary_knots=(-1, 1), knots=(-0.2, 0.2))
    assert bs.shape == (20, 4)
    assert bs.attrs["basis"] == "bspline"
    assert bs.attrs["knots"] == (-0.2, 0.2)
    with pytest.raises(ep.EyeProcessValidationError, match="degree"):
        ep.functional_pupil_basis(time, df=4, degree=0)
    with pytest.raises(ep.EyeProcessValidationError, match="insufficient"):
        ep.functional_pupil_basis([0.0, 0.0, 0.0], df=3)
    with pytest.raises(ep.EyeProcessValidationError, match="boundary_knots"):
        ep.functional_pupil_basis(time, df=4, boundary_knots=(1, -1))


def test_trial_coefficients_length_and_unsupported_basis_contract():
    prepared = ep.prepare_functional_pupil_data(_long_pupil(n_person=2, n_item=2, n_time=9), _event_spec())
    basis = ep.functional_pupil_basis(prepared, df=3)
    with pytest.raises(ep.EyeProcessValidationError, match="one row per prepared"):
        fp._trial_coefficients(prepared, basis.iloc[:-1])
    oversized = pd.DataFrame(np.ones((len(prepared.data), 20)))
    co = fp._trial_coefficients(prepared, oversized)
    assert not co["basis_supported"].any()
    assert co.filter(like="pupil_basis_").isna().all().all()


def test_joint_fit_guards_small_sample_backend_and_mocked_stan(monkeypatch):
    d = _long_pupil(n_person=2, n_item=2, n_time=9)
    with pytest.raises(ep.EyeProcessValidationError, match="functional_pupil_irt_spec"):
        ep.fit_joint_functional_pupil_irt(d, spec=object())
    with pytest.raises(ep.EyeProcessValidationError, match="Too few supported"):
        ep.fit_joint_functional_pupil_irt(d, _event_spec())

    enough = _long_pupil(n_person=5, n_item=3, n_time=9)
    with pytest.raises(ep.EyeProcessBackendError, match="brms"):
        ep.fit_joint_functional_pupil_irt(enough, _event_spec(engine="brms"))

    fake_stan = fp._result(
        "eye_functional_pupil_stan",
        diagnostics=pd.DataFrame([{"converged": True}]),
        summary=pd.DataFrame({"Mean": [0.1], "StdDev": [0.2], "R_hat": [1.0]}, index=["beta[1]"]),
    )
    monkeypatch.setattr(fp, "fit_functional_pupil_stan", lambda *args, **kwargs: fake_stan)
    fitted = ep.fit_joint_functional_pupil_irt(d, _event_spec(engine="stan"))
    assert fitted.model.eyeprocess_class == "eye_functional_pupil_stan"
    assert bool(fitted.diagnostics.loc[0, "converged"])


def test_functional_stan_success_contract_with_fake_cmdstan(monkeypatch):
    prepared = ep.prepare_functional_pupil_data(_long_pupil(n_person=2, n_item=2, n_time=9), _event_spec(engine="stan"))
    basis = ep.functional_pupil_basis(prepared, df=3)

    class FakeFit:
        def summary(self):
            return pd.DataFrame({"Mean": [0.1], "StdDev": [0.2], "R_hat": [1.01]}, index=["beta[1]"])

    class FakeModel:
        def __init__(self, stan_file):
            self.stan_file = stan_file

        def sample(self, **kwargs):
            assert kwargs["data"]["N_trial"] == len(prepared.trials)
            assert kwargs["data"]["N_sample"] == len(prepared.data)
            return FakeFit()

    monkeypatch.setitem(sys.modules, "cmdstanpy", SimpleNamespace(CmdStanModel=FakeModel))
    out = fp.fit_functional_pupil_stan(prepared, basis_matrix=basis, seed=3, refresh=0)
    assert out.eyeprocess_class == "eye_functional_pupil_stan"
    assert bool(out.diagnostics.loc[0, "converged"])


def test_extract_parameters_stan_pattern_and_diagnostics_legacy_paths():
    stan = fp._result(
        "eye_functional_pupil_stan",
        summary=pd.DataFrame(
            {"Mean": [0.2, -0.1], "StdDev": [0.05, 0.08], "R_hat": [1.0, 1.01]},
            index=["beta[1]", "alpha"],
        ),
        diagnostics=pd.DataFrame([{"converged": True}]),
    )
    fit = fp._result("eye_functional_pupil_irt", model=stan, legacy=False)
    pars = ep.extract_functional_pupil_parameters(fit, pattern="beta")
    assert pars["parameter"].tolist() == ["beta[1]"]
    assert pars.loc[0, "lower"] < pars.loc[0, "estimate"] < pars.loc[0, "upper"]

    legacy = fp._result("eye_functional_pupil_irt", model=object(), legacy=True)
    diag = ep.functional_pupil_diagnostics(legacy)
    assert diag.checks.loc[0, "check"] == "legacy_bridge"
    with pytest.raises(ep.EyeProcessValidationError, match="Expected"):
        ep.extract_functional_pupil_parameters(object())
    with pytest.raises(ep.EyeProcessValidationError, match="Expected"):
        ep.functional_pupil_diagnostics(object())


def test_sensitivity_grid_guards_error_capture_and_reraise():
    d = _long_pupil(n_person=2, n_item=2, n_time=9)
    with pytest.raises(ep.EyeProcessValidationError, match="baseline window"):
        ep.pupil_preprocessing_grid(baseline_windows=())
    with pytest.raises(ep.EyeProcessValidationError, match="basis_df"):
        ep.pupil_preprocessing_grid(baseline_windows=((-400, 0),), latency_ms=(0,), basis_df=(1,), baseline_methods=("subtract",), max_interpolated_fraction=(0.2,))
    with pytest.raises(ep.EyeProcessValidationError, match="Interpolation thresholds"):
        ep.pupil_preprocessing_grid(baseline_windows=((-400, 0),), latency_ms=(0,), basis_df=(3,), baseline_methods=("subtract",), max_interpolated_fraction=(2.0,))

    with pytest.raises(ep.EyeProcessValidationError, match="non-empty"):
        ep.pupil_preprocessing_sensitivity(d, grid=pd.DataFrame())
    with pytest.raises(ep.EyeProcessValidationError, match="missing"):
        ep.pupil_preprocessing_sensitivity(d, grid=pd.DataFrame({"df": [3]}))

    bad_grid = pd.DataFrame(
        [
            {
                "baseline_start": -400.0,
                "baseline_end": 0.0,
                "latency_ms": 0.0,
                "df": 3,
                "baseline_method": "subtract",
                "max_interpolated_fraction": 0.2,
            }
        ]
    )
    captured = ep.pupil_preprocessing_sensitivity(d, grid=bad_grid, base_spec=_event_spec(), fit=True, continue_on_error=True)
    assert captured.results.loc[0, "parameter"] == ".error"
    with pytest.raises(ep.EyeProcessValidationError, match="Too few supported"):
        ep.pupil_preprocessing_sensitivity(d, grid=bad_grid, base_spec=_event_spec(), fit=True, continue_on_error=False)


def test_compare_functional_scalar_models_aic_logloss_and_guards():
    prepared = ep.prepare_functional_pupil_data(_long_pupil(n_person=12, n_item=4, n_time=9), _event_spec())
    aic = ep.compare_functional_scalar_models(prepared, criterion="AIC")
    assert set(aic["model"]) == {"functional", "scalar"}
    assert aic["value"].notna().all()
    logloss = ep.compare_functional_scalar_models(prepared, criterion="log_loss", folds=4, seed=4)
    assert set(logloss["model"]) == {"functional", "scalar"}
    assert (logloss["value"] >= 0).all()

    with pytest.raises(ep.EyeProcessValidationError, match="Expected a functional pupil"):
        ep.compare_functional_scalar_models(object())
    with pytest.raises(ep.EyeProcessValidationError, match="No requested scalar"):
        ep.compare_functional_scalar_models(prepared, scalar_features=("missing",))

    small = ep.prepare_functional_pupil_data(_long_pupil(n_person=2, n_item=2, n_time=9), _event_spec())
    with pytest.raises(ep.EyeProcessValidationError, match="Too few complete grouped"):
        ep.compare_functional_scalar_models(small)
