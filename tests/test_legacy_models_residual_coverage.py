from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.legacy_models as lm


def _dataset(seed: int = 701):
    return ep.simulate_eye_dataset(
        n_person=8,
        n_item=3,
        sampling_rate=8,
        trial_duration=0.25,
        samples_per_trial=5,
        include_biometrics=False,
        seed=seed,
    )


def _dataset_with_features(seed: int = 702):
    x = _dataset(seed)
    rows = []
    for i, row in x["responses"].reset_index(drop=True).iterrows():
        for name, value in (
            ("gaze_feature", float(np.sin(i * 0.3) + 0.01 * i)),
            ("pupil_feature", float(np.cos(i * 0.2) - 0.01 * i)),
        ):
            rows.append(
                {
                    "feature_id": f"f{i}_{name}",
                    "recording_id": row["recording_id"],
                    "participant_id": row["participant_id"],
                    "trial_id": row["trial_id"],
                    "item_id": row["item_id"],
                    "stimulus_id": pd.NA,
                    "aoi_id": pd.NA,
                    "feature_name": name,
                    "value": value,
                    "unit": "arbitrary",
                    "level": "trial",
                    "method": "residual_test",
                    "parameters": "none",
                }
            )
    x["features"] = ep.standardize_eye_table(pd.DataFrame(rows), "features")
    return x


def _fake_model(engine: str = "other", *, fit=None, data=None, metadata=None, experimental=False):
    return lm._new_model(
        SimpleNamespace() if fit is None else fit,
        engine,
        "residual_test",
        [] if data is None else data,
        metadata,
        experimental=experimental,
    )


def test_private_dataset_statsmodels_formula_and_feature_guards(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="EyeDataset"):
        lm._require_dataset(pd.DataFrame())

    assert lm._formula_text("score ~ 1") == "score ~ 1"
    with pytest.raises(ep.EyeProcessValidationError, match="formula"):
        lm._formula_text("score")

    real_import = builtins.__import__

    def fail_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("blocked statsmodels")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", fail_statsmodels)
        with pytest.raises(ep.EyeProcessBackendError, match="statsmodels"):
            lm._statsmodels()

    x = _dataset_with_features()
    empty = x.copy()
    empty["features"] = lm.empty_eye_table("features")
    assert lm._features_wide(empty).empty

    nameless = x.copy()
    f = x["features"].copy()
    f["feature_name"] = pd.NA
    nameless["features"] = f
    assert lm._features_wide(nameless).empty

    fallback = lm._features_wide(x, aggregate=object())
    assert {"gaze_feature", "pupil_feature"} <= set(fallback.columns)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "pivot_table", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pivot failed")))
        with pytest.raises(ep.EyeProcessValidationError, match="aggregate feature"):
            lm._features_wide(x)


def test_response_matrix_validation_response_mapping_and_duplicate_edges():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="value"):
        ep.response_matrix(x, value="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="duplicate"):
        ep.response_matrix(x, duplicate="bad")

    empty = x.copy()
    empty["responses"] = x["responses"].iloc[0:0].copy()
    with pytest.raises(ep.EyeProcessValidationError, match="No responses"):
        ep.response_matrix(empty)

    categorical = x.copy()
    categorical["responses"] = x["responses"].copy()
    categorical["responses"]["response"] = np.where(
        categorical["responses"]["score"].astype(int).to_numpy() == 1,
        "yes",
        "no",
    )
    categorical["responses"].loc[categorical["responses"].index[0], "response"] = pd.NA
    mapped = ep.response_matrix(categorical, value="response")
    assert mapped.shape == (8, 3)
    assert np.isnan(mapped.to_numpy()).any()

    dup = categorical.copy()
    extra = dup["responses"].iloc[[1]].copy()
    extra["response"] = "not_numeric"
    dup["responses"] = pd.concat([dup["responses"], extra], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="requires a numeric"):
        ep.response_matrix(dup, value="response", duplicate="mean")


def test_response_time_matrix_duplicate_and_alignment_edges():
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="duplicate"):
        ep.response_time_matrix(x, duplicate="bad")

    no_rt = x.copy()
    no_rt["responses"] = x["responses"].copy()
    no_rt["responses"]["response_time"] = 0.0
    with pytest.raises(ep.EyeProcessValidationError, match="positive response times"):
        ep.response_time_matrix(no_rt)

    dup = x.copy()
    dup["responses"] = pd.concat([x["responses"], x["responses"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="Duplicate participant-item response times"):
        ep.response_time_matrix(dup)
    assert ep.response_time_matrix(dup, duplicate="last").shape == (8, 3)
    assert ep.response_time_matrix(dup, duplicate="mean").shape == (8, 3)

    with pytest.raises(ep.EyeProcessValidationError, match="no common"):
        ep.align_response_matrices(pd.DataFrame([[1]], index=["P1"], columns=["I1"]), pd.DataFrame([[1]], index=["P2"], columns=["I2"]))


def test_model_data_and_explanatory_irt_validation_branches():
    x = _dataset_with_features()
    with pytest.raises(ep.EyeProcessValidationError, match="No responses"):
        empty = x.copy(); empty["responses"] = x["responses"].iloc[0:0].copy(); ep.model_data(empty)

    with pytest.raises(ep.EyeProcessValidationError, match="Formula response"):
        ep.fit_explanatory_irt(x, "missing_response ~ gaze_feature", engine="glm")
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_explanatory_irt(x, "score ~ gaze_feature", engine="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="binomial"):
        ep.fit_explanatory_irt(x, "score ~ gaze_feature", engine="glm", family="gaussian")

    fit = ep.fit_explanatory_irt(
        x,
        "score ~ gaze_feature",
        engine="glm",
        participant_random=False,
        item_random=False,
    )
    assert "C(participant_id)" not in fit.metadata["formula"]
    assert "C(item_id)" not in fit.metadata["formula"]


def test_fit_irt_accuracy_and_dif_guard_failure_paths(monkeypatch):
    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_irt(x, engine="bad")

    no_score = x.copy()
    no_score["responses"] = x["responses"].copy()
    no_score["responses"]["score"] = np.nan
    with pytest.raises(ep.EyeProcessValidationError, match="finite scores"):
        ep.fit_irt(no_score, engine="rasch_glm")

    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_accuracy_rt(x, engine="bad")

    with pytest.raises(ep.EyeProcessValidationError, match="one value per response row"):
        ep.fit_dif(x, group=["A"], engine="logistic")
    with pytest.raises(ep.EyeProcessBackendError):
        ep.fit_dif(x, group=np.tile(["A", "B"], len(x["responses"]) // 2), engine="mirt")
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_dif(x, group=np.tile(["A", "B"], len(x["responses"]) // 2), engine="bad")

    grouped = x.copy()
    grouped["responses"] = x["responses"].copy()
    grouped["responses"]["group"] = np.tile(["A", "B"], len(grouped["responses"]) // 2)
    small = ep.fit_dif(grouped, group="group", engine="logistic")
    assert set(small.fit["status"]) == {"insufficient_data"}

    enough = ep.simulate_eye_dataset(
        n_person=12,
        n_item=2,
        sampling_rate=5,
        trial_duration=0.2,
        samples_per_trial=5,
        include_pupil=False,
        include_biometrics=False,
        seed=703,
    )
    participants = list(pd.unique(enough["responses"]["participant_id"]))
    group_map = {pid: ("A" if i < len(participants) // 2 else "B") for i, pid in enumerate(participants)}
    groups = enough["responses"]["participant_id"].map(group_map).to_numpy()
    with monkeypatch.context() as patch:
        patch.setattr(lm, "_fit_binomial", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")))
        failed = ep.fit_dif(enough, group=groups, engine="logistic", items=["I001"])
    assert failed.fit.loc[0, "status"] == "fit_failed"


def test_shared_factor_guards_noncentered_unscaled_and_no_append():
    x = _dataset_with_features()
    with pytest.raises(ep.EyeProcessValidationError, match="absent"):
        ep.fit_shared_process_factor(x, ["missing_feature"])
    with pytest.raises(ep.EyeProcessValidationError, match="Insufficient"):
        ep.fit_shared_process_factor(x, ["gaze_feature", "pupil_feature"], n_factors=len(x["responses"]))

    result = ep.fit_shared_process_factor(
        x,
        ["gaze_feature", "pupil_feature"],
        n_factors=1,
        center=False,
        scale_=False,
        append=False,
    )
    assert np.allclose(result.model.fit.center, 0.0)
    assert np.allclose(result.model.fit.scale, 1.0)
    assert "process_factor_1" not in set(result.data["features"]["feature_name"].astype(str))


def test_model_extractors_statistics_and_local_dependence_guards():
    for fun in (ep.item_parameters, ep.person_scores, ep.model_fit_statistics, ep.check_local_dependence):
        with pytest.raises(ep.EyeProcessValidationError, match="eyeprocess_model"):
            fun({})

    mirt = _fake_model("mirt")
    with pytest.raises(ep.EyeProcessBackendError):
        ep.item_parameters(mirt)
    with pytest.raises(ep.EyeProcessBackendError):
        ep.person_scores(mirt)
    with pytest.raises(ep.EyeProcessBackendError):
        ep.check_local_dependence(mirt)

    other = _fake_model("other")
    assert ep.item_parameters(other).empty
    assert ep.person_scores(other).empty
    with pytest.warns(RuntimeWarning, match="Local-dependence"):
        assert ep.check_local_dependence(other) is None

    item_fit = SimpleNamespace(params=pd.Series({"C(item_id)[I1]": 0.5}))
    person_fit = SimpleNamespace(params=pd.Series({"C(participant_id)[P1]": -0.3}))
    assert ep.item_parameters(_fake_model("rasch_glm", fit=item_fit)).loc[0, "item_id"] == "I1"
    assert ep.person_scores(_fake_model("rasch_glm", fit=person_fit)).loc[0, "participant_id"] == "P1"

    stats = ep.model_fit_statistics(_fake_model("other", fit=SimpleNamespace(), data=[1, 2, 3]))
    assert stats.loc[0, "nobs"] == 3
    assert np.isnan(stats.loc[0, "logLik"])

    no_len = lm._new_model(SimpleNamespace(), "other", "residual_test", object())
    stats = ep.model_fit_statistics(no_len)
    assert np.isnan(stats.loc[0, "nobs"])


def test_joint_dynamic_and_simulation_alternate_paths():
    x = _dataset_with_features()
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_joint_process_model(x, "score ~ gaze_feature", "log_response_time ~ gaze_feature", engine="bad")

    joint = ep.fit_joint_process_model(
        x,
        "score ~ gaze_feature",
        "log_response_time ~ gaze_feature",
        process_formulas=["pupil_feature ~ gaze_feature"],
        engine="separate",
    )
    assert len(joint.fit.process) == 1

    with pytest.raises(ep.EyeProcessValidationError, match="Invalid dynamic AOI"):
        ep.fit_dynamic_aoi_model(x, source="bad")

    empty = x.copy(); empty["gaze_samples"] = x["gaze_samples"].iloc[0:0].copy()
    with pytest.raises(ep.EyeProcessValidationError, match="No AOI transitions"):
        ep.fit_dynamic_aoi_model(empty, source="samples")

    constant = x.copy(); constant["gaze_samples"] = x["gaze_samples"].copy(); constant["gaze_samples"]["true_aoi"] = "prompt"
    with pytest.raises(ep.EyeProcessValidationError, match="No AOI transitions"):
        ep.fit_dynamic_aoi_model(constant, source="samples")

    ungrouped = x.copy()
    ungrouped["gaze_samples"] = pd.DataFrame({"true_aoi": ["prompt", "options", "prompt"]})
    dynamic = ep.fit_dynamic_aoi_model(ungrouped, source="samples", smoothing=0.0)
    assert dynamic.fit.counts.to_numpy().sum() == 2

    with pytest.raises(ep.EyeProcessValidationError, match="at least two"):
        ep.simulate_eye_dataset(n_person=1, n_item=2)
    stripped = ep.simulate_eye_dataset(
        n_person=2,
        n_item=2,
        sampling_rate=5,
        trial_duration=0.2,
        samples_per_trial=5,
        include_pupil=False,
        include_biometrics=False,
        seed=704,
    )
    assert stripped["eye_samples"].empty
    assert stripped["biometrics"].empty


def test_parameter_recovery_power_process_and_process_spec_failure_paths(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="must be functions"):
        ep.parameter_recovery(1, lambda x: x, lambda x: x, lambda x: x)

    failed = ep.parameter_recovery(
        simulator=lambda: (_ for _ in ()).throw(RuntimeError("simulation failed")),
        estimator=lambda x: x,
        extractor=lambda x: x,
        truth_extractor=lambda x: x,
        replications=1,
    )
    assert "simulation failed" in failed.loc[0, "error"]

    with monkeypatch.context() as patch:
        patch.setattr(lm, "_fit_binomial", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")))
        power = ep.power_process_simulation(4, 2, 0.2, replications=1, seed=3)
    assert power.loc[0, "power"] == 0.0
    assert np.isnan(power.loc[0, "mean_estimate"])

    x = _dataset()
    with pytest.raises(ep.EyeProcessValidationError, match="process_irt_spec"):
        ep.fit_process_irt(x, {})

    spec = ep.process_irt_spec(participant_effect=False, item_effect=False)
    fit = ep.fit_process_irt(x, spec, engine="glm")
    assert fit.metadata["formula"] == "score ~ 1"


def test_process_diagnostics_functional_pupil_and_strategy_guards(monkeypatch):
    with pytest.raises(ep.EyeProcessValidationError, match="eyeprocess_model"):
        ep.process_irt_diagnostics({})

    fake = _fake_model(
        "other",
        fit=SimpleNamespace(),
        data=[1],
        metadata={"warning": "w1", "estimand_warning": "w2"},
        experimental=True,
    )
    diag = ep.process_irt_diagnostics(fake)
    assert "Model is marked experimental." in diag.warnings
    assert {"w1", "w2"} <= set(diag.warnings)

    x = _dataset()
    empty_eye = x.copy(); empty_eye["eye_samples"] = x["eye_samples"].iloc[0:0].copy()
    assert ep.functional_pupil_features(empty_eye, append=False).empty
    assert ep.functional_pupil_features(empty_eye, append=True) is empty_eye

    orphan = x.copy()
    orphan["eye_samples"] = x["eye_samples"].head(4).copy()
    orphan["eye_samples"]["trial_id"] = "not_a_trial"
    assert ep.functional_pupil_features(orphan, df=3, append=False).empty

    real_import = builtins.__import__

    def fail_patsy(name, *args, **kwargs):
        if name == "patsy" or name.startswith("patsy."):
            raise ImportError("blocked patsy")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", fail_patsy)
        with pytest.raises(ep.EyeProcessBackendError, match="patsy"):
            ep.functional_pupil_features(x, df=3, append=False)

    xf = _dataset_with_features()
    with pytest.raises(ep.EyeProcessValidationError, match="Missing strategy"):
        ep.fit_strategy_mixture(xf, ["missing_feature"])
    with pytest.raises(ep.EyeProcessValidationError, match="Insufficient complete"):
        ep.fit_strategy_mixture(xf, ["gaze_feature"], centers=20)

    def fail_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("blocked sklearn")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", fail_sklearn)
        with pytest.raises(ep.EyeProcessBackendError, match="scikit-learn"):
            ep.fit_strategy_mixture(xf, ["gaze_feature"], centers=2)


def test_ez_diffusion_and_missing_process_validation_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="Missing required"):
        ep.estimate_ez_diffusion(pd.DataFrame({"score": [1]}), by="item_id")

    tiny = pd.DataFrame(
        {
            "item_id": ["I1", "I2", "I2", "I2"],
            "score": [1.0, 0.0, 1.0, 0.0],
            "response_time": [1.0, 1.0, 1.0, 1.0],
        }
    )
    ez = ep.estimate_ez_diffusion(tiny, by="item_id")
    assert set(ez["status"]) <= {"insufficient_data", "undefined_at_chance"}

    chance = pd.DataFrame(
        {
            "item_id": ["I1"] * 4,
            "score": [0.0, 1.0, 0.0, 1.0],
            "response_time": [1.0, 1.2, 1.4, 1.6],
        }
    )
    assert ep.estimate_ez_diffusion(chance, by="item_id").loc[0, "status"] == "undefined_at_chance"

    x = _dataset_with_features()
    with pytest.raises(ep.EyeProcessValidationError, match="absent"):
        ep.model_missing_process(x, "missing_feature")
    with pytest.raises(ep.EyeProcessBackendError):
        ep.model_missing_process(x, "gaze_feature", engine="lme4")
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.model_missing_process(x, "gaze_feature", engine="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="absent"):
        ep.sensitivity_missing_process(x, "missing_feature", "score ~ 1")
