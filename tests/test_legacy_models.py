from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import eyeprocesspy as ep

LEGACY_EXPORTS = [
    "align_response_matrices", "check_local_dependence", "fit_accuracy_rt", "fit_dif",
    "fit_dynamic_aoi_model", "fit_explanatory_irt", "fit_irt", "fit_joint_process_model",
    "fit_shared_process_factor", "item_parameters", "model_data", "model_fit_statistics",
    "person_scores", "response_matrix", "response_time_matrix", "parameter_recovery",
    "power_process_simulation", "simulate_eye_dataset", "simulate_process_irt",
    "estimate_ez_diffusion", "fit_gaze_informed_irt", "fit_gaze_weighted_choice",
    "fit_multimodal_irt", "fit_process_irt", "fit_pupil_informed_irt", "fit_strategy_mixture",
    "functional_pupil_features", "model_missing_process", "process_irt_diagnostics",
    "process_irt_spec", "sensitivity_missing_process",
]


def _dataset_with_features(seed=501):
    x = ep.simulate_eye_dataset(n_person=16, n_item=6, sampling_rate=15, trial_duration=.35, samples_per_trial=8, seed=seed)
    r = x["responses"].reset_index(drop=True)
    rows = []
    for i, row in r.iterrows():
        for j, (name, value) in enumerate([
            ("gaze_feature", np.sin(i * .37) + (i % 5) * .03),
            ("pupil_feature", np.cos(i * .29) + .1 * np.log(float(row["response_time"]))),
        ]):
            if name == "gaze_feature" and i % 13 == 0:
                value = np.nan
            rows.append({
                "feature_id": f"f{i:04d}_{j}", "recording_id": row["recording_id"],
                "participant_id": row["participant_id"], "trial_id": row["trial_id"],
                "item_id": row["item_id"], "stimulus_id": pd.NA, "aoi_id": pd.NA,
                "feature_name": name, "value": value, "unit": "arbitrary",
                "level": "trial", "method": "test_fixture", "parameters": "seed=501",
            })
    x["features"] = ep.standardize_eye_table(pd.DataFrame(rows), "features")
    return x


def test_legacy_export_and_signature_smoke():
    assert len(LEGACY_EXPORTS) == 31
    assert all(callable(getattr(ep, n, None)) for n in LEGACY_EXPORTS)
    signatures = json.loads((Path(__file__).resolve().parents[1] / "reference" / "R_SIGNATURES.json").read_text())
    for name in LEGACY_EXPORTS:
        r_names = [a["name"] for a in signatures[name]["args"]]
        r_names = ["kwargs" if n == "..." else ("scale_" if n == "scale." else n) for n in r_names]
        assert list(inspect.signature(getattr(ep, name)).parameters) == r_names, name


def test_simulation_and_response_matrix_contracts():
    a = ep.simulate_eye_dataset(n_person=8, n_item=5, sampling_rate=20, trial_duration=.5, seed=5)
    b = ep.simulate_eye_dataset(n_person=8, n_item=5, sampling_rate=20, trial_duration=.5, seed=5)
    pd.testing.assert_frame_equal(a["responses"], b["responses"])
    assert a["responses"].participant_id.nunique() == 8
    assert a["responses"].item_id.nunique() == 5
    assert np.isfinite(pd.to_numeric(a["responses"].response_time)).all()
    y = ep.response_matrix(a); rt = ep.response_time_matrix(a)
    assert y.shape == (8, 5) and rt.shape == (8, 5)
    aligned = ep.align_response_matrices(y, rt)
    assert aligned.Y.shape == aligned.RT.shape == (8, 5)

    dup = a.copy()
    dup["responses"] = pd.concat([dup["responses"], dup["responses"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ep.EyeProcessValidationError, match="Duplicate participant-item"):
        ep.response_matrix(dup)
    assert ep.response_matrix(dup, duplicate="last").shape == (8, 5)
    assert ep.response_matrix(dup, duplicate="mean").shape == (8, 5)

    proc = ep.simulate_process_irt(n_person=8, n_item=5, seed=6)
    assert len(proc.data) == 40
    assert {"data", "truth"} <= set(proc)


def test_rasch_glm_two_stage_dif_and_model_extraction():
    x = ep.simulate_eye_dataset(n_person=20, n_item=6, sampling_rate=10, trial_duration=.3, samples_per_trial=5, seed=9)
    model = ep.fit_irt(x, engine="rasch_glm")
    assert model.eyeprocess_class == "eyeprocess_model" and model.engine == "rasch_glm"
    assert len(ep.item_parameters(model)) >= 5
    assert len(ep.person_scores(model)) >= 19
    fit_stats = ep.model_fit_statistics(model)
    assert fit_stats.loc[0, "nobs"] == 120
    two = ep.fit_accuracy_rt(x, engine="two_stage")
    assert np.isfinite(two.fit.ability_speed_correlation)

    group_by_person = {p: ("A" if i < 10 else "B") for i, p in enumerate(sorted(x["responses"].participant_id.unique()))}
    group = x["responses"].participant_id.map(group_by_person).to_numpy()
    dif = ep.fit_dif(x, group=group, engine="logistic")
    assert len(dif.fit) == 6
    assert set(dif.fit.status) <= {"estimated", "insufficient_data", "fit_failed"}


def test_feature_models_process_irt_strategy_and_missingness():
    x = _dataset_with_features()
    d = ep.model_data(x, include_features=True)
    assert {"gaze_feature", "pupil_feature"} <= set(d.columns)

    glm = ep.fit_explanatory_irt(x, "score ~ gaze_feature", engine="glm")
    assert glm.engine == "glm"
    spec = ep.process_irt_spec(gaze_features=("gaze_feature",), pupil_features=("pupil_feature",))
    proc = ep.fit_process_irt(x, spec, engine="glm")
    assert proc.model_type == "explanatory_irt"
    assert ep.fit_gaze_informed_irt(x, gaze_features=("gaze_feature",), engine="glm").engine == "glm"
    assert ep.fit_pupil_informed_irt(x, pupil_features=("pupil_feature",), engine="glm").engine == "glm"
    assert ep.fit_multimodal_irt(x, gaze_features=("gaze_feature",), pupil_features=("pupil_feature",), engine="glm").engine == "glm"
    assert ep.fit_gaze_weighted_choice(x, dwell_features=("gaze_feature",), engine="glm").engine == "glm"

    shared = ep.fit_shared_process_factor(x, features=("gaze_feature", "pupil_feature"), n_factors=1)
    assert shared.model.model_type == "shared_process_factor"
    assert "process_factor_1" in set(shared.data["features"].feature_name.dropna())

    strategy = ep.fit_strategy_mixture(x, features=("gaze_feature", "pupil_feature"), centers=2, seed=1)
    assert strategy.model.experimental
    assert "strategy_class" in set(strategy.data["features"].feature_name.dropna())

    missing = ep.model_missing_process(x, "gaze_feature", engine="glm")
    assert missing.model_type == "missing_process_model"
    sensitivity = ep.sensitivity_missing_process(x, "gaze_feature", "score ~ gaze_feature")
    assert {"complete_case", "median_indicator"} <= set(sensitivity.fits)

    diag = ep.process_irt_diagnostics(proc)
    assert "fit" in diag and "warnings" in diag


def test_joint_dynamic_diffusion_functional_pupil_and_recovery():
    x = _dataset_with_features(seed=502)
    joint = ep.fit_joint_process_model(x, "score ~ gaze_feature", "log_response_time ~ gaze_feature", engine="separate")
    assert joint.experimental and joint.engine == "separate"
    dyn = ep.fit_dynamic_aoi_model(x, source="samples", smoothing=.5)
    np.testing.assert_allclose(dyn.fit.probabilities.sum(axis=1), 1.0)

    ez = ep.estimate_ez_diffusion(x, by=("item_id",))
    assert len(ez) == 6
    assert {"drift_rate", "boundary_separation", "nondecision_time", "status"} <= set(ez)

    f = ep.functional_pupil_features(x, df=3, append=False)
    assert not f.empty
    assert f.feature_name.astype(str).str.startswith("pupil_basis_").all()

    recovery = ep.parameter_recovery(
        simulator=lambda truth=1.5: {"truth": truth},
        estimator=lambda sim: {"estimate": sim["truth"] + .1},
        extractor=lambda fit: {"theta": fit["estimate"]},
        truth_extractor=lambda sim: {"theta": sim["truth"]},
        replications=3,
        truth=1.5,
    )
    assert len(recovery) == 3
    np.testing.assert_allclose(recovery.bias, .1)
    ax = ep.plot_eye_parameter_recovery(recovery)
    try:
        assert hasattr(ax, "gp3_data") and len(ax.gp3_data) == 3
    finally:
        plt.close(ax.figure)


def test_external_r_engine_paths_are_explicit_gates():
    x = ep.simulate_eye_dataset(n_person=8, n_item=5, sampling_rate=10, samples_per_trial=5, seed=77)
    for call in [
        lambda: ep.fit_irt(x, engine="mirt"),
        lambda: ep.fit_irt(x, engine="TAM"),
        lambda: ep.fit_accuracy_rt(x, engine="LNIRT"),
        lambda: ep.fit_explanatory_irt(x, "score ~ 1", engine="lme4"),
        lambda: ep.fit_explanatory_irt(x, "score ~ 1", engine="brms"),
    ]:
        with pytest.raises(ep.EyeProcessBackendError):
            call()


def test_power_process_simulation_small_smoke():
    out = ep.power_process_simulation(n_person=8, n_item=4, effect=.2, replications=2, seed=10)
    assert len(out) == 1
    assert 0 <= out.power.iloc[0] <= 1
