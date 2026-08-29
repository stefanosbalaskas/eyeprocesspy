from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError
from eyeprocesspy.irt import EyeResult

FROZEN = [
    "audit_candidate_item_bank", "audit_process_external_validity", "audit_visual_context_dependence",
    "compare_process_criterion_models", "compare_process_profile_solutions", "compare_visual_context_irt",
    "context_factor_effects", "fit_item_parameter_seed_model", "fit_multiblock_process_map",
    "fit_process_profile_mixture", "fit_visual_context_irt", "incremental_process_validity",
    "multiblock_contributions", "multiblock_person_coordinates", "multiblock_variable_coordinates",
    "predict_item_parameter_priors", "process_criterion_associations", "process_feature_blocks",
    "process_profile_probabilities", "process_profile_summary", "visual_context_registry",
    "audit_frontier_model_contract", "fit_crossclassified_process_irt_mhrm", "fit_kde_latent_distribution_irt",
    "fit_nonignorable_missing_irt", "fit_persistence_gaze_diffusion_irt", "prepare_structured_unstructured_process_features",
    "audit_biometric_imputation", "audit_item_reduction_sensitivity", "audit_nonparametric_rasch",
    "biometric_imputation_sensitivity", "compare_bayesian_process_models", "fit_mixture_irt_process_classes",
    "fit_process_rasch_tree", "map_latent_classes_to_process_profiles",
    "audit_3pl_process_signatures", "bayesian_process_diagnostic_flags", "bayesian_process_diagnostics_dashboard",
    "fit_gaze_anchored_3pl_audit", "gaze_anchored_3pl_alignment",
]


def test_all_frozen_08_irt_exports_resolve():
    missing = [n for n in FROZEN if not callable(getattr(ep, n, None))]
    assert missing == []


def test_visual_context_registry_identifies_shared_contexts():
    m = pd.DataFrame({"item_id": [f"i{i}" for i in range(1, 9)],
                      "screen_id": ["screen_A"] * 4 + [f"u{i}" for i in range(5, 9)]})
    r = ep.visual_context_registry(m, context="screen_id")
    assert r.eyeprocess_class == "eye_visual_context_registry"
    assert r.mapping["shared_context"].any()
    X = pd.DataFrame(np.random.default_rng(6).binomial(1, .6, size=(200, 8)), columns=m.item_id)
    with pytest.raises(EyeProcessBackendError, match="mirt"):
        ep.fit_visual_context_irt(X, r)


def test_multiblock_mapping_has_reference_fallback():
    rng = np.random.default_rng(7)
    d = pd.DataFrame({
        "person_id": [f"P{i}" for i in range(1, 31)], "theta": rng.normal(size=30),
        "accuracy": rng.random(30), "dwell": rng.normal(700, 70, 30), "entropy": rng.random(30),
        "pupil": rng.normal(size=30), "validity": rng.uniform(.8, 1, 30),
    })
    b = ep.process_feature_blocks(d, {"Psychometric": ["theta", "accuracy"], "Gaze": ["dwell", "entropy"],
                                      "Pupil": ["pupil"], "Quality": ["validity"]}, id="person_id")
    fit = ep.fit_multiblock_process_map(b, engine="pca_block_scaled")
    assert fit.eyeprocess_class == "eye_multiblock_process_map"
    assert len(ep.multiblock_person_coordinates(fit)) == 30
    assert len(ep.multiblock_variable_coordinates(fit)) == 6
    assert len(ep.multiblock_contributions(fit)) == 4


def test_process_profile_reference_is_explicitly_descriptive():
    rng = np.random.default_rng(8)
    d = pd.DataFrame({"person_id": [f"P{i}" for i in range(1, 51)],
                      "a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
    fit = ep.fit_process_profile_mixture(d, ["a", "b", "c"], k=3, engine="kmeans_reference")
    assert fit.eyeprocess_class == "eye_process_profile_mixture"
    assert "not_finite_mixture" in fit.status
    assert len(ep.process_profile_probabilities(fit)) == 50
    assert len(ep.process_profile_summary(fit)) == 3
    cmp = ep.compare_process_profile_solutions(d, ["a", "b", "c"], [2, 3, 4])
    assert cmp.k.tolist() == [2, 3, 4]


def test_external_validity_and_item_seed_reference_algorithms():
    rng = np.random.default_rng(81)
    n = 50
    d = pd.DataFrame({"criterion": rng.normal(size=n), "base": rng.normal(size=n),
                      "dwell": rng.normal(size=n), "pupil": rng.normal(size=n)})
    a = ep.audit_process_external_validity(d, "criterion", ["dwell", "pupil"], ["base"])
    assert a.eyeprocess_class == "eye_process_external_validity"
    assert set(ep.process_criterion_associations(a).predictor) == {"dwell", "pupil"}
    assert ep.incremental_process_validity(a).shape == (1, 3)
    assert not ep.compare_process_criterion_models(a).empty

    item = pd.DataFrame({"irt_difficulty": rng.normal(size=24), "irt_discrimination": rng.uniform(.5, 2, 24),
                         "words": rng.uniform(20, 100, 24), "visual": rng.normal(size=24)})
    sm = ep.fit_item_parameter_seed_model(item, predictors=["words", "visual"], engine="lm")
    pred = ep.predict_item_parameter_priors(sm, item[["words", "visual"]].iloc[:5])
    assert {"predicted_pre_pilot_difficulty", "predicted_pre_pilot_discrimination"}.issubset(pred.columns)
    audit = ep.audit_candidate_item_bank(sm, item[["words", "visual"]].iloc[:5])
    assert audit.eyeprocess_class == "eye_candidate_item_bank_audit"
    assert "review_required" in audit.table


def test_frontier_estimators_remain_gated_without_external_engines():
    X = np.random.default_rng(9).binomial(1, .5, size=(20, 5))
    k = ep.fit_kde_latent_distribution_irt(X)
    p = ep.fit_persistence_gaze_diffusion_irt(pd.DataFrame({"y": range(5)}))
    n = ep.fit_nonignorable_missing_irt(pd.DataFrame({"y": range(5)}))
    c = ep.fit_crossclassified_process_irt_mhrm(pd.DataFrame({"y": range(5)}))
    for x in (k, p, n, c):
        assert x.eyeprocess_class == "eye_gated_process_model"
        assert x.status == "gated"
        assert ep.audit_frontier_model_contract(x).status.iloc[0] == "remains_gated"
    k2 = ep.fit_kde_latent_distribution_irt(X, engine=lambda z: {"n": len(z)})
    assert ep.audit_frontier_model_contract(k2, {"recovery": 1, "misspec": 1, "benchmark": 1}).status.iloc[0] == "candidate_for_validation_review"


def test_structured_unstructured_contract_preserves_fold_locality():
    d = pd.DataFrame({"person_id": [f"P{i}" for i in range(12)], "fold": np.repeat([1, 2, 3], 4), "x": np.arange(12)})
    x = ep.prepare_structured_unstructured_process_features(d, fold="fold")
    assert x.eyeprocess_class == "eye_structured_unstructured_process_features"
    assert "training fold only" in x.contract["leakage_rule"]
    built = ep.prepare_structured_unstructured_process_features(
        d, fold="fold", builder=lambda train_s, train_u, test_s, test_u, fold_value: {"fold": fold_value, "n_train": len(train_s)})
    assert built.status == "fold_local_representation_built" and len(built.folds) == 3


def test_latent_class_process_alignment_and_imputation_contracts():
    rng = np.random.default_rng(13)
    cls = pd.DataFrame({"person_id": [f"P{i}" for i in range(1, 21)], "class": ["A"] * 10 + ["B"] * 10})
    proc = pd.DataFrame({"person_id": np.repeat(cls.person_id, 2), "dwell": rng.normal(size=40), "pupil": rng.normal(size=40)})
    x = ep.map_latent_classes_to_process_profiles(cls, proc, process_features=["dwell", "pupil"])
    assert x.eyeprocess_class == "eye_latent_process_alignment"
    assert len(x.summary) == 2
    assert "cannot prove" in x.caveat
    imp = ep.biometric_imputation_sensitivity(pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]}), ["a", "b"], methods=[])
    assert imp.eyeprocess_class == "eye_biometric_imputation_sensitivity"
    assert len(imp.missingness) == 2 and len(imp.results) == 0
    assert ep.audit_biometric_imputation(pd.DataFrame({"a": [1, np.nan], "b": [2, 3]}), ["a", "b"], methods=[]).eyeprocess_class == "eye_biometric_imputation_sensitivity"


def test_exact_r_specific_sensitivity_engines_are_explicit_gates():
    X = np.ones((100, 6))
    with pytest.raises(EyeProcessBackendError, match="mirt"):
        ep.fit_mixture_irt_process_classes(X)
    with pytest.raises(EyeProcessBackendError, match="eRm"):
        ep.audit_nonparametric_rasch(X)
    with pytest.raises(EyeProcessBackendError, match="psychotree"):
        ep.fit_process_rasch_tree(X, pd.DataFrame({"dwell": np.arange(100)}))
    with pytest.raises(EyeProcessBackendError, match="brms"):
        ep.compare_bayesian_process_models(object(), object())


def test_bayesian_and_3pl_diagnostics_helpers_preserve_review_semantics():
    posterior = pd.DataFrame({"model": ["m1", "m1"], "variable": ["a", "b"], "rhat": [1.0, 1.02],
                              "ess_bulk": [800, 200], "ess_tail": [900, 300]})
    dash = EyeResult({"posterior": posterior, "loo_table": pd.DataFrame()}, eyeprocess_class="eye_bayesian_process_dashboard")
    flags = ep.bayesian_process_diagnostic_flags(dash)
    assert flags.review_required.tolist() == [False, True]
    with pytest.raises(EyeProcessBackendError, match="brms"):
        ep.bayesian_process_diagnostics_dashboard(object())
    with pytest.raises(EyeProcessBackendError, match="mirt"):
        ep.fit_gaze_anchored_3pl_audit(np.ones((120, 6)))

    items = pd.DataFrame({"item_id": [f"i{i}" for i in range(1, 6)], "lower_asymptote": [.1, .15, .2, .3, .4],
                          "rt_ms": [100, 200, 300, 400, 500], "ttff_ms": [50, 60, 70, 80, 90],
                          "a": [1, 1.1, .9, 1.2, 1.3], "b": [-1, -.5, 0, .5, 1]})
    audit = EyeResult({"item_parameters": items, "alignment": pd.DataFrame({"feature": ["rt_ms"], "correlation": [.2]}),
                       "process_features": ["rt_ms", "ttff_ms"]}, eyeprocess_class="eye_gaze_anchored_3pl_audit")
    sig = ep.audit_3pl_process_signatures(audit)
    assert {"process_review_count", "review_label"}.issubset(sig.columns)
    assert ep.gaze_anchored_3pl_alignment(audit).shape[0] == 1


def test_08_plot_counterparts_return_axes():
    mpl = pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt
    gated = ep.fit_kde_latent_distribution_irt(np.ones((20, 5)))
    ax = ep.plot_eye_gated_process_model(gated); assert ax is not None; plt.close(ax.figure)
    cls = pd.DataFrame({"person_id": ["P1", "P2"], "class": ["A", "B"]})
    proc = pd.DataFrame({"person_id": ["P1", "P2"], "dwell": [1.0, 2.0]})
    align = ep.map_latent_classes_to_process_profiles(cls, proc, process_features=["dwell"])
    ax = ep.plot_eye_latent_process_alignment(align); assert len(ax.lines) == 2; plt.close(ax.figure)
    imp = ep.biometric_imputation_sensitivity(pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 2]}), ["a", "b"], methods=[])
    ax = ep.plot_eye_biometric_imputation_sensitivity(imp); assert len(ax.patches) == 2; plt.close(ax.figure)
    dash = EyeResult({"posterior": pd.DataFrame({"model": ["m1", "m1"], "rhat": [1.0, 1.02], "ess_bulk": [800, 200]}),
                      "loo_table": pd.DataFrame()}, eyeprocess_class="eye_bayesian_process_dashboard")
    ax = ep.plot_eye_bayesian_process_dashboard(dash, type="rhat"); assert ax is not None; plt.close(ax.figure)
    items = pd.DataFrame({"item_id": ["i1", "i2"], "lower_asymptote": [.1, .2], "a": [1, 1.2], "b": [-.2, .4], "rt_ms": [300, 400]})
    a3 = EyeResult({"item_parameters": items, "process_features": ["rt_ms"]}, eyeprocess_class="eye_gaze_anchored_3pl_audit")
    ax = ep.plot_eye_gaze_anchored_3pl_audit(a3); assert len(ax.patches) == 2; plt.close(ax.figure)


def test_context_structure_plot_counterparts_return_axes():
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(108)
    d = pd.DataFrame({"person_id": [f"P{i}" for i in range(30)], "a": rng.normal(size=30),
                      "b": rng.normal(size=30), "c": rng.normal(size=30), "criterion": rng.normal(size=30)})
    blocks = ep.process_feature_blocks(d, {"A": ["a", "b"], "B": ["c"]}, id="person_id")
    mb = ep.fit_multiblock_process_map(blocks, engine="pca_block_scaled")
    for typ in ["individuals", "variables", "blocks"]:
        ax = ep.plot_eye_multiblock_process_map(mb, type=typ); assert ax is not None; plt.close(ax.figure)
    prof = ep.fit_process_profile_mixture(d, ["a", "b", "c"], k=2, engine="kmeans_reference")
    for typ in ["profiles", "posterior", "scatter"]:
        ax = ep.plot_eye_process_profile_mixture(prof, type=typ); assert ax is not None; plt.close(ax.figure)
    valid = ep.audit_process_external_validity(d, "criterion", ["a", "b"])
    for typ in ["associations", "incremental", "observed_fitted", "residuals"]:
        ax = ep.plot_eye_process_external_validity(valid, type=typ); assert ax is not None; plt.close(ax.figure)
    items = pd.DataFrame({"irt_difficulty": rng.normal(size=20), "irt_discrimination": rng.uniform(.5, 2, 20),
                          "a": rng.normal(size=20), "b": rng.normal(size=20)})
    seed = ep.fit_item_parameter_seed_model(items, predictors=["a", "b"], engine="lm")
    ax = ep.plot_eye_item_parameter_seed(seed); assert ax is not None; plt.close(ax.figure)
    audit = ep.audit_candidate_item_bank(seed, items[["a", "b"]].iloc[:6])
    ax = ep.plot_eye_candidate_item_bank_audit(audit); assert ax is not None; plt.close(ax.figure)
    meta = pd.DataFrame({"item_id": ["i1", "i2", "i3", "i4"], "screen_id": ["s", "s", "s", "u"]})
    registry = ep.visual_context_registry(meta, context="screen_id")
    vfit = EyeResult({"registry": registry}, eyeprocess_class="eye_visual_context_irt")
    ax = ep.plot_eye_visual_context_irt(vfit, type="context_registry"); assert len(ax.patches) == 2; plt.close(ax.figure)
