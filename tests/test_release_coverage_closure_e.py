from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.frontier_08 as fr
import eyeprocesspy.importers as imp
import eyeprocesspy.plots_functional_pupil as pf
import eyeprocesspy.plots_irt as pi
import eyeprocesspy.sensitivity_08 as se


def _close(ax):
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_frontier_invalid_engines_fold_builder_and_contract_guards():
    class BadFrame:
        def __iter__(self):
            raise RuntimeError("cannot coerce")
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        fr._df(BadFrame())

    for fn in (
        fr.fit_kde_latent_distribution_irt,
        fr.fit_persistence_gaze_diffusion_irt,
        fr.fit_nonignorable_missing_irt,
        fr.fit_crossclassified_process_irt_mhrm,
    ):
        with pytest.raises(ep.EyeProcessValidationError, match="engine"):
            fn([[1]], engine=3)

    structured = pd.DataFrame({"person_id": [1, 2, 3, 4], "fold": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        fr.prepare_structured_unstructured_process_features(structured, fold="missing")
    reg = fr.prepare_structured_unstructured_process_features(structured, fold="fold")
    assert reg.status == "fold_registry_requires_builder"
    with pytest.raises(ep.EyeProcessValidationError, match="builder"):
        fr.prepare_structured_unstructured_process_features(structured, fold="fold", builder=3)
    one = structured.assign(fold="a")
    with pytest.raises(ep.EyeProcessValidationError, match="At least two"):
        fr.prepare_structured_unstructured_process_features(one, fold="fold", builder=lambda *a: a)

    unstructured = pd.DataFrame({"fold": ["a", "b"], "text": ["x", "y"]})
    out = fr.prepare_structured_unstructured_process_features(
        structured,
        unstructured=unstructured,
        fold="fold",
        builder=lambda train_s, train_u, test_s, test_u, v: {
            "fold": v, "n_train": len(train_s), "n_test_u": len(test_u)
        },
    )
    assert set(out.folds) == {"a", "b"}

    with pytest.raises(ep.EyeProcessValidationError, match="eye_gated"):
        fr.audit_frontier_model_contract({})
    gated = fr.fit_kde_latent_distribution_irt([[1]])
    with pytest.raises(ep.EyeProcessValidationError, match="mapping"):
        fr.audit_frontier_model_contract(gated, evidence=[])


def test_sensitivity_guard_and_backend_boundaries():
    class BadFrame:
        def __iter__(self):
            raise RuntimeError("bad")
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        se._df(BadFrame())
    assert np.isnan(se._mean(["bad", np.nan]))

    with pytest.raises(ep.EyeProcessValidationError, match="n_classes"):
        se.fit_mixture_irt_process_classes([[1]], n_classes=3)
    with pytest.raises(ep.EyeProcessBackendError):
        se.fit_mixture_irt_process_classes([[1]], n_classes=2)

    cm = pd.DataFrame({"person_id": ["P1", "P1"], "class": [1, 2]})
    proc = pd.DataFrame({"person_id": ["P1"], "x": [1]})
    with pytest.raises(ep.EyeProcessValidationError, match="one assignment"):
        se.map_latent_classes_to_process_profiles(cm, proc, process_features=["x"])
    with pytest.raises(ep.EyeProcessValidationError, match="at least one process feature"):
        se.map_latent_classes_to_process_profiles(pd.DataFrame({"person_id": ["P1"], "class": [1]}), proc)

    with pytest.raises(ep.EyeProcessValidationError, match="NPtest"):
        se.audit_nonparametric_rasch([[1]], methods=[])
    with pytest.raises(ep.EyeProcessValidationError, match="positive"):
        se.audit_nonparametric_rasch([[1]], n=0)
    for alpha, maxstep, msg in ((0.0, 5, "alpha"), (1.0, 5, "alpha"), (0.05, 0, "maxstep")):
        with pytest.raises(ep.EyeProcessValidationError, match=msg):
            se.audit_item_reduction_sensitivity(None, alpha=alpha, maxstep=maxstep)

    data = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ep.EyeProcessValidationError, match="variable"):
        se.biometric_imputation_sensitivity(data, [])
    with pytest.raises(ep.EyeProcessValidationError, match="positive"):
        se.biometric_imputation_sensitivity(data, ["x"], m=0)
    ok = se.biometric_imputation_sensitivity(data, ["x"], methods=["mice", "bad", "mice"])
    assert ok.status.method.tolist() == ["mice"]

    with pytest.raises(ep.EyeProcessValidationError, match="same number"):
        se.fit_process_rasch_tree([[1], [0]], pd.DataFrame({"z": [1]}))
    with pytest.raises(ep.EyeProcessValidationError, match="splitting covariate"):
        se.fit_process_rasch_tree([[1]], pd.DataFrame(index=[0]))
    with pytest.raises(ep.EyeProcessValidationError, match="maxit"):
        se.fit_process_rasch_tree([[1]], pd.DataFrame({"z": [1]}), maxit=0)
    with pytest.raises(ep.EyeProcessValidationError, match="method"):
        se.compare_bayesian_process_models(1, 2, method="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="at least two"):
        se.compare_bayesian_process_models(1)


def test_importer_private_mapping_residuals(tmp_path):
    with pytest.raises(ValueError, match="Unsupported time"):
        imp._time_multiplier("fortnight")
    f = tmp_path / "x.txt"
    f.write_text("a;b\n1;2\n", encoding="utf-8")
    assert imp._read_delimited(f).iloc[0, 1] == 2

    d = pd.DataFrame({"a": ["1,5", "bad"], "b": [1, 2]})
    assert imp._map_column(d, {"x": []}, "x", default=7).tolist() == [7, 7]
    assert imp._map_column(d, {"x": ["b", "a"]}, "x").tolist() == [1, 2]
    with pytest.raises(ValueError, match="does not exist"):
        imp._map_column(d, {"x": "missing"}, "x")
    vals = imp._safe_numeric(d.a)
    assert vals.iloc[0] == pytest.approx(1.5) and pd.isna(vals.iloc[1])

    with pytest.raises(TypeError, match="mapping"):
        imp.validate_eye_mapping([], data=d)
    with pytest.raises(TypeError, match="DataFrame"):
        imp.validate_eye_mapping({"timestamp": "a", "x": "a", "y": "a"}, data=[])
    mapping = {
        "timestamp": "a", "x": "b", "y": "b",
        "nested": {"p": "a"}, "seq": ["a", "b"], "none": None,
    }
    assert imp.validate_eye_mapping(mapping, data=d) is mapping
    with pytest.raises(TypeError, match="DataFrame"):
        imp.infer_eye_mapping([])


def test_functional_pupil_plot_legacy_trajectory_and_posterior_paths():
    legacy_features = pd.DataFrame({
        "participant_id": ["P1", "P1"], "trial_id": ["T1", "T1"], "item_id": ["I1", "I1"],
        "feature_name": ["b1", "b2"], "value": [1.0, 2.0]
    })
    legacy = SimpleNamespace(legacy=True, data={"features": legacy_features}, feature_names=["b1", "b2"])
    ax = pf.plot_eye_functional_pupil_irt(legacy)
    assert len(ax.lines) == 1
    _close(ax)

    traj = pd.DataFrame({"trial_index": [1, 1, 2], ".time": [0, 1, 0], "pupil_adjusted": [1.0, 1.2, 0.9]})
    modern = SimpleNamespace(legacy=False, data=SimpleNamespace(data=traj), trial_coefficients=pd.DataFrame({"b1": [1.0]}), feature_names=["b1"], model=None)
    ax = pf.plot_eye_functional_pupil_irt(modern, type="trajectories")
    assert len(ax.lines) == 2
    _close(ax)

    with pytest.raises(ValueError, match="Stan"):
        pf.plot_eye_functional_pupil_irt(modern, type="posterior")
    stan_summary = pd.DataFrame({"variable": ["theta_loading[1]", "other"], "mean": [0.5, 0.2]})
    stan = SimpleNamespace(eyeprocess_class="eye_functional_pupil_stan", summary=stan_summary)
    modern.model = stan
    ax = pf.plot_eye_functional_pupil_irt(modern, type="posterior")
    assert len(ax.lines) == 1
    _close(ax)

    sens = SimpleNamespace(results=pd.DataFrame({"parameter": ["p1", "p2"], "specification": ["a", "b"], "estimate": [1.0, 2.0]}))
    ax = pf.plot_eye_functional_pupil_sensitivity(sens, parameter=["p2"])
    assert len(ax.gp3_data) == 1
    _close(ax)


def test_irt_plot_empty_and_validation_branches():
    import matplotlib.pyplot as plt
    base_ax = plt.subplots()[1]
    assert pi._ax(base_ax) is base_ax
    assert pi._df([[1]]).shape == (1, 1)
    with pytest.raises(ep.EyeProcessValidationError, match="mapping"):
        pi._mapping([])
    assert pi._empty(base_ax, "empty").get_title() == "empty"
    _close(base_ax)

    empty = pd.DataFrame()
    empty_calls = [
        lambda: pi.plot_eye_irt_information_profile(empty),
        lambda: pi.plot_eye_irt_test_characteristic_curve(empty),
        lambda: pi.plot_eye_irt_sparse_design_audit({"person_counts": []}),
        lambda: pi.plot_eye_irt_q3_matrix([]),
        lambda: pi.plot_eye_irt_item_fit(empty),
        lambda: pi.plot_eye_irt_person_fit(pd.DataFrame({"infit": [np.nan]})),
        lambda: pi.plot_eye_irt_score_uncertainty({}),
        lambda: pi.plot_eye_irt_adaptive_trace(empty),
        lambda: pi.plot_eye_irt_link_stability({"table": []}),
        lambda: pi.plot_eye_irt_dif_curve(empty),
        lambda: pi.plot_eye_irt_dtf_curve(empty),
        lambda: pi.plot_eye_irt_process_alignment({"table": [], "correlations": []}),
        lambda: pi.plot_eye_irt_recovery_result({"estimates": []}),
        lambda: pi.plot_eye_irt_sbc_evidence({"diagnostics": {"counts": []}}),
        lambda: pi.plot_eye_cdm_qmatrix_audit({"attribute": []}),
        lambda: pi.plot_eye_irt_bank_coverage({"curve": []}),
        lambda: pi.plot_eye_irt_targeting_gap({"table": []}),
        lambda: pi.plot_eye_irt_prior_sensitivity({"table": []}),
    ]
    for call in empty_calls:
        ax = call()
        _close(ax)

    for call in (
        lambda: pi.plot_eye_irt_item_fit([], statistic="bad"),
        lambda: pi.plot_eye_irt_person_fit([], statistic="bad"),
        lambda: pi.plot_eye_irt_link_stability({}, parameter="bad"),
        lambda: pi.plot_eye_irt_process_alignment({}, parameter="x"),
        lambda: pi.plot_eye_irt_recovery_result({}, parameter="x"),
    ):
        with pytest.raises(ep.EyeProcessValidationError):
            call()
