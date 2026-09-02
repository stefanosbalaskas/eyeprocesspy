from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.advanced_process_irt_07 as ap


def _small_process_frame(response=(2, 2, 2, 2)):
    return pd.DataFrame(
        {
            "trial_id": ["t1", "t1", "t2", "t2"],
            "timestamp": [0.0, 1.0, 0.0, 1.0],
            "x": [0.0, 0.2, 1.0, 1.2],
            "y": [0.1, 0.0, 1.1, 1.0],
            "response": list(response),
            "participant_id": ["p1", "p1", "p2", "p2"],
            "item_id": ["i1", "i1", "i2", "i2"],
        }
    )


def test_hmm_validation_singleton_sequences_zero_iteration_and_accessor_guards():
    d = _small_process_frame()
    with pytest.raises(ep.EyeProcessValidationError, match="n_states"):
        ep.fit_process_hmm_irt(d, n_states=1)
    with pytest.raises(ep.EyeProcessValidationError, match="Too few"):
        ep.fit_process_hmm_irt(d.iloc[:1], n_states=2)

    singleton = pd.DataFrame(
        {
            "trial_id": ["a", "b"],
            "timestamp": [0.0, 0.0],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "response": [2, 2],
            "participant_id": ["p1", "p2"],
            "item_id": ["i1", "i2"],
        }
    )
    fit = ep.fit_process_hmm_irt(singleton, n_states=2, max_iter=0, seed=4)
    assert fit.response_model is None
    assert np.isnan(fit.logLik)
    assert fit.logLik_history.size == 0
    assert len(fit.occupancy) == 2

    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_hmm_irt"):
        ep.process_state_occupancy(SimpleNamespace())
    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_hmm_irt"):
        ep.process_state_transition_summary(SimpleNamespace())


def test_hmm_convergence_break_and_nonbinary_response_path():
    d = _small_process_frame()
    fit = ep.fit_process_hmm_irt(d, n_states=2, max_iter=4, tol=1e9, seed=2)
    assert fit.response_model is None
    assert 1 <= len(fit.logLik_history) <= 2


def test_cognitive_diagnosis_guards_external_and_process_surrogates():
    X = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=float)
    Q = np.eye(2)
    with pytest.raises(ep.EyeProcessValidationError, match="Q-matrix"):
        ep.fit_cognitive_diagnosis_process(X, np.eye(3), engine="external", external_engine=lambda **_: {})
    with pytest.raises(ep.EyeProcessBackendError, match="GDINA"):
        ep.fit_cognitive_diagnosis_process(X, Q)
    with pytest.raises(ep.EyeProcessBackendError, match="external_engine"):
        ep.fit_cognitive_diagnosis_process(X, Q, engine="other")

    base = ep.fit_cognitive_diagnosis_process(
        X, Q, engine="external", external_engine=lambda **kw: {"shape": kw["response_matrix"].shape}
    )
    assert base.process_summary is None

    pd_one = pd.DataFrame({"pid": [1, 1, 2, 2, 3, 3], "f1": [1, 2, 2, 3, 3, 4]})
    with pytest.raises(ep.EyeProcessValidationError, match="person_id"):
        ep.fit_cognitive_diagnosis_process(
            X,
            Q,
            process_data=pd_one,
            process_features=["f1"],
            engine="external",
            external_engine=lambda **_: {},
        )
    one = ep.fit_cognitive_diagnosis_process(
        X,
        Q,
        process_data=pd_one,
        process_features=["f1"],
        person_id="pid",
        engine="external",
        external_engine=lambda **_: {},
    )
    assert len(one.process_summary) == 3

    pd_two = pd_one.assign(f2=[4, 3, 3, 2, 2, 1])
    two = ep.fit_cognitive_diagnosis_process(
        X,
        Q,
        process_data=pd_two,
        process_features=["f1", "f2"],
        person_id="pid",
        engine="external",
        external_engine=lambda **_: {},
    )
    assert np.isfinite(two.process_summary["process_surrogate"]).all()


def test_latent_class_empty_features_and_nonbinary_response_skip_model():
    with pytest.raises(ep.EyeProcessValidationError, match="process_features"):
        ep.fit_latent_class_process_irt(pd.DataFrame({"response": [0]}), process_features=[])

    d = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "item_id": ["i1", "i2"] * 3,
            "f1": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            "f2": [2.0, 1.0, 0.5, -0.5, -1.0, -2.0],
            "response": [0, 2, 0, 2, 0, 2],
        }
    )
    fit = ep.fit_latent_class_process_irt(d, process_features=["f1", "f2"], n_classes=2, seed=3)
    assert fit.response_model is None


def test_crossclassified_validation_context_fixed_string_and_all_family_paths(monkeypatch):
    d = pd.DataFrame(
        {
            "participant_id": [f"p{i % 4}" for i in range(12)],
            "item_id": [f"i{i % 3}" for i in range(12)],
            "context": ["a", "b"] * 6,
            "x": np.linspace(-1, 1, 12),
            "cat": ["u", "v"] * 6,
            "gauss": np.linspace(0.1, 1.2, 12),
            "binary": [0, 1] * 6,
            "count": np.arange(12) % 4,
        }
    )
    with pytest.raises(ep.EyeProcessValidationError, match="family"):
        ep.fit_crossclassified_process_irt(d, "gauss", family="bad")

    calls = []

    def fake_ols(X, y):
        calls.append(("ols", np.asarray(y).copy()))
        return {"coefficients": np.zeros(X.shape[1]), "fitted": np.zeros(len(y))}

    def fake_logit(X, y):
        calls.append(("logit", np.asarray(y).copy()))
        return {"coefficients": np.zeros(X.shape[1]), "fitted": np.repeat(0.5, len(y))}

    monkeypatch.setattr(ap, "_ols_fit", fake_ols)
    monkeypatch.setattr(ap, "_logistic_fit", fake_logit)

    g = ep.fit_crossclassified_process_irt(d, "gauss", context="context", fixed="x", family="gaussian")
    b = ep.fit_crossclassified_process_irt(d, "binary", fixed=["x", "cat"], family="binomial")
    p = ep.fit_crossclassified_process_irt(d, "count", family="poisson")
    nb = ep.fit_crossclassified_process_irt(d, "count", family="negative_binomial")
    assert [g.family, b.family, p.family, nb.family] == ["gaussian", "binomial", "poisson", "negative_binomial"]
    assert [kind for kind, _ in calls].count("logit") == 1
    assert [kind for kind, _ in calls].count("ols") == 3


def _latent_space_stub(with_ids=True):
    obj = SimpleNamespace(
        eyeprocess_class="eye_latent_space_irt",
        person_coordinates=np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 1.5]]),
        item_coordinates=np.asarray([[0.0, 1.0], [1.0, 2.0]]),
    )
    if with_ids:
        obj.person_ids = ["p1", "p2", "p3"]
        obj.item_ids = ["i1", "i2"]
    return obj


def test_latent_space_map_and_similarity_all_guards_entities_and_empty_distance():
    with pytest.raises(ep.EyeProcessValidationError, match="eye_latent_space_irt"):
        ep.process_residual_map(SimpleNamespace())
    obj = _latent_space_stub()
    with pytest.raises(ep.EyeProcessValidationError, match="entity"):
        ep.process_residual_map(obj, entity="bad")
    assert set(ep.process_residual_map(obj, "person")["entity_type"]) == {"person"}
    assert set(ep.process_residual_map(obj, "item")["entity_type"]) == {"item"}
    assert set(ep.process_residual_map(obj, "both")["entity_type"]) == {"person", "item"}
    fallback = ep.process_residual_map(_latent_space_stub(with_ids=False), "both")
    assert fallback["entity_id"].notna().all()

    with pytest.raises(ep.EyeProcessValidationError, match="eye_latent_space_irt"):
        ep.validate_latent_space_process_similarity(SimpleNamespace(), [[1]])
    with pytest.raises(ep.EyeProcessValidationError, match="rows"):
        ep.validate_latent_space_process_similarity(obj, [[1], [2]], entity="person")
    person = ep.validate_latent_space_process_similarity(obj, [[0, 0], [1, 1], [2, 3]], entity="person")
    item = ep.validate_latent_space_process_similarity(obj, [[0, 1], [1, 2]], entity="item")
    assert np.isfinite(person.spearman_distance_correlation)
    assert np.isfinite(item.spearman_distance_correlation)

    singleton = SimpleNamespace(
        eyeprocess_class="eye_latent_space_irt",
        person_coordinates=np.asarray([[0.0, 0.0]]),
        item_coordinates=np.asarray([[0.0, 0.0]]),
    )
    one = ep.validate_latent_space_process_similarity(singleton, [[1.0, 2.0]])
    assert np.isnan(one.spearman_distance_correlation)


def test_equating_guards_mean_mean_stocking_lord_and_haebara_paths():
    ref = pd.DataFrame({"a": [1.0, 1.2, 0.9], "b": [-1.0, 0.0, 1.0]})
    new = pd.DataFrame({"a": [0.9, 1.1, 0.8], "b": [-0.8, 0.2, 1.2]})
    with pytest.raises(ep.EyeProcessValidationError, match="same number"):
        ep.equate_irt_scales(ref, new.iloc[:2])
    with pytest.raises(ep.EyeProcessValidationError, match="Unsupported"):
        ep.equate_irt_scales(ref, new, method="bad")

    mm = ep.equate_irt_scales(ref, new, method="mean-mean")
    sl = ep.equate_irt_scales(ref, new, method="stocking-lord")
    hb = ep.equate_irt_scales(ref, new, method="haebara", theta_grid=[-2, -1, 0, 1, 2])
    assert all(np.isfinite([mm.A, mm.B, sl.A, sl.B, hb.A, hb.B]))


def test_process_person_fit_guards():
    with pytest.raises(ep.EyeProcessValidationError, match="eye_joint_gaze_rt_irt"):
        ep.process_person_fit(SimpleNamespace(), data=pd.DataFrame())
    stub = SimpleNamespace(eyeprocess_class="eye_joint_gaze_rt_irt")
    with pytest.raises(ep.EyeProcessValidationError, match="Supply the data"):
        ep.process_person_fit(stub)


def test_process_dif_surrogate_aggregate_variants_and_adjusted_dif_guard():
    d = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0, 0.0],
        }
    )
    one = ep.process_dif_nuisance_surrogate(d, ["f1"], aggregate=False)
    two = ep.process_dif_nuisance_surrogate(d, ["f1", "f2"], aggregate=True)
    assert len(one) == 4 and len(two) == 2
    with pytest.raises(ep.EyeProcessValidationError, match="ability and group"):
        ep.audit_process_adjusted_dif(d, process_features=["f1"])


def test_sequence_coercion_empty_embedding_and_alignment_guard():
    assert ep.process_ngram_features("a>b>a", n=(1, 0, -1)).shape[0] == 1
    assert ep.process_ngram_features(["a", "b", "c"], n=(1,)).shape[0] == 1
    assert ep.process_ngram_features([["a", "b"], ["b", "c"]], n=(1,)).shape[0] == 2
    with pytest.raises(ep.EyeProcessValidationError, match="No n-grams"):
        ep.process_sequence_embedding([[]], n=(1, 2))

    d = pd.DataFrame({"response": [0, 1], "participant_id": ["p1", "p2"], "item_id": ["i1", "i2"]})
    with pytest.raises(ep.EyeProcessValidationError, match="row-for-row"):
        ep.fit_response_process_embedding_irt(d, [["a", "b"]], dimensions=1, n=(1,))


def test_gpirt_validation_external_comparison_and_custom_grid_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="two-dimensional"):
        ep.fit_gpirt([0, 1, 1])
    R = np.asarray([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float)
    with pytest.raises(ep.EyeProcessBackendError, match="external_engine"):
        ep.fit_gpirt(R, engine="external")
    ext = ep.fit_gpirt(R, engine="external", external_engine=lambda **kw: {"n": len(kw["response_matrix"])})
    assert ext.exact_gpirt is True
    with pytest.raises(ep.EyeProcessValidationError, match="engine"):
        ep.fit_gpirt(R, engine="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="spline-reference"):
        ep.compare_parametric_nonparametric_irf(R, ext)

    gp = ep.fit_gpirt(R, spline_df=3)
    cmp = ep.compare_parametric_nonparametric_irf(R, gp, theta_grid=[-1.0, 0.0, 1.0])
    assert len(cmp) == 6


def test_external_engine_success_gates_and_variational_failure():
    engine = lambda **kw: {"keys": sorted(kw), "value": next(iter(kw.values()))}
    d = pd.DataFrame({"x": [1, 2]})
    R = np.asarray([[0, 1], [1, 0]])
    assert ep.fit_dynamic_gpirt(d, external_engine=engine, alpha=1).engine == "external"
    assert ep.fit_continuous_time_irt(d, external_engine=engine).engine == "external"
    assert ep.fit_flow_mirt(R, external_engine=engine).engine == "external"
    assert ep.fit_variational_irt(R, external_engine=engine).engine == "external"
    with pytest.raises(ep.EyeProcessBackendError, match="external_engine"):
        ep.fit_variational_irt(R)


def test_trajectory_validation_smoothing_and_prediction_guard():
    with pytest.raises(ep.EyeProcessValidationError, match="four finite"):
        ep.latent_trait_trajectory([0, 1, 2], [0.0, 0.2, 0.4])
    tr = ep.latent_trait_trajectory([3, 0, 2, 1, 4], [0.7, 0.0, 0.5, 0.2, 1.0], spar=0.2)
    assert np.all(np.diff(tr.time) >= 0)
    with pytest.raises(ep.EyeProcessValidationError, match="latent_trait_trajectory"):
        ep.predict_theta_at_time(SimpleNamespace(), [0])


def test_process_information_sequence_weights_expected_and_selection_edges():
    with pytest.raises(ep.EyeProcessValidationError, match="equal length"):
        ep.process_item_information([0], [1, 2], [0])

    empty_w = ep.process_item_information([0], [1, 1.2], [0, 0.5], weights=[])
    full_w = ep.process_item_information(
        [0],
        [1, 1.2],
        [0, 0.5],
        process_information=[0.1, 0.2],
        rt_information=[0.3, 0.4],
        expected_time=[1.0, 2.0],
        weights=[1.0, 0.5, 0.25],
        burden_weight=0.1,
    )
    assert empty_w.utility.shape == (1, 2)
    weighted = ep.expected_process_information(full_w, theta_weights=[2.0])
    assert weighted.shape == (2,)
    with pytest.raises(ep.EyeProcessValidationError, match="process_item_information"):
        ep.expected_process_information(SimpleNamespace())

    bank = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "a": [1.0, 1.2],
            "b": [0.0, 0.5],
            "process_information": [0.1, 0.2],
            "rt_information": [0.3, 0.4],
            "expected_time": [2.0, 1.0],
        }
    )
    sel = ep.select_next_item_process(0, bank, weights=[1, 1, 1], burden_weight=0.1)
    assert sel.item_id in {"a", "b"}
    with pytest.raises(ep.EyeProcessValidationError, match="No unused"):
        ep.select_next_item_process(0, bank, used=["a", "b"])
