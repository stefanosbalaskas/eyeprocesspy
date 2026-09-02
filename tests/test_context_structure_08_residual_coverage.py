from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.context_structure_08 as cs
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError


class _BadFrame:
    def __iter__(self):
        raise RuntimeError("cannot iterate")


class _Obj(dict):
    def __init__(self, eyeprocess_class: str, **kwargs):
        super().__init__(**kwargs)
        self.eyeprocess_class = eyeprocess_class


def _registry(shared: bool = True):
    mapping = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "visual_context_id": ["ctx", "ctx", "ctx", "other"],
            "n_items": [3, 3, 3, 1],
            "shared_context": [shared, shared, shared, False],
        }
    )
    return _Obj(
        "eye_visual_context_registry",
        mapping=mapping,
        source_item_column="item_id",
        source_context_column="screen_id",
        min_items_per_context=3,
    )


def _numeric_data(n: int = 30) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "person_id": [f"p{i}" for i in range(n)],
            "x1": x,
            "x2": 2.0 * x + 1.0,
            "x3": np.sin(x / 3.0),
            "criterion": 0.25 * x + np.cos(x / 5.0),
        }
    )


def _seed_data(n: int = 12) -> pd.DataFrame:
    x = np.linspace(0.1, 1.2, n)
    return pd.DataFrame(
        {
            "p1": x,
            "p2": x**2,
            "irt_difficulty": 0.5 * x + 0.1,
            "irt_discrimination": 1.2 - 0.2 * x,
        }
    )


def test_low_level_dataframe_numeric_and_linear_model_edges():
    original = pd.DataFrame({"a": [1, 2]})
    copied = cs._df(original)
    assert copied.equals(original)
    assert copied is not original

    with pytest.raises(EyeProcessValidationError, match="coercible"):
        cs._df(_BadFrame(), "broken")

    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs._req(original, ["missing"])

    assert np.isnan(cs._sd([1.0]))
    assert np.isnan(cs._mean([np.nan]))
    assert np.isnan(cs._r2(np.array([1.0]), np.array([1.0])))
    assert np.isnan(cs._r2(np.array([2.0, 2.0]), np.array([2.0, 2.0])))

    with pytest.raises(EyeProcessValidationError, match="No complete cases"):
        cs._lm_fit(
            pd.DataFrame({"y": [np.nan], "x": [np.nan]}),
            "y",
            ["x"],
        )

    model = cs._lm_fit(pd.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]}), "y", ["x"])
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs._lm_predict(model, pd.DataFrame({"z": [1.0]}))


def test_visual_context_registry_validation_and_auto_context_paths():
    meta = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "screen_id": ["s", "s", None, ""],
        }
    )
    auto = cs.visual_context_registry(meta)
    mapping = auto["mapping"]
    assert auto["source_context_column"] == "screen_id"
    assert mapping.loc[2, "visual_context_id"] == "unique_context__i3"
    assert mapping.loc[3, "visual_context_id"] == "unique_context__i4"

    with pytest.raises(EyeProcessValidationError, match="at least 2"):
        cs.visual_context_registry(meta, min_items_per_context=1)

    duplicate = pd.DataFrame({"item_id": ["i1", "i1"], "screen_id": ["s", "s"]})
    with pytest.raises(EyeProcessValidationError, match="one row per item"):
        cs.visual_context_registry(duplicate)

    with pytest.raises(EyeProcessValidationError, match="No visual-context column"):
        cs.visual_context_registry(pd.DataFrame({"item_id": ["i1", "i2"]}))

    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs.visual_context_registry(meta, context="not_here")


def test_context_position_and_visual_irt_gates():
    with pytest.raises(EyeProcessValidationError, match="visual_context_registry"):
        cs._context_positions({}, ["i1", "i2", "i3", "i4"])

    no_shared = _registry(shared=False)
    with pytest.raises(EyeProcessValidationError, match="No shared visual context"):
        cs._context_positions(no_shared, ["i1", "i2", "i3", "i4"])

    registry = _registry()
    assert cs._context_positions(registry, ["i1", "i2", "i3", "i4"]) == [1, 2, 3]
    with pytest.raises(EyeProcessValidationError, match="selected context"):
        cs._context_positions(registry, ["i1", "i2", "i3", "i4"], "absent")

    with pytest.raises(EyeProcessValidationError, match="At least four"):
        cs.fit_visual_context_irt(pd.DataFrame(np.ones((3, 3))), registry)

    numeric_registry = cs.visual_context_registry(
        pd.DataFrame(
            {
                "item_id": ["Item1", "Item2", "Item3", "Item4"],
                "screen_id": ["ctx", "ctx", "ctx", "other"],
            }
        ),
        context="screen_id",
    )
    numeric_columns = pd.DataFrame(np.ones((4, 4)))
    with pytest.raises(EyeProcessBackendError, match="mirt"):
        cs.fit_visual_context_irt(numeric_columns, numeric_registry)

    all_context = cs.visual_context_registry(
        pd.DataFrame(
            {
                "item_id": ["i1", "i2", "i3", "i4"],
                "screen_id": ["ctx"] * 4,
            }
        ),
        context="screen_id",
    )
    with pytest.raises(EyeProcessValidationError, match="not all items"):
        cs.fit_visual_context_irt(
            pd.DataFrame(np.ones((4, 4)), columns=["i1", "i2", "i3", "i4"]),
            all_context,
        )


def test_visual_context_result_accessors_and_audit_branches():
    with pytest.raises(EyeProcessValidationError, match="eye_visual_context_irt"):
        cs.compare_visual_context_irt({})

    comparison = pd.DataFrame({"model": ["base"], "aic": [1.0]})
    visual = _Obj("eye_visual_context_irt", comparison=comparison)
    copied = cs.compare_visual_context_irt(visual)
    assert copied.equals(comparison)
    assert copied is not comparison

    assert cs.compare_visual_context_irt(_Obj("eye_visual_context_irt", comparison=None)).empty
    coerced = cs.compare_visual_context_irt(
        _Obj("eye_visual_context_irt", comparison={"model": ["base"]})
    )
    assert coerced["model"].tolist() == ["base"]

    with pytest.raises(EyeProcessValidationError, match="eye_visual_context_irt"):
        cs.context_factor_effects({})

    effects = pd.DataFrame({"term": ["ctx"], "estimate": [0.2]})
    visual = _Obj("eye_visual_context_irt", context_factor_effects=effects)
    out = cs.context_factor_effects(visual, IRTpars=True)
    assert out.equals(effects) and out is not effects

    with pytest.raises(EyeProcessBackendError, match="mirt backend"):
        cs.context_factor_effects(_Obj("eye_visual_context_irt"))

    with pytest.raises(EyeProcessValidationError, match="eye_visual_context_irt"):
        cs.audit_visual_context_dependence({})

    empty_visual = _Obj(
        "eye_visual_context_irt",
        registry={"mapping": pd.DataFrame({"item_id": []})},
        positions=[],
        context="ctx",
        comparison=None,
    )
    audit = cs.audit_visual_context_dependence(empty_visual)
    assert np.isnan(audit.loc[0, "context_fraction"])
    assert not bool(audit.loc[0, "comparison_available"])


def test_process_feature_blocks_validation_and_constant_handling():
    data = pd.DataFrame({"id": ["a", "b", "c"], "x": [1, 2, 3], "y": [2, 3, 4]})

    for blocks in ({}, {"": ["x"], "b": ["y"]}):
        with pytest.raises(EyeProcessValidationError, match="named list"):
            cs.process_feature_blocks(data, blocks)

    with pytest.raises(EyeProcessValidationError, match="only one block"):
        cs.process_feature_blocks(data, {"a": ["x"], "b": ["x"]})

    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs.process_feature_blocks(data, {"a": ["x"], "b": ["missing"]})

    constants = pd.DataFrame({"x": [1, 1, 1], "y": [2, 2, 2]})
    with pytest.raises(EyeProcessValidationError, match="At least two non-empty"):
        cs.process_feature_blocks(constants, {"a": "x", "b": "y"})

    kept = cs.process_feature_blocks(
        constants,
        {"a": "x", "b": "y"},
        drop_constant=False,
    )
    assert kept["blocks"] == {"a": ["x"], "b": ["y"]}


def test_multiblock_validation_fallback_and_accessors():
    data = _numeric_data(8)
    blocks = {"a": ["x1"], "b": ["x3"]}

    with pytest.raises(EyeProcessValidationError, match="engine"):
        cs.fit_multiblock_process_map(data, blocks, engine="bad")

    with pytest.raises(EyeProcessValidationError, match="ncp"):
        cs.fit_multiblock_process_map(data, blocks, ncp=0)

    with pytest.raises(EyeProcessValidationError, match="At least three rows"):
        cs.fit_multiblock_process_map(data.iloc[:2], blocks)

    prepared = cs.process_feature_blocks(data, blocks, id="person_id")
    with pytest.raises(EyeProcessBackendError, match="FactoMineR"):
        cs.fit_multiblock_process_map(prepared, engine="FactoMineR")

    data_with_missing = data.copy()
    data_with_missing.loc[0, "x3"] = np.nan
    mapped = cs.fit_multiblock_process_map(
        data_with_missing,
        blocks,
        id="person_id",
        engine="pca_block_scaled",
        ncp=2,
    )
    assert mapped["engine"] == "pca_block_scaled"
    assert mapped["person_coordinates"]["id"].tolist() == data["person_id"].tolist()

    mapped_no_id = cs.fit_multiblock_process_map(
        data[["x1", "x3"]],
        blocks,
        engine="auto",
        ncp=1,
    )
    assert mapped_no_id["person_coordinates"]["id"].tolist() == [str(i) for i in range(1, 9)]

    for accessor in (
        cs.multiblock_contributions,
        cs.multiblock_person_coordinates,
        cs.multiblock_variable_coordinates,
    ):
        with pytest.raises(EyeProcessValidationError, match="eye_multiblock_process_map"):
            accessor({})

    assert not cs.multiblock_contributions(mapped).empty
    assert not cs.multiblock_person_coordinates(mapped).empty
    assert not cs.multiblock_variable_coordinates(mapped).empty


def test_multiblock_nonfinite_mean_defensive_path(monkeypatch):
    raw = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})
    prepared = _Obj(
        "eye_process_feature_blocks",
        data=raw,
        blocks={"a": ["x"], "b": ["y"]},
        id=None,
    )

    def stop_svd(*args, **kwargs):
        raise RuntimeError("stop after nonfinite-mean fallback")

    monkeypatch.setattr(cs.np.linalg, "svd", stop_svd)
    with pytest.raises(RuntimeError, match="nonfinite-mean fallback"):
        cs.fit_multiblock_process_map(prepared, engine="pca_block_scaled")


def test_soft_cluster_probabilities_are_row_normalized():
    z = np.array([[0.0, 0.0], [2.0, 2.0]])
    centers = np.array([[0.0, 0.0], [2.0, 2.0]])
    probs = cs._soft_cluster_prob(z, centers)
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_process_profile_validation_backend_and_accessors():
    data = _numeric_data(30)

    with pytest.raises(EyeProcessValidationError, match="Invalid profile engine"):
        cs.fit_process_profile_mixture(data, ["x1", "x3"], engine="bad")
    with pytest.raises(EyeProcessValidationError, match="at least one"):
        cs.fit_process_profile_mixture(data, [])
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs.fit_process_profile_mixture(data, ["x1", "missing"])
    with pytest.raises(EyeProcessValidationError, match="k must be"):
        cs.fit_process_profile_mixture(data, ["x1", "x3"], k=1)

    one_varying = data.assign(constant=1.0)
    with pytest.raises(EyeProcessValidationError, match="two varying"):
        cs.fit_process_profile_mixture(one_varying, ["x1", "constant"])

    with pytest.raises(EyeProcessValidationError, match="Too few complete cases"):
        cs.fit_process_profile_mixture(data.iloc[:10], ["x1", "x3"], k=3)

    with pytest.raises(EyeProcessBackendError, match="tidyLPA"):
        cs.fit_process_profile_mixture(data, ["x1", "x3"], k=2, engine="tidyLPA")

    fit = cs.fit_process_profile_mixture(
        data,
        ["x1", "x3"],
        k=2,
        id="person_id",
        engine="kmeans_reference",
        seed=10,
    )
    assert len(fit["assignment"]) == len(data)
    assert {"profile_probability_1", "profile_probability_2"}.issubset(fit["assignment"])

    no_id = cs.fit_process_profile_mixture(
        data.drop(columns="person_id"),
        ["x1", "x3"],
        k=2,
        engine="auto",
        seed=10,
    )
    assert no_id["assignment"]["id"].iloc[0] == "1"

    with pytest.raises(EyeProcessValidationError, match="eye_process_profile_mixture"):
        cs.process_profile_probabilities({})
    with pytest.raises(EyeProcessValidationError, match="eye_process_profile_mixture"):
        cs.process_profile_summary({})
    assert not cs.process_profile_probabilities(fit).empty
    assert not cs.process_profile_summary(fit).empty


def test_compare_profile_solution_validation_and_valid_k_filtering():
    data = _numeric_data(30)
    with pytest.raises(EyeProcessValidationError, match="two varying"):
        cs.compare_process_profile_solutions(
            data.assign(constant=1.0),
            ["x1", "constant"],
        )

    with pytest.raises(EyeProcessValidationError, match="No valid k_values"):
        cs.compare_process_profile_solutions(data, ["x1", "x3"], k_values=[0, 1, 30, 31])

    result = cs.compare_process_profile_solutions(
        data,
        ["x1", "x3"],
        k_values=[3, 2, 2, 99],
        seed=2,
    )
    assert result["k"].tolist() == [2, 3]
    assert result["between_over_total"].notna().all()


def test_external_validity_validation_constant_correlation_and_accessors():
    data = _numeric_data(30)

    with pytest.raises(EyeProcessValidationError, match="at least one process predictor"):
        cs.audit_process_external_validity(data, "criterion", [])
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs.audit_process_external_validity(data, "criterion", ["missing"])
    with pytest.raises(EyeProcessValidationError, match="At least 20 complete"):
        cs.audit_process_external_validity(data.iloc[:10], "criterion", ["x1"])

    constant = data.assign(flat=1.0)
    fit = cs.audit_process_external_validity(
        constant,
        "criterion",
        ["x1", "flat"],
        baseline_predictors=["x3"],
    )
    flat_corr = fit["associations"].loc[
        fit["associations"]["predictor"].eq("flat"), "correlation"
    ].iloc[0]
    assert np.isnan(flat_corr)

    for accessor in (
        cs.process_criterion_associations,
        cs.incremental_process_validity,
        cs.compare_process_criterion_models,
    ):
        with pytest.raises(EyeProcessValidationError, match="eye_process_external_validity"):
            accessor({})

    assert not cs.process_criterion_associations(fit).empty
    assert not cs.incremental_process_validity(fit).empty
    assert not cs.compare_process_criterion_models(fit).empty

    no_comparison = _Obj(
        "eye_process_external_validity",
        comparison=None,
        associations=pd.DataFrame(),
        baseline_model={"r_squared": 0.0},
        full_model={"r_squared": 0.0},
        incremental_r2=0.0,
    )
    assert cs.compare_process_criterion_models(no_comparison).empty


def test_item_parameter_seed_validation_backend_prediction_and_audit():
    data = _seed_data()

    with pytest.raises(EyeProcessValidationError, match="Invalid seed-model engine"):
        cs.fit_item_parameter_seed_model(data, predictors=["p1"], engine="bad")
    with pytest.raises(EyeProcessValidationError, match="at least one item"):
        cs.fit_item_parameter_seed_model(data, predictors=[])
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs.fit_item_parameter_seed_model(data, predictors=["missing"])

    constant = data.assign(flat=1.0)
    with pytest.raises(EyeProcessValidationError, match="predictors must vary"):
        cs.fit_item_parameter_seed_model(constant, predictors=["p1", "flat"])

    with pytest.raises(EyeProcessValidationError, match="Too few complete calibrated items"):
        cs.fit_item_parameter_seed_model(data.iloc[:4], predictors=["p1", "p2"])

    with pytest.raises(EyeProcessBackendError, match="ranger"):
        cs.fit_item_parameter_seed_model(data, predictors=["p1", "p2"], engine="ranger")

    model = cs.fit_item_parameter_seed_model(data, predictors=["p1", "p2"], engine="lm")
    with pytest.raises(EyeProcessValidationError, match="eye_item_parameter_seed"):
        cs.predict_item_parameter_priors({}, data[["p1", "p2"]])
    with pytest.raises(EyeProcessValidationError, match="missing required"):
        cs.predict_item_parameter_priors(model, data[["p1"]])

    low_disc = _Obj(
        "eye_item_parameter_seed",
        predictors=["p1"],
        difficulty_model=cs._lm_fit(
            pd.DataFrame({"y": [-1.0, 1.0], "p1": [0.0, 1.0]}),
            "y",
            ["p1"],
        ),
        discrimination_model=cs._lm_fit(
            pd.DataFrame({"y": [-2.0, -1.0], "p1": [0.0, 1.0]}),
            "y",
            ["p1"],
        ),
        caveat="caveat",
    )
    pred = cs.predict_item_parameter_priors(low_disc, pd.DataFrame({"p1": [0.5]}))
    assert pred.loc[0, "predicted_pre_pilot_discrimination"] == pytest.approx(0.05)

    for bad_range in ([1.0], [np.nan, 1.0], [2.0, 1.0]):
        with pytest.raises(EyeProcessValidationError, match="difficulty_range"):
            cs.audit_candidate_item_bank(model, data[["p1", "p2"]].iloc[:2], bad_range)

    for minimum in (0.0, -1.0, np.nan):
        with pytest.raises(EyeProcessValidationError, match="discrimination_min"):
            cs.audit_candidate_item_bank(
                model,
                data[["p1", "p2"]].iloc[:2],
                discrimination_min=minimum,
            )

    audited = cs.audit_candidate_item_bank(
        model,
        pd.DataFrame({"p1": [10.0], "p2": [100.0]}),
        difficulty_range=(-0.1, 0.1),
        discrimination_min=5.0,
    )
    assert bool(audited["table"].loc[0, "review_required"])
