from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.irt as irt
from eyeprocesspy.exceptions import EyeProcessValidationError


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["I1", "I2", "I3"],
            "a": [0.9, 1.2, 1.05],
            "b": [-0.6, 0.1, 0.9],
            "c": [0.0, 0.1, 0.05],
            "d": [1.0, 0.95, 0.98],
        }
    )


def _focal() -> pd.DataFrame:
    d = _items().copy()
    d["a"] *= 1.05
    d["b"] = d["b"] * 0.95 + 0.12
    return d


def test_item_information_all_families_and_guards():
    theta = np.array([-1.0, 0.0, 1.0])
    for family, kwargs in [
        ("2pl", {"a": 1.2, "b": 0.1}),
        ("3pl", {"a": 1.2, "b": 0.1, "c": 0.15}),
        ("4pl", {"a": 1.2, "b": 0.1, "c": 0.05, "d": 0.95}),
        ("grm", {"a": 1.1, "thresholds": [-0.5, 0.5]}),
        ("gpcm", {"a": 1.1, "steps": [-0.4, 0.6]}),
        ("nominal", {"slopes": [-0.5, 0.2, 0.8], "intercepts": [0.1, -0.2, 0.0]}),
    ]:
        value = irt.eyeprocess_irt_item_information(theta, family=family, **kwargs)
        assert value.shape == theta.shape
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0)

    with pytest.raises(EyeProcessValidationError, match="unsupported IRT family"):
        irt.eyeprocess_irt_item_information(theta, family="bad")
    with pytest.raises(EyeProcessValidationError, match="3PL information"):
        irt.eyeprocess_irt_item_information(theta, family="3pl", c=1.0)
    with pytest.raises(EyeProcessValidationError, match="4PL information"):
        irt.eyeprocess_irt_item_information(theta, family="4pl", c=0.8, d=0.7)


def test_numeric_information_and_expected_score_family_branches():
    theta = np.array([-0.5, 0.5])
    info = irt._numeric_information(
        lambda t: irt.eyeprocess_irt_grm_probability([t], thresholds=[-0.25, 0.4])[0],
        theta,
    )
    assert info.shape == (2,) and np.all(info >= 0)

    expected = {
        "2pl": irt.eyeprocess_irt_expected_score(theta, family="2pl"),
        "3pl": irt.eyeprocess_irt_expected_score(theta, family="3pl", c=0.2),
        "4pl": irt.eyeprocess_irt_expected_score(theta, family="4pl", c=0.05, d=0.95),
        "grm": irt.eyeprocess_irt_expected_score(theta, family="grm", thresholds=[-0.4, 0.4]),
        "gpcm": irt.eyeprocess_irt_expected_score(theta, family="gpcm", steps=[-0.3, 0.5]),
        "nominal": irt.eyeprocess_irt_expected_score(
            theta,
            family="nominal",
            slopes=[-0.5, 0.2, 0.7],
            intercepts=[0.0, 0.1, -0.1],
        ),
    }
    assert all(v.shape == theta.shape for v in expected.values())
    with pytest.raises(EyeProcessValidationError, match="unsupported IRT family"):
        irt.eyeprocess_irt_expected_score(theta, family="bad")


def test_plausible_values_scalar_conversion_and_reproducibility_guards():
    score = irt.eyeprocess_irt_eap_score([1, 0, 1], _items())
    a = irt.eyeprocess_irt_plausible_values(score, n=4, seed=9)
    b = irt.eyeprocess_irt_plausible_values(score, n=np.array(4), seed=np.array(9))
    assert np.array_equal(a, b)
    with pytest.raises(EyeProcessValidationError, match="EAP eye_irt_score"):
        irt.eyeprocess_irt_plausible_values({})
    for kwargs in (
        {"n": [2, 3]},
        {"seed": [1, 2]},
        {"n": "bad"},
        {"n": 0},
        {"seed": 0},
    ):
        with pytest.raises(EyeProcessValidationError, match="positive scalar integers"):
            irt.eyeprocess_irt_plausible_values(score, **kwargs)


def test_item_bank_selection_exposure_content_and_no_eligible_paths():
    items = _items()
    bank = irt.eyeprocess_irt_item_bank(items, content=["A", "B", "A"], exposure_limit=0.5)
    assert irt.validate_eyeprocess_irt_item_bank(bank)
    with pytest.raises(EyeProcessValidationError, match="exposure_limit"):
        irt.eyeprocess_irt_item_bank(items, exposure_limit=0)
    with pytest.raises(EyeProcessValidationError, match="content must match"):
        irt.eyeprocess_irt_item_bank(items, content=["A"])

    exposure = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "rate": [0.9, 0.0, 0.8]})
    selected = irt.eyeprocess_irt_item_selection(bank, theta=0.0, exposure=exposure, content_required=["B"])
    assert selected.selected == "I2"
    none = irt.eyeprocess_irt_item_selection(bank, theta=0.0, administered=["I1", "I2", "I3"])
    assert none.selected is None and none.reason == "no_eligible_item"

    bare = irt.eyeprocess_irt_item_bank(items)
    with pytest.raises(EyeProcessValidationError, match="no content labels"):
        irt.eyeprocess_irt_item_selection(bare, theta=0.0, content_required=["A"])
    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_irt_item_bank({})


def test_stopping_rule_continue_precision_maximum_and_scalar_guards():
    assert irt.eyeprocess_irt_stopping_rule(3, se=0.2, min_items=5, max_items=10)["reason"] == "continue"
    assert irt.eyeprocess_irt_stopping_rule(5, se=0.2, min_items=5, max_items=10)["reason"] == "target_precision"
    assert irt.eyeprocess_irt_stopping_rule(10, se=1.0, min_items=5, max_items=10)["reason"] == "maximum_items"
    with pytest.raises(EyeProcessValidationError, match="invalid stopping-rule"):
        irt.eyeprocess_irt_stopping_rule([1, 2])
    with pytest.raises(EyeProcessValidationError, match="invalid stopping-rule"):
        irt.eyeprocess_irt_stopping_rule("bad")
    with pytest.raises(EyeProcessValidationError, match="invalid stopping-rule"):
        irt.eyeprocess_irt_stopping_rule(1, min_items=5, max_items=4)


def test_content_balance_default_target_custom_target_and_guards():
    bank = irt.eyeprocess_irt_item_bank(_items(), content=["A", "B", "A"])
    default = irt.eyeprocess_irt_content_balance_audit(["I1", "I2", "I3"], bank)
    assert default["observed"].sum() == pytest.approx(1.0)
    custom = irt.eyeprocess_irt_content_balance_audit(
        ["I1", "I2", "I3"], bank, target={"A": 2.0, "B": 1.0, "C": 1.0}
    )
    assert custom["target"].sum() == pytest.approx(1.0)
    assert "C" in set(custom["content"])
    with pytest.raises(EyeProcessValidationError, match="absent from bank"):
        irt.eyeprocess_irt_content_balance_audit(["missing"], bank)
    with pytest.raises(EyeProcessValidationError, match="at least one item"):
        irt.eyeprocess_irt_content_balance_audit([], bank)
    with pytest.raises(EyeProcessValidationError, match="target must"):
        irt.eyeprocess_irt_content_balance_audit(["I1"], bank, target={"A": -1.0})


def test_link_optimizers_weight_start_and_stability_paths():
    ref, focal = _items(), _focal()
    theta = np.linspace(-2, 2, 21)
    sl = irt.eyeprocess_irt_stocking_lord_link(ref, focal, theta=theta)
    hb = irt.eyeprocess_irt_haebara_link(ref, focal, theta=theta)
    assert sl.A > 0 and hb.A > 0
    with pytest.raises(EyeProcessValidationError, match="weights must"):
        irt.eyeprocess_irt_stocking_lord_link(ref, focal, theta=theta, weights=[1, 2])
    with pytest.raises(EyeProcessValidationError, match="start must"):
        irt.eyeprocess_irt_haebara_link(ref, focal, theta=theta, start=(0, 0))

    stability = irt.eyeprocess_irt_link_stability(
        ref,
        focal,
        anchor_sets={"all": ["I1", "I2", "I3"], "pair": ["I1", "I2"]},
        method="mean-mean",
    )
    assert len(stability.table) == 2
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        irt.eyeprocess_irt_link_stability(ref, focal, anchor_sets=[])
    with pytest.raises(EyeProcessValidationError, match="invalid linking method"):
        irt.eyeprocess_irt_link_stability(ref, focal, anchor_sets=[["I1", "I2"]], method="bad")


def test_anchor_audit_and_purification_iteration_paths():
    dif = pd.DataFrame({"item_id": ["I1", "I2"], "effect": [0.3, 0.01]})
    audit = irt.eyeprocess_irt_anchor_audit(_items(), dif=dif, max_abs_effect=0.1, min_information=0.05)
    assert not bool(audit.loc[audit.item_id.eq("I1"), "eligible"].iloc[0])
    assert "information_theta0" in audit

    calls = []
    def effects(anchors):
        calls.append(tuple(anchors))
        effect = [0.0 for _ in anchors]
        if len(calls) == 1 and "I3" in anchors:
            effect[anchors.index("I3")] = 0.5
        return pd.DataFrame({"item_id": anchors, "effect": effect})

    purified = irt.eyeprocess_irt_anchor_purification(_items(), effects, threshold=0.1)
    assert purified.anchors == ["I1", "I2"]
    assert len(purified.history) == 2

    with pytest.raises(EyeProcessValidationError, match="effect_fun"):
        irt.eyeprocess_irt_anchor_purification(_items(), object())
    with pytest.raises(EyeProcessValidationError, match="fewer than two"):
        irt.eyeprocess_irt_anchor_purification(
            _items(),
            lambda anchors: pd.DataFrame({"item_id": anchors, "effect": [1.0] * len(anchors)}),
            initial=["I1", "I2"],
        )


def test_sbc_rank_generation_and_diagnostic_guards():
    deterministic = irt.eyeprocess_irt_sbc_ranks([0.0], [[-1.0, 0.0, 0.0, 1.0]], randomize_ties=False)
    randomized = irt.eyeprocess_irt_sbc_ranks([0.0], [[-1.0, 0.0, 0.0, 1.0]], randomize_ties=True, seed=2)
    assert deterministic.tolist() == [1]
    assert 1 <= randomized[0] <= 3
    with pytest.raises(EyeProcessValidationError, match="draws rows"):
        irt.eyeprocess_irt_sbc_ranks([0.0, 1.0], [[0.0, 1.0]])
    with pytest.raises(EyeProcessValidationError, match="seed must"):
        irt.eyeprocess_irt_sbc_ranks([0.0], np.empty((1, 0)), seed=1)

    diag = irt._sbc_rank_diagnostics([0, 1, 2, np.nan], n_draws=2, bins=10)
    assert diag.bins == 3
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        irt._sbc_rank_diagnostics([0], 0)
    with pytest.raises(EyeProcessValidationError, match="integer-valued"):
        irt._sbc_rank_diagnostics([0.5], 2)
    with pytest.raises(EyeProcessValidationError, match="inclusive"):
        irt._sbc_rank_diagnostics([3], 2)
    with pytest.raises(EyeProcessValidationError, match=">= 2"):
        irt._sbc_rank_diagnostics([0], 2, bins=1)


def test_ability_sbc_all_input_guard_branches_without_expensive_simulation():
    items = _items()
    cases = [
        ({"replications": 19}, "replications"),
        ({"posterior_draws": 8}, "posterior_draws"),
        ({"seed": 0}, "seed"),
        ({"theta_grid": np.linspace(-2, 2, 100)}, "theta_grid"),
        ({"prior_sd": 0}, "prior"),
        ({"interval": 1.0}, "interval"),
    ]
    for kwargs, message in cases:
        with pytest.raises(EyeProcessValidationError, match=message):
            irt.run_eyeprocess_irt_ability_sbc(items, **kwargs)


def test_mirt_loading_directional_and_testlet_residual_guards():
    spec = irt.eyeprocess_mirt_loading_spec(
        ["I1", "I2", "I3"],
        [[1, 1], [1, 0], [0, 1]],
        dimension_names=["A", "B"],
        simple_structure=True,
    )
    assert spec.violations == ["I1"]
    with pytest.raises(EyeProcessValidationError, match="loadings"):
        irt.eyeprocess_mirt_loading_spec(["I1"], [[1, np.nan]])
    with pytest.raises(EyeProcessValidationError, match="dimension_names"):
        irt.eyeprocess_mirt_loading_spec(["I1"], [[1, 0]], dimension_names=["A", "A"])
    with pytest.raises(EyeProcessValidationError, match="positive scalar"):
        irt.eyeprocess_mirt_loading_audit(spec, min_items_per_dimension=0)
    assert len(irt.eyeprocess_mirt_loading_audit(spec, min_items_per_dimension=1)) == 2

    info = irt.eyeprocess_mirt_directional_information([0.0, 0.0], [1.0, 0.5], direction=[1.0, 0.0])
    assert info >= 0
    with pytest.raises(EyeProcessValidationError, match="equal dimension"):
        irt.eyeprocess_mirt_directional_information([0], [1, 2])
    with pytest.raises(EyeProcessValidationError, match="invalid direction"):
        irt.eyeprocess_mirt_directional_information([0, 0], [1, 2], direction=[0, 0])

    testlets = irt.eyeprocess_irt_testlet_spec(["I1", "I2", "I3"], ["T1", "T1", "T2"])
    audit = irt.eyeprocess_irt_testlet_audit(testlets, min_items=2)
    assert bool(audit.loc[audit.testlet.eq("T2"), "singleton"].iloc[0])
    with pytest.raises(EyeProcessValidationError, match="eye_irt_testlet_spec"):
        irt.eyeprocess_irt_testlet_audit(pd.DataFrame({"testlet": ["T1"]}))
    with pytest.raises(EyeProcessValidationError, match="positive scalar"):
        irt.eyeprocess_irt_testlet_audit(testlets, min_items=0)


def test_latent_regression_numeric_categorical_interaction_centering_and_guards():
    data = pd.DataFrame(
        {
            "age": [10.0, 12.0, 14.0, np.nan],
            "dose": [1.0, 2.0, 3.0, 4.0],
            "group": ["A", "B", "A", "B"],
        }
    )
    centered = irt.eyeprocess_irt_latent_regression_design(data, "~ age + group + age:dose")
    assert "(Intercept)" in centered.matrix
    assert "group_B" in centered.matrix
    assert "age:dose" in centered.matrix
    assert centered.complete.tolist() == [True, True, True, False]
    assert centered.matrix.loc[:2, "age"].mean() == pytest.approx(0.0)

    raw = irt.eyeprocess_irt_latent_regression_design(data.fillna(16), "0 + age + group", center_numeric=False)
    assert "(Intercept)" not in raw.matrix
    assert raw.centers["age"] == 0.0
    with pytest.raises(EyeProcessValidationError, match="missing column"):
        irt.eyeprocess_irt_latent_regression_design(data, "~ absent")
    with pytest.raises(EyeProcessValidationError, match="missing column"):
        irt.eyeprocess_irt_latent_regression_design(data, "~ age:absent")


def test_cdm_qmatrix_profiles_probability_and_uncertainty_guards():
    q = np.array([[1, 0], [0, 1], [1, 1]])
    qa = irt.eyeprocess_cdm_qmatrix_audit(q, item_ids=["I1", "I2", "I3"], attribute_names=["A", "B"])
    assert qa.complete_identity_block
    with pytest.raises(EyeProcessValidationError, match="binary matrix"):
        irt.eyeprocess_cdm_qmatrix_audit([[1, 2]])
    with pytest.raises(EyeProcessValidationError, match="identifier lengths"):
        irt.eyeprocess_cdm_qmatrix_audit(q, item_ids=["I1"])

    profiles = irt.eyeprocess_cdm_attribute_profiles(2, ["A", "B"])
    assert len(profiles) == 4
    with pytest.raises(EyeProcessValidationError, match="between 1 and 20"):
        irt.eyeprocess_cdm_attribute_profiles(0)
    with pytest.raises(EyeProcessValidationError, match="attribute_names"):
        irt.eyeprocess_cdm_attribute_profiles(2, ["A", "A"])

    ideal = irt.eyeprocess_cdm_dina_ideal_response(q, profiles[["A", "B"]])
    prob = irt.eyeprocess_cdm_dina_probability(ideal, slip=[0.1, 0.2, 0.3], guess=0.2)
    assert prob.shape == ideal.shape
    with pytest.raises(EyeProcessValidationError, match="invalid slip/guess"):
        irt.eyeprocess_cdm_dina_probability(ideal, slip=[0.1, 0.2])

    uncertainty = irt.eyeprocess_cdm_classification_uncertainty([[0.8, 0.2], [0.4, 0.6]])
    assert (uncertainty["normalized_entropy"] >= 0).all()
    with pytest.raises(EyeProcessValidationError, match="row-normalized"):
        irt.eyeprocess_cdm_classification_uncertainty([[0.8, 0.3]])


def test_classification_precision_zero_se_multiple_cuts_and_guards():
    out = irt.eyeprocess_irt_classification_precision(
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        cut_score=[0.0, 0.5],
        confidence=0.9,
    )
    at_zero = out[(out["theta"] == 0.0) & (out["cut_score"] == 0.0)].iloc[0]
    assert at_zero.probability_above == pytest.approx(0.5)
    assert len(out) == 6
    with pytest.raises(EyeProcessValidationError, match="compatible finite vectors"):
        irt.eyeprocess_irt_classification_precision([0, 1], [0.1, -0.1])
    with pytest.raises(EyeProcessValidationError, match="cut_score"):
        irt.eyeprocess_irt_classification_precision([0], [0.1], cut_score=[])
    with pytest.raises(EyeProcessValidationError, match="confidence"):
        irt.eyeprocess_irt_classification_precision([0], [0.1], confidence=1.0)


def test_missing_by_design_declared_structural_unexpected_and_guards():
    responses = np.array([[1.0, np.nan, np.nan], [0.0, 1.0, np.nan]])
    design = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    declared = irt.eyeprocess_irt_missing_by_design_audit(responses, design=design, min_administered=2)
    assert declared.structural_missing == 2
    assert declared.unexpected_missing == 1
    assert declared.has_declared_design
    undeclared = irt.eyeprocess_irt_missing_by_design_audit(responses)
    assert undeclared.structural_missing == 0
    assert undeclared.unexpected_missing == 3
    with pytest.raises(EyeProcessValidationError, match="numeric/integer matrix"):
        irt.eyeprocess_irt_missing_by_design_audit([1, 0])
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        irt.eyeprocess_irt_missing_by_design_audit(responses, min_administered=0)
    with pytest.raises(EyeProcessValidationError, match="same dimensions"):
        irt.eyeprocess_irt_missing_by_design_audit(responses, design=[[1, 1]])
    bad_design = design.copy(); bad_design[0, 0] = 2
    with pytest.raises(EyeProcessValidationError, match="0/1/NA"):
        irt.eyeprocess_irt_missing_by_design_audit(responses, design=bad_design)
