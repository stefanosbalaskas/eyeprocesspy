from __future__ import annotations

import math

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
    out = _items().copy()
    out["a"] *= 1.05
    out["b"] = out["b"] * 0.95 + 0.12
    return out


def test_foundation_residual_guards_and_precision_success():
    spec = irt.eyeprocess_irt_model_spec(family="2pl", identification="anchor")
    audit = irt.eyeprocess_irt_identification_audit(spec, constraints={"anchor_items": ["I1", "I2"]}, n_items=6, n_persons=4)
    assert audit.valid
    with pytest.raises(EyeProcessValidationError, match="n_persons"):
        irt.eyeprocess_irt_identification_audit(spec, n_persons=-1)

    sparse = pd.DataFrame({"person": [], "item": []})
    empty = irt.eyeprocess_irt_sparse_design_audit(sparse, "person", "item")
    assert math.isnan(empty.density)
    with pytest.raises(EyeProcessValidationError, match="non-empty scalar"):
        irt.eyeprocess_irt_sparse_design_audit(pd.DataFrame({"": [1], "item": ["I1"]}), "", "item")
    with pytest.raises(EyeProcessValidationError, match="minimum counts"):
        irt.eyeprocess_irt_sparse_design_audit(pd.DataFrame({"person": [1], "item": ["I1"]}), "person", "item", min_person_items=0)

    with pytest.raises(EyeProcessValidationError, match="steps"):
        irt.eyeprocess_irt_gpcm_probability([0.0], steps=[])
    with pytest.raises(EyeProcessValidationError, match="slopes and intercepts"):
        irt.eyeprocess_irt_nominal_probability([0.0], slopes=[1.0], intercepts=[0.0])

    prof = irt.eyeprocess_irt_measurement_precision_profile(np.linspace(-2, 2, 9), _items(), target=(-1, 1))
    assert prof.area >= 0


def test_empty_fit_tables_score_loop_and_marginal_reliability_edges():
    item_fit = irt.eyeprocess_irt_item_fit_residuals([[np.nan]], [[np.nan]])
    assert item_fit.loc[0, "n"] == 0
    person_fit = irt.eyeprocess_irt_person_fit_residuals([[np.nan]], [[np.nan]])
    assert person_fit.loc[0, "n"] == 0
    empty_fit = irt.eyeprocess_irt_item_fit_residuals(np.empty((1, 0)), np.empty((1, 0)))
    assert empty_fit.empty

    empty_scores = irt.eyeprocess_irt_score_table(np.empty((0, 3)), _items(), method="EAP")
    assert empty_scores.empty
    with pytest.raises(EyeProcessValidationError, match="method must"):
        irt.eyeprocess_irt_score_table([[1, 0, 1]], _items(), method="bad")
    with pytest.raises(EyeProcessValidationError, match="person_ids"):
        irt.eyeprocess_irt_score_table([[1, 0, 1]], _items(), person_ids=[])

    assert math.isnan(irt.eyeprocess_irt_marginal_reliability([0.0], [0.1]))
    assert math.isnan(irt.eyeprocess_irt_marginal_reliability([1.0, 1.0], [0.1, 0.2]))


def test_information_targeting_adaptive_and_penalty_short_circuits():
    default = irt.eyeprocess_irt_information_targeting(_items(), [-1.0, 0.0, 1.0])
    assert default.weighted_information >= 0
    custom = irt.eyeprocess_irt_information_targeting(_items(), [-1.0, 0.0, 1.0], weights=[1.0, 2.0, 1.0])
    assert custom.weights.sum() == pytest.approx(1.0)
    for weights in ([1.0], [1.0, np.nan, 1.0], [1.0, -1.0, 1.0], [0.0, 0.0, 0.0]):
        with pytest.raises(EyeProcessValidationError, match="weights"):
            irt.eyeprocess_irt_information_targeting(_items(), [-1.0, 0.0, 1.0], weights=weights)

    bare = irt.eyeprocess_irt_item_bank(_items())
    with pytest.raises(EyeProcessValidationError, match="no content labels"):
        irt.eyeprocess_irt_content_balance_audit(["I1"], bare)

    trace = irt.eyeprocess_irt_adaptive_trace(["I1", "I2"], [0, 0], [0.1, 0.2], [0.8, 0.7], [1.0, 1.2])
    assert trace.response.isna().all()
    one = irt.eyeprocess_irt_adaptive_trace(["I1"], [0], [0.1], [0.8], [1.0], response=1)
    assert one.loc[0, "response"] == 1
    with pytest.raises(EyeProcessValidationError, match="equal length"):
        irt.eyeprocess_irt_adaptive_trace(["I1", "I2"], [0], [0.1, 0.2], [0.8, 0.7], [1.0, 1.2])

    with pytest.raises(EyeProcessValidationError, match="SE values"):
        irt.eyeprocess_irt_information_gain([1.0], [1.0, 2.0])
    with pytest.raises(EyeProcessValidationError, match="SE values"):
        irt.eyeprocess_irt_information_gain([1.0], [0.0])

    score = irt.eyeprocess_irt_process_aware_selection_penalty([2.0, 3.0], 1.0, burden_weight=0.2, quality_risk=0.5, quality_weight=0.1)
    assert score.shape == (2,)
    with pytest.raises(EyeProcessValidationError, match="compatible vectors"):
        irt.eyeprocess_irt_process_aware_selection_penalty([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(EyeProcessValidationError, match="compatible vectors"):
        irt.eyeprocess_irt_process_aware_selection_penalty([1.0, np.nan], [1.0, 2.0])


def test_linking_anchor_loop_and_empty_effect_paths(monkeypatch):
    with pytest.raises(EyeProcessValidationError, match="At least two"):
        irt._anchor_merge(_items(), _focal(), anchors=["I1"])

    flat = _focal()
    flat["b"] = 0.0
    with pytest.raises(EyeProcessValidationError, match="difficulty SD"):
        irt.eyeprocess_irt_mean_sigma_link(_items(), flat)

    fake_r = _items().copy(); fake_f = _focal().copy()
    fake_r["a"] = 0.0
    monkeypatch.setattr(irt, "_anchor_merge", lambda *args, **kwargs: (fake_r, fake_f, ["I1", "I2", "I3"]))
    with pytest.raises(EyeProcessValidationError, match="mean discrimination"):
        irt.eyeprocess_irt_mean_mean_link(_items(), _focal())
    monkeypatch.undo()

    with pytest.raises(EyeProcessValidationError, match="A must"):
        irt._apply_link_df(_items(), 0, 0)
    with pytest.raises(EyeProcessValidationError, match="link must"):
        irt.eyeprocess_irt_apply_link(_items(), {})

    seq = irt.eyeprocess_irt_link_stability(_items(), _focal(), [["I1", "I2"], ["I2", "I3"]], method="mean-mean")
    assert len(seq.table) == 2

    plain_audit = irt.eyeprocess_irt_anchor_audit(_items())
    assert plain_audit.eligible.all()
    zero_iter = irt.eyeprocess_irt_anchor_purification(_items(), lambda ids: pd.DataFrame({"item_id": ids, "effect": 0.0}), max_iter=0)
    assert zero_iter.history.empty

    with pytest.raises(EyeProcessValidationError, match="exactly one item"):
        irt.eyeprocess_irt_dif_effect_curve(_items().iloc[:2], _focal().iloc[:1])
    no_common = _focal().copy(); no_common["item_id"] = ["J1", "J2", "J3"]
    with pytest.raises(EyeProcessValidationError, match="No common items"):
        irt.eyeprocess_irt_dtf_curve(_items(), no_common)

    summary = irt.eyeprocess_irt_functioning_effect_summary(pd.DataFrame({"absolute_difference": [np.nan], "signed_difference": [np.nan]}))
    assert math.isnan(summary["max_abs"]) and math.isnan(summary["signed_area"])


def test_process_concordance_drift_and_empty_group_loops():
    dif = pd.DataFrame({"item_id": ["I1"], "effect": [0.1]})
    proc = pd.DataFrame({"item_id": ["J1"], "effect": [0.2]})
    empty = irt.eyeprocess_irt_process_dif_concordance(dif, proc)
    assert empty.n == 0

    d3 = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "effect": [0.1, 0.2, 0.4]})
    p3 = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "effect": [0.2, 0.5, 0.8]})
    full = irt.eyeprocess_irt_process_dif_concordance(d3, p3)
    assert np.isfinite(full.correlation)

    cols_session = pd.DataFrame(columns=["item_id", "session", "b"])
    cols_device = pd.DataFrame(columns=["item_id", "device", "b"])
    assert irt.eyeprocess_irt_session_drift(cols_session).empty
    assert irt.eyeprocess_irt_device_drift(cols_device).empty
    nan_s = irt.eyeprocess_irt_session_drift(pd.DataFrame({"item_id": ["I1"], "session": [1], "b": [np.nan]}))
    assert math.isnan(nan_s.loc[0, "change"])
    nan_d = irt.eyeprocess_irt_device_drift(pd.DataFrame({"item_id": ["I1"], "device": ["D1"], "b": [np.nan]}))
    assert math.isnan(nan_d.loc[0, "mean"])


def test_joint_process_contract_validation_and_empty_group_profiles():
    for kwargs in (
        {"response_family": "bad"},
        {"time_model": "bad"},
        {"missingness": "bad"},
        {"status": "bad"},
    ):
        with pytest.raises(EyeProcessValidationError, match="invalid joint-process"):
            irt.eyeprocess_joint_process_irt_spec(**kwargs)
    with pytest.raises(EyeProcessValidationError, match="joint_process_irt_spec"):
        irt.validate_eyeprocess_joint_process_irt_spec({})

    base = pd.DataFrame({"p": ["P1"], "i": ["I1"], "y": [1]})
    bundle = irt.eyeprocess_process_irt_data_bundle(base, "p", "i", "y")
    assert bundle.response_time is None
    with pytest.raises(EyeProcessValidationError, match="binary"):
        irt.eyeprocess_process_irt_data_bundle(pd.DataFrame({"p": [1], "i": [1], "y": [2]}), "p", "i", "y")
    with pytest.raises(EyeProcessValidationError, match="positive"):
        irt.eyeprocess_process_irt_data_bundle(pd.DataFrame({"p": [1], "i": [1], "y": [1], "rt": [0]}), "p", "i", "y", response_time="rt")
    with pytest.raises(EyeProcessValidationError, match="No positive"):
        irt.eyeprocess_response_time_profile(pd.DataFrame({"p": [1], "i": [1], "rt": [np.nan]}), "p", "i", "rt")

    empty_speed = irt.eyeprocess_speed_accuracy_profile(pd.DataFrame(columns=["p", "y", "rt"]), "p", "y", "rt")
    assert empty_speed.person.empty and math.isnan(empty_speed.pooled_correlation)
    empty_group = irt._group_channel_profile(pd.DataFrame(columns=["id", "x"]), "id", ["x"], "id")
    assert empty_group.empty


def test_process_alignment_explicit_missing_and_correlation_paths():
    profile = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "mean_dwell": [1.0, 2.0, 4.0]})
    out = irt.eyeprocess_irt_process_alignment(_items(), profile, process_columns=["missing", "mean_dwell"])
    assert out.correlations["channel"].tolist() == ["mean_dwell"]
    assert np.isfinite(out.correlations.loc[0, "correlation_discrimination"])
    assert np.isfinite(out.correlations.loc[0, "correlation_difficulty"])

    flat = profile.copy(); flat["mean_dwell"] = 1.0
    out2 = irt.eyeprocess_irt_process_alignment(_items(), flat)
    assert math.isnan(out2.correlations.loc[0, "correlation_discrimination"])


def test_external_engine_and_fit_validation_residuals():
    with pytest.raises(EyeProcessValidationError, match="non-empty scalar"):
        irt.run_eyeprocess_equateirt("")
    with pytest.raises(EyeProcessValidationError, match="only accepts"):
        irt.fit_eyeprocess_mirt(None, engine="wrong")

    gated = irt._gated_engine("mirt", "fit_eyeprocess_mirt")
    assert irt.validate_eyeprocess_external_irt_fit(gated, engine="mirt")
    with pytest.raises(EyeProcessValidationError, match="Engine mismatch"):
        irt.validate_eyeprocess_external_irt_fit(gated, engine="TAM")

    bad_external = irt.EyeResult({"engine": "mirt", "fit": None}, eyeprocess_class="eye_external_irt_fit")
    with pytest.raises(EyeProcessValidationError, match="NULL fit"):
        irt.validate_eyeprocess_external_irt_fit(bad_external)
    bad_gated = irt.EyeResult({"engine": "mirt", "fit": object()}, eyeprocess_class="eye_gated_irt_engine")
    with pytest.raises(EyeProcessValidationError, match="must have NULL fit"):
        irt.validate_eyeprocess_external_irt_fit(bad_gated)


def test_simulation_recovery_design_short_circuits_and_success_variants():
    items = _items()
    with pytest.raises(EyeProcessValidationError, match="n_persons"):
        irt.simulate_eyeprocess_irt_binary(0, items)
    with pytest.raises(EyeProcessValidationError, match="missing_rate"):
        irt.simulate_eyeprocess_irt_binary(2, items, missing_rate=1)
    with pytest.raises(EyeProcessValidationError, match="missing_rate"):
        irt.simulate_eyeprocess_irt_binary(2, items, testlet_sd=-1)
    with pytest.raises(EyeProcessValidationError, match="theta must"):
        irt.simulate_eyeprocess_irt_binary(2, items, theta=[0])
    sim = irt.simulate_eyeprocess_irt_binary(4, items, missing_rate=0.2, testlet_sd=0.3, seed=4)
    assert sim.responses.shape == (4, 3)

    bad_designs = [
        {"sample_size": []},
        {"sample_size": [19]},
        {"n_items": []},
        {"n_items": [3]},
        {"missing_rate": []},
        {"missing_rate": [np.nan]},
        {"missing_rate": [-0.1]},
        {"testlet_sd": []},
        {"testlet_sd": [np.nan]},
        {"testlet_sd": [-0.1]},
        {"replications": 0},
        {"seed": 0},
    ]
    for kwargs in bad_designs:
        with pytest.raises(EyeProcessValidationError, match="invalid recovery design"):
            irt.eyeprocess_irt_recovery_design(**kwargs)

    design = irt.eyeprocess_irt_recovery_design(sample_size=[20], n_items=[4], missing_rate=[0], testlet_sd=[0], replications=1, seed=1)
    with pytest.raises(EyeProcessValidationError, match="design must"):
        irt.run_eyeprocess_irt_recovery(pd.DataFrame())
    with pytest.raises(EyeProcessValidationError, match="requires engine"):
        irt.run_eyeprocess_irt_recovery(design, engine="TAM")


def test_recovery_summary_failures_and_one_dimensional_sbc():
    with pytest.raises(EyeProcessValidationError, match="recovery_result"):
        irt.eyeprocess_irt_recovery_summary({})
    empty_result = irt.EyeResult({"estimates": pd.DataFrame()}, eyeprocess_class="eye_irt_recovery_result")
    assert irt.eyeprocess_irt_recovery_summary(empty_result).empty

    estimates = pd.DataFrame(
        {
            "scenario_id": ["S1", "S1", "S1"],
            "a_truth": [0.8, 1.0, 1.2],
            "a_estimate": [0.9, 1.1, 1.1],
            "b_truth": [-0.5, 0.0, 0.5],
            "b_estimate": [-0.4, 0.1, 0.4],
        }
    )
    res = irt.EyeResult({"estimates": estimates}, eyeprocess_class="eye_irt_recovery_result")
    summary = irt.eyeprocess_irt_recovery_summary(res)
    assert len(summary) == 2 and summary.correlation.notna().all()

    with pytest.raises(EyeProcessValidationError, match="recovery_result"):
        irt.eyeprocess_irt_recovery_failures({})
    design = pd.DataFrame({"scenario_id": ["S1"], "replications": [2]})
    nofail = irt.EyeResult({"design": design, "failures": pd.DataFrame()}, eyeprocess_class="eye_irt_recovery_result")
    ft = irt.eyeprocess_irt_recovery_failures(nofail)
    assert ft.loc[0, "failures"] == 0
    somefail = irt.EyeResult({"design": design, "failures": pd.DataFrame({"scenario_id": ["S1"]})}, eyeprocess_class="eye_irt_recovery_result")
    assert irt.eyeprocess_irt_recovery_failures(somefail).loc[0, "failures"] == 1

    ranks = irt.eyeprocess_irt_sbc_ranks([0.0], [-1.0, 0.0, 1.0], randomize_ties=False)
    assert ranks.tolist() == [1]


def test_mirt_cdm_remaining_guards_and_scalar_expansion():
    with pytest.raises(EyeProcessValidationError, match="created by"):
        irt.eyeprocess_mirt_loading_audit({})
    with pytest.raises(EyeProcessValidationError, match="dimension mismatch"):
        irt.eyeprocess_mirt_information_matrix([0.0], [1.0, 2.0])
    with pytest.raises(EyeProcessValidationError, match="item_id/testlet"):
        irt.eyeprocess_irt_testlet_spec(["I1", "I1"], ["T1", "T2"])
    with pytest.raises(EyeProcessValidationError, match="compatible binary"):
        irt.eyeprocess_cdm_dina_ideal_response([[1, 0]], [[1]])

    eta = np.array([[1, 0], [0, 1]])
    p = irt.eyeprocess_cdm_dina_probability(eta, slip=0.1, guess=[0.2, 0.3])
    assert p.shape == eta.shape

    invalid_profiles = [
        [0.5, 0.5],
        np.empty((0, 2)),
        [[1.0]],
        [[np.nan, 0.0]],
        [[-0.1, 1.1]],
        [[0.8, 0.3]],
    ]
    for bad in invalid_profiles:
        with pytest.raises(EyeProcessValidationError, match="profile probabilities"):
            irt.eyeprocess_cdm_classification_uncertainty(bad)


def test_advanced_fit_targeting_and_zero_observation_paths():
    obs = pd.DataFrame([[np.nan, np.nan], [1.0, 0.0]], index=["P0", "P1"], columns=["I1", "I2"])
    exp = pd.DataFrame([[np.nan, np.nan], [0.8, 0.2]], index=obs.index, columns=obs.columns)
    item = irt.eyeprocess_irt_infit_outfit(obs, exp, by="item")
    person = irt.eyeprocess_irt_infit_outfit(obs, exp, by="person")
    assert len(item) == 2 and person.loc[0, "n"] == 0 and math.isnan(person.loc[0, "outfit"])
    with pytest.raises(EyeProcessValidationError, match="by must"):
        irt.eyeprocess_irt_infit_outfit([[1]], [[0.5]], by="bad")

    lz = irt.eyeprocess_irt_person_fit_lz(obs, exp)
    assert math.isnan(lz.loc[0, "lz"])
    with pytest.raises(EyeProcessValidationError, match="0, 0.5"):
        irt.eyeprocess_irt_person_fit_lz([[1]], [[0.5]], min_probability=0.5)

    with pytest.raises(EyeProcessValidationError, match="target must"):
        irt.eyeprocess_irt_bank_coverage(_items(), target=(1, -1))
    with pytest.raises(EyeProcessValidationError, match="does not overlap"):
        irt.eyeprocess_irt_bank_coverage(_items(), theta=[-4, -3], target=(1, 2))

    with pytest.raises(EyeProcessValidationError, match="at least one finite"):
        irt.eyeprocess_irt_targeting_gap([np.nan], _items())
    for breaks in ([0, 1], [0, np.nan, 1], [0, 1, 0.5]):
        with pytest.raises(EyeProcessValidationError, match="strictly increasing"):
            irt.eyeprocess_irt_targeting_gap([0.5], _items(), breaks=breaks)
    with pytest.raises(EyeProcessValidationError, match="span"):
        irt.eyeprocess_irt_targeting_gap([3.0], _items(), breaks=[-1, 0, 1])


def test_prior_grid_summary_model_card_short_circuit_guards():
    for kwargs in (
        {"discrimination": "bad"},
        {"difficulty": "bad"},
        {"guessing": "bad"},
    ):
        with pytest.raises(EyeProcessValidationError, match="invalid prior family"):
            irt.eyeprocess_irt_prior_spec(**kwargs)
    for shape in ([1.0], [1.0, np.nan], [1.0, 0.0]):
        with pytest.raises(EyeProcessValidationError, match="guessing_shape"):
            irt.eyeprocess_irt_prior_spec(guessing_shape=shape)

    grid_cases = [
        {"discrimination_scale": [np.nan]},
        {"discrimination_scale": [0]},
        {"difficulty_scale": [np.nan]},
        {"difficulty_scale": [0]},
        {"guessing_mean": [np.nan]},
        {"guessing_mean": [0]},
        {"guessing_mean": [1]},
    ]
    for kwargs in grid_cases:
        with pytest.raises(EyeProcessValidationError, match="Prior grid"):
            irt.eyeprocess_irt_prior_sensitivity_grid(**kwargs)

    empty = irt.eyeprocess_irt_prior_sensitivity_summary(pd.DataFrame({"prior_id": ["P1"], "estimate": [np.nan]}))
    assert empty.n_finite == 0 and math.isnan(empty.median)
    finite = irt.eyeprocess_irt_prior_sensitivity_summary(pd.DataFrame({"prior_id": ["P1", "P2"], "estimate": [1.0, 2.0]}))
    assert finite.n_finite == 2 and np.isfinite(finite.sd)

    with pytest.raises(EyeProcessValidationError, match="spec must"):
        irt.eyeprocess_irt_model_card({})
    spec = irt.eyeprocess_joint_process_irt_spec()
    card = irt.eyeprocess_irt_model_card(spec, engine_status={}, identification=[], fit_evidence="present")
    audit = irt.eyeprocess_irt_model_card_audit(card)
    present = dict(zip(audit.field, audit.present))
    assert not present["engine_status"] and not present["identification"] and present["fit_evidence"]
    with pytest.raises(EyeProcessValidationError, match="card must"):
        irt.eyeprocess_irt_model_card_audit({})
