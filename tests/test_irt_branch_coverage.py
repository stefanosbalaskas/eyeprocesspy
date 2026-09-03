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
            "a": [0.8, 1.2, 1.0],
            "b": [-0.5, 0.2, 1.0],
            "c": [0.0, 0.1, 0.2],
            "d": [1.0, 0.95, 0.9],
        }
    )


def _responses() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 0.0, 1.0], [0.0, 1.0, np.nan], [1.0, np.nan, 0.0], [np.nan, np.nan, np.nan]],
        index=["P1", "P2", "P3", "P4"],
        columns=["I1", "I2", "I3"],
    )


def test_private_dataframe_and_numeric_helpers_cover_error_contracts():
    scalar = irt._as_df({"a": 1}, "scalar")
    assert scalar.shape == (1, 1)
    with pytest.raises(EyeProcessValidationError):
        irt._as_df(object(), "bad")

    with pytest.raises(EyeProcessValidationError):
        irt._req_cols(pd.DataFrame({"a": [1]}), ["b"])

    assert math.isnan(irt._finite_mean([np.nan]))
    assert math.isnan(irt._finite_sd([1.0]))
    assert math.isnan(irt._safe_quantile([], 0.5))
    assert np.isnan(irt._safe_quantile([], [0.1, 0.9])).all()
    assert irt._safe_quantile([1, 2, 3], 0.5) == 2.0

    h1 = irt._stable_hash({"x": np.array([1, 2]), "y": np.float64(1.5)})
    h2 = irt._stable_hash({"y": 1.5, "x": [1, 2]})
    assert h1 == h2

    for bad in ([], [1.0, np.nan]):
        with pytest.raises(EyeProcessValidationError):
            irt._theta(bad)


def test_item_binary_probability_and_scalar_validation():
    with pytest.raises(EyeProcessValidationError):
        irt._item_pars(pd.DataFrame({"item_id": ["I1"], "a": [1.0]}))

    dup = _items()
    dup.loc[1, "item_id"] = "I1"
    with pytest.raises(EyeProcessValidationError):
        irt._item_pars(dup)

    bad_a = _items()
    bad_a.loc[0, "a"] = 0
    with pytest.raises(EyeProcessValidationError):
        irt._item_pars(bad_a)

    bad_cd = _items()
    bad_cd.loc[0, "c"] = 0.8
    bad_cd.loc[0, "d"] = 0.7
    with pytest.raises(EyeProcessValidationError):
        irt._item_pars(bad_cd)

    defaults = irt._item_pars(_items().drop(columns=["c", "d"]))
    assert np.allclose(defaults["c"], 0)
    assert np.allclose(defaults["d"], 1)

    with pytest.raises(EyeProcessValidationError):
        irt._binary_matrix([0, 1])
    with pytest.raises(EyeProcessValidationError):
        irt._binary_matrix([[0, 2]])
    with pytest.raises(EyeProcessValidationError):
        irt._binary_matrix([[0, np.nan]], allow_na=False)

    with pytest.raises(EyeProcessValidationError):
        irt._prob_matrix([[0.2]], (1, 2))
    with pytest.raises(EyeProcessValidationError):
        irt._prob_matrix([[0.0, 1.0]], (1, 2))

    with pytest.raises(EyeProcessValidationError):
        irt._scalar([1, 2], "x")
    with pytest.raises(EyeProcessValidationError):
        irt._scalar(0, "x", positive=True)
    with pytest.raises(EyeProcessValidationError):
        irt._scalar(-1, "x", nonnegative=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": "bad"},
        {"identification": "bad"},
        {"engine": "bad"},
        {"status": "bad"},
        {"dimensions": 0},
        {"notes": 123},
    ],
)
def test_model_spec_invalid_choices(kwargs):
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_model_spec(**kwargs)


def test_model_spec_validation_and_identification_audit_branches():
    spec = irt.eyeprocess_irt_model_spec(
        family="2pl",
        process_channels=["dwell", "dwell", "pupil"],
        notes="declared",
    )
    assert spec.process_channels == ["dwell", "pupil"]
    assert irt.validate_eyeprocess_irt_model_spec(spec)

    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_irt_model_spec({})
    broken = irt.EyeResult(spec.copy(), eyeprocess_class="eyeprocess_irt_model_spec")
    broken.pop("family")
    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_irt_model_spec(broken)

    weak = irt.eyeprocess_irt_identification_audit(spec, n_items=2, n_persons=5)
    assert not weak.valid
    assert len(weak.warnings) >= 2

    strong = irt.eyeprocess_irt_identification_audit(
        spec,
        constraints={"theta_mean_fixed": True, "theta_sd_fixed": True, "anchor_items": ["I1", "I2"]},
        n_items=10,
        n_persons=20,
    )
    assert strong.valid

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_identification_audit(spec, n_items=-1)
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_identification_audit(spec, n_persons=-1)


def test_sparse_design_and_core_probability_guardrails():
    data = pd.DataFrame(
        {
            "person": ["P1", "P1", "P2", "P3"],
            "item": ["I1", "I2", "I1", "I2"],
            "response": [1, np.nan, 0, 1],
        }
    )
    audit = irt.eyeprocess_irt_sparse_design_audit(
        data, "person", "item", response="response", min_person_items=2, min_item_persons=2
    )
    assert audit.n_observed == 3
    assert "P2" in audit.sparse_persons

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_sparse_design_audit(data, "", "item")
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_sparse_design_audit(data, "person", "item", min_person_items=0)

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_3pl_probability([0], c=1.0)
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_4pl_probability([0], c=0.8, d=0.7)
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_grm_probability([0], thresholds=[])
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_grm_probability([0], thresholds=[0.5, 0.1])

    p = irt.eyeprocess_irt_2pl_probability([-1, 0, 1], a=1.2, b=0.1)
    assert np.all(np.diff(p) > 0)


def test_information_precision_and_diagnostic_error_paths():
    items = _items()
    theta = np.linspace(-2, 2, 9)
    info = irt.eyeprocess_irt_test_information(theta, items)
    assert np.all(info["information"] > 0)

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_information_area([0], [1])
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_measurement_precision_profile(theta, items, target=[1, -1])
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_measurement_precision_profile([-5, 5], items, target=[-1, 1])

    y = _responses().to_numpy()
    probs = np.full_like(y, 0.5, dtype=float)
    probs[np.isnan(y)] = np.nan

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_item_fit_residuals(y, probs, item_ids=["I1"])
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_person_fit_residuals(y, probs, person_ids=["P1"])

    item_fit = irt.eyeprocess_irt_item_fit_residuals(y, probs)
    person_fit = irt.eyeprocess_irt_person_fit_residuals(y, probs)
    assert (item_fit["n"] >= 0).all()
    assert (person_fit["n"] >= 0).all()
    assert person_fit.iloc[-1]["n"] == 0

    q3 = irt.eyeprocess_irt_q3(y, probs)
    assert q3.shape == (3, 3)
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_local_dependence_pairs(np.zeros((2, 3)))

    pair = pd.DataFrame([[np.nan, -0.3], [-0.3, np.nan]], columns=["A", "B"], index=["A", "B"])
    abs_pairs = irt.eyeprocess_irt_local_dependence_pairs(pair, threshold=0.2, absolute=True)
    pos_pairs = irt.eyeprocess_irt_local_dependence_pairs(pair, threshold=0.2, absolute=False)
    assert len(abs_pairs) == 1
    assert pos_pairs.empty


def test_diagnostic_audits_and_ppc_statistic_branches():
    y = _responses().to_numpy()
    extreme = irt.eyeprocess_irt_extreme_score_audit(y, lower_fraction=0.1, upper_fraction=0.9)
    assert len(extreme) == 4
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_extreme_score_audit(y, lower_fraction=0.9, upper_fraction=0.1)

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_threshold_order_audit("", [0, 1])
    reversed_thresholds = irt.eyeprocess_irt_threshold_order_audit("I1", [0.5, 0.0, 1.0])
    assert not reversed_thresholds.ordered
    assert reversed_thresholds.reversals

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_monotonicity_audit([0], [0.5])
    mono = irt.eyeprocess_irt_monotonicity_audit([0, 1, 2], [0.2, 0.4, 0.3])
    assert not mono.monotone_non_decreasing

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_category_function_audit([[1.0]])
    cat = irt.eyeprocess_irt_category_function_audit([[0.6, 0.6], [0.4, 0.6]])
    assert not cat.rows_sum_to_one

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_parameter_plausibility_audit(_items(), discrimination=[2, 1])

    observed = np.array([[1, 0], [0, 1]], dtype=float)
    replicated = np.array([observed, 1 - observed], dtype=float)
    for statistic in ("mean_score", "score_sd", "item_means", "max_item_residual"):
        out = irt.eyeprocess_irt_ppc_discrepancy(observed, replicated, statistic=statistic)
        assert 0 <= out.posterior_predictive_p <= 1

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_ppc_discrepancy(observed, replicated, statistic="bad")
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_ppc_discrepancy(observed, np.zeros((2, 2)), statistic="mean_score")


def test_score_likelihood_eap_map_mle_guardrails():
    items = _items()
    response = [1, 0, np.nan]

    assert irt._response_loglik(0.0, [np.nan, np.nan, np.nan], items) == 0.0
    with pytest.raises(EyeProcessValidationError):
        irt._response_loglik(0.0, [1, 2, 0], items)

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_eap_score(response, items, theta_grid=[-1, 0, 1])
    eap = irt.eyeprocess_irt_eap_score(response, items)
    assert np.isfinite(eap.estimate)
    assert eap.se > 0

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_map_score(response, items, bounds=[1, -1])
    mapped = irt.eyeprocess_irt_map_score(response, items)
    assert -6 <= mapped.estimate <= 6

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_mle_score(response, items, bounds=[1, -1])
    mle = irt.eyeprocess_irt_mle_score(response, items)
    assert -6 <= mle.estimate <= 6


def test_process_bundle_profiles_and_missingness_contracts():
    data = pd.DataFrame(
        {
            "person": ["P1", "P1", "P2", "P2"],
            "item": ["I1", "I2", "I1", "I2"],
            "response": [1, 0, 1, np.nan],
            "rt": [0.5, 0.8, 0.6, 0.7],
            "dwell": [2.0, np.nan, 1.0, 3.0],
        }
    )
    spec = irt.eyeprocess_joint_process_irt_spec(
        response_family="2pl", time_model="lognormal", process_channels=["dwell", "dwell"]
    )
    assert spec.process_channels == ["dwell"]
    assert irt.validate_eyeprocess_joint_process_irt_spec(spec)
    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_joint_process_irt_spec({})

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_joint_process_irt_spec(response_family="bad")

    bundle = irt.eyeprocess_process_irt_data_bundle(
        data, "person", "item", "response", response_time="rt", process=["dwell"]
    )
    assert bundle.n_persons == 2

    bad_y = data.copy()
    bad_y.loc[0, "response"] = 2
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_process_irt_data_bundle(bad_y, "person", "item", "response")

    bad_rt = data.copy()
    bad_rt.loc[0, "rt"] = 0
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_process_irt_data_bundle(
            bad_rt, "person", "item", "response", response_time="rt"
        )

    profile = irt.eyeprocess_response_time_profile(data, "person", "item", "rt")
    assert profile.n == 4
    no_rt = data.assign(rt=np.nan)
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_response_time_profile(no_rt, "person", "item", "rt")

    speed = irt.eyeprocess_speed_accuracy_profile(data, "person", "response", "rt")
    assert len(speed.person) == 2

    item_profile = irt.eyeprocess_process_item_profile(data, "item", ["dwell"])
    person_profile = irt.eyeprocess_process_person_profile(data, "person", ["dwell"])
    assert len(item_profile) == 2
    assert len(person_profile) == 2

    empty_pattern = irt.eyeprocess_process_missingness_pattern(
        data.iloc[0:0], "response", ["dwell"]
    )
    assert empty_pattern.empty
    pattern = irt.eyeprocess_process_missingness_pattern(data, "response", ["dwell"])
    assert math.isclose(pattern["fraction"].sum(), 1.0)

    mmap = irt.eyeprocess_multichannel_measurement_map(
        channels=["rt", "rt", "dwell"], role=["timing", "gaze"]
    )
    assert mmap["channel"].tolist() == ["accuracy", "rt", "dwell"]
    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_multichannel_measurement_map(channels=["rt", "dwell"], role=["one"])


def test_external_engine_registry_and_gate_validation():
    registry = irt.eyeprocess_irt_engine_registry()
    assert not registry["available"].any()
    assert len(registry) == 8

    for engine in registry["engine"]:
        status = irt.eyeprocess_irt_engine_status(engine)
        assert status.iloc[0]["engine"] == engine

    with pytest.raises(EyeProcessValidationError):
        irt.eyeprocess_irt_engine_status("missing")

    gated = irt.fit_eyeprocess_mirt([[1, 0]])
    assert gated.status == "gated"
    assert irt.validate_eyeprocess_external_irt_fit(gated, engine="mirt")

    with pytest.raises(EyeProcessValidationError):
        irt.fit_eyeprocess_mirt([[1]], engine="TAM")
    with pytest.raises(EyeProcessValidationError):
        irt.fit_eyeprocess_tam([[1]], model="bad")
    with pytest.raises(EyeProcessValidationError):
        irt.fit_eyeprocess_erm([[1]], model="bad")
    with pytest.raises(EyeProcessValidationError):
        irt.run_eyeprocess_equateirt("")
    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_external_irt_fit({})
    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_external_irt_fit(gated, engine="TAM")

    fake = irt._result("eye_external_irt_fit", engine="mirt", fit=None)
    with pytest.raises(EyeProcessValidationError):
        irt.validate_eyeprocess_external_irt_fit(fake)
