from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.negative_controls_09 as nc


class _BadArray:
    def __array__(self, *args, **kwargs):
        raise RuntimeError("cannot array")


class _BadFrame:
    def __iter__(self):
        raise RuntimeError("cannot iterate")


def _control(results: pd.DataFrame):
    return {
        "results": results,
        "eyeprocess_class": "eye_process_negative_controls",
    }


def test_private_coercion_collection_group_and_capture_paths(monkeypatch):
    df = pd.DataFrame({"x": [1, 2], "g": ["a", "b"]})
    assert nc._as_frame(df).equals(df)
    assert nc._as_frame({"x": [1]}).shape == (1, 1)
    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        nc._as_frame(_BadFrame())

    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        nc._require_columns(df, ["x", "missing"])
    nc._require_columns(df, [None, "x"])

    np.testing.assert_allclose(nc._numeric_vector(pd.Series([1, "2"])), [1, 2])
    np.testing.assert_allclose(nc._numeric_vector(np.array([3, 4])), [3, 4])
    assert math.isnan(nc._numeric_vector("bad")[0])
    np.testing.assert_allclose(nc._numeric_vector(5), [5])
    assert math.isnan(nc._numeric_vector(_BadArray())[0])
    assert math.isnan(nc._first_numeric([]))

    assert nc._as_list(None) == []
    assert nc._as_list("x") == ["x"]
    assert nc._as_list(pd.Series([1, 2])) == [1, 2]
    assert nc._as_list((1, 2)) == [1, 2]
    assert nc._as_list(4) == [4]
    assert nc._recycle([], 3, missing="M") == ["M"] * 3
    assert nc._recycle([1, 2], 3) == [1, 2, 1]
    assert nc._normalize_group_columns(None) == []
    assert nc._normalize_group_columns("g") == ["g"]
    assert nc._normalize_group_columns(["g", 2]) == ["g", "2"]

    groups = nc._group_indices(df, [])
    assert len(groups) == 1 and groups[0].tolist() == [0, 1]
    grouped = nc._group_indices(pd.DataFrame({"g": ["a", "a", "b"]}), ["g"])
    assert sorted(len(x) for x in grouped) == [1, 2]

    def warns(x):
        warnings.warn("hello", RuntimeWarning)
        return x + 1

    captured = nc._capture_call(warns, 1)
    assert captured["value"] == 2 and captured["warnings"] == ["hello"]
    failed = nc._capture_call(lambda _: (_ for _ in ()).throw(RuntimeError("boom")), None)
    assert failed["value"] is None and "boom" in failed["error"]

    assert nc._default_control_extract(1.5).loc[0, "effect"] == pytest.approx(1.5)
    assert nc._default_control_extract({"effect": 2.5}).loc[0, "effect"] == pytest.approx(2.5)
    frame = pd.DataFrame({"effect": [3.5]})
    assert nc._default_control_extract(frame).equals(frame)
    with pytest.raises(ep.EyeProcessValidationError, match="analysis_fun output"):
        nc._default_control_extract("unsupported")

    assert math.isnan(nc._quantile(np.array([np.nan]), 0.5))
    real_quantile = np.quantile
    calls = []

    def legacy_quantile(a, q, **kwargs):
        calls.append(kwargs)
        if "method" in kwargs:
            raise TypeError("legacy")
        return real_quantile(a, q, **kwargs)

    monkeypatch.setattr(nc.np, "quantile", legacy_quantile)
    assert nc._quantile(np.array([1.0, 2.0, 3.0]), 0.5) == pytest.approx(2.0)
    assert len(calls) == 2


def test_provenance_unit_recycling_missing_and_leakage_empty_paths():
    p = ep.process_feature_time_provenance(
        ["a", "b"],
        [1],
        [2, 3],
        source=None,
        transformation=None,
        unit=[],
    )
    assert p["unit"].tolist() == ["None", "None"]
    assert p["source"].isna().all()
    with pytest.raises(ep.EyeProcessValidationError, match="feature names"):
        ep.process_feature_time_provenance([None], [1], [2])
    with pytest.raises(ep.EyeProcessValidationError, match="finite"):
        ep.process_feature_time_provenance(["x"], [1], [np.inf])

    empty = pd.DataFrame(columns=["feature", "available_at", "outcome_at"])
    audit = ep.audit_temporal_leakage(empty)
    assert audit["n_features"] == 0 and math.isnan(audit["flagged_fraction"])
    strict = ep.audit_temporal_leakage(
        pd.DataFrame({"feature": ["x"], "available_at": [1.0], "outcome_at": [1.0]}),
        allow_equal=False,
    )
    assert strict["n_flagged"] == 1
    with pytest.raises(ep.EyeProcessValidationError, match="tolerance"):
        ep.audit_temporal_leakage(p, tolerance=np.nan)


def test_feature_availability_all_cutoff_forms_and_guards():
    p = ep.process_feature_time_provenance(["a", "b"], [1, 3], [5, 5])
    with pytest.raises(ep.EyeProcessValidationError, match="cutoff cannot"):
        ep.validate_feature_availability(p, None)
    with pytest.raises(ep.EyeProcessValidationError, match="cutoff cannot"):
        ep.validate_feature_availability(p, [])

    s = pd.Series([1.0, 4.0], index=["a", "b"])
    out = ep.validate_feature_availability(p, s)
    assert out["available"].tolist() == [True, True]

    with pytest.raises(ep.EyeProcessValidationError, match="named by feature"):
        ep.validate_feature_availability(p, [1, 4])
    with pytest.raises(ep.EyeProcessValidationError, match="required for every feature"):
        ep.validate_feature_availability(p, {"a": 1, "c": 4})
    with pytest.raises(ep.EyeProcessValidationError, match="finite"):
        ep.validate_feature_availability(p, {"a": 1, "b": np.nan})


def test_outcome_blind_missing_noncallable_and_warning_paths():
    data = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(ep.EyeProcessValidationError, match="outcome column"):
        ep.outcome_blind_feature_audit(data, "missing", lambda d: d)
    with pytest.raises(ep.EyeProcessValidationError, match="feature_fun"):
        ep.outcome_blind_feature_audit(data, "y", object())

    def warn_fun(frame):
        warnings.warn("feature-warning", UserWarning)
        return frame["x"].sum()

    res = ep.outcome_blind_feature_audit(data, "y", warn_fun)
    assert res["status"] == "pass" and res["warnings"] == ["feature-warning"]


def test_permutation_seed_outcome_group_and_shift_helper_edges():
    data = pd.DataFrame({"g": ["a", "a", "b", "b"], "y": [1, 2, 3, 4]})
    with pytest.raises(ep.EyeProcessValidationError, match="outcome must"):
        ep.process_negative_control_permute(data, "")
    with pytest.raises(ep.EyeProcessValidationError, match="seed"):
        ep.process_negative_control_permute(data, "y", seed=np.nan)
    huge = np.iinfo(np.int32).max + 5
    p = ep.process_negative_control_permute(data, "y", seed=huge)
    assert p.attrs["negative_control"]["seed"] == 5
    grouped = ep.process_negative_control_permute(data, "y", within="g", seed=1)
    assert grouped.attrs["negative_control"]["within"] == ["g"]

    empty = pd.Series([], dtype=float)
    assert nc._shift_values(empty, 1).empty
    zero = nc._shift_values(pd.Series([1, 2]), 0)
    assert zero.tolist() == [1, 2]
    assert nc._shift_values(pd.Series([1, 2]), -2).isna().all()
    assert nc._shift_values(pd.Series([1, 2, 3]), -1).tolist()[:2] == [2, 3]

    with pytest.raises(ep.EyeProcessValidationError, match="lag"):
        ep.process_negative_control_shift(data, "y", lag=1.5)
    shift0 = ep.process_negative_control_shift(data, "y", lag=0)
    assert shift0["y"].tolist() == data["y"].tolist()


def test_placebo_summary_empty_single_and_group_empty_subset_guards():
    empty = nc._placebo_summary(pd.DataFrame({"v": [np.nan]}), "v", 0.0)
    assert empty["n"] == 0 and math.isnan(empty["mean"])
    single = nc._placebo_summary(pd.DataFrame({"v": [2.0]}), "v", 1.0)
    assert single["n"] == 1 and math.isnan(single["sd"])

    data = pd.DataFrame({"g": ["a", "b"], "t": [0.0, 1.0], "v": [1.0, 2.0]})
    ungrouped = ep.placebo_window_audit(data, "t", "v", [0, 1], expected=0)
    assert len(ungrouped) == 1 and ungrouped.loc[0, "n"] == 2
    grouped_empty = ep.placebo_window_audit(data, "t", "v", [5, 6], by=["g"])
    assert grouped_empty.empty

    with pytest.raises(ep.EyeProcessValidationError, match="window"):
        ep.placebo_window_audit(data, "t", "v", [0, np.nan])
    with pytest.raises(ep.EyeProcessValidationError, match="expected"):
        ep.placebo_window_audit(data, "t", "v", [0, 1], expected=np.nan)


def test_run_negative_controls_validation_extraction_and_warning_paths():
    data = pd.DataFrame({"g": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0]})

    for controls in ([], ["bad"]):
        with pytest.raises(ep.EyeProcessValidationError, match="controls must"):
            ep.run_process_negative_controls(data, "y", lambda d: 0.0, controls=controls)
    for reps in (np.nan, 1.5):
        with pytest.raises(ep.EyeProcessValidationError, match="replications"):
            ep.run_process_negative_controls(data, "y", lambda d: 0.0, replications=reps)

    with pytest.raises(ep.EyeProcessValidationError, match="shift_lags cannot"):
        ep.run_process_negative_controls(data, "y", lambda d: 0.0, controls="shift", shift_lags=[])
    for lags in ([np.nan], [1.5], [0]):
        with pytest.raises(ep.EyeProcessValidationError, match="shift_lags"):
            ep.run_process_negative_controls(data, "y", lambda d: 0.0, controls="shift", shift_lags=lags)

    with pytest.raises(ep.EyeProcessValidationError, match="functions"):
        ep.run_process_negative_controls(data, "y", object())
    with pytest.raises(ep.EyeProcessValidationError, match="functions"):
        ep.run_process_negative_controls(data, "y", lambda d: 0.0, extract_fun=object())
    with pytest.raises(ep.EyeProcessValidationError, match="seed"):
        ep.run_process_negative_controls(data, "y", lambda d: 0.0, seed=np.nan)

    both = ep.run_process_negative_controls(
        data,
        "y",
        lambda d: {"effect": float(pd.to_numeric(d["y"]).mean())},
        controls=["permutation", "shift"],
        replications=2,
        shift_lags=[-1, 1],
        within="g",
        seed=2,
    )
    assert set(both["results"]["control"]) == {"permutation", "shift"}

    dfout = ep.run_process_negative_controls(
        data,
        "y",
        lambda d: pd.DataFrame({"effect": [1.0, 2.0]}),
        controls="permutation",
        replications=1,
    )
    assert len(dfout["results"]) == 2

    unsupported = ep.run_process_negative_controls(
        data,
        "y",
        lambda d: "bad",
        controls="permutation",
        replications=1,
    )
    assert unsupported["results"]["effect"].isna().all()
    assert unsupported["results"]["error"].notna().all()

    def warns(d):
        warnings.warn("analysis warning", RuntimeWarning)
        return 1.0

    warned = ep.run_process_negative_controls(data, "y", warns, controls="permutation", replications=1)
    assert "analysis warning" in warned["results"].loc[0, "warnings"]

    def extract_fails(value):
        raise RuntimeError("extract failed")

    ef = ep.run_process_negative_controls(
        data, "y", lambda d: 1.0, controls="permutation", replications=1, extract_fun=extract_fails
    )
    assert ef["results"]["effect"].isna().all()
    assert ef["results"]["error"].str.contains("extract failed").all()


def test_run_negative_controls_uncoercible_and_empty_extractor_outputs():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    class BadOutput:
        def __iter__(self):
            raise RuntimeError("no frame")

    bad = ep.run_process_negative_controls(
        data,
        "y",
        lambda d: 1.0,
        controls="permutation",
        replications=1,
        extract_fun=lambda value: BadOutput(),
    )
    assert bad["results"]["effect"].isna().all()

    empty = ep.run_process_negative_controls(
        data,
        "y",
        lambda d: 1.0,
        controls="permutation",
        replications=1,
        extract_fun=lambda value: pd.DataFrame(),
    )
    assert empty["results"]["effect"].isna().all()


def test_summary_object_threshold_empty_finite_and_null_benchmark_edges():
    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_negative_controls"):
        ep.summarise_process_negative_controls({})
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        ep.summarise_process_negative_controls(_control(pd.DataFrame({"control": ["p"]})))
    with pytest.raises(ep.EyeProcessValidationError, match="threshold"):
        ep.summarise_process_negative_controls(_control(pd.DataFrame({"control": ["p"], "effect": [0]})), threshold=np.nan)

    c = _control(pd.DataFrame({"control": ["p", "q"], "effect": [np.nan, 1.0]}))
    summary = ep.summarise_process_negative_controls(c, threshold=0.5)
    p = summary.loc[summary.control.eq("p")].iloc[0]
    q = summary.loc[summary.control.eq("q")].iloc[0]
    assert p["n_finite"] == 0 and math.isnan(p["mean"]) and math.isnan(p["exceedance_rate"])
    assert q["n_finite"] == 1 and math.isnan(q["sd"])

    empty = ep.process_null_benchmark(np.nan, [])
    assert empty["n_null"] == 0 and math.isnan(empty["percentile"])
    one = ep.process_null_benchmark(2.0, [1.0])
    assert one["n_null"] == 1 and math.isnan(one["null_sd"]) and math.isnan(one["standardized_distance"])
    flat = ep.process_null_benchmark(2.0, [1.0, 1.0])
    assert flat["null_sd"] == 0 and math.isnan(flat["standardized_distance"])


def test_concordance_invalid_no_finite_false_and_true_paths():
    c_nan = _control(pd.DataFrame({"control": ["p"], "effect": [np.nan]}))
    with pytest.raises(ep.EyeProcessValidationError, match="tolerance"):
        ep.negative_control_concordance(c_nan, tolerance=-1)
    none = ep.negative_control_concordance(c_nan)
    assert none["all_within_tolerance"] is None
    c_bad = _control(pd.DataFrame({"control": ["p", "q"], "effect": [0.01, 0.2]}))
    assert ep.negative_control_concordance(c_bad, tolerance=0.05)["all_within_tolerance"] is False
    c_good = _control(pd.DataFrame({"control": ["p", "q"], "effect": [0.01, -0.02]}))
    assert ep.negative_control_concordance(c_good, tolerance=0.05)["all_within_tolerance"] is True
