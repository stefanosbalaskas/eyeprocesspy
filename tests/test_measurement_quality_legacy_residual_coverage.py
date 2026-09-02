from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.measurement_quality_legacy as mq


def _unc_data():
    return pd.DataFrame(
        {
            "person": ["p1", "p1", "p2", "p2"],
            "item": ["i1", "i2", "i1", "i2"],
            "metric": [1.0, 2.0, 3.0, 4.0],
            "metric2": [2.0, 4.0, 6.0, 8.0],
        }
    )


def _calibration_data():
    return pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.8, 0.9],
            "y": [0.2, 0.3, 0.7, 0.8],
            "target_x": [0.12, 0.22, 0.82, 0.92],
            "target_y": [0.18, 0.28, 0.68, 0.78],
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )


def _close(ax):
    plt.close(ax.figure)


def test_private_coercion_numeric_empty_and_axis_paths():
    assert mq._df({"a": [1, 2]}).shape == (2, 1)

    class BadFrame:
        def __iter__(self):
            raise RuntimeError("nope")

    with pytest.raises(ep.EyeProcessValidationError, match="coercible"):
        mq._df(BadFrame())

    assert math.isnan(mq._mean([np.nan, np.inf]))
    assert math.isnan(mq._sd([1.0]))
    assert math.isnan(mq._q([np.nan], 0.5))
    assert mq._q([np.nan], 0.5, 7.0) == 7.0
    ax = plt.subplots()[1]
    assert mq._ax(ax) is ax
    _close(ax)


def test_uncertainty_spec_sequence_vector_and_estimation_guards_cluster_paths():
    spec = ep.process_uncertainty_spec(
        calibration=False,
        source_sd=[0.1, 0.2, 0.3],
        draws=4,
        seed=9,
    )
    assert spec.included["calibration"] is False
    assert spec.source_sd["calibration"] == pytest.approx(0.1)
    assert spec.source_sd["preprocessing"] == pytest.approx(0.3)

    with pytest.raises(ep.EyeProcessValidationError, match="spec"):
        ep.estimate_process_uncertainty([1.0, 2.0], spec={})
    with pytest.raises(ep.EyeProcessValidationError, match="No numeric"):
        ep.estimate_process_uncertainty(pd.DataFrame({"label": ["a", "b"]}))

    one = ep.estimate_process_uncertainty([1.0, 2.0, 3.0], spec=spec)
    assert one.metrics == ["metric"]
    d = _unc_data()
    clustered = ep.estimate_process_uncertainty(d, spec=spec, metrics=["metric", "missing"], cluster="person")
    assert clustered.metrics == ["metric"]
    assert (clustered.components.loc[clustered.components.source == "calibration", "source_sd"] == 0).all()

    singleton = ep.estimate_process_uncertainty(pd.DataFrame({"x": [1.0]}), metrics=["x"])
    assert len(singleton.summary) == 1


def test_uncertainty_propagation_invalid_posterior_empty_simulation_and_failed_estimand():
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown propagation"):
        ep.propagate_process_uncertainty([1, 2], method="bad")
    post = ep.propagate_process_uncertainty({"draws": [1.0, 2.0, np.nan]}, method="posterior", draws=5, seed=1)
    assert len(post.draws) == 5
    with pytest.raises(ep.EyeProcessValidationError, match="numeric draws"):
        ep.propagate_process_uncertainty({"draws": [np.nan]}, method="posterior")
    with pytest.raises(ep.EyeProcessValidationError, match="No observations"):
        ep.propagate_process_uncertainty([], method="bootstrap", draws=2)

    unc = ep.estimate_process_uncertainty(_unc_data(), metrics=["metric"])
    sim = ep.propagate_process_uncertainty(unc, method="simulation", draws=3, seed=3)
    assert sim.summary.loc[0, "draws"] == 3
    bad = ep.propagate_process_uncertainty(
        _unc_data()[["metric"]],
        estimand=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        draws=2,
    )
    assert bad.draws.size == 0
    assert math.isnan(bad.summary.loc[0, "mean"])


def test_uncertainty_budget_comparison_guards_positional_kwargs_and_plot_aliases():
    unc1 = ep.estimate_process_uncertainty(_unc_data(), metrics=["metric"])
    unc2 = ep.estimate_process_uncertainty(_unc_data(), metrics=["metric2"])
    with pytest.raises(ep.EyeProcessValidationError, match="eye_process_uncertainty"):
        ep.uncertainty_budget(object())
    with pytest.raises(ep.EyeProcessValidationError, match="at least one"):
        ep.compare_uncertainty_budgets()
    with pytest.raises(ep.EyeProcessValidationError, match="All objects"):
        ep.compare_uncertainty_budgets(unc1, object())

    positional = ep.compare_uncertainty_budgets(unc1, unc2)
    named = ep.compare_uncertainty_budgets(first=unc1, second=unc2)
    assert set(positional.combined.budget) == {"budget_1", "budget_2"}
    assert set(named.combined.budget) == {"first", "second"}

    for kind in ["waterfall", "tornado"]:
        ax = ep.plot_eye_process_uncertainty(unc1, type=kind, metric="metric")
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    prop = ep.propagate_process_uncertainty(unc1, draws=4, seed=2)
    for kind in ["distribution", "sensitivity"]:
        ax = ep.plot_eye_process_uncertainty_propagation(prop, type=kind)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
    ax = ep.plot_eye_uncertainty_budget_comparison(named)
    _close(ax)


def test_reference_table_roll_groups_and_drift_alternate_paths():
    d = _calibration_data()
    refs = pd.DataFrame({"x": d.target_x, "y": d.target_y})
    table = mq._reference_table(d, refs, "x", "y")
    assert len(table) == len(d)
    with pytest.raises(ep.EyeProcessValidationError, match="one row"):
        mq._reference_table(d, refs.iloc[:2], "x", "y")
    with pytest.raises(ep.EyeProcessValidationError, match="Missing reference x"):
        mq._reference_table(d.drop(columns="target_x"), None, "x", "y")

    np.testing.assert_array_equal(mq._roll_groups([0, 5, 10], "5 sec"), [1, 2, 3])
    assert np.all(mq._roll_groups([0, 1], "nonsense") == 1)
    assert np.all(mq._roll_groups([0, 1], -2) == 1)
    all_nan_groups = mq._roll_groups([np.nan, np.nan], 10)
    assert all_nan_groups.shape == (2,)

    no_time = d.drop(columns="time")
    drift = ep.detect_calibration_drift(no_time, references=refs, window="nonsense")
    assert drift.time_col is None
    assert len(drift.summary) >= 1


def test_recalibration_guards_all_methods_application_audit_threshold_and_plots():
    d = _calibration_data()
    drift = ep.detect_calibration_drift(d, window=1)
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown recalibration"):
        ep.fit_offline_recalibration(drift, method="bad")
    with pytest.raises(ep.EyeProcessValidationError, match="three complete"):
        ep.fit_offline_recalibration(d.iloc[:2])

    models = [
        ep.fit_offline_recalibration(drift, method="translation"),
        ep.fit_offline_recalibration(drift, method="affine"),
        ep.fit_offline_recalibration(drift, method="polynomial"),
    ]
    corrected = []
    for model in models:
        out = ep.apply_offline_recalibration(d[["x", "y"]], model)
        assert "x_recalibrated" in out and "y_recalibrated" in out
        corrected.append(out)

    audit_default = ep.audit_recalibration(drift, corrected[0])
    audit_strict = ep.audit_recalibration(drift, corrected[0], minimum_improvement=2.0)
    assert bool(audit_default.summary.passed.iloc[0]) is True
    assert bool(audit_strict.summary.passed.iloc[0]) is False

    for kind in ["vector_field", "error_ellipses", "drift_over_time", "screen_coverage"]:
        ax = ep.plot_eye_calibration_drift(drift, type=kind)
        _close(ax)
    ax = ep.plot_eye_recalibration_audit(audit_default)
    _close(ax)


def test_gstudy_missing_columns_no_facets_singletons_and_dstudy_paths():
    d = _unc_data()
    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        ep.fit_process_gstudy(d, "missing", facets=["person"])
    with pytest.raises(ep.EyeProcessValidationError, match="At least one facet"):
        ep.fit_process_gstudy(d, "metric", facets=["absent"])

    single = ep.fit_process_gstudy(pd.DataFrame({"metric": [1.0], "person": ["p1"]}), "metric", facets=["person"])
    assert single.variance_components.variance.ge(0).all()
    gs = ep.fit_process_gstudy(d, "metric", facets=["person", "item"])
    ds = ep.design_process_dstudy(gs, persons=4, items=[0, 2], sessions=[0, 2], devices=[0, 1])
    assert len(ds.design_grid) == 8
    assert ds.persons == 4
    assert math.isnan(mq._icc([1, 2], ["a", "a"]))
    assert np.isfinite(mq._icc([1, 2, 3, 4], ["a", "a", "b", "b"]))


def test_reliability_icc_gtheory_split_half_bootstrap_and_labels():
    d = pd.DataFrame(
        {
            "person_id": np.repeat(["p1", "p2", "p3", "p4"], 4),
            "item_id": np.tile(["i1", "i2", "i3", "i4"], 4),
            "metric": [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7],
        }
    )
    for method in ["icc", "gtheory", "split_half", "bootstrap"]:
        fit = ep.audit_process_reliability(d, "metric", method=method, draws=3)
        assert fit.summary.loc[0, "method"] == method
        assert fit.summary.loc[0, "interpretation"] in {"limited", "moderate", "good", "excellent"}

    sparse = pd.DataFrame({"person_id": ["p1", "p2"], "item_id": ["i1", "i1"], "metric": [1.0, 2.0]})
    split = ep.audit_process_reliability(sparse, "metric", method="split_half")
    assert math.isnan(split.summary.loc[0, "estimate"])

    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        ep.audit_process_reliability(d.drop(columns="person_id"), "metric")


def test_reliability_plot_dispatch_for_gstudy_dstudy_and_audit_aliases():
    d = _unc_data().rename(columns={"person": "person_id", "item": "item_id"})
    gs = ep.fit_process_gstudy(d, "metric", facets=["person_id", "item_id"])
    ds = ep.design_process_dstudy(gs, items=[1, 2], sessions=[1], devices=[1])
    audit = ep.audit_process_reliability(d, "metric", method="icc")
    for obj, fun in [
        (gs, ep.plot_eye_process_gstudy),
        (ds, ep.plot_eye_process_dstudy),
        (audit, ep.plot_eye_process_reliability_audit),
    ]:
        ax = fun(obj)
        assert hasattr(ax, "eyeprocess_plot_data")
        _close(ax)
