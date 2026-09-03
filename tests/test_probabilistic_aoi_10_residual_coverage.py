from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.probabilistic_aoi_10 as pa


def _gaze() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": ["P1", "P1", "P2", "P2"],
            "x": [0.10, 0.35, 0.65, 0.90],
            "y": [0.50, 0.55, 0.45, 0.50],
            "time": [0.0, 0.1, 0.2, 0.3],
            "duration": [10.0, 20.0, 30.0, 40.0],
        }
    )


def _aois() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aoi_id": ["A", "B"],
            "xmin": [0.0, 0.5],
            "xmax": [0.5, 1.0],
            "ymin": [0.0, 0.0],
            "ymax": [1.0, 1.0],
        }
    )


def _close(axis) -> None:
    import matplotlib.pyplot as plt

    if axis is not None and hasattr(axis, "figure"):
        plt.close(axis.figure)


def test_private_validation_numeric_scale_and_softmax_residuals():
    with pytest.raises(ep.EyeProcessValidationError, match="data frame"):
        pa._assert_data_frame([1, 2, 3], name="bad")

    with pytest.raises(ep.EyeProcessValidationError, match="Could not identify coordinate"):
        pa._first_column(pd.DataFrame({"z": [1]}), ("x", "y"), label="coordinate")

    assert np.isnan(pa._safe_quantile([np.nan], 0.5))
    assert pa._entropy([0.0, np.nan, -1.0]) == pytest.approx(0.0)

    with pytest.raises(ep.EyeProcessValidationError, match="two-dimensional"):
        pa._softmax(np.asarray([1.0, 2.0]))

    assert pa._softmax(np.empty((0, 2))).shape == (0, 2)
    assert pa._softmax(np.empty((2, 0))).shape == (2, 0)

    soft = pa._softmax(
        np.asarray(
            [
                [np.nan, np.nan],
                [1000.0, -1000.0],
            ]
        )
    )
    assert np.isnan(soft[0]).all()
    assert np.isfinite(soft[1]).all()
    assert soft[1].sum() == pytest.approx(1.0)

    np.testing.assert_allclose(pa._resolve_two_scale(None, 0.2), [0.2, 0.2])
    np.testing.assert_allclose(pa._resolve_two_scale([np.nan, -1.0, 0.0], 0.2), [0.2, 0.2])
    np.testing.assert_allclose(pa._resolve_two_scale([0.3], 0.2), [0.3, 0.3])
    np.testing.assert_allclose(pa._resolve_two_scale([0.3, 0.4, 0.5], 0.2), [0.3, 0.4])

    np.testing.assert_allclose(pa._resolve_bias(None), [0.0, 0.0])
    np.testing.assert_allclose(pa._resolve_bias([np.nan]), [0.0, 0.0])
    np.testing.assert_allclose(pa._resolve_bias([0.1]), [0.1, 0.1])
    np.testing.assert_allclose(pa._resolve_bias([0.1, -0.2, 9.0]), [0.1, -0.2])


def test_assignment_validation_tuple_ellipse_and_degenerate_geometry_paths():
    gaze = _gaze()
    aois = _aois()

    with pytest.raises(ep.EyeProcessValidationError, match="error_model"):
        ep.assign_aois_probabilistic(gaze, aois, error_model="unsupported")

    with pytest.raises(ep.EyeProcessValidationError, match="gaze x coordinate"):
        ep.assign_aois_probabilistic(pd.DataFrame({"z": [1.0]}), aois)

    with pytest.raises(ep.EyeProcessValidationError, match="missing required"):
        ep.assign_aois_probabilistic(gaze, aois, x_col="missing", y_col="y")

    bad_aois = pd.DataFrame(
        {
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )
    with pytest.raises(ep.EyeProcessValidationError, match="AOI identifier"):
        ep.assign_aois_probabilistic(gaze, bad_aois)

    empty_tuple = ep.assign_aois_probabilistic(gaze, aois, error_model=(), precision=0.05)
    assert empty_tuple.error_model == "empirical"

    ellipse = ep.assign_aois_probabilistic(
        gaze,
        aois,
        error_model="ellipse",
        precision=[0.04, 0.08, 9.0],
        accuracy=[0.01, -0.02, 9.0],
        id_cols=["person_id", "person_id", "missing"],
    )
    assert ellipse.error_model == "ellipse"
    np.testing.assert_allclose(ellipse.spread, [0.04, 0.08])
    np.testing.assert_allclose(ellipse.accuracy, [0.01, -0.02])
    assert list(ellipse.wide.columns).count("person_id") == 1
    assert list(ellipse.membership.columns).count("person_id") == 2

    degenerate_aois = pd.DataFrame(
        {
            "aoi_id": ["A"],
            "xmin": [np.nan],
            "xmax": [np.nan],
            "ymin": [np.nan],
            "ymax": [np.nan],
        }
    )
    degenerate = ep.assign_aois_probabilistic(gaze.iloc[:2], degenerate_aois)
    assert degenerate.eyeprocess_class == "eye_probabilistic_aoi"
    assert list(degenerate.probabilities.columns) == ["A", "outside"]


def test_aoi_audit_empty_pairwise_and_uncertain_metric_fallback_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="Supply `aois`"):
        ep.audit_aoi_separation()

    single_aoi = pd.DataFrame(
        {
            "aoi_id": ["A"],
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )
    audit = ep.audit_aoi_separation(aois=single_aoi)
    assert audit.pairwise.empty
    assert audit.probability_summary is None
    assert audit.summary.loc[0, "pairs"] == 0
    assert np.isnan(audit.summary.loc[0, "minimum_gap"])

    fallback = pa._uncertain_aoi_metrics(
        np.asarray(["A"], dtype=object),
        pd.DataFrame({"other": [1]}),
        "missing_time",
        "missing_duration",
        ["A", "outside"],
    )
    assert fallback["dwell__A"] == pytest.approx(1.0)
    assert fallback["ttff__A"] == pytest.approx(0.0)
    assert fallback["transitions"] == pytest.approx(0.0)

    nonfinite = pa._uncertain_aoi_metrics(
        np.asarray(["outside", "outside"], dtype=object),
        pd.DataFrame({"time": [np.nan, np.nan], "duration": [np.nan, np.nan]}),
        "time",
        "duration",
        ["A", "outside"],
    )
    assert nonfinite["dwell__A"] == pytest.approx(0.0)
    assert np.isnan(nonfinite["ttff__A"])


def test_propagation_invalid_draw_conversion_string_metrics_and_deduplication():
    fit = ep.assign_aois_probabilistic(_gaze(), _aois(), precision=0.05)

    with pytest.raises(ep.EyeProcessValidationError, match="draws"):
        ep.propagate_aoi_uncertainty(fit, draws=np.inf)

    entropy_only = ep.propagate_aoi_uncertainty(
        fit,
        metrics="entropy",
        draws=2,
        seed=7,
    )
    assert entropy_only.metrics == ["entropy"]
    assert list(entropy_only.draws.columns) == ["draw", "entropy"]

    deduplicated = ep.propagate_aoi_uncertainty(
        fit,
        metrics=["entropy", "entropy", "unsupported"],
        draws=2,
        seed=7,
    )
    assert deduplicated.metrics == ["entropy"]


def test_plot_backend_custom_axis_empty_and_invalid_class_residuals(monkeypatch):
    original_import = builtins.__import__

    def deny_matplotlib(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("forced matplotlib absence")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", deny_matplotlib)
        with pytest.raises(ep.EyeProcessBackendError, match="requires matplotlib"):
            pa._get_plt()

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        assert pa._axis(axis) is axis
        empty = pa._empty_plot("Nothing here", "Empty", ax=axis)
        assert empty.eyeprocess_plot_data.empty
    finally:
        plt.close(figure)

    single_aoi = pd.DataFrame(
        {
            "aoi_id": ["A"],
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )
    audit = ep.audit_aoi_separation(aois=single_aoi)
    figure, axis = plt.subplots()
    try:
        out = ep.plot_aoi_boundary_risk(audit, ax=axis)
        assert out.eyeprocess_plot_data.empty
    finally:
        plt.close(figure)

    figure, axis = plt.subplots()
    try:
        with pytest.raises(ep.EyeProcessValidationError, match="must be"):
            ep.plot_aoi_boundary_risk(object(), ax=axis)
    finally:
        plt.close(figure)

    empty_uncertainty = pa._result(
        "eye_aoi_uncertainty",
        draws=pd.DataFrame({"draw": [1, 2]}),
    )
    figure, axis = plt.subplots()
    try:
        out = ep.plot_aoi_metric_uncertainty(empty_uncertainty, ax=axis)
        assert out.eyeprocess_plot_data.empty
    finally:
        plt.close(figure)

    one_step = pa._result(
        "eye_aoi_uncertainty",
        source={
            "probabilities": pd.DataFrame(
                [[1.0, 0.0]],
                columns=["A", "outside"],
            )
        },
    )
    figure, axis = plt.subplots()
    try:
        out = ep.plot_fuzzy_transition_matrix(one_step, ax=axis)
        np.testing.assert_allclose(out.eyeprocess_plot_matrix, np.zeros((2, 2)))
    finally:
        plt.close(figure)
