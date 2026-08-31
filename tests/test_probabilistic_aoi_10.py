from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.probabilistic_aoi_10 import _softmax

TARGETS = [
    "assign_aois_probabilistic",
    "audit_aoi_separation",
    "summarise_aoi_membership",
    "propagate_aoi_uncertainty",
    "plot_aoi_probability_map",
    "plot_aoi_boundary_risk",
    "plot_probabilistic_scanpath",
    "plot_fuzzy_transition_matrix",
    "plot_aoi_metric_uncertainty",
]


def _gaze(n=40):
    time = np.arange(n, dtype=float) * 0.01
    x = np.linspace(0.08, 0.92, n)
    y = 0.5 + 0.12 * np.sin(np.linspace(0, 3 * np.pi, n))
    return pd.DataFrame(
        {
            "person_id": np.where(np.arange(n) < n / 2, "P1", "P2"),
            "time": time,
            "duration": np.repeat(10.0, n),
            "x": x,
            "y": y,
        }
    )


def _aois():
    return pd.DataFrame(
        {
            "aoi_id": ["A", "B", "C"],
            "xmin": [0.00, 0.30, 0.70],
            "xmax": [0.40, 0.70, 1.00],
            "ymin": [0.20, 0.20, 0.20],
            "ymax": [0.80, 0.80, 0.80],
        }
    )


def _close(axis):
    import matplotlib.pyplot as plt

    if axis is not None and hasattr(axis, "figure"):
        plt.close(axis.figure)


def test_public_r031_exports_are_callable():
    assert len(TARGETS) == 9
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_internal_softmax_preserves_shape_and_row_closure():
    logits = np.asarray(
        [
            [0.0, 1.0, -1.0],
            [2.0, 0.0, 3.0],
        ]
    )
    softmax = _softmax(logits)
    assert softmax.shape == logits.shape
    np.testing.assert_allclose(
        softmax.sum(axis=1),
        np.ones(logits.shape[0]),
        atol=1e-12,
        rtol=0,
    )


def test_probabilistic_assignment_closes_probabilities_like_frozen_r():
    gaze = _gaze(40)
    fit = ep.assign_aois_probabilistic(
        gaze,
        _aois(),
        precision=0.05,
        id_cols=["person_id"],
    )

    assert fit.eyeprocess_class == "eye_probabilistic_aoi"
    np.testing.assert_allclose(
        fit["probabilities"].sum(axis=1).to_numpy(dtype=float),
        np.ones(len(gaze)),
        atol=1e-8,
        rtol=0,
    )
    assert (fit["classification"]["maximum_probability"].to_numpy(dtype=float) >= 0).all()
    assert list(fit["probabilities"].columns) == [
        "A",
        "B",
        "C",
        "outside",
    ]
    assert len(fit["membership"]) == len(gaze) * 4
    assert "person_id" in fit["membership"].columns


def test_nonfinite_gaze_is_outside_with_probability_one():
    gaze = _gaze(6)
    gaze.loc[2, "x"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = ep.assign_aois_probabilistic(
            gaze,
            _aois(),
            precision=0.05,
        )

    assert any("outside-AOI probability one" in str(item.message) for item in caught)
    row = fit["probabilities"].iloc[2]
    assert row["outside"] == pytest.approx(1.0)
    assert row[["A", "B", "C"]].sum() == pytest.approx(0.0)


def test_aoi_separation_audit_covers_overlap_touching_and_separated():
    fit = ep.assign_aois_probabilistic(
        _gaze(20),
        _aois(),
        precision=0.05,
    )
    audit = ep.audit_aoi_separation(fit)

    assert audit.eyeprocess_class == "eye_aoi_separation_audit"
    statuses = set(audit["pairwise"]["separation_status"])
    assert statuses == {"overlap", "touching", "separated"}
    assert len(audit["probability_summary"]) == 1
    assert audit["summary"]["pairs"].iloc[0] == 3


def test_membership_summary_supports_ungrouped_and_grouped_forms():
    fit = ep.assign_aois_probabilistic(
        _gaze(20),
        _aois(),
        precision=0.05,
        id_cols=["person_id"],
    )
    overall = ep.summarise_aoi_membership(fit)
    grouped = ep.summarise_aoi_membership(fit, by=["person_id", "missing"])

    assert list(overall.columns) == ["aoi", "mean_probability"]
    assert list(grouped.columns) == [
        "person_id",
        "aoi",
        "mean_probability",
    ]
    assert set(overall["aoi"]) == {"A", "B", "C", "outside"}


def test_uncertainty_propagation_matches_frozen_structure_and_is_seeded():
    fit = ep.assign_aois_probabilistic(
        _gaze(40),
        _aois(),
        precision=0.05,
    )
    first = ep.propagate_aoi_uncertainty(
        fit,
        draws=20,
        time_col="time",
        duration_col="duration",
        seed=20260807,
    )
    second = ep.propagate_aoi_uncertainty(
        fit,
        draws=20,
        time_col="time",
        duration_col="duration",
        seed=20260807,
    )

    assert first.eyeprocess_class == "eye_aoi_uncertainty"
    assert len(first["draws"]) == 20
    assert set(first["summary"].columns) == {
        "metric",
        "mean",
        "sd",
        "lower",
        "median",
        "upper",
    }
    pd.testing.assert_frame_equal(first["draws"], second["draws"])
    pd.testing.assert_frame_equal(first["summary"], second["summary"])


def test_probabilistic_aoi_plot_surface_executes():
    fit = ep.assign_aois_probabilistic(
        _gaze(30),
        _aois(),
        precision=0.05,
    )
    audit = ep.audit_aoi_separation(fit)
    propagated = ep.propagate_aoi_uncertainty(
        fit,
        draws=20,
        time_col="time",
        duration_col="duration",
    )

    axes = [
        ep.plot_aoi_probability_map(fit),
        ep.plot_aoi_boundary_risk(fit),
        ep.plot_aoi_boundary_risk(audit),
        ep.plot_probabilistic_scanpath(fit),
        ep.plot_fuzzy_transition_matrix(propagated),
        ep.plot_aoi_metric_uncertainty(propagated),
    ]
    for axis in axes:
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_fuzzy_transition_matrix_matches_outer_product_definition():
    fit = ep.assign_aois_probabilistic(
        _gaze(8),
        _aois(),
        precision=0.05,
    )
    propagated = ep.propagate_aoi_uncertainty(fit, draws=2)
    axis = ep.plot_fuzzy_transition_matrix(propagated)

    p = fit["probabilities"].to_numpy(dtype=float)
    expected = np.zeros((p.shape[1], p.shape[1]))
    for index in range(len(p) - 1):
        expected += np.outer(p[index], p[index + 1])

    np.testing.assert_allclose(
        axis.eyeprocess_plot_matrix,
        expected,
        atol=1e-12,
        rtol=0,
    )
    _close(axis)


def test_validation_boundaries_are_explicit():
    with pytest.raises(ep.EyeProcessValidationError):
        ep.assign_aois_probabilistic(
            pd.DataFrame({"x": [0.1], "y": [0.2]}),
            pd.DataFrame(),
        )

    fit = ep.assign_aois_probabilistic(
        _gaze(5),
        _aois(),
        precision=0.05,
    )
    with pytest.raises(ep.EyeProcessValidationError, match="at least 2"):
        ep.propagate_aoi_uncertainty(fit, draws=1)
    with pytest.raises(ep.EyeProcessValidationError, match="No supported"):
        ep.propagate_aoi_uncertainty(fit, metrics=["unsupported"])
    with pytest.raises(ep.EyeProcessValidationError):
        ep.summarise_aoi_membership({})
