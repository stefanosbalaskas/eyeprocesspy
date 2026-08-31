from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.plots_completion_08 import _transition_table

TARGETS = [
    "plot_aoi_transition_matrix",
    "plot_aoi_transition_rank",
    "plot_process_channel_ablation_delta",
    "plot_pupil_components",
    "plot_pupil_preprocessing_audit",
]


def _close(axis):
    import matplotlib.pyplot as plt

    if axis is not None and hasattr(axis, "figure"):
        plt.close(axis.figure)


def _transitions():
    return pd.DataFrame(
        {
            "from": ["A", "A", "B", "B"],
            "to": ["B", "C", "A", "C"],
        }
    )


def _pupil():
    x = np.linspace(0, 1, 20)
    return pd.DataFrame(
        {
            "time_ms": np.arange(1, 21, dtype=float),
            "pupil_raw": x,
            "pupil_interpolated": x + 0.1,
            "pupil_smoothed": x + 0.2,
            "pupil_bc": x - 0.5,
            "pupil_tonic": x + 0.15,
            "pupil_phasic": np.sin(x * np.pi),
        }
    )


def test_public_r064_completion_exports_are_callable():
    assert len(TARGETS) == 5
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_transition_table_normalization_contract():
    data = _transitions()

    raw = _transition_table(data, normalize="none")
    expected = pd.DataFrame(
        [[0, 1, 1], [1, 0, 1]],
        index=["A", "B"],
        columns=["A", "B", "C"],
    )
    pd.testing.assert_frame_equal(raw, expected)

    by_from = _transition_table(data, normalize="from")
    np.testing.assert_allclose(
        by_from.sum(axis=1).to_numpy(dtype=float),
        np.ones(2),
        atol=1e-12,
        rtol=0,
    )

    global_prob = _transition_table(data, normalize="all")
    assert float(global_prob.to_numpy(dtype=float).sum()) == pytest.approx(1.0)


def test_frozen_base_plot_helpers_execute_on_synthetic_inputs():
    matrix_axis = ep.plot_aoi_transition_matrix(_transitions())
    rank_axis = ep.plot_aoi_transition_rank(_transitions())

    signal = _pupil().loc[
        :,
        [
            "time_ms",
            "pupil_raw",
            "pupil_smoothed",
            "pupil_bc",
        ],
    ]
    pupil_axis = ep.plot_pupil_preprocessing_audit(signal)

    for axis in (matrix_axis, rank_axis, pupil_axis):
        assert hasattr(axis, "eyeprocess_plot_data")
        _close(axis)


def test_transition_matrix_retains_source_normalized_matrix():
    axis = ep.plot_aoi_transition_matrix(
        _transitions(),
        normalize="from",
    )
    expected = _transition_table(
        _transitions(),
        normalize="from",
    )
    pd.testing.assert_frame_equal(
        axis.eyeprocess_plot_data,
        expected,
    )
    np.testing.assert_allclose(
        axis.eyeprocess_plot_matrix,
        expected.to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    )
    _close(axis)


def test_transition_rank_matches_frozen_sort_and_top_n_contract():
    axis = ep.plot_aoi_transition_rank(
        _transitions(),
        normalize="none",
        top_n=3,
    )
    ranked = axis.eyeprocess_plot_data
    assert len(ranked) == 3
    assert ranked["value"].is_monotonic_decreasing
    assert set(ranked.columns) == {
        "from",
        "to",
        "value",
        "transition",
    }
    _close(axis)


def test_pupil_preprocessing_audit_intersects_requested_signals():
    axis = ep.plot_pupil_preprocessing_audit(
        _pupil(),
        signals=[
            "pupil_raw",
            "missing_signal",
            "pupil_bc",
        ],
    )
    assert list(axis.eyeprocess_plot_data.columns) == [
        "time_ms",
        "pupil_raw",
        "pupil_bc",
    ]
    _close(axis)


def test_pupil_components_is_true_source_alias_to_preprocessing_surface():
    axis = ep.plot_pupil_components(_pupil())
    assert list(axis.eyeprocess_plot_data.columns) == [
        "time_ms",
        "pupil_smoothed",
        "pupil_tonic",
        "pupil_phasic",
    ]
    _close(axis)


def test_channel_ablation_delta_uses_max_full_reference():
    table = pd.DataFrame(
        {
            "channel": ["full", "full", "no_gaze", "no_pupil"],
            "metric": ["auc"] * 4,
            "value": [0.80, 0.82, 0.75, 0.78],
        }
    )
    axis = ep.plot_process_channel_ablation_delta(
        table=table,
        metric="auc",
    )

    result = axis.eyeprocess_plot_data
    assert axis.eyeprocess_plot_reference == pytest.approx(0.82)
    np.testing.assert_allclose(
        result["delta_from_reference"].to_numpy(dtype=float),
        [-0.02, 0.0, -0.07, -0.04],
        atol=1e-12,
        rtol=0,
    )
    _close(axis)


def test_channel_ablation_accepts_mapping_candidates_and_fallback_reference():
    table = pd.DataFrame(
        {
            "channel": ["gaze", "pupil"],
            "value": [0.71, 0.74],
        }
    )
    axis = ep.plot_process_channel_ablation_delta(
        {"results": table},
    )
    assert axis.eyeprocess_plot_reference == pytest.approx(0.74)
    _close(axis)


def test_validation_boundaries_are_explicit():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="No complete AOI transitions",
    ):
        ep.plot_aoi_transition_matrix(
            pd.DataFrame(
                {
                    "from": [None],
                    "to": [None],
                }
            )
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="No requested pupil",
    ):
        ep.plot_pupil_preprocessing_audit(
            pd.DataFrame({"time_ms": [1, 2]}),
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="positive integer",
    ):
        ep.plot_aoi_transition_rank(
            _transitions(),
            top_n=0,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="No finite ablation",
    ):
        ep.plot_process_channel_ablation_delta(
            table=pd.DataFrame(
                {
                    "channel": ["full"],
                    "value": [np.nan],
                }
            )
        )
