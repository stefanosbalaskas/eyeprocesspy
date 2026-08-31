from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
from eyeprocesspy.irt import EyeResult

TARGETS = [
    "eye_plot_spec",
    "plot_diagnostics",
    "plot_evidence",
    "plot_sensitivity",
    "autoplot_eyeprocess",
]


def _close(axis):
    import matplotlib.pyplot as plt

    if axis is not None and hasattr(axis, "figure"):
        plt.close(axis.figure)


def test_public_r030_exports_are_callable():
    assert len(TARGETS) == 5
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_eye_plot_spec_matches_frozen_contract():
    spec = ep.eye_plot_spec(
        type="diagnostics",
        title="Calibration audit",
        xlab="Observed",
        ylab="Expected",
        caption="Review uncertainty.",
        show_uncertainty=False,
        show_raw=True,
        facet_by="device",
        label_items=True,
        interactive=True,
    )

    assert spec.eyeprocess_class == "eye_plot_spec"
    assert spec.type == "diagnostics"
    assert spec.title == "Calibration audit"
    assert spec.xlab == "Observed"
    assert spec.ylab == "Expected"
    assert spec.caption == "Review uncertainty."
    assert spec.show_uncertainty is False
    assert spec.show_raw is True
    assert spec.facet_by == "device"
    assert spec.label_items is True
    assert spec.interactive is True
    assert "eyeprocess plot specification" in repr(spec)
    assert "Type: diagnostics" in repr(spec)


def test_autoplot_dispatches_existing_class_named_plotter():
    graph = ep.build_evidence_graph(
        raw_data=["raw"],
        transformations=["clean"],
        metrics=["dwell"],
        models=["irt"],
        diagnostics=["ppc"],
        decisions=["retain"],
    )

    axis = ep.autoplot_eyeprocess(graph)
    assert hasattr(axis, "eyeprocess_plot_data")
    assert len(axis.eyeprocess_plot_data) == len(graph["nodes"])
    _close(axis)


def test_evidence_dispatch_uses_existing_evidence_graph_plot():
    graph = ep.build_evidence_graph(
        raw_data=["raw"],
        transformations=["clean"],
        metrics=["dwell"],
        decisions=["retain"],
    )

    axis = ep.plot_evidence(graph)
    assert hasattr(axis, "eyeprocess_plot_data")
    assert set(axis.eyeprocess_plot_data.columns) >= {
        "node_id",
        "label",
        "stage",
    }
    _close(axis)


def test_diagnostics_dispatches_eyeprocess_model():
    dataset = ep.simulate_eye_dataset(
        n_person=3,
        n_item=3,
        samples_per_trial=12,
        seed=33,
    )
    model = ep.fit_irt(dataset, engine="rasch_glm")

    axis = ep.plot_diagnostics(model)
    assert hasattr(axis, "eyeprocess_plot_data")
    _close(axis)


def test_sensitivity_generic_summary_fallback():
    result = EyeResult(
        {
            "summary": pd.DataFrame(
                {
                    "scenario": ["base", "alt"],
                    "estimate": [0.10, 0.16],
                    "delta": [0.00, 0.06],
                }
            )
        },
        eyeprocess_class="eye_unregistered_sensitivity_result",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        axis = ep.plot_sensitivity(result)

    assert hasattr(axis, "eyeprocess_plot_data")
    assert len(axis.eyeprocess_plot_data) == 2
    assert not caught
    _close(axis)


def test_generic_dataframe_autoplot_is_transparent():
    data = pd.DataFrame(
        {
            "x": np.arange(4, dtype=float),
            "y": [1.0, 2.0, 1.5, 3.0],
        }
    )
    axis = ep.autoplot_eyeprocess(data)
    pd.testing.assert_frame_equal(axis.eyeprocess_plot_data, data)
    _close(axis)


def test_unplottable_object_raises_explicit_validation_error():
    with pytest.raises(ep.EyeProcessValidationError):
        ep.autoplot_eyeprocess(object())
