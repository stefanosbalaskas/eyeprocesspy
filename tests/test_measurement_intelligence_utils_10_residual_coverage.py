from __future__ import annotations

import builtins
import warnings

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.measurement_intelligence_utils_10 as mi


def _close(axis) -> None:
    import matplotlib.pyplot as plt

    if axis is not None and hasattr(axis, "figure"):
        plt.close(axis.figure)


def test_plot_spec_attribute_repr_and_helper_guards(monkeypatch):
    spec = mi.eye_plot_spec()
    with pytest.raises(AttributeError, match="missing"):
        _ = spec.missing
    text = repr(spec)
    assert "Title:" not in text
    assert "Note:" not in text

    original_import = builtins.__import__

    def deny_matplotlib(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("forced matplotlib absence")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", deny_matplotlib)
        with pytest.raises(ep.EyeProcessBackendError, match="requires matplotlib"):
            mi._get_plt()

    marker = object()
    assert mi._new_axis(marker) is marker

    class RejectPlotData:
        def __setattr__(self, name, value):
            raise RuntimeError("read only")

    locked = RejectPlotData()
    assert mi._set_plot_data(locked, {"x": 1}) is locked


def test_call_supported_signature_variants(monkeypatch):
    def permissive(x, **kwargs):
        return x, kwargs

    def fail_signature(_function):
        raise ValueError("no signature")

    with monkeypatch.context() as patch:
        patch.setattr(mi.inspect, "signature", fail_signature)
        value, kwargs = mi._call_supported(permissive, "x", alpha=1)
    assert value == "x"
    assert kwargs == {"alpha": 1}

    value, kwargs = mi._call_supported(permissive, "x", alpha=2)
    assert value == "x"
    assert kwargs == {"alpha": 2}

    def limited(x, ax=None):
        return x, ax

    sentinel = object()
    value, axis = mi._call_supported(
        limited,
        "x",
        ax=sentinel,
        unsupported=9,
    )
    assert value == "x"
    assert axis is sentinel


def test_class_specialized_and_candidate_frame_residuals(monkeypatch):
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    frame.attrs["eyeprocess_class"] = "eye_attr_result"
    assert mi._class_name(frame) == "eye_attr_result"

    class Dummy:
        eyeprocess_class = "eye_dummy"

    def plot_dummy(x, ax=None):
        return x, ax

    with monkeypatch.context() as patch:
        patch.setattr(ep, "plot_eye_dummy", plot_dummy, raising=False)
        assert mi._specialized_plotter(Dummy()) is plot_dummy

    assert mi._specialized_plotter(object()) is None

    direct = mi._candidate_frame(frame, "diagnostics")
    pd.testing.assert_frame_equal(direct, frame)
    assert direct is not frame
    assert mi._candidate_frame(object(), "default") is None

    series = pd.Series([1.0, 2.0], name=None)
    candidate = mi._candidate_frame({"summary": series}, "unknown_type")
    assert list(candidate.columns) == ["value"]
    assert mi._candidate_frame({"unrelated": 1}, "default") is None


def test_generic_frame_plot_empty_single_and_diagnostics_paths():
    string_axis = mi._generic_frame_plot(
        pd.DataFrame({"label": ["a", "b"]}),
        plot_type="evidence",
    )
    assert string_axis.eyeprocess_plot_data.shape == (2, 1)
    assert not string_axis.get_xticks().size
    _close(string_axis)

    one_axis = mi._generic_frame_plot(
        pd.DataFrame({"estimate": [1.0, 2.0, 3.0]}),
        plot_type="default",
    )
    assert one_axis.get_ylabel() == "estimate"
    _close(one_axis)

    diag_axis = mi._generic_frame_plot(
        pd.DataFrame({"observed": [1.0, 2.0], "residual": [-0.1, 0.2]}),
        plot_type="diagnostics",
    )
    assert diag_axis.get_xlabel() == "observed"
    assert diag_axis.get_ylabel() == "residual"
    _close(diag_axis)


def test_generic_mapping_plot_frame_numeric_and_empty_paths():
    series_axis = mi._generic_mapping_plot(
        {"summary": pd.Series([1.0, 2.0], name="estimate")},
        plot_type="sensitivity",
    )
    assert list(series_axis.eyeprocess_plot_data.columns) == ["estimate"]
    _close(series_axis)

    numeric_axis = mi._generic_mapping_plot(
        {
            "alpha": 1.0,
            "beta": 2,
            "text": "skip",
            "flag": True,
            "missing": np.nan,
        },
        plot_type="evidence",
    )
    assert numeric_axis.eyeprocess_plot_data["alpha"] == 1.0
    assert len(numeric_axis.patches) == 2
    _close(numeric_axis)

    empty_axis = mi._generic_mapping_plot(
        {"text": "skip", "flag": True, "missing": np.nan},
        plot_type="default",
    )
    assert empty_axis.eyeprocess_plot_data["text"] == "skip"
    assert not empty_axis.get_xticks().size
    _close(empty_axis)


def test_dispatch_object_plot_success_typeerror_fallback_and_validation():
    class PlotWithKwargs:
        def plot(self, **kwargs):
            return kwargs

    assert mi._dispatch_plot(PlotWithKwargs(), alpha=3) == {"alpha": 3}

    class PlotWithoutKwargs:
        def plot(self, **kwargs):
            if kwargs:
                raise TypeError("no kwargs")
            return "fallback"

    assert mi._dispatch_plot(PlotWithoutKwargs(), alpha=3) == "fallback"

    with pytest.raises(ep.EyeProcessValidationError, match="No eyeprocess plotting method"):
        mi._dispatch_plot(object())


def test_plot_with_fallback_success_and_terminal_error(monkeypatch):
    calls = []

    def recover_dispatch(x, *, plot_type="default", ax=None, **kwargs):
        calls.append(plot_type)
        if plot_type != "default":
            raise ep.EyeProcessValidationError("forced specialized failure")
        return "default-result"

    with monkeypatch.context() as patch:
        patch.setattr(mi, "_dispatch_plot", recover_dispatch)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mi._plot_with_fallback({}, "diagnostics")
    assert result == "default-result"
    assert calls == ["diagnostics", "default"]
    assert any("using the default plot" in str(item.message) for item in caught)

    def fail_dispatch(x, *, plot_type="default", ax=None, **kwargs):
        raise ep.EyeProcessValidationError(f"forced {plot_type} failure")

    with monkeypatch.context() as patch:
        patch.setattr(mi, "_dispatch_plot", fail_dispatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(
                ep.EyeProcessValidationError,
                match="No default eyeprocess plot",
            ):
                mi._plot_with_fallback({}, "evidence")
