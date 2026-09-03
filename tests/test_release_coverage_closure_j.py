from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.dynamic_irt as dy
import eyeprocesspy.plots_governance_09 as pg
import eyeprocesspy.plots_process_quality_09 as pq
import eyeprocesspy.timebase as tb


class R(dict):
    def __init__(self, cls, **kwargs):
        super().__init__(**kwargs)
        self.eyeprocess_class = cls

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _close(ax):
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_process_quality_plot_validation_summary_and_empty_probability():
    bad = {}
    for fn in [
        pq.plot_eye_process_reliability_profile,
        pq.plot_eye_calibration_error_model,
        pq.plot_eye_calibration_drift_profile,
        pq.plot_eye_data_quality_profile,
        pq.plot_eye_probabilistic_aoi_assignment,
        pq.plot_eye_sampling_irregularity_audit,
    ]:
        with pytest.raises(ep.EyeProcessValidationError):
            fn(bad)

    rel = R(
        "eye_process_reliability_profile",
        icc=pd.DataFrame({"icc_a1": [0.8]}),
        bland_altman={"summary": pd.DataFrame({"bias": [0.1]})},
    )
    ax = pq.plot_eye_process_reliability_profile(rel, type="summary")
    assert len(ax.eyeprocess_plot_data) == 2
    _close(ax)
    with pytest.raises(ep.EyeProcessValidationError, match="type"):
        pq.plot_eye_process_reliability_profile(rel, type="bad")

    quality = R("eye_data_quality_profile", table=pd.DataFrame({"x": [1]}))
    ax = pq.plot_eye_data_quality_profile(quality, metric="missing")
    assert len(ax.eyeprocess_plot_data) == 1
    _close(ax)

    prob = R("eye_probabilistic_aoi_assignment", probabilities=pd.DataFrame())
    ax = pq.plot_eye_probabilistic_aoi_assignment(prob)
    assert ax.eyeprocess_plot_matrix.shape == (0, 0)
    _close(ax)


def test_timebase_stream_default_mapping_and_errors():
    x = ep.new_eye_dataset(validate=False)
    x["streams"] = ep.standardize_eye_table(
        pd.DataFrame({"stream_id": ["s"], "timestamp_unit": [pd.NA]}), "streams"
    )
    x["gaze_samples"] = ep.standardize_eye_table(
        pd.DataFrame(
            {
                "recording_id": ["R1", "R1"],
                "sample_id": ["s1", "s2"],
                "timestamp_native": [1000.0, 2000.0],
                "timestamp_seconds": [np.nan, np.nan],
            }
        ),
        "gaze_samples",
    )
    out = tb.normalize_timebase(x, component="gaze_samples", native_unit=None)
    assert out["gaze_samples"].timestamp_seconds.tolist() == [0.0, 1000.0]

    mapped = tb.apply_clock_transform(x, {"offset": 1.0, "slope": 2.0}, components="gaze_samples")
    assert ep.is_eye_dataset(mapped)
    with pytest.raises(ep.EyeProcessTimebaseError, match="transform"):
        tb.apply_clock_transform(x, object())

    x["gaze_samples"] = pd.DataFrame({"x": [1]})
    with pytest.raises(ep.EyeProcessTimebaseError, match="lacks required"):
        tb.audit_timebase(x)
    with pytest.raises(ValueError, match="Invalid method"):
        tb.estimate_clock_transform([1], [1], method="bad")
    with pytest.raises(ep.EyeProcessTimebaseError, match="No valid"):
        tb.estimate_clock_transform([np.nan], [np.nan])


def test_dynamic_import_prepare_decode_and_control(monkeypatch):
    fake = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)
    assert dy._cmdstanpy() is fake

    spec = dy.dynamic_irtree_spec(missing_state="drop")
    d = pd.DataFrame({"f": [pd.NA], "t": [pd.NA]})
    with pytest.raises(ep.EyeProcessValidationError, match="No transitions remain"):
        dy.prepare_dynamic_irtree_data(d, spec=spec, **{"from": "f", "to": "t"})

    model = dy._result(
        "eye_multinomial_transition",
        probabilities=pd.DataFrame([[0.2, 0.8]], columns=["A", "B"]),
    )
    wrapper = dy._result("eye_dynamic_irtree", model=model)
    mode = dy.decode_dynamic_states(wrapper, method="mode")
    assert mode.decoded_state.iloc[0] == "B"
    monkeypatch.setattr(np.random, "default_rng", lambda: SimpleNamespace(choice=lambda n, p: 0))
    draw = dy.decode_dynamic_states(wrapper, method="draw")
    assert draw.decoded_state.iloc[0] == "A"

    with pytest.raises(ep.EyeProcessValidationError, match="All inputs"):
        dy.compare_dynamic_transition_models(object())


def test_governance_plot_backend_failure_and_monkeypatched_edges(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="matplotlib"):
        pg._plt()
    monkeypatch.setattr(builtins, "__import__", real_import)

    result = SimpleNamespace(estimates=pd.DataFrame(), failures=pd.DataFrame())
    monkeypatch.setattr(
        pg,
        "validation_failure_profile",
        lambda x: pd.DataFrame({"stage": ["fit"], "failure_rate": [0.25]}),
    )
    ax = pg.plot_eye_process_validation_result(result, type="failure")
    assert len(ax.eyeprocess_plot_data) == 1
    _close(ax)

    graph = SimpleNamespace(
        vertices=pd.DataFrame({"step": ["a", "b"], "order": [1, 2]}),
        edges=pd.DataFrame({"from": ["a"], "to": ["b"]}),
    )
    monkeypatch.setattr(pg, "eye_pipeline_graph", lambda x: graph)
    ax = pg.plot_eye_analysis_pipeline(object())
    assert ax.eyeprocess_plot_data is graph
    _close(ax)
