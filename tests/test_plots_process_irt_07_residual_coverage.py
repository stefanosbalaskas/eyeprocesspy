from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy.plots_process_irt_07 as pp


def _close(ax):
    plt.close(ax.figure)


def test_joint_gaze_rt_plot_true_inner_false_inner_and_outer_fallbacks():
    rich = SimpleNamespace(
        person_scores=pd.DataFrame(
            {
                "theta": [-1.0, 0.0, 1.0],
                "speed": [0.5, 0.0, -0.5],
                "label": ["a", "b", "c"],
            }
        )
    )
    ax = pp.plot_eye_joint_gaze_rt_irt(rich, type="latent")
    assert len(ax.collections) == 1
    assert list(ax.gp3_data.columns) == ["theta", "speed", "label"]
    _close(ax)

    one_numeric = SimpleNamespace(
        person_scores=pd.DataFrame({"theta": [0.0, 1.0], "label": ["a", "b"]})
    )
    ax = pp.plot_eye_joint_gaze_rt_irt(one_numeric, type="latent")
    assert ax.gp3_data.empty
    _close(ax)

    ax = pp.plot_eye_joint_gaze_rt_irt(rich, type="summary")
    assert ax.gp3_data.empty
    _close(ax)


def test_joint_graded_plot_dataframe_inner_branches_and_nonframe_fallback():
    rich = SimpleNamespace(
        person_scores=pd.DataFrame({"ability": [-1.0, 1.0], "speed": [0.2, -0.2]})
    )
    ax = pp.plot_eye_joint_graded_rt_process_irt(rich)
    assert len(ax.collections) == 1
    assert len(ax.gp3_data) == 2
    _close(ax)

    one_numeric = SimpleNamespace(
        person_scores=pd.DataFrame({"ability": [0.0, 1.0], "label": ["a", "b"]})
    )
    ax = pp.plot_eye_joint_graded_rt_process_irt(one_numeric)
    assert ax.gp3_data.empty
    _close(ax)

    ax = pp.plot_eye_joint_graded_rt_process_irt(SimpleNamespace(person_scores=None))
    assert ax.gp3_data.empty
    _close(ax)


def test_nominal_plot_nonempty_and_empty_map_branches(monkeypatch):
    nonempty = pd.DataFrame(
        {
            "response_category": ["A", "B"],
            "gaze_channel": ["gA", "gB"],
            "coefficient": [0.4, -0.2],
        }
    )
    monkeypatch.setattr(pp, "distractor_process_map", lambda x: nonempty.copy())
    ax = pp.plot_eye_nominal_gaze_irt(object())
    assert len(ax.patches) == 2
    assert len(ax.gp3_data) == 2
    _close(ax)

    monkeypatch.setattr(pp, "distractor_process_map", lambda x: nonempty.iloc[0:0].copy())
    ax = pp.plot_eye_nominal_gaze_irt(object())
    assert ax.gp3_data.empty
    _close(ax)


def test_omission_plot_missingness_table_and_fallback_branch():
    x = SimpleNamespace(
        missingness=pd.DataFrame(
            {"missingness_class": ["answered", "omitted", "answered"]}
        )
    )
    ax = pp.plot_eye_omission_survival_irt(x)
    assert set(ax.gp3_data["missingness_class"]) == {"answered", "omitted"}
    _close(ax)

    fallback = SimpleNamespace(classified_missingness=pd.DataFrame({"other": [1, 2]}))
    ax = pp.plot_eye_omission_survival_irt(fallback)
    assert ax.gp3_data.empty
    _close(ax)


def test_manyfacet_plot_effects_empty_and_exception_paths(monkeypatch):
    effects = pd.DataFrame({"level": ["d1", "d2"], "estimate": [0.2, -0.1]})
    monkeypatch.setattr(
        pp,
        "facet_effects",
        lambda x, channel: SimpleNamespace(effects=effects.copy()),
    )
    ax = pp.plot_eye_manyfacet_process_irt(SimpleNamespace(facets={"device": object()}))
    assert len(ax.patches) == 2
    assert len(ax.gp3_data) == 2
    _close(ax)

    monkeypatch.setattr(
        pp,
        "facet_effects",
        lambda x, channel: SimpleNamespace(effects=pd.DataFrame()),
    )
    ax = pp.plot_eye_manyfacet_process_irt(SimpleNamespace(facets={}))
    assert ax.gp3_data.empty
    _close(ax)

    def fail(*args, **kwargs):
        raise RuntimeError("fixture")

    monkeypatch.setattr(pp, "facet_effects", fail)
    ax = pp.plot_eye_manyfacet_process_irt(SimpleNamespace(facets={}))
    assert ax.gp3_data.empty
    _close(ax)


def test_changepoint_plot_line_scatter_and_no_position_paths():
    d = pd.DataFrame(
        {
            "person": ["p1", "p2"],
            "changepoint": [2, 3],
            "score": [1.5, 0.5],
        }
    )
    ax = pp.plot_eye_irt_changepoints(SimpleNamespace(results=d), person="p1")
    assert len(ax.lines) == 1
    assert ax.gp3_data["person"].tolist() == ["p1"]
    _close(ax)

    pos_only = pd.DataFrame({"position": [2, 4]})
    ax = pp.plot_eye_irt_changepoints(SimpleNamespace(data=pos_only))
    assert len(ax.collections) == 1
    _close(ax)

    no_pos = pd.DataFrame({"evidence": [0.1, 0.2]})
    ax = pp.plot_eye_irt_changepoints(no_pos)
    assert len(ax.gp3_data) == 2
    _close(ax)


def test_hmm_transition_and_occupancy_dispatch(monkeypatch):
    transitions = pd.DataFrame(
        {
            "from_state": ["A", "A", "B", "B"],
            "to_state": ["A", "B", "A", "B"],
            "probability": [0.7, 0.3, 0.2, 0.8],
        }
    )
    monkeypatch.setattr(pp, "process_state_transition_summary", lambda x: transitions.copy())
    ax = pp.plot_eye_process_hmm_irt(object(), type="transition")
    assert len(ax.images) == 1
    assert len(ax.gp3_data) == 4
    _close(ax)

    occupancy = pd.DataFrame(
        {"state_a_occupancy": [0.7, 0.5], "state_b_occupancy": [0.3, 0.5]}
    )
    monkeypatch.setattr(pp, "process_state_occupancy", lambda x: occupancy.copy())
    ax = pp.plot_eye_process_hmm_irt(object(), type="occupancy")
    assert len(ax.patches) == 2
    assert len(ax.gp3_data) == 2
    _close(ax)

    no_occupancy = pd.DataFrame({"person": ["p1"]})
    monkeypatch.setattr(pp, "process_state_occupancy", lambda x: no_occupancy.copy())
    ax = pp.plot_eye_process_hmm_irt(object(), type="occupancy")
    assert len(ax.patches) == 0
    _close(ax)


def test_latent_space_plot_two_numeric_and_single_numeric_paths(monkeypatch):
    two = pd.DataFrame({"x": [-1.0, 1.0], "y": [0.5, -0.5]})
    monkeypatch.setattr(pp, "process_residual_map", lambda x: two.copy())
    ax = pp.plot_eye_latent_space_irt(object())
    assert len(ax.collections) == 1
    _close(ax)

    one = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]})
    monkeypatch.setattr(pp, "process_residual_map", lambda x: one.copy())
    ax = pp.plot_eye_latent_space_irt(object())
    assert len(ax.collections) == 0
    _close(ax)


def test_person_fit_score_and_no_score_paths():
    scored = pd.DataFrame(
        {
            "person": ["p1", "p2", "p3"],
            "joint_discrepancy": [0.2, 1.1, 0.6],
        }
    )
    ax = pp.plot_eye_process_person_fit(scored, top=2)
    assert len(ax.gp3_data) == 2
    assert ax.gp3_data["joint_discrepancy"].min() >= 0.6
    _close(ax)

    plain = pd.DataFrame({"person": ["p1", "p2"]})
    ax = pp.plot_eye_process_person_fit(plain)
    assert len(ax.gp3_data) == 2
    _close(ax)


def test_gpirt_item_selection_and_negative_control_paths(monkeypatch):
    comparison = pd.DataFrame(
        {
            "item": ["i1", "i1", "i2", "i2"],
            "theta": [-1.0, 1.0, -1.0, 1.0],
            "parametric": [0.2, 0.8, 0.3, 0.7],
            "flexible": [0.25, 0.75, 0.35, 0.65],
        }
    )
    monkeypatch.setattr(
        pp,
        "compare_parametric_nonparametric_irf",
        lambda response_matrix, gpirt_object=None: comparison.copy(),
    )
    fit = SimpleNamespace(response_matrix=np.array([[0, 1], [1, 0]]))

    ax = pp.plot_eye_gpirt(fit)
    assert set(ax.gp3_data["item"]) == {"i1"}
    assert len(ax.lines) == 2
    _close(ax)

    ax = pp.plot_eye_gpirt(fit, item=2)
    assert set(ax.gp3_data["item"]) == {"i2"}
    _close(ax)

    control = SimpleNamespace(null=np.linspace(-1.0, 1.0, 10), observed=0.25)
    ax = pp.plot_eye_process_negative_control(control)
    assert ax.gp3_data.shape == (10, 1)
    assert len(ax.lines) == 1
    _close(ax)
