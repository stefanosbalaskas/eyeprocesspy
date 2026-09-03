from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.plots_multimodal_staged as pm
from eyeprocesspy.exceptions import EyeProcessValidationError


def _close(ax):
    if ax is not None and hasattr(ax, "figure"):
        plt.close(ax.figure)


def _frame():
    return pd.DataFrame(
        {
            "person_id": ["P1", "P1", "P2", "P2"],
            "response": [0, 1, 1, 0],
            "rt": [0.5, 0.7, 0.6, 0.8],
            "gaze": [1.0, 2.0, 3.0, 4.0],
            "gaze_fixation_count": [1, 2, 3, 4],
            "pupil": [0.1, 0.2, 0.3, 0.4],
            "pupil_response": [0.2, 0.3, 0.4, 0.5],
            "pupil_nuisance_effect": [-0.1, 0.0, 0.1, 0.2],
        }
    )


def test_private_axis_dataframe_and_empty_helpers_cover_all_shapes():
    supplied = plt.subplots()[1]
    assert pm._ax(supplied) is supplied
    _close(supplied)

    frame = _frame()
    pd.testing.assert_frame_equal(pm._df(frame), frame)
    pd.testing.assert_frame_equal(pm._df({"data": frame}), frame)
    pd.testing.assert_frame_equal(pm._df(SimpleNamespace(data=frame)), frame)
    assert pm._df(object()).empty

    ax = pm._empty("empty", "nothing")
    assert ax.get_title() == "empty"
    assert ax.gp3_data.empty
    _close(ax)


def test_measurement_simulation_information_and_validation_dispatch(monkeypatch):
    channels = pd.DataFrame(
        {
            "channel": ["response", "rt"],
            "observed": [3, 2],
            "n": [4, 4],
            "missing_fraction": [0.25, 0.5],
        }
    )
    monkeypatch.setattr(
        pm,
        "audit_multimodal_measurement",
        lambda x: SimpleNamespace(channel_table=channels),
    )

    for kind in ("availability", "missingness"):
        ax = pm.plot_eye_multimodal_measurement(object(), type=kind)
        assert ax.gp3_data.equals(channels)
        _close(ax)
    with pytest.raises(EyeProcessValidationError):
        pm.plot_eye_multimodal_measurement(object(), type="bad")

    persons = pd.DataFrame(
        {
            "theta": [-1.0, 0.0, 1.0],
            "tau": [0.5, 0.0, -0.5],
        }
    )
    latent = SimpleNamespace(truth={"persons": persons}, data=_frame())
    ax = pm.plot_eye_multimodal_simulation(latent, type="latent_correlation")
    assert "latent" in ax.gp3_data
    _close(ax)

    ax = pm.plot_eye_multimodal_simulation({"data": _frame()}, type="channels")
    assert {"channel", "mean"} <= set(ax.gp3_data)
    _close(ax)

    with pytest.raises(EyeProcessValidationError):
        pm.plot_eye_multimodal_simulation(
            SimpleNamespace(truth={"persons": None}, data=_frame()),
            type="latent_correlation",
        )
    with pytest.raises(EyeProcessValidationError):
        pm.plot_eye_multimodal_simulation({"data": _frame()}, type="bad")

    nonempty = pd.DataFrame(
        {"value": [0.2], "target": ["gaze"], "metric": ["precision_gain"]}
    )
    ax = pm.plot_eye_process_information(nonempty)
    assert ax.get_xlabel() == "precision_gain"
    _close(ax)

    empty = pd.DataFrame(columns=["value", "target", "metric"])
    ax = pm.plot_eye_process_information(empty)
    assert ax.get_xlabel() == "information"
    _close(ax)

    checks = pd.DataFrame({"check": ["a", "b"], "pass": [True, False]})
    ax = pm.plot_eye_multimodal_validation(SimpleNamespace(checks=checks))
    assert ax.gp3_data.equals(checks)
    _close(ax)


def test_m2_plot_family_all_dispatch_and_fit_branches():
    d = _frame().rename(columns={"person_id": "person"})
    sim = SimpleNamespace(
        truth={
            "theta": np.array([-1.0, 0.0, 1.0]),
            "tau": np.array([0.5, 0.0, -0.5]),
            "omega": np.array([0.2, 0.4, 0.6]),
        },
        data=d,
    )
    for kind in ("person_latent", "channel_distributions", "missingness"):
        ax = pm.plot_eye_multimodal_m2_simulation(sim, type=kind)
        assert hasattr(ax, "gp3_data")
        _close(ax)
    with pytest.raises(EyeProcessValidationError):
        pm.plot_eye_multimodal_m2_simulation(sim, type="bad")

    empty_fit = pm.plot_eye_multimodal_m2_fit(SimpleNamespace(summary=pd.DataFrame()))
    assert empty_fit.gp3_data.empty
    _close(empty_fit)

    one_numeric = SimpleNamespace(
        summary=pd.DataFrame({"estimate": [0.1, 0.2], "term": ["a", "b"]})
    )
    ax = pm.plot_eye_multimodal_m2_fit(one_numeric)
    assert len(ax.gp3_data) == 2
    _close(ax)

    two_numeric = SimpleNamespace(
        summary=pd.DataFrame({"estimate": [0.1, 0.2], "se": [0.01, 0.02]})
    )
    ax = pm.plot_eye_multimodal_m2_fit(two_numeric)
    _close(ax)

    ppc = SimpleNamespace(
        summary=pd.DataFrame(
            {"channel": ["rt", "gaze"], "observed_mean": [0.6, 2.0]}
        )
    )
    _close(pm.plot_eye_multimodal_m2_ppc(ppc))

    info_observed = SimpleNamespace(
        summary=pd.DataFrame(
            {"channel": ["rt"], "observed_fraction": [0.9], "other": [1.0]}
        )
    )
    _close(pm.plot_eye_multimodal_m2_information(info_observed))
    info_fallback = SimpleNamespace(
        summary=pd.DataFrame({"channel": ["rt"], "score": [0.7]})
    )
    _close(pm.plot_eye_multimodal_m2_information(info_fallback))

    validation = SimpleNamespace(
        checks=pd.DataFrame({"criterion": ["c"], "pass": [True]})
    )
    _close(pm.plot_eye_multimodal_m2_validation(validation))
    recovery = SimpleNamespace(
        design=pd.DataFrame({"replicate": [1, 2], "seed": [10, 11]})
    )
    _close(pm.plot_eye_multimodal_m2_recovery(recovery))
    negative = SimpleNamespace(
        diagnostics=pd.DataFrame(
            {
                "dataset": ["observed", "control"],
                "channel": ["rt", "rt"],
                "mean": [0.6, 0.7],
            }
        )
    )
    _close(pm.plot_eye_multimodal_m2_negative_controls(negative))


def test_m3_plot_family_all_dispatch_and_fit_branches():
    sim = SimpleNamespace(data=_frame())
    for kind in ("pupil_nuisance", "missingness"):
        ax = pm.plot_eye_multimodal_m3_simulation(sim, type=kind)
        assert hasattr(ax, "gp3_data")
        _close(ax)
    with pytest.raises(EyeProcessValidationError):
        pm.plot_eye_multimodal_m3_simulation(sim, type="bad")

    empty_fit = pm.plot_eye_multimodal_m3_fit(SimpleNamespace(summary=pd.DataFrame()))
    assert empty_fit.gp3_data.empty
    _close(empty_fit)

    one_numeric = SimpleNamespace(
        summary=pd.DataFrame({"estimate": [0.1, 0.2], "term": ["a", "b"]})
    )
    _close(pm.plot_eye_multimodal_m3_fit(one_numeric))
    two_numeric = SimpleNamespace(
        summary=pd.DataFrame({"estimate": [0.1, 0.2], "se": [0.01, 0.02]})
    )
    _close(pm.plot_eye_multimodal_m3_fit(two_numeric))

    summary = pd.DataFrame({"channel": ["pupil"], "observed_mean": [0.3]})
    _close(pm.plot_eye_multimodal_m3_ppc(SimpleNamespace(summary=summary), type="item"))
    _close(
        pm.plot_eye_multimodal_m3_information(
            SimpleNamespace(
                pupil_observed_fraction=0.9,
                pupil_cost=0.2,
                decisive_z=1.5,
            )
        )
    )
    checks = pd.DataFrame({"criterion": ["c"], "pass": [True]})
    _close(pm.plot_eye_multimodal_m3_validation(SimpleNamespace(checks=checks)))
    design = pd.DataFrame(
        {
            "scenario": ["a", "a", "b"],
            "missingness": ["low", "high", "low"],
        }
    )
    _close(pm.plot_eye_multimodal_m3_recovery(SimpleNamespace(design=design)))

    datasets = {
        "observed": pd.DataFrame({"pupil": [0.1, 0.2, 0.3]}),
        "control": pd.DataFrame({"pupil": [0.3, 0.2, 0.1]}),
    }
    _close(pm.plot_eye_multimodal_m3_negative_controls(SimpleNamespace(datasets=datasets)))

    miss = pd.Series({"response": 0.0, "rt": 0.1, "gaze": 0.2, "pupil": 0.3})
    ident = SimpleNamespace(
        missing_fraction=miss,
        checks=pd.DataFrame({"criterion": ["a"], "pass": [True]}),
    )
    _close(pm.plot_eye_multimodal_m3_identifiability(ident, type="missingness"))
    _close(pm.plot_eye_multimodal_m3_identifiability(ident, type="checks"))


def _m4_sim():
    d = pd.DataFrame(
        {
            "person_id": ["P1", "P1", "P2", "P2"],
            "rt": [0.5, 0.6, 0.7, 0.8],
            "gaze": [1.0, 2.0, 3.0, 4.0],
            "pupil": [0.1, 0.2, 0.3, 0.4],
        }
    )
    return SimpleNamespace(data=d, truth={"state": np.array([0, 1, 1, 0])})


def test_m4_simulation_dispatch_filter_and_invalid_type():
    sim = _m4_sim()
    ax = pm.plot_eye_multimodal_m4_simulation(
        sim, type="state_sequence", person="P1"
    )
    assert set(ax.gp3_data["person_id"]) == {"P1"}
    _close(ax)
    for kind in ("channel_profile", "missingness"):
        _close(pm.plot_eye_multimodal_m4_simulation(sim, type=kind))
    with pytest.raises(EyeProcessValidationError):
        pm.plot_eye_multimodal_m4_simulation(sim, type="bad")


def test_m4_fit_and_state_diagnostic_dispatch(monkeypatch):
    probability = pd.DataFrame(
        {
            "state_1_probability": [0.8, 0.3, 0.4],
            "state_2_probability": [0.2, 0.7, 0.6],
            "posterior_entropy": [0.5, 0.6, 0.7],
        }
    )
    occupancy_mean = pd.DataFrame(
        {"state": [1, 2], "mean_probability": [0.5, 0.5]}
    )
    states = SimpleNamespace(probability=probability, occupancy=occupancy_mean)

    monkeypatch.setattr(pm, "multimodal_m4_state_diagnostics", lambda x: states)
    _close(pm.plot_eye_multimodal_m4_fit(object(), type="state_probability"))
    _close(pm.plot_eye_multimodal_m4_fit(object(), type="entropy"))

    monkeypatch.setattr(
        pm,
        "multimodal_m4_state_diagnostics",
        lambda x: (_ for _ in ()).throw(RuntimeError("no posterior")),
    )
    empty = pm.plot_eye_multimodal_m4_fit(object())
    assert empty.gp3_data.empty
    _close(empty)

    for kind in ("probability", "occupancy", "entropy", "certainty"):
        _close(pm.plot_eye_multimodal_m4_states(states, type=kind))

    occupancy_raw = pd.DataFrame({"state": [1, 2], "occupancy": [0.4, 0.6]})
    raw_states = SimpleNamespace(probability=probability, occupancy=occupancy_raw)
    _close(pm.plot_eye_multimodal_m4_states(raw_states, type="occupancy"))


def test_m4_remaining_plot_surfaces_and_information_empty_branch():
    checks = pd.DataFrame(
        {
            "criterion": ["a", "b", "c", "d", "e", "f"],
            "status": [
                "PASS",
                "PASS_WITH_CAUTION",
                "REVIEW",
                "NOT_EVALUATED",
                "FAIL",
                "UNKNOWN",
            ],
        }
    )
    _close(pm.plot_eye_multimodal_m4_identifiability(SimpleNamespace(checks=checks)))

    measurement = pd.DataFrame({"channel": ["rt", "gaze"], "mean": [0.6, 2.0]})
    _close(pm.plot_eye_multimodal_m4_ppc(SimpleNamespace(measurement=measurement)))

    occ_mean = pd.DataFrame(
        {"state": [1, 2], "mean_probability": [0.45, 0.55]}
    )
    _close(pm.plot_eye_multimodal_m4_information(SimpleNamespace(occupancy=occ_mean)))
    occ_raw = pd.DataFrame({"state": [1, 2], "occupancy": [0.4, 0.6]})
    _close(pm.plot_eye_multimodal_m4_information(SimpleNamespace(occupancy=occ_raw)))
    empty = pd.DataFrame(columns=["state", "occupancy"])
    ax = pm.plot_eye_multimodal_m4_information(SimpleNamespace(occupancy=empty))
    assert ax.gp3_data.empty
    _close(ax)

    _close(
        pm.plot_eye_multimodal_m4_negative_controls(
            SimpleNamespace(controls=["shuffle", "null"])
        )
    )
    _close(
        pm.plot_eye_multimodal_m4_sensitivity(
            SimpleNamespace(design=pd.DataFrame({"n_states": [1, 2, 3]}))
        )
    )
    _close(
        pm.plot_eye_multimodal_m4_recovery(
            SimpleNamespace(design=pd.DataFrame({"scenario": ["clear", "weak"]}))
        )
    )
    _close(pm.plot_eye_multimodal_m4_validation(SimpleNamespace(checks=checks)))
