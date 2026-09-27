from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyeprocesspy.bayesian_networks import (
    cpt_sensitivity_analysis,
    discretization_sensitivity,
    fit_bayesian_network,
    measurement_noise_sensitivity,
    plot_bn_cpt_sensitivity,
    plot_bn_predictive_calibration,
    plot_bn_sample_size_stability,
    predictive_calibration,
    prepare_bayesian_network_data,
    sample_size_stability_curve,
    structural_perturbation_sensitivity,
)

pytest.importorskip("pgmpy")


def _discrete_spec(n: int = 120):
    rng = np.random.default_rng(8)
    condition = rng.choice(["A", "B"], n)
    trust = np.where(rng.random(n) < np.where(condition == "A", 0.75, 0.30), "H", "L")
    choice = np.where(rng.random(n) < np.where(trust == "H", 0.80, 0.20), "Y", "N")
    data = pd.DataFrame(
        {
            "participant_id": [f"p{i // 3}" for i in range(n)],
            "trial_id": [f"t{i}" for i in range(n)],
            "condition": condition,
            "trust": trust,
            "choice": choice,
        }
    )
    return prepare_bayesian_network_data(
        data,
        nodes={
            "condition": {"type": "categorical", "modality": "experimental"},
            "trust": {"type": "categorical", "modality": "questionnaire"},
            "choice": {"type": "categorical", "modality": "behavior"},
        },
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id=None,
    )


def _fitted():
    spec = _discrete_spec()
    return fit_bayesian_network(
        [("condition", "trust"), ("trust", "choice")],
        data=spec,
        estimator="bayesian",
        equivalent_sample_size=5,
    )


def test_cpt_sensitivity_and_predictive_calibration_plots():
    fitted = _fitted()
    cpt = cpt_sensitivity_analysis(
        fitted,
        node="trust",
        state="H",
        values=(0.2, 0.5, 0.8),
        target="choice",
        evidence={"condition": "A"},
        parent_configuration={"condition": "A"},
    )
    assert cpt.eyeprocess_class == "eye_bn_cpt_sensitivity"
    assert set(cpt.table.requested_probability) == {0.2, 0.5, 0.8}
    ax = plot_bn_cpt_sensitivity(cpt, target_state="Y")
    assert hasattr(ax, "eyeprocess_plot_data")
    ax2 = plot_bn_cpt_sensitivity(cpt)
    assert hasattr(ax2, "eyeprocess_plot_data")

    root = cpt_sensitivity_analysis(
        fitted,
        node="condition",
        state="A",
        values=(0.3, 0.7),
        target="choice",
    )
    assert len(root.table) > 0

    calibration = predictive_calibration(fitted, target="choice", n_bins=5)
    assert calibration.summary["n"] == len(fitted.data_spec.data)
    assert calibration.summary["brier_multiclass"] >= 0
    ax = plot_bn_predictive_calibration(calibration)
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(Exception):
        cpt_sensitivity_analysis(fitted, node="trust", state="missing", values=(0.5,), target="choice", parent_configuration={"condition": "A"})
    with pytest.raises(Exception):
        cpt_sensitivity_analysis(fitted, node="trust", state="H", values=(0.5,), target="choice")
    with pytest.raises(Exception):
        cpt_sensitivity_analysis(fitted, node="condition", state="A", values=(0.5,), target="choice", parent_configuration={"trust": "H"})
    with pytest.raises(Exception):
        cpt_sensitivity_analysis(fitted, node="condition", state="A", values=(0, 1), target="choice")
    with pytest.raises(Exception):
        predictive_calibration(fitted, target="missing")
    with pytest.raises(Exception):
        predictive_calibration(fitted, target="choice", positive_state="missing")
    with pytest.raises(Exception):
        predictive_calibration(fitted, target="choice", n_bins=1)
    with pytest.raises(Exception):
        predictive_calibration(
            fitted,
            target="choice",
            data=fitted.data_spec.data.drop(columns="trust"),
        )
    with pytest.raises(Exception):
        plot_bn_cpt_sensitivity({})
    with pytest.raises(Exception):
        plot_bn_predictive_calibration({})


def test_structural_perturbation_and_cycle_accounting():
    fitted = _fitted()
    comparison = structural_perturbation_sensitivity(
        fitted,
        target="choice",
        evidence={"condition": "A"},
        include_delete=True,
        include_reverse=True,
        add_edges=(("condition", "choice"), ("choice", "condition")),
    )
    assert comparison.provenance["planned_perturbations"] == 6
    assert comparison.provenance["failures"]
    assert "baseline" in comparison.edge_table.columns

    with pytest.raises(Exception):
        structural_perturbation_sensitivity(fitted, add_edges=(("missing", "choice"),))


def _continuous_spec(n: int = 100):
    rng = np.random.default_rng(12)
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(scale=0.6, size=n)
    z = 0.5 * y + rng.normal(scale=0.6, size=n)
    data = pd.DataFrame({"x": x, "y": y, "z": z})
    return prepare_bayesian_network_data(
        data,
        nodes={
            "x": {"type": "continuous", "modality": "gaze"},
            "y": {"type": "continuous", "modality": "pupil"},
            "z": {"type": "continuous", "modality": "behavior"},
        },
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
    )


def test_discretization_noise_and_sample_size_stability():
    continuous = _continuous_spec()
    discrete_comparison = discretization_sensitivity(
        continuous,
        columns=("x", "y", "z"),
        schemes={
            "tertiles": {"method": "quantile", "bins": 3},
            "equal_width": {"method": "width", "bins": 3},
            "explicit": {"method": "cuts", "bins": [-10, -0.5, 0.5, 10]},
        },
    )
    assert discrete_comparison.provenance["sensitivity_axis"] == "discretization"

    noisy = measurement_noise_sensitivity(
        continuous,
        columns=("x", "y"),
        noise_scales=(0, 0.1),
        repeats=1,
    )
    assert noisy.provenance["sensitivity_axis"] == "measurement_noise"

    discrete = _discrete_spec()
    curve = sample_size_stability_curve(
        discrete,
        fractions=(0.7, 1.0),
        repeats=1,
        resample_by="participant_id",
    )
    assert len(curve) == 2
    ax = plot_bn_sample_size_stability(curve)
    assert hasattr(ax, "eyeprocess_plot_data")
    ax = plot_bn_sample_size_stability(curve.drop(columns="status"))
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(Exception):
        discretization_sensitivity(continuous, columns=("missing",), schemes={"x": {"method": "quantile", "bins": 3}})
    with pytest.raises(Exception):
        discretization_sensitivity(continuous, columns=("x",), schemes={})
    with pytest.raises(Exception):
        discretization_sensitivity(continuous, columns=("x",), schemes={"bad": {"method": "bad", "bins": 3}})
    incomplete = _continuous_spec()
    incomplete.data.loc[0, "x"] = np.nan
    with pytest.raises(Exception):
        discretization_sensitivity(incomplete, columns=("x",), schemes={"q": {"method": "quantile", "bins": 3}})
    with pytest.raises(Exception):
        measurement_noise_sensitivity(continuous, columns=("missing",))
    with pytest.raises(Exception):
        measurement_noise_sensitivity(continuous, columns=("x",), repeats=0)
    with pytest.raises(Exception):
        measurement_noise_sensitivity(continuous, columns=("x",), noise_scales=(-1,), repeats=1)
    with pytest.raises(Exception):
        sample_size_stability_curve(discrete, fractions=(0,), repeats=1)
    with pytest.raises(Exception):
        sample_size_stability_curve(discrete, fractions=(1,), repeats=0)
    with pytest.raises(Exception):
        sample_size_stability_curve(discrete, fractions=(1,), repeats=1, resample_by="missing")
    missing_group = _discrete_spec()
    missing_group.data.loc[0, "participant_id"] = pd.NA
    with pytest.raises(Exception):
        sample_size_stability_curve(
            missing_group,
            fractions=(1,),
            repeats=1,
            resample_by="participant_id",
        )
    with pytest.raises(Exception):
        plot_bn_sample_size_stability(pd.DataFrame({"x": [1]}))
