from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy.bayesian_networks.fitting as fitting_module
import eyeprocesspy.bayesian_networks.inference as inference_module
import eyeprocesspy.bayesian_networks.learning as learning_module
import eyeprocesspy.bayesian_networks.plotting as plotting_module
import eyeprocesspy.bayesian_networks.reporting as reporting_module
import eyeprocesspy.bayesian_networks.sensitivity as sensitivity_module
import eyeprocesspy.bayesian_networks.stability as stability_module
import eyeprocesspy.bayesian_networks.validation as validation_module
from eyeprocesspy.bayesian_networks import (
    BayesianNetworkQueryResult,
    BayesianNetworkResult,
    BayesianNetworkStabilityResult,
    BayesianNetworkValidationResult,
    bootstrap_bn_structure,
    compare_bayesian_networks,
    compare_bn_across_aoi_specs,
    compare_bn_across_detectors,
    define_bn_constraints,
    fit_bayesian_network,
    learn_bayesian_network,
    plot_bayesian_network,
    plot_bn_detector_robustness,
    plot_bn_direction_stability,
    plot_bn_edge_stability,
    plot_bn_posterior,
    plot_bn_validation,
    predict_bayesian_network,
    prepare_bayesian_network_data,
    query_bayesian_network,
    report_bayesian_network,
    validate_bayesian_network,
)


def discrete_spec(repeated=True):
    d = pd.DataFrame(
        {
            "participant_id": (["p1", "p1", "p2", "p2"] if repeated else ["p1", "p2", "p3", "p4"]),
            "trial_id": (["t1", "t2", "t1", "t2"] if repeated else ["t1"] * 4),
            "condition": ["A", "B", "A", "B"],
            "trust": ["H", "H", "L", "L"],
            "choice": ["Y", "Y", "N", "N"],
        }
    )
    return prepare_bayesian_network_data(
        d,
        nodes={
            "condition": {"type": "categorical", "modality": "experimental"},
            "trust": {"type": "categorical", "modality": "questionnaire"},
            "choice": {"type": "categorical", "modality": "behavior"},
        },
        participant_id="participant_id",
        trial_id="trial_id",
        stimulus_id=None,
    )


def fake_result(
    spec,
    *,
    fitted=False,
    edges=(("condition", "trust"), ("trust", "choice")),
):
    return BayesianNetworkResult(
        edges=tuple(edges),
        nodes=tuple(spec.structure_nodes),
        model_family=spec.model_family,
        backend="pgmpy",
        backend_model=SimpleNamespace(nodes=lambda: list(spec.structure_nodes)),
        data_spec=spec,
        structure_algorithm="hill_climb",
        score="bic-d",
        parameter_estimator="bayesian" if fitted else None,
        fitted=fitted,
        provenance={"seed": 42},
    )


def test_learning_contract(monkeypatch):
    spec = discrete_spec()
    constraints = define_bn_constraints(
        temporal_tiers=[["condition"], ["trust"], ["choice"]],
        nodes=spec.structure_nodes,
    )
    monkeypatch.setattr(
        learning_module.pgmpy_backend,
        "learn_structure",
        lambda *a, **k: ("graph", (("condition", "trust"),), "bic-d"),
    )
    result = learn_bayesian_network(spec, constraints=constraints)
    assert result.edges == (("condition", "trust"),)
    assert result.provenance["random_state"] == 42
    with pytest.warns(UserWarning, match="Unconstrained"):
        unconstrained = learn_bayesian_network(spec)
    assert unconstrained.warnings
    with pytest.raises(ValueError):
        learn_bayesian_network(spec, backend="other")
    with pytest.raises(ValueError):
        learn_bayesian_network(spec, max_parents=0)

    invalid = discrete_spec()
    invalid.node_specs["trust"] = invalid.node_specs["trust"].__class__(
        "trust", "categorical", missing_allowed=False
    )
    invalid.data.loc[0, "trust"] = pd.NA
    with pytest.raises(ValueError, match="Invalid"):
        learn_bayesian_network(invalid, allow_unconstrained=True)


def test_fitting_fixed_and_learned(monkeypatch):
    spec = discrete_spec()
    sentinel = object()
    monkeypatch.setattr(
        fitting_module.pgmpy_backend,
        "fit_parameters",
        lambda **k: sentinel,
    )
    fixed = fit_bayesian_network(
        [("condition", "trust"), ("trust", "choice")],
        data=spec,
    )
    assert fixed.fitted
    assert fixed.backend_model is sentinel
    assert fixed.structure_algorithm == "fixed_dag"
    learned = fake_result(spec)
    fitted = fit_bayesian_network(learned)
    assert fitted.structure_algorithm == "hill_climb"
    with pytest.raises(ValueError):
        fit_bayesian_network([("a", "b")])
    with pytest.raises(ValueError):
        fit_bayesian_network([("a",)], data=spec)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        fit_bayesian_network([("a", "a")], data=spec)
    with pytest.raises(ValueError):
        fit_bayesian_network([("condition", "missing")], data=spec)
    with pytest.raises(ValueError):
        fit_bayesian_network([], data=spec, backend="other")
    with pytest.raises(ValueError):
        fit_bayesian_network([], data=spec, equivalent_sample_size=0)


def test_query_and_predict(monkeypatch):
    spec = discrete_spec()
    fitted = fake_result(spec, fitted=True)
    monkeypatch.setattr(
        inference_module.pgmpy_backend,
        "query",
        lambda *a, **k: (
            {"H": 0.8, "L": 0.2},
            None,
            None,
            "variable_elimination",
        ),
    )
    q = query_bayesian_network(
        fitted,
        target="trust",
        evidence={"choice": "Y"},
    )
    assert q.posterior == {"H": 0.8, "L": 0.2}
    q2 = predict_bayesian_network(
        fitted,
        target="trust",
        evidence={"choice": "Y"},
    )
    assert q2.posterior["H"] == 0.8
    pred = predict_bayesian_network(
        fitted,
        target="trust",
        evidence=pd.DataFrame({"choice": ["Y", "N"], "unused": [pd.NA, pd.NA]}),
    )
    assert len(pred) == 4
    with pytest.raises(ValueError):
        query_bayesian_network(fake_result(spec), target="trust")
    with pytest.raises(TypeError):
        predict_bayesian_network(
            fitted,
            target="trust",
            evidence=1,  # type: ignore[arg-type]
        )

    gaussian = fake_result(spec, fitted=True)
    gaussian.model_family = "gaussian"
    monkeypatch.setattr(
        inference_module.pgmpy_backend,
        "query",
        lambda *a, **k: (
            None,
            1.5,
            0.25,
            "gaussian_conditioning",
        ),
    )
    pred_g = predict_bayesian_network(
        gaussian,
        target="trust",
        evidence=pd.DataFrame({"choice": [1.0]}),
    )
    assert pred_g.loc[0, "mean"] == 1.5


def test_bootstrap_stability_group_and_failures(monkeypatch):
    spec = discrete_spec()
    calls = {"n": 0}

    def fake_learn(sampled, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("fit failed")
        edges = (("condition", "trust"),) if calls["n"] % 2 else (("trust", "condition"),)
        return fake_result(sampled, edges=edges)

    monkeypatch.setattr(
        stability_module,
        "learn_bayesian_network",
        fake_learn,
    )
    out = bootstrap_bn_structure(spec, n_boot=4, random_state=1)
    assert out.resample_by == "participant_id"
    assert out.failed_fits == 1
    assert out.successful_fits == 3
    assert out.edge_table.iloc[0]["edge_strength"] == pytest.approx(3 / 4)
    with pytest.raises(ValueError):
        bootstrap_bn_structure(spec, n_boot=0)
    with pytest.raises(ValueError):
        stability_module._resample_rows(
            spec,
            rng=np.random.default_rng(1),
            resample_by="missing",
        )

    no_groups = discrete_spec()
    no_groups.data["participant_id"] = pd.NA
    with pytest.raises(ValueError):
        stability_module._resample_rows(
            no_groups,
            rng=np.random.default_rng(1),
            resample_by="participant_id",
        )

    row_spec = discrete_spec(repeated=False)
    calls["n"] = 0
    row_out = bootstrap_bn_structure(
        row_spec,
        n_boot=1,
        resample_by=None,
        random_state=2,
    )
    assert row_out.resample_by is None


def test_grouped_validation_success_and_failure(monkeypatch):
    spec = discrete_spec(repeated=True)
    monkeypatch.setattr(
        validation_module,
        "learn_bayesian_network",
        lambda train, **k: fake_result(train),
    )
    monkeypatch.setattr(
        validation_module,
        "fit_bayesian_network",
        lambda learned, **k: fake_result(
            learned.data_spec,
            fitted=True,
        ),
    )
    scores = iter([-2.0, -3.0])
    monkeypatch.setattr(
        validation_module.pgmpy_backend,
        "log_likelihood",
        lambda *a, **k: next(scores),
    )
    out = validate_bayesian_network(spec, n_splits=2, random_state=1)
    assert out.fold_table["status"].eq("ok").all()
    assert out.summary["failed_folds"] == 0
    with pytest.raises(ValueError):
        validate_bayesian_network(spec, method="random")
    with pytest.raises(ValueError):
        validate_bayesian_network(spec, n_splits=1)
    no_group = discrete_spec()
    no_group.participant_id = None
    with pytest.raises(ValueError):
        validate_bayesian_network(no_group, groups=None, n_splits=2)
    with pytest.raises(ValueError):
        validate_bayesian_network(spec, n_splits=3)

    def fail_first(train, **k):
        if train.data["participant_id"].nunique() == 1:
            raise RuntimeError("bad fold")
        return fake_result(train)

    monkeypatch.setattr(
        validation_module,
        "learn_bayesian_network",
        fail_first,
    )
    failed = validate_bayesian_network(spec, n_splits=2, random_state=1)
    assert failed.summary["failed_folds"] == 2


def test_comparisons_and_sensitivity(monkeypatch):
    spec = discrete_spec()
    r1 = fake_result(
        spec,
        fitted=True,
        edges=(("condition", "trust"), ("trust", "choice")),
    )
    r2 = fake_result(
        spec,
        fitted=True,
        edges=(("trust", "condition"),),
    )
    comparison = compare_bayesian_networks({"ivt": r1, "idt": r2})
    assert comparison.edge_table["edge_robustness"].max() == 1.0
    with pytest.raises(ValueError):
        compare_bayesian_networks({})

    monkeypatch.setattr(
        sensitivity_module,
        "query_bayesian_network",
        lambda result, target, evidence: BayesianNetworkQueryResult(
            target=target,
            evidence=dict(evidence),
            posterior={"H": 0.7, "L": 0.3},
        ),
    )
    post = compare_bayesian_networks(
        {"a": r1},
        target="trust",
        evidence={"choice": "Y"},
    )
    assert len(post.posterior_table) == 2
    monkeypatch.setattr(
        sensitivity_module,
        "query_bayesian_network",
        lambda result, target, evidence: BayesianNetworkQueryResult(
            target=target,
            evidence=dict(evidence),
            posterior=None,
            mean=1.0,
            variance=0.2,
        ),
    )
    postg = compare_bayesian_networks({"a": r1}, target="trust")
    assert postg.posterior_table.loc[0, "mean"] == 1.0

    monkeypatch.setattr(
        sensitivity_module,
        "learn_bayesian_network",
        lambda data_spec, **k: fake_result(data_spec),
    )
    det = compare_bn_across_detectors({"ivt": spec, "idt": spec})
    assert det.provenance["sensitivity_axis"] == "event_detector"
    aoi = compare_bn_across_aoi_specs({"nominal": spec})
    assert aoi.provenance["sensitivity_axis"] == "aoi_specification"
    with pytest.raises(ValueError):
        compare_bn_across_detectors({})


def test_plots_and_report(tmp_path: Path):
    spec = discrete_spec()
    result = fake_result(spec, fitted=True)
    stab = BayesianNetworkStabilityResult(
        edge_table=pd.DataFrame(
            {
                "node_a": ["condition"],
                "node_b": ["trust"],
                "edge": ["condition--trust"],
                "edge_strength": [0.9],
                "direction": ["condition->trust"],
                "direction_strength": [0.8],
                "present_count": [9],
                "successful_fits": [10],
            }
        ),
        n_boot=10,
        resample_by="participant_id",
        successful_fits=10,
        failed_fits=0,
    )
    val = BayesianNetworkValidationResult(
        fold_table=pd.DataFrame({"fold": [0, 1], "mean_log_likelihood": [-1.0, -1.2]}),
        method="group_kfold",
        group_column="participant_id",
        summary={"mean_log_likelihood": -1.1},
    )
    comp = compare_bayesian_networks({"a": result})
    for ax in [
        plot_bayesian_network(result, edge_strength=stab.edge_table),
        plot_bn_edge_stability(stab),
        plot_bn_direction_stability(stab),
        plot_bn_posterior(
            BayesianNetworkQueryResult(
                "trust",
                {},
                {"H": 0.8, "L": 0.2},
            )
        ),
        plot_bn_posterior(
            BayesianNetworkQueryResult(
                "x",
                {},
                None,
                mean=1.0,
                variance=0.25,
            )
        ),
        plot_bn_detector_robustness(comp),
        plot_bn_validation(val),
    ]:
        assert hasattr(ax, "eyeprocess_plot_data")
        plt.close(ax.figure)
    with pytest.raises(ValueError):
        plot_bn_posterior(BayesianNetworkQueryResult("x", {}, None))

    path = tmp_path / "report.md"
    report = report_bayesian_network(
        result,
        stability=stab,
        validation=val,
        sensitivity=comp,
        path=path,
    )
    assert "not, by itself, evidence of a causal effect" in report
    assert "condition" in report
    assert path.read_text() == report

    empty_stab = BayesianNetworkStabilityResult(
        pd.DataFrame(),
        2,
        None,
        0,
        2,
    )
    empty_comp = comp.__class__(pd.DataFrame())
    report2 = report_bayesian_network(
        result,
        stability=empty_stab,
        sensitivity=empty_comp,
    )
    assert "No edges were recovered" in report2


def test_additional_fitting_stability_plot_reporting_branches(monkeypatch):
    spec = discrete_spec()
    monkeypatch.setattr(
        fitting_module.pgmpy_backend,
        "fit_parameters",
        lambda **k: object(),
    )
    with pytest.raises(ValueError, match="Each edge"):
        fit_bayesian_network([("a",)], data=spec)  # type: ignore[list-item]
    invalid = discrete_spec()
    invalid.node_specs["trust"] = invalid.node_specs["trust"].__class__(
        "trust",
        "categorical",
        missing_allowed=False,
    )
    invalid.data.loc[0, "trust"] = pd.NA
    with pytest.raises(ValueError, match="Invalid"):
        fit_bayesian_network([], data=invalid)

    no_id = prepare_bayesian_network_data(
        pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]}),
        nodes={
            "a": {"type": "categorical"},
            "b": {"type": "categorical"},
        },
        participant_id=None,
        trial_id=None,
        stimulus_id=None,
        observation_level="participant",
    )
    monkeypatch.setattr(
        stability_module,
        "learn_bayesian_network",
        lambda sampled, **k: fake_result(sampled, edges=()),
    )
    out = bootstrap_bn_structure(no_id, n_boot=1, random_state=1)
    assert out.resample_by is None
    assert out.edge_table.empty

    result = fake_result(spec, fitted=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    stab = BayesianNetworkStabilityResult(
        edge_table=pd.DataFrame(
            {
                "node_a": ["condition"],
                "node_b": ["trust"],
                "edge": ["condition--trust"],
                "edge_strength": [0.5],
                "direction": ["condition->trust"],
                "direction_strength": [0.6],
                "present_count": [1],
                "successful_fits": [1],
            }
        ),
        n_boot=1,
        resample_by=None,
        successful_fits=1,
        failed_fits=0,
    )
    val = BayesianNetworkValidationResult(
        fold_table=pd.DataFrame({"fold": [0], "mean_log_likelihood": [-1.0]}),
        method="group_kfold",
        group_column=None,
        summary={"mean_log_likelihood": -1.0},
    )
    comp = compare_bayesian_networks({"one": result})
    plot_bayesian_network(result, edge_strength=None, ax=axes[0, 0])
    plot_bn_edge_stability(stab, ax=axes[0, 1])
    plot_bn_direction_stability(stab, ax=axes[0, 2])
    plot_bn_posterior(
        BayesianNetworkQueryResult("trust", {}, {"H": 1.0}),
        ax=axes[1, 0],
    )
    plot_bn_detector_robustness(comp, ax=axes[1, 1])
    plot_bn_validation(val, ax=axes[1, 2])
    plt.close(fig)

    original_import = plotting_module.import_module
    monkeypatch.setattr(
        plotting_module,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(ImportError, match="matplotlib"):
        plotting_module._plt()
    monkeypatch.setattr(
        plotting_module,
        "import_module",
        lambda name: (
            original_import(name)
            if name == "matplotlib.pyplot"
            else (_ for _ in ()).throw(ImportError(name))
        ),
    )
    with pytest.raises(ImportError, match="networkx"):
        plot_bayesian_network(result)

    assert reporting_module._markdown_table(pd.DataFrame()) == ""
    empty_result = fake_result(spec, fitted=True, edges=())
    empty_result.warnings = ("warning text",)
    minimal = report_bayesian_network(empty_result)
    assert "Warnings" in minimal
    assert "warning text" in minimal
