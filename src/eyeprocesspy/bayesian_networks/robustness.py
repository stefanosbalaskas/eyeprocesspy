"""Robustness, stability, sensitivity, and predictive calibration for Bayesian networks."""

from __future__ import annotations

import copy
import importlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import EyeProcessValidationError
from ..irt import EyeResult
from .backends import pgmpy_backend
from .fitting import fit_bayesian_network
from .inference import query_bayesian_network
from .learning import learn_bayesian_network
from .schema import (
    BayesianConstraintSpec,
    BayesianDataSpec,
    BayesianNetworkComparisonResult,
    BayesianNetworkResult,
)
from .sensitivity import compare_bayesian_networks


def _require_result(result: BayesianNetworkResult, *, fitted: bool = False) -> None:
    if not isinstance(result, BayesianNetworkResult):
        raise EyeProcessValidationError("result must be a BayesianNetworkResult.")
    if fitted and not result.fitted:
        raise EyeProcessValidationError("This analysis requires a fitted Bayesian network.")


def _acyclic(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        if source not in adjacency or target not in adjacency or source == target:
            return False
        adjacency[source].append(target)
        indegree[target] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(nodes)


def cpt_sensitivity_analysis(
    result: BayesianNetworkResult,
    *,
    node: str,
    state: Any,
    values: Sequence[float],
    target: str,
    evidence: Mapping[str, Any] | None = None,
    parent_configuration: Mapping[str, Any] | None = None,
) -> EyeResult:
    """Vary one discrete CPT probability and propagate the change to a target query.

    The remaining probabilities in the same CPT column are rescaled
    proportionally. The original fitted model is never modified.
    """
    _require_result(result, fitted=True)
    if result.model_family != "discrete":
        raise EyeProcessValidationError("CPT sensitivity currently requires a discrete BN.")
    model = result.backend_model
    cpd = model.get_cpds(node)
    if cpd is None:
        raise EyeProcessValidationError(f"No CPT is available for node {node!r}.")
    state_names = [str(v) for v in cpd.state_names[node]]
    state_text = str(state)
    if state_text not in state_names:
        raise EyeProcessValidationError(f"Unknown state {state!r} for node {node!r}.")
    row_index = state_names.index(state_text)
    parents = list(cpd.variables[1:])
    config = dict(parent_configuration or {})
    if parents and set(config) != set(parents):
        raise EyeProcessValidationError(
            "parent_configuration must provide exactly the CPT parent variables."
        )
    if not parents and config:
        raise EyeProcessValidationError("parent_configuration is not valid for a root node.")

    column_index = 0
    if parents:
        parent_states = [list(cpd.state_names[parent]) for parent in parents]
        configurations = list(itertools.product(*parent_states))
        wanted = tuple(config[parent] for parent in parents)
        try:
            column_index = configurations.index(wanted)
        except ValueError as exc:
            raise EyeProcessValidationError("parent_configuration contains an unknown parent state.") from exc

    probabilities = np.asarray(cpd.get_values(), dtype=float)
    requested = [float(value) for value in values]
    if not requested or not np.isfinite(requested).all() or any(value <= 0 or value >= 1 for value in requested):
        raise EyeProcessValidationError("values must contain finite probabilities strictly inside (0, 1).")

    discrete = importlib.import_module("pgmpy.factors.discrete")
    rows = []
    for requested_probability in requested:
        changed = probabilities.copy()
        old_column = changed[:, column_index].copy()
        old_other = float(old_column.sum() - old_column[row_index])
        remaining = 1.0 - requested_probability
        if len(old_column) == 1:
            raise EyeProcessValidationError("A one-state CPT cannot be sensitivity-perturbed.")
        if old_other > 0:
            changed[:, column_index] = old_column * (remaining / old_other)
            changed[row_index, column_index] = requested_probability
        else:
            changed[:, column_index] = remaining / (len(old_column) - 1)
            changed[row_index, column_index] = requested_probability

        cloned = copy.deepcopy(model)
        cloned.remove_cpds(cloned.get_cpds(node))
        kwargs: dict[str, Any] = {
            "variable": node,
            "variable_card": int(cpd.variable_card),
            "values": changed,
            "state_names": {name: list(cpd.state_names[name]) for name in cpd.variables},
        }
        if parents:
            kwargs["evidence"] = parents
            kwargs["evidence_card"] = [int(value) for value in cpd.cardinality[1:]]
        cloned.add_cpds(discrete.TabularCPD(**kwargs))
        cloned.check_model()
        posterior, mean, variance, _ = pgmpy_backend.query(
            cloned,
            family="discrete",
            target=target,
            evidence=dict(evidence or {}),
        )
        if posterior is not None:
            for target_state, probability in posterior.items():
                rows.append(
                    {
                        "node": node,
                        "state": state_text,
                        "requested_probability": requested_probability,
                        "target": target,
                        "target_state": target_state,
                        "target_probability": probability,
                    }
                )
        else:
            rows.append(
                {
                    "node": node,
                    "state": state_text,
                    "requested_probability": requested_probability,
                    "target": target,
                    "target_mean": mean,
                    "target_variance": variance,
                }
            )
    return EyeResult(
        {
            "table": pd.DataFrame(rows),
            "parent_configuration": config,
            "evidence": dict(evidence or {}),
            "caveat": "CPT sensitivity measures dependence on a fitted probability assumption; it does not identify a causal effect.",
        },
        eyeprocess_class="eye_bn_cpt_sensitivity",
    )


def structural_perturbation_sensitivity(
    result: BayesianNetworkResult,
    *,
    target: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    include_delete: bool = True,
    include_reverse: bool = True,
    add_edges: Sequence[tuple[str, str]] = (),
    estimator: str | None = None,
    prior: str = "BDeu",
    equivalent_sample_size: float = 10.0,
) -> BayesianNetworkComparisonResult:
    """Delete, reverse, or add declared arcs and refit parameters on the same data."""
    _require_result(result, fitted=True)
    estimator = result.parameter_estimator or "bayesian" if estimator is None else estimator
    nodes = list(result.nodes)
    baseline_edges = tuple(result.edges)
    candidates: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    if include_delete:
        for edge in baseline_edges:
            candidates.append((f"delete:{edge[0]}->{edge[1]}", tuple(e for e in baseline_edges if e != edge)))
    if include_reverse:
        for source, target_node in baseline_edges:
            changed = tuple(e for e in baseline_edges if e != (source, target_node)) + ((target_node, source),)
            candidates.append((f"reverse:{source}->{target_node}", changed))
    for source, target_node in add_edges:
        if source not in nodes or target_node not in nodes:
            raise EyeProcessValidationError("add_edges references an unknown BN node.")
        changed = baseline_edges + ((str(source), str(target_node)),)
        candidates.append((f"add:{source}->{target_node}", changed))

    successful: dict[str, BayesianNetworkResult] = {"baseline": result}
    failures = []
    for name, edges in candidates:
        unique = tuple(sorted(set(edges)))
        if not _acyclic(nodes, unique):
            failures.append({"specification": name, "error": "perturbation_not_acyclic"})
            continue
        try:
            successful[name] = fit_bayesian_network(
                unique,
                data=result.data_spec,
                estimator=str(estimator),
                prior=prior,
                equivalent_sample_size=equivalent_sample_size,
                constraints=result.constraints,
                backend=result.backend,
            )
        except Exception as exc:
            failures.append({"specification": name, "error": str(exc)})
    comparison = compare_bayesian_networks(
        successful,
        target=target,
        evidence=dict(evidence or {}) if target is not None else None,
    )
    comparison.provenance.update(
        {
            "sensitivity_axis": "structural_perturbation",
            "planned_perturbations": len(candidates),
            "successful_perturbations": len(successful) - 1,
            "failures": failures,
        }
    )
    return comparison


def _discretize(series: pd.Series, *, method: str, bins: Any) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise EyeProcessValidationError("Discretization sensitivity does not silently impute missing values.")
    if method == "quantile":
        codes = pd.qcut(numeric, q=int(bins), labels=False, duplicates="drop")
    elif method == "width":
        codes = pd.cut(numeric, bins=int(bins), labels=False, include_lowest=True)
    elif method == "cuts":
        cuts = np.asarray(bins, dtype=float)
        if cuts.ndim != 1 or len(cuts) < 3 or not np.all(np.diff(cuts) > 0):
            raise EyeProcessValidationError("Explicit cuts must be a strictly increasing vector with at least three boundaries.")
        codes = pd.cut(numeric, bins=cuts, labels=False, include_lowest=True)
    else:
        raise EyeProcessValidationError("Discretization method must be quantile, width, or cuts.")
    if pd.Series(codes).isna().any():
        raise EyeProcessValidationError("The declared discretization left observations outside valid bins.")
    return pd.Series(codes, index=series.index).astype(int).map(lambda value: f"bin_{value}")


def discretization_sensitivity(
    data_spec: BayesianDataSpec,
    *,
    columns: Sequence[str],
    schemes: Mapping[str, Mapping[str, Any]],
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    random_state: int | None = 42,
) -> BayesianNetworkComparisonResult:
    """Relearn a BN across explicitly declared discretization schemes."""
    if not schemes:
        raise EyeProcessValidationError("schemes must contain at least one named discretization.")
    unknown = [column for column in columns if column not in data_spec.node_specs]
    if unknown:
        raise EyeProcessValidationError(f"Unknown discretization node(s): {unknown}.")
    learned = {}
    for name, scheme in schemes.items():
        transformed = data_spec.data.copy()
        specs = dict(data_spec.node_specs)
        for column in columns:
            transformed[column] = _discretize(
                transformed[column],
                method=str(scheme.get("method", "quantile")).lower(),
                bins=scheme.get("bins", 3),
            )
            states = tuple(sorted(transformed[column].astype(str).unique()))
            specs[column] = replace(specs[column], variable_type="ordinal", states=states)
        branch = replace(
            data_spec,
            data=transformed,
            node_specs=specs,
            provenance={**data_spec.provenance, "discretization_scheme": str(name), "discretization": dict(scheme)},
        )
        learned[str(name)] = learn_bayesian_network(
            branch,
            algorithm=algorithm,
            score=score,
            constraints=constraints,
            random_state=random_state,
            allow_unconstrained=True,
        )
    comparison = compare_bayesian_networks(learned)
    comparison.provenance["sensitivity_axis"] = "discretization"
    return comparison


def measurement_noise_sensitivity(
    data_spec: BayesianDataSpec,
    *,
    columns: Sequence[str],
    noise_scales: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    repeats: int = 3,
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    seed: int = 1,
) -> BayesianNetworkComparisonResult:
    """Relearn structures after declared Gaussian measurement perturbations."""
    repeats, seed = int(repeats), int(seed)
    if repeats < 1 or seed < 0:
        raise EyeProcessValidationError("repeats must be positive and seed non-negative.")
    for column in columns:
        if column not in data_spec.data:
            raise EyeProcessValidationError(f"Unknown noise column: {column!r}.")
        if data_spec.node_specs[column].variable_type != "continuous":
            raise EyeProcessValidationError("Gaussian measurement noise is only applied to declared continuous nodes.")
    rng = np.random.default_rng(seed)
    learned = {}
    for scale in noise_scales:
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0:
            raise EyeProcessValidationError("noise_scales must be finite and non-negative.")
        for repeat_id in range(1, repeats + 1):
            data = data_spec.data.copy()
            for column in columns:
                values = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all():
                    raise EyeProcessValidationError("Measurement-noise sensitivity does not silently impute missing values.")
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                data[column] = values + rng.normal(0, scale * sd, len(values))
            branch = replace(
                data_spec,
                data=data,
                provenance={**data_spec.provenance, "measurement_noise_scale": scale, "repeat": repeat_id},
            )
            label = f"noise={scale:g}:rep={repeat_id}"
            learned[label] = learn_bayesian_network(
                branch,
                algorithm=algorithm,
                score=score,
                constraints=constraints,
                random_state=seed + repeat_id,
                allow_unconstrained=True,
            )
    comparison = compare_bayesian_networks(learned)
    comparison.provenance["sensitivity_axis"] = "measurement_noise"
    return comparison


def sample_size_stability_curve(
    data_spec: BayesianDataSpec,
    *,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    repeats: int = 10,
    resample_by: str | None = None,
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    """Estimate skeleton recovery against the full-data learned graph by sample fraction."""
    repeats, seed = int(repeats), int(seed)
    if repeats < 1 or seed < 0:
        raise EyeProcessValidationError("repeats must be positive and seed non-negative.")
    if resample_by is not None and resample_by not in data_spec.data:
        raise EyeProcessValidationError("resample_by is not present in BayesianDataSpec.data.")
    baseline = learn_bayesian_network(
        data_spec,
        algorithm=algorithm,
        score=score,
        constraints=constraints,
        random_state=seed,
        allow_unconstrained=True,
    )
    baseline_skeleton = {tuple(sorted(edge)) for edge in baseline.edges}
    rng = np.random.default_rng(seed)
    rows = []
    for fraction in fractions:
        fraction = float(fraction)
        if not 0 < fraction <= 1:
            raise EyeProcessValidationError("fractions must lie in (0, 1].")
        for repeat_id in range(1, repeats + 1):
            if resample_by is None:
                n = max(2, int(round(len(data_spec.data) * fraction)))
                indices = rng.choice(len(data_spec.data), size=min(n, len(data_spec.data)), replace=False)
                sampled_data = data_spec.data.iloc[np.sort(indices)].reset_index(drop=True)
            else:
                groups = pd.unique(data_spec.data[resample_by].dropna())
                n = max(1, int(round(len(groups) * fraction)))
                chosen = rng.choice(groups, size=min(n, len(groups)), replace=False)
                sampled_data = data_spec.data.loc[data_spec.data[resample_by].isin(chosen)].reset_index(drop=True)
            sampled = replace(data_spec, data=sampled_data)
            try:
                fitted = learn_bayesian_network(
                    sampled,
                    algorithm=algorithm,
                    score=score,
                    constraints=constraints,
                    random_state=seed + repeat_id,
                    allow_unconstrained=True,
                )
                skeleton = {tuple(sorted(edge)) for edge in fitted.edges}
                union = baseline_skeleton | skeleton
                agreement = len(baseline_skeleton & skeleton) / len(union) if union else 1.0
                rows.append({"fraction": fraction, "repeat": repeat_id, "n_rows": len(sampled_data), "skeleton_jaccard": agreement, "n_edges": len(fitted.edges), "status": "success", "error": None})
            except Exception as exc:
                rows.append({"fraction": fraction, "repeat": repeat_id, "n_rows": len(sampled_data), "skeleton_jaccard": np.nan, "n_edges": np.nan, "status": "failed", "error": str(exc)})
    return pd.DataFrame(rows)


def predictive_calibration(
    result: BayesianNetworkResult,
    *,
    target: str,
    data: pd.DataFrame | None = None,
    positive_state: Any | None = None,
    n_bins: int = 10,
) -> EyeResult:
    """Compute held-out-style probability calibration diagnostics for a fitted discrete BN."""
    _require_result(result, fitted=True)
    if result.model_family != "discrete":
        raise EyeProcessValidationError("Probability calibration currently requires a discrete BN.")
    frame = result.data_spec.data if data is None else data.copy()
    if target not in frame or target not in result.nodes:
        raise EyeProcessValidationError("target must be present in both data and the fitted BN.")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise EyeProcessValidationError("n_bins must be at least 2.")
    cpd = result.backend_model.get_cpds(target)
    states = [str(value) for value in cpd.state_names[target]]
    if positive_state is None and len(states) == 2:
        positive_state = states[-1]
    if positive_state is not None and str(positive_state) not in states:
        raise EyeProcessValidationError("positive_state is not a state of the target node.")

    rows = []
    multiclass_brier = []
    log_losses = []
    for row_id, row in frame.reset_index(drop=True).iterrows():
        observed = row[target]
        if pd.isna(observed):
            continue
        evidence = {
            node: row[node]
            for node in result.nodes
            if node != target and node in frame and not pd.isna(row[node])
        }
        query = query_bayesian_network(result, target=target, evidence=evidence)
        probabilities = dict(query.posterior or {})
        observed_text = str(observed)
        if observed_text not in probabilities:
            raise EyeProcessValidationError("Observed target state is not represented by the fitted CPT.")
        vector = np.asarray([float(probabilities.get(state, 0.0)) for state in states])
        truth = np.asarray([float(state == observed_text) for state in states])
        multiclass_brier.append(float(np.sum((vector - truth) ** 2)))
        probability_observed = max(float(probabilities[observed_text]), np.finfo(float).eps)
        log_losses.append(float(-np.log(probability_observed)))
        if positive_state is not None:
            predicted = float(probabilities[str(positive_state)])
            outcome = float(observed_text == str(positive_state))
        else:
            predicted = float(vector.max())
            outcome = float(states[int(vector.argmax())] == observed_text)
        rows.append({"row": row_id, "predicted_probability": predicted, "observed": outcome})

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise EyeProcessValidationError("No complete target observations are available for calibration.")
    predictions["bin"] = pd.cut(
        predictions.predicted_probability,
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        labels=False,
    )
    calibration = (
        predictions.groupby("bin", observed=True, dropna=True)
        .agg(n=("observed", "size"), mean_predicted=("predicted_probability", "mean"), observed_rate=("observed", "mean"))
        .reset_index()
    )
    summary = {
        "n": int(len(predictions)),
        "brier_multiclass": float(np.mean(multiclass_brier)),
        "log_loss": float(np.mean(log_losses)),
        "calibration_target": str(positive_state) if positive_state is not None else "maximum_confidence_correctness",
    }
    return EyeResult(
        {"predictions": predictions, "calibration": calibration, "summary": summary},
        eyeprocess_class="eye_bn_predictive_calibration",
    )


def plot_bn_cpt_sensitivity(result: Any, *, target_state: Any | None = None, ax: Any = None) -> Any:
    """Plot posterior response to a CPT perturbation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(result, "eyeprocess_class", None) != "eye_bn_cpt_sensitivity":
        raise EyeProcessValidationError("result must be an eye_bn_cpt_sensitivity.")
    table = result["table"]
    if target_state is not None:
        table = table.loc[table.target_state.astype(str).eq(str(target_state))]
    axis = plt.subplots()[1] if ax is None else ax
    for state_name, group in table.groupby("target_state", sort=False):
        axis.plot(group.requested_probability, group.target_probability, marker="o", label=str(state_name))
    axis.set_xlabel("Perturbed CPT probability")
    axis.set_ylabel("Target posterior probability")
    axis.set_title("Bayesian-network CPT sensitivity")
    axis.legend()
    axis.eyeprocess_plot_data = table.copy()
    return axis


def plot_bn_sample_size_stability(table: Any, ax: Any = None) -> Any:
    """Plot sample-fraction structure stability with branch-level points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    frame = pd.DataFrame(table).copy()
    required = {"fraction", "skeleton_jaccard"}
    if not required.issubset(frame):
        raise EyeProcessValidationError("table must contain fraction and skeleton_jaccard.")
    axis = plt.subplots()[1] if ax is None else ax
    good = frame if "status" not in frame else frame.loc[frame["status"].eq("success")]
    axis.scatter(good.fraction, good.skeleton_jaccard)
    means = good.groupby("fraction", observed=True).skeleton_jaccard.mean()
    axis.plot(means.index, means.values)
    axis.set_xlabel("Sample fraction")
    axis.set_ylabel("Skeleton Jaccard vs full data")
    axis.set_title("Bayesian-network sample-size stability")
    axis.eyeprocess_plot_data = frame.copy()
    return axis


def plot_bn_predictive_calibration(result: Any, ax: Any = None) -> Any:
    """Plot reliability curve for BN predictive probabilities."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(result, "eyeprocess_class", None) != "eye_bn_predictive_calibration":
        raise EyeProcessValidationError("result must be an eye_bn_predictive_calibration.")
    table = result["calibration"]
    axis = plt.subplots()[1] if ax is None else ax
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.plot(table.mean_predicted, table.observed_rate, marker="o")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Bayesian-network predictive calibration")
    axis.eyeprocess_plot_data = table.copy()
    return axis


__all__ = [
    "cpt_sensitivity_analysis",
    "discretization_sensitivity",
    "measurement_noise_sensitivity",
    "plot_bn_cpt_sensitivity",
    "plot_bn_predictive_calibration",
    "plot_bn_sample_size_stability",
    "predictive_calibration",
    "sample_size_stability_curve",
    "structural_perturbation_sensitivity",
]
