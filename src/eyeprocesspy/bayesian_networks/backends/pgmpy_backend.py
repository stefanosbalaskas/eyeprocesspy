"""Optional pgmpy adapter for static Bayesian-network analysis."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from ..schema import BayesianConstraintSpec, ModelFamily

_INSTALL = (
    "Bayesian-network functionality requires the optional 'bayesnet' dependency. "
    "Install it with `pip install eyeprocesspy[bayesnet]`."
)


def _import(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc


def _score_name(score: str | None, family: ModelFamily) -> str | None:
    if score is None:
        return None
    aliases = {"bic", "aic", "ll"}
    if score not in aliases:
        return score
    suffix = {"discrete": "d", "gaussian": "g", "mixed": "cg"}[family]
    return f"{score}-{suffix}"


def _expert_knowledge(constraints: BayesianConstraintSpec | None) -> Any:
    if constraints is None:
        return None
    cd = _import("pgmpy.causal_discovery")
    return cd.ExpertKnowledge(
        required_edges=list(constraints.required_edges),
        forbidden_edges=list(constraints.forbidden_edges),
        temporal_order=[list(tier) for tier in constraints.temporal_tiers],
    )


def learn_structure(
    data: pd.DataFrame,
    *,
    family: ModelFamily,
    algorithm: str,
    score: str | None,
    constraints: BayesianConstraintSpec | None,
    max_parents: int | None,
) -> tuple[Any, tuple[tuple[str, str], ...], str | None]:
    """Learn a static graph using pgmpy's causal-discovery interface."""
    cd = _import("pgmpy.causal_discovery")
    algorithm = algorithm.lower()
    score_name = _score_name(score, family)
    expert = _expert_knowledge(constraints)
    max_indegree = (
        max_parents
        if max_parents is not None
        else (constraints.max_parents if constraints else None)
    )

    if algorithm in {"hill_climb", "hc"}:
        est = cd.HillClimbSearch(
            scoring_method=score_name,
            max_indegree=max_indegree,
            expert_knowledge=expert,
            return_type="dag",
            show_progress=False,
        ).fit(data)
        used = score_name
    elif algorithm == "pc":
        if family == "mixed":
            raise NotImplementedError(
                "PC is not exposed for mixed data in the first eyeprocesspy BN backend."
            )
        ci_test = "chi_square" if family == "discrete" else "pearsonr"
        est = cd.PC(
            ci_test=ci_test,
            return_type="dag",
            expert_knowledge=expert,
            enforce_expert_knowledge=expert is not None,
            variant="stable",
            n_jobs=1,
            show_progress=False,
        ).fit(data)
        used = None
    elif algorithm == "ges":
        est = cd.GES(
            scoring_method=score_name,
            return_type="dag",
            expert_knowledge=expert,
            show_progress=False,
        ).fit(data)
        used = score_name
    else:
        raise ValueError("algorithm must be one of: 'hill_climb', 'pc', or 'ges'.")

    graph = est.causal_graph_
    edges = tuple(sorted((str(a), str(b)) for a, b in graph.edges()))
    return graph, edges, used


def fit_parameters(
    *,
    edges: tuple[tuple[str, str], ...],
    nodes: list[str],
    data: pd.DataFrame,
    family: ModelFamily,
    estimator: str,
    prior: str,
    equivalent_sample_size: float,
) -> Any:
    """Fit parameters for a fixed static DAG."""
    models = _import("pgmpy.models")
    estimator = estimator.lower()
    if family == "mixed":
        raise NotImplementedError(
            "Mixed conditional-Gaussian structure learning is supported, but pgmpy parameter "
            "fitting for a mixed BN is not exposed by this first eyeprocesspy backend. "
            "No discretization is performed."
        )
    if family == "discrete":
        estimators = _import("pgmpy.estimators")
        model = models.DiscreteBayesianNetwork(list(edges))
        model.add_nodes_from(nodes)
        if estimator in {"mle", "maximum_likelihood"}:
            model.fit(data[nodes], estimator=estimators.MaximumLikelihoodEstimator)
        elif estimator in {"bayesian", "bayes"}:
            normalized_prior = {
                "bdeu": "BDeu",
                "k2": "K2",
                "dirichlet": "dirichlet",
            }.get(prior.lower(), prior)
            bayes = estimators.BayesianEstimator(model, data[nodes])
            cpds = bayes.get_parameters(
                prior_type=normalized_prior,
                equivalent_sample_size=equivalent_sample_size,
                n_jobs=1,
            )
            model.add_cpds(*cpds)
        else:
            raise ValueError("Discrete estimator must be 'mle' or 'bayesian'.")
        model.check_model()
        return model

    if estimator not in {"mle", "maximum_likelihood"}:
        raise ValueError("Gaussian Bayesian networks currently support estimator='mle'.")
    model = models.LinearGaussianBayesianNetwork(list(edges))
    model.add_nodes_from(nodes)
    model.fit(data[nodes], estimator="mle", std_estimator="unbiased")
    model.check_model()
    return model


def query(
    model: Any,
    *,
    family: ModelFamily,
    target: str,
    evidence: dict[str, Any],
) -> tuple[dict[str, float] | None, float | None, float | None, str]:
    """Return a discrete posterior or Gaussian conditional mean/variance."""
    if target in evidence:
        raise ValueError("target must not also be supplied as evidence.")
    if target not in model.nodes():
        raise ValueError(f"Unknown target node: {target!r}.")
    unknown = sorted(set(evidence).difference(model.nodes()))
    if unknown:
        raise ValueError(f"Evidence contains unknown nodes: {unknown}.")

    if family == "discrete":
        inference = _import("pgmpy.inference")
        factor = inference.VariableElimination(model).query(
            variables=[target], evidence=evidence, show_progress=False
        )
        states = factor.state_names[target]
        values = np.asarray(factor.values, dtype=float).reshape(-1)
        posterior = {str(state): float(value) for state, value in zip(states, values, strict=True)}
        return posterior, None, None, "variable_elimination"

    if family == "mixed":
        raise NotImplementedError(
            "Posterior inference for mixed BNs is not exposed by this backend."
        )

    nx = _import("networkx")
    ordering = list(nx.topological_sort(model))
    mean, cov = model.to_joint_gaussian()
    index = {name: i for i, name in enumerate(ordering)}
    target_i = index[target]
    if not evidence:
        return (
            None,
            float(mean[target_i]),
            float(cov[target_i, target_i]),
            "gaussian_conditioning",
        )
    ev_names = list(evidence)
    ev_idx = [index[name] for name in ev_names]
    ev_values = np.asarray([float(evidence[name]) for name in ev_names], dtype=float)
    mu_e = mean[ev_idx]
    cov_ee = cov[np.ix_(ev_idx, ev_idx)]
    cov_te = cov[target_i, ev_idx]
    adjustment = cov_te @ np.linalg.pinv(cov_ee)
    cond_mean = mean[target_i] + adjustment @ (ev_values - mu_e)
    cond_var = cov[target_i, target_i] - adjustment @ cov[np.ix_(ev_idx, [target_i])].reshape(-1)
    return (
        None,
        float(cond_mean),
        float(max(cond_var, 0.0)),
        "gaussian_conditioning",
    )


def log_likelihood(
    model: Any,
    *,
    family: ModelFamily,
    data: pd.DataFrame,
    nodes: list[str],
) -> float:
    """Compute held-out log likelihood for a fitted model."""
    if family == "discrete":
        metrics = _import("pgmpy.metrics")
        return float(metrics.log_likelihood_score(model, data[nodes]))
    if family == "gaussian":
        return float(model.log_likelihood(data[nodes]))
    raise NotImplementedError("Held-out likelihood for mixed BNs is not exposed by this backend.")
