"""Publication-oriented plots for Bayesian-network structure and diagnostics."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    BayesianNetworkComparisonResult,
    BayesianNetworkQueryResult,
    BayesianNetworkResult,
    BayesianNetworkStabilityResult,
    BayesianNetworkValidationResult,
)


def _plt() -> Any:
    try:
        return import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "Bayesian-network plotting requires matplotlib. Install eyeprocesspy[plots]."
        ) from exc


def plot_bayesian_network(
    result: BayesianNetworkResult,
    *,
    edge_strength: pd.DataFrame | None = None,
    ax: Any = None,
    seed: int = 42,
) -> Any:
    """Plot the learned/fixed DAG, optionally scaling edges by bootstrap strength."""
    plt = _plt()
    try:
        nx = import_module("networkx")
    except ImportError as exc:
        raise ImportError(
            "Bayesian-network graph plotting requires networkx "
            "(installed with eyeprocesspy[bayesnet])."
        ) from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    graph = nx.DiGraph()
    graph.add_nodes_from(result.nodes)
    graph.add_edges_from(result.edges)
    pos = nx.spring_layout(graph, seed=seed)
    widths = [1.8] * len(result.edges)
    if edge_strength is not None and not edge_strength.empty:
        strength_map = {
            tuple(sorted((str(row.node_a), str(row.node_b)))): float(row.edge_strength)
            for row in edge_strength.itertuples()
        }
        widths = [
            0.8 + 3.2 * strength_map.get(tuple(sorted(edge)), 0.5)
            for edge in result.edges
        ]
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=1800)
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        width=widths,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.03",
    )
    ax.set_title("Bayesian-network structure")
    ax.axis("off")
    ax.eyeprocess_plot_data = pd.DataFrame(
        result.edges,
        columns=["source", "target"],
    )
    return ax


def plot_bn_edge_stability(
    stability: BayesianNetworkStabilityResult,
    *,
    ax: Any = None,
) -> Any:
    """Plot bootstrap edge-presence strength."""
    plt = _plt()
    table = stability.edge_table.sort_values("edge_strength", ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.35 * max(len(table), 1))))
    ax.barh(table["edge"], table["edge_strength"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Bootstrap edge strength")
    ax.set_title("Bayesian-network edge stability")
    ax.eyeprocess_plot_data = table.copy()
    return ax


def plot_bn_direction_stability(
    stability: BayesianNetworkStabilityResult,
    *,
    ax: Any = None,
) -> Any:
    """Plot conditional direction stability for observed edges."""
    plt = _plt()
    table = stability.edge_table.sort_values("direction_strength", ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.35 * max(len(table), 1))))
    ax.barh(table["direction"], table["direction_strength"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Direction strength conditional on edge presence")
    ax.set_title("Bayesian-network direction stability")
    ax.eyeprocess_plot_data = table.copy()
    return ax


def plot_bn_posterior(
    query: BayesianNetworkQueryResult,
    *,
    ax: Any = None,
) -> Any:
    """Plot a discrete posterior or Gaussian posterior summary."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if query.posterior is not None:
        table = pd.DataFrame(
            {
                "state": list(query.posterior),
                "probability": list(query.posterior.values()),
            }
        )
        ax.bar(table["state"], table["probability"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Posterior probability")
    else:
        if query.mean is None or query.variance is None:
            raise ValueError("Gaussian posterior requires mean and variance.")
        sd = float(np.sqrt(max(query.variance, 0.0)))
        table = pd.DataFrame(
            {"target": [query.target], "mean": [query.mean], "sd": [sd]}
        )
        ax.errorbar([query.target], [query.mean], yerr=[sd], fmt="o", capsize=5)
        ax.set_ylabel("Conditional mean ± 1 SD")
    ax.set_title(f"Posterior: {query.target}")
    ax.eyeprocess_plot_data = table
    return ax


def plot_bn_detector_robustness(
    comparison: BayesianNetworkComparisonResult,
    *,
    ax: Any = None,
) -> Any:
    """Plot edge robustness across detector/AOI specifications."""
    plt = _plt()
    table = comparison.edge_table.sort_values("edge_robustness", ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.35 * max(len(table), 1))))
    ax.barh(table["edge"], table["edge_robustness"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of specifications containing edge")
    ax.set_title("Bayesian-network specification robustness")
    ax.eyeprocess_plot_data = table.copy()
    return ax


def plot_bn_validation(
    validation: BayesianNetworkValidationResult,
    *,
    ax: Any = None,
) -> Any:
    """Plot participant-grouped held-out mean log likelihood by fold."""
    plt = _plt()
    table = validation.fold_table.copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["fold"], table["mean_log_likelihood"], marker="o")
    ax.set_xlabel("Held-out fold")
    ax.set_ylabel("Mean held-out log likelihood")
    ax.set_title("Participant-grouped BN validation")
    ax.eyeprocess_plot_data = table
    return ax
