"""Cross-specification robustness for Bayesian-network results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .inference import query_bayesian_network
from .learning import learn_bayesian_network
from .schema import (
    BayesianConstraintSpec,
    BayesianDataSpec,
    BayesianNetworkComparisonResult,
    BayesianNetworkResult,
)


def compare_bayesian_networks(
    results: Mapping[str, BayesianNetworkResult],
    *,
    target: str | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> BayesianNetworkComparisonResult:
    """Summarize edge robustness and optional posterior sensitivity across networks."""
    if not results:
        raise ValueError("results must contain at least one named BayesianNetworkResult.")
    names = list(results)
    all_skeletons = {
        name: {tuple(sorted(edge)) for edge in result.edges}
        for name, result in results.items()
    }
    edge_union = sorted(set().union(*all_skeletons.values()))
    edge_rows: list[dict[str, Any]] = []
    for a, b in edge_union:
        row: dict[str, Any] = {"edge": f"{a}--{b}"}
        present = 0
        for name in names:
            result = results[name]
            if (a, b) in result.edges:
                direction = f"{a}->{b}"
            elif (b, a) in result.edges:
                direction = f"{b}->{a}"
            else:
                direction = "absent"
            row[name] = direction
            present += direction != "absent"
        row["edge_robustness"] = present / len(names)
        edge_rows.append(row)
    edge_table = pd.DataFrame(
        edge_rows,
        columns=["edge", *names, "edge_robustness"],
    )

    posterior_table: pd.DataFrame | None = None
    if target is not None:
        posterior_rows: list[dict[str, Any]] = []
        for name, result in results.items():
            query = query_bayesian_network(
                result,
                target=target,
                evidence=evidence or {},
            )
            if query.posterior is not None:
                for state, probability in query.posterior.items():
                    posterior_rows.append(
                        {
                            "specification": name,
                            "target": target,
                            "state": state,
                            "probability": probability,
                        }
                    )
            else:
                posterior_rows.append(
                    {
                        "specification": name,
                        "target": target,
                        "mean": query.mean,
                        "variance": query.variance,
                    }
                )
        posterior_table = pd.DataFrame(posterior_rows)
    return BayesianNetworkComparisonResult(
        edge_table=edge_table,
        posterior_table=posterior_table,
        provenance={
            "n_specifications": len(names),
            "evidence": dict(evidence or {}),
        },
    )


def compare_bn_across_detectors(
    analyses: Mapping[str, BayesianDataSpec],
    *,
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    max_parents: int | None = None,
    random_state: int | None = 42,
    backend: str = "pgmpy",
) -> BayesianNetworkComparisonResult:
    """Learn the same constrained BN across detector-derived feature tables."""
    if not analyses:
        raise ValueError(
            "analyses must contain at least one detector-labelled BayesianDataSpec."
        )
    learned = {
        name: learn_bayesian_network(
            data_spec,
            algorithm=algorithm,
            score=score,
            constraints=constraints,
            max_parents=max_parents,
            random_state=random_state,
            backend=backend,
            allow_unconstrained=True,
        )
        for name, data_spec in analyses.items()
    }
    comparison = compare_bayesian_networks(learned)
    comparison.provenance.update(
        {
            "sensitivity_axis": "event_detector",
            "algorithm": algorithm,
            "score": score,
        }
    )
    return comparison


def compare_bn_across_aoi_specs(
    analyses: Mapping[str, BayesianDataSpec],
    **kwargs: Any,
) -> BayesianNetworkComparisonResult:
    """Learn the same BN across AOI-derived feature tables."""
    comparison = compare_bn_across_detectors(analyses, **kwargs)
    comparison.provenance["sensitivity_axis"] = "aoi_specification"
    return comparison
