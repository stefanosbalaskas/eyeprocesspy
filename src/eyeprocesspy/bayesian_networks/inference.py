"""Posterior querying and prediction from fitted Bayesian networks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .backends import pgmpy_backend
from .schema import BayesianNetworkQueryResult, BayesianNetworkResult


def query_bayesian_network(
    result: BayesianNetworkResult,
    *,
    target: str,
    evidence: Mapping[str, Any] | None = None,
) -> BayesianNetworkQueryResult:
    """Query one target conditional on observed evidence."""
    if not isinstance(result, BayesianNetworkResult) or not result.fitted:
        raise ValueError("query_bayesian_network requires a fitted BayesianNetworkResult.")
    evidence_dict = dict(evidence or {})
    posterior, mean, variance, method = pgmpy_backend.query(
        result.backend_model,
        family=result.model_family,
        target=target,
        evidence=evidence_dict,
    )
    return BayesianNetworkQueryResult(
        target=target,
        evidence=evidence_dict,
        posterior=posterior,
        mean=mean,
        variance=variance,
        method=method,
        backend=result.backend,
        provenance={"model": result.provenance},
    )


def predict_bayesian_network(
    result: BayesianNetworkResult,
    *,
    target: str,
    evidence: Mapping[str, Any] | pd.DataFrame,
) -> BayesianNetworkQueryResult | pd.DataFrame:
    """Predict one target for one evidence mapping or each row of a DataFrame."""
    if isinstance(evidence, Mapping):
        return query_bayesian_network(result, target=target, evidence=evidence)
    if not isinstance(evidence, pd.DataFrame):
        raise TypeError("evidence must be a mapping or pandas DataFrame.")
    rows: list[dict[str, Any]] = []
    for index, row in evidence.iterrows():
        observed = {
            name: value
            for name, value in row.items()
            if pd.notna(value) and name != target
        }
        q = query_bayesian_network(result, target=target, evidence=observed)
        if q.posterior is not None:
            for state, probability in q.posterior.items():
                rows.append(
                    {
                        "row": index,
                        "target": target,
                        "state": state,
                        "probability": probability,
                    }
                )
        else:
            rows.append(
                {
                    "row": index,
                    "target": target,
                    "mean": q.mean,
                    "variance": q.variance,
                }
            )
    return pd.DataFrame(rows)
