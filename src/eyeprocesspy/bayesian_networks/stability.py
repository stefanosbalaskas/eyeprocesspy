"""Bootstrap stability for learned Bayesian-network structure."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .learning import learn_bayesian_network
from .schema import BayesianConstraintSpec, BayesianDataSpec, BayesianNetworkStabilityResult


def _resample_rows(
    data_spec: BayesianDataSpec,
    *,
    rng: np.random.Generator,
    resample_by: str | None,
) -> BayesianDataSpec:
    data = data_spec.data
    if resample_by is None:
        sampled = data.iloc[rng.integers(0, len(data), len(data))].reset_index(drop=True)
    else:
        if resample_by not in data:
            raise ValueError(f"resample_by column {resample_by!r} is absent.")
        groups = pd.Index(data[resample_by].dropna().unique())
        if groups.empty:
            raise ValueError(f"resample_by column {resample_by!r} has no observed groups.")
        sampled_groups = groups[rng.integers(0, len(groups), len(groups))]
        chunks: list[pd.DataFrame] = []
        for bootstrap_id, group in enumerate(sampled_groups):
            chunk = data.loc[data[resample_by].eq(group)].copy()
            chunk[resample_by] = f"bootstrap_{bootstrap_id}:{group}"
            chunks.append(chunk)
        sampled = pd.concat(chunks, ignore_index=True)
    return BayesianDataSpec(
        data=sampled,
        node_specs=data_spec.node_specs,
        participant_id=data_spec.participant_id,
        trial_id=None if resample_by == data_spec.participant_id else data_spec.trial_id,
        stimulus_id=data_spec.stimulus_id,
        observation_level=data_spec.observation_level,
        provenance={**data_spec.provenance, "bootstrap": True},
        source_columns=data_spec.source_columns,
    )


def bootstrap_bn_structure(
    data_spec: BayesianDataSpec,
    *,
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    n_boot: int = 1000,
    resample_by: str | None = None,
    max_parents: int | None = None,
    random_state: int | None = 42,
    backend: str = "pgmpy",
) -> BayesianNetworkStabilityResult:
    """Estimate edge presence and direction stability across bootstrap samples."""
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if resample_by is None and data_spec.participant_id is not None:
        repeated = data_spec.data[data_spec.participant_id].duplicated().any()
        if repeated:
            resample_by = data_spec.participant_id

    rng = np.random.default_rng(random_state)
    directed: Counter[tuple[str, str]] = Counter()
    undirected: Counter[tuple[str, str]] = Counter()
    success = 0
    failures = 0
    for _ in range(n_boot):
        sampled = _resample_rows(data_spec, rng=rng, resample_by=resample_by)
        try:
            result = learn_bayesian_network(
                sampled,
                algorithm=algorithm,
                score=score,
                constraints=constraints,
                max_parents=max_parents,
                random_state=random_state,
                backend=backend,
                allow_unconstrained=True,
            )
        except (ValueError, RuntimeError, ImportError):
            failures += 1
            continue
        success += 1
        for source, target in set(result.edges):
            directed[(source, target)] += 1
            undirected[tuple(sorted((source, target)))] += 1

    rows: list[dict[str, object]] = []
    for a, b in sorted(undirected):
        present = undirected[(a, b)]
        a_to_b = directed[(a, b)]
        b_to_a = directed[(b, a)]
        rows.append(
            {
                "node_a": a,
                "node_b": b,
                "edge": f"{a}--{b}",
                "edge_strength": present / n_boot,
                "direction": f"{a}->{b}" if a_to_b >= b_to_a else f"{b}->{a}",
                "direction_strength": max(a_to_b, b_to_a) / present,
                "present_count": present,
                "successful_fits": success,
            }
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "node_a",
            "node_b",
            "edge",
            "edge_strength",
            "direction",
            "direction_strength",
            "present_count",
            "successful_fits",
        ],
    )
    return BayesianNetworkStabilityResult(
        edge_table=table,
        n_boot=n_boot,
        resample_by=resample_by,
        successful_fits=success,
        failed_fits=failures,
        provenance={"random_state": random_state, "algorithm": algorithm, "score": score},
    )
