"""Theory-aware static Bayesian-network structure learning."""

from __future__ import annotations

import platform
import warnings
from importlib import metadata
from typing import Any

from .backends import pgmpy_backend
from .prepare import validate_bayesian_network_data
from .schema import BayesianConstraintSpec, BayesianDataSpec, BayesianNetworkResult


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _provenance(**extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "eyeprocesspy": _version("eyeprocesspy"),
        "pgmpy": _version("pgmpy"),
    }
    out.update(extra)
    return out


def learn_bayesian_network(
    data_spec: BayesianDataSpec,
    *,
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    max_parents: int | None = None,
    random_state: int | None = 42,
    backend: str = "pgmpy",
    allow_unconstrained: bool = False,
) -> BayesianNetworkResult:
    """Learn a static graph while preserving scientific ordering constraints."""
    validation = validate_bayesian_network_data(data_spec)
    if not validation.valid:
        raise ValueError("Invalid Bayesian-network data: " + "; ".join(validation.errors))
    if backend != "pgmpy":
        raise ValueError("The first eyeprocesspy BN tranche supports backend='pgmpy' only.")
    if max_parents is not None and max_parents < 1:
        raise ValueError("max_parents must be at least 1 when supplied.")

    messages = list(validation.warnings)
    has_experimental = any(
        spec.modality == "experimental" and bool(spec.include_in_structure)
        for spec in data_spec.node_specs.values()
    )
    if constraints is None and has_experimental and not allow_unconstrained:
        message = (
            "Unconstrained structure learning includes an experimental node. Encode temporal tiers "
            "or forbidden edges so the search cannot reverse known experimental ordering."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        messages.append(message)

    nodes = data_spec.structure_nodes
    graph, edges, used_score = pgmpy_backend.learn_structure(
        data_spec.data[nodes],
        family=data_spec.model_family,
        algorithm=algorithm,
        score=score,
        constraints=constraints,
        max_parents=max_parents,
    )
    return BayesianNetworkResult(
        edges=edges,
        nodes=tuple(nodes),
        model_family=data_spec.model_family,
        backend=backend,
        backend_model=graph,
        data_spec=data_spec,
        constraints=constraints,
        structure_algorithm=algorithm,
        score=used_score,
        fitted=False,
        provenance=_provenance(
            random_state=random_state,
            observation_level=data_spec.observation_level,
        ),
        warnings=tuple(messages),
    )
