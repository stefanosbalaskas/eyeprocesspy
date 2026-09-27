"""Parameter estimation for fixed or learned static Bayesian networks."""

from __future__ import annotations

from collections.abc import Sequence

from .backends import pgmpy_backend
from .learning import _provenance
from .prepare import validate_bayesian_network_data
from .schema import BayesianConstraintSpec, BayesianDataSpec, BayesianNetworkResult, Edge


def _normalize_edges(edges: Sequence[Sequence[str]]) -> tuple[Edge, ...]:
    out: set[Edge] = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"Each edge must contain source and target; received {edge!r}.")
        source, target = str(edge[0]), str(edge[1])
        if source == target:
            raise ValueError(f"Self edges are not allowed: {(source, target)!r}.")
        out.add((source, target))
    return tuple(sorted(out))


def fit_bayesian_network(
    graph: BayesianNetworkResult | Sequence[Sequence[str]],
    *,
    data: BayesianDataSpec | None = None,
    estimator: str = "bayesian",
    prior: str = "BDeu",
    equivalent_sample_size: float = 10.0,
    constraints: BayesianConstraintSpec | None = None,
    backend: str = "pgmpy",
) -> BayesianNetworkResult:
    """Fit parameters for a learned graph or an explicit theoretical DAG."""
    if isinstance(graph, BayesianNetworkResult):
        data_spec = graph.data_spec if data is None else data
        edges = graph.edges
        constraints = graph.constraints if constraints is None else constraints
        structure_algorithm = graph.structure_algorithm
        score = graph.score
    else:
        if data is None:
            raise ValueError("data is required when fitting an explicit edge list.")
        data_spec = data
        edges = _normalize_edges(graph)
        structure_algorithm = "fixed_dag"
        score = None
    if backend != "pgmpy":
        raise ValueError("The first eyeprocesspy BN tranche supports backend='pgmpy' only.")
    if equivalent_sample_size <= 0:
        raise ValueError("equivalent_sample_size must be positive.")

    validation = validate_bayesian_network_data(data_spec)
    if not validation.valid:
        raise ValueError("Invalid Bayesian-network data: " + "; ".join(validation.errors))
    nodes = data_spec.structure_nodes
    unknown = sorted({n for edge in edges for n in edge}.difference(nodes))
    if unknown:
        raise ValueError(f"Graph references nodes excluded from structure data: {unknown}.")

    model = pgmpy_backend.fit_parameters(
        edges=edges,
        nodes=nodes,
        data=data_spec.data,
        family=data_spec.model_family,
        estimator=estimator,
        prior=prior,
        equivalent_sample_size=equivalent_sample_size,
    )
    return BayesianNetworkResult(
        edges=edges,
        nodes=tuple(nodes),
        model_family=data_spec.model_family,
        backend=backend,
        backend_model=model,
        data_spec=data_spec,
        constraints=constraints,
        structure_algorithm=structure_algorithm,
        score=score,
        parameter_estimator=estimator,
        fitted=True,
        provenance=_provenance(
            prior=prior,
            equivalent_sample_size=equivalent_sample_size,
            source_provenance=data_spec.provenance,
        ),
        warnings=validation.warnings,
    )
