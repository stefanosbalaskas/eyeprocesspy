"""Scientific structure constraints for Bayesian-network learning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .schema import BayesianConstraintSpec, Edge


def _edge_tuple(edge: Sequence[str]) -> Edge:
    if len(edge) != 2:
        raise ValueError(f"Edges must contain exactly two node names; received {edge!r}.")
    source, target = str(edge[0]), str(edge[1])
    if not source or not target or source == target:
        raise ValueError(f"Invalid directed edge: {(source, target)!r}.")
    return source, target


def define_temporal_tiers(
    tiers: Sequence[Sequence[str]],
    *,
    nodes: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Validate a partial temporal order used to forbid backwards edges."""
    if not tiers:
        return ()
    normalized = tuple(tuple(str(node) for node in tier) for tier in tiers)
    if any(not tier for tier in normalized):
        raise ValueError("Temporal tiers must not contain empty tiers.")
    flat = [node for tier in normalized for node in tier]
    if len(flat) != len(set(flat)):
        raise ValueError("A node may appear in only one temporal tier.")
    if nodes is not None:
        allowed = set(nodes)
        unknown = sorted(set(flat).difference(allowed))
        if unknown:
            raise ValueError(f"Temporal tiers contain unknown nodes: {unknown}.")
    return normalized


def _has_cycle(edges: Sequence[Edge]) -> bool:
    graph: dict[str, list[str]] = {}
    for source, target in edges:
        graph.setdefault(source, []).append(target)
        graph.setdefault(target, [])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for child in graph.get(node, []):
            if visit(child):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _temporal_forbidden(tiers: tuple[tuple[str, ...], ...]) -> set[Edge]:
    out: set[Edge] = set()
    for earlier_index, earlier in enumerate(tiers):
        later_nodes = [node for tier in tiers[earlier_index + 1 :] for node in tier]
        for late in later_nodes:
            for early in earlier:
                out.add((late, early))
    return out


def define_bn_constraints(
    *,
    temporal_tiers: Sequence[Sequence[str]] | tuple[tuple[str, ...], ...] = (),
    required_edges: Sequence[Sequence[str]] = (),
    forbidden_edges: Sequence[Sequence[str]] = (),
    max_parents: int | None = None,
    nodes: Iterable[str] | None = None,
) -> BayesianConstraintSpec:
    """Create a validated constraint set for theory-aware structure learning."""
    node_set = set(nodes) if nodes is not None else None
    tiers = define_temporal_tiers(temporal_tiers, nodes=node_set)
    required = {_edge_tuple(edge) for edge in required_edges}
    forbidden = {_edge_tuple(edge) for edge in forbidden_edges}
    forbidden.update(_temporal_forbidden(tiers))

    if max_parents is not None and max_parents < 1:
        raise ValueError("max_parents must be at least 1 when supplied.")
    all_edge_nodes = {node for edge in required | forbidden for node in edge}
    if node_set is not None:
        unknown = sorted(all_edge_nodes.difference(node_set))
        if unknown:
            raise ValueError(f"Constraints reference unknown nodes: {unknown}.")
    conflict = sorted(required.intersection(forbidden))
    if conflict:
        raise ValueError(f"Edges cannot be both required and forbidden: {conflict}.")
    if _has_cycle(tuple(required)):
        raise ValueError("required_edges contains a directed cycle.")

    return BayesianConstraintSpec(
        temporal_tiers=tiers,
        required_edges=tuple(sorted(required)),
        forbidden_edges=tuple(sorted(forbidden)),
        max_parents=max_parents,
    )
