"""Typed contracts for downstream Bayesian-network analysis.

Scientific contract
-------------------
Bayesian-network objects consume already-qualified analysis features. They do
not silently preprocess gaze, redefine AOIs, impute missing values, discretize
continuous variables, exclude trials, or upgrade a probabilistic edge to a
causal claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

VariableType = Literal["categorical", "ordinal", "count", "continuous"]
NodeRole = Literal["substantive", "quality", "id", "provenance"]
ModelFamily = Literal["discrete", "gaussian", "mixed"]
Edge = tuple[str, str]


@dataclass(frozen=True)
class BayesianNodeSpec:
    """Scientific metadata for one Bayesian-network node."""

    name: str
    variable_type: VariableType
    modality: str = "unspecified"
    level: str = "trial"
    unit: str | None = None
    value_range: tuple[float, float] | None = None
    role: NodeRole = "substantive"
    include_in_structure: bool | None = None
    states: tuple[str, ...] | None = None
    missing_allowed: bool = True
    zero_is_missing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BayesianNodeSpec.name must be non-empty.")
        if self.variable_type not in {"categorical", "ordinal", "count", "continuous"}:
            raise ValueError(f"Unsupported variable_type: {self.variable_type!r}.")
        if self.role not in {"substantive", "quality", "id", "provenance"}:
            raise ValueError(f"Unsupported node role: {self.role!r}.")
        if self.value_range is not None and self.value_range[0] > self.value_range[1]:
            raise ValueError("value_range lower bound must not exceed upper bound.")
        if self.states is not None and len(set(self.states)) != len(self.states):
            raise ValueError("states must not contain duplicates.")
        if self.include_in_structure is None:
            object.__setattr__(self, "include_in_structure", self.role == "substantive")


@dataclass
class BayesianDataSpec:
    """Prepared observational table plus node and provenance contracts."""

    data: pd.DataFrame
    node_specs: dict[str, BayesianNodeSpec]
    participant_id: str | None = None
    trial_id: str | None = None
    stimulus_id: str | None = None
    observation_level: str = "trial"
    provenance: dict[str, Any] = field(default_factory=dict)
    source_columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structure_nodes(self) -> list[str]:
        return [name for name, spec in self.node_specs.items() if bool(spec.include_in_structure)]

    @property
    def model_family(self) -> ModelFamily:
        kinds = {self.node_specs[name].variable_type for name in self.structure_nodes}
        discrete_kinds = {"categorical", "ordinal", "count"}
        if kinds <= discrete_kinds:
            return "discrete"
        if kinds <= {"continuous"}:
            return "gaussian"
        return "mixed"


@dataclass(frozen=True)
class BayesianConstraintSpec:
    """Structure-learning constraints and temporal ordering."""

    temporal_tiers: tuple[tuple[str, ...], ...] = ()
    required_edges: tuple[Edge, ...] = ()
    forbidden_edges: tuple[Edge, ...] = ()
    max_parents: int | None = None


@dataclass(frozen=True)
class BayesianDataValidationResult:
    """Validation result that preserves warnings instead of silently repairing data."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    n_rows: int = 0
    n_nodes: int = 0
    model_family: ModelFamily | None = None


@dataclass
class BayesianNetworkResult:
    """Backend-neutral learned or fitted Bayesian-network result."""

    edges: tuple[Edge, ...]
    nodes: tuple[str, ...]
    model_family: ModelFamily
    backend: str
    backend_model: Any
    data_spec: BayesianDataSpec
    constraints: BayesianConstraintSpec | None = None
    structure_algorithm: str | None = None
    score: str | None = None
    parameter_estimator: str | None = None
    fitted: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BayesianNetworkQueryResult:
    """Posterior query result with explicit evidence and provenance."""

    target: str
    evidence: dict[str, Any]
    posterior: dict[str, float] | None
    mean: float | None = None
    variance: float | None = None
    method: str = "variable_elimination"
    backend: str = "pgmpy"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianNetworkStabilityResult:
    """Bootstrap edge and direction stability."""

    edge_table: pd.DataFrame
    n_boot: int
    resample_by: str | None
    successful_fits: int
    failed_fits: int
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianNetworkValidationResult:
    """Participant-aware validation evidence."""

    fold_table: pd.DataFrame
    method: str
    group_column: str | None
    summary: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianNetworkComparisonResult:
    """Cross-network edge and optional posterior robustness summary."""

    edge_table: pd.DataFrame
    posterior_table: pd.DataFrame | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
