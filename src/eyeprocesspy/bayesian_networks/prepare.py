"""Preparation and validation for Bayesian-network feature tables."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    BayesianDataSpec,
    BayesianDataValidationResult,
    BayesianNodeSpec,
    VariableType,
)


def _infer_variable_type(series: pd.Series) -> VariableType:
    if pd.api.types.is_bool_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "continuous"
    return "categorical"


def define_bn_nodes(
    specifications: Mapping[str, BayesianNodeSpec | Mapping[str, Any]],
) -> dict[str, BayesianNodeSpec]:
    """Create validated node specifications from mappings or existing specs."""
    if not isinstance(specifications, Mapping) or not specifications:
        raise ValueError("specifications must be a non-empty mapping.")
    out: dict[str, BayesianNodeSpec] = {}
    for name, value in specifications.items():
        if isinstance(value, BayesianNodeSpec):
            if value.name != name:
                raise ValueError(f"Node key {name!r} does not match spec name {value.name!r}.")
            out[name] = value
            continue
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Node specification for {name!r} must be a mapping or BayesianNodeSpec."
            )
        payload = dict(value)
        payload.setdefault("name", name)
        if "type" in payload and "variable_type" not in payload:
            payload["variable_type"] = payload.pop("type")
        if "range" in payload and "value_range" not in payload:
            payload["value_range"] = tuple(payload.pop("range"))
        if "states" in payload and payload["states"] is not None:
            payload["states"] = tuple(str(x) for x in payload["states"])
        out[name] = BayesianNodeSpec(**payload)
    return out


def prepare_bayesian_network_data(
    data: pd.DataFrame,
    *,
    nodes: Sequence[str] | Mapping[str, BayesianNodeSpec | Mapping[str, Any]],
    node_specs: Mapping[str, BayesianNodeSpec | Mapping[str, Any]] | None = None,
    participant_id: str | None = "participant_id",
    trial_id: str | None = "trial_id",
    stimulus_id: str | None = "stimulus_id",
    observation_level: str = "trial",
    provenance: Mapping[str, Any] | None = None,
    provenance_columns: Sequence[str] | None = None,
) -> BayesianDataSpec:
    """Prepare an analysis table without silently transforming scientific values.

    Missing values are retained. Continuous variables are not discretized, and
    rows are not dropped. When explicit node metadata is absent, variable types
    are inferred from pandas dtypes and a warning is emitted.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if observation_level not in {"participant", "trial", "bin"}:
        raise ValueError("observation_level must be 'participant', 'trial', or 'bin'.")

    if isinstance(nodes, Mapping):
        specs = define_bn_nodes(nodes)
        node_names = list(specs)
        if node_specs is not None:
            raise ValueError("Supply node metadata through either nodes or node_specs, not both.")
    else:
        node_names = list(nodes)
        if not node_names or len(set(node_names)) != len(node_names):
            raise ValueError("nodes must be a non-empty sequence without duplicates.")
        if node_specs is None:
            missing = [name for name in node_names if name not in data]
            if missing:
                raise ValueError(f"data is missing requested nodes: {missing}.")
            warnings.warn(
                "Node types were inferred from pandas dtypes. For scientific analyses, "
                "define_bn_nodes() is recommended so modality, units, roles, and ranges are explicit.",
                UserWarning,
                stacklevel=2,
            )
            specs = {
                name: BayesianNodeSpec(name=name, variable_type=_infer_variable_type(data[name]))
                for name in node_names
            }
        else:
            all_specs = define_bn_nodes(node_specs)
            missing_specs = [name for name in node_names if name not in all_specs]
            if missing_specs:
                raise ValueError(f"node_specs is missing requested nodes: {missing_specs}.")
            specs = {name: all_specs[name] for name in node_names}

    missing_nodes = [name for name in node_names if name not in data]
    if missing_nodes:
        raise ValueError(f"data is missing requested nodes: {missing_nodes}.")

    id_columns = [
        x
        for x in (participant_id, trial_id, stimulus_id)
        if x is not None and x in data
    ]
    provenance_columns = tuple(provenance_columns or ())
    absent_prov = [name for name in provenance_columns if name not in data]
    if absent_prov:
        raise ValueError(f"data is missing provenance columns: {absent_prov}.")

    keep: list[str] = []
    for name in [*id_columns, *node_names, *provenance_columns]:
        if name not in keep:
            keep.append(name)
    prepared = data.loc[:, keep].copy()

    if observation_level == "trial" and participant_id and trial_id:
        if participant_id not in prepared or trial_id not in prepared:
            raise ValueError(
                "Trial-level Bayesian data requires participant_id and trial_id columns when "
                "those identifiers are declared."
            )
        if prepared.duplicated([participant_id, trial_id]).any():
            example = prepared.loc[
                prepared.duplicated([participant_id, trial_id], keep=False),
                [participant_id, trial_id],
            ].head(5)
            raise ValueError(
                "Trial-level Bayesian data must contain one row per participant/trial; "
                f"duplicates include {example.to_dict('records')}."
            )

    return BayesianDataSpec(
        data=prepared,
        node_specs=specs,
        participant_id=participant_id if participant_id in prepared else None,
        trial_id=trial_id if trial_id in prepared else None,
        stimulus_id=stimulus_id if stimulus_id in prepared else None,
        observation_level=observation_level,
        provenance=dict(provenance or {}),
        source_columns=tuple(data.columns),
    )


def validate_bayesian_network_data(
    data_spec: BayesianDataSpec,
    *,
    strict: bool = False,
) -> BayesianDataValidationResult:
    """Validate node types, ranges, states, missingness, and structural roles."""
    if not isinstance(data_spec, BayesianDataSpec):
        raise TypeError("data_spec must be a BayesianDataSpec.")
    errors: list[str] = []
    notes: list[str] = []
    data = data_spec.data

    if not data_spec.structure_nodes:
        errors.append("No nodes are eligible for structure learning.")

    for name, spec in data_spec.node_specs.items():
        if name not in data:
            errors.append(f"Node {name!r} is absent from the prepared data.")
            continue
        series = data[name]
        missing_count = int(series.isna().sum())
        if missing_count and not spec.missing_allowed:
            errors.append(
                f"Node {name!r} contains {missing_count} missing values "
                "but missing_allowed=False."
            )
        if (
            spec.zero_is_missing
            and pd.api.types.is_numeric_dtype(series)
            and bool(series.eq(0).any())
        ):
            notes.append(
                f"Node {name!r} declares zero_is_missing=True, but zeros are preserved; "
                "convert them explicitly upstream if they truly represent missingness."
            )
        if spec.variable_type in {"continuous", "count"}:
            converted = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & converted.isna()
            if bool(invalid.any()):
                errors.append(
                    f"Node {name!r} contains non-numeric values incompatible with "
                    f"{spec.variable_type}."
                )
            if spec.variable_type == "count" and bool((converted.dropna() < 0).any()):
                errors.append(f"Count node {name!r} contains negative values.")
            if spec.value_range is not None:
                low, high = spec.value_range
                bad = converted.notna() & ((converted < low) | (converted > high))
                if bool(bad.any()):
                    errors.append(
                        f"Node {name!r} contains values outside declared range [{low}, {high}]."
                    )
        if spec.states is not None:
            observed = {str(x) for x in series.dropna().unique()}
            unexpected = sorted(observed.difference(spec.states))
            if unexpected:
                errors.append(
                    f"Node {name!r} contains states not declared in states: {unexpected}."
                )
        if spec.role == "quality" and bool(spec.include_in_structure):
            notes.append(
                f"Quality node {name!r} is explicitly included in structure learning; "
                "interpret this as a substantive modeling decision, not a default QC behavior."
            )

    if data_spec.model_family == "mixed":
        notes.append(
            "Mixed categorical/continuous data can be used for pgmpy conditional-Gaussian "
            "structure scores, but the first eyeprocesspy backend does not silently "
            "discretize them."
        )

    if strict and notes:
        errors.extend(notes)
        notes = []
    return BayesianDataValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(notes),
        n_rows=len(data),
        n_nodes=len(data_spec.structure_nodes),
        model_family=data_spec.model_family,
    )
