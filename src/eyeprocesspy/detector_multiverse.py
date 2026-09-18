"""Event-detector multiverse and inference-robustness workflows.

This module keeps event detection as an explicit analytical decision and propagates
that decision through events, AOI assignment, derived features, and statistical
inference. It is deliberately vendor-neutral. Vendor-specific adapters belong in
adapter packages; mature external detectors such as REMoDNaV are bridged rather
than reimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import importlib.metadata
import inspect
import itertools
import json
import math
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import EyeDataset, add_provenance, is_eye_dataset
from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .foundation_09 import _aoi_contains
from .preprocess_features_09 import (
    detect_fixations_idt,
    detect_fixations_ivt,
    detect_saccades,
)
from .schema import empty_eye_table, standardize_eye_table


_RAW_ALGORITHMS = {"ivt", "idt", "adaptive_velocity", "remodnav"}
_SUPPORTED_ALGORITHMS = _RAW_ALGORITHMS | {"external", "vendor"}
_EVENT_LABELS = {
    "FIXA": "fixation",
    "FIX": "fixation",
    "fixation": "fixation",
    "SACC": "saccade",
    "ISAC": "saccade",
    "saccade": "saccade",
    "PURS": "pursuit",
    "PUR": "pursuit",
    "pursuit": "pursuit",
    "HPSO": "pso",
    "IHPS": "pso",
    "LPSO": "pso",
    "ILPS": "pso",
    "pso": "pso",
    "blink": "blink",
    "BLINK": "blink",
}


def _scalar_string(value: Any, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise EyeProcessValidationError(f"`{name}` must be a non-empty scalar string.")
    return value


def _finite_positive(value: Any, name: str, allow_none: bool = True) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if allow_none:
            return None
        raise EyeProcessValidationError(f"`{name}` is required.")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError(f"`{name}` must be numeric.") from exc
    if not np.isfinite(out) or out <= 0:
        raise EyeProcessValidationError(f"`{name}` must be finite and > 0.")
    return out


def _normalise_parameters(parameters: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if parameters is None:
        return tuple()
    if not isinstance(parameters, Mapping):
        raise EyeProcessValidationError("`parameters` must be a mapping.")
    return tuple(sorted(((str(k), v) for k, v in parameters.items()), key=lambda item: item[0]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


@dataclass(frozen=True)
class EventDetectorSpec:
    """Canonical, immutable event-detector specification."""

    detector_id: str
    algorithm: str
    velocity_threshold: float | None = None
    dispersion_threshold: float | None = None
    minimum_duration_ms: float | None = None
    maximum_gap_ms: float | None = None
    merge_rule: str = "none"
    sampling_rate: float | None = None
    smoothing: str | None = None
    filter: str | None = None
    coordinate_unit: str = "degrees"
    implementation: str = "eyeprocesspy"
    implementation_version: str | None = None
    parameters: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    callback: Callable[..., Any] | None = field(default=None, repr=False, compare=False)

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    @property
    def fingerprint(self) -> str:
        payload = {
            "detector_id": self.detector_id,
            "algorithm": self.algorithm,
            "velocity_threshold": self.velocity_threshold,
            "dispersion_threshold": self.dispersion_threshold,
            "minimum_duration_ms": self.minimum_duration_ms,
            "maximum_gap_ms": self.maximum_gap_ms,
            "merge_rule": self.merge_rule,
            "sampling_rate": self.sampling_rate,
            "smoothing": self.smoothing,
            "filter": self.filter,
            "coordinate_unit": self.coordinate_unit,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "parameters": self.parameter_dict,
        }
        blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
        return sha256(blob.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "algorithm": self.algorithm,
            "velocity_threshold": self.velocity_threshold,
            "dispersion_threshold": self.dispersion_threshold,
            "minimum_duration_ms": self.minimum_duration_ms,
            "maximum_gap_ms": self.maximum_gap_ms,
            "merge_rule": self.merge_rule,
            "sampling_rate": self.sampling_rate,
            "smoothing": self.smoothing,
            "filter": self.filter,
            "coordinate_unit": self.coordinate_unit,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "parameters": self.parameter_dict,
            "detector_spec_hash": self.fingerprint,
        }


@dataclass(frozen=True)
class DetectorMultiverse:
    specs: tuple[EventDetectorSpec, ...]
    label: str = "event_detector_multiverse"

    @property
    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame([spec.as_dict() for spec in self.specs])


@dataclass(frozen=True)
class DetectorMultiverseResult:
    multiverse: DetectorMultiverse
    branches: Mapping[str, EyeDataset]
    events: pd.DataFrame
    status: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_fingerprint: str | None = None


@dataclass(frozen=True)
class DetectorInferenceResult:
    multiverse: DetectorMultiverse
    coefficients: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    model_spec: Mapping[str, Any]
    feature_fingerprint: str | None = None


def define_event_detector_spec(
    detector_id: str,
    algorithm: str,
    *,
    velocity_threshold: float | None = None,
    dispersion_threshold: float | None = None,
    minimum_duration_ms: float | None = None,
    maximum_gap_ms: float | None = None,
    merge_rule: str = "none",
    sampling_rate: float | None = None,
    smoothing: str | None = None,
    filter: str | None = None,
    coordinate_unit: str = "degrees",
    implementation: str | None = None,
    implementation_version: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    callback: Callable[..., Any] | None = None,
) -> EventDetectorSpec:
    """Create and validate a canonical detector specification.

    Thresholds are intentionally not given universal defaults. A threshold must
    be supplied when the selected algorithm requires one.
    """
    detector_id = _scalar_string(detector_id, "detector_id")
    algorithm = _scalar_string(algorithm, "algorithm").lower().replace("-", "_")
    if algorithm == "i_vt":
        algorithm = "ivt"
    if algorithm == "i_dt":
        algorithm = "idt"
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise EyeProcessValidationError(
            "`algorithm` must be one of: ivt, idt, adaptive_velocity, remodnav, external, vendor."
        )
    if implementation is None:
        implementation = "REMoDNaV" if algorithm == "remodnav" else "eyeprocesspy"
    spec = EventDetectorSpec(
        detector_id=detector_id,
        algorithm=algorithm,
        velocity_threshold=None if velocity_threshold is None else float(velocity_threshold),
        dispersion_threshold=None if dispersion_threshold is None else float(dispersion_threshold),
        minimum_duration_ms=None if minimum_duration_ms is None else float(minimum_duration_ms),
        maximum_gap_ms=None if maximum_gap_ms is None else float(maximum_gap_ms),
        merge_rule=_scalar_string(merge_rule, "merge_rule"),
        sampling_rate=None if sampling_rate is None else float(sampling_rate),
        smoothing=None if smoothing is None else str(smoothing),
        filter=None if filter is None else str(filter),
        coordinate_unit=_scalar_string(coordinate_unit, "coordinate_unit").lower(),
        implementation=_scalar_string(implementation, "implementation"),
        implementation_version=None if implementation_version is None else str(implementation_version),
        parameters=_normalise_parameters(parameters),
        callback=callback,
    )
    validate_event_detector_spec(spec)
    return spec


def validate_event_detector_spec(spec: EventDetectorSpec) -> bool:
    """Validate a detector specification; returns ``True`` invisibly on success."""
    if not isinstance(spec, EventDetectorSpec):
        raise EyeProcessValidationError("`spec` must be an EventDetectorSpec.")
    if spec.algorithm not in _SUPPORTED_ALGORITHMS:
        raise EyeProcessValidationError("Unsupported detector algorithm.")
    if spec.coordinate_unit not in {"degrees", "pixels", "normalized"}:
        raise EyeProcessValidationError("`coordinate_unit` must be degrees, pixels, or normalized.")
    if spec.algorithm in _RAW_ALGORITHMS:
        _finite_positive(spec.sampling_rate, "sampling_rate", allow_none=False)
        _finite_positive(spec.minimum_duration_ms, "minimum_duration_ms", allow_none=False)
    if spec.algorithm in {"ivt"}:
        _finite_positive(spec.velocity_threshold, "velocity_threshold", allow_none=False)
    if spec.algorithm == "idt":
        _finite_positive(spec.dispersion_threshold, "dispersion_threshold", allow_none=False)
    if spec.maximum_gap_ms is not None:
        _finite_positive(spec.maximum_gap_ms, "maximum_gap_ms", allow_none=False)
    if spec.algorithm == "adaptive_velocity":
        params = spec.parameter_dict
        _finite_positive(params.get("noise_factor"), "parameters['noise_factor']", allow_none=False)
        _finite_positive(
            params.get("minimum_velocity_threshold"),
            "parameters['minimum_velocity_threshold']",
            allow_none=False,
        )
    if spec.algorithm == "remodnav" and spec.coordinate_unit == "pixels":
        _finite_positive(spec.parameter_dict.get("px2deg"), "parameters['px2deg']", allow_none=False)
    if spec.algorithm == "external" and not callable(spec.callback):
        raise EyeProcessValidationError("External detector specs require a callable `callback`.")
    if spec.algorithm == "vendor" and spec.callback is not None:
        raise EyeProcessValidationError("Vendor-event specs do not use `callback`.")
    return True


def create_detector_multiverse(
    specs: Sequence[EventDetectorSpec] | None = None,
    *,
    base_spec: EventDetectorSpec | None = None,
    parameter_grid: Mapping[str, Sequence[Any]] | None = None,
    id_template: str = "{base}_{index:03d}",
    label: str = "event_detector_multiverse",
) -> DetectorMultiverse:
    """Create a deterministic detector multiverse from explicit specs or a grid."""
    assembled: list[EventDetectorSpec] = list(specs or [])
    if parameter_grid is not None:
        if base_spec is None:
            raise EyeProcessValidationError("`base_spec` is required when `parameter_grid` is supplied.")
        if not isinstance(parameter_grid, Mapping) or not parameter_grid:
            raise EyeProcessValidationError("`parameter_grid` must be a non-empty mapping.")
        keys = list(parameter_grid)
        values = [list(parameter_grid[key]) for key in keys]
        if any(len(v) == 0 for v in values):
            raise EyeProcessValidationError("Every detector-grid dimension must contain at least one value.")
        allowed = set(EventDetectorSpec.__dataclass_fields__) - {"callback", "parameters"}
        params_allowed = set(base_spec.parameter_dict)
        for index, combination in enumerate(itertools.product(*values), start=1):
            changes: dict[str, Any] = {}
            parameter_changes = dict(base_spec.parameter_dict)
            for key, value in zip(keys, combination):
                if key.startswith("parameters."):
                    parameter_changes[key.split(".", 1)[1]] = value
                elif key in allowed:
                    changes[key] = value
                elif key in params_allowed:
                    parameter_changes[key] = value
                else:
                    raise EyeProcessValidationError(f"Unknown detector-grid field: {key}")
            changes["detector_id"] = id_template.format(base=base_spec.detector_id, index=index, **changes)
            changes["parameters"] = tuple(sorted(parameter_changes.items()))
            candidate = replace(base_spec, **changes)
            validate_event_detector_spec(candidate)
            assembled.append(candidate)
    if not assembled:
        raise EyeProcessValidationError("Supply at least one detector specification.")
    for spec in assembled:
        validate_event_detector_spec(spec)
    ids = [spec.detector_id for spec in assembled]
    if len(set(ids)) != len(ids):
        raise EyeProcessValidationError("Detector ids must be unique within a multiverse.")
    ordered = tuple(sorted(assembled, key=lambda s: (s.detector_id, s.fingerprint)))
    return DetectorMultiverse(specs=ordered, label=_scalar_string(label, "label"))


def _dataset_fingerprint(x: EyeDataset) -> str:
    pieces = []
    for table_name in ("recordings", "gaze_samples", "intervals", "aoi_definitions", "aoi_geometry"):
        table = x[table_name]
        if table.empty:
            pieces.append(f"{table_name}:empty")
        else:
            stable = table.copy()
            for column in stable.columns:
                stable[column] = stable[column].map(_jsonable)
            pieces.append(table_name + ":" + stable.to_json(orient="split", index=False, date_format="iso"))
    return sha256("|".join(pieces).encode()).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty:
        return None
    stable = frame.copy()
    for column in stable.columns:
        stable[column] = stable[column].map(_jsonable)
    payload = stable.to_json(orient="split", index=False, date_format="iso")
    return sha256(payload.encode()).hexdigest()


def _lineage_fields(x: EyeDataset, spec: EventDetectorSpec | None = None) -> dict[str, Any]:
    return {
        "source_data_hash": _dataset_fingerprint(x),
        "preprocessing_provenance_hash": _frame_hash(x["provenance"]),
        "aoi_spec_hash": _frame_hash(pd.concat([x["aoi_definitions"], x["aoi_geometry"]], ignore_index=True, sort=False))
        if (not x["aoi_definitions"].empty or not x["aoi_geometry"].empty) else None,
        "quality_spec_hash": _frame_hash(x["quality"]),
        "detector_spec_hash": spec.fingerprint if spec is not None else None,
        "detector_implementation": spec.implementation if spec is not None else None,
        "detector_implementation_version": spec.implementation_version if spec is not None else None,
        "software": "eyeprocesspy",
        "software_version": "0.1.0",
    }


def _clean_branch_dataset(x: EyeDataset) -> EyeDataset:
    out = x.copy()
    episodes = out["episodes"].copy()
    if not episodes.empty:
        remove = episodes["episode_type"].isin(["fixation", "saccade", "pursuit", "pso"])
        episodes = episodes.loc[~remove].copy()
    out["episodes"] = standardize_eye_table(episodes, "episodes")
    return out


def _attach_detector_columns(events: pd.DataFrame, spec: EventDetectorSpec) -> pd.DataFrame:
    events = standardize_eye_table(events, "episodes")
    if events.empty:
        for name, value in {
            "detector_id": spec.detector_id,
            "detector_algorithm": spec.algorithm,
            "detector_spec_hash": spec.fingerprint,
            "detector_implementation": spec.implementation,
            "detector_implementation_version": spec.implementation_version,
        }.items():
            events[name] = pd.Series(dtype="object")
        return events
    events = events.copy()
    events["detector_id"] = spec.detector_id
    events["detector_algorithm"] = spec.algorithm
    events["detector_spec_hash"] = spec.fingerprint
    events["detector_implementation"] = spec.implementation
    events["detector_implementation_version"] = spec.implementation_version
    return events


def _check_sampling_rate(x: EyeDataset, spec: EventDetectorSpec, tolerance_fraction: float = 0.10) -> list[str]:
    data = x["gaze_samples"]
    if data.empty:
        return []
    rates = []
    for _, group in data.groupby(["recording_id", "trial_id"], dropna=False, sort=False):
        t = pd.to_numeric(group["timestamp_seconds"], errors="coerce").dropna().sort_values().to_numpy(float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            rates.append(1.0 / np.median(dt))
    if not rates:
        return ["Sampling rate could not be verified from timestamps."]
    empirical = float(np.median(rates))
    expected = float(spec.sampling_rate)
    if abs(empirical - expected) / expected > float(tolerance_fraction):
