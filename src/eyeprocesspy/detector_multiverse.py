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
        return [f"Empirical sampling rate ({empirical:.3f} Hz) differs from the detector specification ({expected:.3f} Hz)."]
    return []


def _run_adaptive_velocity(x: EyeDataset, spec: EventDetectorSpec) -> EyeDataset:
    """Transparent robust-MAD adaptive velocity reference detector.

    This is intentionally labelled a reference implementation, not an implementation
    of REMoDNaV, Nyström-Holmqvist, Engbert-Kliegl, or any other named detector.
    """
    params = spec.parameter_dict
    noise_factor = float(params["noise_factor"])
    min_threshold = float(params["minimum_velocity_threshold"])
    out = _clean_branch_dataset(x)
    rows: list[dict[str, Any]] = []
    counter = 0
    for _, group in out["gaze_samples"].groupby(["recording_id", "trial_id"], dropna=False, sort=False):
        z = group.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
        t = pd.to_numeric(z["timestamp_seconds"], errors="coerce").to_numpy(float)
        gx = pd.to_numeric(z["gaze_x"], errors="coerce").to_numpy(float)
        gy = pd.to_numeric(z["gaze_y"], errors="coerce").to_numpy(float)
        valid = z["valid"].astype("boolean").fillna(False).to_numpy(bool)
        dt = np.r_[np.nan, np.diff(t)]
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.r_[np.nan, np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2) / np.diff(t)]
        usable = velocity[np.isfinite(velocity) & valid]
        if usable.size < 3:
            continue
        centre = float(np.median(usable))
        mad = float(np.median(np.abs(usable - centre)))
        robust_sigma = 1.4826 * mad
        threshold = max(min_threshold, centre + noise_factor * robust_sigma)
        is_fix = np.isfinite(velocity) & valid & (velocity <= threshold)
        if len(is_fix):
            is_fix[0] = bool(is_fix[1]) if len(is_fix) > 1 else False
        gap_limit = float(spec.maximum_gap_ms or (1000.0 / float(spec.sampling_rate) * 2.5))
        run_ids = np.zeros(len(z), dtype=int)
        run = 0
        for i in range(len(z)):
            if i == 0 or (not is_fix[i]) or (not is_fix[i - 1]) or (np.isfinite(dt[i]) and dt[i] * 1000 > gap_limit):
                run += 1
            run_ids[i] = run
        for run_id in pd.unique(run_ids[is_fix]):
            pos = np.flatnonzero((run_ids == run_id) & is_fix)
            if not len(pos):
                continue
            duration = (np.nanmax(t[pos]) - np.nanmin(t[pos])) * 1000
            if duration < float(spec.minimum_duration_ms):
                continue
            counter += 1
            rows.append(
                {
                    "episode_id": f"{z.iloc[0]['recording_id']}_adaptive_fix_{counter:07d}",
                    "recording_id": z.iloc[0]["recording_id"],
                    "episode_type": "fixation",
                    "eye": "combined",
                    "start_time": float(t[pos.min()]),
                    "end_time": float(t[pos.max()]),
                    "duration_ms": float(duration),
                    "start_x": float(gx[pos.min()]),
                    "start_y": float(gy[pos.min()]),
                    "end_x": float(gx[pos.max()]),
                    "end_y": float(gy[pos.max()]),
                    "centroid_x": float(np.nanmean(gx[pos])),
                    "centroid_y": float(np.nanmean(gy[pos])),
                    "amplitude": np.nan,
                    "peak_velocity": float(np.nanmax(velocity[pos])) if np.isfinite(velocity[pos]).any() else np.nan,
                    "dispersion": float((np.nanmax(gx[pos]) - np.nanmin(gx[pos])) + (np.nanmax(gy[pos]) - np.nanmin(gy[pos]))),
                    "coordinate_space_id": z.iloc[0]["coordinate_space_id"],
                    "source_algorithm": "adaptive velocity (robust-MAD reference)",
                    "source_parameters": f"noise_factor={noise_factor:g};minimum_velocity_threshold={min_threshold:g};adaptive_threshold={threshold:g}",
                    "derived_by": "eyeprocess",
                    "trial_id": z.iloc[0]["trial_id"],
                    "stimulus_id": z.iloc[0]["stimulus_id"],
                    "aoi_id": pd.NA,
                }
            )
    detected = standardize_eye_table(pd.DataFrame(rows), "episodes") if rows else empty_eye_table("episodes")
    existing = out["episodes"]
    out["episodes"] = standardize_eye_table(pd.concat([existing, detected], ignore_index=True, sort=False), "episodes")
    return add_provenance(
        out,
        "detect_fixations_adaptive_velocity",
        "episodes",
        f"detector_id={spec.detector_id};spec_hash={spec.fingerprint};n={len(detected)}",
    )


def _remodnav_version() -> str | None:
    try:
        return importlib.metadata.version("remodnav")
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_remodnav(x: EyeDataset, spec: EventDetectorSpec) -> EyeDataset:
    try:
        import remodnav
    except ImportError as exc:
        raise EyeProcessBackendError(
            "REMoDNaV is not installed. Install the optional gaze dependency and rerun; no surrogate detector is substituted."
        ) from exc
    params = dict(spec.parameter_dict)
    px2deg = 1.0 if spec.coordinate_unit == "degrees" else params.pop("px2deg", None)
    if px2deg is None:
        raise EyeProcessValidationError("REMoDNaV with pixel coordinates requires an explicit `px2deg` parameter.")
    if spec.coordinate_unit == "normalized":
        raise EyeProcessValidationError("REMoDNaV cannot consume normalized coordinates without an explicit coordinate conversion first.")

    classifier_keys = set(inspect.signature(remodnav.EyegazeClassifier.__init__).parameters) - {"self"}
    preproc_keys = set(inspect.signature(remodnav.EyegazeClassifier.preproc).parameters) - {"self", "data"}
    classifier_args = {
        "px2deg": float(px2deg),
        "sampling_rate": float(spec.sampling_rate),
        "min_fixation_duration": float(spec.minimum_duration_ms) / 1000.0,
    }
    preproc_args: dict[str, Any] = {}
    unused: dict[str, Any] = {}
    for key, value in params.items():
        if key in classifier_keys:
            classifier_args[key] = value
        elif key in preproc_keys:
            preproc_args[key] = value
        elif key not in {"include_labels"}:
            unused[key] = value
    if unused:
        raise EyeProcessValidationError("Unknown REMoDNaV parameter(s): " + ", ".join(sorted(unused)))
    include_labels = set(params.get("include_labels", ["FIXA", "SACC", "ISAC", "PURS", "HPSO", "IHPS", "LPSO", "ILPS"]))

    out = _clean_branch_dataset(x)
    rows = []
    counter = 0
    for _, group in out["gaze_samples"].groupby(["recording_id", "trial_id"], dropna=False, sort=False):
        z = group.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
        if len(z) < 3:
            continue
        coords = np.recarray(len(z), dtype=[("x", "f8"), ("y", "f8")])
        coords.x = pd.to_numeric(z["gaze_x"], errors="coerce").to_numpy(float)
        coords.y = pd.to_numeric(z["gaze_y"], errors="coerce").to_numpy(float)
        invalid = ~z["valid"].astype("boolean").fillna(False).to_numpy(bool)
        coords.x[invalid] = np.nan
        coords.y[invalid] = np.nan
        clf = remodnav.EyegazeClassifier(**classifier_args)
        pp = clf.preproc(coords, **preproc_args)
        detected = clf(pp, classify_isp=True, sort_events=True)
        t0 = float(pd.to_numeric(z["timestamp_seconds"], errors="coerce").min())
        for event in detected:
            label = str(event.get("label", ""))
            if label not in include_labels:
                continue
            event_type = _EVENT_LABELS.get(label)
            if event_type is None:
                continue
            counter += 1
            start = t0 + float(event["start_time"])
            end = t0 + float(event["end_time"])
            rows.append(
                {
                    "episode_id": f"{z.iloc[0]['recording_id']}_remodnav_{counter:07d}",
                    "recording_id": z.iloc[0]["recording_id"],
                    "episode_type": event_type,
                    "eye": "combined",
                    "start_time": start,
                    "end_time": end,
                    "duration_ms": (end - start) * 1000.0,
                    "start_x": event.get("start_x", np.nan),
                    "start_y": event.get("start_y", np.nan),
                    "end_x": event.get("end_x", np.nan),
                    "end_y": event.get("end_y", np.nan),
                    "centroid_x": np.nanmean([event.get("start_x", np.nan), event.get("end_x", np.nan)]),
                    "centroid_y": np.nanmean([event.get("start_y", np.nan), event.get("end_y", np.nan)]),
                    "amplitude": event.get("amp", np.nan),
                    "peak_velocity": event.get("peak_vel", np.nan),
                    "dispersion": np.nan,
                    "coordinate_space_id": z.iloc[0]["coordinate_space_id"],
                    "source_algorithm": "REMoDNaV",
                    "source_parameters": json.dumps(_jsonable({**classifier_args, **preproc_args}), sort_keys=True),
                    "derived_by": "external",
                    "trial_id": z.iloc[0]["trial_id"],
                    "stimulus_id": z.iloc[0]["stimulus_id"],
                    "aoi_id": pd.NA,
                }
            )
    detected = standardize_eye_table(pd.DataFrame(rows), "episodes") if rows else empty_eye_table("episodes")
    out["episodes"] = standardize_eye_table(pd.concat([out["episodes"], detected], ignore_index=True, sort=False), "episodes")
    return add_provenance(
        out,
        "detect_events_remodnav",
        "episodes",
        f"detector_id={spec.detector_id};spec_hash={spec.fingerprint};remodnav_version={_remodnav_version()};n={len(detected)}",
    )


def import_external_detector_events(
    events: pd.DataFrame,
    spec: EventDetectorSpec,
    *,
    dataset: EyeDataset | None = None,
) -> pd.DataFrame:
    """Validate and normalize an external detector event catalogue."""
    validate_event_detector_spec(spec)
    if not isinstance(events, pd.DataFrame):
        raise EyeProcessValidationError("External detector output must be a pandas DataFrame.")
    data = events.copy()
    rename = {}
    if "onset" in data and "start_time" not in data:
        rename["onset"] = "start_time"
    if "label" in data and "episode_type" not in data:
        rename["label"] = "episode_type"
    data = data.rename(columns=rename)
    if "end_time" not in data and {"start_time", "duration"}.issubset(data):
        data["end_time"] = pd.to_numeric(data["start_time"], errors="coerce") + pd.to_numeric(data["duration"], errors="coerce")
    required = {"recording_id", "episode_type", "start_time", "end_time"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise EyeProcessValidationError("External detector events are missing: " + ", ".join(missing))
    data["episode_type"] = data["episode_type"].map(lambda value: _EVENT_LABELS.get(str(value), str(value).lower()))
    start = pd.to_numeric(data["start_time"], errors="coerce")
    end = pd.to_numeric(data["end_time"], errors="coerce")
    if start.isna().any() or end.isna().any() or (end < start).any():
        raise EyeProcessValidationError("External detector event times must be finite with end_time >= start_time.")
    if "duration_ms" not in data:
        data["duration_ms"] = (end - start) * 1000.0
    if "episode_id" not in data:
        data["episode_id"] = [f"{spec.detector_id}_external_{i:07d}" for i in range(1, len(data) + 1)]
    if "eye" not in data:
        data["eye"] = "combined"
    if "source_algorithm" not in data:
        data["source_algorithm"] = spec.implementation
    if "source_parameters" not in data:
        data["source_parameters"] = json.dumps(_jsonable(spec.parameter_dict), sort_keys=True)
    if "derived_by" not in data:
        data["derived_by"] = "external"
    if "trial_id" not in data:
        data["trial_id"] = pd.NA
    if dataset is not None && data["trial_id"].isna().any() && !dataset["intervals"].empty:
        intervals = dataset["intervals"].loc[dataset["intervals"]["interval_type"].eq("trial")]
        for idx in data.index[data["trial_id"].isna()]:
            rec = data.at[idx, "recording_id"]
            time = float(data.at[idx, "start_time"])
            hit = intervals[
                intervals["recording_id"].eq(rec)
                & (pd.to_numeric(intervals["start_time"], errors="coerce") <= time)
                & (pd.to_numeric(intervals["end_time"], errors="coerce") >= time)
            ]
            if len(hit) == 1:
                data.at[idx, "trial_id"] = hit.iloc[0]["trial_id"]
            elif len(hit) > 1:
                raise EyeProcessValidationError("An external event maps to multiple trial intervals; resolve the trial definition explicitly.")
    return _attach_detector_columns(data, spec)


def detect_events_with_spec(x: EyeDataset, spec: EventDetectorSpec) -> EyeDataset:
    """Run one explicit detector specification and return an isolated EyeDataset branch."""
    if not is_eye_dataset(x):
        raise EyeProcessValidationError("`x` must be an EyeDataset.")
    validate_event_detector_spec(spec)
    branch = _clean_branch_dataset(x)
    sampling_warnings = _check_sampling_rate(branch, spec) if spec.algorithm in _RAW_ALGORITHMS else []
    for message in sampling_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    if spec.algorithm == "ivt":
        branch = detect_fixations_ivt(
            branch,
            velocity_threshold=float(spec.velocity_threshold),
            minimum_duration_ms=float(spec.minimum_duration_ms),
            maximum_gap_ms=float(spec.maximum_gap_ms or 75.0),
            coordinate_units=spec.coordinate_unit,
            overwrite=False,
        )
        if bool(spec.parameter_dict.get("include_saccades", False)):
            branch = detect_saccades(
                branch,
                velocity_threshold=float(spec.velocity_threshold),
                minimum_duration_ms=float(spec.parameter_dict.get("minimum_saccade_duration_ms", 10.0)),
                overwrite=False,
            )
    elif spec.algorithm == "idt":
        branch = detect_fixations_idt(
            branch,
            dispersion_threshold=float(spec.dispersion_threshold),
            minimum_duration_ms=float(spec.minimum_duration_ms),
            coordinate_units=spec.coordinate_unit,
            overwrite=False,
        )
    elif spec.algorithm == "adaptive_velocity":
        branch = _run_adaptive_velocity(branch, spec)
    elif spec.algorithm == "remodnav":
        branch = _run_remodnav(branch, spec)
    elif spec.algorithm == "external":
        output = spec.callback(data=branch.copy(), spec=spec)
        if is_eye_dataset(output):
            external = output["episodes"].copy()
        else:
            external = output
        external = import_external_detector_events(external, spec, dataset=branch)
        branch["episodes"] = standardize_eye_table(
            pd.concat([branch["episodes"], external], ignore_index=True, sort=False),
            "episodes",
        )
        branch = add_provenance(
            branch,
            "detect_events_external",
            "episodes",
            f"detector_id={spec.detector_id};spec_hash={spec.fingerprint};n={len(external)}",
        )
    elif spec.algorithm == "vendor":
        vendor = x["episodes"].copy()
        params = spec.parameter_dict
        if "derived_by" in vendor:
            vendor = vendor[vendor["derived_by"].eq(params.get("derived_by", "vendor"))]
        vendor = _attach_detector_columns(vendor, spec)
        branch["episodes"] = standardize_eye_table(
            pd.concat([branch["episodes"], vendor], ignore_index=True, sort=False),
            "episodes",
        )
        branch = add_provenance(
            branch,
            "import_vendor_detector_events",
            "episodes",
            f"detector_id={spec.detector_id};spec_hash={spec.fingerprint};n={len(vendor)}",
        )

    relevant = branch["episodes"].copy()
    if spec.algorithm != "vendor":
        relevant = relevant[relevant["episode_type"].isin(["fixation", "saccade", "pursuit", "pso"])]
    relevant = _attach_detector_columns(relevant, spec)
    lineage = _lineage_fields(x, spec)
    for column, value in lineage.items():
        relevant[column] = value
    keep_ids = set(relevant["episode_id"].astype(str)) if not relevant.empty else set()
    episodes = branch["episodes"].copy()
    for column in relevant.columns:
        if column not in episodes.columns:
            episodes[column] = pd.NA
    if keep_ids:
        mask = episodes["episode_id"].astype(str).isin(keep_ids)
        provenance_columns = [
            c for c in relevant.columns
            if c.startswith("detector_") or c in {
                "source_data_hash", "preprocessing_provenance_hash", "aoi_spec_hash",
                "quality_spec_hash", "software", "software_version"
            }
        ]
        for column in provenance_columns:
            lookup = relevant.set_index(relevant["episode_id"].astype(str))[column]
            episodes.loc[mask, column] = episodes.loc[mask, "episode_id"].astype(str).map(lookup)
    branch["episodes"] = episodes
    return add_provenance(
        branch,
        "detect_events_with_spec",
        "episodes",
        f"detector_id={spec.detector_id};algorithm={spec.algorithm};spec_hash={spec.fingerprint}",
        warnings=" | ".join(sampling_warnings) if sampling_warnings else pd.NA,
    )


def run_detector_multiverse(
    x: EyeDataset,
    multiverse: DetectorMultiverse | Sequence[EventDetectorSpec],
    *,
    continue_on_error: bool = True,
) -> DetectorMultiverseResult:
    """Run all detector branches independently in deterministic detector-id order."""
    if not is_eye_dataset(x):
