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
    if dataset is not None and data["trial_id"].isna().any() and not dataset["intervals"].empty:
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
        raise EyeProcessValidationError("`x` must be an EyeDataset.")
    if not isinstance(multiverse, DetectorMultiverse):
        multiverse = create_detector_multiverse(multiverse)
    branches: dict[str, EyeDataset] = {}
    event_frames = []
    statuses = []
    failures = []
    warning_rows = []
    for spec in multiverse.specs:
        captured: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                branch = detect_events_with_spec(x, spec)
            captured = [str(item.message) for item in caught]
            branches[spec.detector_id] = branch
            events = branch["episodes"].copy()
            if "detector_id" in events:
                events = events[events["detector_id"].eq(spec.detector_id)]
            else:
                events = events.iloc[0:0]
            event_frames.append(events)
            statuses.append(
                {
                    "detector_id": spec.detector_id,
                    "detector_spec_hash": spec.fingerprint,
                    "status": "ok",
                    "n_events": len(events),
                    "n_fixations": int(events["episode_type"].eq("fixation").sum()) if not events.empty else 0,
                }
            )
            for message in captured:
                warning_rows.append({"detector_id": spec.detector_id, "stage": "detection", "warning": message})
        except Exception as exc:  # branch failure is recorded, never treated as a valid result
            failures.append(
                {
                    "detector_id": spec.detector_id,
                    "detector_spec_hash": spec.fingerprint,
                    "stage": "detection",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            statuses.append(
                {
                    "detector_id": spec.detector_id,
                    "detector_spec_hash": spec.fingerprint,
                    "status": "failed",
                    "n_events": np.nan,
                    "n_fixations": np.nan,
                }
            )
            if not continue_on_error:
                raise
    events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    return DetectorMultiverseResult(
        multiverse=multiverse,
        branches=branches,
        events=events,
        status=pd.DataFrame(statuses),
        failures=pd.DataFrame(failures),
        warnings=pd.DataFrame(warning_rows),
        source_fingerprint=_dataset_fingerprint(x),
    )


def _event_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else float(a_start == b_start and a_end == b_end)


def match_detected_events(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    event_type: str = "fixation",
    onset_tolerance_ms: float = 75.0,
    minimum_overlap: float = 0.10,
) -> pd.DataFrame:
    """One-to-one temporal matching robust to different event catalogue lengths."""
    if onset_tolerance_ms < 0 or minimum_overlap < 0 or minimum_overlap > 1:
        raise EyeProcessValidationError("Invalid event-matching tolerance.")
    required = {"recording_id", "episode_type", "start_time", "end_time"}
    for name, frame in {"reference": reference, "candidate": candidate}.items():
        if not isinstance(frame, pd.DataFrame):
            raise EyeProcessValidationError(f"`{name}` must be a DataFrame.")
        missing = required - set(frame.columns)
        if missing:
            raise EyeProcessValidationError(f"`{name}` is missing: {', '.join(sorted(missing))}.")
    ref = reference[reference["episode_type"].eq(event_type)].copy().reset_index(drop=True)
    cand = candidate[candidate["episode_type"].eq(event_type)].copy().reset_index(drop=True)
    if ref.empty or cand.empty:
        return pd.DataFrame(
            columns=[
                "reference_index", "candidate_index", "recording_id", "trial_id",
                "event_type", "overlap_iou", "onset_difference_ms", "offset_difference_ms",
                "duration_difference_ms",
            ]
        )
    candidates = []
    for i, a in ref.iterrows():
        for j, b in cand.iterrows():
            if str(a["recording_id"]) != str(b["recording_id"]):
                continue
            if "trial_id" in ref and "trial_id" in cand and pd.notna(a.get("trial_id")) and pd.notna(b.get("trial_id")):
                if str(a.get("trial_id")) != str(b.get("trial_id")):
                    continue
            a_start, a_end = float(a["start_time"]), float(a["end_time"])
            b_start, b_end = float(b["start_time"]), float(b["end_time"])
            onset_diff = abs(a_start - b_start) * 1000.0
            iou = _event_iou(a_start, a_end, b_start, b_end)
            if iou < minimum_overlap and onset_diff > onset_tolerance_ms:
                continue
            candidates.append((iou, -onset_diff, i, j))
    candidates.sort(reverse=True)
    used_ref: set[int] = set()
    used_cand: set[int] = set()
    rows = []
    for iou, neg_onset, i, j in candidates:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        a = ref.iloc[i]
        b = cand.iloc[j]
        rows.append(
            {
                "reference_index": i,
                "candidate_index": j,
                "recording_id": a["recording_id"],
                "trial_id": a.get("trial_id", pd.NA),
                "event_type": event_type,
                "overlap_iou": float(iou),
                "onset_difference_ms": float(-neg_onset),
                "offset_difference_ms": abs(float(a["end_time"]) - float(b["end_time"])) * 1000.0,
                "duration_difference_ms": float(b["end_time"] - b["start_time"] - (a["end_time"] - a["start_time"])) * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def compare_event_catalogues(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    event_type: str = "fixation",
    onset_tolerance_ms: float = 75.0,
    minimum_overlap: float = 0.10,
) -> pd.DataFrame:
    """Compare two event catalogues using one-to-one temporal matching."""
    matches = match_detected_events(
        reference,
        candidate,
        event_type=event_type,
        onset_tolerance_ms=onset_tolerance_ms,
        minimum_overlap=minimum_overlap,
    )
    n_ref = int(reference["episode_type"].eq(event_type).sum())
    n_cand = int(candidate["episode_type"].eq(event_type).sum())
    n_match = len(matches)
    precision = n_match / n_cand if n_cand else np.nan
    recall = n_match / n_ref if n_ref else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
    return pd.DataFrame(
        [{
            "event_type": event_type,
            "reference_events": n_ref,
            "candidate_events": n_cand,
            "matched_events": n_match,
            "matched_event_precision": precision,
            "matched_event_recall": recall,
            "f1": f1,
            "mean_event_overlap": matches["overlap_iou"].mean() if n_match else np.nan,
            "median_event_overlap": matches["overlap_iou"].median() if n_match else np.nan,
            "mean_onset_difference_ms": matches["onset_difference_ms"].mean() if n_match else np.nan,
            "mean_offset_difference_ms": matches["offset_difference_ms"].mean() if n_match else np.nan,
            "mean_duration_difference_ms": matches["duration_difference_ms"].mean() if n_match else np.nan,
        }]
    )


def estimate_detector_agreement(
    x: DetectorMultiverseResult | pd.DataFrame,
    *,
    event_type: str = "fixation",
    onset_tolerance_ms: float = 75.0,
    minimum_overlap: float = 0.10,
) -> pd.DataFrame:
    """Pairwise detector agreement across all successful detector branches."""
    events = x.events if isinstance(x, DetectorMultiverseResult) else x
    if not isinstance(events, pd.DataFrame) or "detector_id" not in events:
        raise EyeProcessValidationError("Detector-labelled event data are required.")
    ids = sorted(events["detector_id"].dropna().astype(str).unique())
    rows = []
    for left, right in itertools.combinations(ids, 2):
        a = events[events["detector_id"].astype(str).eq(left)]
        b = events[events["detector_id"].astype(str).eq(right)]
        comp = compare_event_catalogues(
            a, b, event_type=event_type, onset_tolerance_ms=onset_tolerance_ms, minimum_overlap=minimum_overlap
        )
        row = comp.iloc[0].to_dict()
        row.update({"detector_a": left, "detector_b": right})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_detector_events(x: DetectorMultiverseResult | pd.DataFrame) -> pd.DataFrame:
    """Detector-level event counts and fixation-duration summaries."""
    events = x.events if isinstance(x, DetectorMultiverseResult) else x
    if events.empty:
        return pd.DataFrame()
    if "detector_id" not in events:
        raise EyeProcessValidationError("`events` must include detector_id.")
    rows = []
    for detector_id, group in events.groupby("detector_id", sort=True, dropna=False):
        fix = group[group["episode_type"].eq("fixation")]
        duration = pd.to_numeric(fix["duration_ms"], errors="coerce")
        rows.append(
            {
                "detector_id": detector_id,
                "number_of_events": len(group),
                "number_of_fixations": len(fix),
                "number_of_saccades": int(group["episode_type"].eq("saccade").sum()),
                "mean_fixation_duration_ms": duration.mean() if len(fix) else np.nan,
                "median_fixation_duration_ms": duration.median() if len(fix) else np.nan,
                "total_fixation_duration_ms": duration.sum(min_count=1) if len(fix) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarise_detector_disagreement(x: DetectorMultiverseResult, *, event_type: str = "fixation") -> pd.DataFrame:
    """Pairwise detector disagreement expressed without assuming a gold standard."""
    agreement = estimate_detector_agreement(x, event_type=event_type)
    if agreement.empty:
        return agreement
    out = agreement.copy()
    out["unmatched_reference"] = out["reference_events"] - out["matched_events"]
    out["unmatched_candidate"] = out["candidate_events"] - out["matched_events"]
    out["event_count_difference"] = out["candidate_events"] - out["reference_events"]
    return out


def _assign_episode_aois_explicit(branch: EyeDataset, overlap: str) -> EyeDataset:
    if overlap not in {"error", "first", "smallest", "all"}:
        raise EyeProcessValidationError("`overlap` must be error, first, smallest, or all.")
    if branch["aoi_definitions"].empty or branch["aoi_geometry"].empty:
        raise EyeProcessValidationError("No AOIs are registered; detector-to-AOI propagation cannot continue.")
    out = branch.copy()
    episodes = out["episodes"].copy()
    if episodes.empty:
        return out
    definitions = out["aoi_definitions"].reset_index(drop=True)
    geometries = out["aoi_geometry"]
    assignments: list[Any] = []
    for _, event in episodes.iterrows():
        if event["episode_type"] not in {"fixation", "pursuit"} or not np.isfinite(pd.to_numeric(pd.Series([event["centroid_x"]]), errors="coerce").iloc[0]):
            assignments.append(event.get("aoi_id", pd.NA))
            continue
        hits: list[tuple[str, float, int]] = []
        for order, definition in definitions.iterrows():
            if pd.notna(definition["stimulus_id"]) and str(definition["stimulus_id"]).strip():
                if str(event.get("stimulus_id")) != str(definition["stimulus_id"]):
                    continue
            selected = geometries[geometries["aoi_id"].astype(str).eq(str(definition["aoi_id"]))]
            for _, geometry in selected.iterrows():
                if str(event.get("coordinate_space_id")) != str(geometry["coordinate_space_id"]):
                    continue
                hit = _aoi_contains(
                    [event["centroid_x"]], [event["centroid_y"]], [event["start_time"]], definition, geometry
                )[0]
                if hit:
                    width = pd.to_numeric(pd.Series([geometry["width"]]), errors="coerce").iloc[0]
                    height = pd.to_numeric(pd.Series([geometry["height"]]), errors="coerce").iloc[0]
                    area = float(width * height) if np.isfinite(width) and np.isfinite(height) else np.inf
                    hits.append((str(definition["aoi_id"]), area, order))
        unique = []
        seen = set()
        for hit in hits:
            if hit[0] not in seen:
                unique.append(hit)
                seen.add(hit[0])
        if len(unique) > 1 and overlap == "error":
            raise EyeProcessValidationError(
                f"Ambiguous AOI assignment for episode {event.get('episode_id')}: "
                + ", ".join(item[0] for item in unique)
                + ". Choose an overlap rule explicitly."
            )
        if not unique:
            assignments.append(pd.NA)
        elif overlap == "all":
            assignments.append("|".join(item[0] for item in unique))
        elif overlap == "smallest":
            assignments.append(min(unique, key=lambda item: (item[1], item[2]))[0])
        else:
            assignments.append(unique[0][0])
    episodes["aoi_id"] = assignments
    out["episodes"] = episodes
    return add_provenance(out, "propagate_detector_to_aoi", "episodes", f"overlap={overlap}")


def propagate_detector_to_aoi(
    x: DetectorMultiverseResult,
    *,
    overlap: str = "error",
    continue_on_error: bool = True,
) -> DetectorMultiverseResult:
    """Assign AOIs independently within every detector branch."""
    if not isinstance(x, DetectorMultiverseResult):
        raise EyeProcessValidationError("`x` must be a DetectorMultiverseResult.")
    branches: dict[str, EyeDataset] = {}
    failures = [] if x.failures.empty else x.failures.to_dict("records")
    warnings_rows = [] if x.warnings.empty else x.warnings.to_dict("records")
    event_frames = []
    for spec in x.multiverse.specs:
        if spec.detector_id not in x.branches:
            continue
        try:
            branch = _assign_episode_aois_explicit(x.branches[spec.detector_id], overlap)
            branches[spec.detector_id] = branch
            events = branch["episodes"].copy()
            if "detector_id" in events:
                events = events[events["detector_id"].eq(spec.detector_id)]
            event_frames.append(events)
        except Exception as exc:
            failures.append({
                "detector_id": spec.detector_id,
                "detector_spec_hash": spec.fingerprint,
                "stage": "aoi_assignment",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if not continue_on_error:
                raise
    events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    status = x.status.copy()
    failed_ids = {row["detector_id"] for row in failures if row.get("stage") == "aoi_assignment"}
    if not status.empty and failed_ids:
        status.loc[status["detector_id"].isin(failed_ids), "status"] = "failed_aoi"
    return replace(
        x,
        branches=branches,
        events=events,
        status=status,
        failures=pd.DataFrame(failures),
        warnings=pd.DataFrame(warnings_rows),
    )


def _trial_rows(branch: EyeDataset) -> pd.DataFrame:
    intervals = branch["intervals"].copy()
    trials = intervals[intervals["interval_type"].eq("trial")].copy()
    if trials.empty:
        raise EyeProcessValidationError("Explicit trial intervals are required for detector-to-feature propagation.")
    if trials["trial_id"].isna().any():
        raise EyeProcessValidationError("Trial intervals must have non-missing trial_id values.")
    if trials.duplicated(["recording_id", "trial_id"]).any():
        raise EyeProcessValidationError("Trial intervals must be unique by recording_id and trial_id.")
    return trials


def _trial_valid_fraction(branch: EyeDataset, recording_id: Any, trial_id: Any) -> tuple[int, float]:
    gaze = branch["gaze_samples"]
    subset = gaze[gaze["recording_id"].eq(recording_id) & gaze["trial_id"].eq(trial_id)]
    if subset.empty:
        return 0, np.nan
    valid = subset["valid"].astype("boolean").fillna(False).to_numpy(bool)
    finite = np.isfinite(pd.to_numeric(subset["gaze_x"], errors="coerce")) & np.isfinite(pd.to_numeric(subset["gaze_y"], errors="coerce"))
    observed = valid & finite
    return len(subset), float(observed.mean())


def _pupil_within_fixations(branch: EyeDataset, fixations: pd.DataFrame, recording_id: Any, trial_id: Any) -> float:
    eye = branch["eye_samples"]
    if eye.empty or fixations.empty:
        return np.nan
    data = eye[eye["recording_id"].eq(recording_id) & eye["trial_id"].eq(trial_id)].copy()
    if data.empty:
        return np.nan
    times = pd.to_numeric(data["timestamp_seconds"], errors="coerce").to_numpy(float)
    pupil = pd.to_numeric(data["pupil_diameter"], errors="coerce").to_numpy(float)
    valid = data["pupil_valid"].astype("boolean").fillna(False).to_numpy(bool)
    mask = np.zeros(len(data), dtype=bool)
    for _, fix in fixations.iterrows():
        mask |= (times >= float(fix["start_time"])) & (times <= float(fix["end_time"]))
    values = pupil[mask & valid & np.isfinite(pupil)]
    return float(np.mean(values)) if values.size else np.nan


def _derive_branch_features(branch: EyeDataset, spec: EventDetectorSpec) -> pd.DataFrame:
    trials = _trial_rows(branch)
    aois = branch["aoi_definitions"].copy()
    if aois.empty:
        raise EyeProcessValidationError("AOI definitions are required before feature propagation.")
    episodes = branch["episodes"].copy()
    fixations = episodes[episodes["episode_type"].eq("fixation")].copy()
    if "detector_id" in fixations:
        fixations = fixations[fixations["detector_id"].eq(spec.detector_id)]
    rows = []
    for _, trial in trials.iterrows():
        rec = trial["recording_id"]
        trial_id = trial["trial_id"]
        start = float(trial["start_time"])
        end = float(trial["end_time"])
        duration_ms = (end - start) * 1000.0
        n_samples, valid_fraction = _trial_valid_fraction(branch, rec, trial_id)
        trial_fix = fixations[fixations["recording_id"].eq(rec) & fixations["trial_id"].eq(trial_id)].sort_values("start_time")
        seq = [str(value) for value in trial_fix["aoi_id"].dropna().tolist()]
        collapsed = [value for i, value in enumerate(seq) if i == 0 or value != seq[i - 1]]
        for _, aoi in aois.iterrows():
            aoi_id = str(aoi["aoi_id"])
            target = trial_fix[trial_fix["aoi_id"].astype("string").eq(aoi_id)].copy()
            observed_trial = n_samples > 0 and np.isfinite(valid_fraction) and valid_fraction > 0
            count = float(len(target)) if observed_trial else np.nan
            dwell = float(pd.to_numeric(target["duration_ms"], errors="coerce").sum()) if observed_trial and len(target) else (0.0 if observed_trial else np.nan)
            mean_duration = float(pd.to_numeric(target["duration_ms"], errors="coerce").mean()) if len(target) else np.nan
            first_latency = (float(target["start_time"].min()) - start) * 1000.0 if len(target) else np.nan
            entries = sum(1 for i, state in enumerate(collapsed) if state == aoi_id and (i == 0 or collapsed[i - 1] != aoi_id))
            revisits = max(entries - 1, 0) if observed_trial else np.nan
            transitions_from = sum(1 for left, right in zip(collapsed[:-1], collapsed[1:]) if left == aoi_id and right != aoi_id)
            transitions_to = sum(1 for left, right in zip(collapsed[:-1], collapsed[1:]) if right == aoi_id and left != aoi_id)
            row = {
                "detector_id": spec.detector_id,
                "detector_algorithm": spec.algorithm,
                "detector_spec_hash": spec.fingerprint,
                "recording_id": rec,
                "participant_id": trial.get("participant_id", pd.NA),
                "trial_id": trial_id,
                "item_id": trial.get("item_id", pd.NA),
                "stimulus_id": trial.get("stimulus_id", pd.NA),
                "condition_id": trial.get("condition_id", pd.NA),
                "aoi_id": aoi_id,
                "trial_duration_ms": duration_ms,
                "n_gaze_samples": n_samples,
                "valid_data_fraction": valid_fraction,
                "fixation_count": count,
                "dwell_time_ms": dwell,
                "mean_fixation_duration_ms": mean_duration,
                "first_fixation_latency_ms": first_latency,
                "ttff_ms": first_latency,
                "ttff_event_observed": bool(len(target)) if observed_trial else pd.NA,
                "ttff_censor_time_ms": duration_ms if observed_trial else np.nan,
                "revisits": revisits,
                "transition_count_from_aoi": float(transitions_from) if observed_trial else np.nan,
                "transition_count_to_aoi": float(transitions_to) if observed_trial else np.nan,
                "scanpath_sequence": " > ".join(collapsed),
                "pupil_within_fixation_mean": _pupil_within_fixations(branch, target, rec, trial_id),
                "feature_review_required": bool(np.isfinite(valid_fraction) and valid_fraction < 0.5),
            }
            row.update(_lineage_fields(branch, spec))
            rows.append(row)
    return pd.DataFrame(rows)


def propagate_detector_to_features(
    x: DetectorMultiverseResult,
    *,
    continue_on_error: bool = True,
) -> DetectorMultiverseResult:
    """Recompute AOI features for every successful detector branch, including zero-event trials."""
    if not isinstance(x, DetectorMultiverseResult):
        raise EyeProcessValidationError("`x` must be a DetectorMultiverseResult.")
    frames = []
    failures = [] if x.failures.empty else x.failures.to_dict("records")
    for spec in x.multiverse.specs:
        branch = x.branches.get(spec.detector_id)
        if branch is None:
            continue
        try:
            frames.append(_derive_branch_features(branch, spec))
        except Exception as exc:
            failures.append({
                "detector_id": spec.detector_id,
                "detector_spec_hash": spec.fingerprint,
                "stage": "feature_propagation",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if not continue_on_error:
                raise
    features = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return replace(x, features=features, failures=pd.DataFrame(failures))


def _fit_statsmodels(data: pd.DataFrame, model_spec: Mapping[str, Any]) -> tuple[Any, list[str]]:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise EyeProcessBackendError("This model engine requires statsmodels; no replacement estimator is selected automatically.") from exc
    formula = model_spec.get("formula")
    if not isinstance(formula, str) or "~" not in formula:
        raise EyeProcessValidationError("`model_spec['formula']` must be an explicit formula string.")
    engine = model_spec.get("engine")
    captured = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if engine == "statsmodels_ols":
            fit = smf.ols(formula, data=data, missing="raise").fit()
            converged = True
        elif engine == "statsmodels_mixedlm":
            groups = model_spec.get("groups")
            if not groups or groups not in data.columns:
                raise EyeProcessValidationError("MixedLM requires an explicit existing `groups` column.")
            re_formula = model_spec.get("re_formula")
            model = smf.mixedlm(formula, data=data, groups=data[groups], re_formula=re_formula, missing="raise")
            fit = model.fit(
                reml=bool(model_spec.get("reml", False)),
                method=model_spec.get("method", "lbfgs"),
                maxiter=int(model_spec.get("maxiter", 200)),
                disp=False,
            )
            converged = bool(getattr(fit, "converged", False))
        else:
            raise EyeProcessValidationError("Unknown statsmodels engine.")
        captured = [str(item.message) for item in caught]
    return (fit, captured if converged else captured + ["Model did not converge; estimates are retained for diagnosis but excluded from stability summaries."])


def _tidy_statsmodels(fit: Any, converged: bool, n: int) -> pd.DataFrame:
    params = pd.Series(fit.params)
    se = pd.Series(fit.bse).reindex(params.index)
    p = pd.Series(getattr(fit, "pvalues", np.nan), index=params.index).reindex(params.index)
    try:
        conf = fit.conf_int().reindex(params.index)
        lower = conf.iloc[:, 0]
        upper = conf.iloc[:, 1]
    except Exception:
        lower = pd.Series(np.nan, index=params.index)
        upper = pd.Series(np.nan, index=params.index)
    return pd.DataFrame({
        "term": params.index.astype(str),
        "estimate": params.to_numpy(float),
        "SE": pd.to_numeric(se, errors="coerce").to_numpy(float),
        "CI_lower": pd.to_numeric(lower, errors="coerce").to_numpy(float),
        "CI_upper": pd.to_numeric(upper, errors="coerce").to_numpy(float),
        "p": pd.to_numeric(p, errors="coerce").to_numpy(float),
        "converged": bool(converged),
        "N": int(n),
    })


def run_detector_inference_multiverse(
    x: DetectorMultiverseResult,
    model_spec: Mapping[str, Any],
    *,
    model_callback: Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame] | None = None,
    minimum_valid_fraction: float | None = None,
) -> DetectorInferenceResult:
    """Fit the identical explicit model specification across detector branches."""
    if not isinstance(x, DetectorMultiverseResult):
        raise EyeProcessValidationError("`x` must be a DetectorMultiverseResult.")
    if x.features.empty:
        raise EyeProcessValidationError("Run `propagate_detector_to_features()` before inference propagation.")
    if not isinstance(model_spec, Mapping):
        raise EyeProcessValidationError("`model_spec` must be a mapping.")
    engine = model_spec.get("engine")
    if engine not in {"statsmodels_ols", "statsmodels_mixedlm", "callback"}:
        raise EyeProcessValidationError("Choose an explicit model engine: statsmodels_ols, statsmodels_mixedlm, or callback.")
    if engine == "callback" and not callable(model_callback):
        raise EyeProcessValidationError("`engine='callback'` requires a callable `model_callback`.")
    outcome = str(model_spec.get("outcome", "")).strip()
    if not outcome:
        formula = str(model_spec.get("formula", ""))
        outcome = formula.split("~", 1)[0].strip() if "~" in formula else ""
    if not outcome or outcome not in x.features.columns:
        raise EyeProcessValidationError("The model outcome must be named explicitly and exist in propagated features.")
    aoi_id = model_spec.get("aoi_id")
    rows = []
    failures = []
    warning_rows = []
    for spec in x.multiverse.specs:
        data = x.features[x.features["detector_id"].eq(spec.detector_id)].copy()
        if aoi_id is not None:
            data = data[data["aoi_id"].astype(str).eq(str(aoi_id))]
        if minimum_valid_fraction is not None:
            if not 0 <= float(minimum_valid_fraction) <= 1:
                raise EyeProcessValidationError("`minimum_valid_fraction` must lie in [0, 1].")
            data = data[pd.to_numeric(data["valid_data_fraction"], errors="coerce") >= float(minimum_valid_fraction)]
        data = data[np.isfinite(pd.to_numeric(data[outcome], errors="coerce"))].copy()
        if data.empty:
            failures.append({"detector_id": spec.detector_id, "stage": "model", "error_type": "NoModelData", "error": "No finite model rows remained for this detector."})
            continue
        try:
            if engine == "callback":
                tidy = model_callback(data.copy(), dict(model_spec))
                if not isinstance(tidy, pd.DataFrame):
                    raise EyeProcessValidationError("`model_callback` must return a DataFrame.")
                required = {"term", "estimate", "SE", "CI_lower", "CI_upper", "p", "converged", "N"}
                missing = required - set(tidy.columns)
                if missing:
                    raise EyeProcessValidationError("Model callback output is missing: " + ", ".join(sorted(missing)))
                model_warnings: list[str] = []
            else:
                fit, model_warnings = _fit_statsmodels(data, model_spec)
                converged = bool(getattr(fit, "converged", True)) if engine == "statsmodels_mixedlm" else True
                tidy = _tidy_statsmodels(fit, converged, len(data))
            tidy = tidy.copy()
            tidy["detector_id"] = spec.detector_id
            tidy["detector_algorithm"] = spec.algorithm
            tidy["detector_spec_hash"] = spec.fingerprint
            tidy["parameter_spec"] = json.dumps(_jsonable(spec.as_dict()), sort_keys=True)
            tidy["model_engine"] = engine
            tidy["model_formula"] = model_spec.get("formula", pd.NA)
            model_spec_blob = json.dumps(_jsonable(dict(model_spec)), sort_keys=True, separators=(",", ":"))
            tidy["model_spec_hash"] = sha256(model_spec_blob.encode()).hexdigest()
            tidy["feature_fingerprint"] = sha256(data.to_json(orient="split", index=False, default_handler=str).encode()).hexdigest()
            tidy["software"] = "eyeprocesspy"
            tidy["software_version"] = "0.1.0"
            tidy["warnings"] = " | ".join(model_warnings) if model_warnings else pd.NA
            rows.append(tidy)
            for message in model_warnings:
                warning_rows.append({"detector_id": spec.detector_id, "stage": "model", "warning": message})
        except Exception as exc:
            failures.append({"detector_id": spec.detector_id, "stage": "model", "error_type": type(exc).__name__, "error": str(exc)})
    coefficients = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    feature_blob = x.features.to_json(orient="split", index=False, default_handler=str)
    return DetectorInferenceResult(
        multiverse=x.multiverse,
        coefficients=coefficients,
        failures=pd.DataFrame(failures),
        warnings=pd.DataFrame(warning_rows),
        model_spec=dict(model_spec),
        feature_fingerprint=sha256(feature_blob.encode()).hexdigest(),
    )


def assess_detector_inference_stability(
    x: DetectorInferenceResult,
    *,
    term: str,
    substantive_threshold: float | None = None,
    direction: str = "above",
) -> pd.DataFrame:
    """Summarize coefficient stability across converged detector specifications."""
    if not isinstance(x, DetectorInferenceResult):
        raise EyeProcessValidationError("`x` must be a DetectorInferenceResult.")
    if direction not in {"above", "below", "absolute"}:
        raise EyeProcessValidationError("`direction` must be above, below, or absolute.")
    data = x.coefficients[x.coefficients["term"].astype(str).eq(str(term))].copy()
    all_n = len(data)
    if data.empty:
        return pd.DataFrame([{
            "term": term, "specifications": 0, "converged_specifications": 0, "convergence_rate": np.nan,
            "median_estimate": np.nan, "estimate_min": np.nan, "estimate_max": np.nan, "estimate_range": np.nan,
            "same_sign_proportion": np.nan, "ci_overlap": pd.NA, "ci_overlap_lower": np.nan, "ci_overlap_upper": np.nan,
            "substantive_conclusion_stability": np.nan,
        }])
    converged_mask = data["converged"].astype("boolean").fillna(False)
    valid = data[converged_mask].copy()
    estimates = pd.to_numeric(valid["estimate"], errors="coerce")
    valid = valid[np.isfinite(estimates)]
    estimates = pd.to_numeric(valid["estimate"], errors="coerce")
    if valid.empty:
        return pd.DataFrame([{
            "term": term, "specifications": all_n, "converged_specifications": 0, "convergence_rate": 0.0,
            "median_estimate": np.nan, "estimate_min": np.nan, "estimate_max": np.nan, "estimate_range": np.nan,
            "same_sign_proportion": np.nan, "ci_overlap": pd.NA, "ci_overlap_lower": np.nan, "ci_overlap_upper": np.nan,
            "substantive_conclusion_stability": np.nan,
        }])
    nonzero = estimates[estimates != 0]
    if len(nonzero):
        majority_positive = float((nonzero > 0).mean()) >= 0.5
        same_sign = float((nonzero > 0).mean() if majority_positive else (nonzero < 0).mean())
    else:
        same_sign = np.nan
    lowers = pd.to_numeric(valid["CI_lower"], errors="coerce")
    uppers = pd.to_numeric(valid["CI_upper"], errors="coerce")
    finite_ci = np.isfinite(lowers) & np.isfinite(uppers)
    if finite_ci.any():
        overlap_lower = float(lowers[finite_ci].max())
        overlap_upper = float(uppers[finite_ci].min())
        ci_overlap: Any = bool(overlap_lower <= overlap_upper)
    else:
        overlap_lower = overlap_upper = np.nan
        ci_overlap = pd.NA
    threshold_stability = np.nan
    if substantive_threshold is not None:
        threshold = float(substantive_threshold)
        if direction == "above":
            decisions = estimates >= threshold
        elif direction == "below":
            decisions = estimates <= threshold
        else:
            decisions = estimates.abs() >= abs(threshold)
        threshold_stability = float(max(decisions.mean(), (~decisions).mean()))
    return pd.DataFrame([{
        "term": term,
        "specifications": all_n,
        "converged_specifications": len(valid),
        "convergence_rate": len(valid) / all_n if all_n else np.nan,
        "median_estimate": float(estimates.median()),
        "estimate_min": float(estimates.min()),
        "estimate_max": float(estimates.max()),
        "estimate_range": float(estimates.max() - estimates.min()),
        "same_sign_proportion": same_sign,
        "ci_overlap": ci_overlap,
        "ci_overlap_lower": overlap_lower,
        "ci_overlap_upper": overlap_upper,
        "substantive_conclusion_stability": threshold_stability,
    }])


def _feature_sensitivity(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    metrics = [
        "dwell_time_ms", "fixation_count", "mean_fixation_duration_ms", "ttff_ms", "revisits",
        "transition_count_from_aoi", "transition_count_to_aoi", "pupil_within_fixation_mean",
    ]
    keys = ["recording_id", "trial_id", "aoi_id"]
    rows = []
    for metric in metrics:
        if metric not in features:
            continue
        spreads = []
        for _, group in features.groupby(keys, dropna=False, sort=False):
            values = pd.to_numeric(group[metric], errors="coerce")
            values = values[np.isfinite(values)]
            if len(values) >= 2:
                spreads.append(float(values.max() - values.min()))
        rows.append({
            "feature": metric,
            "units_with_multiple_detectors": len(spreads),
            "median_detector_range": float(np.median(spreads)) if spreads else np.nan,
            "max_detector_range": float(np.max(spreads)) if spreads else np.nan,
        })
    return pd.DataFrame(rows)


def summarise_detector_robustness(
    x: DetectorMultiverseResult,
    *,
    inference: DetectorInferenceResult | None = None,
    term: str | None = None,
    substantive_threshold: float | None = None,
    direction: str = "above",
) -> dict[str, pd.DataFrame]:
    """Return event-, feature-, and inference-level robustness summaries."""
    out = {
        "event_summary": summarise_detector_events(x),
        "event_agreement": estimate_detector_agreement(x) if not x.events.empty else pd.DataFrame(),
        "feature_sensitivity": _feature_sensitivity(x.features),
        "failures": x.failures.copy(),
    }
    if inference is not None:
        if term is None:
            raise EyeProcessValidationError("`term` is required when inference results are supplied.")
        out["inference_stability"] = assess_detector_inference_stability(
            inference, term=term, substantive_threshold=substantive_threshold, direction=direction
        )
        out["model_failures"] = inference.failures.copy()
    else:
        out["inference_stability"] = pd.DataFrame()
        out["model_failures"] = pd.DataFrame()
    return out


def plot_detector_event_timeline(
    x: DetectorMultiverseResult | pd.DataFrame,
    *,
    trial_id: str | None = None,
    event_type: str = "fixation",
    ax: Any = None,
):
    """Plot detected event intervals by detector."""
    import matplotlib.pyplot as plt

    events = x.events if isinstance(x, DetectorMultiverseResult) else x
    data = events[events["episode_type"].eq(event_type)].copy()
    if trial_id is not None:
        data = data[data["trial_id"].astype(str).eq(str(trial_id))]
    if ax is None:
        _, ax = plt.subplots()
    ids = sorted(data["detector_id"].dropna().astype(str).unique()) if not data.empty else []
    for y, detector_id in enumerate(ids):
        subset = data[data["detector_id"].astype(str).eq(detector_id)]
        for _, row in subset.iterrows():
            ax.plot([row["start_time"], row["end_time"]], [y, y], linewidth=4)
    ax.set_yticks(range(len(ids)), ids)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Detector")
    ax.set_title(f"{event_type.title()} event timeline")
    return ax


def plot_detector_agreement(x: DetectorMultiverseResult, *, metric: str = "mean_event_overlap", ax: Any = None):
    """Plot pairwise detector agreement as a matrix."""
    import matplotlib.pyplot as plt

    agreement = estimate_detector_agreement(x)
    if metric not in agreement.columns and not agreement.empty:
        raise EyeProcessValidationError(f"Unknown agreement metric: {metric}")
    ids = sorted(set(agreement.get("detector_a", [])) | set(agreement.get("detector_b", [])))
    matrix = np.full((len(ids), len(ids)), np.nan)
    np.fill_diagonal(matrix, 1.0)
    lookup = {name: i for i, name in enumerate(ids)}
    for _, row in agreement.iterrows():
        i, j = lookup[row["detector_a"]], lookup[row["detector_b"]]
        matrix[i, j] = matrix[j, i] = row[metric]
    if ax is None:
        _, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
    ax.set_yticks(range(len(ids)), ids)
    ax.set_title(f"Detector agreement: {metric}")
    ax.figure.colorbar(image, ax=ax)
    return ax


def plot_detector_feature_distributions(
    x: DetectorMultiverseResult,
    *,
    feature: str = "dwell_time_ms",
    aoi_id: str | None = None,
    ax: Any = None,
):
    """Plot feature distributions by detector."""
    import matplotlib.pyplot as plt

    data = x.features.copy()
    if data.empty or feature not in data:
        raise EyeProcessValidationError("Requested propagated feature is unavailable.")
    if aoi_id is not None:
        data = data[data["aoi_id"].astype(str).eq(str(aoi_id))]
    ids = sorted(data["detector_id"].dropna().astype(str).unique())
    values = [pd.to_numeric(data.loc[data["detector_id"].astype(str).eq(detector_id), feature], errors="coerce").dropna() for detector_id in ids]
    if ax is None:
        _, ax = plt.subplots()
    ax.boxplot(values, tick_labels=ids)
    ax.set_ylabel(feature)
    ax.set_xlabel("Detector")
    ax.set_title(f"Detector sensitivity of {feature}")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_detector_coefficient_stability(
    x: DetectorInferenceResult,
    *,
    term: str,
    ax: Any = None,
):
    """Plot coefficient estimates and confidence intervals by detector."""
    import matplotlib.pyplot as plt

    data = x.coefficients[x.coefficients["term"].astype(str).eq(str(term))].copy()
    if data.empty:
        raise EyeProcessValidationError("Requested model term is unavailable.")
    if ax is None:
        _, ax = plt.subplots()
    data = data.sort_values("estimate")
    y = np.arange(len(data))
    est = pd.to_numeric(data["estimate"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(data["CI_lower"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(data["CI_upper"], errors="coerce").to_numpy(float)
    ax.errorbar(est, y, xerr=np.vstack([est - low, high - est]), fmt="o")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, data["detector_id"].astype(str))
    ax.set_xlabel("Estimate (95% CI)")
    ax.set_ylabel("Detector")
    ax.set_title(f"Coefficient stability: {term}")
    return ax


def plot_detector_multiverse(
    x: DetectorMultiverseResult,
    *,
    inference: DetectorInferenceResult | None = None,
    term: str | None = None,
    feature: str = "dwell_time_ms",
    aoi_id: str | None = None,
):
    """Return a dictionary of separate detector-multiverse diagnostic figures."""
    plots = {
        "agreement": plot_detector_agreement(x).figure,
        "feature": plot_detector_feature_distributions(x, feature=feature, aoi_id=aoi_id).figure,
    }
    if inference is not None and term is not None:
        plots["coefficient"] = plot_detector_coefficient_stability(inference, term=term).figure
    return plots


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    cols = [str(c) for c in frame.columns]
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).replace("|", "\\|").replace("\n", " ")
        return text
    header = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(cell(row[c]) for c in frame.columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, rule, *rows])


def report_detector_multiverse(
    x: DetectorMultiverseResult,
    *,
    inference: DetectorInferenceResult | None = None,
    term: str | None = None,
    substantive_threshold: float | None = None,
    path: str | None = None,
) -> str:
    """Create a compact reproducible Markdown detector-sensitivity report."""
    summary = summarise_detector_robustness(
        x,
        inference=inference,
        term=term,
        substantive_threshold=substantive_threshold,
    )
    manifest = x.multiverse.manifest
    lines = [
        "# Event-detector multiverse report",
        "",
        "## Scope",
        "",
        "This report evaluates whether events, AOI features, and statistical conclusions change across the supplied defensible detector specifications. The specification set is not evidence that omitted detector choices are valid or irrelevant.",
        "",
        "## Detector specifications",
        "",
        _markdown_table(manifest),
        "",
        "## Event-level sensitivity",
        "",
        _markdown_table(summary["event_summary"]) if not summary["event_summary"].empty else "No successful event catalogues were available.",
        "",
        "## AOI-feature sensitivity",
        "",
        _markdown_table(summary["feature_sensitivity"]) if not summary["feature_sensitivity"].empty else "AOI features were not propagated or no cross-detector comparison was estimable.",
        "",
    ]
    if inference is not None and term is not None:
        lines.extend([
            "## Inference stability",
            "",
            _markdown_table(summary["inference_stability"]),
            "",
            "Non-converged model branches are retained as diagnostic failures and are excluded from coefficient-stability calculations.",
            "",
        ])
    if not x.failures.empty:
        lines.extend(["## Branch failures", "", _markdown_table(x.failures), ""])
    lines.extend([
        "## Reporting guidance",
        "",
        "Report the detector family and parameters, sampling rate and coordinate units, AOI assignment rule, number of successful/failed specifications, event-level agreement, the range of key AOI features, coefficient distributions with uncertainty, convergence failures, and any substantive threshold used. Do not summarize robustness by counting p-values alone.",
        "",
        "## Limitations",
        "",
        "Detector sensitivity is conditional on the supplied preprocessing, AOIs, quality rules, model specification, and detector set. Agreement between detectors does not establish event validity, and disagreement does not identify which detector is correct without external evidence.",
    ])
    text = "\n".join(lines)
    if path is not None:
        from pathlib import Path
        Path(path).write_text(text, encoding="utf-8")
    return text


def simulate_detector_multiverse_data(
    *,
    n_participants: int = 12,
    sampling_rate: float = 60.0,
    trial_duration_s: float = 2.0,
    seed: int = 20260918,
) -> EyeDataset:
    """Create a small synthetic 60-Hz dataset with a known disclosure-dwell effect."""
    from .dataset import new_eye_dataset
    from .schema import new_coordinate_space
    from .foundation_09 import new_aoi, register_aois

    if int(n_participants) < 4:
        raise EyeProcessValidationError("Use at least four participants for the worked multiverse example.")
    _finite_positive(sampling_rate, "sampling_rate", allow_none=False)
    _finite_positive(trial_duration_s, "trial_duration_s", allow_none=False)
    rng = np.random.default_rng(int(seed))
    recordings = []
    gaze_rows = []
    intervals = []
    sample_counter = 0
    n_samples = int(round(float(sampling_rate) * float(trial_duration_s)))
    for participant in range(1, int(n_participants) + 1):
        rec = f"R{participant:03d}"
        recordings.append({"recording_id": rec, "participant_id": f"P{participant:03d}", "vendor": "synthetic", "nominal_sampling_rate": sampling_rate})
        for condition_index, condition in enumerate(("control", "disclosure")):
            trial_id = f"{rec}_T{condition_index + 1}"
            start = condition_index * (trial_duration_s + 0.25)
            end = start + trial_duration_s
            intervals.append({
                "interval_id": f"I_{trial_id}", "recording_id": rec, "interval_type": "trial",
                "start_time": start, "end_time": end, "trial_id": trial_id,
                "participant_id": f"P{participant:03d}", "item_id": "ad_01", "stimulus_id": "stim_01",
                "condition_id": condition, "valid_interval": True,
            })
            local_t = np.arange(n_samples) / sampling_rate
            # Piecewise fixation centres; disclosure condition has a longer target-AOI segment.
            target_start = 0.55 + rng.normal(0, 0.03)
            target_end = (1.42 if condition == "disclosure" else 1.02) + rng.normal(0, 0.04)
            x = np.where((local_t >= target_start) & (local_t <= target_end), 6.0, 2.1)
            y = np.where((local_t >= target_start) & (local_t <= target_end), 2.4, 5.8)
            x = x + rng.normal(0, 0.10, n_samples)
            y = y + rng.normal(0, 0.10, n_samples)
            # Add a brief revisit in a subset of disclosure trials.
            if condition == "disclosure" and participant % 3 == 0:
                revisit = (local_t >= 1.62) & (local_t <= 1.82)
                x[revisit] = 6.1 + rng.normal(0, 0.10, revisit.sum())
                y[revisit] = 2.3 + rng.normal(0, 0.10, revisit.sum())
            valid = rng.random(n_samples) > 0.015
            for k in range(n_samples):
                sample_counter += 1
                gaze_rows.append({
                    "recording_id": rec, "sample_id": f"S{sample_counter:08d}",
                    "timestamp_seconds": start + local_t[k], "gaze_x": x[k], "gaze_y": y[k],
                    "valid": bool(valid[k]), "trial_id": trial_id, "stimulus_id": "stim_01",
                    "coordinate_space_id": "deg_display",
                })
    spaces = new_coordinate_space(
        coordinate_space_id="deg_display",
        space_type="custom",
        width=12,
        height=8,
        origin="center",
        x_unit="degrees",
        y_unit="degrees",
    )
    dataset = new_eye_dataset(
        recordings=pd.DataFrame(recordings),
        gaze_samples=pd.DataFrame(gaze_rows),
        intervals=pd.DataFrame(intervals),
        coordinate_spaces=spaces,
        validate=False,
    )
    dataset = register_aois(
        dataset,
        new_aoi(
            "disclosure", "Disclosure", "stim_01", "rectangle",
            x=4.0, y=1.0, width=4.0, height=3.0, coordinate_space_id="deg_display",
        ),
        new_aoi(
            "main_content", "Main content", "stim_01", "rectangle",
            x=0.5, y=4.5, width=3.5, height=2.5, coordinate_space_id="deg_display",
        ),
    )
    return add_provenance(
        dataset,
        "simulate_detector_multiverse_data",
        "dataset",
        f"n_participants={n_participants};sampling_rate={sampling_rate};seed={seed}",
    )


__all__ = [
    "EventDetectorSpec",
    "DetectorMultiverse",
    "DetectorMultiverseResult",
    "DetectorInferenceResult",
    "define_event_detector_spec",
    "validate_event_detector_spec",
    "create_detector_multiverse",
    "detect_events_with_spec",
    "import_external_detector_events",
    "run_detector_multiverse",
    "compare_event_catalogues",
    "match_detected_events",
    "estimate_detector_agreement",
    "summarise_detector_events",
    "summarise_detector_disagreement",
    "propagate_detector_to_aoi",
    "propagate_detector_to_features",
    "run_detector_inference_multiverse",
    "assess_detector_inference_stability",
    "summarise_detector_robustness",
    "plot_detector_event_timeline",
    "plot_detector_agreement",
    "plot_detector_feature_distributions",
    "plot_detector_coefficient_stability",
    "plot_detector_multiverse",
    "report_detector_multiverse",
    "simulate_detector_multiverse_data",
]
