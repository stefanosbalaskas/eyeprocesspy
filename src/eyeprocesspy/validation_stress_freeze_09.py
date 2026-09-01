"""Frozen R/091 validation stress/freeze contracts.

Source reference:
``R/091-validation-stress-freeze-0-9.R`` from eyeprocess 0.11.1.

The source algorithms for stress-plan expansion, stress execution, claim
matrices, integrity hashing, readiness, and release gating are ported directly.
Native RDS persistence is intentionally not emulated. Python persistence uses a
deterministic JSON representation and rejects ``.rds`` paths explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .reproducibility_provenance_09 import object_hash
from .validation_evidence_programs_09 import eyeprocess_validation_seed

__all__ = [
    "expand_eyeprocess_stress_evidence_plan",
    "eyeprocess_negative_control_evidence_plan",
    "eyeprocess_reliability_evidence_plan",
    "eyeprocess_stress_evidence_plan",
    "eyeprocess_validation_claim_matrix",
    "eyeprocess_validation_evidence_manifest",
    "eyeprocess_validation_readiness",
    "eyeprocess_validation_release_gate",
    "freeze_eyeprocess_validation_evidence",
    "read_eyeprocess_validation_evidence",
    "run_eyeprocess_stress_evidence",
    "summarise_eyeprocess_stress_evidence",
    "verify_eyeprocess_validation_evidence",
    "write_eyeprocess_validation_evidence",
]

_STRESS_PLAN_CLASS = "eye_stress_evidence_plan"
_RELIABILITY_PLAN_CLASS = "eye_reliability_evidence_plan"
_NEGATIVE_PLAN_CLASS = "eye_negative_control_evidence_plan"
_MANIFEST_CLASS = "eye_validation_evidence_manifest"
_FREEZE_CLASS = "eye_validation_evidence_freeze"
_READINESS_CLASS = "eye_validation_readiness"
_RELEASE_GATE_CLASS = "eye_validation_release_gate"
_STRESS_RESULT_CLASS = "eye_stress_evidence_result"

_STRESS_FIELDS = (
    "missing_gaze",
    "pupil_dropout",
    "calibration_offset",
    "sampling_jitter",
    "aoi_label_noise",
    "device_shift",
    "trial_imbalance",
)
_PROPORTION_FIELDS = {
    "missing_gaze",
    "pupil_dropout",
    "aoi_label_noise",
    "trial_imbalance",
}
_RELIABILITY_METRICS = (
    "split_half",
    "icc",
    "temporal_stability",
    "bland_altman",
)
_NEGATIVE_CONTROLS = (
    "permutation",
    "temporal_shift",
    "placebo_window",
    "known_leakage",
)
_CLAIM_STATUS = (
    "qualified",
    "supported",
    "pending",
    "not_supported",
)


def _tag(value: dict[str, Any], class_name: str) -> dict[str, Any]:
    value["eyeprocess_class"] = class_name
    return value


def _class_is(value: Any, class_name: str) -> bool:
    return isinstance(value, Mapping) and value.get("eyeprocess_class") == class_name


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)):
        return [value]
    if isinstance(value, pd.Series):
        return value.tolist()
    if np.isscalar(value):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _unique(values: Sequence[Any]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _numeric_vector(value: Any, *, name: str) -> list[float]:
    raw = pd.to_numeric(
        pd.Series(_as_list(value)),
        errors="coerce",
    ).to_numpy(dtype=float)
    if len(raw) == 0 or np.any(~np.isfinite(raw)) or np.any(raw < 0):
        raise EyeProcessValidationError(f"{name} must contain finite non-negative values.")
    return _unique([float(item) for item in raw])


def _now_utc_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "__eyeprocess_type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "index": [_clean_scalar(item) for item in value.index.tolist()],
            "records": [
                {str(key): _jsonify(item) for key, item in row.items()} for row in value.to_dict(orient="records")
            ],
            "attrs": _jsonify(dict(value.attrs)),
        }
    if isinstance(value, pd.Series):
        return {
            "__eyeprocess_type__": "series",
            "name": _clean_scalar(value.name),
            "index": [_clean_scalar(item) for item in value.index.tolist()],
            "data": [_jsonify(item) for item in value.tolist()],
        }
    if isinstance(value, np.ndarray):
        return {
            "__eyeprocess_type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _jsonify(value.tolist()),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    cleaned = _clean_scalar(value)
    if cleaned is None or isinstance(
        cleaned,
        (str, int, float, bool),
    ):
        return cleaned
    return {
        "__eyeprocess_type__": (f"{type(value).__module__}.{type(value).__qualname__}"),
        "repr": repr(value),
    }


def _restore_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json(item) for item in value]
    if not isinstance(value, Mapping):
        return value

    marker = value.get("__eyeprocess_type__")
    if marker == "dataframe":
        frame = pd.DataFrame(
            [{key: _restore_json(item) for key, item in row.items()} for row in value.get("records", [])],
            columns=value.get("columns", []),
        )
        frame.index = value.get(
            "index",
            list(range(len(frame))),
        )
        frame.attrs.update(_restore_json(value.get("attrs", {})))
        return frame
    if marker == "series":
        return pd.Series(
            [_restore_json(item) for item in value.get("data", [])],
            index=value.get("index"),
            name=value.get("name"),
        )
    if marker == "ndarray":
        array = np.asarray(_restore_json(value.get("data", [])))
        shape = tuple(value.get("shape", array.shape))
        try:
            return array.reshape(shape)
        except ValueError:
            return array

    return {key: _restore_json(item) for key, item in value.items()}


def eyeprocess_stress_evidence_plan(
    missing_gaze=(0, 0.05, 0.15, 0.30),
    pupil_dropout=(0, 0.05, 0.15),
    calibration_offset=(0, 0.01, 0.03, 0.06),
    sampling_jitter=(0, 0.05, 0.15),
    aoi_label_noise=(0, 0.02, 0.10),
    device_shift=(0, 0.02, 0.05),
    trial_imbalance=(0, 0.10, 0.25),
    seed=20260811,
):
    """Declare the frozen one-factor-at-a-time stress-evidence plan."""
    values = {
        "missing_gaze": missing_gaze,
        "pupil_dropout": pupil_dropout,
        "calibration_offset": calibration_offset,
        "sampling_jitter": sampling_jitter,
        "aoi_label_noise": aoi_label_noise,
        "device_shift": device_shift,
        "trial_imbalance": trial_imbalance,
    }

    output: dict[str, Any] = {}
    for name, value in values.items():
        vector = _numeric_vector(value, name=name)
        if name in _PROPORTION_FIELDS and any(item >= 1 for item in vector):
            raise EyeProcessValidationError(f"{name} must lie in [0,1).")
        output[name] = vector

    try:
        seed0 = int(seed)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("seed must be a positive scalar integer.") from exc
    if seed0 < 1:
        raise EyeProcessValidationError("seed must be a positive scalar integer.")

    output["seed"] = seed0
    return _tag(output, _STRESS_PLAN_CLASS)


def expand_eyeprocess_stress_evidence_plan(plan):
    """Expand a stress plan in frozen R family/value order."""
    if not _class_is(plan, _STRESS_PLAN_CLASS):
        raise EyeProcessValidationError("plan must be an eye_stress_evidence_plan.")

    rows = []
    index = 0
    for corruption in _STRESS_FIELDS:
        for severity in plan[corruption]:
            index += 1
            rows.append(
                {
                    "scenario_id": f"STRESS{index:03d}",
                    "corruption": corruption,
                    "severity": float(severity),
                    "seed": eyeprocess_validation_seed(
                        plan["seed"],
                        index,
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "scenario_id",
            "corruption",
            "severity",
            "seed",
        ],
    )


def eyeprocess_reliability_evidence_plan(
    metrics=_RELIABILITY_METRICS,
    bootstrap=200,
    seed=20260811,
):
    """Declare frozen reliability-evidence targets."""
    metrics0 = _unique([str(value) for value in _as_list(metrics)])
    if not metrics0 or any(value not in _RELIABILITY_METRICS for value in metrics0):
        raise EyeProcessValidationError("unsupported reliability metric.")
    try:
        bootstrap0 = int(bootstrap)
        seed0 = int(seed)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("invalid bootstrap/seed.") from exc
    if bootstrap0 < 0 or seed0 < 1:
        raise EyeProcessValidationError("invalid bootstrap/seed.")
    return _tag(
        {
            "metrics": metrics0,
            "bootstrap": bootstrap0,
            "seed": seed0,
            "guardrail": ("Reliability is repeatability evidence, not construct validity."),
        },
        _RELIABILITY_PLAN_CLASS,
    )


def eyeprocess_negative_control_evidence_plan(
    controls=_NEGATIVE_CONTROLS,
    replications=100,
    seed=20260811,
):
    """Declare frozen negative-control evidence targets."""
    controls0 = _unique([str(value) for value in _as_list(controls)])
    if not controls0 or any(value not in _NEGATIVE_CONTROLS for value in controls0):
        raise EyeProcessValidationError("unsupported negative control.")
    try:
        replications0 = int(replications)
        seed0 = int(seed)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("replications/seed must be positive scalar integers.") from exc
    if replications0 < 1 or seed0 < 1:
        raise EyeProcessValidationError("replications/seed must be positive scalar integers.")
    return _tag(
        {
            "controls": controls0,
            "replications": replications0,
            "seed": seed0,
            "guardrail": (
                "Negative controls diagnose analysis behavior; they do not label analyst conduct or participant state."
            ),
        },
        _NEGATIVE_PLAN_CLASS,
    )


def _recycle(value: Any, n: int) -> list[Any]:
    values = _as_list(value)
    return [values[index % len(values)] for index in range(n)]


def _as_character_values(value: Any, n: int) -> list[str | None]:
    output: list[str | None] = []
    for item in _recycle(value, n):
        if item is None or item is pd.NA:
            output.append(None)
            continue
        if isinstance(item, (float, np.floating)) and np.isnan(item):
            output.append(None)
            continue
        output.append(str(item))
    return output


def eyeprocess_validation_claim_matrix(
    claim_id,
    claim,
    evidence_id,
    evidence_type,
    status="qualified",
    boundary=None,
):
    """Build the frozen machine-readable claim/evidence matrix."""
    fields = {
        "claim_id": claim_id,
        "claim": claim,
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "status": status,
        "boundary": boundary,
    }
    lengths = {name: len(_as_list(value)) for name, value in fields.items()}
    if boundary is None:
        lengths["boundary"] = 1

    if any(length == 0 for length in lengths.values()):
        raise EyeProcessValidationError("claim/evidence fields must be non-empty.")

    n = max(lengths.values())
    if any(length not in {1, n} for length in lengths.values()):
        raise EyeProcessValidationError("claim/evidence fields must have length 1 or a common maximum length.")

    status0 = [str(value) for value in _recycle(status, n)]
    if any(value not in _CLAIM_STATUS for value in status0):
        raise EyeProcessValidationError("invalid claim status.")

    boundary_values = [None] * n if boundary is None else _recycle(boundary, n)
    output = pd.DataFrame(
        {
            "claim_id": _as_character_values(claim_id, n),
            "claim": _as_character_values(claim, n),
            "evidence_id": _as_character_values(evidence_id, n),
            "evidence_type": _as_character_values(evidence_type, n),
            "status": status0,
            "boundary": boundary_values,
        }
    )

    ids = output["claim_id"]
    if ids.isna().any() or ids.fillna("").str.len().eq(0).any() or ids.duplicated().any():
        raise EyeProcessValidationError("claim_id must be unique, non-missing, and non-empty.")

    for name in ("claim", "evidence_id", "evidence_type"):
        values = output[name]
        if values.isna().any() or values.fillna("").str.len().eq(0).any():
            raise EyeProcessValidationError("claim, evidence_id, and evidence_type must be non-missing and non-empty.")

    return output


def eyeprocess_validation_evidence_manifest(
    files=(),
    objects=(),
    source_commit=None,
    label="eyeprocess-0.9-m2",
):
    """Create the frozen evidence manifest from files and objects."""
    paths = [Path(value) for value in _as_list(files)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise EyeProcessValidationError("Evidence files not found: " + ", ".join(missing))

    file_rows = []
    for path in paths:
        resolved = path.resolve(strict=True)
        file_rows.append(
            {
                "path": resolved.as_posix(),
                "md5": hashlib.md5(resolved.read_bytes()).hexdigest(),
            }
        )
    file_table = pd.DataFrame(
        file_rows,
        columns=["path", "md5"],
    )

    if isinstance(objects, Mapping):
        object_items = list(objects.items())
    else:
        object_values = _as_list(objects)
        object_items = [
            (f"object_{index}", value)
            for index, value in enumerate(
                object_values,
                start=1,
            )
        ]
    object_table = pd.DataFrame(
        [
            {
                "name": str(name),
                "hash": object_hash(value),
            }
            for name, value in object_items
        ],
        columns=["name", "hash"],
    )

    return _tag(
        {
            "label": str(label),
            "source_commit": (None if source_commit is None else str(source_commit)),
            "files": file_table,
            "objects": object_table,
            "generated_at": _now_utc_string(),
        },
        _MANIFEST_CLASS,
    )


def freeze_eyeprocess_validation_evidence(
    design,
    recovery=None,
    sbc=None,
    stress=None,
    reliability=None,
    negative_controls=None,
    irt=None,
    claims=None,
    provenance=None,
    source_commit=None,
):
    """Freeze a complete validation-evidence bundle with integrity hash."""
    components = {
        "design": design,
        "recovery": recovery,
        "sbc": sbc,
        "stress": stress,
        "reliability": reliability,
        "negative_controls": negative_controls,
        "irt": irt,
        "claims": claims,
        "provenance": provenance,
    }
    presence = {name: value is not None for name, value in components.items()}

    output = _tag(
        {
            "components": components,
            "presence": presence,
            "source_commit": (None if source_commit is None else str(source_commit)),
            "frozen_at": _now_utc_string(),
            "scientific_scope": (
                "software validation and measurement-behavior evidence; not construct-validity certification"
            ),
        },
        _FREEZE_CLASS,
    )
    output["hash"] = object_hash(output)
    return output


def verify_eyeprocess_validation_evidence(x):
    """Verify the integrity hash of a frozen evidence bundle."""
    if not _class_is(x, _FREEZE_CLASS):
        raise EyeProcessValidationError("x must be a frozen validation evidence bundle.")
    stored = x.get("hash")
    payload = {key: value for key, value in x.items() if key != "hash"}
    return stored == object_hash(payload)


def write_eyeprocess_validation_evidence(x, path):
    """Write frozen validation evidence as deterministic JSON."""
    if not _class_is(x, _FREEZE_CLASS):
        raise EyeProcessValidationError("x must be a frozen validation evidence bundle.")
    if not verify_eyeprocess_validation_evidence(x):
        raise EyeProcessValidationError("evidence hash verification failed before writing.")

    output = Path(path)
    if output.suffix.lower() == ".rds":
        raise EyeProcessValidationError(
            "Native RDS serialization is R-specific and is intentionally "
            "not emulated by eyeprocesspy. Use a .json output path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _jsonify(x),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output.resolve().as_posix()


def read_eyeprocess_validation_evidence(
    path,
    verify=True,
):
    """Read a Python-native frozen validation-evidence JSON file."""
    source = Path(path)
    if source.suffix.lower() == ".rds":
        raise EyeProcessValidationError(
            "Native RDS serialization is R-specific and is intentionally "
            "not emulated by eyeprocesspy. Read the JSON freeze instead."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EyeProcessValidationError("Could not read frozen validation evidence.") from exc

    output = _restore_json(payload)
    if not _class_is(output, _FREEZE_CLASS):
        raise EyeProcessValidationError("file is not an eyeprocess validation evidence freeze.")
    if verify and not verify_eyeprocess_validation_evidence(output):
        raise EyeProcessValidationError("frozen evidence hash verification failed.")
    return output


def eyeprocess_validation_readiness(
    x,
    required=(
        "design",
        "recovery",
        "stress",
        "reliability",
        "negative_controls",
        "claims",
        "provenance",
    ),
):
    """Evaluate readiness of a frozen validation-evidence bundle."""
    if not _class_is(x, _FREEZE_CLASS):
        raise EyeProcessValidationError("x must be a frozen validation evidence bundle.")
    required0 = _unique([str(value) for value in _as_list(required)])
    present = [bool(x["presence"].get(name, False)) for name in required0]
    table = pd.DataFrame(
        {
            "requirement": required0,
            "satisfied": present,
        }
    )
    return _tag(
        {
            "ready": bool(all(present)),
            "table": table,
            "hash_valid": verify_eyeprocess_validation_evidence(x),
            "source_commit": x.get("source_commit"),
        },
        _READINESS_CLASS,
    )


def eyeprocess_validation_release_gate(
    readiness,
    acceptance=None,
    require_hash=True,
):
    """Apply the frozen conservative software-release evidence gate."""
    if not _class_is(readiness, _READINESS_CLASS):
        raise EyeProcessValidationError("readiness must come from eyeprocess_validation_readiness().")

    acceptance_ok = True
    if acceptance is not None:
        if not isinstance(acceptance, pd.DataFrame):
            raise EyeProcessValidationError("acceptance must be a data.frame.")
        if "pass" not in acceptance.columns:
            raise EyeProcessValidationError("acceptance is missing required columns: pass")
        acceptance_ok = bool(
            len(acceptance) > 0 and acceptance["pass"].notna().all() and acceptance["pass"].astype(bool).all()
        )

    passed = bool(readiness["ready"] and acceptance_ok and (not bool(require_hash) or bool(readiness["hash_valid"])))
    return _tag(
        {
            "pass": passed,
            "readiness": bool(readiness["ready"]),
            "acceptance": acceptance_ok,
            "hash": bool(readiness["hash_valid"]),
            "interpretation": (
                "A passing gate supports software-release readiness only; it is not a scientific validity certificate."
            ),
        },
        _RELEASE_GATE_CLASS,
    )


def _coerce_metric_output(value: Any) -> dict[str, float]:
    if isinstance(value, pd.Series):
        items = list(value.items())
    elif isinstance(value, Mapping):
        items = list(value.items())
    else:
        array = np.asarray(value)
        names = getattr(value, "name", None)
        if names is None:
            raise EyeProcessValidationError("metric_fun must return a non-empty uniquely named vector.")
        items = [(str(names), array.item())]

    if not items or any(not str(name) for name, _ in items) or len({str(name) for name, _ in items}) != len(items):
        raise EyeProcessValidationError("metric_fun must return a non-empty uniquely named vector.")

    output: dict[str, float] = {}
    for name, raw in items:
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as exc:
            raise EyeProcessValidationError("metric_fun must return numeric values.") from exc
        if math.isinf(numeric):
            raise EyeProcessValidationError("metric_fun cannot return infinite values.")
        output[str(name)] = numeric
    return output


def run_eyeprocess_stress_evidence(
    data,
    plan,
    corruptors,
    metric_fun: Callable[[Any], Any],
):
    """Execute the frozen measurement-stress evidence programme."""
    if not _class_is(plan, _STRESS_PLAN_CLASS):
        raise EyeProcessValidationError("plan must be an eye_stress_evidence_plan.")
    if (
        not isinstance(corruptors, Mapping)
        or not corruptors
        or any(not str(name) for name in corruptors)
        or len(set(map(str, corruptors))) != len(corruptors)
        or any(not callable(fun) for fun in corruptors.values())
    ):
        raise EyeProcessValidationError("corruptors must be a uniquely named list of functions.")
    if not callable(metric_fun):
        raise EyeProcessValidationError("metric_fun must be a function.")

    scenarios = expand_eyeprocess_stress_evidence_plan(plan)
    missing = sorted(set(scenarios["corruption"]) - set(map(str, corruptors)))
    if missing:
        raise EyeProcessValidationError("Missing corruptors for: " + ", ".join(missing))

    baseline = _coerce_metric_output(metric_fun(data))

    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for _, scenario in scenarios.iterrows():
        corruption = str(scenario["corruption"])
        severity = float(scenario["severity"])
        seed = int(scenario["seed"])

        try:
            corrupted = corruptors[corruption](
                copy.deepcopy(data),
                severity,
                seed,
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "corruption": corruption,
                    "severity": severity,
                    "seed": seed,
                    "error": str(exc),
                }
            )
            continue

        try:
            value = _coerce_metric_output(metric_fun(corrupted))
        except Exception as exc:
            failure_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "corruption": corruption,
                    "severity": severity,
                    "seed": seed,
                    "error": str(exc),
                }
            )
            continue

        if set(value) != set(baseline):
            failure_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "corruption": corruption,
                    "severity": severity,
                    "seed": seed,
                    "error": ("metric_fun names changed after corruption"),
                }
            )
            continue

        value = {name: value[name] for name in baseline}

        for name, baseline_value in baseline.items():
            observed = value[name]
            finite_pair = np.isfinite(baseline_value) and np.isfinite(observed)
            delta = observed - baseline_value if finite_pair else np.nan
            relative_change = (
                (observed - baseline_value) / abs(baseline_value) if finite_pair and baseline_value != 0 else np.nan
            )
            result_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "corruption": corruption,
                    "severity": severity,
                    "seed": seed,
                    "metric": name,
                    "baseline": baseline_value,
                    "value": observed,
                    "delta": delta,
                    "relative_change": relative_change,
                }
            )

    results = pd.DataFrame(
        result_rows,
        columns=[
            "scenario_id",
            "corruption",
            "severity",
            "seed",
            "metric",
            "baseline",
            "value",
            "delta",
            "relative_change",
        ],
    )
    failures = pd.DataFrame(
        failure_rows,
        columns=[
            "scenario_id",
            "corruption",
            "severity",
            "seed",
            "error",
        ],
    )

    return _tag(
        {
            "plan": plan,
            "scenarios": scenarios,
            "baseline": baseline,
            "results": results,
            "failures": failures,
            "guardrail": (
                "Stress evidence describes software/measurement behavior "
                "under declared synthetic corruptions; thresholds remain "
                "study-specific."
            ),
        },
        _STRESS_RESULT_CLASS,
    )


def summarise_eyeprocess_stress_evidence(x):
    """Summarise executed measurement-stress evidence."""
    if not _class_is(x, _STRESS_RESULT_CLASS):
        raise EyeProcessValidationError("x must be an eye_stress_evidence_result.")
    results = x["results"]
    if not isinstance(results, pd.DataFrame) or results.empty:
        return pd.DataFrame()

    output = []
    for (corruption, metric), frame in results.groupby(
        ["corruption", "metric"],
        sort=True,
        dropna=False,
    ):
        finite_delta = pd.to_numeric(
            frame["delta"],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite_delta = finite_delta[np.isfinite(finite_delta)]
        output.append(
            {
                "corruption": corruption,
                "metric": metric,
                "n_scenarios": len(frame),
                "min_severity": float(frame["severity"].min()),
                "max_severity": float(frame["severity"].max()),
                "mean_delta": (float(np.mean(finite_delta)) if len(finite_delta) else np.nan),
                "max_abs_delta": (float(np.max(np.abs(finite_delta))) if len(finite_delta) else np.nan),
            }
        )
    return pd.DataFrame(output)
