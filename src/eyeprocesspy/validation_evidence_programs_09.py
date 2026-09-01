"""Frozen R/082 validation-evidence orchestration contracts.

Source reference:
``R/082-validation-evidence-programs-0-9.R`` from eyeprocess 0.11.1.

This module ports the deterministic validation-plan, acceptance-rule, Monte
Carlo uncertainty, scenario-manifest, and evidence-grade surface. Native RDS
manifest serialization is not emulated; JSON is the Python-native persistence
format and ``.rds`` paths are explicitly rejected.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .reproducibility_provenance_09 import object_hash

__all__ = [
    "evaluate_validation_acceptance",
    "expand_eyeprocess_validation_plan",
    "eyeprocess_validation_evidence_grade",
    "eyeprocess_validation_plan",
    "eyeprocess_validation_seed",
    "read_validation_scenario_manifest",
    "summarise_validation_acceptance",
    "validate_eyeprocess_validation_plan",
    "validation_acceptance_matrix",
    "validation_acceptance_rule",
    "validation_mcse_profile",
    "validation_replication_budget",
    "validation_scenario_manifest",
    "write_validation_scenario_manifest",
]

_PLAN_CLASS = "eye_validation_evidence_plan"
_RULE_CLASS = "eye_validation_acceptance_rule"
_MANIFEST_CLASS = "eye_validation_scenario_manifest"
_GRADE_CLASS = "eye_validation_evidence_grade"

_ALLOWED_FAMILIES = (
    "recovery",
    "sbc",
    "stress",
    "reliability",
    "negative_control",
)
_ALLOWED_SPECIFICATION = ("correct", "misspecified")
_ALLOWED_DIRECTIONS = ("max", "min", "between", "equals")


def _tag(value: dict[str, Any], class_name: str) -> dict[str, Any]:
    value["eyeprocess_class"] = class_name
    return value


def _class_is(value: Any, class_name: str) -> bool:
    if isinstance(value, Mapping):
        return value.get("eyeprocess_class") == class_name
    return getattr(value, "eyeprocess_class", None) == class_name


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
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
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _as_frame(value: Any, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise EyeProcessValidationError(f"{name} must be a data.frame.")
    return value.copy()


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required columns: " + ", ".join(missing))


def _finite_numbers(
    values: Any,
    *,
    integer: bool = False,
) -> list[float] | list[int]:
    out = pd.to_numeric(
        pd.Series(_sequence(values)),
        errors="coerce",
    ).to_numpy(dtype=float)
    if integer:
        return [int(value) for value in out]
    return [float(value) for value in out]


def _utc_string(value: Any | None = None) -> str:
    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)
    else:
        try:
            moment = pd.Timestamp(value)
        except Exception as exc:
            raise EyeProcessValidationError("generated_at must be datetime-like.") from exc
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        else:
            moment = moment.tz_convert("UTC")
        moment = moment.to_pydatetime()
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "__eyeprocess_type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "records": [
                {str(key): _json_safe(item) for key, item in row.items()} for row in value.to_dict(orient="records")
            ],
            "attrs": _json_safe(dict(value.attrs)),
        }
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_restore(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    if value.get("__eyeprocess_type__") == "dataframe":
        frame = pd.DataFrame(
            [{key: _json_restore(item) for key, item in row.items()} for row in value.get("records", [])],
            columns=value.get("columns", []),
        )
        frame.attrs.update(_json_restore(value.get("attrs", {})))
        return frame
    return {key: _json_restore(item) for key, item in value.items()}


def eyeprocess_validation_plan(
    families=_ALLOWED_FAMILIES,
    sample_size=(250, 750),
    n_items=(12, 24),
    missing_rate=(0.0, 0.15),
    noise_level=("reference", "elevated"),
    specification=_ALLOWED_SPECIFICATION,
    replications=20,
    seed=20260811,
    label="eyeprocess-0.9-m2",
):
    """Declare the frozen deterministic validation-evidence plan."""
    families0 = _unique([str(value) for value in _sequence(families)])
    if not families0 or any(value not in _ALLOWED_FAMILIES for value in families0):
        raise EyeProcessValidationError("families must be drawn from: " + ", ".join(_ALLOWED_FAMILIES))

    sample0 = _unique(_finite_numbers(sample_size, integer=True))
    if not sample0 or any(value < 20 for value in sample0):
        raise EyeProcessValidationError("sample_size must contain integers >= 20.")

    items0 = _unique(_finite_numbers(n_items, integer=True))
    if not items0 or any(value < 3 for value in items0):
        raise EyeProcessValidationError("n_items must contain integers >= 3.")

    missing0 = _unique(_finite_numbers(missing_rate))
    if not missing0 or any(not np.isfinite(value) or value < 0 or value >= 1 for value in missing0):
        raise EyeProcessValidationError("missing_rate must lie in [0, 1).")

    noise0 = _unique([str(value) for value in _sequence(noise_level)])
    if not noise0 or any(not value for value in noise0):
        raise EyeProcessValidationError("noise_level must be non-empty.")

    specification0 = _unique([str(value) for value in _sequence(specification)])
    if not specification0 or any(value not in _ALLOWED_SPECIFICATION for value in specification0):
        raise EyeProcessValidationError("specification must be 'correct' and/or 'misspecified'.")

    try:
        replications0 = int(replications)
        seed0 = int(seed)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("replications and seed must be scalar integers.") from exc

    if replications0 < 1:
        raise EyeProcessValidationError("replications must be >= 1.")
    if seed0 < 1:
        raise EyeProcessValidationError("seed must be a positive integer.")
    if not isinstance(label, str):
        raise EyeProcessValidationError("label must be character.")

    return _tag(
        {
            "families": families0,
            "sample_size": sample0,
            "n_items": items0,
            "missing_rate": missing0,
            "noise_level": noise0,
            "specification": specification0,
            "replications": replications0,
            "seed": seed0,
            "label": label,
        },
        _PLAN_CLASS,
    )


def validate_eyeprocess_validation_plan(x):
    """Validate the structural contract of a validation-evidence plan."""
    if not _class_is(x, _PLAN_CLASS):
        raise EyeProcessValidationError("x must inherit from eye_validation_evidence_plan.")
    required = {
        "families",
        "sample_size",
        "n_items",
        "missing_rate",
        "noise_level",
        "specification",
        "replications",
        "seed",
        "label",
    }
    missing = sorted(required - set(x))
    if missing:
        raise EyeProcessValidationError("validation plan is missing fields: " + ", ".join(missing))
    return True


def eyeprocess_validation_seed(
    master_seed,
    index,
    stream=0,
):
    """Derive the frozen deterministic bounded validation seed."""
    try:
        master = int(master_seed)
        index0 = int(index)
        stream0 = int(stream)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError(
            "master_seed and index must be positive scalar integers; stream must be a non-negative scalar integer."
        ) from exc

    if master < 1 or index0 < 1 or stream0 < 0:
        raise EyeProcessValidationError(
            "master_seed and index must be positive scalar integers; stream must be a non-negative scalar integer."
        )

    modulus = 2147483646
    value = (float(master) + 104729.0 * float(index0) + 1009.0 * float(stream0)) % modulus
    return int(value + 1)


def _r_expand_grid(columns: list[tuple[str, list[Any]]]) -> pd.DataFrame:
    names = [name for name, _ in columns]
    values = [items for _, items in columns]
    rows = [tuple(reversed(row)) for row in itertools.product(*reversed(values))]
    return pd.DataFrame(rows, columns=names)


def expand_eyeprocess_validation_plan(x):
    """Expand a validation plan using R ``expand.grid`` row ordering."""
    validate_eyeprocess_validation_plan(x)

    grid = _r_expand_grid(
        [
            ("family", list(x["families"])),
            ("sample_size", list(x["sample_size"])),
            ("n_items", list(x["n_items"])),
            ("missing_rate", list(x["missing_rate"])),
            ("noise_level", list(x["noise_level"])),
            ("specification", list(x["specification"])),
        ]
    )

    grid["scenario_id"] = [f"M2S{index:04d}" for index in range(1, len(grid) + 1)]
    grid["replications"] = int(x["replications"])
    grid["master_seed"] = int(x["seed"])
    grid["scenario_seed"] = [eyeprocess_validation_seed(x["seed"], index) for index in range(1, len(grid) + 1)]

    grid = grid[
        [
            "scenario_id",
            "family",
            "sample_size",
            "n_items",
            "missing_rate",
            "noise_level",
            "specification",
            "replications",
            "master_seed",
            "scenario_seed",
        ]
    ]
    grid.attrs["label"] = x["label"]
    grid.attrs["plan_hash"] = object_hash(x)
    return grid


def validation_acceptance_rule(
    metric,
    direction="max",
    threshold=None,
    upper=None,
    tolerance=0,
):
    """Define one frozen validation acceptance rule."""
    if not isinstance(metric, str):
        raise EyeProcessValidationError("metric must be character.")
    if direction not in _ALLOWED_DIRECTIONS:
        raise EyeProcessValidationError("direction must be one of max, min, between, or equals.")

    try:
        threshold0 = float(threshold)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("threshold must be supplied.") from exc

    if direction == "between":
        try:
            upper0 = float(upper)
        except (TypeError, ValueError) as exc:
            raise EyeProcessValidationError("between rules require finite lower threshold <= finite upper.") from exc
        if not np.isfinite(threshold0) or not np.isfinite(upper0) or threshold0 > upper0:
            raise EyeProcessValidationError("between rules require finite lower threshold <= finite upper.")
    else:
        upper0 = upper
        if not np.isfinite(threshold0):
            raise EyeProcessValidationError("threshold must be a finite scalar for this rule.")

    try:
        tolerance0 = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("tolerance must be a finite non-negative scalar.") from exc
    if not np.isfinite(tolerance0) or tolerance0 < 0:
        raise EyeProcessValidationError("tolerance must be a finite non-negative scalar.")

    return _tag(
        {
            "metric": metric,
            "direction": direction,
            "threshold": threshold0,
            "upper": upper0,
            "tolerance": tolerance0,
        },
        _RULE_CLASS,
    )


def evaluate_validation_acceptance(value, rule):
    """Evaluate one value against a frozen acceptance rule."""
    if not _class_is(rule, _RULE_CLASS):
        raise EyeProcessValidationError("rule must be created by validation_acceptance_rule().")
    try:
        value0 = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if not np.isfinite(value0):
        return pd.NA

    threshold = float(rule["threshold"])
    tolerance = float(rule["tolerance"])
    direction = rule["direction"]

    if direction == "max":
        return bool(value0 <= threshold + tolerance)
    if direction == "min":
        return bool(value0 >= threshold - tolerance)
    if direction == "between":
        upper = float(rule["upper"])
        return bool(value0 >= threshold - tolerance and value0 <= upper + tolerance)
    if direction == "equals":
        return bool(abs(value0 - threshold) <= tolerance)
    raise EyeProcessValidationError("Unknown rule direction.")


def validation_acceptance_matrix(
    summary,
    rules,
    id_cols=(),
):
    """Evaluate a validation summary table against named rules."""
    frame = _as_frame(summary, name="summary")
    if not isinstance(rules, Mapping) and not isinstance(rules, (list, tuple)):
        raise EyeProcessValidationError("rules must be a non-empty list of validation_acceptance_rule objects.")

    if isinstance(rules, Mapping):
        entries = list(rules.items())
    else:
        entries = [(f"rule_{index}", rule) for index, rule in enumerate(rules, start=1)]

    if not entries or any(not _class_is(rule, _RULE_CLASS) for _, rule in entries):
        raise EyeProcessValidationError("rules must be a non-empty list of validation_acceptance_rule objects.")

    ids = [str(value) for value in _sequence(id_cols) if str(value)]
    metrics = [str(rule["metric"]) for _, rule in entries]
    _require_columns(
        frame,
        _unique(ids + metrics),
        name="summary",
    )

    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        for index, (name, rule) in enumerate(
            entries,
            start=1,
        ):
            value = record[rule["metric"]]
            evaluated = evaluate_validation_acceptance(
                value,
                rule,
            )
            threshold_values = [rule["threshold"]]
            if rule["upper"] is not None:
                threshold_values.append(rule["upper"])
            threshold_text = ":".join(str(value) for value in threshold_values)
            row = {key: record[key] for key in ids}
            row.update(
                {
                    "rule_id": (str(name) if str(name) else f"rule_{index}"),
                    "metric": rule["metric"],
                    "value": pd.to_numeric(
                        pd.Series([value]),
                        errors="coerce",
                    ).iloc[0],
                    "direction": rule["direction"],
                    "threshold": threshold_text,
                    "pass": evaluated,
                }
            )
            rows.append(row)

    return pd.DataFrame(
        rows,
        columns=ids
        + [
            "rule_id",
            "metric",
            "value",
            "direction",
            "threshold",
            "pass",
        ],
    )


def summarise_validation_acceptance(
    x,
    by=(),
):
    """Summarise a frozen acceptance matrix."""
    frame = _as_frame(x, name="x")
    groups = [str(value) for value in _sequence(by) if str(value)]
    _require_columns(
        frame,
        groups + ["pass"],
        name="x",
    )

    def calc(part: pd.DataFrame) -> dict[str, Any]:
        evaluable = part["pass"].notna()
        n_evaluable = int(evaluable.sum())
        passed = part.loc[evaluable, "pass"].astype(bool)
        return {
            "n": len(part),
            "n_evaluable": n_evaluable,
            "n_pass": int(passed.sum()),
            "pass_fraction": (float(passed.mean()) if n_evaluable else np.nan),
        }

    if not groups:
        return pd.DataFrame([calc(frame)])

    output = []
    grouper = groups[0] if len(groups) == 1 else groups
    for keys, part in frame.groupby(
        grouper,
        dropna=False,
        sort=True,
    ):
        if len(groups) == 1:
            keys = (keys,)
        row = dict(zip(groups, keys))
        row.update(calc(part))
        output.append(row)
    return pd.DataFrame(output)


def validation_mcse_profile(
    x,
    metric,
    by=(),
):
    """Estimate Monte Carlo uncertainty for validation summaries."""
    frame = _as_frame(x, name="x")
    groups = [str(value) for value in _sequence(by) if str(value)]
    metric0 = str(metric)
    _require_columns(
        frame,
        groups + [metric0],
        name="x",
    )

    def calc(part: pd.DataFrame) -> dict[str, Any]:
        values = pd.to_numeric(
            part[metric0],
            errors="coerce",
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        if n:
            mean = float(np.mean(values))
        else:
            mean = np.nan
        if n > 1:
            sd = float(np.std(values, ddof=1))
            mcse = sd / math.sqrt(n)
        else:
            sd = np.nan
            mcse = np.nan
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "mcse_mean": mcse,
        }

    if not groups:
        return pd.DataFrame([calc(frame)])

    output = []
    grouper = groups[0] if len(groups) == 1 else groups
    for keys, part in frame.groupby(
        grouper,
        dropna=False,
        sort=True,
    ):
        if len(groups) == 1:
            keys = (keys,)
        row = dict(zip(groups, keys))
        row.update(calc(part))
        output.append(row)
    return pd.DataFrame(output)


def validation_replication_budget(
    pilot_sd,
    target_mcse,
    minimum=20,
    maximum=10000,
):
    """Compute the frozen replication budget from target MCSE."""
    try:
        pilot = float(pilot_sd)
        target = float(target_mcse)
        minimum0 = int(minimum)
        maximum0 = int(maximum)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("replication-budget inputs must be scalar numeric values.") from exc

    if not np.isfinite(pilot) or pilot < 0:
        raise EyeProcessValidationError("pilot_sd must be finite and non-negative.")
    if not np.isfinite(target) or target <= 0:
        raise EyeProcessValidationError("target_mcse must be positive.")
    if minimum0 < 1 or maximum0 < minimum0:
        raise EyeProcessValidationError("invalid minimum/maximum replication limits.")

    raw = minimum0 if pilot == 0 else math.ceil((pilot / target) ** 2)
    return int(
        min(
            maximum0,
            max(minimum0, raw),
        )
    )


def validation_scenario_manifest(
    plan,
    source_commit=None,
    generated_at=None,
):
    """Create a frozen validation scenario manifest."""
    validate_eyeprocess_validation_plan(plan)
    grid = expand_eyeprocess_validation_plan(plan)

    return _tag(
        {
            "label": plan["label"],
            "plan_hash": object_hash(plan),
            "scenarios": grid,
            "source_commit": (None if source_commit is None else str(source_commit)),
            "generated_at": _utc_string(generated_at),
            "scientific_scope": ("software validation; not construct-validity evidence"),
        },
        _MANIFEST_CLASS,
    )


def write_validation_scenario_manifest(x, path):
    """Persist a validation scenario manifest as Python-native JSON."""
    if not _class_is(x, _MANIFEST_CLASS):
        raise EyeProcessValidationError("x must be a validation scenario manifest.")
    output = Path(path)
    if output.suffix.lower() == ".rds":
        raise EyeProcessValidationError(
            "Native RDS serialization is R-specific and is intentionally "
            "not emulated by eyeprocesspy. Use a .json output path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _json_safe(x),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output.resolve().as_posix()


def read_validation_scenario_manifest(path):
    """Read a Python-native validation scenario manifest."""
    source = Path(path)
    if source.suffix.lower() == ".rds":
        raise EyeProcessValidationError(
            "Native RDS serialization is R-specific and is intentionally "
            "not emulated by eyeprocesspy. Read the JSON manifest instead."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EyeProcessValidationError("Could not read validation scenario manifest.") from exc
    result = _json_restore(payload)
    if not _class_is(result, _MANIFEST_CLASS):
        raise EyeProcessValidationError("File is not an eyeprocess validation scenario manifest.")
    return result


def eyeprocess_validation_evidence_grade(
    components,
    required=(
        "design",
        "execution",
        "summary",
        "provenance",
        "hash",
    ),
):
    """Grade completeness of declared validation-evidence components."""
    components0 = {str(value) for value in _sequence(components) if value is not None}
    required0 = _unique([str(value) for value in _sequence(required) if value is not None])
    if not required0:
        raise EyeProcessValidationError("required must be non-empty.")

    present = {name: name in components0 for name in required0}
    count = sum(present.values())
    if count == len(required0):
        grade = "complete"
    elif count >= math.ceil(len(required0) * 0.6):
        grade = "partial"
    else:
        grade = "insufficient"

    return _tag(
        {
            "grade": grade,
            "required": required0,
            "present": present,
            "coverage": count / len(required0),
        },
        _GRADE_CLASS,
    )
