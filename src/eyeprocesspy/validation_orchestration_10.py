"""Research-scale validation orchestration ported from frozen R/023."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .exceptions import EyeProcessBackendError, EyeProcessValidationError

__all__ = [
    "collect_validation_jobs",
    "prune_validation_checkpoints",
    "read_validation_job_manifest",
    "resume_validation_jobs",
    "run_validation_jobs",
    "split_validation_plan",
    "validation_job_plan",
    "validation_seed",
    "write_validation_job_manifest",
]

_PLAN_FILE = "plan.json"
_RUNNER_FILE = "runner-manifest.json"
_CHECKPOINT_SUFFIX = ".json"
_SCHEMA_VERSION = "1.0.0"


class EyeValidationJobPlan(dict):
    """Python counterpart of ``eye_validation_job_plan``."""

    eyeprocess_class = "eye_validation_job_plan"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class EyeValidationRun(dict):
    """Python counterpart of ``eye_validation_run``."""

    eyeprocess_class = "eye_validation_run"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class EyeValidationCollection(dict):
    """Python counterpart of ``eye_validation_collection``."""

    eyeprocess_class = "eye_validation_collection"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _scalar_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        _stop(f"`{name}` must be a single integer >= {minimum}.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError(f"`{name}` must be a single integer >= {minimum}.") from exc
    try:
        if float(numeric) != float(value):
            _stop(f"`{name}` must be a single integer >= {minimum}.")
    except (TypeError, ValueError, OverflowError):
        _stop(f"`{name}` must be a single integer >= {minimum}.")
    if numeric < minimum:
        _stop(f"`{name}` must be a single integer >= {minimum}.")
    return numeric


def _scalar_num(
    value: Any,
    name: str,
    minimum: float = -math.inf,
    maximum: float = math.inf,
    finite: bool = True,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError(f"`{name}` is outside its allowed range.") from exc
    if math.isnan(numeric) or (finite and not math.isfinite(numeric)) or numeric < minimum or numeric > maximum:
        _stop(f"`{name}` is outside its allowed range.")
    return numeric


def _safe_name(value: Any, fallback: str = "unnamed") -> str:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() == "nan":
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text or fallback


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _canonical_scalar(value: Any) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return ",".join(_canonical_scalar(item) for item in list(value))
    if _is_missing(value):
        return "<NA>"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            # Close analogue of R format(..., digits=17, scientific=TRUE, trim=TRUE).
            text = format(numeric, ".16e")
            mantissa, exponent = text.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            sign = "+" if int(exponent) >= 0 else "-"
            return f"{mantissa}e{sign}{abs(int(exponent)):02d}"
        return str(numeric)
    return str(value)


def _canonical_row(design: Any) -> str:
    if isinstance(design, pd.DataFrame):
        if len(design) != 1:
            _stop("A design cell must be a list or one-row data frame.")
        design = design.iloc[0].to_dict()
    elif isinstance(design, pd.Series):
        design = design.to_dict()
    if not isinstance(design, Mapping):
        _stop("A design cell must be a list or one-row data frame.")
    return "|".join(f"{name}={_canonical_scalar(design[name])}" for name in sorted(design))


def _hash_int(text: Any, modulus: int = 2147483629) -> int:
    if isinstance(text, (list, tuple)):
        text = "\n".join(str(item) for item in text)
    encoded = str(text).encode("utf-8")
    if not encoded:
        return 1
    # Frozen R evaluates this recurrence in double precision.
    value = float(2166136261 % modulus)
    modulus_f = float(modulus)
    for byte in encoded:
        value = math.fmod(value * 16777619.0 + byte + 1.0, modulus_f)
    return int(value)


def validation_seed(design, replication, base_seed=1, stream=1):
    """Allocate a deterministic validation seed from design contents."""
    replication = _scalar_int(replication, "replication", 1)
    base_seed = _scalar_int(base_seed, "base_seed", 0)
    stream = _scalar_int(stream, "stream", 1)
    key = "|".join(
        [
            _canonical_row(design),
            str(replication),
            str(base_seed),
            str(stream),
        ]
    )
    seed = (_hash_int(key) + base_seed + 104729 * stream) % 2147483646
    return int(seed + 1)


def _as_levels(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _expand_grid(grid: Any) -> pd.DataFrame:
    if grid is None:
        return pd.DataFrame({".scenario": [1]})
    if isinstance(grid, pd.DataFrame):
        if grid.empty:
            _stop("`grid` must contain at least one design cell.")
        return grid.reset_index(drop=True).copy()
    if isinstance(grid, Mapping):
        if not grid:
            _stop("`grid` must contain at least one design cell.")
        names = list(grid)
        levels = [_as_levels(grid[name]) for name in names]
        if any(len(level) == 0 for level in levels):
            _stop("`grid` must contain at least one design cell.")
        # Match R expand.grid ordering: first factor varies fastest.
        rows = []
        for reversed_values in itertools.product(*reversed(levels)):
            values = list(reversed(reversed_values))
            rows.append(dict(zip(names, values, strict=True)))
        return pd.DataFrame(rows)
    _stop("`grid` must be a data frame, named list, or NULL.")


def _plan_id(
    grid: pd.DataFrame,
    replications: int,
    base_seed: int,
    model_family: str,
) -> str:
    cells = [_canonical_row(grid.iloc[[index]]) for index in range(len(grid))]
    digest = _hash_int([*cells, replications, base_seed])
    return f"{_safe_name(model_family, 'validation')}-{digest:010d}"


def _serialize(value: Any) -> Any:
    if value is pd.NA:
        return {"__eye_type__": "missing"}
    if isinstance(value, pd.DataFrame):
        return {
            "__eye_type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "data": [[_serialize(item) for item in row] for row in value.itertuples(index=False, name=None)],
        }
    if isinstance(value, pd.Series):
        return {
            "__eye_type__": "series",
            "name": None if value.name is None else str(value.name),
            "data": [_serialize(item) for item in value.tolist()],
        }
    if isinstance(value, np.ndarray):
        return {
            "__eye_type__": "ndarray",
            "data": _serialize(value.tolist()),
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _serialize(value.item())
    if isinstance(value, Path):
        return {"__eye_type__": "path", "value": str(value)}
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"__eye_type__": "float", "value": "nan"}
        if math.isinf(value):
            return {
                "__eye_type__": "float",
                "value": "inf" if value > 0 else "-inf",
            }
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return {"__eye_type__": "datetime", "value": value.isoformat()}
    return {
        "__eye_type__": "python_repr",
        "class": type(value).__name__,
        "value": repr(value),
    }


def _deserialize(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__eye_type__")
    if kind == "missing":
        return pd.NA
    if kind == "dataframe":
        return pd.DataFrame(
            [[_deserialize(item) for item in row] for row in value.get("data", [])],
            columns=value.get("columns", []),
        )
    if kind == "series":
        return pd.Series(
            [_deserialize(item) for item in value.get("data", [])],
            name=value.get("name"),
        )
    if kind == "ndarray":
        return np.asarray(_deserialize(value.get("data", [])))
    if kind == "path":
        return Path(value["value"])
    if kind == "datetime":
        return datetime.fromisoformat(value["value"])
    if kind == "float":
        return {
            "nan": math.nan,
            "inf": math.inf,
            "-inf": -math.inf,
        }[value["value"]]
    if kind == "python_repr":
        return value.get("value")
    return {key: _deserialize(item) for key, item in value.items()}


def _stable_payload(value: Any) -> str:
    return json.dumps(
        _serialize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _object_fingerprint(value: Any, prefix: str = "object") -> str:
    digest = hashlib.sha256(_stable_payload(value).encode("utf-8")).hexdigest()
    number = int(digest[:16], 16) % 10_000_000_000
    return f"{prefix}-{number:010d}"


def _plan_fingerprint(
    jobs: pd.DataFrame,
    model_family: str,
    base_seed: int,
) -> str:
    stable = jobs.drop(columns=[column for column in ("status", "attempt") if column in jobs.columns])
    return _object_fingerprint(
        {
            "model_family": model_family,
            "base_seed": base_seed,
            "jobs": stable,
        },
        "plan",
    )


def validation_job_plan(
    grid=None,
    replications=100,
    base_seed=1,
    model_family="unspecified",
    plan_id=None,
    chunk_size=1,
    metadata=None,
):
    """Create a deterministic research-scale validation job plan."""
    expanded = _expand_grid(grid)
    replications = _scalar_int(replications, "replications", 1)
    base_seed = _scalar_int(base_seed, "base_seed", 0)
    chunk_size = _scalar_int(chunk_size, "chunk_size", 1)
    model_family = _safe_name(model_family, "unspecified")
    if plan_id is None:
        plan_id = _plan_id(
            expanded,
            replications,
            base_seed,
            model_family,
        )
    plan_id = _safe_name(plan_id, "validation-plan")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        _stop("`metadata` must be a mapping.")

    rows: list[dict[str, Any]] = []
    for scenario_index in range(len(expanded)):
        scenario_id = f"S{scenario_index + 1:05d}"
        design = expanded.iloc[scenario_index].to_dict()
        for replication in range(1, replications + 1):
            rows.append(
                {
                    "plan_id": plan_id,
                    "model_family": model_family,
                    "scenario_id": scenario_id,
                    "replication": replication,
                    "seed": validation_seed(
                        design,
                        replication,
                        base_seed,
                    ),
                    "job_id": (f"{scenario_id}-R{replication:06d}"),
                    **design,
                }
            )
    jobs = pd.DataFrame(rows)
    jobs["chunk_id"] = [f"C{((index // chunk_size) + 1):05d}" for index in range(len(jobs))]
    jobs["expected_checkpoint"] = "job-" + jobs["job_id"].astype(str) + _CHECKPOINT_SUFFIX
    jobs["status"] = "planned"
    jobs["attempt"] = 0

    output = EyeValidationJobPlan(
        plan_id=plan_id,
        model_family=model_family,
        jobs=jobs,
        grid=expanded,
        replications=replications,
        base_seed=base_seed,
        chunk_size=chunk_size,
        metadata=dict(metadata),
        created_utc=_now_utc(),
        schema_version=_SCHEMA_VERSION,
    )
    output["plan_fingerprint"] = _plan_fingerprint(
        jobs,
        model_family,
        base_seed,
    )
    return output


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
        text=True,
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(
                _serialize(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    return _deserialize(json.loads(path.read_text(encoding="utf-8")))


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".csv",
        dir=path.parent,
        text=True,
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            lineterminator="\n",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _manifest_metadata(plan: EyeValidationJobPlan) -> dict[str, Any]:
    return {
        "schema_version": plan["schema_version"],
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "model_family": plan["model_family"],
        "created_utc": plan["created_utc"],
        "written_utc": _now_utc(),
        "replications": plan["replications"],
        "scenarios": len(plan["grid"]),
        "jobs": len(plan["jobs"]),
        "base_seed": plan["base_seed"],
        "chunk_size": plan["chunk_size"],
        "eyeprocesspy_serialization": "json",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "metadata": plan["metadata"],
    }


def write_validation_job_manifest(plan, path, overwrite=False):
    """Write the validation manifest using transparent Python JSON."""
    if not isinstance(plan, EyeValidationJobPlan):
        _stop("`plan` must be created by `validation_job_plan()`.")
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        _stop(f"Manifest directory already exists: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    (output / "logs").mkdir()
    (output / "artifacts").mkdir()

    _atomic_json(output / _PLAN_FILE, plan)
    _atomic_csv(output / "jobs.csv", plan["jobs"])
    _atomic_csv(output / "design-grid.csv", plan["grid"])
    _atomic_json(output / "manifest.json", _manifest_metadata(plan))
    (output / "serialization-boundary.md").write_text(
        "# Serialization boundary\n\n"
        "The frozen R implementation stores plans and checkpoints with "
        "`saveRDS()`. eyeprocesspy never writes Python objects under an "
        "`.rds` filename. This manifest therefore uses transparent JSON.\n",
        encoding="utf-8",
    )
    return str(output)


def read_validation_job_manifest(path):
    """Read a Python validation manifest created by this source port."""
    output = Path(path).expanduser().resolve()
    if not output.is_dir():
        _stop(f"Validation manifest does not exist: {output}")
    plan_file = output / _PLAN_FILE
    if not plan_file.is_file():
        _stop(f"Validation manifest is missing `{_PLAN_FILE}`.")
    payload = _read_json(plan_file)
    if not isinstance(payload, Mapping):
        _stop("Manifest plan has an unsupported class.")
    required = {
        "plan_id",
        "model_family",
        "jobs",
        "grid",
        "replications",
        "base_seed",
        "chunk_size",
        "metadata",
        "schema_version",
        "plan_fingerprint",
    }
    if not required.issubset(payload):
        _stop("Manifest plan has an unsupported class.")
    plan = EyeValidationJobPlan(payload)
    if not isinstance(plan["jobs"], pd.DataFrame):
        _stop("Manifest jobs table is invalid.")
    if not isinstance(plan["grid"], pd.DataFrame):
        _stop("Manifest design grid is invalid.")
    plan["manifest_path"] = str(output)
    return plan


def split_validation_plan(plan, chunks=None):
    """Split a validation plan into independent chunk-specific plans."""
    if not isinstance(plan, EyeValidationJobPlan):
        _stop("Expected an `eye_validation_job_plan`.")
    available = list(dict.fromkeys(plan["jobs"]["chunk_id"].astype(str).tolist()))
    requested = available if chunks is None else [str(chunk) for chunk in _as_levels(chunks)]
    selected = [chunk for chunk in requested if chunk in available]
    if not selected:
        _stop("No requested chunks exist in the plan.")

    output: dict[str, EyeValidationJobPlan] = {}
    for chunk in selected:
        copy = EyeValidationJobPlan(
            {
                key: (
                    value.copy(deep=True)
                    if isinstance(value, pd.DataFrame)
                    else dict(value)
                    if isinstance(value, dict)
                    else value
                )
                for key, value in plan.items()
                if key != "manifest_path"
            }
        )
        copy["jobs"] = plan["jobs"].loc[plan["jobs"]["chunk_id"].astype(str).eq(chunk)].reset_index(drop=True)
        copy["metadata"] = dict(plan["metadata"])
        copy["metadata"]["parent_plan_id"] = plan["plan_id"]
        copy["metadata"]["selected_chunk"] = chunk
        output[chunk] = copy
    return output


def _function_fingerprint(function: Any) -> str | None:
    if not callable(function):
        return None
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        if code is None:
            source = repr(function)
        else:
            source = "|".join(
                [
                    code.co_name,
                    code.co_code.hex(),
                    repr(code.co_consts),
                    repr(getattr(function, "__defaults__", None)),
                ]
            )
    return "fn-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _call_supported(function, positional=(), named=None):
    named = dict(named or {})
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*positional, **named)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(*positional, **named)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    filtered = {name: value for name, value in named.items() if name in allowed}
    return function(*positional, **filtered)


def _capture_call(function, positional=(), named=None):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        value = _call_supported(
            function,
            positional=positional,
            named=named,
        )
    messages = []
    warning_messages = list(dict.fromkeys(str(item.message) for item in captured))
    return value, warning_messages, messages


def _deep_size(value: Any) -> int:
    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(item)
        if isinstance(item, pd.DataFrame):
            return total + int(item.memory_usage(index=True, deep=True).sum())
        if isinstance(item, pd.Series):
            return total + int(item.memory_usage(index=True, deep=True))
        if isinstance(item, np.ndarray):
            return total + int(item.nbytes)
        if isinstance(item, Mapping):
            return total + sum(visit(key) + visit(value) for key, value in item.items())
        if isinstance(item, (list, tuple, set, frozenset)):
            return total + sum(visit(value) for value in item)
        return total

    return visit(value)


def _truth_mapping(truth: Any, parameters: Sequence[str]) -> dict[str, float]:
    if isinstance(truth, pd.Series):
        truth = truth.to_dict()
    if isinstance(truth, Mapping):
        output = {}
        for key, value in truth.items():
            try:
                output[str(key)] = float(value)
            except (TypeError, ValueError):
                output[str(key)] = math.nan
        return output
    array = np.asarray(truth)
    if array.ndim == 1 and len(array) == len(parameters):
        return {parameter: float(value) for parameter, value in zip(parameters, array, strict=True)}
    return {}


def _standardize_estimates(
    estimates: Any,
    truth: Any = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    if isinstance(estimates, pd.Series):
        estimates = estimates.to_dict()
    if isinstance(estimates, Mapping):
        estimates = pd.DataFrame(
            {
                "parameter": [str(key) for key in estimates],
                "estimate": list(estimates.values()),
            }
        )
    if not isinstance(estimates, pd.DataFrame):
        _stop("The estimator extractor must return a data frame with `parameter` and `estimate`.")
    if not {"parameter", "estimate"}.issubset(estimates.columns):
        _stop("The estimator extractor must return a data frame with `parameter` and `estimate`.")
    output = estimates.copy().reset_index(drop=True)
    output["parameter"] = output["parameter"].astype(str)
    output["estimate"] = pd.to_numeric(
        output["estimate"],
        errors="coerce",
    )
    if "std_error" not in output:
        if "sd" in output:
            output["std_error"] = output["sd"]
        else:
            output["std_error"] = np.nan
    output["std_error"] = pd.to_numeric(
        output["std_error"],
        errors="coerce",
    )

    lower_alias = next(
        (name for name in ("q2.5", "q025", "q5", "q05") if name in output),
        None,
    )
    upper_alias = next(
        (name for name in ("q97.5", "q975", "q95") if name in output),
        None,
    )
    z_value = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    if "lower" not in output:
        output["lower"] = output[lower_alias] if lower_alias else output["estimate"] - z_value * output["std_error"]
    if "upper" not in output:
        output["upper"] = output[upper_alias] if upper_alias else output["estimate"] + z_value * output["std_error"]
    output["lower"] = pd.to_numeric(output["lower"], errors="coerce")
    output["upper"] = pd.to_numeric(output["upper"], errors="coerce")

    mapping = _truth_mapping(
        truth,
        output["parameter"].tolist(),
    )
    output["truth"] = [mapping.get(parameter, math.nan) for parameter in output["parameter"]]
    output["bias"] = output["estimate"] - output["truth"]
    output["squared_error"] = output["bias"] ** 2
    output["absolute_error"] = output["bias"].abs()
    denominator = output["truth"].abs()
    output["relative_bias"] = np.where(
        np.isfinite(denominator) & (denominator > np.sqrt(np.finfo(float).eps)),
        output["bias"] / output["truth"],
        np.nan,
    )
    finite_interval = np.isfinite(output["lower"]) & np.isfinite(output["upper"]) & np.isfinite(output["truth"])
    output["covered"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    output.loc[finite_interval, "covered"] = (
        output.loc[finite_interval, "lower"] <= output.loc[finite_interval, "truth"]
    ) & (output.loc[finite_interval, "upper"] >= output.loc[finite_interval, "truth"])
    output["interval_width"] = output["upper"] - output["lower"]
    return output


def _default_diagnostics(fit: Any) -> pd.DataFrame:
    converged = True
    iterations = np.nan
    divergences = np.nan
    max_rhat = np.nan
    min_ess_bulk = np.nan
    min_ess_tail = np.nan
    if isinstance(fit, Mapping):
        if "converged" in fit:
            converged = bool(fit["converged"])
        if "iterations" in fit:
            iterations = fit["iterations"]
        diagnostics = fit.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            converged = bool(diagnostics.get("converged", converged))
            divergences = diagnostics.get(
                "divergences",
                divergences,
            )
            max_rhat = diagnostics.get("max_rhat", max_rhat)
            min_ess_bulk = diagnostics.get(
                "min_ess_bulk",
                min_ess_bulk,
            )
            min_ess_tail = diagnostics.get(
                "min_ess_tail",
                min_ess_tail,
            )
    return pd.DataFrame(
        [
            {
                "converged": converged,
                "iterations": iterations,
                "divergences": divergences,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_ess_bulk,
                "min_ess_tail": min_ess_tail,
            }
        ]
    )


def _job_result(
    job: Mapping[str, Any],
    status: str,
    stage: str,
    started: float,
    *,
    warning_messages=None,
    messages=None,
    error=None,
    estimates=None,
    diagnostics=None,
    draws=None,
    predictions=None,
    artifacts=None,
) -> dict[str, Any]:
    ended = time.time()
    if diagnostics is None:
        diagnostics = pd.DataFrame([{"converged": status == "complete"}])
    diagnostics = pd.DataFrame(diagnostics).copy()
    diagnostics["job_id"] = str(job["job_id"])
    return {
        "schema_version": _SCHEMA_VERSION,
        "plan_id": str(job["plan_id"]),
        "job": dict(job),
        "status": status,
        "stage": stage,
        "started_utc": datetime.fromtimestamp(
            started,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ended_utc": datetime.fromtimestamp(
            ended,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "elapsed_seconds": float(ended - started),
        "warnings": list(dict.fromkeys(warning_messages or [])),
        "messages": list(dict.fromkeys(messages or [])),
        "error": error,
        "estimates": (pd.DataFrame() if estimates is None else pd.DataFrame(estimates)),
        "diagnostics": diagnostics,
        "draws": draws,
        "predictions": predictions,
        "artifacts": dict(artifacts or {}),
        "session": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _run_job_core(
    job,
    simulator,
    fitter,
    extractor,
    truth_extractor,
    *,
    simulation_args=None,
    fit_args=None,
    diagnostics_extractor=None,
    draws_extractor=None,
    predictions_extractor=None,
    confidence=0.95,
    memory_limit_mb=math.inf,
):
    started = time.time()
    warning_messages: list[str] = []
    messages: list[str] = []
    excluded = {
        "plan_id",
        "model_family",
        "scenario_id",
        "replication",
        "seed",
        "job_id",
        "chunk_id",
        "expected_checkpoint",
        "status",
        "attempt",
    }
    design = {key: value for key, value in job.items() if key not in excluded}
    simulation_args = dict(simulation_args or {})
    fit_args = dict(fit_args or {})

    try:
        simulation, captured, emitted = _capture_call(
            simulator,
            named={
                **design,
                **simulation_args,
                "seed": int(job["seed"]),
            },
        )
        warning_messages.extend(captured)
        messages.extend(emitted)
    except Exception as exc:
        return _job_result(
            job,
            "failed",
            "simulation",
            started,
            error=str(exc),
        )

    if math.isfinite(memory_limit_mb):
        simulation_mb = _deep_size(simulation) / 1024**2
        if simulation_mb > memory_limit_mb:
            return _job_result(
                job,
                "failed",
                "memory",
                started,
                warning_messages=warning_messages,
                messages=messages,
                error=(
                    f"Simulation object size {simulation_mb:.2f} MB "
                    f"exceeds the configured {memory_limit_mb:.2f} MB limit."
                ),
            )

    try:
        fit, captured, emitted = _capture_call(
            fitter,
            positional=(simulation,),
            named=fit_args,
        )
        warning_messages.extend(captured)
        messages.extend(emitted)
    except Exception as exc:
        return _job_result(
            job,
            "failed",
            "fit",
            started,
            warning_messages=warning_messages,
            messages=messages,
            error=str(exc),
        )

    if math.isfinite(memory_limit_mb):
        fit_mb = _deep_size(fit) / 1024**2
        if fit_mb > memory_limit_mb:
            return _job_result(
                job,
                "failed",
                "memory",
                started,
                warning_messages=warning_messages,
                messages=messages,
                error=(f"Fit object size {fit_mb:.2f} MB exceeds the configured {memory_limit_mb:.2f} MB limit."),
            )

    try:
        raw_estimates, captured, emitted = _capture_call(
            extractor,
            positional=(fit,),
        )
        warning_messages.extend(captured)
        messages.extend(emitted)
    except Exception as exc:
        return _job_result(
            job,
            "failed",
            "extract",
            started,
            warning_messages=warning_messages,
            messages=messages,
            error=str(exc),
        )

    try:
        truth, captured, emitted = _capture_call(
            truth_extractor,
            positional=(simulation,),
        )
        warning_messages.extend(captured)
        messages.extend(emitted)
    except Exception as exc:
        return _job_result(
            job,
            "failed",
            "truth",
            started,
            warning_messages=warning_messages,
            messages=messages,
            error=str(exc),
        )

    try:
        estimates = _standardize_estimates(
            raw_estimates,
            truth,
            confidence,
        )
    except Exception as exc:
        return _job_result(
            job,
            "failed",
            "standardize",
            started,
            warning_messages=warning_messages,
            messages=messages,
            error=str(exc),
        )

    if callable(diagnostics_extractor):
        try:
            diagnostics = _call_supported(
                diagnostics_extractor,
                positional=(fit,),
            )
            diagnostics = pd.DataFrame(diagnostics)
        except Exception as exc:
            diagnostics = pd.DataFrame(
                [
                    {
                        "converged": False,
                        "diagnostic_error": str(exc),
                    }
                ]
            )
    else:
        diagnostics = _default_diagnostics(fit)
    if "converged" not in diagnostics:
        diagnostics["converged"] = True

    draws = None
    if callable(draws_extractor):
        try:
            draws = _call_supported(
                draws_extractor,
                positional=(fit,),
            )
        except Exception as exc:
            draws = {"error": str(exc)}

    predictions = None
    if callable(predictions_extractor):
        try:
            predictions = _call_supported(
                predictions_extractor,
                positional=(fit, simulation),
            )
        except Exception as exc:
            predictions = {"error": str(exc)}

    convergence = diagnostics["converged"].astype("boolean").dropna()
    converged = bool(len(convergence)) and bool(convergence.all())
    return _job_result(
        job,
        "complete" if converged else "nonconverged",
        "complete",
        started,
        warning_messages=warning_messages,
        messages=messages,
        estimates=estimates,
        diagnostics=diagnostics,
        draws=draws,
        predictions=predictions,
        artifacts={
            "simulation_size_mb": _deep_size(simulation) / 1024**2,
            "fit_size_mb": _deep_size(fit) / 1024**2,
        },
    )


def _checkpoint_path(output_dir: Path, job_id: str) -> Path:
    return output_dir / "checkpoints" / f"job-{job_id}{_CHECKPOINT_SUFFIX}"


def _lock_path(output_dir: Path, job_id: str) -> Path:
    return output_dir / "checkpoints" / f"job-{job_id}.lock"


def _acquire_lock(path: Path, stale_after_seconds: float) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir() and math.isfinite(stale_after_seconds):
        age = time.time() - path.stat().st_mtime
        if age > stale_after_seconds:
            shutil.rmtree(path, ignore_errors=True)
    try:
        path.mkdir()
    except FileExistsError:
        return False
    owner = pd.DataFrame(
        [
            {
                "pid": os.getpid(),
                "host": platform.node(),
                "acquired_utc": _now_utc(),
            }
        ]
    )
    try:
        owner.to_csv(path / "owner.csv", index=False)
    except OSError:
        pass
    return True


def _release_lock(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _run_and_checkpoint(
    job,
    output_dir,
    runner_args,
    *,
    isolation,
    timeout_seconds,
    memory_limit_mb,
    overwrite,
    fail_fast,
    stale_lock_seconds,
):
    checkpoint = _checkpoint_path(output_dir, str(job["job_id"]))
    if checkpoint.exists() and not overwrite:
        try:
            return _read_json(checkpoint)
        except Exception as exc:
            return _job_result(
                job,
                "corrupt",
                "checkpoint",
                time.time(),
                error=str(exc),
            )

    lock = _lock_path(output_dir, str(job["job_id"]))
    if not _acquire_lock(lock, stale_lock_seconds):
        return _job_result(
            job,
            "locked",
            "lock",
            time.time(),
            error="Another process holds this job lock.",
        )
    try:
        if isolation == "callr":
            raise EyeProcessBackendError(
                "Frozen R `callr` process isolation has no exact Python "
                "backend in this source port. Use `isolation='in_process'`."
            )
        result = _run_job_core(
            job,
            **runner_args,
            memory_limit_mb=memory_limit_mb,
        )
        _atomic_json(checkpoint, result)
        log_row = pd.DataFrame(
            [
                {
                    "job_id": job["job_id"],
                    "scenario_id": job["scenario_id"],
                    "replication": job["replication"],
                    "seed": job["seed"],
                    "status": result["status"],
                    "stage": result["stage"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "warnings": len(result["warnings"]),
                    "error": result["error"],
                    "checkpoint": str(checkpoint.resolve()),
                }
            ]
        )
        _atomic_csv(
            output_dir / "logs" / f"job-{job['job_id']}.csv",
            log_row,
        )
        if fail_fast and result["status"] in {
            "failed",
            "nonconverged",
        }:
            _stop(f"Validation job failed: {job['job_id']} ({result['stage']})")
        return result
    finally:
        _release_lock(lock)


def _backend(backend: Any, workers: int) -> str:
    if isinstance(backend, (list, tuple)):
        backend = backend[0]
    backend = str(backend)
    if backend not in {"auto", "sequential", "future"}:
        _stop("`backend` must be auto, sequential, or future.")
    if backend == "auto":
        return "future" if workers > 1 else "sequential"
    return backend


def _isolation(isolation: Any, timeout_seconds: float) -> str:
    if isinstance(isolation, (list, tuple)):
        isolation = isolation[0]
    isolation = str(isolation)
    if isolation not in {"auto", "in_process", "callr"}:
        _stop("`isolation` must be auto, in_process, or callr.")
    if isolation == "auto":
        if math.isfinite(timeout_seconds):
            return "callr"
        return "in_process"
    return isolation


def _status_table(output_dir: Path, plan: EyeValidationJobPlan) -> None:
    logs = sorted((output_dir / "logs").glob("job-*.csv"))
    if not logs:
        return
    frames = []
    for path in logs:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return
    status = pd.concat(frames, ignore_index=True, sort=False)
    order = {str(job_id): index for index, job_id in enumerate(plan["jobs"]["job_id"])}
    status["_order"] = status["job_id"].astype(str).map(order)
    status = status.dropna(subset=["_order"]).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    _atomic_csv(output_dir / "job-status.csv", status)


def _runner_manifest(
    plan,
    simulator,
    fitter,
    extractor,
    truth_extractor,
    diagnostics_extractor,
    draws_extractor,
    predictions_extractor,
    simulation_args,
    fit_args,
    confidence,
    run_metadata,
):
    functions = {
        "simulator": _function_fingerprint(simulator),
        "fitter": _function_fingerprint(fitter),
        "extractor": _function_fingerprint(extractor),
        "truth_extractor": _function_fingerprint(truth_extractor),
        "diagnostics_extractor": _function_fingerprint(diagnostics_extractor),
        "draws_extractor": _function_fingerprint(draws_extractor),
        "predictions_extractor": _function_fingerprint(predictions_extractor),
    }
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "plan_fingerprint": plan["plan_fingerprint"],
        "function_fingerprints": functions,
        "argument_fingerprint": _object_fingerprint(
            {
                "simulation_args": simulation_args,
                "fit_args": fit_args,
                "confidence": confidence,
                "run_metadata": run_metadata,
            },
            "runner-args",
        ),
        "run_metadata": run_metadata,
    }
    payload["fingerprint"] = _object_fingerprint(payload, "runner")
    return payload


def run_validation_jobs(
    plan,
    simulator,
    fitter,
    extractor,
    truth_extractor,
    output_dir,
    workers=1,
    backend=("auto", "sequential", "future"),
    isolation=("auto", "in_process", "callr"),
    timeout_seconds=math.inf,
    memory_limit_mb=math.inf,
    stale_lock_seconds=3600,
    overwrite=False,
    fail_fast=False,
    progress=False,
    job_ids=None,
    chunks=None,
    simulation_args=None,
    fit_args=None,
    diagnostics_extractor=None,
    draws_extractor=None,
    predictions_extractor=None,
    confidence=0.95,
    run_metadata=None,
):
    """Run deterministic validation jobs with checkpointing and resume safety."""
    if isinstance(plan, (str, Path)):
        plan = read_validation_job_manifest(plan)
    if not isinstance(plan, EyeValidationJobPlan):
        _stop("`plan` must be a validation plan or manifest path.")
    functions = [simulator, fitter, extractor, truth_extractor]
    if not all(callable(function) for function in functions):
        _stop("Simulator, fitter, extractor, and truth extractor must be functions.")
    workers = _scalar_int(workers, "workers", 1)
    timeout_seconds = _scalar_num(
        timeout_seconds,
        "timeout_seconds",
        0,
        math.inf,
        finite=False,
    )
    memory_limit_mb = _scalar_num(
        memory_limit_mb,
        "memory_limit_mb",
        0,
        math.inf,
        finite=False,
    )
    stale_lock_seconds = _scalar_num(
        stale_lock_seconds,
        "stale_lock_seconds",
        0,
        math.inf,
        finite=False,
    )
    confidence = _scalar_num(
        confidence,
        "confidence",
        0,
        1,
    )
    if confidence <= 0 or confidence >= 1:
        _stop("`confidence` must lie strictly between 0 and 1.")
    simulation_args = dict(simulation_args or {})
    fit_args = dict(fit_args or {})
    run_metadata = dict(run_metadata or {})

    backend = _backend(backend, workers)
    isolation = _isolation(isolation, timeout_seconds)
    if isolation == "callr":
        raise EyeProcessBackendError(
            "Frozen R `callr` isolation is R-specific. "
            "eyeprocesspy currently supports `isolation='in_process'`; "
            "a finite timeout therefore requires an explicit Python "
            "process-isolation implementation before parity can be claimed."
        )

    output = Path(output_dir).expanduser().resolve()
    plan_file = output / _PLAN_FILE
    if not output.exists():
        write_validation_job_manifest(plan, output)
    elif not plan_file.exists():
        existing = list(output.iterdir())
        if existing:
            _stop(f"Existing output directory is not a validation manifest: {output}")
        write_validation_job_manifest(plan, output, overwrite=True)
    else:
        existing_plan = read_validation_job_manifest(output)
        if existing_plan["plan_id"] != plan["plan_id"] or existing_plan["plan_fingerprint"] != plan["plan_fingerprint"]:
            _stop("The output directory belongs to a different validation plan or plan revision.")

    jobs = plan["jobs"].copy()
    if job_ids is not None:
        selected_ids = {str(value) for value in _as_levels(job_ids)}
        jobs = jobs[jobs["job_id"].astype(str).isin(selected_ids)]
    if chunks is not None:
        selected_chunks = {str(value) for value in _as_levels(chunks)}
        jobs = jobs[jobs["chunk_id"].astype(str).isin(selected_chunks)]
    jobs = jobs.reset_index(drop=True)
    if jobs.empty:
        _stop("No jobs were selected.")

    runner_manifest = _runner_manifest(
        plan,
        simulator,
        fitter,
        extractor,
        truth_extractor,
        diagnostics_extractor,
        draws_extractor,
        predictions_extractor,
        simulation_args,
        fit_args,
        confidence,
        run_metadata,
    )
    runner_path = output / _RUNNER_FILE
    if runner_path.exists():
        existing_runner = _read_json(runner_path)
        if (
            not isinstance(existing_runner, Mapping)
            or existing_runner.get("fingerprint") != runner_manifest["fingerprint"]
        ):
            _stop("Runner functions or declared run metadata differ from the existing validation execution.")
    else:
        _atomic_json(runner_path, runner_manifest)

    runner_args = {
        "simulator": simulator,
        "fitter": fitter,
        "extractor": extractor,
        "truth_extractor": truth_extractor,
        "simulation_args": simulation_args,
        "fit_args": fit_args,
        "diagnostics_extractor": diagnostics_extractor,
        "draws_extractor": draws_extractor,
        "predictions_extractor": predictions_extractor,
        "confidence": confidence,
    }

    def one(row):
        job = row.to_dict()
        return _run_and_checkpoint(
            job,
            output,
            runner_args,
            isolation=isolation,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            overwrite=bool(overwrite),
            fail_fast=bool(fail_fast),
            stale_lock_seconds=stale_lock_seconds,
        )

    started = time.time()
    if backend == "future":
        # R uses future::multisession. Threads preserve Python closure support
        # and filesystem checkpoint semantics without pretending to be R.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    one,
                    [row for _, row in jobs.iterrows()],
                )
            )
    else:
        results = []
        for index, (_, row) in enumerate(jobs.iterrows(), start=1):
            result = one(row)
            results.append(result)
            if progress:
                print(f"[{index}/{len(jobs)}] {row['job_id']}: {result['status']}")

    _status_table(output, plan)
    elapsed = time.time() - started
    return EyeValidationRun(
        plan=plan,
        selected_jobs=jobs,
        results=results,
        output_dir=str(output),
        backend=backend,
        isolation=isolation,
        workers=workers,
        started_utc=datetime.fromtimestamp(
            started,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
        elapsed_seconds=float(elapsed),
        function_fingerprints=runner_manifest["function_fingerprints"],
        runner_fingerprint=runner_manifest["fingerprint"],
        run_metadata=run_metadata,
    )


def resume_validation_jobs(
    plan,
    output_dir,
    retry=("missing", "failed", "nonconverged", "locked", "corrupt"),
    **kwargs,
):
    """Resume only missing or explicitly retryable validation jobs."""
    if isinstance(plan, (str, Path)):
        plan = read_validation_job_manifest(plan)
    if not isinstance(plan, EyeValidationJobPlan):
        _stop("Expected a validation plan.")
    output = Path(output_dir).expanduser().resolve()
    retry_set = {str(value) for value in _as_levels(retry)}

    statuses = []
    for job_id in plan["jobs"]["job_id"].astype(str):
        checkpoint = _checkpoint_path(output, job_id)
        if not checkpoint.exists():
            statuses.append("missing")
            continue
        try:
            payload = _read_json(checkpoint)
            statuses.append(str(payload.get("status", "corrupt")))
        except Exception:
            statuses.append("corrupt")

    selected = (
        plan["jobs"]
        .loc[
            pd.Series(statuses, index=plan["jobs"].index).isin(retry_set),
            "job_id",
        ]
        .astype(str)
        .tolist()
    )
    if not selected:
        return EyeValidationRun(
            plan=plan,
            selected_jobs=plan["jobs"].iloc[0:0].copy(),
            results=[],
            output_dir=str(output),
            backend="none",
            isolation="none",
            workers=0,
            elapsed_seconds=0.0,
        )
    return run_validation_jobs(
        plan,
        output_dir=output,
        job_ids=selected,
        overwrite=True,
        **kwargs,
    )


def _bind_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _annotate_frame(frame: Any, result: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    output = frame.copy()
    job = result["job"]
    for name, value in job.items():
        if name not in output:
            output[name] = value
    output["status"] = result["status"]
    output["stage"] = result["stage"]
    output["elapsed_seconds"] = result["elapsed_seconds"]
    output["warning_count"] = len(result.get("warnings", []))
    output["error"] = result.get("error")
    return output


def _checkpoint_fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_payload(value).encode("utf-8")).hexdigest()


def collect_validation_jobs(path, plan=None, strict=True):
    """Collect JSON validation checkpoints from one or more directories."""
    if isinstance(path, (str, Path)):
        paths = [Path(path).expanduser().resolve()]
    else:
        paths = [Path(item).expanduser().resolve() for item in path]
    if not paths or any(not item.is_dir() for item in paths):
        _stop("Validation path does not exist.")

    files = sorted(
        {
            checkpoint.resolve()
            for root in paths
            for checkpoint in (root / "checkpoints").glob(f"job-*{_CHECKPOINT_SUFFIX}")
        }
    )
    results = []
    corrupt = []
    for file in files:
        try:
            value = _read_json(file)
        except Exception as exc:
            corrupt.append(
                {
                    "status": "corrupt",
                    "error": str(exc),
                    "source_file": str(file),
                }
            )
            continue
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("job"), Mapping)
            or value["job"].get("job_id") is None
        ):
            corrupt.append(
                {
                    "status": "corrupt",
                    "error": "Checkpoint lacks job metadata.",
                    "source_file": str(file),
                }
            )
            continue
        results.append(dict(value))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(
            str(result["job"]["job_id"]),
            [],
        ).append(result)
    deduplicated = []
    for job_id, group in grouped.items():
        if len(group) > 1:
            fingerprints = {_checkpoint_fingerprint(item) for item in group}
            if strict and len(fingerprints) > 1:
                _stop(f"Conflicting checkpoints were found for job `{job_id}`.")
        deduplicated.append(group[-1])
    results = deduplicated

    if plan is None:
        candidate = paths[0] / _PLAN_FILE
        if candidate.exists():
            plan = read_validation_job_manifest(paths[0])

    estimates = _bind_rows([_annotate_frame(result.get("estimates"), result) for result in results])
    diagnostics = _bind_rows([_annotate_frame(result.get("diagnostics"), result) for result in results])
    predictions = _bind_rows([_annotate_frame(result.get("predictions"), result) for result in results])
    draws = _bind_rows([_annotate_frame(result.get("draws"), result) for result in results])

    job_rows = []
    for result in results:
        row = dict(result["job"])
        row.update(
            {
                "status": result["status"],
                "stage": result["stage"],
                "elapsed_seconds": result["elapsed_seconds"],
                "warning_count": len(result.get("warnings", [])),
                "message_count": len(result.get("messages", [])),
                "error": result.get("error"),
            }
        )
        job_rows.append(row)

    return EyeValidationCollection(
        plan=plan,
        paths=[str(item) for item in paths],
        results=results,
        jobs=pd.DataFrame(job_rows),
        estimates=estimates,
        diagnostics=diagnostics,
        predictions=predictions,
        draws=draws,
        corrupt=corrupt,
        collected_utc=_now_utc(),
    )


def prune_validation_checkpoints(
    path,
    statuses=("corrupt", "locked"),
    dry_run=True,
):
    """Report or delete checkpoints with selected execution statuses."""
    output = Path(path).expanduser().resolve()
    if not output.is_dir():
        _stop(f"Validation path does not exist: {output}")
    selected_statuses = {str(value) for value in _as_levels(statuses)}
    rows = []
    for file in sorted((output / "checkpoints").glob(f"job-*{_CHECKPOINT_SUFFIX}")):
        try:
            result = _read_json(file)
            status = str(result.get("status", "corrupt"))
        except Exception:
            status = "corrupt"
        remove = status in selected_statuses
        rows.append(
            {
                "file": str(file.resolve()),
                "status": status,
                "remove": bool(remove),
            }
        )
        if remove and not dry_run:
            file.unlink(missing_ok=True)
    return pd.DataFrame(
        rows,
        columns=["file", "status", "remove"],
    )
