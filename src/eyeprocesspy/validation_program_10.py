"""Validation-programme core ported from frozen R/021-validation-program.R.

This module implements the first coherent validation-programme tranche:
multi-vendor evidence auditing/reporting and Monte Carlo model validation.
The implementations preserve the frozen R contracts while using idiomatic
pandas/numpy data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError

__all__ = [
    "audit_vendor_validation",
    "model_validation_spec",
    "model_validation_summary",
    "run_model_validation",
    "vendor_validation_spec",
    "write_vendor_validation_report",
]


@dataclass(slots=True, frozen=True)
class _VendorValidationSpec:
    required_vendors: tuple[str, ...]
    min_cases_per_vendor: int
    min_pass_rate: float
    require_versions: bool
    require_devices: bool
    require_independent_sources: bool
    require_licence_reviewed: bool


@dataclass(slots=True, frozen=True)
class _ModelValidationSpec:
    replications: int
    confidence: float
    max_abs_bias: float
    min_coverage: float
    max_failure_rate: float


class _ModelValidation(dict):
    """Mapping analogue of the frozen R ``eye_model_validation`` object."""

    def __repr__(self) -> str:
        summary = model_validation_summary(self)
        return f"<eye_model_validation runs={len(self['runs'])} groups={len(summary)}>"


def _raise(message: str) -> None:
    raise EyeProcessValidationError(message)


def _as_nonempty_strings(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    try:
        output = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise EyeProcessValidationError(f"`{name}` must contain non-empty values.") from exc
    if not output or any(not value or value.lower() == "nan" for value in output):
        _raise(f"`{name}` must contain non-empty values.")
    return output


def vendor_validation_spec(
    required_vendors=("gazepoint", "tobii", "pupillabs", "eyelink", "smi"),
    min_cases_per_vendor=2,
    min_pass_rate=0.95,
    require_versions=True,
    require_devices=True,
    require_independent_sources=True,
    require_licence_reviewed=True,
):
    """Specify multi-vendor empirical validation requirements."""
    vendors = tuple(value.lower() for value in _as_nonempty_strings(required_vendors, "required_vendors"))
    try:
        cases = int(min_cases_per_vendor)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`min_cases_per_vendor` must be a positive integer.") from exc
    if cases < 1 or float(cases) != float(min_cases_per_vendor):
        _raise("`min_cases_per_vendor` must be a positive integer.")

    try:
        pass_rate = float(min_pass_rate)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`min_pass_rate` must be between zero and one.") from exc
    if not np.isfinite(pass_rate) or not 0 <= pass_rate <= 1:
        _raise("`min_pass_rate` must be between zero and one.")

    return _VendorValidationSpec(
        required_vendors=vendors,
        min_cases_per_vendor=cases,
        min_pass_rate=pass_rate,
        require_versions=bool(require_versions),
        require_devices=bool(require_devices),
        require_independent_sources=bool(require_independent_sources),
        require_licence_reviewed=bool(require_licence_reviewed),
    )


def _extract_corpus_frames(x: Any) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if isinstance(x, pd.DataFrame):
        return x.copy(), None

    if isinstance(x, Mapping) and "summary" in x:
        summary = x["summary"]
        manifest = x.get("manifest")
    else:
        summary = getattr(x, "summary", None)
        manifest = getattr(x, "manifest", None)

    if not isinstance(summary, pd.DataFrame):
        raise TypeError("`x` must be a validation-summary DataFrame or an object containing a DataFrame `summary`.")
    if manifest is not None and not isinstance(manifest, pd.DataFrame):
        raise TypeError("Corpus `manifest` must be a pandas DataFrame.")
    return summary.copy(), None if manifest is None else manifest.copy()


def _complete_text(series: pd.Series) -> bool:
    if series.empty:
        return False
    values = series.astype("string")
    return bool((values.notna() & values.str.strip().ne("")).all())


def _logical_flags(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    return frame[column].astype("boolean")


def audit_vendor_validation(x, spec=None):
    """Audit a multi-vendor validation corpus using the frozen R rules."""
    if spec is None:
        spec = vendor_validation_spec()
    if not isinstance(spec, _VendorValidationSpec):
        raise TypeError("`spec` must be returned by `vendor_validation_spec()`.")

    data, manifest = _extract_corpus_frames(x)
    missing = {"vendor", "status"} - set(data.columns)
    if missing:
        _raise("Validation summary is missing required column(s): " + ", ".join(sorted(missing)))

    if manifest is not None and not manifest.empty and "case_id" in data.columns and "case_id" in manifest.columns:
        metadata_columns = [
            name
            for name in ("independent_source", "licence_reviewed")
            if name in manifest.columns and name not in data.columns
        ]
        if metadata_columns:
            data = data.merge(
                manifest[["case_id", *metadata_columns]],
                how="left",
                on="case_id",
                sort=False,
                validate="many_to_one",
            )

    data["vendor"] = data["vendor"].astype("string").str.lower()
    observed = [str(value) for value in pd.unique(data["vendor"].dropna()) if str(value).strip()]
    vendors = list(spec.required_vendors)
    vendors.extend(value for value in observed if value not in vendors)

    rows: list[dict[str, Any]] = []
    for vendor in vendors:
        subset = data[data["vendor"].eq(vendor)].copy()
        n_cases = int(len(subset))
        status = subset["status"].astype("string")
        passes = int(status.eq("pass").sum())
        warnings = int(status.eq("warning").sum())
        failures = int(status.eq("fail").sum())
        pass_rate = passes / n_cases if n_cases else 0.0

        versions_ok = not spec.require_versions or (
            "software_version" in subset.columns and n_cases > 0 and _complete_text(subset["software_version"])
        )
        devices_ok = not spec.require_devices or (
            "device_model" in subset.columns and n_cases > 0 and _complete_text(subset["device_model"])
        )

        independent = _logical_flags(subset, "independent_source")
        independent_ok = not spec.require_independent_sources or (
            n_cases > 0 and bool(independent.notna().all()) and bool(independent.fillna(False).all())
        )
        independent_cases = int(independent.fillna(False).sum())

        reviewed = _logical_flags(subset, "licence_reviewed")
        licence_ok = not spec.require_licence_reviewed or (
            n_cases > 0 and bool(reviewed.notna().all()) and bool(reviewed.fillna(False).all())
        )
        reviewed_cases = int(reviewed.fillna(False).sum())

        case_ok = n_cases >= spec.min_cases_per_vendor
        rate_ok = pass_rate >= spec.min_pass_rate
        if case_ok and rate_ok and versions_ok and devices_ok and independent_ok and licence_ok:
            overall = "pass"
        elif n_cases > 0 and failures == 0:
            overall = "warning"
        else:
            overall = "fail"

        rows.append(
            {
                "vendor": vendor,
                "cases": n_cases,
                "independent_cases": independent_cases,
                "licence_reviewed_cases": reviewed_cases,
                "passes": passes,
                "warnings": warnings,
                "failures": failures,
                "pass_rate": float(pass_rate),
                "cases_sufficient": bool(case_ok),
                "versions_complete": bool(versions_ok),
                "devices_complete": bool(devices_ok),
                "independent_sources_complete": bool(independent_ok),
                "licences_reviewed": bool(licence_ok),
                "status": overall,
            }
        )

    output = pd.DataFrame(rows)
    output.attrs["eyeprocess_class"] = "eye_vendor_validation"
    output.attrs["spec"] = spec
    return output


def write_vendor_validation_report(x, path):
    """Write the frozen multi-vendor validation Markdown report."""
    if not isinstance(x, pd.DataFrame) or x.attrs.get("eyeprocess_class") != "eye_vendor_validation":
        _raise("Expected an `eye_vendor_validation` object.")

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Multi-vendor empirical validation audit",
        "",
        f"Generated: {generated}",
        "",
        ("| Vendor | Cases | Independent | Licence reviewed | Passes | Warnings | Failures | Pass rate | Status |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in x.itertuples(index=False):
        lines.append(
            "| "
            f"{row.vendor} | {int(row.cases)} | {int(row.independent_cases)} | "
            f"{int(row.licence_reviewed_cases)} | {int(row.passes)} | "
            f"{int(row.warnings)} | {int(row.failures)} | "
            f"{float(row.pass_rate):.3f} | {row.status} |"
        )
    lines.extend(
        [
            "",
            (
                "A fixture-tested adapter is not classified as empirically "
                "validated unless independent real-export cases satisfy the "
                "declared thresholds."
            ),
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return str(target)


def model_validation_spec(
    replications=100,
    confidence=0.95,
    max_abs_bias=0.10,
    min_coverage=0.90,
    max_failure_rate=0.05,
):
    """Specify a Monte Carlo model-validation programme."""
    try:
        reps = int(replications)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`replications` must be a positive integer.") from exc
    if reps < 1 or float(reps) != float(replications):
        _raise("`replications` must be a positive integer.")

    bounded = {}
    for name, value in (
        ("confidence", confidence),
        ("min_coverage", min_coverage),
        ("max_failure_rate", max_failure_rate),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise EyeProcessValidationError(
                "Confidence, coverage, and failure-rate thresholds must be between zero and one."
            ) from exc
        if not np.isfinite(numeric) or not 0 <= numeric <= 1:
            _raise("Confidence, coverage, and failure-rate thresholds must be between zero and one.")
        bounded[name] = numeric

    try:
        bias = float(max_abs_bias)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`max_abs_bias` must be a finite non-negative value.") from exc
    if not np.isfinite(bias) or bias < 0:
        _raise("`max_abs_bias` must be a finite non-negative value.")

    return _ModelValidationSpec(
        replications=reps,
        confidence=bounded["confidence"],
        max_abs_bias=bias,
        min_coverage=bounded["min_coverage"],
        max_failure_rate=bounded["max_failure_rate"],
    )


def _grid_rows(grid: Any) -> list[dict[str, Any]]:
    if grid is None:
        return [{}]

    if isinstance(grid, pd.DataFrame):
        if grid.empty:
            _raise("`grid` must contain at least one scenario.")
        return [{name: row[name] for name in grid.columns} for _, row in grid.iterrows()]

    if isinstance(grid, Mapping):
        if not grid:
            return [{}]
        names = list(grid)
        values: list[list[Any]] = []
        for name in names:
            value = grid[name]
            if isinstance(value, (str, bytes)) or np.isscalar(value):
                sequence = [value]
            else:
                sequence = list(value)
            if not sequence:
                _raise("`grid` must contain at least one scenario.")
            values.append(sequence)
        return [dict(zip(names, combination, strict=True)) for combination in product(*values)]

    if isinstance(grid, Sequence) and not isinstance(grid, (str, bytes)):
        rows = list(grid)
        if not rows:
            _raise("`grid` must contain at least one scenario.")
        if all(isinstance(row, Mapping) for row in rows):
            return [dict(row) for row in rows]

    _raise("`grid` must be a DataFrame, mapping, sequence of mappings, or None.")
    raise AssertionError("unreachable")


def _failure_row(
    *,
    scenario: int,
    replication: int,
    parameter: str,
    error: BaseException | str,
    scenario_values: Mapping[str, Any],
) -> pd.DataFrame:
    row = {
        "scenario": scenario,
        "replication": replication,
        "parameter": parameter,
        "estimate": np.nan,
        "truth": np.nan,
        "lower": np.nan,
        "upper": np.nan,
        "converged": False,
        "error": str(error),
    }
    row.update(scenario_values)
    return pd.DataFrame([row])


def _estimate_frame(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        if value.index.is_unique:
            return pd.DataFrame(
                {
                    "parameter": value.index.astype(str),
                    "estimate": pd.to_numeric(value.to_numpy(), errors="coerce"),
                }
            )
        return None

    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(
                {
                    "parameter": [str(name) for name in value],
                    "estimate": [value[name] for name in value],
                }
            )
        except Exception:
            return None

    return None


def _truth_mapping(value: Any) -> dict[str, float] | None:
    if isinstance(value, pd.Series):
        if not value.index.is_unique:
            return None
        items = value.items()
    elif isinstance(value, Mapping):
        items = value.items()
    else:
        return None

    result: dict[str, float] = {}
    try:
        for name, item in items:
            result[str(name)] = float(item)
    except (TypeError, ValueError):
        return None
    return result


def _bind_rows(rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    columns: list[str] = []
    for frame in rows:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
    normalized = []
    for frame in rows:
        current = frame.copy()
        for column in columns:
            if column not in current.columns:
                current[column] = pd.NA
        normalized.append(current[columns])
    return pd.concat(normalized, ignore_index=True, sort=False)


def run_model_validation(
    simulator: Callable[..., Any],
    fitter: Callable[[Any], Any],
    extractor: Callable[[Any], Any],
    truth_extractor: Callable[[Any], Any],
    grid=None,
    spec=None,
    seed=1,
    continue_on_error=True,
):
    """Run parameter recovery, coverage, and failure validation."""
    functions = {
        "simulator": simulator,
        "fitter": fitter,
        "extractor": extractor,
        "truth_extractor": truth_extractor,
    }
    if not all(callable(function) for function in functions.values()):
        _raise("All simulator/fitter/extractor arguments must be callable.")

    if spec is None:
        spec = model_validation_spec()
    if not isinstance(spec, _ModelValidationSpec):
        raise TypeError("`spec` must be returned by `model_validation_spec()`.")

    scenarios = _grid_rows(grid)
    rng = np.random.default_rng(int(seed))
    rows: list[pd.DataFrame] = []

    for scenario_number, scenario_values in enumerate(scenarios, start=1):
        for replication in range(1, spec.replications + 1):
            sim_seed = int(rng.integers(1, np.iinfo(np.int32).max))
            np.random.seed(sim_seed)

            try:
                simulation = simulator(**scenario_values)
            except Exception as exc:
                rows.append(
                    _failure_row(
                        scenario=scenario_number,
                        replication=replication,
                        parameter=".simulation",
                        error=exc,
                        scenario_values=scenario_values,
                    )
                )
                if not continue_on_error:
                    raise
                continue

            try:
                fit = fitter(simulation)
            except Exception as exc:
                rows.append(
                    _failure_row(
                        scenario=scenario_number,
                        replication=replication,
                        parameter=".fit",
                        error=exc,
                        scenario_values=scenario_values,
                    )
                )
                if not continue_on_error:
                    raise
                continue

            try:
                extracted = extractor(fit)
            except Exception as exc:
                rows.append(
                    _failure_row(
                        scenario=scenario_number,
                        replication=replication,
                        parameter=".extract",
                        error=exc,
                        scenario_values=scenario_values,
                    )
                )
                if not continue_on_error:
                    raise
                continue

            estimates = _estimate_frame(extracted)
            if estimates is None or not {"parameter", "estimate"} <= set(estimates.columns):
                message = (
                    "Extractor must return a DataFrame with parameter and estimate columns or a named mapping/Series."
                )
                rows.append(
                    _failure_row(
                        scenario=scenario_number,
                        replication=replication,
                        parameter=".extract",
                        error=message,
                        scenario_values=scenario_values,
                    )
                )
                if not continue_on_error:
                    _raise(message)
                continue

            try:
                truth_raw = truth_extractor(simulation)
            except Exception as exc:
                rows.append(
                    _failure_row(
                        scenario=scenario_number,
                        replication=replication,
                        parameter=".truth",
                        error=exc,
                        scenario_values=scenario_values,
                    )
                )
                if not continue_on_error:
                    raise
                continue

            truth = _truth_mapping(truth_raw)
            if truth is None:
                message = "Truth extractor must return named values."
                rows.append(
                    _failure_row(
                        scenario=scenario_number,
                        replication=replication,
                        parameter=".truth",
                        error=message,
                        scenario_values=scenario_values,
                    )
                )
                if not continue_on_error:
                    _raise(message)
                continue

            estimates = estimates.copy()
            estimates["parameter"] = estimates["parameter"].astype(str)
            estimates["estimate"] = pd.to_numeric(estimates["estimate"], errors="coerce")
            estimates["truth"] = estimates["parameter"].map(truth)
            if "lower" not in estimates.columns:
                estimates["lower"] = np.nan
            if "upper" not in estimates.columns:
                estimates["upper"] = np.nan
            estimates["lower"] = pd.to_numeric(estimates["lower"], errors="coerce")
            estimates["upper"] = pd.to_numeric(estimates["upper"], errors="coerce")
            estimates["scenario"] = scenario_number
            estimates["replication"] = replication
            estimates["converged"] = True
            estimates["error"] = pd.NA
            for name, value in scenario_values.items():
                estimates[name] = value
            rows.append(estimates)

    runs = _bind_rows(rows)
    for column in ("estimate", "truth", "lower", "upper"):
        if column not in runs.columns:
            runs[column] = np.nan
        runs[column] = pd.to_numeric(runs[column], errors="coerce")
    if "converged" not in runs.columns:
        runs["converged"] = False
    runs["converged"] = runs["converged"].fillna(False).astype(bool)

    runs["bias"] = runs["estimate"] - runs["truth"]
    runs["squared_error"] = runs["bias"] ** 2
    interval_available = (
        np.isfinite(runs["lower"].to_numpy(dtype=float))
        & np.isfinite(runs["upper"].to_numpy(dtype=float))
        & np.isfinite(runs["truth"].to_numpy(dtype=float))
    )
    covered = np.full(len(runs), pd.NA, dtype=object)
    if interval_available.any():
        truth_values = runs["truth"].to_numpy(dtype=float)
        lower_values = runs["lower"].to_numpy(dtype=float)
        upper_values = runs["upper"].to_numpy(dtype=float)
        covered[interval_available] = (lower_values[interval_available] <= truth_values[interval_available]) & (
            upper_values[interval_available] >= truth_values[interval_available]
        )
    runs["covered"] = pd.Series(covered, dtype="boolean")

    return _ModelValidation(
        runs=runs,
        spec=spec,
        grid=grid,
        call=None,
    )


def model_validation_summary(x):
    """Summarize scenario-by-parameter model-validation metrics."""
    if not isinstance(x, Mapping) or not isinstance(x.get("spec"), _ModelValidationSpec):
        _raise("Expected an `eye_model_validation` object.")

    data = x.get("runs")
    if not isinstance(data, pd.DataFrame):
        _raise("Expected an `eye_model_validation` object.")
    if data.empty:
        return pd.DataFrame()

    excluded = {
        "replication",
        "parameter",
        "estimate",
        "truth",
        "std_error",
        "lower",
        "upper",
        "converged",
        "error",
        "bias",
        "squared_error",
        "covered",
    }
    scenario_columns = [column for column in data.columns if column not in excluded]
    keys: list[str] = []
    for column in ["scenario", *scenario_columns, "parameter"]:
        if column in data.columns and column not in keys:
            keys.append(column)

    rows: list[dict[str, Any]] = []
    grouped = data.groupby(keys, dropna=False, sort=False)
    for key_values, group in grouped:
        if len(keys) == 1:
            key_values = (key_values,)
        base = dict(zip(keys, key_values, strict=True))

        converged = group["converged"].fillna(False).astype(bool).to_numpy()
        estimate = pd.to_numeric(group["estimate"], errors="coerce").to_numpy(dtype=float)
        truth = pd.to_numeric(group["truth"], errors="coerce").to_numpy(dtype=float)
        ok = converged & np.isfinite(estimate) & np.isfinite(truth)
        bias = pd.to_numeric(group["bias"], errors="coerce").to_numpy(dtype=float)
        squared = pd.to_numeric(group["squared_error"], errors="coerce").to_numpy(dtype=float)

        covered_series = group["covered"].astype("boolean")
        coverage_mask = ok & covered_series.notna().to_numpy()
        coverage = (
            float(covered_series[coverage_mask].astype(bool).to_numpy().mean()) if coverage_mask.any() else np.nan
        )

        row = {
            **base,
            "replications": int(group["replication"].nunique(dropna=True)),
            "successful": int(ok.sum()),
            "failure_rate": float((~converged).mean()),
            "bias": float(np.mean(bias[ok])) if ok.any() else np.nan,
            "absolute_bias": (float(np.mean(np.abs(bias[ok]))) if ok.any() else np.nan),
            "rmse": (float(np.sqrt(np.mean(squared[ok]))) if ok.any() else np.nan),
            "coverage": coverage,
        }
        rows.append(row)

    output = pd.DataFrame(rows)
    spec = x["spec"]
    fail = (
        output["successful"].lt(1)
        | ~np.isfinite(output["absolute_bias"].to_numpy(dtype=float))
        | output["failure_rate"].gt(spec.max_failure_rate)
        | output["absolute_bias"].gt(spec.max_abs_bias)
        | (np.isfinite(output["coverage"].to_numpy(dtype=float)) & output["coverage"].lt(spec.min_coverage))
    )
    output["status"] = np.where(fail, "fail", "pass")
    output.attrs["eyeprocess_class"] = "eye_model_validation_summary"
    return output
