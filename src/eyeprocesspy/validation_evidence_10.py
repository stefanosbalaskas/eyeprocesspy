"""Advanced validation evidence from frozen ``R/021-validation-program.R``.

This tranche ports the scientific-promotion evidence gate, simulation-based
calibration (SBC), and the licence-gated Raven empirical-reproduction harness.
The functions are validation infrastructure: passing a software test is not
silently re-labelled as empirical or scientific validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import EyeProcessValidationError
from .validation_program_10 import model_validation_summary

__all__ = [
    "advanced_model_evidence_spec",
    "audit_advanced_model_evidence",
    "raven_reproduction_spec",
    "run_raven_reproduction",
    "sbc_summary",
    "simulation_based_calibration",
    "write_advanced_model_evidence_report",
]


_DEFAULT_ADVANCED_MODELS = (
    "fit_joint_process_model",
    "fit_shared_process_factor",
    "fit_strategy_mixture",
    "fit_process_irt",
    "fit_pupil_informed_irt",
    "fit_multimodal_irt",
    "fit_dynamic_aoi_model",
    "fit_gaze_weighted_choice",
    "fit_dynamic_irtree",
    "fit_joint_functional_pupil_irt",
    "fit_theory_strategy_irt",
    "fit_gaze_diffusion_irt",
)


@dataclass(slots=True, frozen=True)
class _AdvancedEvidenceSpec:
    models: tuple[str, ...]
    require_recovery: bool
    require_calibration: bool
    require_misspecification: bool
    require_grouped_validation: bool
    require_engine_equivalence: bool
    require_empirical_reproduction: bool
    require_sensitivity: bool


@dataclass(slots=True, frozen=True)
class _RavenReproductionSpec:
    data_path: str
    response: str
    strategy_features: tuple[str, ...]
    published_targets: dict[str, float] | None
    licence_reviewed: bool
    citation: Any


class _SBC(dict):
    eyeprocess_class = "eye_sbc"


class _EmpiricalReproduction(dict):
    eyeprocess_class = "eye_empirical_reproduction"


def _raise(message: str) -> None:
    raise EyeProcessValidationError(message)


def _nonempty_names(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    try:
        output = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise EyeProcessValidationError(f"`{name}` must contain non-empty values.") from exc
    if not output or any(not value or value.lower() == "nan" for value in output):
        _raise(f"`{name}` must contain non-empty values.")
    return output


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _class_name(value: Any) -> str | None:
    marker = getattr(value, "eyeprocess_class", None)
    if isinstance(marker, str):
        return marker
    if isinstance(value, pd.DataFrame):
        marker = value.attrs.get("eyeprocess_class")
        return marker if isinstance(marker, str) else None
    if isinstance(value, Mapping):
        for key in ("eyeprocess_class", "_eyeprocess_class"):
            marker = value.get(key)
            if isinstance(marker, str):
                return marker
    return None


def _mapping_component(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def advanced_model_evidence_spec(
    models=_DEFAULT_ADVANCED_MODELS,
    require_recovery=True,
    require_calibration=True,
    require_misspecification=True,
    require_grouped_validation=True,
    require_engine_equivalence=True,
    require_empirical_reproduction=True,
    require_sensitivity=True,
):
    """Specify evidence required to promote advanced model interfaces."""
    names = _unique(_nonempty_names(models, "models"))
    return _AdvancedEvidenceSpec(
        models=names,
        require_recovery=bool(require_recovery),
        require_calibration=bool(require_calibration),
        require_misspecification=bool(require_misspecification),
        require_grouped_validation=bool(require_grouped_validation),
        require_engine_equivalence=bool(require_engine_equivalence),
        require_empirical_reproduction=bool(require_empirical_reproduction),
        require_sensitivity=bool(require_sensitivity),
    )


def _is_model_validation(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("runs"), pd.DataFrame) and value.get("spec") is not None


def _recovery_pass(value: Any) -> bool:
    if not _is_model_validation(value):
        return False
    summary = model_validation_summary(value)
    if summary.empty:
        return False
    coverage = pd.to_numeric(summary["coverage"], errors="coerce")
    spec = value["spec"]
    min_coverage = getattr(spec, "min_coverage", None)
    if min_coverage is None:
        return False
    return bool(
        summary["status"].astype("string").eq("pass").all()
        and np.isfinite(coverage.to_numpy(dtype=float)).all()
        and coverage.ge(float(min_coverage)).all()
    )


def _sbc_pass(value: Any) -> bool:
    if _class_name(value) != "eye_sbc":
        return False
    summary = sbc_summary(value)
    return bool(not summary.empty and summary["status"].astype("string").eq("pass").all())


def _misspecification_pass(value: Any) -> bool:
    if _is_model_validation(value):
        summary = model_validation_summary(value)
        if "expected_failure" not in summary.columns:
            return False
        expected = summary["expected_failure"].astype("boolean")
        mask = expected.fillna(False).to_numpy(dtype=bool)
        if not mask.any():
            return False
        return bool(summary.loc[mask, "status"].astype("string").eq("fail").all())

    if isinstance(value, pd.DataFrame) and {
        "expected_failure",
        "detected",
    } <= set(value.columns):
        expected = value["expected_failure"].astype("boolean")
        detected = value["detected"].astype("boolean")
        mask = expected.fillna(False).to_numpy(dtype=bool)
        if not mask.any():
            return False
        relevant = detected[mask].dropna()
        return bool(not relevant.empty and relevant.astype(bool).all())

    return False


def _grouped_validation_pass(value: Any) -> bool:
    if _class_name(value) not in {"eye_grouped_cv", "eye_crossed_grouped_cv"}:
        return False
    results = _mapping_component(value, "results")
    if not isinstance(results, pd.DataFrame) or results.empty:
        return False
    if "score" not in results.columns:
        return False
    score = pd.to_numeric(results["score"], errors="coerce")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        return False
    if "error" not in results.columns:
        return True
    errors = results["error"].astype("string")
    return bool((errors.isna() | errors.fillna("").str.len().eq(0)).all())


def _engine_pass(value: Any) -> bool:
    if _class_name(value) != "eye_engine_comparison":
        return False
    estimates = _mapping_component(value, "estimates")
    if not isinstance(estimates, pd.DataFrame) or estimates.empty:
        return False
    if "equivalent" not in estimates.columns:
        return False
    equivalent = estimates["equivalent"].astype("boolean")
    observed = equivalent.dropna()
    return bool(not observed.empty and observed.astype(bool).all())


def _empirical_pass(value: Any) -> bool:
    if _class_name(value) != "eye_empirical_reproduction":
        return False
    comparison = _mapping_component(value, "comparison")
    required = {"target", "absolute_difference", "reproduced"}
    if not isinstance(comparison, pd.DataFrame) or not required <= set(comparison.columns):
        return False
    reproduced = comparison["reproduced"].astype("boolean").dropna()
    return bool(not reproduced.empty and reproduced.astype(bool).all())


def _sensitivity_pass(value: Any) -> bool:
    if _class_name(value) != "eye_multiverse":
        return False
    specifications = _mapping_component(value, "specifications")
    results = _mapping_component(value, "results")
    if specifications is None:
        return False
    try:
        n_specifications = len(specifications)
    except TypeError:
        return False
    if n_specifications < 2 or not isinstance(results, pd.DataFrame) or results.empty:
        return False

    numeric = results.select_dtypes(include=[np.number])
    any_finite = any(
        np.isfinite(pd.to_numeric(numeric[column], errors="coerce").to_numpy(dtype=float)).any()
        for column in numeric.columns
    )
    if not any_finite:
        return False
    if "error" not in results.columns:
        return True
    errors = results["error"].astype("string")
    return bool((errors.isna() | errors.fillna("").str.len().eq(0)).all())


def audit_advanced_model_evidence(evidence, spec=None):
    """Audit the independent evidence required for advanced-model promotion."""
    if spec is None:
        spec = advanced_model_evidence_spec()
    if not isinstance(spec, _AdvancedEvidenceSpec):
        raise TypeError("`spec` must be returned by `advanced_model_evidence_spec()`.")
    if not isinstance(evidence, Mapping):
        _raise("`evidence` must be a named mapping.")

    rows: list[dict[str, Any]] = []
    for model in spec.models:
        record = evidence.get(model, {})
        if not isinstance(record, Mapping):
            record = {}

        observed = {
            "recovery": _recovery_pass(record.get("recovery")),
            "calibration": _sbc_pass(record.get("calibration")),
            "misspecification": _misspecification_pass(record.get("misspecification")),
            "grouped_validation": _grouped_validation_pass(record.get("grouped_validation")),
            "engine_equivalence": _engine_pass(record.get("engine_equivalence")),
            "empirical_reproduction": _empirical_pass(record.get("empirical_reproduction")),
            "sensitivity": _sensitivity_pass(record.get("sensitivity")),
        }
        required_flags = {
            "recovery": spec.require_recovery,
            "calibration": spec.require_calibration,
            "misspecification": spec.require_misspecification,
            "grouped_validation": spec.require_grouped_validation,
            "engine_equivalence": spec.require_engine_equivalence,
            "empirical_reproduction": spec.require_empirical_reproduction,
            "sensitivity": spec.require_sensitivity,
        }
        required_results = [observed[name] for name, required in required_flags.items() if required]
        completed = int(sum(required_results))
        required_n = int(len(required_results))
        if not required_n or all(required_results):
            status = "pass"
        elif any(required_results):
            status = "warning"
        else:
            status = "fail"

        rows.append(
            {
                "model": model,
                **observed,
                "completed": completed,
                "required": required_n,
                "status": status,
            }
        )

    output = pd.DataFrame(rows)
    output.attrs["eyeprocess_class"] = "eye_advanced_evidence_audit"
    output.attrs["spec"] = spec
    return output


def write_advanced_model_evidence_report(x, path):
    """Write the frozen advanced-model scientific-evidence Markdown report."""
    if not isinstance(x, pd.DataFrame) or x.attrs.get("eyeprocess_class") != "eye_advanced_evidence_audit":
        _raise("Expected an advanced-model evidence audit.")

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Advanced-model scientific-evidence audit",
        "",
        f"Generated: {generated}",
        "",
        (
            "| Model | Recovery | Calibration | Misspecification | "
            "Grouped validation | Engine equivalence | Empirical reproduction | "
            "Sensitivity | Status |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in x.itertuples(index=False):
        values = [
            row.recovery,
            row.calibration,
            row.misspecification,
            row.grouped_validation,
            row.engine_equivalence,
            row.empirical_reproduction,
            row.sensitivity,
        ]
        values = ["TRUE" if bool(value) else "FALSE" for value in values]
        lines.append(
            "| "
            f"`{row.model}()` | {values[0]} | {values[1]} | {values[2]} | "
            f"{values[3]} | {values[4]} | {values[5]} | {values[6]} | "
            f"{row.status} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return str(target)


def _sbc_failure(replication: int, message: str, draws=np.nan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "replication": replication,
                "parameter": pd.NA,
                "rank": pd.NA,
                "draws": draws,
                "normalized_rank": np.nan,
                "truth": np.nan,
                "posterior_mean": np.nan,
                "posterior_sd": np.nan,
                "error": message,
            }
        ]
    )


def _truth_map(value: Any) -> dict[str, float] | None:
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


def _posterior_frame(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError):
            return None
    return None


def simulation_based_calibration(
    simulator: Callable[..., Any],
    fitter: Callable[[Any], Any],
    posterior_draws: Callable[[Any], Any],
    truth_extractor: Callable[[Any], Any],
    replications=100,
    seed=1,
    **kwargs,
):
    """Run the frozen simulation-based calibration harness."""
    functions = (simulator, fitter, posterior_draws, truth_extractor)
    if not all(callable(function) for function in functions):
        _raise("All SBC components must be callable.")

    try:
        reps = int(replications)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`replications` must be a positive integer.") from exc
    if reps < 1 or float(reps) != float(replications):
        _raise("`replications` must be a positive integer.")

    np.random.seed(int(seed))
    rows: list[pd.DataFrame] = []
    for replication in range(1, reps + 1):
        try:
            simulation = simulator(**kwargs)
        except Exception as exc:
            rows.append(_sbc_failure(replication, str(exc)))
            continue

        try:
            fit = fitter(simulation)
        except Exception as exc:
            rows.append(_sbc_failure(replication, str(exc)))
            continue

        try:
            raw_draws = posterior_draws(fit)
            draws = _posterior_frame(raw_draws)
        except Exception as exc:
            rows.append(_sbc_failure(replication, str(exc)))
            continue

        # Frozen R does not catch truth-extractor failures; preserve that boundary.
        truth_raw = truth_extractor(simulation)
        truth = _truth_map(truth_raw)
        if draws is None or not len(draws.columns) or truth is None:
            rows.append(
                _sbc_failure(
                    replication,
                    "Posterior draws and truth must have matching parameter names.",
                )
            )
            continue

        draw_names = [str(column) for column in draws.columns]
        draws.columns = draw_names
        parameters = [name for name in draw_names if name in truth]
        if not parameters:
            rows.append(
                _sbc_failure(
                    replication,
                    "No matching posterior/truth parameters.",
                    draws=len(draws),
                )
            )
            continue

        parameter_rows: list[dict[str, Any]] = []
        for parameter in parameters:
            values = pd.to_numeric(draws[parameter], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            truth_value = float(truth[parameter])
            if len(values) and np.isfinite(truth_value):
                below = int(np.sum(values < truth_value))
                tied = int(np.sum(values == truth_value))
                tie_offset = int(np.random.randint(0, tied + 1)) if tied else 0
                rank = below + tie_offset
            else:
                rank = pd.NA

            n_draws = int(len(values))
            normalized_rank = (float(rank) + 0.5) / (n_draws + 1) if n_draws and not pd.isna(rank) else np.nan
            parameter_rows.append(
                {
                    "replication": replication,
                    "parameter": parameter,
                    "rank": rank,
                    "draws": n_draws,
                    "normalized_rank": normalized_rank,
                    "truth": truth_value,
                    "posterior_mean": (float(np.mean(values)) if n_draws else np.nan),
                    "posterior_sd": (float(np.std(values, ddof=1)) if n_draws > 1 else np.nan),
                    "error": pd.NA,
                }
            )
        rows.append(pd.DataFrame(parameter_rows))

    ranks = pd.concat(rows, ignore_index=True, sort=False)
    ranks["rank"] = pd.array(ranks["rank"], dtype="Int64")
    output = _SBC(
        ranks=ranks,
        replications=reps,
        seed=seed,
        call=None,
    )
    return output


def sbc_summary(x):
    """Summarize parameter-level SBC ranks and standardized bias."""
    if _class_name(x) != "eye_sbc":
        _raise("Expected an `eye_sbc` object.")
    data = _mapping_component(x, "ranks")
    if not isinstance(data, pd.DataFrame):
        _raise("Expected an `eye_sbc` object.")

    valid_parameters = data["parameter"].dropna().astype(str).unique().tolist()
    rows: list[dict[str, Any]] = []
    for parameter in valid_parameters:
        subset = data[data["parameter"].astype("string").eq(parameter)].copy()
        normalized = pd.to_numeric(subset["normalized_rank"], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(normalized)

        posterior_mean = pd.to_numeric(subset["posterior_mean"], errors="coerce").to_numpy(dtype=float)
        truth = pd.to_numeric(subset["truth"], errors="coerce").to_numpy(dtype=float)
        posterior_sd = pd.to_numeric(subset["posterior_sd"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized = (posterior_mean - truth) / posterior_sd

        successful = int(usable.sum())
        if successful >= 20:
            uniformity_p = float(stats.kstest(normalized[usable], "uniform").pvalue)
        else:
            uniformity_p = np.nan
        mean_rank = float(np.mean(normalized[usable])) if successful else np.nan

        if successful < 20:
            status = "insufficient"
        elif (
            np.isfinite(mean_rank)
            and abs(mean_rank - 0.5) <= 0.10
            and np.isfinite(uniformity_p)
            and uniformity_p >= 0.01
        ):
            status = "pass"
        else:
            status = "fail"

        finite_standardized = np.isfinite(standardized)
        rows.append(
            {
                "parameter": parameter,
                "replications": int(len(subset)),
                "successful": successful,
                "mean_rank": mean_rank,
                "rank_variance": (float(np.var(normalized[usable], ddof=1)) if successful > 1 else np.nan),
                "uniformity_p_value": uniformity_p,
                "mean_standardized_bias": (
                    float(np.mean(standardized[finite_standardized])) if finite_standardized.any() else np.nan
                ),
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def _published_targets(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    targets = _truth_map(value)
    if targets is None or not targets:
        _raise("`published_targets` must be a named finite numeric vector.")
    if not all(np.isfinite(number) for number in targets.values()):
        _raise("`published_targets` must be a named finite numeric vector.")
    return targets


def raven_reproduction_spec(
    data_path,
    response,
    strategy_features,
    published_targets=None,
    licence_reviewed=False,
    citation="10.1016/j.intell.2023.101782",
):
    """Specify the licence-gated published Raven strategy reproduction."""
    path_text = str(data_path).strip() if data_path is not None else ""
    if not path_text:
        _raise("`data_path` must be a non-empty path.")
    response_text = str(response).strip() if response is not None else ""
    if not response_text:
        _raise("`response` must be a non-empty column name.")
    features = _nonempty_names(strategy_features, "strategy_features")
    targets = _published_targets(published_targets)
    normalized = str(Path(path_text).expanduser().resolve(strict=False))
    return _RavenReproductionSpec(
        data_path=normalized,
        response=response_text,
        strategy_features=features,
        published_targets=targets,
        licence_reviewed=bool(licence_reviewed),
        citation=citation,
    )


def _estimate_frame(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        if not value.index.is_unique:
            return None
        return pd.DataFrame(
            {
                "parameter": value.index.astype(str),
                "estimate": value.to_numpy(),
            }
        )
    if isinstance(value, Mapping):
        return pd.DataFrame(
            {
                "parameter": [str(name) for name in value],
                "estimate": list(value.values()),
            }
        )
    return None


def run_raven_reproduction(
    spec,
    importer: Callable[[str], Any],
    fitter: Callable[[Any, _RavenReproductionSpec], Any],
    extractor: Callable[[Any], Any],
    tolerance=0.05,
):
    """Execute a licensed published-model reproduction."""
    if not isinstance(spec, _RavenReproductionSpec):
        _raise("`spec` must be created by `raven_reproduction_spec()`.")
    try:
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`tolerance` must be a finite non-negative value.") from exc
    if not np.isfinite(tol) or tol < 0:
        _raise("`tolerance` must be a finite non-negative value.")

    if not spec.licence_reviewed:
        _raise("Set `licence_reviewed = True` only after reviewing the exact public data/code terms.")

    path = Path(spec.data_path)
    if not path.exists():
        _raise(f"Raven reproduction materials were not found: {spec.data_path}")

    if not all(callable(function) for function in (importer, fitter, extractor)):
        _raise("Importer, fitter, and extractor must be callable.")

    data = importer(spec.data_path)
    fit = fitter(data, spec)
    estimates = _estimate_frame(extractor(fit))
    if estimates is None or not {"parameter", "estimate"} <= set(estimates.columns):
        _raise("Reproduction estimates must contain `parameter` and `estimate`.")

    comparison = estimates.copy()
    comparison["parameter"] = comparison["parameter"].astype(str)
    comparison["estimate"] = pd.to_numeric(comparison["estimate"], errors="coerce")

    if spec.published_targets is not None:
        comparison["target"] = comparison["parameter"].map(spec.published_targets)
        comparison["target"] = pd.to_numeric(comparison["target"], errors="coerce")
        comparison["absolute_difference"] = (comparison["estimate"] - comparison["target"]).abs()
        finite = np.isfinite(comparison["absolute_difference"].to_numpy(dtype=float))
        reproduced = [
            bool(value <= tol) if usable else pd.NA
            for value, usable in zip(comparison["absolute_difference"], finite, strict=True)
        ]
        comparison["reproduced"] = pd.array(reproduced, dtype="boolean")

    return _EmpiricalReproduction(
        spec=spec,
        data=data,
        fit=fit,
        comparison=comparison,
        tolerance=tol,
    )
