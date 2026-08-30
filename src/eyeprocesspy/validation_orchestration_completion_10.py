"""Final validation-orchestration closure ported from frozen R/023.

This module completes the research-scale validation summaries, scientific
completion gates, diagnostic plots, model-promotion audits, and release reports.
It reuses the 966 execution/checkpoint core without changing that frozen module.
"""

from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import chi2

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .validation_orchestration_10 import (
    EyeValidationCollection,
    _now_utc,
    _scalar_int,
    _scalar_num,
)

__all__ = [
    "audit_model_promotion",
    "audit_validation_completion",
    "model_promotion_spec",
    "plot_interval_coverage",
    "plot_parameter_recovery",
    "plot_sbc_rank",
    "plot_validation_failures",
    "plot_validation_runtime",
    "validation_calibration_summary",
    "validation_failure_summary",
    "validation_recovery_summary",
    "validation_runtime_summary",
    "validation_sbc_summary",
    "validation_thresholds",
    "write_model_promotion_report",
    "write_validation_release_report",
]


class _EyeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class EyeValidationThresholds(_EyeDict):
    eyeprocess_class = "eye_validation_thresholds"


class EyeValidationCompletionAudit(_EyeDict):
    eyeprocess_class = "eye_validation_completion_audit"


class EyeModelPromotionSpec(_EyeDict):
    eyeprocess_class = "eye_model_promotion_spec"


class EyeModelPromotionAudit(_EyeDict):
    eyeprocess_class = "eye_model_promotion_audit"


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _is_collection(value: Any) -> bool:
    return isinstance(value, EyeValidationCollection) or (
        getattr(value, "eyeprocess_class", None) == "eye_validation_collection"
    )


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _by_list(by: Any) -> list[str]:
    if by is None:
        return []
    if isinstance(by, str):
        return [by]
    return [str(value) for value in list(by)]


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _group_summary(
    data: pd.DataFrame,
    keys: Sequence[str],
    summarizer,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    keys = [key for key in keys if key in data.columns]
    if not keys:
        return pd.DataFrame([summarizer(data)])
    rows = []
    group_key = keys[0] if len(keys) == 1 else keys
    grouped = data.groupby(
        group_key,
        dropna=False,
        sort=False,
    )
    for values, frame in grouped:
        if len(keys) == 1:
            values = (values,)
        base = dict(zip(keys, values, strict=True))
        base.update(summarizer(frame))
        rows.append(base)
    return pd.DataFrame(rows)


def validation_recovery_summary(x, by=()):
    """Summarize parameter bias, RMSE, coverage, and standard-error recovery."""
    data = x["estimates"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame):
        _stop("Expected a validation collection or estimates data frame.")
    required = {"parameter", "estimate", "truth"}
    if not required.issubset(data.columns):
        _stop("Recovery data require: " + ", ".join(["parameter", "estimate", "truth"]))
    data = data.copy()
    if "status" not in data:
        data["status"] = "complete"
    if "std_error" not in data:
        data["std_error"] = np.nan

    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data["truth"] = pd.to_numeric(data["truth"], errors="coerce")
    data["std_error"] = pd.to_numeric(data["std_error"], errors="coerce")

    if "relative_bias" not in data:
        denominator = data["truth"].abs()
        data["relative_bias"] = np.where(
            np.isfinite(denominator) & (denominator > np.sqrt(np.finfo(float).eps)),
            (data["estimate"] - data["truth"]) / denominator,
            np.nan,
        )

    if "covered" not in data:
        if {"lower", "upper"}.issubset(data.columns):
            lower = pd.to_numeric(data["lower"], errors="coerce")
            upper = pd.to_numeric(data["upper"], errors="coerce")
            data["covered"] = (
                np.isfinite(lower)
                & np.isfinite(upper)
                & np.isfinite(data["truth"])
                & (lower <= data["truth"])
                & (data["truth"] <= upper)
            )
        else:
            data["covered"] = pd.NA

    if "interval_width" not in data:
        if {"lower", "upper"}.issubset(data.columns):
            data["interval_width"] = pd.to_numeric(data["upper"], errors="coerce") - pd.to_numeric(
                data["lower"], errors="coerce"
            )
        else:
            data["interval_width"] = np.nan

    keys = []
    for key in ["model_family", "scenario_id", *_by_list(by), "parameter"]:
        if key in data.columns and key not in keys:
            keys.append(key)

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        estimate = pd.to_numeric(frame["estimate"], errors="coerce")
        truth = pd.to_numeric(frame["truth"], errors="coerce")
        status = frame["status"].astype(str)
        ok = np.isfinite(estimate) & np.isfinite(truth) & ~status.isin(["failed", "nonconverged"])
        covered = pd.Series(frame["covered"], index=frame.index)
        covered_ok = ok & covered.notna()
        model_se = ok & np.isfinite(pd.to_numeric(frame["std_error"], errors="coerce"))

        est_ok = estimate.loc[ok].to_numpy(dtype=float)
        truth_ok = truth.loc[ok].to_numpy(dtype=float)
        delta = est_ok - truth_ok

        if "replication" in frame:
            replications = int(frame["replication"].nunique(dropna=True))
        else:
            replications = int(len(frame))

        relative = pd.to_numeric(frame["relative_bias"], errors="coerce")
        relative_mask = ok & np.isfinite(relative)

        interval_width = pd.to_numeric(frame["interval_width"], errors="coerce")
        se_values = pd.to_numeric(frame["std_error"], errors="coerce")

        empirical_sd = float(np.std(est_ok, ddof=1)) if len(est_ok) > 1 else math.nan
        mean_model_se = float(se_values.loc[model_se].mean()) if bool(model_se.any()) else math.nan
        coverage = (
            float(covered.loc[covered_ok].astype(bool).astype(float).mean()) if bool(covered_ok.any()) else math.nan
        )
        mean_interval_width = float(interval_width.loc[covered_ok].mean()) if bool(covered_ok.any()) else math.nan

        return {
            "replications": replications,
            "successful": int(ok.sum()),
            "failure_rate": float((~ok).mean()) if len(frame) else math.nan,
            "mean_truth": float(np.mean(truth_ok)) if len(truth_ok) else math.nan,
            "mean_estimate": float(np.mean(est_ok)) if len(est_ok) else math.nan,
            "bias": float(np.mean(delta)) if len(delta) else math.nan,
            "absolute_bias": (float(np.mean(np.abs(delta))) if len(delta) else math.nan),
            "relative_bias": (float(relative.loc[relative_mask].mean()) if bool(relative_mask.any()) else math.nan),
            "rmse": (float(np.sqrt(np.mean(delta**2))) if len(delta) else math.nan),
            "empirical_sd": empirical_sd,
            "mean_model_se": mean_model_se,
            "se_ratio": (
                float(mean_model_se / empirical_sd)
                if math.isfinite(mean_model_se) and math.isfinite(empirical_sd) and empirical_sd != 0
                else math.nan
            ),
            "coverage": coverage,
            "mean_interval_width": mean_interval_width,
        }

    return _group_summary(data, keys, summarize)


def validation_failure_summary(
    x,
    by=("model_family", "scenario_id"),
):
    """Summarize execution failures, nonconvergence, locks, and warnings."""
    data = x["jobs"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame):
        _stop("Expected a validation collection or job table.")
    data = data.copy()
    if "status" not in data:
        _stop("Failure data require a `status` column.")
    if "warning_count" not in data:
        data["warning_count"] = 0
    keys = [key for key in _by_list(by) if key in data.columns]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        status = frame["status"].astype(str)
        warning_count = pd.to_numeric(frame["warning_count"], errors="coerce").fillna(0)
        return {
            "jobs": int(len(frame)),
            "successes": int(status.eq("complete").sum()),
            "nonconverged": int(status.eq("nonconverged").sum()),
            "failed": int(status.eq("failed").sum()),
            "locked": int(status.eq("locked").sum()),
            "failure_rate": float(status.ne("complete").mean()),
            "warning_rate": float(warning_count.gt(0).mean()),
        }

    return _group_summary(data, keys, summarize)


def validation_runtime_summary(
    x,
    by=("model_family", "scenario_id"),
):
    """Summarize validation runtime by model family/scenario."""
    data = x["jobs"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or "elapsed_seconds" not in data:
        _stop("Runtime data are unavailable.")
    data = data.copy()
    keys = [key for key in _by_list(by) if key in data.columns]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        values = _finite_numeric(frame["elapsed_seconds"]).to_numpy(dtype=float)
        return {
            "jobs": int(len(frame)),
            "total_seconds": float(values.sum()) if len(values) else 0.0,
            "mean_seconds": float(values.mean()) if len(values) else math.nan,
            "median_seconds": (float(np.median(values)) if len(values) else math.nan),
            "p90_seconds": (float(np.quantile(values, 0.90)) if len(values) else math.nan),
            "max_seconds": float(values.max()) if len(values) else math.nan,
        }

    return _group_summary(data, keys, summarize)


def _weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return math.nan
    return float(np.sum(values * weights) / total)


def _calibration_coefficients(y, p, w) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    lp = logit(p)

    def nll_intercept(beta):
        eta = beta[0] + lp
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))

    def nll_slope(beta):
        eta = beta[0] + beta[1] * lp
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))

    intercept_fit = minimize(
        nll_intercept,
        x0=np.array([0.0]),
        method="BFGS",
    )
    slope_fit = minimize(
        nll_slope,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )
    intercept = float(intercept_fit.x[0]) if intercept_fit.success else math.nan
    slope = float(slope_fit.x[1]) if slope_fit.success else math.nan
    return intercept, slope


def validation_calibration_summary(
    x,
    by=("model_family", "scenario_id"),
    bins=10,
):
    """Summarize binary prediction calibration, Brier/log loss, and ECE."""
    data = x["predictions"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or not {
        "observed",
        "predicted",
    }.issubset(data.columns):
        _stop("Prediction data require `observed` and `predicted` columns.")
    bins = _scalar_int(bins, "bins", 2)
    data = data.copy()
    data["observed"] = pd.to_numeric(data["observed"], errors="coerce")
    finite_observed = data["observed"][np.isfinite(data["observed"])]
    if not finite_observed.isin([0, 1]).all():
        _stop("Calibration outcomes must be binary 0/1.")
    data["predicted"] = np.clip(
        pd.to_numeric(data["predicted"], errors="coerce"),
        1e-8,
        1 - 1e-8,
    )
    if "weight" not in data:
        data["weight"] = 1.0
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce")
    keys = [key for key in _by_list(by) if key in data.columns]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        ok = (
            np.isfinite(frame["observed"])
            & np.isfinite(frame["predicted"])
            & np.isfinite(frame["weight"])
            & (frame["weight"] > 0)
        )
        z = frame.loc[ok]
        if z.empty:
            return {
                "n": 0,
                "calibration_intercept": math.nan,
                "calibration_slope": math.nan,
                "brier": math.nan,
                "log_loss": math.nan,
                "ece": math.nan,
            }

        y = z["observed"].to_numpy(dtype=float)
        p = z["predicted"].to_numpy(dtype=float)
        w = z["weight"].to_numpy(dtype=float)
        intercept, slope = _calibration_coefficients(y, p, w)

        edges = np.linspace(0.0, 1.0, bins + 1)
        assignments = np.searchsorted(edges, p, side="right") - 1
        assignments = np.clip(assignments, 0, bins - 1)
        ece = 0.0
        total_weight = float(w.sum())
        for bin_index in range(bins):
            mask = assignments == bin_index
            if not np.any(mask):
                continue
            bin_weight = float(w[mask].sum())
            ece += bin_weight / total_weight * abs(_weighted_mean(y[mask], w[mask]) - _weighted_mean(p[mask], w[mask]))

        return {
            "n": int(len(z)),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "brier": _weighted_mean((y - p) ** 2, w),
            "log_loss": -_weighted_mean(
                y * np.log(p) + (1 - y) * np.log1p(-p),
                w,
            ),
            "ece": float(ece),
        }

    return _group_summary(data, keys, summarize)


def _sbc_ranks(data: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    grouping = ["job_id", "parameter"]
    for _, frame in data.groupby(grouping, dropna=False, sort=False):
        draw = pd.to_numeric(frame["draw"], errors="coerce")
        truth = pd.to_numeric(frame["truth"], errors="coerce")
        ok = np.isfinite(draw) & np.isfinite(truth)
        z = frame.loc[ok].copy()
        if z.empty:
            continue
        truth_value = float(pd.to_numeric(z["truth"], errors="coerce").iloc[0])
        draw_values = pd.to_numeric(z["draw"], errors="coerce").to_numpy(dtype=float)
        row = {key: z.iloc[0][key] for key in by if key in z.columns}
        row.update(
            {
                "job_id": z.iloc[0]["job_id"],
                "parameter": z.iloc[0]["parameter"],
                "rank": (float(np.sum(draw_values < truth_value)) + 0.5 * float(np.sum(draw_values == truth_value))),
                "draws": int(len(draw_values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validation_sbc_summary(
    x,
    by=("model_family", "scenario_id"),
    bins=10,
):
    """Summarize simulation-based-calibration scaled ranks."""
    data = x["draws"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or not {
        "parameter",
        "draw",
        "truth",
        "job_id",
    }.issubset(data.columns):
        _stop("SBC draws require `job_id`, `parameter`, `draw`, and `truth`.")
    bins = _scalar_int(bins, "bins", 2)
    by_keys = [key for key in _by_list(by) if key in data.columns]
    ranks = _sbc_ranks(data, by_keys)
    if ranks.empty:
        return pd.DataFrame()
    keys = [*by_keys, "parameter"]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        rank = pd.to_numeric(frame["rank"], errors="coerce").to_numpy(dtype=float)
        draws = pd.to_numeric(frame["draws"], errors="coerce").to_numpy(dtype=float)
        scaled = (rank + 0.5) / (draws + 1.0)
        counts, _ = np.histogram(
            scaled,
            bins=np.linspace(0.0, 1.0, bins + 1),
        )
        expected = float(counts.sum() / bins)
        chi_square = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else math.nan
        p_value = float(chi2.sf(chi_square, bins - 1)) if math.isfinite(chi_square) else math.nan
        return {
            "replications": int(len(frame)),
            "mean_scaled_rank": float(np.mean(scaled)),
            "rank_variance": (float(np.var(scaled, ddof=1)) if len(scaled) > 1 else math.nan),
            "chi_square": chi_square,
            "p_value": p_value,
        }

    return _group_summary(ranks, keys, summarize)


def validation_thresholds(
    required_replications=100,
    max_failure_rate=0.05,
    max_absolute_bias=0.10,
    max_rmse=math.inf,
    min_coverage=0.90,
    max_coverage=0.99,
    max_rhat=1.01,
    min_ess_bulk=400,
    max_divergence_rate=0.01,
    require_sbc=True,
    require_empirical_reproduction=True,
):
    """Specify completion and scientific-promotion thresholds."""
    return EyeValidationThresholds(
        required_replications=_scalar_int(
            required_replications,
            "required_replications",
            1,
        ),
        max_failure_rate=_scalar_num(
            max_failure_rate,
            "max_failure_rate",
            0,
            1,
        ),
        max_absolute_bias=_scalar_num(
            max_absolute_bias,
            "max_absolute_bias",
            0,
            math.inf,
            finite=False,
        ),
        max_rmse=_scalar_num(
            max_rmse,
            "max_rmse",
            0,
            math.inf,
            finite=False,
        ),
        min_coverage=_scalar_num(
            min_coverage,
            "min_coverage",
            0,
            1,
        ),
        max_coverage=_scalar_num(
            max_coverage,
            "max_coverage",
            0,
            1,
        ),
        max_rhat=_scalar_num(
            max_rhat,
            "max_rhat",
            1,
            math.inf,
        ),
        min_ess_bulk=_scalar_num(
            min_ess_bulk,
            "min_ess_bulk",
            0,
            math.inf,
            finite=False,
        ),
        max_divergence_rate=_scalar_num(
            max_divergence_rate,
            "max_divergence_rate",
            0,
            1,
        ),
        require_sbc=bool(require_sbc is True),
        require_empirical_reproduction=bool(require_empirical_reproduction is True),
    )


def _class_name(value: Any) -> str | None:
    return getattr(value, "eyeprocess_class", None)


def _field(value: Any, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _frame_field(value: Any, name: str) -> pd.DataFrame:
    field = _field(value, name)
    return field if isinstance(field, pd.DataFrame) else pd.DataFrame()


def _evidence_pass(value: Any, kind: str) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    class_name = _class_name(value)
    if kind == "completion" and (
        isinstance(value, EyeValidationCompletionAudit) or class_name == "eye_validation_completion_audit"
    ):
        return _field(value, "status") == "complete"

    if kind == "grouped_validation" and class_name in {
        "eye_grouped_cv",
        "eye_crossed_grouped_cv",
    }:
        results = _frame_field(value, "results")
        return (
            not results.empty
            and "score" in results
            and np.isfinite(pd.to_numeric(results["score"], errors="coerce")).all()
        )

    if kind == "engine_equivalence" and class_name == "eye_engine_comparison":
        estimates = _frame_field(value, "estimates")
        if estimates.empty or "equivalent" not in estimates:
            return False
        equivalent = estimates["equivalent"].astype("boolean").dropna()
        return bool(len(equivalent)) and bool(equivalent.all())

    if kind == "empirical_reproduction" and class_name == "eye_empirical_reproduction":
        comparison = _frame_field(value, "comparison")
        if comparison.empty or "reproduced" not in comparison:
            return False
        reproduced = comparison["reproduced"].astype("boolean").dropna()
        return bool(len(reproduced)) and bool(reproduced.all())

    if kind == "preprocessing_sensitivity" and class_name == "eye_multiverse":
        specifications = _field(value, "specifications", [])
        results = _frame_field(value, "results")
        return len(specifications) >= 2 and not results.empty

    if kind == "multi_vendor" and class_name == "eye_vendor_validation":
        if isinstance(value, pd.DataFrame):
            return not value.empty and "status" in value and value["status"].astype(str).eq("pass").all()

    if isinstance(value, pd.DataFrame) and "pass" in value:
        passed = value["pass"].astype("boolean").dropna()
        return bool(len(value)) and bool(len(passed)) and bool(passed.all())

    if isinstance(value, Mapping):
        if "pass" in value and isinstance(value["pass"], (bool, np.bool_)):
            return bool(value["pass"])
        status = value.get("status")
        if isinstance(status, str):
            return status.lower() in {"pass", "passed", "complete", "success"}

    return False


def _safe_max(values: pd.Series, fallback: float) -> float:
    finite = _finite_numeric(values)
    return float(finite.max()) if len(finite) else fallback


def _safe_min(values: pd.Series, fallback: float) -> float:
    finite = _finite_numeric(values)
    return float(finite.min()) if len(finite) else fallback


def audit_validation_completion(
    x,
    thresholds=None,
    empirical_reproduction=None,
):
    """Audit whether the validation programme passes frozen completion gates."""
    if not _is_collection(x):
        _stop("Expected an `eye_validation_collection`.")
    if thresholds is None:
        thresholds = validation_thresholds()
    if not isinstance(thresholds, EyeValidationThresholds):
        _stop("`thresholds` must be created by `validation_thresholds()`.")

    expected = (
        x["plan"]["jobs"]
        if x.get("plan") is not None and isinstance(x["plan"].get("jobs"), pd.DataFrame)
        else pd.DataFrame()
    )
    jobs = x["jobs"] if isinstance(x.get("jobs"), pd.DataFrame) else pd.DataFrame()
    observed = set(jobs["job_id"].astype(str)) if not jobs.empty and "job_id" in jobs else set()
    if not expected.empty and "job_id" in expected:
        missing_jobs = expected.loc[~expected["job_id"].astype(str).isin(observed)].copy()
    else:
        missing_jobs = pd.DataFrame()

    failure = validation_failure_summary(x) if not jobs.empty else pd.DataFrame()
    estimates = x["estimates"] if isinstance(x.get("estimates"), pd.DataFrame) else pd.DataFrame()
    recovery = validation_recovery_summary(x) if not estimates.empty else pd.DataFrame()
    draws = x["draws"] if isinstance(x.get("draws"), pd.DataFrame) else pd.DataFrame()
    try:
        sbc = validation_sbc_summary(x) if not draws.empty else pd.DataFrame()
    except EyeProcessValidationError:
        sbc = pd.DataFrame()
    diagnostics = x["diagnostics"] if isinstance(x.get("diagnostics"), pd.DataFrame) else pd.DataFrame()

    if not diagnostics.empty and "divergences" in diagnostics:
        divergences = pd.to_numeric(diagnostics["divergences"], errors="coerce")
        finite = divergences[np.isfinite(divergences)]
        divergence_rate = float((finite > 0).mean()) if len(finite) else math.nan
    else:
        divergence_rate = math.nan

    max_rhat = (
        _safe_max(diagnostics["max_rhat"], math.nan)
        if not diagnostics.empty and "max_rhat" in diagnostics
        else math.nan
    )
    min_ess = (
        _safe_min(diagnostics["min_ess_bulk"], math.nan)
        if not diagnostics.empty and "min_ess_bulk" in diagnostics
        else math.nan
    )

    min_replications = _safe_min(recovery["replications"], 0.0) if not recovery.empty else 0.0
    max_failure = _safe_max(failure["failure_rate"], 1.0) if not failure.empty else 1.0
    max_absolute_bias = _safe_max(recovery["absolute_bias"], math.inf) if not recovery.empty else math.inf
    max_rmse = _safe_max(recovery["rmse"], math.inf) if not recovery.empty else math.inf

    finite_coverage = (
        _finite_numeric(recovery["coverage"])
        if not recovery.empty and "coverage" in recovery
        else pd.Series(dtype=float)
    )
    coverage_observed = (
        f"{round(float(finite_coverage.min()), 3)}-{round(float(finite_coverage.max()), 3)}"
        if len(finite_coverage)
        else None
    )

    finite_sbc = _finite_numeric(sbc["p_value"]) if not sbc.empty and "p_value" in sbc else pd.Series(dtype=float)
    sbc_observed = float(finite_sbc.min()) if len(finite_sbc) else math.nan
    empirical_pass = _evidence_pass(
        empirical_reproduction,
        "empirical_reproduction",
    )

    gates = pd.DataFrame(
        {
            "gate": [
                "all_jobs_present",
                "replications",
                "failure_rate",
                "absolute_bias",
                "rmse",
                "coverage",
                "rhat",
                "ess_bulk",
                "divergences",
                "sbc",
                "empirical_reproduction",
            ],
            "required": [
                True,
                thresholds["required_replications"],
                thresholds["max_failure_rate"],
                thresholds["max_absolute_bias"],
                thresholds["max_rmse"],
                (f"{thresholds['min_coverage']}-{thresholds['max_coverage']}"),
                thresholds["max_rhat"],
                thresholds["min_ess_bulk"],
                thresholds["max_divergence_rate"],
                thresholds["require_sbc"],
                thresholds["require_empirical_reproduction"],
            ],
            "observed": [
                missing_jobs.empty,
                min_replications,
                max_failure,
                max_absolute_bias,
                max_rmse,
                coverage_observed,
                max_rhat,
                min_ess,
                divergence_rate,
                sbc_observed,
                empirical_pass,
            ],
            "pass": [
                missing_jobs.empty,
                (
                    not recovery.empty
                    and bool(
                        (
                            pd.to_numeric(recovery["replications"], errors="coerce")
                            >= thresholds["required_replications"]
                        ).all()
                    )
                ),
                (
                    not failure.empty
                    and bool(
                        (
                            pd.to_numeric(failure["failure_rate"], errors="coerce") <= thresholds["max_failure_rate"]
                        ).all()
                    )
                ),
                (
                    not recovery.empty
                    and np.isfinite(pd.to_numeric(recovery["absolute_bias"], errors="coerce")).all()
                    and bool(
                        (
                            pd.to_numeric(recovery["absolute_bias"], errors="coerce") <= thresholds["max_absolute_bias"]
                        ).all()
                    )
                ),
                (
                    not recovery.empty
                    and np.isfinite(pd.to_numeric(recovery["rmse"], errors="coerce")).all()
                    and bool((pd.to_numeric(recovery["rmse"], errors="coerce") <= thresholds["max_rmse"]).all())
                ),
                (
                    not recovery.empty
                    and np.isfinite(pd.to_numeric(recovery["coverage"], errors="coerce")).all()
                    and bool(
                        (
                            (pd.to_numeric(recovery["coverage"], errors="coerce") >= thresholds["min_coverage"])
                            & (pd.to_numeric(recovery["coverage"], errors="coerce") <= thresholds["max_coverage"])
                        ).all()
                    )
                ),
                (not math.isfinite(max_rhat) or max_rhat <= thresholds["max_rhat"]),
                (not math.isfinite(min_ess) or min_ess >= thresholds["min_ess_bulk"]),
                (not math.isfinite(divergence_rate) or divergence_rate <= thresholds["max_divergence_rate"]),
                (
                    not thresholds["require_sbc"]
                    or (not sbc.empty and len(finite_sbc) == len(sbc) and bool((finite_sbc >= 0.01).all()))
                ),
                (not thresholds["require_empirical_reproduction"] or empirical_pass),
            ],
        }
    )

    return EyeValidationCompletionAudit(
        status="complete" if bool(gates["pass"].all()) else "incomplete",
        gates=gates,
        missing_jobs=missing_jobs,
        failure=failure,
        recovery=recovery,
        sbc=sbc,
        thresholds=thresholds,
        audited_utc=_now_utc(),
    )


def _plot_engine(engine: Any) -> str:
    if isinstance(engine, (list, tuple)):
        engine = engine[0]
    engine = str(engine)
    if engine == "ggplot2":
        raise EyeProcessBackendError(
            "The frozen R `ggplot2` rendering backend is R-specific. "
            "Use engine='auto' or engine='base' for the matplotlib "
            "reference rendering in eyeprocesspy."
        )
    if engine not in {"auto", "base", "matplotlib"}:
        _stop("`engine` must be auto, base, matplotlib, or ggplot2.")
    return "matplotlib"


def _axis(ax=None):
    import matplotlib.pyplot as plt

    return plt.subplots()[1] if ax is None else ax


def plot_parameter_recovery(
    x,
    parameter=None,
    engine=("auto", "ggplot2", "base"),
    ax=None,
    **kwargs,
):
    """Plot estimated versus true parameter values."""
    data = x["estimates"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or not {
        "truth",
        "estimate",
        "parameter",
    }.issubset(data.columns):
        _stop("Recovery estimates are unavailable.")
    data = data.copy()
    data["truth"] = pd.to_numeric(data["truth"], errors="coerce")
    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data = data.loc[np.isfinite(data["truth"]) & np.isfinite(data["estimate"])]
    if parameter is not None:
        requested = {str(value) for value in ([parameter] if isinstance(parameter, str) else parameter)}
        data = data.loc[data["parameter"].astype(str).isin(requested)]
    if data.empty:
        _stop("No recovery rows match the request.")
    _plot_engine(engine)
    ax = _axis(ax)
    ax.scatter(data["truth"], data["estimate"], **kwargs)
    lower = float(min(data["truth"].min(), data["estimate"].min()))
    upper = float(max(data["truth"].max(), data["estimate"].max()))
    ax.plot([lower, upper], [lower, upper])
    ax.set_xlabel("True value")
    ax.set_ylabel("Estimated value")
    ax.set_title("Parameter recovery")
    return ax


def plot_interval_coverage(
    x,
    target=0.95,
    engine=("auto", "ggplot2", "base"),
    ax=None,
    **kwargs,
):
    """Plot observed interval coverage against a nominal target."""
    data = validation_recovery_summary(x) if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or not {
        "parameter",
        "coverage",
    }.issubset(data.columns):
        _stop("Coverage summary is unavailable.")
    data = data.copy()
    data["coverage"] = pd.to_numeric(data["coverage"], errors="coerce")
    data = data.loc[np.isfinite(data["coverage"])]
    if data.empty:
        _stop("No finite coverage values are available.")
    _plot_engine(engine)
    ax = _axis(ax)
    labels = data["parameter"].astype(str).tolist()
    positions = np.arange(len(data))
    ax.bar(positions, data["coverage"].to_numpy(dtype=float), **kwargs)
    ax.axhline(float(target))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Interval coverage")
    return ax


def _plot_sbc_ranks(data: pd.DataFrame, parameter=None) -> pd.DataFrame:
    if not {"job_id", "parameter", "draw", "truth"}.issubset(data.columns):
        _stop("SBC draws are unavailable.")
    if parameter is not None:
        requested = {str(value) for value in ([parameter] if isinstance(parameter, str) else parameter)}
        data = data.loc[data["parameter"].astype(str).isin(requested)]
    ranks = _sbc_ranks(data, [])
    if ranks.empty:
        _stop("No SBC ranks could be calculated.")
    ranks["scaled_rank"] = (pd.to_numeric(ranks["rank"], errors="coerce") + 0.5) / (
        pd.to_numeric(ranks["draws"], errors="coerce") + 1
    )
    return ranks


def plot_sbc_rank(
    x,
    parameter=None,
    bins=10,
    engine=("auto", "ggplot2", "base"),
    ax=None,
    **kwargs,
):
    """Plot simulation-based-calibration scaled-rank histograms."""
    data = x["draws"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame):
        _stop("SBC draws are unavailable.")
    bins = _scalar_int(bins, "bins", 2)
    ranks = _plot_sbc_ranks(data.copy(), parameter=parameter)
    _plot_engine(engine)
    ax = _axis(ax)
    ax.hist(
        ranks["scaled_rank"].to_numpy(dtype=float),
        bins=bins,
        range=(0, 1),
        **kwargs,
    )
    ax.set_xlabel("Scaled rank")
    ax.set_ylabel("Replications")
    ax.set_title("Simulation-based calibration")
    return ax


def plot_validation_failures(
    x,
    engine=("auto", "ggplot2", "base"),
    ax=None,
    **kwargs,
):
    """Plot validation failure rates."""
    data = validation_failure_summary(x) if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or "failure_rate" not in data:
        _stop("Failure summary is unavailable.")
    data = data.copy()
    labels = (
        data["scenario_id"].astype(str).tolist()
        if "scenario_id" in data
        else [str(index + 1) for index in range(len(data))]
    )
    _plot_engine(engine)
    ax = _axis(ax)
    positions = np.arange(len(data))
    ax.bar(
        positions,
        pd.to_numeric(data["failure_rate"], errors="coerce"),
        **kwargs,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Failure rate")
    ax.set_title("Validation failures")
    return ax


def plot_validation_runtime(
    x,
    engine=("auto", "ggplot2", "base"),
    ax=None,
    **kwargs,
):
    """Plot per-job validation runtime."""
    data = x["jobs"] if _is_collection(x) else x
    if not isinstance(data, pd.DataFrame) or "elapsed_seconds" not in data:
        _stop("Runtime data are unavailable.")
    data = data.copy()
    data["elapsed_seconds"] = pd.to_numeric(data["elapsed_seconds"], errors="coerce")
    data = data.loc[np.isfinite(data["elapsed_seconds"])]
    if data.empty:
        _stop("No finite runtime values are available.")
    _plot_engine(engine)
    ax = _axis(ax)
    if "scenario_id" in data:
        groups = [
            group["elapsed_seconds"].to_numpy(dtype=float) for _, group in data.groupby("scenario_id", sort=False)
        ]
        labels = [str(name) for name, _ in data.groupby("scenario_id", sort=False)]
        ax.boxplot(groups, tick_labels=labels, **kwargs)
    else:
        ax.plot(
            np.arange(1, len(data) + 1),
            data["elapsed_seconds"].to_numpy(dtype=float),
            **kwargs,
        )
    ax.set_ylabel("Seconds")
    ax.set_title("Validation runtime")
    return ax


def model_promotion_spec(
    model_families=(
        "dynamic_irtree",
        "functional_pupil_irt",
        "theory_strategy_irt",
        "gaze_diffusion_irt",
    ),
    require_completion=True,
    require_sbc=True,
    require_misspecification=True,
    require_grouped_validation=True,
    require_engine_equivalence=True,
    require_empirical_reproduction=True,
    require_preprocessing_sensitivity=True,
    require_multi_vendor=False,
):
    """Declare required scientific evidence before model promotion."""
    if isinstance(model_families, str):
        model_families = [model_families]
    unique = []
    for family in model_families:
        if family is None or not str(family):
            _stop("`model_families` must contain non-empty names.")
        family = str(family)
        if family not in unique:
            unique.append(family)
    if not unique:
        _stop("`model_families` must contain non-empty names.")
    return EyeModelPromotionSpec(
        model_families=unique,
        require_completion=bool(require_completion is True),
        require_sbc=bool(require_sbc is True),
        require_misspecification=bool(require_misspecification is True),
        require_grouped_validation=bool(require_grouped_validation is True),
        require_engine_equivalence=bool(require_engine_equivalence is True),
        require_empirical_reproduction=bool(require_empirical_reproduction is True),
        require_preprocessing_sensitivity=bool(require_preprocessing_sensitivity is True),
        require_multi_vendor=bool(require_multi_vendor is True),
    )


def audit_model_promotion(evidence, spec=None):
    """Audit promotion readiness for each declared advanced-model family."""
    if not isinstance(evidence, Mapping):
        _stop("`evidence` must be a named list.")
    if spec is None:
        spec = model_promotion_spec()
    if not isinstance(spec, EyeModelPromotionSpec):
        _stop("`spec` must be created by `model_promotion_spec()`.")

    gate_requirements = {
        "completion": spec["require_completion"],
        "sbc": spec["require_sbc"],
        "misspecification": spec["require_misspecification"],
        "grouped_validation": spec["require_grouped_validation"],
        "engine_equivalence": spec["require_engine_equivalence"],
        "empirical_reproduction": spec["require_empirical_reproduction"],
        "preprocessing_sensitivity": spec["require_preprocessing_sensitivity"],
        "multi_vendor": spec["require_multi_vendor"],
    }

    rows = []
    for family in spec["model_families"]:
        family_evidence = evidence.get(family)
        if not isinstance(family_evidence, Mapping):
            family_evidence = {}
        for gate, required in gate_requirements.items():
            passed = _evidence_pass(family_evidence.get(gate), gate) if required else True
            rows.append(
                {
                    "model_family": family,
                    "gate": gate,
                    "required": bool(required),
                    "pass": bool(passed),
                }
            )
    gates = pd.DataFrame(rows)

    models = _group_summary(
        gates,
        ["model_family"],
        lambda frame: {
            "required_gates": int(frame["required"].sum()),
            "passed_required_gates": int((frame["required"] & frame["pass"]).sum()),
            "status": ("promotable" if bool(frame.loc[frame["required"], "pass"].all()) else "experimental"),
        },
    )
    return EyeModelPromotionAudit(
        gates=gates,
        models=models,
        spec=spec,
        audited_utc=_now_utc(),
    )


def _markdown_table(data: pd.DataFrame, digits=4) -> str:
    if not isinstance(data, pd.DataFrame) or data.empty:
        return "_No rows available._"
    frame = data.copy()
    for column in frame:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
            )
        else:
            frame[column] = frame[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_validation_release_report(
    x,
    path,
    completion=None,
    promotion=None,
    title="eyeprocess validation release report",
    include_session=True,
):
    """Write the frozen validation release-report structure as Markdown."""
    if not _is_collection(x):
        _stop("Expected an `eye_validation_collection`.")
    if completion is None:
        completion = audit_validation_completion(x)
    if not isinstance(completion, EyeValidationCompletionAudit):
        _stop("`completion` must be a validation completion audit.")

    estimates = x["estimates"] if isinstance(x.get("estimates"), pd.DataFrame) else pd.DataFrame()
    jobs = x["jobs"] if isinstance(x.get("jobs"), pd.DataFrame) else pd.DataFrame()
    predictions = x["predictions"] if isinstance(x.get("predictions"), pd.DataFrame) else pd.DataFrame()
    draws = x["draws"] if isinstance(x.get("draws"), pd.DataFrame) else pd.DataFrame()

    recovery = validation_recovery_summary(x) if not estimates.empty else pd.DataFrame()
    failure = validation_failure_summary(x) if not jobs.empty else pd.DataFrame()
    runtime = validation_runtime_summary(x) if not jobs.empty else pd.DataFrame()
    calibration = (
        validation_calibration_summary(x)
        if not predictions.empty and {"observed", "predicted"}.issubset(predictions.columns)
        else pd.DataFrame()
    )
    try:
        sbc = validation_sbc_summary(x) if not draws.empty else pd.DataFrame()
    except EyeProcessValidationError:
        sbc = pd.DataFrame()

    lines = [
        f"# {title}",
        "",
        f"Generated: {_now_utc()}",
        "",
        f"Completion status: **{completion['status'].upper()}**",
        "",
        "## Completion gates",
        "",
        _markdown_table(completion["gates"]),
        "",
        "## Job failures and convergence",
        "",
        _markdown_table(failure),
        "",
        "## Parameter recovery",
        "",
        _markdown_table(recovery),
        "",
        "## Runtime",
        "",
        _markdown_table(runtime),
        "",
    ]
    if not calibration.empty:
        lines.extend(
            [
                "## Prediction calibration",
                "",
                _markdown_table(calibration),
                "",
            ]
        )
    if not sbc.empty:
        lines.extend(
            [
                "## Simulation-based calibration",
                "",
                _markdown_table(sbc),
                "",
            ]
        )
    if isinstance(promotion, EyeModelPromotionAudit):
        lines.extend(
            [
                "## Model promotion audit",
                "",
                _markdown_table(promotion["models"]),
                "",
                _markdown_table(promotion["gates"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "Executable software, successful unit tests, and completed "
                "simulations are not by themselves evidence that a model "
                "identifies a named cognitive process. Promotion requires "
                "the prespecified recovery, calibration, misspecification, "
                "grouped-validation, engine-equivalence, sensitivity, and "
                "empirical-reproduction gates."
            ),
            "",
        ]
    )
    if include_session:
        lines.extend(
            [
                "## Session",
                "",
                f"- Python: {platform.python_version()}",
                f"- Platform: {platform.platform()}",
                "",
            ]
        )

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return str(destination)


def write_model_promotion_report(x, path):
    """Write a Markdown report for an advanced-model promotion audit."""
    if not isinstance(x, EyeModelPromotionAudit):
        _stop("Expected an `eye_model_promotion_audit`.")
    lines = [
        "# Advanced-model promotion audit",
        "",
        f"Generated: {_now_utc()}",
        "",
        "## Model status",
        "",
        _markdown_table(x["models"]),
        "",
        "## Evidence gates",
        "",
        _markdown_table(x["gates"]),
        "",
        "## Interpretation boundary",
        "",
        (
            "A promotable status means that the declared evidence gates "
            "passed; it is not a claim that a gaze, pupil, strategy, or "
            "diffusion component has been established as a cognitive state "
            "without substantive external validation."
        ),
        "",
    ]
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return str(destination)
