"""Cross-stage robustness audits for gaze-based HCI claims.

This module is an experimental orchestration layer. It does not select event
detectors, AOIs, quality rules, estimands, or statistical models. Instead, it
records a prespecified decision universe and asks whether one focal scientific
claim is stable across the declared, defensible measurement specifications.

One GazeRobustnessSpec represents one estimand. Alternatives that change the
scientific estimand (for example, event-only TTFF versus censored time-to-event)
must be audited in separate specifications rather than pooled into one
robustness denominator.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError


def _text(value: Any, name: str) -> str:
    out = str(value).strip()
    if not out:
        raise EyeProcessValidationError(f"{name} must be a non-empty string.")
    return out


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
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda z: str(z[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GazeClaimSpec:
    """Prespecified focal HCI claim and its estimand."""

    claim_id: str
    term: str
    estimand_id: str
    substantive_threshold: float | None = None
    threshold_direction: str = "above"

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "claim_id": self.claim_id,
                "term": self.term,
                "estimand_id": self.estimand_id,
                "substantive_threshold": self.substantive_threshold,
                "threshold_direction": self.threshold_direction,
            }
        )


@dataclass(frozen=True)
class GazeRobustnessSpec:
    """Declared measurement-decision universe for one estimand."""

    claim: GazeClaimSpec
    decision_grid: tuple[tuple[str, tuple[Any, ...]], ...]
    primary_decisions: tuple[tuple[str, Any], ...]
    label: str = "gaze_claim_robustness"

    @property
    def decision_dict(self) -> dict[str, tuple[Any, ...]]:
        return dict(self.decision_grid)

    @property
    def primary_dict(self) -> dict[str, Any]:
        return dict(self.primary_decisions)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "claim": self.claim.fingerprint,
                "decision_grid": dict(self.decision_grid),
                "primary_decisions": dict(self.primary_decisions),
                "label": self.label,
            }
        )


@dataclass(frozen=True)
class GazeRobustnessAuditResult:
    """Branch-level outputs for a declared gaze robustness specification."""

    spec: GazeRobustnessSpec
    universes: pd.DataFrame
    results: pd.DataFrame
    failures: pd.DataFrame


def define_gaze_claim_spec(
    claim_id: str,
    *,
    term: str,
    estimand_id: str,
    substantive_threshold: float | None = None,
    threshold_direction: str = "above",
) -> GazeClaimSpec:
    """Define one focal claim without silently changing its estimand."""
    direction = _text(threshold_direction, "threshold_direction").lower()
    if direction not in {"above", "below", "absolute"}:
        raise EyeProcessValidationError(
            "threshold_direction must be 'above', 'below', or 'absolute'."
        )
    threshold = None
    if substantive_threshold is not None:
        threshold = float(substantive_threshold)
        if not np.isfinite(threshold):
            raise EyeProcessValidationError("substantive_threshold must be finite.")
    return GazeClaimSpec(
        claim_id=_text(claim_id, "claim_id"),
        term=_text(term, "term"),
        estimand_id=_text(estimand_id, "estimand_id"),
        substantive_threshold=threshold,
        threshold_direction=direction,
    )


def define_gaze_robustness_spec(
    claim: GazeClaimSpec,
    decision_grid: Mapping[str, Sequence[Any]],
    *,
    primary_decisions: Mapping[str, Any],
    label: str = "gaze_claim_robustness",
) -> GazeRobustnessSpec:
    """Declare a deterministic Cartesian universe of defensible measurement choices."""
    if not isinstance(claim, GazeClaimSpec):
        raise EyeProcessValidationError("claim must be a GazeClaimSpec.")
    if not isinstance(decision_grid, Mapping) or not decision_grid:
        raise EyeProcessValidationError("decision_grid must be a non-empty mapping.")
    if not isinstance(primary_decisions, Mapping):
        raise EyeProcessValidationError("primary_decisions must be a mapping.")

    grid: list[tuple[str, tuple[Any, ...]]] = []
    for raw_name, raw_values in decision_grid.items():
        name = _text(raw_name, "decision name")
        values = tuple(raw_values)
        if not values:
            raise EyeProcessValidationError(f"Decision {name!r} has no declared values.")
        grid.append((name, values))
    grid.sort(key=lambda z: z[0])

    primary = dict(primary_decisions)
    grid_names = {name for name, _ in grid}
    if set(primary) != grid_names:
        raise EyeProcessValidationError(
            "primary_decisions must provide exactly one value for every decision dimension."
        )
    for name, values in grid:
        if not any(primary[name] == value for value in values):
            raise EyeProcessValidationError(
                f"Primary value for {name!r} is not present in its decision grid."
            )

    return GazeRobustnessSpec(
        claim=claim,
        decision_grid=tuple(grid),
        primary_decisions=tuple(sorted(primary.items(), key=lambda z: z[0])),
        label=_text(label, "label"),
    )


def expand_gaze_robustness_spec(spec: GazeRobustnessSpec) -> pd.DataFrame:
    """Expand a declared spec into a stable universe manifest."""
    if not isinstance(spec, GazeRobustnessSpec):
        raise EyeProcessValidationError("spec must be a GazeRobustnessSpec.")
    names = [name for name, _ in spec.decision_grid]
    values = [vals for _, vals in spec.decision_grid]
    primary = spec.primary_dict
    rows: list[dict[str, Any]] = []

    for index, combination in enumerate(itertools.product(*values), start=1):
        choices = dict(zip(names, combination))
        row: dict[str, Any] = {
            "universe_id": f"u{index:04d}",
            "claim_id": spec.claim.claim_id,
            "term": spec.claim.term,
            "estimand_id": spec.claim.estimand_id,
            **choices,
        }
        row["is_primary"] = all(choices[name] == primary[name] for name in names)
        row["specification_hash"] = _fingerprint(
            {
                "claim_spec_hash": spec.claim.fingerprint,
                "choices": choices,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if int(out["is_primary"].sum()) != 1:
        raise EyeProcessValidationError(
            "The declared decision grid must contain exactly one primary universe."
        )
    return out


def _normalise_runner_output(value: Any, claim: GazeClaimSpec) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        if "term" in frame:
            frame = frame.loc[frame["term"].astype(str).eq(claim.term)].copy()
        if len(frame) != 1:
            raise EyeProcessValidationError(
                "Runner DataFrame must yield exactly one row for the focal term."
            )
        raw = frame.iloc[0].to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise EyeProcessValidationError("runner must return a mapping or DataFrame.")

    aliases = {
        "estimate": ("estimate",),
        "SE": ("SE", "se", "standard_error"),
        "CI_lower": ("CI_lower", "CI_low", "ci_lower", "ci_low"),
        "CI_upper": ("CI_upper", "CI_high", "ci_upper", "ci_high"),
        "converged": ("converged", "model_converged"),
        "N": ("N", "n", "model_rows_used"),
    }
    out: dict[str, Any] = {}
    for target, candidates in aliases.items():
        found = next((raw[name] for name in candidates if name in raw), None)
        out[target] = found

    try:
        out["estimate"] = float(out["estimate"])
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("Runner output requires a numeric estimate.") from exc
    if not np.isfinite(out["estimate"]):
        raise EyeProcessValidationError("Runner estimate must be finite.")

    for name in ("SE", "CI_lower", "CI_upper", "N"):
        if out[name] is not None and not pd.isna(out[name]):
            out[name] = float(out[name])
        else:
            out[name] = np.nan

    converged = out["converged"]
    if converged is None or pd.isna(converged):
        converged = True
    out["converged"] = bool(converged)
    return out


def run_gaze_robustness_audit(
    spec: GazeRobustnessSpec,
    runner: Callable[[Mapping[str, Any], GazeClaimSpec], Any],
    *,
    continue_on_error: bool = True,
) -> GazeRobustnessAuditResult:
    """Evaluate every planned universe while retaining failures and non-convergence."""
    if not callable(runner):
        raise EyeProcessValidationError("runner must be callable.")
    universes = expand_gaze_robustness_spec(spec)
    decision_names = [name for name, _ in spec.decision_grid]
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for row in universes.to_dict(orient="records"):
        choices = {name: row[name] for name in decision_names}
        base = {
            "universe_id": row["universe_id"],
            "claim_id": spec.claim.claim_id,
            "term": spec.claim.term,
            "estimand_id": spec.claim.estimand_id,
            "is_primary": bool(row["is_primary"]),
            "specification_hash": row["specification_hash"],
            **choices,
        }
        try:
            tidy = _normalise_runner_output(runner(choices, spec.claim), spec.claim)
            status = "ok" if tidy["converged"] else "non_converged"
            result_rows.append({**base, **tidy, "status": status, "failure_reason": pd.NA})
        except Exception as exc:
            if not continue_on_error:
                raise
            result_rows.append(
                {
                    **base,
                    "estimate": np.nan,
                    "SE": np.nan,
                    "CI_lower": np.nan,
                    "CI_upper": np.nan,
                    "converged": False,
                    "N": np.nan,
                    "status": "failed",
                    "failure_reason": str(exc),
                }
            )
            failure_rows.append(
                {
                    "universe_id": row["universe_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **choices,
                }
            )

    results = pd.DataFrame(result_rows)
    failures = pd.DataFrame(
        failure_rows,
        columns=["universe_id", "error_type", "error", *decision_names],
    )
    return GazeRobustnessAuditResult(
        spec=spec,
        universes=universes,
        results=results,
        failures=failures,
    )


def _threshold_support(estimates: pd.Series, claim: GazeClaimSpec) -> float:
    threshold = claim.substantive_threshold
    if threshold is None or estimates.empty:
        return np.nan
    if claim.threshold_direction == "above":
        decision = estimates >= threshold
    elif claim.threshold_direction == "below":
        decision = estimates <= threshold
    else:
        decision = estimates.abs() >= abs(threshold)
    return float(decision.mean())


def summarise_gaze_claim_robustness(result: GazeRobustnessAuditResult) -> pd.DataFrame:
    """Summarize claim stability using the full planned-universe denominator."""
    if not isinstance(result, GazeRobustnessAuditResult):
        raise EyeProcessValidationError("result must be a GazeRobustnessAuditResult.")
    data = result.results.copy()
    planned = len(data)
    evaluable_mask = data["status"].eq("ok") & pd.to_numeric(
        data["estimate"], errors="coerce"
    ).map(np.isfinite)
    evaluable = data.loc[evaluable_mask].copy()
    estimates = pd.to_numeric(evaluable["estimate"], errors="coerce")

    primary = data.loc[data["is_primary"].astype(bool)].copy()
    primary_estimate = np.nan
    if len(primary) == 1 and primary.iloc[0]["status"] == "ok":
        candidate = pd.to_numeric(pd.Series([primary.iloc[0]["estimate"]]), errors="coerce").iloc[0]
        if np.isfinite(candidate):
            primary_estimate = float(candidate)

    same_direction_evaluable = np.nan
    same_direction_planned = np.nan
    primary_direction = np.nan
    if np.isfinite(primary_estimate):
        primary_direction = float(np.sign(primary_estimate))
        if primary_direction == 0:
            same = estimates.eq(0)
        else:
            same = np.sign(estimates) == primary_direction
        if len(same):
            same_direction_evaluable = float(same.mean())
            same_direction_planned = float(same.sum() / planned) if planned else np.nan

    lowers = pd.to_numeric(evaluable["CI_lower"], errors="coerce")
    uppers = pd.to_numeric(evaluable["CI_upper"], errors="coerce")
    finite_ci = np.isfinite(lowers) & np.isfinite(uppers)
    if finite_ci.any():
        overlap_lower = float(lowers[finite_ci].max())
        overlap_upper = float(uppers[finite_ci].min())
        common_ci_overlap: Any = bool(overlap_lower <= overlap_upper)
    else:
        overlap_lower = overlap_upper = np.nan
        common_ci_overlap = pd.NA

    nvals = pd.to_numeric(evaluable["N"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "claim_id": result.spec.claim.claim_id,
                "term": result.spec.claim.term,
                "estimand_id": result.spec.claim.estimand_id,
                "planned_universes": planned,
                "evaluable_universes": len(evaluable),
                "failed_universes": int(data["status"].eq("failed").sum()),
                "nonconverged_universes": int(data["status"].eq("non_converged").sum()),
                "planned_evaluable_rate": len(evaluable) / planned if planned else np.nan,
                "primary_estimate": primary_estimate,
                "primary_direction": primary_direction,
                "same_direction_proportion_evaluable": same_direction_evaluable,
                "same_direction_proportion_planned": same_direction_planned,
                "median_estimate": float(estimates.median()) if len(estimates) else np.nan,
                "estimate_min": float(estimates.min()) if len(estimates) else np.nan,
                "estimate_max": float(estimates.max()) if len(estimates) else np.nan,
                "estimate_range": (
                    float(estimates.max() - estimates.min()) if len(estimates) else np.nan
                ),
                "common_CI_overlap": common_ci_overlap,
                "common_CI_lower": overlap_lower,
                "common_CI_upper": overlap_upper,
                "median_N": float(nvals.median()) if nvals.notna().any() else np.nan,
                "min_N": float(nvals.min()) if nvals.notna().any() else np.nan,
                "max_N": float(nvals.max()) if nvals.notna().any() else np.nan,
                "substantive_support_proportion": _threshold_support(
                    estimates, result.spec.claim
                ),
            }
        ]
    )


def decompose_gaze_decision_sensitivity(
    result: GazeRobustnessAuditResult,
) -> pd.DataFrame:
    """Describe marginal estimate shifts by declared decision family.

    This is descriptive sensitivity accounting, not causal attribution or a
    variance-decomposition estimand.
    """
    if not isinstance(result, GazeRobustnessAuditResult):
        raise EyeProcessValidationError("result must be a GazeRobustnessAuditResult.")
    data = result.results.loc[result.results["status"].eq("ok")].copy()
    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data = data.loc[np.isfinite(data["estimate"])].copy()
    rows: list[dict[str, Any]] = []
    for name, declared_values in result.spec.decision_grid:
        level_means: list[float] = []
        level_medians: list[float] = []
        evaluable_levels = 0
        for value in declared_values:
            z = data.loc[data[name].map(lambda x: x == value), "estimate"]
            if len(z):
                evaluable_levels += 1
                level_means.append(float(z.mean()))
                level_medians.append(float(z.median()))
        rows.append(
            {
                "decision": name,
                "declared_levels": len(declared_values),
                "evaluable_levels": evaluable_levels,
                "marginal_mean_range": (
                    max(level_means) - min(level_means) if level_means else np.nan
                ),
                "marginal_median_range": (
                    max(level_medians) - min(level_medians) if level_medians else np.nan
                ),
                "evaluable_universes": len(data),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["caveat"] = (
        "Decision-family ranges are descriptive marginal sensitivity summaries; "
        "they are not causal effects or variance-explained fractions."
    )
    return out


def report_gaze_claim_robustness(result: GazeRobustnessAuditResult) -> str:
    """Return a compact Markdown report without assigning an automatic verdict."""
    summary = summarise_gaze_claim_robustness(result).iloc[0]
    lines = [
        "## Gaze claim robustness audit",
        "",
        f"Claim: {summary.claim_id}",
        f"Estimand: {summary.estimand_id}",
        (
            f"Planned universes: {summary.planned_universes}; "
            f"evaluable: {summary.evaluable_universes}; "
            f"failed: {summary.failed_universes}; "
            f"non-converged: {summary.nonconverged_universes}."
        ),
    ]
    if np.isfinite(summary.primary_estimate):
        lines.append(f"Primary estimate: {summary.primary_estimate:.4g}.")
        lines.append(
            "Same-direction proportion: "
            f"{summary.same_direction_proportion_evaluable:.3f} among evaluable universes; "
            f"{summary.same_direction_proportion_planned:.3f} against all planned universes."
        )
    if np.isfinite(summary.median_estimate):
        lines.append(
            f"Estimate median={summary.median_estimate:.4g}; "
            f"range=[{summary.estimate_min:.4g}, {summary.estimate_max:.4g}]."
        )
    if np.isfinite(summary.substantive_support_proportion):
        lines.append(
            "Prespecified substantive-threshold support among evaluable universes: "
            f"{summary.substantive_support_proportion:.3f}."
        )
    lines.extend(
        [
            "",
            (
                "Interpretation: these quantities describe stability only within the "
                "declared, defensible measurement universe. They are not probabilities "
                "that the scientific claim is true."
            ),
            (
                "Estimand-changing alternatives must be reported as separate audits rather "
                "than pooled into this denominator."
            ),
        ]
    )
    return "\n".join(lines)


def plot_gaze_specification_curve(
    result: GazeRobustnessAuditResult,
    *,
    ax: Any = None,
):
    """Plot branch estimates and uncertainty for evaluable specifications."""
    import matplotlib.pyplot as plt

    data = result.results.loc[result.results["status"].eq("ok")].copy()
    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data = data.loc[np.isfinite(data["estimate"])].copy()
    data = data.sort_values(["estimate", "universe_id"]).reset_index(drop=True)

    if ax is None:
        _, ax = plt.subplots()
    if data.empty:
        ax.set_title("Gaze claim specification curve")
        ax.set_xlabel("Specification")
        ax.set_ylabel("Effect estimate")
        return ax

    x = np.arange(len(data))
    estimates = data["estimate"].to_numpy(dtype=float)
    lower = pd.to_numeric(data["CI_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(data["CI_upper"], errors="coerce").to_numpy(dtype=float)
    finite_ci = np.isfinite(lower) & np.isfinite(upper)
    if finite_ci.any():
        yerr = np.vstack(
            [
                estimates[finite_ci] - lower[finite_ci],
                upper[finite_ci] - estimates[finite_ci],
            ]
        )
        ax.errorbar(x[finite_ci], estimates[finite_ci], yerr=yerr, fmt="o", capsize=2)
    if (~finite_ci).any():
        ax.scatter(x[~finite_ci], estimates[~finite_ci])

    primary = data["is_primary"].astype(bool).to_numpy()
    if primary.any():
        ax.scatter(x[primary], estimates[primary], marker="D", s=70)
    ax.axhline(0, linewidth=1)
    ax.set_title("Gaze claim specification curve")
    ax.set_xlabel("Defensible specification (ordered by estimate)")
    ax.set_ylabel("Effect estimate")
    return ax


__all__ = [
    "GazeClaimSpec",
    "GazeRobustnessSpec",
    "GazeRobustnessAuditResult",
    "define_gaze_claim_spec",
    "define_gaze_robustness_spec",
    "expand_gaze_robustness_spec",
    "run_gaze_robustness_audit",
    "summarise_gaze_claim_robustness",
    "decompose_gaze_decision_sensitivity",
    "report_gaze_claim_robustness",
    "plot_gaze_specification_curve",
]
