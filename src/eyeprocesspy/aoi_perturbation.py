"""AOI perturbation and uncertainty analysis public API."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._aoi_assignment_core import (
    _assign_points,
    _validate_model_table,
    compare_aoi_assignments,
    estimate_aoi_assignment_stability,
    estimate_fixation_assignment_probability,
    recompute_aoi_features,
)
from ._aoi_geometry_primitives import (
    AMBIGUOUS,
    OUTSIDE,
    _frame,
    _software_provenance,
    _stable_frame_hash,
)
from ._aoi_geometry_units import (
    _spec_value,
    aoi_perturbation_spec,
    convert_aoi_margin_to_degrees,
    convert_aoi_margin_to_pixels,
)
from ._aoi_geometry_validation import validate_aoi_geometry
from ._aoi_perturb_ops import (
    apply_aoi_perturbation_grid,
    create_aoi_perturbation_grid,
    dilate_aoi,
    erode_aoi,
    jitter_aoi,
    perturb_aoi_geometry,
    translate_aoi,
)
from .exceptions import EyeProcessValidationError
from .irt import EyeResult
from .plots_aoi_perturbation import (
    plot_aoi_assignment_stability,
    plot_aoi_coefficient_stability,
    plot_aoi_perturbations,
    plot_aoi_robustness_surface,
)


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def run_aoi_sensitivity_analysis(
    data: pd.DataFrame,
    aois: pd.DataFrame,
    grid: Any,
    *,
    x_col: str,
    y_col: str,
    observation_id_col: str | None = None,
    participant_col: str | None = None,
    trial_col: str | None = None,
    duration_col: str | None = None,
    time_col: str | None = None,
    observation_level: str = "fixation",
    overlap_policy: str = "ambiguous",
    model_callback: Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame] | None = None,
    preprocessing_specification: Any = None,
    event_detector: Any = None,
    quality_rules: Any = None,
    model_specification: Any = None,
) -> EyeResult:
    """Run geometry perturbation, remapping, feature recomputation, and optional models."""
    frame = _frame(data, "data")
    observation_level = str(observation_level).lower()
    if observation_level not in {"fixation", "sample"}:
        raise EyeProcessValidationError("`observation_level` must be 'fixation' or 'sample'.")
    geometry = validate_aoi_geometry(aois)["geometry"]
    if observation_id_col is not None:
        if observation_id_col not in frame.columns:
            raise EyeProcessValidationError(f"Observation id column `{observation_id_col}` is absent.")
        ids = frame[observation_id_col].tolist()
        id_series = pd.Series(ids)
        if id_series.isna().any() or id_series.duplicated().any():
            raise EyeProcessValidationError(
                "Observation IDs must be unique and non-missing for perturbation tracking."
            )
    else:
        ids = list(range(1, len(frame) + 1))
    grid_result = apply_aoi_perturbation_grid(geometry, grid)
    completed = grid_result["audit"].loc[grid_result["audit"]["status"].eq("completed"), "perturbation_id"].astype(str).tolist()
    if "baseline" not in completed:
        raise EyeProcessValidationError("Sensitivity analysis requires a successful `baseline` perturbation in the grid.")

    assignments: dict[str, pd.Series] = {}
    features: dict[str, pd.DataFrame] = {}
    model_rows: list[pd.DataFrame] = []
    failure_rows = grid_result["audit"].loc[grid_result["audit"]["status"].eq("failed"), ["perturbation_id", "message"]].copy()
    if len(failure_rows):
        failure_rows["stage"] = "geometry"
    else:
        failure_rows = pd.DataFrame(columns=["perturbation_id", "message", "stage"])
    specs = {str(_spec_value(s, "perturbation_id")): s for s in grid["specifications"]}

    for pid in completed:
        geom = grid_result["geometries"][pid]
        assigned = _assign_points(frame, geom, x_col=x_col, y_col=y_col, overlap_policy=overlap_policy)
        assignments[pid] = assigned
        feat = recompute_aoi_features(
            frame,
            assigned,
            participant_col=participant_col,
            trial_col=trial_col,
            duration_col=duration_col,
            time_col=time_col,
            perturbation_id=pid,
            aoi_levels=geometry["aoi_id"].astype(str).tolist(),
            observation_level=observation_level,
        )
        features[pid] = feat
        if model_callback is not None:
            assigned_data = frame.copy()
            assigned_data["aoi_assignment"] = assigned
            assigned_data["perturbation_id"] = pid
            try:
                model_table = _validate_model_table(model_callback(feat.copy(), assigned_data, specs[pid]), pid)
                model_rows.append(model_table)
            except Exception as exc:
                failure_rows = pd.concat([
                    failure_rows,
                    pd.DataFrame([{"perturbation_id": pid, "message": str(exc), "stage": "model"}]),
                ], ignore_index=True)

    baseline = assignments["baseline"]
    comparisons: dict[str, Any] = {}
    for pid, assigned in assignments.items():
        comparisons[pid] = compare_aoi_assignments(baseline, assigned, ids=ids)
    metadata_cols = [c for c in (participant_col, trial_col) if c is not None]
    metadata = frame.loc[:, metadata_cols].copy() if metadata_cols else pd.DataFrame(index=frame.index)
    metadata.insert(0, "observation_id", ids)
    stability = estimate_aoi_assignment_stability(comparisons, metadata=metadata, group_cols=metadata_cols)
    assignment_long = pd.concat([
        pd.DataFrame({"perturbation_id": pid, "observation_id": ids, "aoi_assignment": labels.tolist()})
        for pid, labels in assignments.items()
    ], ignore_index=True)
    assignment_probability = estimate_fixation_assignment_probability(assignment_long)
    models = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame(columns=["perturbation_id", "term", "estimate", "SE", "CI_low", "CI_high", "p_value", "model_converged", "N", "direction"])
    provenance = {
        "source_data_hash": _stable_frame_hash(frame),
        "aoi_specification_hash": _stable_frame_hash(geometry),
        "preprocessing_specification": preprocessing_specification,
        "event_detector": event_detector,
        "quality_rules": quality_rules,
        "model_specification": model_specification,
        "overlap_policy": overlap_policy,
        "observation_level": observation_level,
        "software": _software_provenance(),
    }
    return _result(
        "eye_aoi_sensitivity",
        nominal_aois=geometry,
        grid=grid,
        grid_result=grid_result,
        assignments=assignments,
        assignment_table=assignment_long,
        comparisons=comparisons,
        stability=stability,
        assignment_probability=assignment_probability,
        features=features,
        models=models,
        failures=failure_rows.reset_index(drop=True),
        provenance=provenance,
        caveat="Robustness proportions summarize the declared perturbation set; they are not probabilities that the substantive conclusion is true.",
    )


def assess_aoi_inference_stability(x: Any, *, term: str | None = None) -> pd.DataFrame:
    models = x["models"].copy()
    if models.empty:
        return pd.DataFrame(columns=["term", "n_models", "n_converged", "convergence_proportion", "same_sign_proportion", "median_estimate", "min_estimate", "max_estimate", "median_CI_width", "median_N", "min_N", "max_N"])
    if term is not None:
        models = models.loc[models["term"].astype(str).eq(str(term))].copy()
        if models.empty:
            raise EyeProcessValidationError(f"No model rows found for term `{term}`.")
    rows = []
    for current_term, z in models.groupby("term", sort=True):
        converged = z["model_converged"].fillna(False).astype(bool)
        usable = z.loc[converged].copy()
        baseline = usable.loc[usable["perturbation_id"].eq("baseline")]
        if baseline.empty:
            baseline_sign = np.nan
        else:
            baseline_est = float(pd.to_numeric(baseline.iloc[0]["estimate"], errors="coerce"))
            baseline_sign = np.sign(baseline_est)
        estimates = pd.to_numeric(usable["estimate"], errors="coerce")
        ci_width = pd.to_numeric(usable["CI_high"], errors="coerce") - pd.to_numeric(usable["CI_low"], errors="coerce")
        sample_sizes = pd.to_numeric(usable["N"], errors="coerce")
        signs = np.sign(estimates)
        same = np.nan if not np.isfinite(baseline_sign) or not len(signs) else float(np.mean(signs == baseline_sign))
        rows.append({
            "term": current_term,
            "n_models": len(z),
            "n_converged": int(converged.sum()),
            "convergence_proportion": float(converged.mean()) if len(z) else np.nan,
            "same_sign_proportion": same,
            "median_estimate": float(estimates.median()) if estimates.notna().any() else np.nan,
            "min_estimate": float(estimates.min()) if estimates.notna().any() else np.nan,
            "max_estimate": float(estimates.max()) if estimates.notna().any() else np.nan,
            "median_CI_width": float(ci_width.median()) if ci_width.notna().any() else np.nan,
            "median_N": float(sample_sizes.median()) if sample_sizes.notna().any() else np.nan,
            "min_N": float(sample_sizes.min()) if sample_sizes.notna().any() else np.nan,
            "max_N": float(sample_sizes.max()) if sample_sizes.notna().any() else np.nan,
        })
    out = pd.DataFrame(rows)
    out.attrs["caveat"] = "Same-sign and convergence proportions are descriptive sensitivity summaries, not probabilities that an effect is true."
    return out


def summarise_aoi_sensitivity(x: Any) -> EyeResult:
    stability = x["stability"]["overall"].copy()
    inference = assess_aoi_inference_stability(x)
    failures = x["failures"].copy()
    grid_audit = x["grid_result"]["audit"].copy()
    return _result(
        "eye_aoi_sensitivity_summary",
        assignment_stability=stability,
        inference_stability=inference,
        perturbation_audit=grid_audit,
        failures=failures,
        n_planned=len(grid_audit),
        n_completed=int(grid_audit["status"].eq("completed").sum()),
        n_geometry_failed=int(grid_audit["status"].eq("failed").sum()),
        n_model_failures=int((failures["stage"] == "model").sum()) if len(failures) else 0,
        caveat=x["caveat"],
    )


def report_aoi_sensitivity(x: Any) -> str:
    """Return a compact manuscript-oriented Markdown report."""
    summary = summarise_aoi_sensitivity(x)
    stability = summary["assignment_stability"]
    median_unchanged = float(stability["proportion_unchanged"].median()) if len(stability) else np.nan
    lines = [
        "## AOI perturbation sensitivity analysis",
        "",
        f"Planned perturbations: {summary['n_planned']}; completed geometry branches: {summary['n_completed']}; geometry failures: {summary['n_geometry_failed']}; model callback failures: {summary['n_model_failures']}.",
        f"Median unchanged AOI assignment across completed perturbations: {median_unchanged:.3f}." if np.isfinite(median_unchanged) else "Assignment stability could not be summarized.",
        "",
    ]
    inference = summary["inference_stability"]
    if len(inference):
        lines.append("Model-level sensitivity was summarized using coefficient direction, magnitude, interval width, and convergence rather than significance alone.")
        for row in inference.itertuples(index=False):
            lines.append(
                f"- `{row.term}`: {row.n_converged}/{row.n_models} converged; same-sign frequency={row.same_sign_proportion:.3f}; median estimate={row.median_estimate:.4g}; range=[{row.min_estimate:.4g}, {row.max_estimate:.4g}]; N range=[{row.min_N:.0f}, {row.max_N:.0f}]."
            )
        lines.append("")
    lines.append("Interpretation: these quantities describe robustness to the declared AOI perturbations. They are not probabilities that the scientific conclusion is true.")
    if len(summary["failures"]):
        lines.append("Failed or non-evaluable branches remain in the audit trail and should be reported rather than silently excluded.")
    return "\n".join(lines)


__all__ = [
    "OUTSIDE", "AMBIGUOUS", "aoi_perturbation_spec", "validate_aoi_geometry",
    "convert_aoi_margin_to_degrees", "convert_aoi_margin_to_pixels",
    "dilate_aoi", "erode_aoi", "translate_aoi", "jitter_aoi",
    "perturb_aoi_geometry", "create_aoi_perturbation_grid",
    "apply_aoi_perturbation_grid", "compare_aoi_assignments",
    "estimate_aoi_assignment_stability", "estimate_fixation_assignment_probability",
    "recompute_aoi_features", "run_aoi_sensitivity_analysis",
    "summarise_aoi_sensitivity", "assess_aoi_inference_stability",
    "report_aoi_sensitivity", "plot_aoi_perturbations",
    "plot_aoi_assignment_stability", "plot_aoi_coefficient_stability",
    "plot_aoi_robustness_surface",
]
