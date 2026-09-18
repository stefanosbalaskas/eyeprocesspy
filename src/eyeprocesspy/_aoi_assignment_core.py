"""Internal AOI assignment and feature-recomputation contracts."""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._aoi_geometry_primitives import AMBIGUOUS, OUTSIDE, _frame
from ._aoi_geometry_validation import _aoi_contains, validate_aoi_geometry
from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _assign_points(
    data: pd.DataFrame,
    geometry: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    overlap_policy: str = "ambiguous",
) -> pd.Series:
    frame = _frame(data, "data")
    if x_col not in frame.columns or y_col not in frame.columns:
        raise EyeProcessValidationError(f"`data` must contain `{x_col}` and `{y_col}`.")
    overlap_policy = str(overlap_policy).lower()
    if overlap_policy not in {"ambiguous", "all", "error"}:
        raise EyeProcessValidationError("`overlap_policy` must be ambiguous, all, or error.")
    geom = validate_aoi_geometry(geometry)["geometry"]
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(x) | ~np.isfinite(y)
    memberships = np.zeros((len(frame), len(geom)), dtype=bool)
    for j, (_, row) in enumerate(geom.iterrows()):
        memberships[:, j] = _aoi_contains(row, x, y) & ~missing
    counts = memberships.sum(axis=1)
    labels = np.full(len(frame), OUTSIDE, dtype=object)
    labels[missing] = None
    for i in np.where(counts == 1)[0]:
        labels[i] = str(geom.iloc[np.flatnonzero(memberships[i])[0]]["aoi_id"])
    ambiguous = np.where(counts > 1)[0]
    if len(ambiguous):
        if overlap_policy == "error":
            raise EyeProcessValidationError(f"{len(ambiguous)} observations have ambiguous overlapping AOI membership.")
        if overlap_policy == "ambiguous":
            labels[ambiguous] = AMBIGUOUS
        else:
            for i in ambiguous:
                labels[i] = "|".join(geom.iloc[np.flatnonzero(memberships[i])]["aoi_id"].astype(str).tolist())
    return pd.Series(labels, index=frame.index, dtype="object", name="aoi_assignment")


def compare_aoi_assignments(
    baseline: Sequence[Any],
    perturbed: Sequence[Any],
    *,
    ids: Sequence[Any] | None = None,
) -> EyeResult:
    base = pd.Series(list(baseline), dtype="object")
    alt = pd.Series(list(perturbed), dtype="object")
    if len(base) != len(alt):
        raise EyeProcessValidationError("Baseline and perturbed assignments must have equal length.")
    if ids is None:
        ids = np.arange(1, len(base) + 1)
    if len(ids) != len(base):
        raise EyeProcessValidationError("`ids` must have the same length as assignments.")
    missing = base.isna() | alt.isna()
    unchanged = (~missing) & base.eq(alt)
    newly = (~missing) & base.eq(OUTSIDE) & ~alt.eq(OUTSIDE)
    lost = (~missing) & ~base.eq(OUTSIDE) & alt.eq(OUTSIDE)
    reassigned = (~missing) & ~unchanged & ~newly & ~lost
    detail = pd.DataFrame({
        "observation_id": list(ids),
        "baseline_aoi": base,
        "perturbed_aoi": alt,
        "unchanged": unchanged,
        "newly_assigned": newly,
        "lost_assignment": lost,
        "reassigned": reassigned,
        "missing_comparison": missing,
    })
    comparable = ~missing
    n = int(comparable.sum())
    summary = pd.DataFrame([{
        "n_total": len(base),
        "n_comparable": n,
        "proportion_unchanged": float(unchanged[comparable].mean()) if n else np.nan,
        "proportion_newly_assigned": float(newly[comparable].mean()) if n else np.nan,
        "proportion_lost": float(lost[comparable].mean()) if n else np.nan,
        "proportion_reassigned": float(reassigned[comparable].mean()) if n else np.nan,
        "proportion_missing_comparison": float(missing.mean()) if len(base) else np.nan,
    }])
    valid_pairs = detail.loc[comparable, ["baseline_aoi", "perturbed_aoi"]]
    matrix = pd.crosstab(valid_pairs["baseline_aoi"], valid_pairs["perturbed_aoi"], dropna=False)
    return _result("eye_aoi_assignment_comparison", detail=detail, summary=summary, reassignment_matrix=matrix)


def estimate_aoi_assignment_stability(
    comparisons: Mapping[str, Any] | pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
) -> EyeResult:
    rows: list[pd.DataFrame] = []
    if isinstance(comparisons, pd.DataFrame):
        required = {"perturbation_id", "baseline_aoi", "perturbed_aoi"}
        if not required.issubset(comparisons.columns):
            raise EyeProcessValidationError("Comparison data must contain perturbation_id, baseline_aoi, and perturbed_aoi.")
        source = comparisons.copy()
    else:
        for pid, comparison in comparisons.items():
            detail = comparison["detail"].copy()
            detail["perturbation_id"] = str(pid)
            rows.append(detail)
        source = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if source.empty:
        raise EyeProcessValidationError("No assignment comparisons are available.")
    if "observation_id" not in source.columns:
        source["observation_id"] = source.groupby("perturbation_id").cumcount() + 1
    source_ids = source["observation_id"].drop_duplicates().tolist()
    if pd.Series(source_ids).isna().any():
        raise EyeProcessValidationError("Comparison observation IDs must be non-missing.")
    if metadata is not None:
        meta = _frame(metadata, "metadata")
        if len(meta) != len(source_ids):
            raise EyeProcessValidationError("`metadata` must contain one row per observation.")
        if "observation_id" not in meta.columns:
            meta["observation_id"] = source_ids
        else:
            if meta["observation_id"].isna().any() or meta["observation_id"].duplicated().any():
                raise EyeProcessValidationError("Metadata observation IDs must be unique and non-missing.")
            if set(meta["observation_id"].tolist()) != set(source_ids):
                raise EyeProcessValidationError("Metadata observation IDs must match the comparison observation IDs exactly.")
        source = source.merge(meta, on="observation_id", how="left", validate="many_to_one")
    source["comparable"] = ~(source["baseline_aoi"].isna() | source["perturbed_aoi"].isna())
    source["unchanged"] = source["comparable"] & source["baseline_aoi"].eq(source["perturbed_aoi"])
    source["newly_assigned"] = source["comparable"] & source["baseline_aoi"].eq(OUTSIDE) & ~source["perturbed_aoi"].eq(OUTSIDE)
    source["lost_assignment"] = source["comparable"] & ~source["baseline_aoi"].eq(OUTSIDE) & source["perturbed_aoi"].eq(OUTSIDE)
    source["reassigned"] = source["comparable"] & ~(source["unchanged"] | source["newly_assigned"] | source["lost_assignment"])

    def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        out_rows = []
        grouped = [((), df)] if not keys else df.groupby(keys, dropna=False, sort=True)
        for key, z in grouped:
            key = (key,) if keys and not isinstance(key, tuple) else key
            comparable = z["comparable"]
            n = int(comparable.sum())
            row = {k: v for k, v in zip(keys, key)}
            row.update({
                "n_total": len(z),
                "n_comparable": n,
                "proportion_unchanged": float(z.loc[comparable, "unchanged"].mean()) if n else np.nan,
                "proportion_newly_assigned": float(z.loc[comparable, "newly_assigned"].mean()) if n else np.nan,
                "proportion_lost": float(z.loc[comparable, "lost_assignment"].mean()) if n else np.nan,
                "proportion_reassigned": float(z.loc[comparable, "reassigned"].mean()) if n else np.nan,
            })
            out_rows.append(row)
        return pd.DataFrame(out_rows)

    overall = summarise(source, ["perturbation_id"])
    by_group: dict[str, pd.DataFrame] = {}
    for col in group_cols or []:
        if col not in source.columns:
            raise EyeProcessValidationError(f"Grouping column `{col}` is not available in comparison metadata.")
        by_group[col] = summarise(source, ["perturbation_id", col])
    aoi_level = summarise(source.assign(aoi=source["baseline_aoi"]), ["perturbation_id", "aoi"])
    return _result(
        "eye_aoi_assignment_stability",
        detail=source,
        overall=overall,
        group_summaries=by_group,
        aoi_level=aoi_level,
        caveat="Stability frequencies describe sensitivity to the specified perturbations; they are not probabilities that AOI assignments are scientifically true.",
    )


def estimate_fixation_assignment_probability(
    assignments: Mapping[str, Sequence[Any]] | pd.DataFrame,
    *,
    ids: Sequence[Any] | None = None,
    include_baseline: bool = True,
) -> pd.DataFrame:
    """Estimate empirical assignment frequencies across a declared perturbation set."""
    if isinstance(assignments, pd.DataFrame):
        required = {"perturbation_id", "observation_id", "aoi_assignment"}
        if not required.issubset(assignments.columns):
            raise EyeProcessValidationError("Assignment table must contain perturbation_id, observation_id, and aoi_assignment.")
        long = assignments.loc[:, list(required)].copy()
    else:
        rows = []
        lengths = {len(v) for v in assignments.values()}
        if len(lengths) != 1:
            raise EyeProcessValidationError("All perturbation assignment vectors must have the same length.")
        n = next(iter(lengths), 0)
        obs = list(ids) if ids is not None else list(range(1, n + 1))
        if len(obs) != n:
            raise EyeProcessValidationError("`ids` must match the assignment vector length.")
        for pid, labels in assignments.items():
            if not include_baseline and str(pid) == "baseline":
                continue
            rows.append(pd.DataFrame({"perturbation_id": str(pid), "observation_id": obs, "aoi_assignment": list(labels)}))
        long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["perturbation_id", "observation_id", "aoi_assignment"])
    long = long.dropna(subset=["aoi_assignment"])
    if long.empty:
        return pd.DataFrame(columns=["observation_id", "aoi", "assignment_count", "n_perturbations", "assignment_frequency"])
    counts = long.groupby(["observation_id", "aoi_assignment"], dropna=False).size().rename("assignment_count").reset_index()
    denominators = long.groupby("observation_id")["perturbation_id"].nunique().rename("n_perturbations").reset_index()
    out = counts.merge(denominators, on="observation_id", how="left")
    out["assignment_frequency"] = out["assignment_count"] / out["n_perturbations"]
    out = out.rename(columns={"aoi_assignment": "aoi"})
    out.attrs["caveat"] = "Assignment frequency is a descriptive perturbation frequency, not a posterior probability of true AOI membership."
    return out


def recompute_aoi_features(
    data: pd.DataFrame,
    assignments: Sequence[Any],
    *,
    participant_col: str | None = None,
    trial_col: str | None = None,
    duration_col: str | None = None,
    time_col: str | None = None,
    perturbation_id: str | None = None,
    aoi_levels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Recompute AOI features without dropping observed zero-assignment cells.

    A participant/trial × AOI cell receives a structural zero count only when
    the group contains at least one non-missing AOI assignment opportunity.
    Groups whose assignments are all missing retain NA counts/inspection.
    Dwell is NA when an assigned observation has missing duration, and
    first-fixation timing is never inferred from row order.
    """
    frame = _frame(data, "data")
    if len(assignments) != len(frame):
        raise EyeProcessValidationError("`assignments` must contain one value per data row.")
    frame = frame.copy()
    frame["aoi_assignment"] = list(assignments)
    group_cols = [c for c in (participant_col, trial_col) if c is not None]
    missing_groups = [c for c in group_cols if c not in frame.columns]
    if missing_groups:
        raise EyeProcessValidationError("Missing grouping column(s): " + ", ".join(missing_groups))
    if duration_col is not None and duration_col not in frame.columns:
        raise EyeProcessValidationError(f"`duration_col` `{duration_col}` is absent.")
    if time_col is not None and time_col not in frame.columns:
        raise EyeProcessValidationError(f"`time_col` `{time_col}` is absent.")
    if duration_col is None:
        warnings.warn(
            "No `duration_col` supplied; dwell is returned as NA rather than inferred.",
            RuntimeWarning,
            stacklevel=2,
        )
    if time_col is None:
        warnings.warn(
            "No `time_col` supplied; first_fixation is returned as NA rather than inferred from row order.",
            RuntimeWarning,
            stacklevel=2,
        )

    if aoi_levels is None:
        inferred = (
            frame.loc[
                frame["aoi_assignment"].notna()
                & ~frame["aoi_assignment"].isin([OUTSIDE, AMBIGUOUS]),
                "aoi_assignment",
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        levels = inferred
    else:
        levels = [str(v) for v in aoi_levels]
        if any(not v for v in levels) or len(set(levels)) != len(levels):
            raise EyeProcessValidationError("`aoi_levels` must contain unique non-empty AOI identifiers.")

    columns = group_cols + [
        "aoi",
        "fixation_count",
        "dwell",
        "first_fixation",
        "inspected",
        "n_valid_observations",
        "n_missing_observations",
        "duration_complete",
        "time_complete",
        "perturbation_id",
    ]
    if not levels:
        return pd.DataFrame(columns=columns)

    grouped: Any
    if group_cols:
        grouped = frame.groupby(group_cols, dropna=False, sort=True)
    else:
        grouped = [((), frame)]

    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        elif not group_cols:
            key = ()
        group_values = {k: v for k, v in zip(group_cols, key)}
        valid_assignment = group["aoi_assignment"].notna()
        n_valid = int(valid_assignment.sum())
        n_missing = int((~valid_assignment).sum())

        for aoi in levels:
            row: dict[str, Any] = dict(group_values)
            row["aoi"] = aoi
            selected = group.loc[group["aoi_assignment"].astype("string").eq(aoi).fillna(False)]
            count = int(len(selected))

            if n_valid == 0:
                row["fixation_count"] = pd.NA
                row["dwell"] = np.nan
                row["first_fixation"] = np.nan
                row["inspected"] = pd.NA
                row["duration_complete"] = pd.NA
                row["time_complete"] = pd.NA
            else:
                row["fixation_count"] = count
                row["inspected"] = bool(count > 0)
                if duration_col is None:
                    row["dwell"] = np.nan
                    row["duration_complete"] = pd.NA
                elif count == 0:
                    row["dwell"] = 0.0
                    row["duration_complete"] = True
                else:
                    durations = pd.to_numeric(selected[duration_col], errors="coerce")
                    duration_complete = bool(durations.notna().all())
                    row["duration_complete"] = duration_complete
                    row["dwell"] = float(durations.sum()) if duration_complete else np.nan

                if time_col is None:
                    row["first_fixation"] = np.nan
                    row["time_complete"] = pd.NA
                elif count == 0:
                    row["first_fixation"] = np.nan
                    row["time_complete"] = True
                else:
                    times = pd.to_numeric(selected[time_col], errors="coerce")
                    time_complete = bool(times.notna().all())
                    row["time_complete"] = time_complete
                    row["first_fixation"] = float(times.min()) if times.notna().any() else np.nan

            row["n_valid_observations"] = n_valid
            row["n_missing_observations"] = n_missing
            row["perturbation_id"] = perturbation_id
            rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    if "fixation_count" in out:
        out["fixation_count"] = out["fixation_count"].astype("Int64")
    if "inspected" in out:
        out["inspected"] = out["inspected"].astype("boolean")
    if "duration_complete" in out:
        out["duration_complete"] = out["duration_complete"].astype("boolean")
    if "time_complete" in out:
        out["time_complete"] = out["time_complete"].astype("boolean")
    return out


def _validate_model_table(table: Any, perturbation_id: str) -> pd.DataFrame:
    frame = _frame(table, "model callback result")
    required = ["term", "estimate", "SE", "CI_low", "CI_high", "p_value", "model_converged", "N"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise EyeProcessValidationError(
            "Model callback result is missing required column(s): " + ", ".join(missing)
        )
    out = frame.copy()

    raw_convergence = out["model_converged"]
    valid_boolean = raw_convergence.map(
        lambda v: pd.isna(v) or isinstance(v, (bool, np.bool_)) or v in (0, 1)
    )
    if not bool(valid_boolean.all()):
        raise EyeProcessValidationError(
            "`model_converged` must contain booleans, 0/1, or missing values; strings are not accepted."
        )
    out["model_converged"] = raw_convergence.astype("boolean")

    for column in ("estimate", "SE", "CI_low", "CI_high", "N"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if (out["N"].dropna() < 0).any():
        raise EyeProcessValidationError("Model callback `N` must be non-negative when supplied.")

    converged = out["model_converged"].fillna(False).astype(bool)
    if converged.any():
        required_finite = ["estimate", "SE", "CI_low", "CI_high", "N"]
        bad = ~np.isfinite(out.loc[converged, required_finite].to_numpy(dtype=float))
        if bad.any():
            raise EyeProcessValidationError(
                "Converged model rows must contain finite estimate, SE, CI_low, CI_high, and N values."
            )
        if (out.loc[converged, "N"] <= 0).any():
            raise EyeProcessValidationError("Converged model rows must report N > 0.")

    out["perturbation_id"] = perturbation_id
    est = out["estimate"].to_numpy(dtype=float)
    out["direction"] = np.where(
        ~np.isfinite(est),
        "missing",
        np.where(est > 0, "positive", np.where(est < 0, "negative", "zero")),
    )
    ordered = ["perturbation_id"] + required + ["direction"]
    extras = [c for c in out.columns if c not in set(ordered)]
    return out[ordered + extras]
