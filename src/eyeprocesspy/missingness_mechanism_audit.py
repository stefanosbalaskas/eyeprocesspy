"""Experimental descriptive audits for explicitly observed versus unavailable samples.

No MCAR/MAR/MNAR mechanism is inferred, no values are silently interpolated,
and pattern-mixture deltas are descriptive sensitivity scenarios, not identification.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _frame(data: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a nonempty pandas DataFrame")
    if len(set(required)) != len(required):
        raise ValueError("required column names must be unique")
    absent = sorted(set(required) - set(data.columns))
    if absent:
        raise ValueError(f"missing required columns: {absent}")
    return data.copy(deep=True)


def _observed(values: pd.Series) -> np.ndarray:
    if values.isna().any():
        raise ValueError("observation indicator cannot be missing")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    if not values.isin([0, 1, True, False]).all():
        raise ValueError("observation indicator must contain only booleans or 0/1")
    return values.to_numpy(dtype=bool)


def profile_missingness_mechanism(
    data: pd.DataFrame,
    *,
    observed_col: str,
    group_cols: Sequence[str] = ("participant_id", "trial_id"),
    time_col: str | None = None,
) -> dict[str, pd.DataFrame | str]:
    """Describe availability and consecutive observed-row dropout runs.

    A recorded row with observation=False is distinct from a row absent from the
    export. Gaps between entirely absent rows cannot be recovered without an
    independently declared expected sample schedule.
    """
    groups = tuple(group_cols)
    if not groups:
        raise ValueError("at least one grouping column is required")
    required = [*groups, observed_col, *([time_col] if time_col else [])]
    frame = _frame(data, required)
    if frame[groups].isna().any().any():
        raise ValueError("group identities cannot be missing")
    frame["_observed_audit"] = _observed(frame[observed_col])
    if time_col:
        times = pd.to_numeric(frame[time_col], errors="coerce")
        if times.isna().any() or not np.isfinite(times.to_numpy(dtype=float)).all():
            raise ValueError("timestamps must be finite")
        frame["_time_audit"] = times
    summary, runs = [], []
    grouper = groups[0] if len(groups) == 1 else list(groups)
    for key, subset in frame.groupby(grouper, sort=False, dropna=False):
        keys = (key,) if len(groups) == 1 else key
        identity = dict(zip(groups, keys))
        if time_col:
            subset = subset.sort_values("_time_audit", kind="stable")
            t = subset["_time_audit"].to_numpy(dtype=float)
            if np.any(np.diff(t) <= 0):
                raise ValueError("timestamps must strictly increase within each group")
        indicators = subset["_observed_audit"].to_numpy(dtype=bool)
        n = len(indicators)
        n_missing = int((~indicators).sum())
        summary.append({
            **identity, "n_rows": n, "n_observed": n - n_missing,
            "n_missing": n_missing, "missing_fraction": n_missing / n,
            "mechanism": "not_identified",
        })
        active_start = None
        missing_positions = np.flatnonzero(~indicators)
        for position in range(n + 1):
            missing = position < n and not indicators[position]
            if missing and active_start is None:
                active_start = position
            if not missing and active_start is not None:
                end = position - 1
                runs.append({
                    **identity, "start_row": active_start, "end_row": end,
                    "n_recorded_missing_rows": position - active_start,
                    "start_time": float(t[active_start]) if time_col else np.nan,
                    "end_time": float(t[end]) if time_col else np.nan,
                })
                active_start = None
        assert n_missing == len(missing_positions)
    return {
        "summary": pd.DataFrame(summary),
        "gap_runs": pd.DataFrame(
            runs, columns=[*groups, "start_row", "end_row", "n_recorded_missing_rows", "start_time", "end_time"]
        ),
        "interpretation": (
            "Descriptive row-availability only; missingness mechanisms and unrecorded samples "
            "are not identified from observed rows."
        ),
    }


def compare_missingness_sensitivity(
    data: pd.DataFrame,
    *,
    outcome_col: str,
    observed_col: str,
    group_cols: Sequence[str] = ("condition",),
    deltas: Sequence[float] = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Descriptive pattern-mixture mean scenarios for unavailable outcome values.

    For each group, missing outcomes are assigned observed_group_mean + delta.
    Deltas must use the actual outcome units. No inferential CI or MNAR mechanism
    is identified by this calculation.
    """
    groups = tuple(group_cols)
    if not groups or not deltas:
        raise ValueError("group_cols and deltas must be nonempty")
    frame = _frame(data, [*groups, observed_col, outcome_col])
    if frame[list(groups)].isna().any().any():
        raise ValueError("group identities cannot be missing")
    frame["_observed_audit"] = _observed(frame[observed_col])
    numeric = pd.to_numeric(frame[outcome_col], errors="coerce")
    observed_rows = frame["_observed_audit"].to_numpy(dtype=bool)
    if not np.isfinite(numeric.to_numpy(dtype=float)[observed_rows]).all():
        raise ValueError("observed outcomes must be finite numeric values")
    delta_values = [float(v) for v in deltas]
    if not all(np.isfinite(delta_values)):
        raise ValueError("deltas must be finite")
    frame["_outcome_audit"] = numeric
    grouper = groups[0] if len(groups) == 1 else list(groups)
    rows = []
    for key, subset in frame.groupby(grouper, sort=False, dropna=False):
        keys = (key,) if len(groups) == 1 else key
        identity = dict(zip(groups, keys))
        mask = subset["_observed_audit"].to_numpy(dtype=bool)
        if not mask.any():
            raise ValueError("each group must have at least one observed outcome")
        mean = float(subset.loc[mask, "_outcome_audit"].mean())
        frac = float((~mask).mean())
        for delta in delta_values:
            rows.append({
                **identity, "delta_outcome_units": delta, "n_rows": len(mask),
                "n_observed": int(mask.sum()), "missing_fraction": frac,
                "observed_mean": mean, "scenario_mean": mean + frac * delta,
                "inference_status": "descriptive_sensitivity_only",
            })
    return pd.DataFrame(rows)
