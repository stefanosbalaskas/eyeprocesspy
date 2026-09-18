"""Vendor-neutral censored gaze-latency survival analysis.

Scientific contract
-------------------
A trial with a known usable observation window but no target event is retained
as a right-censored observation. A trial whose event status is unknown because
the observation window is incomplete or gaze quality is unusable is retained as
a review row and is never silently converted to censoring.
"""
from __future__ import annotations

import json
import math
import platform
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

CANONICAL_GAZE_SURVIVAL_COLUMNS = [
    "participant_id",
    "trial_id",
    "stimulus_id",
    "condition",
    "target_aoi",
    "time_origin",
    "event_time",
    "censor_time",
    "analysis_time",
    "event_observed",
    "event_type",
    "n_valid_samples",
    "valid_data_fraction",
    "trial_duration",
]


@dataclass
class GazeSurvivalFit:
    """Container for a fitted gaze-survival model."""

    model_family: str
    backend: str
    result: Any
    data: pd.DataFrame
    formula: str | None = None
    covariate_names: list[str] = field(default_factory=list)
    design_info: Any = None
    participant_col: str = "participant_id"
    time_col: str = "analysis_time"
    event_col: str = "event_observed"
    repeated_structure: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _software_version() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "statsmodels": _package_version("statsmodels"),
        "lifelines": _package_version("lifelines"),
        "patsy": _package_version("patsy"),
        "matplotlib": _package_version("matplotlib"),
        "eyeprocesspy": _package_version("eyeprocesspy"),
    }


def _require_optional(package: str, purpose: str) -> None:
    if _package_version(package) is None:
        raise ImportError(
            f"Optional dependency {package!r} is required {purpose}. "
            "Install eyeprocesspy with the survival extra or install the dependency explicitly."
        )


def _provenance(**kwargs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"software": _software_version()}
    out.update({key: value for key, value in kwargs.items() if value is not None})
    return out


def _as_dataframe(x: Any, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return x.copy()


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    cols = set(columns)
    return next((candidate for candidate in candidates if candidate in cols), None)


def _event_time_for_trial(
    events: pd.DataFrame,
    target_aoi: str,
    event_type: str,
    time_col: str,
    aoi_col: str,
    episode_type_col: str | None,
) -> float | None:
    z = events.sort_values(time_col, kind="mergesort").copy()
    if episode_type_col and episode_type_col in z:
        episode_type = z[episode_type_col].astype(str).str.lower()
        if event_type == "first_fixation":
            z = z[episode_type.eq("fixation")]
        elif event_type in {
            "first_aoi_entry",
            "first_evidence_inspection",
            "first_revisit",
            "first_transition_into_target",
            "disengagement",
        }:
            keep = episode_type.isin(["aoi_visit", "fixation"])
            if keep.any():
                z = z[keep]
    if z.empty:
        return None

    labels = z[aoi_col].astype(str)
    target = labels.eq(str(target_aoi))
    if event_type in {"first_fixation", "first_aoi_entry", "first_evidence_inspection"}:
        q = z[target]
        return None if q.empty else float(q.iloc[0][time_col])

    # Revisit/transition/disengagement are visit-level concepts. Collapse
    # consecutive identical AOI labels so repeated fixations within one visit do
    # not become false revisits.
    run_start = labels.ne(labels.shift(1))
    runs = z.loc[run_start].copy()
    run_labels = runs[aoi_col].astype(str)
    run_target = run_labels.eq(str(target_aoi))

    if event_type == "first_revisit":
        q = runs[run_target]
        return None if len(q) < 2 else float(q.iloc[1][time_col])
    if event_type == "first_transition_into_target":
        prev = run_labels.shift(1)
        q = runs[run_target & prev.notna() & ~prev.eq(str(target_aoi))]
        return None if q.empty else float(q.iloc[0][time_col])
    if event_type == "disengagement":
        idx = np.flatnonzero(run_target.to_numpy())
        if not idx.size:
            return None
        after = runs.iloc[idx[0] + 1 :]
        q = after[~after[aoi_col].astype(str).eq(str(target_aoi))]
        return None if q.empty else float(q.iloc[0][time_col])
    raise ValueError(f"Unsupported event_type: {event_type!r}.")


def prepare_gaze_survival_data(
    trials: pd.DataFrame,
    events: pd.DataFrame | None = None,
    *,
    target_aoi: str | None = None,
    event_type: str = "first_fixation",
    participant_col: str = "participant_id",
    trial_col: str = "trial_id",
    recording_col: str = "recording_id",
    stimulus_col: str = "stimulus_id",
    condition_col: str = "condition",
    trial_start_col: str | None = None,
    trial_end_col: str | None = None,
    time_origin_col: str | None = None,
    observation_end_reason_col: str | None = None,
    event_time_col: str = "start_time",
    event_aoi_col: str = "aoi_id",
    event_trial_col: str = "trial_id",
    event_participant_col: str = "participant_id",
    event_recording_col: str = "recording_id",
    episode_type_col: str = "episode_type",
    event_observed_col: str = "event_observed",
    supplied_event_time_col: str = "event_time",
    supplied_censor_time_col: str = "censor_time",
    n_valid_samples_col: str = "n_valid_samples",
    valid_fraction_col: str = "valid_data_fraction",
    valid_observation_col: str | None = None,
    min_valid_fraction: float | None = None,
    time_origin: str = "trial_start",
    time_unit: str = "seconds",
    source_data: str | None = None,
    preprocessing_specification: str | None = None,
    event_detector: str | None = None,
    aoi_specification: str | None = None,
    quality_rules: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Construct the canonical one-row-per-trial gaze-survival table.

    Absence of a target event becomes right censoring only when ``events`` is
    supplied and a complete usable trial observation window is known. Without
    event evidence, callers must provide ``event_observed`` explicitly.
    """
    d = _as_dataframe(trials, "trials")
    if participant_col not in d or trial_col not in d:
        raise ValueError(f"trials must contain {participant_col!r} and {trial_col!r}.")
    if d.duplicated([participant_col, trial_col]).any():
        dup = d.loc[
            d.duplicated([participant_col, trial_col], keep=False),
            [participant_col, trial_col],
        ].head()
        raise ValueError(
            "Duplicated participant/trial rows are not allowed: "
            f"{dup.to_dict('records')}"
        )
    if time_unit not in {"seconds", "milliseconds"}:
        raise ValueError("time_unit must be 'seconds' or 'milliseconds'.")
    if min_valid_fraction is not None and not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in [0, 1].")

    start_col = trial_start_col or _first_existing(
        d.columns, ["start_time", "trial_start", "trial_start_time"]
    )
    end_col = trial_end_col or _first_existing(
        d.columns, ["end_time", "trial_end", "trial_end_time"]
    )
    if (start_col is None or end_col is None) and supplied_censor_time_col not in d:
        raise ValueError(
            "A complete observation window requires trial start/end columns "
            "or an explicit censor_time column."
        )

    scale = 0.001 if time_unit == "milliseconds" else 1.0
    starts = (
        pd.to_numeric(d[start_col], errors="coerce") * scale
        if start_col is not None
        else pd.Series(np.zeros(len(d)), index=d.index, dtype=float)
    )
    ends = (
        pd.to_numeric(d[end_col], errors="coerce") * scale
        if end_col is not None
        else pd.Series(np.nan, index=d.index, dtype=float)
    )

    origin_col = time_origin_col
    if origin_col is None and time_origin != "trial_start" and time_origin in d.columns:
        origin_col = time_origin
    if origin_col is not None:
        if origin_col not in d:
            raise ValueError(f"time_origin_col {origin_col!r} is absent.")
        origins = pd.to_numeric(d[origin_col], errors="coerce") * scale
    elif time_origin == "trial_start":
        origins = starts.copy()
    elif end_col is None and supplied_censor_time_col in d:
        # Supplied censor times are assumed to already be durations from the
        # declared origin. This convention is explicit rather than inferred.
        origins = pd.Series(np.zeros(len(d)), index=d.index, dtype=float)
    else:
        raise ValueError(
            "A non-trial-start time origin requires time_origin_col (or a trial "
            "column with the same name as time_origin)."
        )

    if end_col is not None:
        censor = ends - origins
        trial_duration = ends - starts
    else:
        censor = pd.to_numeric(d[supplied_censor_time_col], errors="coerce") * scale
        trial_duration = (
            pd.to_numeric(d["trial_duration"], errors="coerce") * scale
            if "trial_duration" in d
            else censor.copy()
        )

    out = pd.DataFrame(index=d.index)
    out["participant_id"] = d[participant_col].astype(str)
    out["trial_id"] = d[trial_col].astype(str)
    if recording_col in d:
        out["recording_id"] = d[recording_col].astype(str)
    out["stimulus_id"] = d[stimulus_col].astype(str) if stimulus_col in d else pd.NA
    condition_source = (
        condition_col
        if condition_col in d
        else "condition_id"
        if condition_col == "condition" and "condition_id" in d
        else None
    )
    out["condition"] = d[condition_source].astype(str) if condition_source else pd.NA
    out["target_aoi"] = (
        str(target_aoi)
        if target_aoi is not None
        else d["target_aoi"].astype(str)
        if "target_aoi" in d
        else pd.NA
    )
    out["time_origin"] = time_origin
    out["event_time"] = np.nan
    out["censor_time"] = censor.astype(float)
    out["analysis_time"] = np.nan
    out["event_observed"] = np.nan
    out["event_type"] = event_type
    out["n_valid_samples"] = (
        pd.to_numeric(d[n_valid_samples_col], errors="coerce")
        if n_valid_samples_col in d
        else np.nan
    )
    out["valid_data_fraction"] = (
        pd.to_numeric(d[valid_fraction_col], errors="coerce")
        if valid_fraction_col in d
        else np.nan
    )
    out["trial_duration"] = trial_duration.astype(float)
    out["censor_reason"] = "target_event_not_observed"
    if observation_end_reason_col is not None:
        if observation_end_reason_col not in d:
            raise ValueError(
                f"observation_end_reason_col {observation_end_reason_col!r} is absent."
            )
        out["observation_end_reason"] = d[observation_end_reason_col].astype("string")
    else:
        out["observation_end_reason"] = (
            "trial_window_end" if end_col is not None else "explicit_censor_time"
        )
    out["analysis_eligible"] = True
    out["review_required"] = False

    complete_window = (
        out["censor_time"].notna()
        & np.isfinite(out["censor_time"])
        & out["censor_time"].ge(0)
        & out["trial_duration"].notna()
        & np.isfinite(out["trial_duration"])
        & out["trial_duration"].ge(0)
    )

    if valid_observation_col is not None:
        if valid_observation_col not in d:
            raise ValueError(f"valid_observation_col {valid_observation_col!r} is absent.")
        valid_obs = d[valid_observation_col].fillna(False).astype(bool)
        invalid_obs = ~valid_obs
        complete_window &= valid_obs
        out.loc[invalid_obs, "censor_reason"] = "invalid_observation_flag"
        out.loc[invalid_obs, "analysis_eligible"] = False
        out.loc[invalid_obs, "review_required"] = True

    if min_valid_fraction is not None and valid_fraction_col in d:
        poor = out["valid_data_fraction"].notna() & out["valid_data_fraction"].lt(
            min_valid_fraction
        )
        complete_window &= ~poor
        out.loc[poor, "censor_reason"] = "unusable_gaze_quality"
        out.loc[poor, "analysis_eligible"] = False
        out.loc[poor, "review_required"] = True

    event_key_mode: str | None = None
    if events is not None:
        if target_aoi is None:
            raise ValueError(
                "target_aoi must be supplied when deriving events from an event table."
            )
        ev = _as_dataframe(events, "events")
        missing = {event_time_col, event_aoi_col, event_trial_col}.difference(ev.columns)
        if missing:
            raise ValueError(f"events is missing required columns: {sorted(missing)}")
        participant_key_available = (
            event_participant_col in ev
            and ev[event_participant_col].notna().all()
            and ev[event_participant_col].astype(str).str.len().gt(0).all()
        )
        recording_key_available = (
            recording_col in d
            and event_recording_col in ev
            and ev[event_recording_col].notna().all()
            and ev[event_recording_col].astype(str).str.len().gt(0).all()
        )
        if participant_key_available:
            event_key_mode = "participant_trial"
        elif recording_key_available:
            event_key_mode = "recording_trial"
        elif not d[trial_col].astype(str).duplicated().any():
            event_key_mode = "trial_only"
        else:
            raise ValueError(
                "events lacks participant and recording identity while trial IDs "
                "repeat across participants. Provide participant or recording keys; "
                "trial-only matching would be ambiguous."
            )

        event_times: list[float | None] = []
        for _, trial in d.iterrows():
            mask = ev[event_trial_col].astype(str).eq(str(trial[trial_col]))
            if event_key_mode == "participant_trial":
                mask &= ev[event_participant_col].astype(str).eq(
                    str(trial[participant_col])
                )
            elif event_key_mode == "recording_trial":
                mask &= ev[event_recording_col].astype(str).eq(str(trial[recording_col]))
            q = ev[mask].copy()
            absolute_event_time = _event_time_for_trial(
                q,
                str(target_aoi),
                event_type,
                event_time_col,
                event_aoi_col,
                episode_type_col if episode_type_col in q else None,
            )
            if absolute_event_time is None:
                event_times.append(None)
            else:
                event_times.append(
                    absolute_event_time * scale - float(origins.loc[trial.name])
                )
        out["event_time"] = pd.Series(event_times, index=d.index, dtype=float)
        observed = out["event_time"].notna() & complete_window
        out.loc[complete_window, "event_observed"] = observed[complete_window].astype(int)
        out.loc[observed, "censor_reason"] = "event_observed"
    else:
        if event_observed_col not in d:
            raise ValueError(
                "Without an event table, event_observed must be supplied explicitly; "
                "missing event times are not automatically censored."
            )
        observed = pd.to_numeric(d[event_observed_col], errors="coerce")
        if not observed.dropna().isin([0, 1]).all():
            raise ValueError("event_observed must contain only 0/1 (plus NA).")
        if supplied_event_time_col in d:
            out["event_time"] = (
                pd.to_numeric(d[supplied_event_time_col], errors="coerce") * scale
            )
        out["event_observed"] = observed.astype(float)
        out.loc[out["event_observed"].eq(1), "censor_reason"] = "event_observed"
        out.loc[~complete_window, "event_observed"] = np.nan

    out.loc[~complete_window, "analysis_eligible"] = False
    out.loc[~complete_window, "review_required"] = True
    incomplete_reason = ~complete_window & out["censor_reason"].eq(
        "target_event_not_observed"
    )
    out.loc[incomplete_reason, "censor_reason"] = "incomplete_observation_window"
    out["analysis_time"] = np.where(
        out["event_observed"].eq(1),
        out["event_time"],
        np.where(out["event_observed"].eq(0), out["censor_time"], np.nan),
    )

    qrules = dict(quality_rules or {})
    if min_valid_fraction is not None:
        qrules.setdefault("min_valid_fraction", float(min_valid_fraction))
    if valid_observation_col is not None:
        qrules.setdefault("valid_observation_col", valid_observation_col)

    out["source_data"] = source_data or (
        "trial_table+event_table" if events is not None else "trial_level_survival_inputs"
    )
    if event_key_mode is not None:
        out["event_join_key"] = event_key_mode
    out["event_detector"] = event_detector or (
        "canonical_episodes" if events is not None else "supplied"
    )
    out["aoi_specification"] = aoi_specification or (
        f"target_aoi={target_aoi}" if target_aoi is not None else pd.NA
    )
    out["quality_rules"] = json.dumps(qrules, sort_keys=True)
    out["preprocessing_specification"] = (
        preprocessing_specification if preprocessing_specification is not None else pd.NA
    )
    out["model_specification"] = pd.NA
    out["software_version"] = json.dumps(_software_version(), sort_keys=True)

    issues = validate_gaze_survival_data(out, raise_on_error=False)
    errors = issues[issues["severity"].eq("error")]
    if not errors.empty:
        raise ValueError(
            "Invalid gaze survival data: " + "; ".join(errors["message"].tolist())
        )
    for message in issues.loc[issues["severity"].eq("warning"), "message"]:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return out.reset_index(drop=True)


def validate_gaze_survival_data(
    data: pd.DataFrame, *, raise_on_error: bool = True
) -> pd.DataFrame:
    """Validate canonical gaze-survival rows without modifying them."""
    d = _as_dataframe(data, "data")
    issues: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, n: int = 0) -> None:
        issues.append(
            {"severity": severity, "code": code, "n": int(n), "message": message}
        )

    missing = [column for column in CANONICAL_GAZE_SURVIVAL_COLUMNS if column not in d]
    if missing:
        add(
            "error",
            "missing_columns",
            f"Missing canonical columns: {', '.join(missing)}",
            len(missing),
        )
    else:
        duplicate = d.duplicated(["participant_id", "trial_id"])
        if duplicate.any():
            add(
                "error",
                "duplicated_trials",
                "Participant/trial keys must be unique.",
                int(duplicate.sum()),
            )
        observed = pd.to_numeric(d["event_observed"], errors="coerce")
        invalid_indicator = ~observed.dropna().isin([0, 1])
        if invalid_indicator.any():
            add(
                "error",
                "invalid_event_indicator",
                "event_observed must be 0/1 or NA for non-analyzable rows.",
                int(invalid_indicator.sum()),
            )
        analysis_time = pd.to_numeric(d["analysis_time"], errors="coerce")
        censor_time = pd.to_numeric(d["censor_time"], errors="coerce")
        event_time = pd.to_numeric(d["event_time"], errors="coerce")
        trial_duration = pd.to_numeric(d["trial_duration"], errors="coerce")
        negative = (
            analysis_time.lt(0)
            | censor_time.lt(0)
            | event_time.lt(0)
            | trial_duration.lt(0)
        )
        if negative.fillna(False).any():
            add(
                "error",
                "negative_time",
                "Latency, censoring, and trial-duration values cannot be negative.",
                int(negative.fillna(False).sum()),
            )
        missing_window = censor_time.isna() | trial_duration.isna()
        analyzable = observed.isin([0, 1])
        if (missing_window & analyzable).any():
            add(
                "error",
                "missing_observation_window",
                "Analyzable rows require known censor_time and trial_duration.",
                int((missing_window & analyzable).sum()),
            )
        observed_missing = observed.eq(1) & event_time.isna()
        if observed_missing.any():
            add(
                "error",
                "observed_missing_event_time",
                "Observed events require event_time.",
                int(observed_missing.sum()),
            )
        event_after_censor = (
            observed.eq(1)
            & event_time.notna()
            & censor_time.notna()
            & event_time.gt(censor_time + 1e-12)
        )
        if event_after_censor.any():
            add(
                "error",
                "event_after_censor",
                "event_time cannot exceed censor_time.",
                int(event_after_censor.sum()),
            )
        event_mismatch = (
            observed.eq(1)
            & event_time.notna()
            & ~np.isclose(analysis_time, event_time, equal_nan=False)
        )
        if event_mismatch.any():
            add(
                "error",
                "analysis_time_event_mismatch",
                "Observed rows require analysis_time == event_time.",
                int(event_mismatch.sum()),
            )
        censor_mismatch = (
            observed.eq(0)
            & censor_time.notna()
            & ~np.isclose(analysis_time, censor_time, equal_nan=False)
        )
        if censor_mismatch.any():
            add(
                "error",
                "analysis_time_censor_mismatch",
                "Right-censored rows require analysis_time == censor_time.",
                int(censor_mismatch.sum()),
            )
        zero_event = observed.eq(1) & event_time.eq(0)
        if zero_event.any():
            add(
                "warning",
                "event_at_time_zero",
                "Target event occurs at time zero; verify time origin and initial fixation.",
                int(zero_event.sum()),
            )
        zero_followup = observed.eq(0) & censor_time.eq(0)
        if zero_followup.any():
            add(
                "warning",
                "zero_followup_censoring",
                "Right-censored trials with zero follow-up contribute no time at risk; verify the observation window.",
                int(zero_followup.sum()),
            )
        review = observed.isna()
        if review.any():
            add(
                "warning",
                "non_analyzable_rows",
                "Rows with unknown event status are retained for review and are not valid right-censored observations.",
                int(review.sum()),
            )
        n_event = int(observed.eq(1).sum())
        n_analyzable = int(analyzable.sum())
        if n_analyzable and n_event < 5:
            add(
                "warning",
                "sparse_events",
                "Fewer than five observed events are available; inferential survival models may be unstable.",
                n_event,
            )
        if n_analyzable and n_event / n_analyzable < 0.1:
            add(
                "warning",
                "extreme_censoring",
                "Observed-event proportion is below 10%; report censoring and consider sensitivity analyses.",
                n_event,
            )
        valid_fraction = pd.to_numeric(d["valid_data_fraction"], errors="coerce")
        invalid_fraction = valid_fraction.notna() & ~valid_fraction.between(0, 1)
        if invalid_fraction.any():
            add(
                "error",
                "invalid_valid_fraction",
                "valid_data_fraction must lie in [0, 1].",
                int(invalid_fraction.sum()),
            )

    out = pd.DataFrame(issues, columns=["severity", "code", "n", "message"])
    if raise_on_error and not out.empty and out["severity"].eq("error").any():
        raise ValueError(
            "; ".join(out.loc[out["severity"].eq("error"), "message"].tolist())
        )
    return out


def _analysis_rows(data: pd.DataFrame) -> pd.DataFrame:
    validate_gaze_survival_data(data)
    d = data.copy()
    if d["event_observed"].isna().any():
        raise ValueError(
            "Non-analyzable rows are present. Resolve/review them explicitly before "
            "model fitting; they are not censored trials."
        )
    if "analysis_eligible" in d and (~d["analysis_eligible"].astype(bool)).any():
        raise ValueError(
            "analysis_eligible=FALSE rows are present. Explicitly resolve or exclude "
            "them before model fitting."
        )
    return d


def summarise_gaze_censoring(
    data: pd.DataFrame, by: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Summarize observed, censored, and review-required trials."""
    d = _as_dataframe(data, "data")
    validate_gaze_survival_data(d, raise_on_error=False)
    groups = [] if by is None else ([by] if isinstance(by, str) else list(by))
    for group in groups:
        if group not in d:
            raise ValueError(f"Grouping column {group!r} is absent.")
    iterator = [((), d)] if not groups else d.groupby(groups, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for key, z in iterator:
        if groups and not isinstance(key, tuple):
            key = (key,)
        observed = pd.to_numeric(z["event_observed"], errors="coerce")
        analyzable = observed.isin([0, 1])
        row = {group: value for group, value in zip(groups, key)}
        row.update(
            {
                "n_trials": len(z),
                "n_analyzable": int(analyzable.sum()),
                "n_observed_events": int(observed.eq(1).sum()),
                "n_censored": int(observed.eq(0).sum()),
                "n_review_required": int(observed.isna().sum()),
                "censoring_fraction": (
                    float(observed.eq(0).sum() / analyzable.sum())
                    if analyzable.sum()
                    else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_gaze_survival(
    data: pd.DataFrame, *, group: str | None = None, conf_level: float = 0.95
) -> pd.DataFrame:
    """Estimate Kaplan-Meier survival curves for gaze latency."""
    d = _analysis_rows(data)
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1.")
    if group is not None and group not in d:
        raise ValueError(f"Group column {group!r} is absent.")
    zcrit = stats.norm.ppf(1 - (1 - conf_level) / 2)
    rows: list[dict[str, Any]] = []
    iterator = [("all", d)] if group is None else d.groupby(group, dropna=False, sort=True)
    for label, z in iterator:
        time = pd.to_numeric(z["analysis_time"], errors="raise").to_numpy(float)
        event = pd.to_numeric(z["event_observed"], errors="raise").to_numpy(int)
        survival = 1.0
        greenwood = 0.0
        for point in np.sort(np.unique(time)):
            n_risk = int(np.sum(time >= point))
            n_events = int(np.sum((time == point) & (event == 1)))
            n_censored = int(np.sum((time == point) & (event == 0)))
            if n_events:
                survival *= 1 - n_events / n_risk
                if n_risk > n_events:
                    greenwood += n_events / (n_risk * (n_risk - n_events))
            se = survival * math.sqrt(greenwood)
            rows.append(
                {
                    "group": label,
                    "time": float(point),
                    "n_risk": n_risk,
                    "n_events": n_events,
                    "n_censored": n_censored,
                    "survival": survival,
                    "std_error": se,
                    "lower": max(0.0, survival - zcrit * se),
                    "upper": min(1.0, survival + zcrit * se),
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["provenance"] = _provenance(estimator="Kaplan-Meier", group=group)
    return out


def _design_matrix(formula: str, data: pd.DataFrame):
    _require_optional("patsy", "to build survival-model design matrices")
    import patsy

    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()
    if not rhs:
        raise ValueError("Formula must contain at least one predictor.")
    design = patsy.dmatrix(rhs, data, return_type="dataframe", NA_action="raise")
    info = design.design_info
    if "Intercept" in design:
        design = design.drop(columns="Intercept")
    if design.shape[1] == 0:
        raise ValueError("Cox/AFT models require at least one estimable covariate.")
    return design, info


def _base_provenance(
    data: pd.DataFrame, model_specification: str, estimator: str
) -> dict[str, Any]:
    fields = [
        "source_data",
        "preprocessing_specification",
        "event_detector",
        "event_join_key",
        "aoi_specification",
        "quality_rules",
        "time_origin",
    ]
    provenance = {
        field: sorted(set(data[field].dropna().astype(str)))
        for field in fields
        if field in data
    }
    provenance.update(
        {"model_specification": model_specification, "estimator": estimator}
    )
    return _provenance(**provenance)


def fit_gaze_cox_model(
    data: pd.DataFrame,
    formula: str,
    *,
    ties: str = "breslow",
    cluster: str | None = None,
) -> GazeSurvivalFit:
    """Fit a Cox PH model, optionally with cluster-robust uncertainty."""
    d = _analysis_rows(data)
    _require_optional("statsmodels", "to fit Cox proportional-hazards models")
    _require_optional("patsy", "to build survival-model design matrices")
    from statsmodels.duration.hazard_regression import PHReg

    if ties not in {"breslow", "efron"}:
        raise ValueError("ties must be 'breslow' or 'efron'.")
    design, info = _design_matrix(formula, d)
    model = PHReg(
        d["analysis_time"].to_numpy(float),
        design,
        status=d["event_observed"].to_numpy(int),
        ties=ties,
    )
    groups = None
    repeated = None
    if cluster is not None:
        if cluster not in d:
            raise ValueError(f"Cluster column {cluster!r} is absent.")
        groups = d[cluster].to_numpy()
        repeated = f"cluster_robust:{cluster}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(groups=groups)
    convergence_messages = [
        str(w.message)
        for w in caught
        if "converg" in str(w.message).lower() or "infinite" in str(w.message).lower()
    ]
    if convergence_messages:
        raise RuntimeError("Cox convergence failure: " + " | ".join(convergence_messages))
    for caught_warning in caught:
        warnings.warn(
            str(caught_warning.message), caught_warning.category, stacklevel=2
        )
    return GazeSurvivalFit(
        model_family="cox",
        backend="statsmodels.PHReg",
        result=result,
        data=d,
        formula=formula,
        covariate_names=list(design.columns),
        design_info=info,
        repeated_structure=repeated,
        provenance=_base_provenance(d, formula, f"Cox PH ({ties})"),
    )


def fit_gaze_mixed_cox_model(
    data: pd.DataFrame,
    formula: str,
    *,
    participant_col: str = "participant_id",
    structure: str | None = None,
    ties: str = "breslow",
) -> GazeSurvivalFit:
    """Fit a repeated-participant Cox model.

    Python currently provides participant-clustered sandwich uncertainty. It
    does not mislabel this estimator as latent frailty. ``structure='frailty'``
    fails explicitly; R eyeprocess provides the frailty implementation.
    """
    if structure is None:
        raise ValueError(
            "structure must be specified explicitly as 'cluster_robust' or 'frailty'."
        )
    if structure == "frailty":
        raise NotImplementedError(
            "A participant-level latent frailty Cox estimator is not available "
            "in the current Python backend. Use structure='cluster_robust' or "
            "the R eyeprocess frailty implementation."
        )
    if structure != "cluster_robust":
        raise ValueError("structure must be 'cluster_robust' or 'frailty'.")
    fit = fit_gaze_cox_model(
        data, formula, ties=ties, cluster=participant_col
    )
    fit.model_family = "cox_repeated"
    fit.participant_col = participant_col
    return fit


def fit_gaze_aft_model(
    data: pd.DataFrame,
    formula: str,
    *,
    distribution: str | None = None,
    maxiter: int = 2000,
) -> GazeSurvivalFit:
    """Fit an explicitly selected Weibull or log-normal AFT model.

    Estimation is delegated to the specialist lifelines survival backend;
    eyeprocesspy owns validation, provenance, and semantic output rather than
    reimplementing the AFT likelihood.
    """
    d = _analysis_rows(data)
    _require_optional("lifelines", "for parametric AFT regression")
    if distribution is None:
        raise ValueError(
            "distribution must be specified explicitly as 'weibull' or 'lognormal'."
        )
    distribution = distribution.lower().replace("-", "")
    if distribution not in {"weibull", "lognormal"}:
        raise ValueError("distribution must be 'weibull' or 'lognormal'.")
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if d["analysis_time"].le(0).any():
        raise ValueError(
            "AFT models require strictly positive analysis_time; zero-time events "
            "must be resolved or shifted by a pre-specified measurement-resolution rule."
        )

    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()
    if not rhs:
        raise ValueError("Formula must contain at least one predictor.")

    from lifelines import LogNormalAFTFitter, WeibullAFTFitter
    from lifelines.exceptions import ConvergenceError

    fitter = WeibullAFTFitter() if distribution == "weibull" else LogNormalAFTFitter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fitter.fit(
                d,
                duration_col="analysis_time",
                event_col="event_observed",
                formula=rhs,
                ancillary=False,
                fit_options={"maxiter": int(maxiter)},
            )
    except ConvergenceError as exc:
        raise RuntimeError(f"AFT convergence failure: {exc}") from exc

    convergence_messages = [
        str(w.message)
        for w in caught
        if "converg" in str(w.message).lower()
        or "singular" in str(w.message).lower()
        or "invert" in str(w.message).lower()
    ]
    if convergence_messages:
        raise RuntimeError("AFT convergence failure: " + " | ".join(convergence_messages))
    for caught_warning in caught:
        warnings.warn(
            str(caught_warning.message), caught_warning.category, stacklevel=2
        )

    if not np.isfinite(float(result.log_likelihood_)):
        raise RuntimeError("AFT convergence failure: non-finite log-likelihood.")
    params = result.params_
    if not np.isfinite(np.asarray(params, dtype=float)).all():
        raise RuntimeError("AFT convergence failure: non-finite parameter estimate.")

    location_param = "lambda_" if distribution == "weibull" else "mu_"
    if not isinstance(params.index, pd.MultiIndex):
        raise RuntimeError("Unexpected lifelines AFT parameter contract.")
    location_mask = params.index.get_level_values(0).astype(str) == location_param
    covariates = params.index.get_level_values(1)[location_mask].astype(str).tolist()
    if not covariates:
        raise RuntimeError("AFT backend returned no location-model coefficients.")

    return GazeSurvivalFit(
        model_family=f"aft_{distribution}",
        backend=f"lifelines.{type(result).__name__}",
        result=result,
        data=d,
        formula=formula,
        covariate_names=covariates,
        design_info=None,
        provenance=_base_provenance(d, formula, f"{distribution} AFT via lifelines"),
    )


def tidy_gaze_survival_model(
    fit: GazeSurvivalFit, *, conf_level: float = 0.95
) -> pd.DataFrame:
    """Return exponentiated Cox hazard ratios or AFT time ratios."""
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1.")
    zcrit = stats.norm.ppf(1 - (1 - conf_level) / 2)
    if fit.model_family.startswith("cox"):
        beta = np.asarray(fit.result.params)
        se = np.asarray(fit.result.bse)
        statistic = np.asarray(fit.result.tvalues)
        p_value = np.asarray(fit.result.pvalues)
        names = fit.covariate_names
        measure = "hazard_ratio"
    elif fit.model_family.startswith("aft"):
        location_param = "lambda_" if fit.model_family == "aft_weibull" else "mu_"
        summary = fit.result.summary
        if not isinstance(summary.index, pd.MultiIndex):
            raise RuntimeError("Unexpected lifelines AFT summary contract.")
        location = summary.xs(location_param, level=0)
        beta = location["coef"].to_numpy(float)
        se = location["se(coef)"].to_numpy(float)
        statistic = location["z"].to_numpy(float)
        p_value = location["p"].to_numpy(float)
        names = location.index.astype(str).tolist()
        measure = "time_ratio"
    else:
        raise TypeError("Unsupported fitted object.")
    return pd.DataFrame(
        {
            "term": names,
            "estimate_log_scale": beta,
            "std_error": se,
            measure: np.exp(beta),
            "conf_low": np.exp(beta - zcrit * se),
            "conf_high": np.exp(beta + zcrit * se),
            "statistic": statistic,
            "p_value": p_value,
            "effect_measure": measure,
        }
    )


def check_gaze_proportional_hazards(
    fit: GazeSurvivalFit, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Evaluate Cox PH using Schoenfeld-residual time trends.

    This is a scientific-contract analogue to ``survival::cox.zph``; exact
    numerical parity with R is not claimed.
    """
    if not fit.model_family.startswith("cox"):
        raise TypeError("PH diagnostics require a Cox model.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    residuals = np.asarray(fit.result.schoenfeld_residuals, float)
    event = fit.data[fit.event_col].to_numpy(int) == 1
    times = fit.data.loc[event, fit.time_col].to_numpy(float)
    if residuals.shape[0] == len(fit.data):
        residuals = residuals[event]
    rows = []
    for column, name in enumerate(fit.covariate_names):
        values = residuals[:, column]
        valid = np.isfinite(values) & np.isfinite(times) & (times > 0)
        if valid.sum() < 4:
            rho = p_value = np.nan
        else:
            rho, p_value = stats.spearmanr(np.log(times[valid]), values[valid])
        rows.append(
            {
                "term": name,
                "rho": rho,
                "p_value": p_value,
                "alpha": alpha,
                "ph_flag": bool(np.isfinite(p_value) and p_value < alpha),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["method"] = (
        "Schoenfeld residual Spearman trend; contract analogue to cox.zph, "
        "not exact numerical parity."
    )
    return out


def compare_gaze_survival_models(*fits: GazeSurvivalFit) -> pd.DataFrame:
    """Summarize fitted models without implying a cross-family winner.

    Cox information criteria use partial likelihood; AFT information criteria use
    full likelihood. The returned comparability flag is therefore false when
    these bases are mixed or analysis-row counts differ.
    """
    if not fits:
        raise ValueError("Supply one or more fitted gaze-survival models.")
    rows = []
    likelihood_bases: set[str] = set()
    sample_sizes: set[int] = set()
    for index, fit in enumerate(fits, 1):
        if not isinstance(fit, GazeSurvivalFit):
            raise TypeError("All inputs must be GazeSurvivalFit objects.")
        if fit.model_family.startswith("cox"):
            loglik = float(fit.result.llf)
            n_parameters = len(fit.result.params)
            n = len(fit.data)
            basis = "cox_partial_likelihood"
        else:
            loglik = float(fit.result.log_likelihood_)
            n_parameters = len(fit.result.params_)
            n = len(fit.data)
            basis = "full_likelihood"
        likelihood_bases.add(basis)
        sample_sizes.add(n)
        rows.append(
            {
                "model": f"model_{index}",
                "family": fit.model_family,
                "backend": fit.backend,
                "logLik": loglik,
                "AIC": -2 * loglik + 2 * n_parameters,
                "BIC": -2 * loglik + math.log(n) * n_parameters,
                "n": n,
                "likelihood_basis": basis,
            }
        )
    comparable = len(likelihood_bases) == 1 and len(sample_sizes) == 1
    if not comparable:
        warnings.warn(
            "Information criteria are not directly comparable across Cox partial-"
            "likelihood and AFT full-likelihood models or across different analysis-"
            "row counts. Use diagnostics and estimand-specific interpretation instead "
            "of ranking them by AIC/BIC.",
            RuntimeWarning,
            stacklevel=2,
        )
    out = pd.DataFrame(rows)
    out["information_criteria_comparable"] = comparable
    return out


def _cox_baseline(fit: GazeSurvivalFit) -> pd.DataFrame:
    design, _ = _design_matrix(fit.formula or "", fit.data)
    beta = np.asarray(fit.result.params)
    linear_predictor = design.to_numpy(float) @ beta
    time = fit.data[fit.time_col].to_numpy(float)
    event = fit.data[fit.event_col].to_numpy(int)
    cumulative_hazard = 0.0
    rows = []
    for point in np.sort(np.unique(time[event == 1])):
        risk = np.exp(np.clip(linear_predictor[time >= point], -700, 700)).sum()
        n_event = int(np.sum((time == point) & (event == 1)))
        cumulative_hazard += n_event / risk
        rows.append(
            {
                "time": float(point),
                "cum_hazard": cumulative_hazard,
                "survival": math.exp(-cumulative_hazard),
            }
        )
    return pd.DataFrame(rows)


def predict_gaze_survival(
    fit: GazeSurvivalFit, newdata: pd.DataFrame, times: Sequence[float]
) -> pd.DataFrame:
    """Predict survival probabilities at requested times."""
    new = _as_dataframe(newdata, "newdata")
    requested = np.asarray(times, float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("times must be a non-empty one-dimensional sequence.")
    if not np.isfinite(requested).all() or (requested < 0).any():
        raise ValueError("Prediction times must be finite and non-negative.")
    rows = []
    if fit.model_family.startswith("cox"):
        _require_optional("patsy", "to build survival-model design matrices")
        import patsy

        design = patsy.build_design_matrices(
            [fit.design_info], new, return_type="dataframe"
        )[0]
        if "Intercept" in design:
            design = design.drop(columns="Intercept")
        linear_predictor = design.to_numpy(float) @ np.asarray(fit.result.params)
        baseline = _cox_baseline(fit)
        for row, lp in enumerate(linear_predictor):
            for point in requested:
                q = baseline[baseline["time"] <= point]
                h0 = 0.0 if q.empty else float(q.iloc[-1]["cum_hazard"])
                rows.append(
                    {
                        "row": row,
                        "time": float(point),
                        "survival": math.exp(-h0 * math.exp(float(lp))),
                    }
                )
    elif fit.model_family.startswith("aft"):
        prediction = fit.result.predict_survival_function(new, times=requested)
        if prediction.shape[1] != len(new):
            raise RuntimeError("Unexpected lifelines AFT prediction contract.")
        for row in range(len(new)):
            for point in requested:
                rows.append(
                    {
                        "row": row,
                        "time": float(point),
                        "survival": float(prediction.loc[point].iloc[row]),
                    }
                )
    else:
        raise TypeError("Unsupported fitted object.")
    return pd.DataFrame(rows)


def estimate_gaze_latency_quantiles(
    obj: pd.DataFrame | GazeSurvivalFit,
    *,
    probs: Sequence[float] = (0.25, 0.5, 0.75),
    group: str | None = None,
    newdata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate event-time quantiles from KM or fitted models."""
    probabilities = np.asarray(probs, float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probs must be a non-empty one-dimensional sequence.")
    if not np.isfinite(probabilities).all() or ((probabilities <= 0) | (probabilities >= 1)).any():
        raise ValueError("probs must be finite and lie strictly between 0 and 1.")
    if isinstance(obj, pd.DataFrame):
        km = estimate_gaze_survival(obj, group=group)
        rows = []
        for label, z in km.groupby("group", dropna=False):
            for probability in probabilities:
                q = z[z["survival"] <= 1 - probability]
                rows.append(
                    {
                        "group": label,
                        "prob": float(probability),
                        "quantile": np.nan if q.empty else float(q.iloc[0]["time"]),
                    }
                )
        return pd.DataFrame(rows)

    fit = obj
    if not isinstance(fit, GazeSurvivalFit):
        raise TypeError("obj must be a survival-ready DataFrame or GazeSurvivalFit.")
    if newdata is None:
        newdata = fit.data.iloc[[0]].copy()
    new = _as_dataframe(newdata, "newdata")
    rows = []
    if fit.model_family.startswith("cox"):
        _require_optional("patsy", "to build survival-model design matrices")
        import patsy

        design = patsy.build_design_matrices(
            [fit.design_info], new, return_type="dataframe"
        )[0]
        if "Intercept" in design:
            design = design.drop(columns="Intercept")
        baseline = _cox_baseline(fit)
        lp = design.to_numpy(float) @ np.asarray(fit.result.params)
        for row, value in enumerate(lp):
            for probability in probabilities:
                target = -math.log(1 - float(probability)) / math.exp(float(value))
                q = baseline[baseline["cum_hazard"] >= target]
                rows.append(
                    {
                        "row": row,
                        "prob": float(probability),
                        "quantile": np.nan if q.empty else float(q.iloc[0]["time"]),
                    }
                )
    elif fit.model_family.startswith("aft"):
        for probability in probabilities:
            predicted = fit.result.predict_percentile(
                new, p=1 - float(probability)
            )
            values = np.asarray(predicted, dtype=float).reshape(-1)
            if len(values) != len(new):
                raise RuntimeError("Unexpected lifelines AFT quantile contract.")
            for row, value in enumerate(values):
                rows.append(
                    {
                        "row": row,
                        "prob": float(probability),
                        "quantile": float(value),
                    }
                )
    else:
        raise TypeError("Unsupported fitted object.")
    return pd.DataFrame(rows)


def _require_plotting() -> None:
    _require_optional("matplotlib", "to plot gaze-survival results")


def plot_gaze_survival_curve(
    data: pd.DataFrame, *, group: str | None = None, ax=None
):
    """Plot Kaplan-Meier gaze-survival curves."""
    _require_plotting()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    km = estimate_gaze_survival(data, group=group)
    for label, z in km.groupby("group", dropna=False):
        ax.step(
            np.r_[0, z["time"]],
            np.r_[1, z["survival"]],
            where="post",
            label=str(label),
        )
    ax.set(xlabel="Latency", ylabel="P(target event not yet observed)")
    if group is not None:
        ax.legend(title=group)
    return ax


def plot_gaze_cumulative_incidence(
    data: pd.DataFrame, *, group: str | None = None, ax=None
):
    """Plot 1-KM for one target-event definition.

    This is not a competing-risks cumulative-incidence estimator.
    """
    _require_plotting()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    km = estimate_gaze_survival(data, group=group)
    for label, z in km.groupby("group", dropna=False):
        ax.step(
            np.r_[0, z["time"]],
            np.r_[0, 1 - z["survival"]],
            where="post",
            label=str(label),
        )
    ax.set(xlabel="Latency", ylabel="Cumulative target-event probability (1-KM)")
    if group is not None:
        ax.legend(title=group)
    return ax


def plot_gaze_hazard(data: pd.DataFrame, *, group: str | None = None, ax=None):
    """Plot discrete Nelson-Aalen hazard increments for descriptive use."""
    _require_plotting()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    d = _analysis_rows(data)
    if group is not None and group not in d:
        raise ValueError(f"Group column {group!r} is absent.")
    iterator = [("all", d)] if group is None else d.groupby(group, dropna=False, sort=True)
    for label, z in iterator:
        time = z["analysis_time"].to_numpy(float)
        event = z["event_observed"].to_numpy(int)
        points = np.sort(np.unique(time[event == 1]))
        increments = [
            np.sum((time == point) & (event == 1)) / np.sum(time >= point)
            for point in points
        ]
        ax.step(points, increments, where="mid", label=str(label))
    ax.set(xlabel="Latency", ylabel="Nelson-Aalen hazard increment")
    if group is not None:
        ax.legend(title=group)
    return ax


def plot_gaze_cox_diagnostics(fit: GazeSurvivalFit, *, ax=None):
    """Plot Schoenfeld residuals against log event time for the first Cox term."""
    _require_plotting()
    import matplotlib.pyplot as plt

    if not fit.model_family.startswith("cox"):
        raise TypeError("Cox diagnostics require a Cox model.")
    if ax is None:
        _, ax = plt.subplots()
    residuals = np.asarray(fit.result.schoenfeld_residuals, float)
    event = fit.data[fit.event_col].to_numpy(int) == 1
    times = fit.data.loc[event, fit.time_col].to_numpy(float)
    if residuals.shape[0] == len(fit.data):
        residuals = residuals[event]
    if residuals.shape[1] == 0:
        return ax
    valid = np.isfinite(times) & (times > 0) & np.isfinite(residuals[:, 0])
    ax.scatter(np.log(times[valid]), residuals[valid, 0], s=15)
    ax.axhline(0, linewidth=1)
    ax.set(
        xlabel="log(event time)",
        ylabel=f"Schoenfeld residual: {fit.covariate_names[0]}",
    )
    return ax


def compare_gaze_survival_specifications(
    specifications: Mapping[str, pd.DataFrame],
    formula: str,
    *,
    model_families: Sequence[str],
    participant_col: str = "participant_id",
    ties: str = "breslow",
) -> pd.DataFrame:
    """Fit explicitly requested estimators across named survival specifications.

    ``specifications`` should contain separately prepared canonical survival
    tables representing alternative AOIs, detectors, time origins, fixation
    thresholds, or quality rules. No estimator is selected implicitly.
    """
    if not isinstance(specifications, Mapping) or not specifications:
        raise ValueError(
            "specifications must be a non-empty mapping of name -> survival table."
        )
    families = list(model_families)
    allowed = {"cox", "cox_cluster_robust", "aft_weibull", "aft_lognormal"}
    if not families or any(family not in allowed for family in families):
        raise ValueError(
            "model_families must explicitly contain one or more of: "
            "cox, cox_cluster_robust, aft_weibull, aft_lognormal."
        )
    rows: list[pd.DataFrame] = []
    for specification_name, data in specifications.items():
        d = _as_dataframe(data, f"specifications[{specification_name!r}]")
        censoring = summarise_gaze_censoring(d).iloc[0]
        for family in families:
            if family == "cox":
                fit = fit_gaze_cox_model(d, formula, ties=ties)
            elif family == "cox_cluster_robust":
                fit = fit_gaze_mixed_cox_model(
                    d,
                    formula,
                    participant_col=participant_col,
                    structure="cluster_robust",
                    ties=ties,
                )
            elif family == "aft_weibull":
                fit = fit_gaze_aft_model(d, formula, distribution="weibull")
            else:
                fit = fit_gaze_aft_model(d, formula, distribution="lognormal")
            tidy = tidy_gaze_survival_model(fit).copy()
            tidy.insert(0, "specification", str(specification_name))
            tidy.insert(1, "model_family", fit.model_family)
            tidy["n_trials"] = int(censoring["n_trials"])
            tidy["n_observed_events"] = int(censoring["n_observed_events"])
            tidy["n_censored"] = int(censoring["n_censored"])
            tidy["censoring_fraction"] = float(censoring["censoring_fraction"])
            for field_name in (
                "event_detector",
                "aoi_specification",
                "quality_rules",
                "preprocessing_specification",
                "time_origin",
            ):
                if field_name in d:
                    values = sorted(set(d[field_name].dropna().astype(str)))
                    tidy[field_name] = " | ".join(values)
            rows.append(tidy)
    return pd.concat(rows, ignore_index=True)


def report_gaze_survival_model(
    fit: GazeSurvivalFit, *, conf_level: float = 0.95
) -> dict[str, Any]:
    """Create a concise manuscript-ready reporting bundle."""
    d = fit.data
    censoring = summarise_gaze_censoring(d).iloc[0].to_dict()
    effects = tidy_gaze_survival_model(fit, conf_level=conf_level)
    diagnostic = "not applicable to AFT model"
    if fit.model_family.startswith("cox"):
        ph = check_gaze_proportional_hazards(fit)
        diagnostic = (
            "flagged PH time trend"
            if ph["ph_flag"].any()
            else "no PH time-trend flag at alpha=.05"
        )
    return {
        "N_participants": int(d[fit.participant_col].nunique()),
        "N_trials": int(len(d)),
        "N_observed_events": int(censoring["n_observed_events"]),
        "N_censored_trials": int(censoring["n_censored"]),
        "N_review_required": int(censoring["n_review_required"]),
        "censoring_percentage": 100 * float(censoring["censoring_fraction"]),
        "event_type": sorted(set(d["event_type"].dropna().astype(str))),
        "target_aoi": sorted(set(d["target_aoi"].dropna().astype(str))),
        "time_origin": sorted(set(d["time_origin"].dropna().astype(str))),
        "model_formula": fit.formula,
        "model_family": fit.model_family,
        "backend": fit.backend,
        "effect_measure": (
            "hazard ratio" if fit.model_family.startswith("cox") else "time ratio"
        ),
        "effects": effects,
        "random_or_frailty_structure": fit.repeated_structure,
        "diagnostic_result": diagnostic,
        "provenance": fit.provenance,
    }


def simulate_gaze_survival_inputs(
    kind: str = "disclosure",
    *,
    seed: int = 20260918,
    n_participants: int = 36,
    trials_per_participant: int = 3,
) -> dict[str, pd.DataFrame]:
    """Generate deterministic synthetic trial windows and canonical event rows."""
    if kind not in {"disclosure", "verification"}:
        raise ValueError("kind must be 'disclosure' or 'verification'.")
    if n_participants < 1 or trials_per_participant < 1:
        raise ValueError(
            "n_participants and trials_per_participant must be positive integers."
        )
    rng = np.random.default_rng(seed)
    conditions = (
        ["control", "minimal_disclosure", "detailed_disclosure"]
        if kind == "disclosure"
        else ["standard", "evidence_prompt"]
    )
    shifts = {
        "control": 0.35,
        "minimal_disclosure": 0.05,
        "detailed_disclosure": -0.20,
        "standard": 0.20,
        "evidence_prompt": -0.25,
    }
    target = "disclosure" if kind == "disclosure" else "source_evidence"
    episode_type = "fixation" if kind == "disclosure" else "aoi_visit"
    trial_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for participant_index in range(n_participants):
        participant = f"P{participant_index + 1:03d}"
        recording = f"R{participant_index + 1:03d}"
        participant_shift = rng.normal(0, 0.25)
        for trial_index in range(trials_per_participant):
            trial = f"{participant}_T{trial_index + 1:02d}"
            condition = conditions[
                (participant_index + trial_index) % len(conditions)
            ]
            trial_duration = 4.0
            latent = float(
                np.exp(
                    0.8
                    + shifts[condition]
                    + participant_shift
                    + rng.normal(0, 0.45)
                )
            )
            target_observed = latent <= trial_duration
            trial_rows.append(
                {
                    "recording_id": recording,
                    "participant_id": participant,
                    "trial_id": trial,
                    "stimulus_id": f"S{trial_index + 1:02d}",
                    "condition_id": condition,
                    "start_time": 0.0,
                    "end_time": trial_duration,
                    "n_valid_samples": 230 + int(rng.integers(-10, 11)),
                    "valid_data_fraction": float(rng.uniform(0.91, 0.995)),
                    "observation_end_reason": "scheduled_trial_end",
                }
            )
            body_time = min(0.35 + 0.04 * trial_index, trial_duration - 0.05)
            event_rows.append(
                {
                    "recording_id": recording,
                    "trial_id": trial,
                    "start_time": body_time,
                    "aoi_id": "body",
                    "episode_type": episode_type,
                }
            )
            if target_observed:
                event_rows.append(
                    {
                        "recording_id": recording,
                        "trial_id": trial,
                        "start_time": latent,
                        "aoi_id": target,
                        "episode_type": episode_type,
                    }
                )
    events = pd.DataFrame(event_rows).sort_values(
        ["recording_id", "trial_id", "start_time"], kind="mergesort"
    )
    return {
        "trials": pd.DataFrame(trial_rows).reset_index(drop=True),
        "events": events.reset_index(drop=True),
    }


def simulate_gaze_survival_example(
    kind: str = "disclosure",
    *,
    seed: int = 20260918,
    n_participants: int = 36,
    trials_per_participant: int = 3,
) -> pd.DataFrame:
    """Generate a deterministic canonical censored gaze-latency example."""
    inputs = simulate_gaze_survival_inputs(
        kind,
        seed=seed,
        n_participants=n_participants,
        trials_per_participant=trials_per_participant,
    )
    target = "disclosure" if kind == "disclosure" else "source_evidence"
    event_type = "first_fixation" if kind == "disclosure" else "first_aoi_entry"
    return prepare_gaze_survival_data(
        inputs["trials"],
        inputs["events"],
        target_aoi=target,
        event_type=event_type,
        condition_col="condition_id",
        observation_end_reason_col="observation_end_reason",
        source_data=f"synthetic_{kind}_trial_event_inputs",
        preprocessing_specification="synthetic_truth_no_filtering",
        event_detector="synthetic_truth",
        aoi_specification=f"fixed synthetic AOI:{target}",
        quality_rules={"valid_fraction_min": 0.90, "synthetic_truth": True},
    )