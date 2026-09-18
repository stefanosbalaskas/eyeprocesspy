"""Vendor-neutral trial-level preparation for multilevel mediation.

Scientific contract
-------------------
This module prepares repeated-measures mediation data without collapsing trials
or silently changing observation states. In particular, missing gaze remains
missing, observed zero remains an observed zero, poor-quality trials are flagged
unless the caller explicitly requests mediator masking, and no analysis rows are
silently removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd


_ALLOWED_QUALITY_ACTIONS = {"flag", "mask_mediator"}


def _software_version() -> str:
    try:
        return version("eyeprocesspy")
    except PackageNotFoundError:
        return "development"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _fingerprint_frame(data: pd.DataFrame, columns: Sequence[str]) -> str:
    subset = data.loc[:, list(columns)].copy()
    hashed = pd.util.hash_pandas_object(subset, index=True).to_numpy(dtype="uint64")
    payload = hashed.tobytes() + "|".join(columns).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MultilevelMediationData:
    """Prepared trial-level mediation data plus audits and provenance."""

    data: pd.DataFrame
    columns: Mapping[str, str | None]
    levels: pd.DataFrame
    variance: pd.DataFrame
    missingness: pd.DataFrame
    trial_counts: pd.DataFrame
    validation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...]
    preparation_version: str = "0.1"
    fit_performed: bool = False

    @property
    def n_participants(self) -> int:
        return int(self.data[self.columns["participant"]].nunique(dropna=True))

    @property
    def n_rows(self) -> int:
        return int(len(self.data))

    @property
    def n_analysis_eligible(self) -> int:
        return int(self.data["mediation_analysis_eligible"].sum())


def _require_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    return data.copy(deep=True)


def _require_nonempty_name(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty column name.")
    return value


def _require_columns(data: pd.DataFrame, columns: Iterable[str | None]) -> None:
    required = [c for c in columns if c is not None]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def _numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(data[column], errors="coerce")
    invalid = data[column].notna() & values.isna()
    if bool(invalid.any()):
        examples = data.loc[invalid, column].astype(str).head(3).tolist()
        raise ValueError(
            f"`{column}` must be numeric for mediation preparation; "
            f"non-numeric examples: {examples}."
        )
    return values.astype(float)


def _indicator_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a strict boolean observation indicator without truthiness coercion."""
    raw = data[column]
    nonmissing = raw.dropna()
    if pd.api.types.is_bool_dtype(nonmissing.dtype):
        return raw.fillna(False).astype(bool)
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid_numeric = raw.notna() & numeric.isna()
    if bool(invalid_numeric.any()) or not bool(numeric.dropna().isin([0, 1]).all()):
        raise ValueError(f"`{column}` must contain only TRUE/FALSE or 0/1 values (plus missing).")
    return numeric.fillna(0).astype(int).astype(bool)


def _quality_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric quality values while rejecting nonnumeric observations."""
    raw = data[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    if bool(invalid.any()):
        examples = raw.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"`{column}` must be numeric; non-numeric examples: {examples}.")
    return numeric.astype(float)


def _check_derived_column_collisions(data: pd.DataFrame, columns: Iterable[str]) -> None:
    collisions = [name for name in columns if name in data.columns]
    if collisions:
        raise ValueError(
            "Input data already contains derived mediation column(s): "
            + ", ".join(collisions)
            + ". Rename or remove them explicitly before preparation."
        )


def center_within_participant(
    data: pd.DataFrame,
    value_col: str,
    participant_col: str = "participant_id",
    *,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Participant-mean center a numeric trial-level variable.

    Missing observations remain missing and are excluded only from the group
    mean used for centering; rows themselves are preserved.
    """
    out = _require_dataframe(data)
    _require_columns(out, [participant_col, value_col])
    values = _numeric_series(out, value_col)
    means = values.groupby(out[participant_col], dropna=False).transform("mean")
    out[output_col or f"{value_col}_within"] = values - means
    return out


def decompose_within_between(
    data: pd.DataFrame,
    columns: str | Iterable[str],
    participant_col: str = "participant_id",
    *,
    grand_mean_center_between: bool = False,
) -> pd.DataFrame:
    """Add within-person and between-person components for numeric variables."""
    out = _require_dataframe(data)
    names = [columns] if isinstance(columns, str) else list(columns)
    if not names:
        raise ValueError("`columns` must contain at least one variable.")
    _require_columns(out, [participant_col, *names])
    for name in names:
        values = _numeric_series(out, name)
        group_mean = values.groupby(out[participant_col], dropna=False).transform("mean")
        between = group_mean.copy()
        if grand_mean_center_between:
            grand = float(values.mean(skipna=True))
            between = between - grand
        out[f"{name}_within"] = values - group_mean
        out[f"{name}_between"] = between
    return out


def summarise_within_between_variance(
    data: pd.DataFrame,
    columns: str | Iterable[str],
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """Summarise observed within- and between-participant variance."""
    frame = _require_dataframe(data)
    names = [columns] if isinstance(columns, str) else list(columns)
    if not names:
        raise ValueError("`columns` must contain at least one variable.")
    _require_columns(frame, [participant_col, *names])
    rows: list[dict[str, Any]] = []
    for name in names:
        values = _numeric_series(frame, name)
        tmp = pd.DataFrame({"participant": frame[participant_col], "value": values})
        tmp = tmp.dropna(subset=["participant"])
        means = tmp.groupby("participant", dropna=False)["value"].mean()
        centered = tmp["value"] - tmp.groupby("participant", dropna=False)["value"].transform("mean")
        n_obs = int(tmp["value"].notna().sum())
        total_var = float(tmp["value"].var(ddof=1)) if n_obs > 1 else np.nan
        within_var = float(centered.var(ddof=1)) if centered.notna().sum() > 1 else np.nan
        between_var = float(means.var(ddof=1)) if means.notna().sum() > 1 else np.nan
        denom = within_var + between_var
        between_share = between_var / denom if np.isfinite(denom) and denom > 0 else np.nan
        rows.append(
            {
                "variable": name,
                "n_observed": n_obs,
                "n_missing": int(values.isna().sum()),
                "n_participants_observed": int(means.notna().sum()),
                "total_variance": total_var,
                "within_variance": within_var,
                "between_variance": between_var,
                "between_share": between_share,
            }
        )
    return pd.DataFrame(rows)


def identify_mediation_levels(
    data: pd.DataFrame,
    columns: str | Iterable[str],
    participant_col: str = "participant_id",
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Classify variables as within, between, both, or unidentified."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("`tolerance` must be a finite non-negative number.")
    summary = summarise_within_between_variance(data, columns, participant_col)
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        within = float(row["within_variance"]) if pd.notna(row["within_variance"]) else np.nan
        between = float(row["between_variance"]) if pd.notna(row["between_variance"]) else np.nan
        has_within = bool(np.isfinite(within) and within > tolerance)
        has_between = bool(np.isfinite(between) and between > tolerance)
        if has_within and has_between:
            level = "within_and_between"
        elif has_within:
            level = "within_only"
        elif has_between:
            level = "between_only"
        else:
            level = "constant_or_unidentified"
        rows.append(
            {
                "variable": row["variable"],
                "has_within_variation": has_within,
                "has_between_variation": has_between,
                "level": level,
            }
        )
    return pd.DataFrame(rows)


def audit_mediation_missingness(
    data: pd.DataFrame,
    *,
    x_col: str,
    mediator_col: str,
    outcome_col: str,
    participant_col: str = "participant_id",
    trial_col: str = "trial_id",
    quality_col: str | None = None,
    minimum_quality: float | None = None,
    mediator_observed_col: str | None = None,
    response_observed_col: str | None = None,
) -> pd.DataFrame:
    """Audit missingness while distinguishing missing gaze from true zero gaze."""
    frame = _require_dataframe(data)
    for value, name in [
        (x_col, "x_col"), (mediator_col, "mediator_col"), (outcome_col, "outcome_col"),
        (participant_col, "participant_col"), (trial_col, "trial_col"),
    ]:
        _require_nonempty_name(value, name)
    _require_columns(
        frame,
        [participant_col, trial_col, x_col, mediator_col, outcome_col, quality_col,
         mediator_observed_col, response_observed_col],
    )
    if (quality_col is None) != (minimum_quality is None):
        raise ValueError("`quality_col` and `minimum_quality` must be supplied together.")
    if minimum_quality is not None:
        if isinstance(minimum_quality, bool) or not isinstance(minimum_quality, (int, float)) or not np.isfinite(float(minimum_quality)):
            raise ValueError("`minimum_quality` must be one finite numeric value.")
    m = _numeric_series(frame, mediator_col)
    mediator_observed = m.notna()
    if mediator_observed_col is not None:
        mediator_observed &= _indicator_series(frame, mediator_observed_col)
    response_observed = frame[outcome_col].notna()
    if response_observed_col is not None:
        response_observed &= _indicator_series(frame, response_observed_col)
    poor_quality = pd.Series(False, index=frame.index, dtype=bool)
    if quality_col is not None:
        quality = _quality_series(frame, quality_col)
        poor_quality = quality.isna() | quality.lt(float(minimum_quality))
    observed_zero = mediator_observed & m.eq(0)
    categories = {
        "x_missing": frame[x_col].isna(),
        "mediator_not_observed": ~mediator_observed,
        "mediator_observed_zero": observed_zero,
        "mediator_observed_nonzero": mediator_observed & ~observed_zero,
        "poor_quality_trial": poor_quality,
        "response_missing": ~response_observed,
    }
    n = len(frame)
    return pd.DataFrame(
        [
            {
                "issue": key,
                "n": int(mask.sum()),
                "proportion": float(mask.mean()) if n else np.nan,
            }
            for key, mask in categories.items()
        ]
    )


def check_mediation_trial_counts(
    data: pd.DataFrame,
    participant_col: str = "participant_id",
    trial_col: str = "trial_id",
    *,
    minimum_trials: int = 2,
) -> pd.DataFrame:
    """Summarise repeated-measures support without excluding participants."""
    frame = _require_dataframe(data)
    _require_columns(frame, [participant_col, trial_col])
    if isinstance(minimum_trials, bool) or int(minimum_trials) != minimum_trials or minimum_trials < 1:
        raise ValueError("`minimum_trials` must be an integer of at least 1.")
    counts = (
        frame.groupby(participant_col, dropna=False)[trial_col]
        .agg(n_rows="size", n_trials="nunique")
        .reset_index()
    )
    counts["singleton"] = counts["n_trials"].eq(1)
    counts["below_minimum"] = counts["n_trials"].lt(int(minimum_trials))
    return counts


def validate_multilevel_mediation_data(
    data: pd.DataFrame,
    *,
    x_col: str,
    mediator_col: str,
    outcome_col: str,
    participant_col: str = "participant_id",
    trial_col: str = "trial_id",
    require_within_x: bool = True,
) -> dict[str, Any]:
    """Validate identifiers, decomposition inputs, and repeated-measures support."""
    frame = _require_dataframe(data)
    for value, name in [
        (x_col, "x_col"), (mediator_col, "mediator_col"), (outcome_col, "outcome_col"),
        (participant_col, "participant_col"), (trial_col, "trial_col")
    ]:
        _require_nonempty_name(value, name)
    _require_columns(frame, [participant_col, trial_col, x_col, mediator_col, outcome_col])
    issues: list[str] = []
    notes: list[str] = []
    if frame.empty:
        issues.append("data has no rows")
    if frame[participant_col].isna().any():
        issues.append("participant identifiers contain missing values")
    if frame[trial_col].isna().any():
        issues.append("trial identifiers contain missing values")
    if frame.duplicated([participant_col, trial_col]).any():
        issues.append("participant/trial identifiers are not unique")
    _numeric_series(frame, mediator_col)
    x_numeric = pd.to_numeric(frame[x_col], errors="coerce")
    if bool((frame[x_col].notna() & x_numeric.isna()).any()):
        issues.append("X must be numeric or explicitly coded before decomposition")
    levels = identify_mediation_levels(frame, [x_col, mediator_col], participant_col)
    lookup = levels.set_index("variable")
    if require_within_x and not bool(lookup.loc[x_col, "has_within_variation"]):
        issues.append("X has no within-participant variation")
    if not bool(lookup.loc[mediator_col, "has_within_variation"]):
        notes.append("mediator has no detectable within-participant variation")
    counts = check_mediation_trial_counts(frame, participant_col, trial_col)
    if bool(counts["singleton"].any()):
        notes.append("one or more participants have only one observed trial")
    if frame[outcome_col].isna().any():
        notes.append("outcome contains missing values; rows are preserved and flagged")
    return {
        "valid": len(issues) == 0,
        "issues": tuple(issues),
        "warnings": tuple(notes),
        "levels": levels,
        "trial_counts": counts,
        "n_rows": len(frame),
        "n_participants": int(frame[participant_col].nunique(dropna=True)),
    }


def _build_provenance(
    frame: pd.DataFrame,
    *,
    source_id: str | None,
    source_columns: Sequence[str],
    preprocessing_spec: Mapping[str, Any] | None,
    event_detector: Mapping[str, Any] | str | None,
    aoi_specification: Mapping[str, Any] | str | None,
    quality_rules: Mapping[str, Any],
    preparation_spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_fingerprint_sha256": _fingerprint_frame(frame, source_columns),
        "source_columns": list(source_columns),
        "preprocessing_specification": _json_safe(preprocessing_spec),
        "event_detector": _json_safe(event_detector),
        "aoi_specification": _json_safe(aoi_specification),
        "quality_rules": _json_safe(quality_rules),
        "preparation_specification": _json_safe(preparation_spec),
        "row_position_convention": {
            "base": 0,
            "meaning": "zero-based input row position",
        },
        "software": {"package": "eyeprocesspy", "version": _software_version()},
    }


def prepare_multilevel_mediation_data(
    data: pd.DataFrame,
    *,
    x_col: str,
    mediator_col: str,
    outcome_col: str,
    participant_col: str = "participant_id",
    trial_col: str = "trial_id",
    quality_col: str | None = None,
    minimum_quality: float | None = None,
    mediator_observed_col: str | None = None,
    response_observed_col: str | None = None,
    quality_action: str = "flag",
    grand_mean_center_between: bool = False,
    require_within_x: bool = True,
    source_id: str | None = None,
    preprocessing_spec: Mapping[str, Any] | None = None,
    event_detector: Mapping[str, Any] | str | None = None,
    aoi_specification: Mapping[str, Any] | str | None = None,
    warn: bool = True,
) -> MultilevelMediationData:
    """Prepare trial-level multilevel mediation data.

    No rows are dropped. ``quality_action='flag'`` is the default and preserves
    mediator values on poor-quality trials while marking those trials as not
    analysis eligible. ``quality_action='mask_mediator'`` is an explicit request
    to set poor-quality mediator values to missing before decomposition.
    """
    frame = _require_dataframe(data)
    for value, name in [
        (x_col, "x_col"), (mediator_col, "mediator_col"), (outcome_col, "outcome_col"),
        (participant_col, "participant_col"), (trial_col, "trial_col"),
    ]:
        _require_nonempty_name(value, name)
    _require_columns(
        frame,
        [participant_col, trial_col, x_col, mediator_col, outcome_col, quality_col,
         mediator_observed_col, response_observed_col],
    )
    derived = [
        f"{x_col}_within", f"{x_col}_between", f"{mediator_col}_within",
        f"{mediator_col}_between", "X_within", "X_between", "M_within",
        "M_between", "mediation_mediator_observed", "mediation_mediator_true_zero",
        "mediation_poor_quality", "mediation_response_observed",
        "mediation_mediator_state", "mediation_analysis_eligible",
    ]
    _check_derived_column_collisions(frame, derived)
    if quality_action not in _ALLOWED_QUALITY_ACTIONS:
        raise ValueError(
            "`quality_action` must be one of: " + ", ".join(sorted(_ALLOWED_QUALITY_ACTIONS))
        )
    if (quality_col is None) != (minimum_quality is None):
        raise ValueError("`quality_col` and `minimum_quality` must be supplied together.")
    if minimum_quality is not None:
        if isinstance(minimum_quality, bool) or not isinstance(minimum_quality, (int, float)) or not np.isfinite(float(minimum_quality)):
            raise ValueError("`minimum_quality` must be one finite numeric value.")

    validation = validate_multilevel_mediation_data(
        frame,
        x_col=x_col,
        mediator_col=mediator_col,
        outcome_col=outcome_col,
        participant_col=participant_col,
        trial_col=trial_col,
        require_within_x=require_within_x,
    )
    if not validation["valid"]:
        raise ValueError("Invalid multilevel mediation data: " + "; ".join(validation["issues"]))

    m_original = _numeric_series(frame, mediator_col)
    mediator_observed = m_original.notna()
    if mediator_observed_col is not None:
        mediator_observed &= _indicator_series(frame, mediator_observed_col)

    poor_quality = pd.Series(False, index=frame.index, dtype=bool)
    if quality_col is not None:
        quality = _quality_series(frame, quality_col)
        poor_quality = quality.isna() | quality.lt(float(minimum_quality))

    if quality_action == "mask_mediator":
        frame.loc[poor_quality, mediator_col] = np.nan
        mediator_observed &= ~poor_quality

    m = _numeric_series(frame, mediator_col)
    response_observed = frame[outcome_col].notna()
    if response_observed_col is not None:
        response_observed &= _indicator_series(frame, response_observed_col)

    decomposed = decompose_within_between(
        frame,
        [x_col, mediator_col],
        participant_col,
        grand_mean_center_between=grand_mean_center_between,
    )
    decomposed["X_within"] = decomposed[f"{x_col}_within"]
    decomposed["X_between"] = decomposed[f"{x_col}_between"]
    decomposed["M_within"] = decomposed[f"{mediator_col}_within"]
    decomposed["M_between"] = decomposed[f"{mediator_col}_between"]
    decomposed["mediation_mediator_observed"] = mediator_observed
    decomposed["mediation_mediator_true_zero"] = mediator_observed & m.eq(0)
    decomposed["mediation_poor_quality"] = poor_quality
    decomposed["mediation_response_observed"] = response_observed
    decomposed["mediation_mediator_state"] = np.select(
        [poor_quality, ~mediator_observed, mediator_observed & m.eq(0)],
        ["poor_quality", "not_observed", "observed_zero"],
        default="observed_nonzero",
    )
    decomposed["mediation_analysis_eligible"] = (
        decomposed[x_col].notna()
        & mediator_observed
        & response_observed
        & ~poor_quality
    )

    variance = summarise_within_between_variance(
        decomposed, [x_col, mediator_col], participant_col
    )
    levels = identify_mediation_levels(decomposed, [x_col, mediator_col], participant_col)
    missingness = audit_mediation_missingness(
        decomposed,
        x_col=x_col,
        mediator_col=mediator_col,
        outcome_col=outcome_col,
        participant_col=participant_col,
        trial_col=trial_col,
        quality_col=quality_col,
        minimum_quality=minimum_quality,
        mediator_observed_col=None,
        response_observed_col=response_observed_col,
    )
    trial_counts = check_mediation_trial_counts(decomposed, participant_col, trial_col)

    notes = list(validation["warnings"])
    if bool(poor_quality.any()):
        notes.append(
            "poor-quality trials are flagged; mediator values were "
            + ("explicitly masked before decomposition" if quality_action == "mask_mediator" else "retained")
        )
    if bool((~mediator_observed).any()):
        notes.append("missing mediator observations remain missing and were not converted to zero")
    if bool((mediator_observed & m.eq(0)).any()):
        notes.append("observed zero mediator values are retained and separately identified")

    source_columns = list(dict.fromkeys(
        [participant_col, trial_col, x_col, mediator_col, outcome_col]
        + [c for c in [quality_col, mediator_observed_col, response_observed_col] if c is not None]
    ))
    quality_rules = {
        "quality_col": quality_col,
        "minimum_quality": minimum_quality,
        "quality_action": quality_action,
        "mediator_observed_col": mediator_observed_col,
        "response_observed_col": response_observed_col,
    }
    preparation_spec = {
        "x_col": x_col,
        "mediator_col": mediator_col,
        "outcome_col": outcome_col,
        "participant_col": participant_col,
        "trial_col": trial_col,
        "grand_mean_center_between": grand_mean_center_between,
        "require_within_x": require_within_x,
        "missingness_policy": "preserve_and_flag",
    }
    provenance = _build_provenance(
        data,
        source_id=source_id,
        source_columns=source_columns,
        preprocessing_spec=preprocessing_spec,
        event_detector=event_detector,
        aoi_specification=aoi_specification,
        quality_rules=quality_rules,
        preparation_spec=preparation_spec,
    )

    unique_notes = tuple(dict.fromkeys(notes))
    if warn:
        for note in unique_notes:
            warnings.warn(note, UserWarning, stacklevel=2)

    return MultilevelMediationData(
        data=decomposed,
        columns={
            "participant": participant_col,
            "trial": trial_col,
            "x": x_col,
            "mediator": mediator_col,
            "outcome": outcome_col,
            "quality": quality_col,
        },
        levels=levels,
        variance=variance,
        missingness=missingness,
        trial_counts=trial_counts,
        validation=validation,
        provenance=provenance,
        warnings=unique_notes,
    )


def mediation_provenance_json(prepared: MultilevelMediationData, *, indent: int = 2) -> str:
    """Serialize preparation provenance for reports or cross-language fixtures."""
    if not isinstance(prepared, MultilevelMediationData):
        raise TypeError("`prepared` must be a MultilevelMediationData object.")
    return json.dumps(_json_safe(prepared.provenance), sort_keys=True, indent=indent)


__all__ = [
    "MultilevelMediationData",
    "prepare_multilevel_mediation_data",
    "validate_multilevel_mediation_data",
    "identify_mediation_levels",
    "decompose_within_between",
    "center_within_participant",
    "summarise_within_between_variance",
    "audit_mediation_missingness",
    "check_mediation_trial_counts",
    "mediation_provenance_json",
]


def add_multilevel_mediation_component(
    prepared: MultilevelMediationData,
    *,
    value_col: str,
    semantic: str,
    within_col: str,
    between_col: str,
    grand_mean_center_between: bool | None = None,
) -> MultilevelMediationData:
    """Add another decomposed trial-level variable to a prepared mediation object.

    This helper supports serial mediators and moderators while keeping all
    within/between decomposition inside eyeprocesspy rather than statistical
    backend packages.
    """
    if not isinstance(prepared, MultilevelMediationData):
        raise TypeError("`prepared` must be a MultilevelMediationData object.")
    for value, name in [
        (value_col, "value_col"), (semantic, "semantic"),
        (within_col, "within_col"), (between_col, "between_col")
    ]:
        _require_nonempty_name(value, name)
    if within_col == between_col:
        raise ValueError("`within_col` and `between_col` must be different.")
    data = prepared.data.copy(deep=True)
    _require_columns(data, [value_col])
    generated = [f"{value_col}_within", f"{value_col}_between", within_col, between_col]
    _check_derived_column_collisions(data, generated)
    if grand_mean_center_between is None:
        grand_mean_center_between = bool(
            prepared.provenance.get("preparation_specification", {}).get(
                "grand_mean_center_between", False
            )
        )
    tmp = decompose_within_between(
        data,
        value_col,
        str(prepared.columns["participant"]),
        grand_mean_center_between=grand_mean_center_between,
    )
    tmp[within_col] = tmp[f"{value_col}_within"]
    tmp[between_col] = tmp[f"{value_col}_between"]
    columns = dict(prepared.columns)
    columns[semantic] = value_col
    columns[f"{semantic}_within"] = within_col
    columns[f"{semantic}_between"] = between_col
    provenance = dict(prepared.provenance)
    additions = list(provenance.get("additional_mediation_components", []))
    additions.append(
        {
            "semantic": semantic,
            "source_column": value_col,
            "within_column": within_col,
            "between_column": between_col,
            "grand_mean_center_between": grand_mean_center_between,
        }
    )
    provenance["additional_mediation_components"] = additions
    return MultilevelMediationData(
        data=tmp,
        columns=columns,
        levels=prepared.levels.copy(),
        variance=prepared.variance.copy(),
        missingness=prepared.missingness.copy(),
        trial_counts=prepared.trial_counts.copy(),
        validation=dict(prepared.validation),
        provenance=provenance,
        warnings=prepared.warnings,
        preparation_version=prepared.preparation_version,
        fit_performed=False,
    )


__all__.append("add_multilevel_mediation_component")
