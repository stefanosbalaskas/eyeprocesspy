"""Research-only geometry-based gaze event reconstruction benchmark.

The input is synchronized reference and predicted gaze in the same declared
angular coordinate system. Not a general fixation detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_reconstruction_audit(
    reference: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    trial_col: str = "trial_id",
    time_col: str = "time_s",
    x_col: str = "x_deg",
    y_col: str = "y_deg",
    event_col: str = "event",
) -> dict:
    """Audit sample angular displacement and reference-event geometry.

    All positions MUST be in degrees on common calibrated axes. Reference
    fixation segments and saccade segments are consecutive runs of event
    labels. This quantifies geometry but does not establish event-classification
    agreement, sensor accuracy or calibration validity.
    """
    keys = [participant_col, trial_col, time_col]
    cols = keys + [x_col, y_col]
    if not set(cols + [event_col]).issubset(reference.columns):
        raise ValueError("reference columns missing")
    if not set(cols).issubset(predicted.columns):
        raise ValueError("prediction columns missing")
    if len(reference) != len(predicted) or len(reference) < 3:
        raise ValueError("matched input lengths and three samples required")
    if not reference[keys].equals(predicted[keys]):
        raise ValueError("reference and predicted keys must match in exact row order")
    # Use positional indexing below, regardless of caller-provided DataFrame labels.
    reference = reference.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)
    if reference[keys].isna().any().any():
        raise ValueError("missing sample identity")
    for frame in (reference, predicted):
        coords = frame[[x_col, y_col]].to_numpy(dtype=float)
        if not np.all(np.isfinite(coords)):
            raise ValueError("angular gaze coordinates must be finite")
    for _, g in reference.groupby([participant_col, trial_col], sort=False):
        t = g[time_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("timestamps must strictly increase within trials")
    if not reference[event_col].isin(["fixation", "saccade", "other"]).all():
        raise ValueError("reference events must be fixation, saccade or other")
    r = reference[[x_col, y_col]].to_numpy(dtype=float)
    p = predicted[[x_col, y_col]].to_numpy(dtype=float)
    displacement = np.linalg.norm(p - r, axis=1)
    events = []
    for key, group in reference.groupby([participant_col, trial_col], sort=False):
        indexes = group.index.to_numpy()
        if not np.array_equal(indexes, np.arange(indexes[0], indexes[0] + len(indexes))):
            raise ValueError("samples must be contiguous within each participant/trial")
        labels = group[event_col].to_numpy()
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        ends = np.r_[starts[1:], len(labels)]
        for start, end in zip(starts, ends):
            label = labels[start]
            if label == "other":
                continue
            positions = indexes[start:end]
            ref = r[positions]
            est = p[positions]
            record = {"participant_id": key[0], "trial_id": key[1],
                      "label": label, "n_samples": int(end - start),
                      "fixation_centroid_error_deg": np.nan,
                      "saccade_amplitude_error_deg": np.nan,
                      "saccade_direction_error_deg": np.nan}
            if label == "fixation":
                record["fixation_centroid_error_deg"] = float(
                    np.linalg.norm(est.mean(axis=0) - ref.mean(axis=0)))
            elif len(positions) >= 2:
                true_vector = ref[-1] - ref[0]
                estimated_vector = est[-1] - est[0]
                true_magnitude = np.linalg.norm(true_vector)
                estimated_magnitude = np.linalg.norm(estimated_vector)
                record["saccade_amplitude_error_deg"] = float(abs(
                    estimated_magnitude - true_magnitude))
                if true_magnitude > 0 and estimated_magnitude > 0:
                    cosine = np.clip(
                        np.dot(true_vector, estimated_vector) /
                        (true_magnitude * estimated_magnitude), -1.0, 1.0)
                    record["saccade_direction_error_deg"] = float(
                        np.degrees(np.arccos(cosine)))
            events.append(record)
    event_table = pd.DataFrame(events)
    return {
        "angular_error_mean_deg": float(displacement.mean()),
        "angular_error_median_deg": float(np.median(displacement)),
        "event_geometry": event_table,
        "claim_boundary": "Matched-reference geometry only; no detector agreement or external validity.",
    }


def confidence_threshold_audit(
    frame: pd.DataFrame, *,
    participant_col: str = "participant_id",
    score_col: str = "vendor_confidence",
    error_col: str = "reference_error_deg",
    confidence_threshold: float = 0.5,
    maximum_acceptable_error_deg: float = 2.0,
) -> pd.DataFrame:
    """Report participant-specific vendor score behavior, not a calibrated probability."""
    required = [participant_col, score_col, error_col]
    if not set(required).issubset(frame.columns):
        raise ValueError("confidence audit requires reference errors")
    if not 0 <= confidence_threshold <= 1 or maximum_acceptable_error_deg <= 0:
        raise ValueError("invalid confidence or error threshold")
    rows = []
    for subject, group in frame.groupby(participant_col, sort=False, dropna=False):
        if pd.isna(subject):
            raise ValueError("participant identity missing")
        score = group[score_col].to_numpy(dtype=float)
        error = group[error_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(score)) or not np.all(np.isfinite(error)):
            raise ValueError("confidence and measured error must be finite")
        if np.any((score < 0) | (score > 1)) or np.any(error < 0):
            raise ValueError("invalid confidence scale or negative reference error")
        selected = score >= confidence_threshold
        rows.append({
            "participant_id": subject,
            "n": len(group),
            "selected_n": int(selected.sum()),
            "fraction_selected": float(selected.mean()),
            "acceptable_fraction_selected": (
                float(np.mean(error[selected] <= maximum_acceptable_error_deg))
                if selected.any() else np.nan),
            "calibration_scope": "within_participant_descriptive_only",
        })
    return pd.DataFrame(rows)
