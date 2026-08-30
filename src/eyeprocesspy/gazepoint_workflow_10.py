"""Integrated Gazepoint downstream workflow for frozen eyeprocess 0.11.1.

Ports the nine ledger exports from R/019-gazepoint-downstream-workflow.R.
The workflow preserves the R package's governance rule that process data may
prepare IRT-ready tables but observed responses/scores are never invented.

Python-specific reproducibility files use JSON/Python rather than pretending
that Python can emit native R RDS objects.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .coordinates import audit_coordinate_spaces
from .dataset import (
    EyeDataset,
    _assert_eye_dataset,
    add_provenance,
    validate_eye_dataset,
)
from .foundation_09 import (
    analysis_readiness,
    assign_trials,
    audit_aois,
    audit_clock_sync,
    audit_missingness,
    audit_pupil_quality,
    audit_sampling_rate,
    audit_signal_quality,
)
from .gazepoint import gp_audit_file_pairs, read_gazepoint_folder
from .io_validation_10 import export_canonical, report_eye_dataset
from .preprocess_features_09 import (
    baseline_pupil,
    derive_gaze_features,
    derive_pupil_features,
    derive_rt_features,
    detect_blinks,
    filter_pupil,
    interpolate_pupil,
    trial_table,
)
from .schema import empty_eye_table, standardize_eye_table

__all__ = [
    "build_gazepoint_media_trials",
    "derive_gazepoint_workflow_features",
    "gazepoint_analysis_tables",
    "gazepoint_irt_tables",
    "gazepoint_workflow_spec",
    "plot_gazepoint_workflow",
    "run_gazepoint_workflow",
    "validate_gazepoint_workflow",
    "write_gazepoint_workflow_report",
]


@dataclass(slots=True)
class GazepointWorkflowSpec:
    expected_sampling_rate: float = 60.0
    sampling_tolerance_hz: float = 5.0
    minimum_valid_gaze: float = 0.80
    minimum_valid_pupil: float = 0.70
    pupil_interpolation: str = "linear"
    pupil_max_gap_ms: float = 150.0
    pupil_filter: str = "median"
    pupil_window: int = 5
    pupil_baseline: str = "none"
    pupil_baseline_window: tuple[float, float] = (0.0, 0.5)
    detect_blinks: bool = True
    biometric_channels: tuple[str, ...] = (
        "eda",
        "skin_conductance_level",
        "skin_conductance_response",
        "heart_rate",
        "interbeat_interval",
        "engagement_dial",
    )
    create_plots: bool = True
    create_html_report: bool = True
    retain_raw: bool = True

    def __repr__(self) -> str:
        return (
            "<eye_gazepoint_workflow_spec "
            f"expected_rate={self.expected_sampling_rate:g}Hz "
            f"pupil_interpolation={self.pupil_interpolation!r} "
            f"pupil_filter={self.pupil_filter!r} "
            f"pupil_baseline={self.pupil_baseline!r}>"
        )


class GazepointWorkflow(dict):
    """Dict/list-style Python counterpart of R ``eye_gazepoint_workflow``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __repr__(self) -> str:
        trials = len(self.get("tables", {}).get("trials", []))
        return (
            "<eye_gazepoint_workflow "
            f"status={self.get('status')!r} trials={trials} "
            f"output_dir={self.get('output_dir')!r}>"
        )


def gazepoint_workflow_spec(
    expected_sampling_rate=60,
    sampling_tolerance_hz=5,
    minimum_valid_gaze=0.80,
    minimum_valid_pupil=0.70,
    pupil_interpolation="linear",
    pupil_max_gap_ms=150,
    pupil_filter="median",
    pupil_window=5,
    pupil_baseline="none",
    pupil_baseline_window=(0, 0.5),
    detect_blinks=True,
    biometric_channels=(
        "eda",
        "skin_conductance_level",
        "skin_conductance_response",
        "heart_rate",
        "interbeat_interval",
        "engagement_dial",
    ),
    create_plots=True,
    create_html_report=True,
    retain_raw=True,
):
    """Create the declarative integrated Gazepoint workflow specification."""
    if pupil_interpolation not in {"linear", "constant", "none"}:
        raise ValueError("Invalid `pupil_interpolation`.")
    if pupil_filter not in {
        "median",
        "mean",
        "moving_median",
        "moving_average",
        "none",
    }:
        raise ValueError("Invalid `pupil_filter`.")
    if pupil_baseline not in {"none", "subtract", "divide", "percent", "zscore"}:
        raise ValueError("Invalid `pupil_baseline`.")
    window = tuple(float(v) for v in pupil_baseline_window)
    if len(window) != 2 or not np.isfinite(window).all() or window[1] < window[0]:
        raise ValueError("`pupil_baseline_window` must contain two ordered finite values.")
    return GazepointWorkflowSpec(
        expected_sampling_rate=float(expected_sampling_rate),
        sampling_tolerance_hz=float(sampling_tolerance_hz),
        minimum_valid_gaze=float(minimum_valid_gaze),
        minimum_valid_pupil=float(minimum_valid_pupil),
        pupil_interpolation=str(pupil_interpolation),
        pupil_max_gap_ms=float(pupil_max_gap_ms),
        pupil_filter=str(pupil_filter),
        pupil_window=int(pupil_window),
        pupil_baseline=str(pupil_baseline),
        pupil_baseline_window=window,
        detect_blinks=bool(detect_blinks),
        biometric_channels=tuple(dict.fromkeys(map(str, biometric_channels))),
        create_plots=bool(create_plots),
        create_html_report=bool(create_html_report),
        retain_raw=bool(retain_raw),
    )


def _workflow_token(value: Any) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "missing"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return re.sub(r"_+", "_", text)


def _first_nonmissing(values: Any, default=pd.NA):
    for value in pd.Series(values).tolist():
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _finite(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )


def _clean_output(path: str | Path, overwrite: bool) -> Path:
    output = Path(path).expanduser()
    if output.exists() and output.is_file():
        raise ValueError(f"Output path exists as a file: {output}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise ValueError(f"Output directory is not empty; use `overwrite=True`: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


def _item_map(item_map: Any, stimuli: Sequence[Any]) -> pd.DataFrame:
    unique_stimuli = [str(value) for value in pd.unique(pd.Series(stimuli).dropna()) if str(value).strip()]
    if item_map is None:
        return pd.DataFrame(
            {
                "stimulus_id": unique_stimuli,
                "item_id": unique_stimuli,
                "condition_id": pd.NA,
            }
        )

    if isinstance(item_map, (str, Path)):
        path = Path(item_map).expanduser()
        if not path.is_file():
            raise ValueError(f"Item map does not exist: {path}")
        item_map = pd.read_csv(path)
    if not isinstance(item_map, pd.DataFrame):
        raise TypeError("`item_map` must be a DataFrame, CSV path, or None.")

    data = item_map.copy()
    missing = {"stimulus_id", "item_id"} - set(data.columns)
    if missing:
        raise ValueError("`item_map` is missing required column(s): " + ", ".join(sorted(missing)))
    if "condition_id" not in data:
        data["condition_id"] = pd.NA
    data["stimulus_id"] = data["stimulus_id"].astype("string")
    data["item_id"] = data["item_id"].astype("string")
    data["condition_id"] = data["condition_id"].astype("string")
    if data["stimulus_id"].duplicated().any():
        raise ValueError("`item_map['stimulus_id']` must be unique.")

    absent = [value for value in unique_stimuli if value not in set(data["stimulus_id"])]
    if absent:
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    {
                        "stimulus_id": absent,
                        "item_id": absent,
                        "condition_id": pd.NA,
                    }
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    return data[["stimulus_id", "item_id", "condition_id"]].reset_index(drop=True)


def build_gazepoint_media_trials(x, item_map=None, overwrite=True):
    """Reconstruct contiguous Gazepoint media presentations as trial intervals."""
    _assert_eye_dataset(x)
    gaze = x["gaze_samples"]
    if gaze.empty or gaze["stimulus_id"].isna().all():
        raise ValueError("Gazepoint gaze samples do not contain media/stimulus identifiers.")

    mapping = _item_map(item_map, gaze["stimulus_id"])
    rows: list[dict[str, Any]] = []
    run_counter: dict[tuple[str, str], int] = {}

    for recording_id in pd.unique(gaze["recording_id"].dropna()):
        z = gaze[gaze["recording_id"].eq(recording_id) & gaze["stimulus_id"].notna()].copy()
        z["timestamp_seconds"] = pd.to_numeric(
            z["timestamp_seconds"],
            errors="coerce",
        )
        z = z[np.isfinite(z["timestamp_seconds"])]
        if z.empty:
            continue
        sort_cols = ["timestamp_seconds"]
        if "sample_id" in z:
            sort_cols.append("sample_id")
        z = z.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        stimulus = z["stimulus_id"].astype("string")
        changed = stimulus.ne(stimulus.shift()).fillna(True)
        run_id = np.cumsum(changed.to_numpy(dtype=bool))

        participant = _first_nonmissing(
            x["recordings"].loc[
                x["recordings"]["recording_id"].eq(recording_id),
                "participant_id",
            ],
            pd.NA,
        )

        for _, group in z.groupby(run_id, sort=False):
            stim = str(group["stimulus_id"].iloc[0])
            key = (str(recording_id), stim)
            run_counter[key] = run_counter.get(key, 0) + 1
            this_run = run_counter[key]
            map_row = mapping[mapping["stimulus_id"].astype(str).eq(stim)]
            item = stim if map_row.empty else map_row["item_id"].iloc[0]
            condition = pd.NA if map_row.empty else map_row["condition_id"].iloc[0]
            trial_id = f"{recording_id}__media_{_workflow_token(stim)}__run_{this_run:02d}"
            rows.append(
                {
                    "interval_id": f"interval_{trial_id}",
                    "recording_id": recording_id,
                    "interval_type": "trial",
                    "start_time": float(group["timestamp_seconds"].min()),
                    "end_time": float(group["timestamp_seconds"].max()),
                    "trial_id": trial_id,
                    "participant_id": participant,
                    "item_id": item,
                    "stimulus_id": stim,
                    "condition_id": condition,
                    "parent_interval_id": pd.NA,
                    "valid_interval": True,
                    "source_method": "Gazepoint contiguous media run",
                }
            )

    if not rows:
        raise ValueError("No Gazepoint media trials could be reconstructed.")

    built = standardize_eye_table(pd.DataFrame(rows), "intervals")
    out = x.copy()
    if overwrite:
        out["intervals"] = out["intervals"][~out["intervals"]["interval_type"].eq("trial")].reset_index(drop=True)
    out["intervals"] = standardize_eye_table(
        pd.concat([out["intervals"], built], ignore_index=True, sort=False),
        "intervals",
    )
    out = assign_trials(out, overwrite=True)
    out.vendor_metadata["gazepoint_item_map"] = mapping.to_dict(orient="records")
    return add_provenance(
        out,
        "build_gazepoint_media_trials",
        "intervals",
        f"{len(built)} media trials; item map rows={len(mapping)}",
    )


def _link_features_to_trials(x: EyeDataset) -> EyeDataset:
    if x["features"].empty:
        return x
    trials = trial_table(x)
    if trials.empty:
        return x

    out = x.copy()
    features = out["features"].copy()
    missing = features["trial_id"].isna() | features["trial_id"].astype("string").str.strip().eq("")

    for index in features.index[missing]:
        stimulus = features.at[index, "stimulus_id"]
        recording = features.at[index, "recording_id"]
        participant = features.at[index, "participant_id"]

        candidate = trials.iloc[0:0]
        if pd.notna(recording) and str(recording).strip():
            candidate = trials[trials["recording_id"].eq(recording) & trials["stimulus_id"].eq(stimulus)]
        if candidate.empty and pd.notna(participant):
            candidate = trials[trials["participant_id"].eq(participant) & trials["stimulus_id"].eq(stimulus)]
        if len(candidate) == 1:
            row = candidate.iloc[0]
            features.at[index, "trial_id"] = row["trial_id"]
            features.at[index, "item_id"] = row["item_id"]
            if pd.isna(recording) or not str(recording).strip():
                features.at[index, "recording_id"] = row["recording_id"]

    out["features"] = standardize_eye_table(features, "features")
    return out


def _prepare_responses(x, responses=None, score_key=None):
    trials = trial_table(x)
    template = trials[
        [
            "recording_id",
            "participant_id",
            "trial_id",
            "item_id",
            "stimulus_id",
            "condition_id",
        ]
    ].copy()
    template["response"] = pd.NA
    template["score"] = np.nan
    template["response_time"] = np.nan
    template["trial_duration_seconds"] = (
        pd.to_numeric(trials["end_time"], errors="coerce") - pd.to_numeric(trials["start_time"], errors="coerce")
    ).to_numpy()

    if responses is None:
        return {
            "dataset": x,
            "response_template": template,
            "supplied": False,
        }

    if isinstance(responses, (str, Path)):
        path = Path(responses).expanduser()
        if not path.is_file():
            raise ValueError(f"Response file does not exist: {path}")
        responses = pd.read_csv(path)
    if not isinstance(responses, pd.DataFrame):
        raise TypeError("`responses` must be a DataFrame, CSV path, or None.")

    data = responses.copy()
    missing = {"participant_id", "item_id"} - set(data.columns)
    if missing:
        raise ValueError("`responses` is missing required column(s): " + ", ".join(sorted(missing)))
    for column, default in (
        ("response", pd.NA),
        ("score", np.nan),
        ("response_time", np.nan),
        ("trial_id", pd.NA),
        ("recording_id", pd.NA),
    ):
        if column not in data:
            data[column] = default

    for index, row in data.iterrows():
        candidate = trials[
            trials["participant_id"].astype(str).eq(str(row["participant_id"]))
            & trials["item_id"].astype(str).eq(str(row["item_id"]))
        ]
        if pd.notna(row["trial_id"]) and str(row["trial_id"]).strip():
            candidate = candidate[candidate["trial_id"].astype(str).eq(str(row["trial_id"]))]
        if len(candidate) != 1:
            raise ValueError(
                "Each supplied response must identify exactly one trial. "
                f"Problem row: {index + 1}. Supply `trial_id` when an item is repeated."
            )
        match = candidate.iloc[0]
        data.at[index, "trial_id"] = match["trial_id"]
        data.at[index, "recording_id"] = match["recording_id"]

    if score_key is not None:
        if isinstance(score_key, pd.Series):
            key = score_key.to_dict()
        elif isinstance(score_key, Mapping):
            key = dict(score_key)
        else:
            raise TypeError("`score_key` must be a named mapping/Series by item id.")
        numeric_score = pd.to_numeric(data["score"], errors="coerce")
        missing_score = ~np.isfinite(numeric_score.to_numpy(dtype=float))
        for index in data.index[missing_score]:
            item = str(data.at[index, "item_id"])
            if item in key and pd.notna(data.at[index, "response"]):
                data.at[index, "score"] = float(str(data.at[index, "response"]) == str(key[item]))

    canonical = pd.DataFrame(
        {
            "response_id": [f"response_{_workflow_token(value)}" for value in data["trial_id"]],
            "recording_id": data["recording_id"],
            "participant_id": data["participant_id"],
            "trial_id": data["trial_id"],
            "item_id": data["item_id"],
            "response": data["response"].astype("string"),
            "score": pd.to_numeric(data["score"], errors="coerce"),
            "response_time": pd.to_numeric(data["response_time"], errors="coerce"),
            "response_timestamp": np.nan,
            "response_type": "observed",
            "valid_response": True,
        }
    )
    out = x.copy()
    out["responses"] = standardize_eye_table(canonical, "responses")
    out = add_provenance(
        out,
        "add_gazepoint_workflow_responses",
        "responses",
        f"{len(canonical)} observed response row(s)",
    )

    observed = canonical[
        ["recording_id", "participant_id", "trial_id", "item_id", "response", "score", "response_time"]
    ]
    merged = template.drop(columns=["response", "score", "response_time"]).merge(
        observed,
        on=["recording_id", "participant_id", "trial_id", "item_id"],
        how="left",
        sort=False,
    )
    return {
        "dataset": out,
        "response_template": merged,
        "supplied": True,
    }


def _prepare_biometrics(x):
    if x["biometrics"].empty:
        return x
    out = x.copy()
    data = out["biometrics"].copy()
    values = pd.to_numeric(data["value"], errors="coerce")
    valid = data["valid"].fillna(False).astype(bool) & np.isfinite(values.to_numpy(dtype=float))
    data["analysis_value"] = values.where(valid, np.nan)
    data["analysis_valid"] = valid
    out["biometrics"] = data
    return add_provenance(
        out,
        "prepare_workflow_biometrics",
        "biometrics",
        f"{int(valid.sum())} valid observations retained in analysis_value; raw values preserved",
    )


def _preprocess_pupil(x, spec: GazepointWorkflowSpec):
    if x["eye_samples"].empty:
        return x
    out = x
    if spec.detect_blinks:
        out = detect_blinks(out, source="validity", overwrite=True)
    if spec.pupil_interpolation != "none":
        out = interpolate_pupil(
            out,
            method=spec.pupil_interpolation,
            max_gap_ms=spec.pupil_max_gap_ms,
        )
    if spec.pupil_filter != "none":
        out = filter_pupil(
            out,
            method=spec.pupil_filter,
            window=spec.pupil_window,
        )
    if spec.pupil_baseline != "none":
        out = baseline_pupil(
            out,
            method=spec.pupil_baseline,
            baseline_window=spec.pupil_baseline_window,
            anchor="trial_start",
        )
        out = add_provenance(
            out,
            "workflow_pupil_baseline_notice",
            "eye_samples",
            "Baseline window is relative to media onset and is not necessarily pre-stimulus.",
            warnings=("Do not interpret media-onset baselines as equivalent to a true pre-stimulus baseline."),
        )
    return out


def _feature_rows_for_trial(x, trial: pd.Series) -> list[dict[str, Any]]:
    recording_id = trial["recording_id"]
    trial_id = trial["trial_id"]
    gaze = x["gaze_samples"][
        x["gaze_samples"]["recording_id"].eq(recording_id) & x["gaze_samples"]["trial_id"].eq(trial_id)
    ]
    fixation = x["episodes"][
        x["episodes"]["recording_id"].eq(recording_id)
        & x["episodes"]["trial_id"].eq(trial_id)
        & x["episodes"]["episode_type"].eq("fixation")
    ]
    pupil = x["eye_samples"][
        x["eye_samples"]["recording_id"].eq(recording_id) & x["eye_samples"]["trial_id"].eq(trial_id)
    ]

    gaze_x = pd.to_numeric(gaze["gaze_x"], errors="coerce")
    gaze_y = pd.to_numeric(gaze["gaze_y"], errors="coerce")
    gaze_ok = (
        (
            gaze["valid"].fillna(False).astype(bool)
            & np.isfinite(gaze_x.to_numpy(dtype=float))
            & np.isfinite(gaze_y.to_numpy(dtype=float))
        )
        if len(gaze)
        else np.array([], dtype=bool)
    )

    pupil_value = pd.to_numeric(pupil["pupil_diameter"], errors="coerce")
    pupil_ok = (
        (pupil["pupil_valid"].fillna(False).astype(bool) & np.isfinite(pupil_value.to_numpy(dtype=float)))
        if len(pupil)
        else np.array([], dtype=bool)
    )

    values = {
        "trial_duration_seconds": (float(trial["end_time"]) - float(trial["start_time"])),
        "gaze_sample_count": float(len(gaze)),
        "gaze_valid_fraction": (float(np.mean(gaze_ok)) if len(gaze_ok) else np.nan),
        "fixation_count_vendor": float(
            (fixation["derived_by"].astype("string").eq("vendor") if len(fixation) else pd.Series(dtype=bool)).sum()
        ),
        "pupil_observation_count": float(len(pupil)),
        "pupil_valid_fraction": (float(np.mean(pupil_ok)) if len(pupil_ok) else np.nan),
    }
    units = {
        "trial_duration_seconds": "seconds",
        "gaze_sample_count": "count",
        "gaze_valid_fraction": "proportion",
        "fixation_count_vendor": "count",
        "pupil_observation_count": "count",
        "pupil_valid_fraction": "proportion",
    }

    rows = []
    for feature_name, value in values.items():
        rows.append(
            {
                "feature_id": (f"feature_{_workflow_token(recording_id)}_{_workflow_token(trial_id)}_{feature_name}"),
                "recording_id": recording_id,
                "participant_id": trial["participant_id"],
                "trial_id": trial_id,
                "item_id": trial["item_id"],
                "stimulus_id": trial["stimulus_id"],
                "aoi_id": pd.NA,
                "feature_name": feature_name,
                "value": value,
                "unit": units[feature_name],
                "level": "trial",
                "window_start": trial["start_time"],
                "window_end": trial["end_time"],
                "observed_fraction": np.nan,
                "method": "eyeprocess Gazepoint downstream workflow",
                "parameters": pd.NA,
                "derived_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )
    return rows


def _workflow_trial_features(x) -> pd.DataFrame:
    trials = trial_table(x)
    if trials.empty:
        return empty_eye_table("features")
    rows: list[dict[str, Any]] = []
    for _, trial in trials.iterrows():
        rows.extend(_feature_rows_for_trial(x, trial))
    return standardize_eye_table(pd.DataFrame(rows), "features")


def _workflow_biometric_features(x) -> pd.DataFrame:
    data = x["biometrics"]
    if data.empty:
        return empty_eye_table("features")
    data = data[data["trial_id"].notna() & data["trial_id"].astype("string").str.strip().ne("")].copy()
    if data.empty:
        return empty_eye_table("features")

    value_col = "analysis_value" if "analysis_value" in data else "value"
    rows: list[dict[str, Any]] = []
    keys = ["recording_id", "trial_id", "channel"]
    for _, group in data.groupby(keys, sort=False, dropna=False):
        values = pd.to_numeric(group[value_col], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        base = {
            "recording_id": group["recording_id"].iloc[0],
            "participant_id": pd.NA,
            "trial_id": group["trial_id"].iloc[0],
            "item_id": pd.NA,
            "stimulus_id": _first_nonmissing(group["stimulus_id"], pd.NA),
            "aoi_id": pd.NA,
            "level": "trial",
            "window_start": np.nan,
            "window_end": np.nan,
            "observed_fraction": (float(len(finite) / len(group)) if len(group) else np.nan),
            "method": "eyeprocess Gazepoint downstream workflow",
            "parameters": f"channel={group['channel'].iloc[0]}",
            "derived_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        channel = _workflow_token(group["channel"].iloc[0])
        stats = {
            f"{channel}_mean": (
                float(finite.mean()) if len(finite) else np.nan,
                _first_nonmissing(group["unit"], pd.NA),
            ),
            f"{channel}_sd": (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                _first_nonmissing(group["unit"], pd.NA),
            ),
            f"{channel}_valid_fraction": (
                float(len(finite) / len(group)) if len(group) else np.nan,
                "proportion",
            ),
        }
        for name, (value, unit) in stats.items():
            rows.append(
                {
                    **base,
                    "feature_id": (
                        f"feature_{_workflow_token(base['recording_id'])}_{_workflow_token(base['trial_id'])}_{name}"
                    ),
                    "feature_name": name,
                    "value": value,
                    "unit": unit,
                }
            )

    features = standardize_eye_table(pd.DataFrame(rows), "features")
    if features.empty:
        return features
    trials = trial_table(x)[["recording_id", "trial_id", "participant_id", "item_id", "stimulus_id"]]
    lookup = trials.set_index(["recording_id", "trial_id"])
    for index, row in features.iterrows():
        key = (row["recording_id"], row["trial_id"])
        if key in lookup.index:
            trial = lookup.loc[key]
            if isinstance(trial, pd.DataFrame):
                trial = trial.iloc[0]
            features.at[index, "participant_id"] = trial["participant_id"]
            features.at[index, "item_id"] = trial["item_id"]
            features.at[index, "stimulus_id"] = trial["stimulus_id"]
    return standardize_eye_table(features, "features")


def derive_gazepoint_workflow_features(x, reset_workflow_features=True):
    """Derive the frozen R workflow's trial/AOI/process feature collection."""
    _assert_eye_dataset(x)
    if trial_table(x).empty:
        raise ValueError("Media/trial intervals are required.")

    out = x.copy()
    if reset_workflow_features and not out["features"].empty:
        methods = out["features"]["method"].astype("string")
        workflow = methods.str.contains(
            r"^(derive_gaze_features|derive_pupil_features|derive_rt_features|"
            r"eyeprocess Gazepoint downstream workflow)",
            regex=True,
            na=False,
        )
        out["features"] = out["features"].loc[~workflow].reset_index(drop=True)

    out = _link_features_to_trials(out)
    fixations = out["episodes"][out["episodes"]["episode_type"].eq("fixation")]
    if not fixations.empty:
        out = derive_gaze_features(
            out,
            level="trial",
            source="fixations",
            append=True,
        )
        has_aoi = fixations["aoi_id"].notna() & fixations["aoi_id"].astype("string").str.strip().ne("")
        if has_aoi.any():
            out = derive_gaze_features(
                out,
                level="trial_aoi",
                source="fixations",
                append=True,
            )
    else:
        out = derive_gaze_features(
            out,
            level="trial",
            source="samples",
            append=True,
        )

    if not out["eye_samples"].empty:
        out = derive_pupil_features(out, level="trial", append=True)
    if not out["responses"].empty:
        out = derive_rt_features(out, append=True)

    extra = pd.concat(
        [_workflow_trial_features(out), _workflow_biometric_features(out)],
        ignore_index=True,
        sort=False,
    )
    if not extra.empty:
        out["features"] = standardize_eye_table(
            pd.concat([out["features"], extra], ignore_index=True, sort=False),
            "features",
        )
    out = _link_features_to_trials(out)
    return add_provenance(
        out,
        "derive_gazepoint_workflow_features",
        "features",
        f"total feature rows={len(out['features'])}",
    )


def _wide_features(features: pd.DataFrame, id_cols: Sequence[str]) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    ids = [column for column in id_cols if column in features]
    if not ids:
        return pd.DataFrame()

    data = features[ids + ["feature_name", "value"]].copy()
    data = data[data["feature_name"].notna()]
    if data.empty:
        return pd.DataFrame(columns=ids)

    # Frozen R semantics: retain exactly the observed id combinations, then
    # attach heterogeneous feature columns one at a time. This avoids the
    # Cartesian expansion that pandas pivot_table(dropna=False) can produce.
    result = data[ids].drop_duplicates().reset_index(drop=True)
    for feature_name in pd.unique(data["feature_name"]):
        subset = data[data["feature_name"].eq(feature_name)].copy()
        subset = subset.drop_duplicates(subset=ids, keep="last")
        subset = subset[ids + ["value"]].rename(columns={"value": str(feature_name)})
        result = result.merge(subset, on=ids, how="left", sort=False)
    return result


def _summarize_fixations(
    x,
    by=("recording_id", "trial_id"),
    source="vendor",
) -> pd.DataFrame:
    data = x["episodes"]
    if data.empty:
        return pd.DataFrame()
    data = data[data["episode_type"].eq("fixation")].copy()
    if source != "all":
        data = data[data["derived_by"].astype("string").eq(source)]
    keys = [column for column in by if column in data]
    if data.empty or not keys:
        return pd.DataFrame()

    rows = []
    for values, group in data.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        duration = pd.to_numeric(group["duration_ms"], errors="coerce")
        finite = duration[np.isfinite(duration.to_numpy(dtype=float))]
        row = dict(zip(keys, values, strict=False))
        row.update(
            fixation_count=int(len(group)),
            fixation_duration_total_ms=(float(finite.sum()) if len(finite) else np.nan),
            fixation_duration_mean_ms=(float(finite.mean()) if len(finite) else np.nan),
            fixation_duration_sd_ms=(float(finite.std(ddof=1)) if len(finite) > 1 else np.nan),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _pupil_summary(x, trial_base: pd.DataFrame) -> pd.DataFrame:
    data = x["eye_samples"]
    if data.empty:
        return pd.DataFrame()
    data = data[data["trial_id"].notna() & data["trial_id"].astype("string").str.strip().ne("")]
    if data.empty:
        return pd.DataFrame()

    rows = []
    for _, group in data.groupby(
        ["recording_id", "trial_id", "eye"],
        sort=False,
        dropna=False,
    ):
        pupil = pd.to_numeric(group["pupil_diameter"], errors="coerce")
        valid = group["pupil_valid"].fillna(False).astype(bool) & np.isfinite(pupil.to_numpy(dtype=float))
        values = pupil[valid]
        if "interpolated" in group:
            interpolated = group["interpolated"].fillna(False).astype(bool)
            interp_fraction = float(interpolated.mean())
        else:
            interp_fraction = 0.0
        rows.append(
            {
                "recording_id": group["recording_id"].iloc[0],
                "trial_id": group["trial_id"].iloc[0],
                "eye": group["eye"].iloc[0],
                "pupil_unit": _first_nonmissing(group["pupil_unit"], pd.NA),
                "n_observations": len(group),
                "valid_fraction": float(valid.mean()),
                "interpolated_fraction": interp_fraction,
                "pupil_mean": float(values.mean()) if len(values) else np.nan,
                "pupil_sd": (float(values.std(ddof=1)) if len(values) > 1 else np.nan),
                "pupil_min": float(values.min()) if len(values) else np.nan,
                "pupil_max": float(values.max()) if len(values) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.merge(
        trial_base[
            [
                "recording_id",
                "participant_id",
                "trial_id",
                "item_id",
                "stimulus_id",
            ]
        ],
        on=["recording_id", "trial_id"],
        how="left",
        sort=False,
    )


def _biometric_summary(x, trial_base: pd.DataFrame) -> pd.DataFrame:
    data = x["biometrics"]
    if data.empty:
        return pd.DataFrame()
    data = data[data["trial_id"].notna() & data["trial_id"].astype("string").str.strip().ne("")]
    if data.empty:
        return pd.DataFrame()

    value_col = "analysis_value" if "analysis_value" in data else "value"
    rows = []
    for _, group in data.groupby(
        ["recording_id", "trial_id", "channel"],
        sort=False,
        dropna=False,
    ):
        values = pd.to_numeric(group[value_col], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        rows.append(
            {
                "recording_id": group["recording_id"].iloc[0],
                "trial_id": group["trial_id"].iloc[0],
                "channel": group["channel"].iloc[0],
                "unit": _first_nonmissing(group["unit"], pd.NA),
                "n_observations": len(group),
                "valid_fraction": (float(len(finite) / len(group)) if len(group) else np.nan),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "sd": (float(finite.std(ddof=1)) if len(finite) > 1 else np.nan),
                "minimum": float(finite.min()) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.merge(
        trial_base[
            [
                "recording_id",
                "participant_id",
                "trial_id",
                "item_id",
                "stimulus_id",
            ]
        ],
        on=["recording_id", "trial_id"],
        how="left",
        sort=False,
    )


def _feature_dictionary(x) -> pd.DataFrame:
    features = x["features"]
    if features.empty:
        return pd.DataFrame(columns=["feature_name", "unit", "level", "method", "n_rows"])
    rows = []
    for feature_name, group in features.groupby(
        "feature_name",
        sort=False,
        dropna=False,
    ):
        rows.append(
            {
                "feature_name": feature_name,
                "unit": _first_nonmissing(group["unit"], pd.NA),
                "level": _first_nonmissing(group["level"], pd.NA),
                "method": _first_nonmissing(group["method"], pd.NA),
                "n_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def gazepoint_analysis_tables(x):
    """Build person-by-item-by-trial analysis tables."""
    _assert_eye_dataset(x)
    trials = trial_table(x)
    if trials.empty:
        raise ValueError("No trial intervals are available.")

    trial_base = trials[
        [
            "recording_id",
            "participant_id",
            "trial_id",
            "item_id",
            "stimulus_id",
            "condition_id",
            "start_time",
            "end_time",
        ]
    ].copy()
    trial_base["trial_duration_seconds"] = pd.to_numeric(trial_base["end_time"], errors="coerce") - pd.to_numeric(
        trial_base["start_time"], errors="coerce"
    )

    features = x["features"]
    no_aoi = features["aoi_id"].isna() | features["aoi_id"].astype("string").str.strip().eq("")
    trial_features = features[features["trial_id"].notna() & no_aoi]
    wide = _wide_features(
        trial_features,
        [
            "recording_id",
            "participant_id",
            "trial_id",
            "item_id",
            "stimulus_id",
        ],
    )
    if not wide.empty:
        process = trial_base.merge(
            wide,
            on=[
                "recording_id",
                "participant_id",
                "trial_id",
                "item_id",
                "stimulus_id",
            ],
            how="left",
            sort=False,
        )
    else:
        process = trial_base.copy()

    fixations = _summarize_fixations(
        x,
        by=("recording_id", "trial_id"),
        source="vendor",
    )
    if not fixations.empty:
        fixations = fixations.merge(
            trial_base[
                [
                    "recording_id",
                    "participant_id",
                    "trial_id",
                    "item_id",
                    "stimulus_id",
                ]
            ],
            on=["recording_id", "trial_id"],
            how="left",
            sort=False,
        )

    aoi_fixations = _summarize_fixations(
        x,
        by=("recording_id", "trial_id", "aoi_id"),
        source="vendor",
    )
    if not aoi_fixations.empty:
        aoi_fixations = aoi_fixations.merge(
            trial_base[
                [
                    "recording_id",
                    "participant_id",
                    "trial_id",
                    "item_id",
                    "stimulus_id",
                ]
            ],
            on=["recording_id", "trial_id"],
            how="left",
            sort=False,
        )

    # Preserve the 2026-08 R fix: an empty AOI feature table is a valid case
    # and must not fail the entire downstream analysis-table construction.
    has_aoi = features["aoi_id"].notna() & features["aoi_id"].astype("string").str.strip().ne("")
    aoi_features = features[features["trial_id"].notna() & has_aoi]
    aoi_summary = _wide_features(
        aoi_features,
        [
            "recording_id",
            "participant_id",
            "trial_id",
            "item_id",
            "stimulus_id",
            "aoi_id",
        ],
    )
    if not aoi_summary.empty and not x["aoi_definitions"].empty:
        definitions = x["aoi_definitions"][["aoi_id", "aoi_name"]].drop_duplicates()
        aoi_summary = aoi_summary.merge(
            definitions,
            on="aoi_id",
            how="left",
            sort=False,
        )

    return {
        "trials": trial_base.reset_index(drop=True),
        "process": process.reset_index(drop=True),
        "fixation_summary": fixations.reset_index(drop=True),
        "aoi_fixation_summary": aoi_fixations.reset_index(drop=True),
        "aoi_summary": aoi_summary.reset_index(drop=True),
        "pupil_summary": _pupil_summary(x, trial_base).reset_index(drop=True),
        "biometric_summary": _biometric_summary(x, trial_base).reset_index(drop=True),
        "feature_dictionary": _feature_dictionary(x).reset_index(drop=True),
    }


def _response_matrix(responses: pd.DataFrame, value: str) -> pd.DataFrame | None:
    if responses.empty or value not in responses:
        return None
    data = responses.copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data[
        data["participant_id"].notna() & data["item_id"].notna() & np.isfinite(data[value].to_numpy(dtype=float))
    ]
    if data.empty:
        return None
    data = data.drop_duplicates(
        subset=["participant_id", "item_id"],
        keep="last",
    )
    return data.pivot(
        index="participant_id",
        columns="item_id",
        values=value,
    )


def gazepoint_irt_tables(x, process_table=None):
    """Create response/process tables without fitting an IRT model."""
    _assert_eye_dataset(x)
    if process_table is None:
        process_table = gazepoint_analysis_tables(x)["process"]
    trials = trial_table(x)

    response_template = trials[
        [
            "recording_id",
            "participant_id",
            "trial_id",
            "item_id",
            "stimulus_id",
            "condition_id",
        ]
    ].copy()
    response_template["response"] = pd.NA
    response_template["score"] = np.nan
    response_template["response_time"] = np.nan

    if not x["responses"].empty:
        observed = x["responses"][
            [
                "recording_id",
                "participant_id",
                "trial_id",
                "item_id",
                "response",
                "score",
                "response_time",
            ]
        ]
        response_template = response_template.drop(columns=["response", "score", "response_time"]).merge(
            observed,
            on=["recording_id", "participant_id", "trial_id", "item_id"],
            how="left",
            sort=False,
        )

    irt_long = process_table.merge(
        response_template[
            ["recording_id", "participant_id", "trial_id", "item_id", "response", "score", "response_time"]
        ],
        on=["recording_id", "participant_id", "trial_id", "item_id"],
        how="left",
        sort=False,
    )

    n_persons = trials["participant_id"].dropna().astype(str).nunique()
    n_items = trials["item_id"].dropna().astype(str).nunique()
    n_trials = len(trials)
    observed_response = (
        response_template["response"].notna() & response_template["response"].astype("string").str.strip().ne("")
    ).sum()
    score = pd.to_numeric(response_template["score"], errors="coerce")
    observed_score = int(np.isfinite(score.to_numpy(dtype=float)).sum())
    numeric_process = sum(
        pd.api.types.is_numeric_dtype(process_table[column])
        for column in process_table.columns
        if column
        not in {
            "start_time",
            "end_time",
            "trial_duration_seconds",
        }
    )

    if observed_score == 0:
        status = "process_ready_response_pending"
    elif n_persons < 100 or n_items < 5:
        status = "structurally_ready_validation_only"
    else:
        status = "model_ready_subject_to_diagnostics"

    readiness = pd.DataFrame(
        {
            "metric": [
                "participants",
                "items",
                "trials",
                "observed_responses",
                "observed_scores",
                "numeric_process_covariates",
            ],
            "value": [
                n_persons,
                n_items,
                n_trials,
                int(observed_response),
                observed_score,
                numeric_process,
            ],
            "status": [
                "available" if n_persons else "missing",
                "available" if n_items else "missing",
                "available" if n_trials else "missing",
                "available" if observed_response else "pending",
                (
                    "complete"
                    if observed_score == n_trials and n_trials
                    else "incomplete"
                    if observed_score
                    else "pending"
                ),
                "available",
            ],
        }
    )

    response_matrix = _response_matrix(x["responses"], "score") if observed_score else None
    rt = pd.to_numeric(x["responses"]["response_time"], errors="coerce")
    response_time_matrix = (
        _response_matrix(x["responses"], "response_time")
        if len(rt) and (np.isfinite(rt.to_numpy(dtype=float)) & (rt > 0)).any()
        else None
    )

    return {
        "status": status,
        "readiness": readiness,
        "response_template": response_template.reset_index(drop=True),
        "process_covariates": process_table.reset_index(drop=True),
        "irt_long": irt_long.reset_index(drop=True),
        "response_matrix": response_matrix,
        "response_time_matrix": response_time_matrix,
        "guidance": [
            "No IRT model is fitted automatically.",
            ("Provide observed responses and defensible scoring before estimating ability or item parameters."),
            ("Use grouped, person-aware validation and prespecified process covariates in substantive studies."),
        ],
    }


def _save_workflow_plot(path: Path, draw) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    try:
        fig = draw()
        if fig is None:
            fig = plt.gcf()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return {
            "plot": path.name,
            "path": str(path.resolve()),
            "status": "created",
            "message": "",
        }
    except Exception as exc:
        plt.close("all")
        return {
            "plot": path.name,
            "path": str(path.resolve()),
            "status": "failed",
            "message": str(exc),
        }


def plot_gazepoint_workflow(x, directory, channels=None, expected_hz=60):
    """Create the workflow plot suite and return a plot manifest."""
    _assert_eye_dataset(x)
    import matplotlib.pyplot as plt

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    def add(name: str, draw, subdir="summary"):
        manifest.append(_save_workflow_plot(root / subdir / name, draw))

    def overview():
        fig, ax = plt.subplots()
        names = ["gaze", "eye", "fixations", "biometrics", "features"]
        values = [
            len(x["gaze_samples"]),
            len(x["eye_samples"]),
            int(x["episodes"]["episode_type"].eq("fixation").sum()),
            len(x["biometrics"]),
            len(x["features"]),
        ]
        ax.bar(names, values)
        ax.set_ylabel("Rows")
        ax.set_title("Gazepoint workflow dataset overview")
        ax.tick_params(axis="x", rotation=30)
        return fig

    add("dataset-overview.png", overview)

    def sampling():
        fig, ax = plt.subplots()
        t = pd.to_numeric(x["gaze_samples"]["timestamp_seconds"], errors="coerce")
        t = np.sort(t[np.isfinite(t)])
        if len(t) > 1:
            hz = 1.0 / np.diff(t)
            hz = hz[np.isfinite(hz) & (hz > 0)]
            ax.hist(hz, bins=min(30, max(5, len(hz))))
        ax.axvline(float(expected_hz), linestyle="--")
        ax.set_xlabel("Observed instantaneous Hz")
        ax.set_title("Sampling-rate evidence")
        return fig

    add("sampling-rate.png", sampling)

    def quality():
        fig, ax = plt.subplots()
        gaze = x["gaze_samples"]
        eye = x["eye_samples"]
        gaze_valid = float(gaze["valid"].fillna(False).astype(bool).mean()) if len(gaze) else np.nan
        pupil_valid = float(eye["pupil_valid"].fillna(False).astype(bool).mean()) if len(eye) else np.nan
        ax.bar(["gaze", "pupil"], [gaze_valid, pupil_valid])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Valid fraction")
        ax.set_title("Signal quality")
        return fig

    add("signal-quality.png", quality)

    trials = trial_table(x)
    if not trials.empty:

        def timeline():
            fig, ax = plt.subplots()
            for index, row in trials.reset_index(drop=True).iterrows():
                start = float(row["start_time"])
                width = float(row["end_time"]) - start
                ax.barh(index, width, left=start)
            ax.set_yticks(range(len(trials)))
            ax.set_yticklabels(trials["trial_id"].astype(str).tolist())
            ax.set_xlabel("Seconds")
            ax.set_title("Trial timeline")
            return fig

        add("trial-timeline.png", timeline)

        for _, trial in trials.iterrows():
            recording_id = trial["recording_id"]
            trial_id = trial["trial_id"]
            stem = f"{_workflow_token(recording_id)}__{_workflow_token(trial_id)}"
            gaze = x["gaze_samples"][
                x["gaze_samples"]["recording_id"].eq(recording_id) & x["gaze_samples"]["trial_id"].eq(trial_id)
            ]
            fixation = x["episodes"][
                x["episodes"]["recording_id"].eq(recording_id)
                & x["episodes"]["trial_id"].eq(trial_id)
                & x["episodes"]["episode_type"].eq("fixation")
            ]
            pupil = x["eye_samples"][
                x["eye_samples"]["recording_id"].eq(recording_id) & x["eye_samples"]["trial_id"].eq(trial_id)
            ]

            if not gaze.empty:

                def gaze_trace(data=gaze, title=trial_id):
                    fig, ax = plt.subplots()
                    ax.plot(
                        pd.to_numeric(data["gaze_x"], errors="coerce"),
                        pd.to_numeric(data["gaze_y"], errors="coerce"),
                        marker="o",
                    )
                    ax.invert_yaxis()
                    ax.set_title(f"Gaze trace: {title}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    return fig

                add(f"{stem}__gaze-trace.png", gaze_trace, "trials")

            if not fixation.empty:

                def fixation_plot(data=fixation, title=trial_id):
                    fig, ax = plt.subplots()
                    ax.scatter(
                        pd.to_numeric(data["centroid_x"], errors="coerce"),
                        pd.to_numeric(data["centroid_y"], errors="coerce"),
                    )
                    ax.invert_yaxis()
                    ax.set_title(f"Vendor fixations: {title}")
                    return fig

                add(f"{stem}__fixations.png", fixation_plot, "trials")

            if not pupil.empty:

                def pupil_plot(data=pupil, title=trial_id):
                    fig, ax = plt.subplots()
                    for eye_name, group in data.groupby(
                        "eye",
                        sort=False,
                        dropna=False,
                    ):
                        ax.plot(
                            pd.to_numeric(
                                group["timestamp_seconds"],
                                errors="coerce",
                            ),
                            pd.to_numeric(
                                group["pupil_diameter"],
                                errors="coerce",
                            ),
                            label=str(eye_name),
                        )
                    ax.set_title(f"Pupil: {title}")
                    ax.set_xlabel("Seconds")
                    ax.legend()
                    return fig

                add(f"{stem}__pupil.png", pupil_plot, "trials")

    selected = set(channels or [])
    bio = x["biometrics"]
    if not bio.empty:
        for channel, group in bio.groupby("channel", sort=False, dropna=False):
            if selected and str(channel) not in selected:
                continue

            def bio_plot(data=group, name=channel):
                fig, ax = plt.subplots()
                values = data["analysis_value"] if "analysis_value" in data else data["value"]
                ax.plot(
                    pd.to_numeric(data["timestamp_seconds"], errors="coerce"),
                    pd.to_numeric(values, errors="coerce"),
                )
                ax.set_title(f"Biometric channel: {name}")
                ax.set_xlabel("Seconds")
                return fig

            add(
                f"biometric-{_workflow_token(channel)}.png",
                bio_plot,
                "biometrics",
            )

    return pd.DataFrame(
        manifest,
        columns=["plot", "path", "status", "message"],
    )


def _markdown_table(data: pd.DataFrame, max_rows=30) -> str:
    if not isinstance(data, pd.DataFrame) or data.empty:
        return "_No rows._"
    table = data.head(max_rows).copy()
    for column in table:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else str(value).replace("|", r"\|"))
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule, *rows])


def write_gazepoint_workflow_report(
    workflow,
    path=None,
    render_html=None,
):
    """Write the reproducible workflow Markdown report and optional HTML copy."""
    if not isinstance(workflow, GazepointWorkflow):
        raise TypeError("`workflow` must be returned by `run_gazepoint_workflow()`.")
    if path is None:
        path = Path(workflow.output_dir) / "gazepoint-workflow-report.md"
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if render_html is None:
        render_html = workflow.spec.create_html_report

    trials = workflow.tables["trials"]
    lines = [
        "# Gazepoint downstream workflow report",
        "",
        f"- Status: **{str(workflow.status).upper()}**",
        f"- Source: `{workflow.source_path}`",
        f"- Recordings: {len(workflow.dataset['recordings'])}",
        f"- Trials: {len(trials)}",
        f"- Responses supplied: {workflow.responses_supplied}",
        f"- IRT readiness: `{workflow.irt['status']}`",
        "",
        "## Workflow validation",
        "",
        _markdown_table(workflow.workflow_checks),
        "",
        "## IRT readiness",
        "",
        _markdown_table(workflow.irt["readiness"]),
        "",
        "## Trial/process table",
        "",
        _markdown_table(workflow.tables["process"]),
        "",
        "## Fixation and AOI summaries",
        "",
        (f"- Vendor trial-level fixation rows: {len(workflow.tables['fixation_summary'])}"),
        (f"- Vendor AOI-fixation rows: {len(workflow.tables['aoi_fixation_summary'])}"),
        f"- Combined AOI summary rows: {len(workflow.tables['aoi_summary'])}",
        "",
        "## Pupil summary",
        "",
        _markdown_table(workflow.tables["pupil_summary"]),
        "",
        "## Biometric summary",
        "",
        _markdown_table(workflow.tables["biometric_summary"]),
        "",
        "## Governance",
        "",
        "- No IRT model is fitted automatically.",
        ("- Observed responses and defensible scoring are required before estimating ability or item parameters."),
        ("- Process covariates should use grouped, person-aware validation and prespecification."),
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if render_html:
        html_path = destination.with_suffix(".html")
        body = escape(destination.read_text(encoding="utf-8"))
        html_path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Gazepoint downstream workflow report</title></head>"
            f"<body><pre>{body}</pre></body></html>",
            encoding="utf-8",
        )
        workflow.paths["report_html"] = str(html_path.resolve())

    return str(destination.resolve())


def _write_csv(data: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, lineterminator="\n")
    return str(path.resolve())


def _write_tables(tables, irt, qc, directory: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, data in tables.items():
        if isinstance(data, pd.DataFrame):
            paths[f"table_{name}"] = _write_csv(
                data,
                directory / "tables" / f"{name.replace('_', '-')}.csv",
            )
    for name, data in qc.items():
        if isinstance(data, pd.DataFrame):
            paths[f"qc_{name}"] = _write_csv(
                data,
                directory / "qc" / f"{name.replace('_', '-')}.csv",
            )
    for key, filename in (
        ("readiness", "irt-readiness.csv"),
        ("response_template", "response-template.csv"),
        ("process_covariates", "process-covariates.csv"),
        ("irt_long", "irt-long.csv"),
    ):
        paths[f"irt_{key}"] = _write_csv(
            irt[key],
            directory / "irt" / filename,
        )
    if irt["response_matrix"] is not None:
        paths["irt_response_matrix"] = _write_csv(
            irt["response_matrix"].reset_index(),
            directory / "irt" / "response-matrix.csv",
        )
    if irt["response_time_matrix"] is not None:
        paths["irt_response_time_matrix"] = _write_csv(
            irt["response_time_matrix"].reset_index(),
            directory / "irt" / "response-time-matrix.csv",
        )
    return paths


def _qc_tables(x, source_path: Path, spec: GazepointWorkflowSpec):
    return {
        "file_pairs": gp_audit_file_pairs(source_path),
        "validation": validate_eye_dataset(x),
        "readiness": analysis_readiness(x),
        "sampling_rate": audit_sampling_rate(
            x,
            expected_hz=spec.expected_sampling_rate,
            tolerance_hz=spec.sampling_tolerance_hz,
        ),
        "signal_quality": audit_signal_quality(
            x,
            minimum_valid_gaze=spec.minimum_valid_gaze,
            minimum_valid_pupil=spec.minimum_valid_pupil,
        ),
        "pupil_quality": audit_pupil_quality(x),
        "clock_sync": audit_clock_sync(x),
        "coordinate_spaces": audit_coordinate_spaces(x),
        "aoi": audit_aois(x),
        "gaze_missingness": audit_missingness(x, "gaze_samples"),
        "pupil_missingness": audit_missingness(x, "eye_samples"),
        "biometric_missingness": audit_missingness(x, "biometrics"),
    }


def _reproducibility_files(workflow: GazepointWorkflow) -> None:
    root = Path(workflow.output_dir)
    spec_path = root / "workflow-spec.json"
    spec_path.write_text(
        json.dumps(asdict(workflow.spec), indent=2),
        encoding="utf-8",
    )
    workflow.paths["workflow_spec"] = str(spec_path.resolve())

    summary = {
        "status": workflow.status,
        "source_path": workflow.source_path,
        "output_dir": workflow.output_dir,
        "responses_supplied": workflow.responses_supplied,
        "irt_status": workflow.irt["status"],
        "n_recordings": len(workflow.dataset["recordings"]),
        "n_trials": len(workflow.tables["trials"]),
        "n_features": len(workflow.dataset["features"]),
        "note": (
            "Python reproducibility metadata is JSON. Native R RDS files are "
            "not impersonated by another serialization format."
        ),
    }
    result_path = root / "workflow-result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    workflow.paths["workflow_result"] = str(result_path.resolve())

    item_map = workflow.item_map
    if isinstance(item_map, pd.DataFrame):
        workflow.paths["item_map"] = _write_csv(
            item_map,
            root / "item-map.csv",
        )
    if workflow.responses_supplied and not workflow.dataset["responses"].empty:
        workflow.paths["responses_supplied"] = _write_csv(
            workflow.dataset["responses"],
            root / "responses-supplied.csv",
        )

    py = root / "rerun-workflow.py"
    py.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import eyeprocesspy as ep\n\n"
        "root = Path(__file__).resolve().parent\n"
        "spec_data = json.loads((root / 'workflow-spec.json').read_text())\n"
        "spec_data['pupil_baseline_window'] = "
        "tuple(spec_data['pupil_baseline_window'])\n"
        "spec = ep.gazepoint_workflow_spec(**spec_data)\n"
        "item_map = root / 'item-map.csv'\n"
        "responses = root / 'responses-supplied.csv'\n"
        f"source_path = {workflow.source_path!r}\n"
        "ep.run_gazepoint_workflow(\n"
        "    source_path,\n"
        "    output_dir=root,\n"
        "    item_map=item_map if item_map.exists() else None,\n"
        "    responses=responses if responses.exists() else None,\n"
        "    spec=spec,\n"
        "    overwrite=True,\n"
        "    quiet=False,\n"
        ")\n",
        encoding="utf-8",
    )
    workflow.paths["rerun_python"] = str(py.resolve())

    # Retain a transparent frozen-R reproduction script as documentation.
    # It is text only; no fake RDS files are created.
    rerun_r = root / "rerun-workflow.R"
    rerun_r.write_text(
        "library(eyeprocess)\n\n"
        f"source_path <- {json.dumps(workflow.source_path)}\n"
        f"output_dir <- {json.dumps(workflow.output_dir)}\n"
        "# Recreate the equivalent R workflow with the frozen eyeprocess 0.11.1 "
        "package.\n"
        "run_gazepoint_workflow(source_path, output_dir=output_dir, "
        "overwrite=TRUE)\n",
        encoding="utf-8",
    )
    workflow.paths["rerun_r_reference"] = str(rerun_r.resolve())


def validate_gazepoint_workflow(x):
    """Validate structural and reproducibility invariants of workflow output."""
    if not isinstance(x, GazepointWorkflow):
        raise TypeError("`x` must be returned by `run_gazepoint_workflow()`.")

    trials = x.tables["trials"]
    process = x.tables["process"]
    trial_keys = set(
        zip(
            trials["recording_id"].astype(str),
            trials["trial_id"].astype(str),
            strict=False,
        )
    )
    process_keys = set(
        zip(
            process["recording_id"].astype(str),
            process["trial_id"].astype(str),
            strict=False,
        )
    )

    required = [
        Path(x.paths.get("canonical_dataset", "")),
        Path(x.paths.get("report", "")),
        Path(x.paths.get("workflow_result", "")),
        Path(x.paths.get("workflow_spec", "")),
        Path(x.paths.get("rerun_r_reference", "")),
        Path(x.paths.get("rerun_python", "")),
        Path(x.output_dir) / "irt" / "response-template.csv",
        Path(x.output_dir) / "tables" / "process.csv",
    ]
    checks = [
        (
            "status_pass",
            x.status == "pass",
            f"Workflow status is {x.status!r}.",
        ),
        (
            "trial_keys_unique",
            len(trial_keys) == len(trials),
            "Trial recording/trial keys must be unique.",
        ),
        (
            "process_matches_trials",
            trial_keys == process_keys,
            "Process table must contain exactly one row per reconstructed trial.",
        ),
        (
            "irt_rows_match_trials",
            len(x.irt["response_template"]) == len(trials),
            "IRT response template must contain one row per trial.",
        ),
        (
            "required_outputs_exist",
            all(path.exists() for path in required),
            "Canonical/report/reproducibility/table outputs must exist.",
        ),
    ]
    return pd.DataFrame(
        [{"check": name, "passed": bool(passed), "message": message} for name, passed, message in checks]
    )


def run_gazepoint_workflow(
    path,
    output_dir=None,
    responses=None,
    score_key=None,
    item_map=None,
    spec=None,
    overwrite=False,
    quiet=False,
):
    """Run the complete frozen-R Gazepoint downstream workflow."""
    if spec is None:
        spec = gazepoint_workflow_spec()
    if not isinstance(spec, GazepointWorkflowSpec):
        raise TypeError("`spec` must be created with `gazepoint_workflow_spec()`.")

    source = Path(path).expanduser()
    if not source.is_dir():
        raise ValueError(f"Gazepoint source directory does not exist: {source}")
    source = source.resolve()
    if output_dir is None:
        output_dir = Path.cwd() / "eyeprocess-gazepoint-workflow"
    output = _clean_output(output_dir, overwrite)

    x = read_gazepoint_folder(
        source,
        include=("gaze", "fixations", "events", "biometrics", "aoi"),
        keep_raw=spec.retain_raw,
        quiet=quiet,
    )
    resolved_item_map = _item_map(item_map, x["gaze_samples"]["stimulus_id"])
    x = build_gazepoint_media_trials(
        x,
        item_map=resolved_item_map,
        overwrite=True,
    )

    response_result = _prepare_responses(x, responses, score_key)
    x = response_result["dataset"]
    x = _preprocess_pupil(x, spec)
    x = _prepare_biometrics(x)
    x = derive_gazepoint_workflow_features(
        x,
        reset_workflow_features=True,
    )

    qc = _qc_tables(x, source, spec)
    validation = qc["validation"]
    validation_errors = int(validation["severity"].eq("error").sum()) if not validation.empty else 0
    status = "fail" if validation_errors else "pass"

    tables = gazepoint_analysis_tables(x)
    irt = gazepoint_irt_tables(x, tables["process"])

    paths = _write_tables(tables, irt, qc, output)
    canonical_dir = output / "canonical-dataset"
    paths["canonical_dataset"] = export_canonical(
        x,
        canonical_dir,
        include_raw=spec.retain_raw,
        overwrite=True,
        manifest=True,
    )
    paths["canonical_report"] = report_eye_dataset(
        x,
        path=output / "canonical-validation-report.md",
        title="eyeprocess canonical Gazepoint validation report",
    )

    plot_manifest = (
        plot_gazepoint_workflow(
            x,
            output / "plots",
            channels=spec.biometric_channels,
            expected_hz=spec.expected_sampling_rate,
        )
        if spec.create_plots
        else pd.DataFrame(columns=["plot", "path", "status", "message"])
    )
    if not plot_manifest.empty:
        paths["plot_manifest"] = _write_csv(
            plot_manifest,
            output / "plots" / "plot-manifest.csv",
        )

    result = GazepointWorkflow(
        status=status,
        source_path=str(source),
        output_dir=str(output),
        spec=spec,
        dataset=x,
        item_map=resolved_item_map,
        responses_supplied=bool(response_result["supplied"]),
        response_template=response_result["response_template"],
        qc=qc,
        tables=tables,
        irt=irt,
        plot_manifest=plot_manifest,
        paths=paths,
        workflow_checks=pd.DataFrame(),
    )

    _reproducibility_files(result)
    # Report is generated after reproducibility paths exist so validation can
    # cover the complete output set.
    report_path = output / "gazepoint-workflow-report.md"
    result.paths["report"] = str(report_path.resolve())
    result.workflow_checks = validate_gazepoint_workflow(result)
    write_gazepoint_workflow_report(
        result,
        path=report_path,
        render_html=spec.create_html_report,
    )
    result.workflow_checks = validate_gazepoint_workflow(result)
    return result
