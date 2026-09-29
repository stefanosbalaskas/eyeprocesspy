#!/usr/bin/env python3
"""Build the prespecified SRL participant × stimulus × eye design manifest.

The manifest is structural evidence only. It does not calculate gaze outcomes
or fit the focal Text-versus-Multimedia contrast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PARTICIPANT_REQUIRED = {"part_ID", "experiment_condition"}
STIMULI_REQUIRED = {
    "part_ID",
    "stimulus_name",
    "stimulus_type",
    "stimulus_time",
    "tracking_ratio",
}
RAW_ID_REQUIRED = {"Participant"}
TASKS = tuple(f"Task_{i}" for i in range(1, 9))
EYES = ("left", "right")


def _read_table(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    return pd.read_csv(path, sep=None, engine="python", nrows=nrows)


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}.")


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _stem(value: object) -> str:
    text = _clean(value)
    return Path(text).stem if text else ""


def _raw_files(raw_dir: Path) -> list[Path]:
    allowed = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def _raw_file_index(raw_dir: Path) -> tuple[dict[str, Path], list[dict[str, object]]]:
    index: dict[str, Path] = {}
    issues: list[dict[str, object]] = []

    for path in _raw_files(raw_dir):
        try:
            head = _read_table(path, nrows=250)
            _require(head, RAW_ID_REQUIRED, str(path))
            participants = sorted(
                {_clean(value) for value in head["Participant"] if _clean(value)}
            )
            if len(participants) != 1:
                issues.append(
                    {
                        "severity": "error",
                        "scope": "raw_file",
                        "participant_id": "",
                        "stimulus_name": "",
                        "message": (
                            f"{path.name}: expected one Participant value in sample; "
                            f"found {participants}."
                        ),
                    }
                )
                continue
            participant = participants[0]
            if participant in index:
                issues.append(
                    {
                        "severity": "error",
                        "scope": "raw_file",
                        "participant_id": participant,
                        "stimulus_name": "",
                        "message": (
                            "Multiple raw files map to the same participant: "
                            f"{index[participant].name}, {path.name}."
                        ),
                    }
                )
                continue
            index[participant] = path
        except Exception as exc:
            issues.append(
                {
                    "severity": "error",
                    "scope": "raw_file",
                    "participant_id": "",
                    "stimulus_name": "",
                    "message": f"{path.name}: {type(exc).__name__}: {exc}",
                }
            )
    return index, issues


def build_manifest(
    participants_csv: Path,
    stimuli_csv: Path,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = _read_table(participants_csv)
    stimuli = _read_table(stimuli_csv)
    _require(participants, PARTICIPANT_REQUIRED, str(participants_csv))
    _require(stimuli, STIMULI_REQUIRED, str(stimuli_csv))

    participants = participants.copy()
    participants["participant_id"] = participants["part_ID"].map(_clean)
    stimuli = stimuli.copy()
    stimuli["participant_id"] = stimuli["part_ID"].map(_clean)
    stimuli["stimulus"] = stimuli["stimulus_name"].map(_stem)

    issues: list[dict[str, object]] = []
    raw_index, raw_issues = _raw_file_index(raw_dir)
    issues.extend(raw_issues)

    duplicate_participants = participants[
        participants["participant_id"].duplicated(keep=False)
    ]
    for participant in sorted(duplicate_participants["participant_id"].unique()):
        issues.append(
            {
                "severity": "error",
                "scope": "participants",
                "participant_id": participant,
                "stimulus_name": "",
                "message": "participants.csv contains duplicate participant rows.",
            }
        )

    participant_ids = sorted(
        participant
        for participant in participants["participant_id"].unique()
        if participant
    )
    if len(participant_ids) != 84:
        issues.append(
            {
                "severity": "warning",
                "scope": "design",
                "participant_id": "",
                "stimulus_name": "",
                "message": (
                    "The descriptor reports 84 complete recordings; "
                    f"participants.csv contains {len(participant_ids)} unique IDs. "
                    "Verify whether the release includes additional/excluded records."
                ),
            }
        )

    rows: list[dict[str, object]] = []
    participant_lookup = participants.set_index("participant_id", drop=False)

    for participant in participant_ids:
        if participant not in participant_lookup.index:
            continue
        p_row = participant_lookup.loc[participant]
        if isinstance(p_row, pd.DataFrame):
            continue

        condition = _clean(p_row["experiment_condition"])
        if condition not in {"Prompt", "Non-prompt"}:
            issues.append(
                {
                    "severity": "error",
                    "scope": "participants",
                    "participant_id": participant,
                    "stimulus_name": "",
                    "message": f"Unexpected experiment_condition={condition!r}.",
                }
            )

        p_stimuli = stimuli.loc[stimuli["participant_id"].eq(participant)].copy()
        observed_tasks = set(p_stimuli["stimulus"])
        missing_tasks = sorted(set(TASKS) - observed_tasks)
        extra_tasks = sorted(observed_tasks - set(TASKS))
        if missing_tasks or extra_tasks:
            issues.append(
                {
                    "severity": "error",
                    "scope": "stimuli",
                    "participant_id": participant,
                    "stimulus_name": "",
                    "message": (
                        f"Expected Task_1..Task_8; missing={missing_tasks}; "
                        f"extra={extra_tasks}."
                    ),
                }
            )

        type_counts = p_stimuli.groupby("stimulus_type")["stimulus"].nunique().to_dict()
        if type_counts.get("Text", 0) != 4 or type_counts.get("Multimedia", 0) != 4:
            issues.append(
                {
                    "severity": "error",
                    "scope": "stimuli",
                    "participant_id": participant,
                    "stimulus_name": "",
                    "message": (
                        "Expected four Text and four Multimedia tasks; "
                        f"found {type_counts}."
                    ),
                }
            )

        raw_path = raw_index.get(participant)
        if raw_path is None:
            issues.append(
                {
                    "severity": "error",
                    "scope": "raw_file",
                    "participant_id": participant,
                    "stimulus_name": "",
                    "message": "No unique raw participant file was identified.",
                }
            )

        for task in TASKS:
            task_rows = p_stimuli.loc[p_stimuli["stimulus"].eq(task)]
            if len(task_rows) != 1:
                issues.append(
                    {
                        "severity": "error",
                        "scope": "stimuli",
                        "participant_id": participant,
                        "stimulus_name": task,
                        "message": (
                            f"Expected one stimuli.csv row; found {len(task_rows)}."
                        ),
                    }
                )
                continue
            task_row = task_rows.iloc[0]
            stimulus_type = _clean(task_row["stimulus_type"])
            for eye in EYES:
                rows.append(
                    {
                        "participant_id": participant,
                        "experiment_condition": condition,
                        "stimulus_name": task,
                        "stimulus_type": stimulus_type,
                        "eye": eye,
                        "design_cell": f"{condition}|{stimulus_type}",
                        "raw_file": (
                            raw_path.relative_to(raw_dir).as_posix()
                            if raw_path is not None
                            else pd.NA
                        ),
                        "metadata_stimulus_time_seconds": task_row["stimulus_time"],
                        "metadata_tracking_ratio_percent": task_row["tracking_ratio"],
                        "nominal_sampling_rate_hz": 250.0,
                        "screen_width_px": 1600,
                        "screen_height_px": 900,
                        "primary_estimand_family": "between_aoi_transition_rate",
                        "focal_factor": "experiment_condition",
                        "focal_contrast": "Prompt_minus_Non-prompt",
                        "outcome_frozen": True,
                        "event_detector_frozen": False,
                        "aoi_geometry_frozen": False,
                        "quality_rule_frozen": False,
                        "model_frozen": False,
                    }
                )

    manifest = pd.DataFrame(rows)
    issue_frame = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "scope",
            "participant_id",
            "stimulus_name",
            "message",
        ],
    )
    return manifest, issue_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("participants_csv", type=Path)
    parser.add_argument("stimuli_csv", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_design_manifest"),
    )
    args = parser.parse_args()

    manifest, issues = build_manifest(
        args.participants_csv,
        args.stimuli_csv,
        args.raw_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "srl_trial_eye_manifest.csv", index=False)
    issues.to_csv(args.output_dir / "srl_design_issues.csv", index=False)

    print(
        f"Planned participant × stimulus × eye branches: {len(manifest)}"
    )
    if not manifest.empty:
        print(
            manifest.groupby(
                ["experiment_condition", "stimulus_type", "eye"],
                dropna=False,
            ).size().rename("branches").reset_index().to_string(index=False)
        )

    if len(issues):
        print("\nDesign audit issues:")
        print(issues.to_string(index=False))
    if not issues.empty and issues["severity"].eq("error").any():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
