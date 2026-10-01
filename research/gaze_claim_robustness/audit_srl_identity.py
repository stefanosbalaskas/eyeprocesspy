#!/usr/bin/env python3
"""Reconcile SRL participant identifiers across released data sources.

This audit does not estimate any gaze effect. It distinguishes the full
recruitment metadata from the released event/raw eye-tracking cohorts and
records identifier mismatches without silently repairing them.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _read(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", nrows=nrows)


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _unique(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name}; found {len(matches)}.")
    return matches[0]


def _raw_index(root: Path) -> pd.DataFrame:
    raw_dirs = sorted(path for path in root.rglob("ET_data_raw") if path.is_dir())
    if len(raw_dirs) != 1:
        raise ValueError(f"Expected one ET_data_raw directory; found {len(raw_dirs)}.")
    rows = []
    pattern = re.compile(r"ET_data_raw_(.+)\.txt$")
    for path in sorted(raw_dirs[0].glob("*.txt")):
        match = pattern.fullmatch(path.name)
        filename_id = match.group(1) if match else ""
        head = _read(path, nrows=250)
        embedded = []
        if "Participant" in head:
            embedded = sorted(
                {_clean_id(value) for value in head["Participant"] if _clean_id(value)}
            )
        rows.append(
            {
                "file": path.name,
                "filename_participant_id": filename_id,
                "embedded_participant_ids": "|".join(embedded),
                "n_embedded_participant_ids": len(embedded),
                "filename_matches_embedded": (
                    len(embedded) == 1 and embedded[0] == filename_id
                ),
            }
        )
    return pd.DataFrame(rows)


def _event_index(root: Path) -> pd.DataFrame:
    event_dirs = sorted(path for path in root.rglob("ET_event_data") if path.is_dir())
    if len(event_dirs) != 1:
        raise ValueError(
            f"Expected one ET_event_data directory; found {len(event_dirs)}."
        )
    pattern = re.compile(r"Event_data_(.+)_Task_([1-8])\.csv$")
    rows = []
    for path in sorted(event_dirs[0].glob("Event_data_*_Task_*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            rows.append(
                {
                    "file": path.name,
                    "participant_id": "",
                    "task": "",
                    "filename_parsed": False,
                }
            )
            continue
        rows.append(
            {
                "file": path.name,
                "participant_id": match.group(1),
                "task": f"Task_{match.group(2)}",
                "filename_parsed": True,
            }
        )
    return pd.DataFrame(rows)


def audit_identity(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = _read(_unique(root, "participants.csv"))
    stimuli = _read(_unique(root, "stimuli.csv"))
    if "part_ID" not in participants or "experiment_condition" not in participants:
        raise ValueError("participants.csv lacks part_ID/experiment_condition.")
    if "part_ID" not in stimuli or "stimulus_name" not in stimuli:
        raise ValueError("stimuli.csv lacks part_ID/stimulus_name.")

    p = participants.copy()
    p["participant_id"] = p["part_ID"].map(_clean_id)
    s = stimuli.copy()
    s["participant_id"] = s["part_ID"].map(_clean_id)
    s["stimulus_clean"] = s["stimulus_name"].map(
        lambda value: Path(str(value).strip()).stem if not pd.isna(value) else ""
    )

    metadata_ids = set(p["participant_id"]) - {""}
    stimulus_ids = set(s["participant_id"]) - {""}
    raw = _raw_index(root)
    raw_ids = set(raw["filename_participant_id"]) - {""}
    events = _event_index(root)
    event_counts = (
        events.loc[events["filename_parsed"]]
        .groupby("participant_id")["task"]
        .nunique()
        .rename("n_event_tasks")
    )
    event_ids = set(event_counts.index)

    condition_map = (
        p.drop_duplicates("participant_id")
        .set_index("participant_id")["experiment_condition"]
        .astype(str)
        .to_dict()
    )
    stimulus_task_counts = (
        s.groupby("participant_id")["stimulus_clean"].nunique().to_dict()
    )

    union = sorted(
        metadata_ids | stimulus_ids | raw_ids | event_ids,
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )
    presence_rows = []
    for participant_id in union:
        presence_rows.append(
            {
                "participant_id": participant_id,
                "in_participants_csv": participant_id in metadata_ids,
                "in_stimuli_csv": participant_id in stimulus_ids,
                "raw_file_available": participant_id in raw_ids,
                "event_files_available": participant_id in event_ids,
                "n_event_tasks": int(event_counts.get(participant_id, 0)),
                "n_stimulus_rows_or_levels": int(
                    stimulus_task_counts.get(participant_id, 0)
                ),
                "experiment_condition": condition_map.get(participant_id, ""),
                "exact_cross_source_raw_eligible": (
                    participant_id in metadata_ids
                    and participant_id in stimulus_ids
                    and participant_id in raw_ids
                    and int(stimulus_task_counts.get(participant_id, 0)) >= 8
                ),
                "exact_cross_source_event_complete": (
                    participant_id in metadata_ids
                    and participant_id in stimulus_ids
                    and participant_id in event_ids
                    and int(event_counts.get(participant_id, 0)) == 8
                ),
            }
        )
    presence = pd.DataFrame(presence_rows)

    metadata_only = sorted(metadata_ids - (raw_ids | event_ids))
    unmatched_rows = []
    for source, ids in (
        ("raw", sorted(raw_ids - metadata_ids)),
        ("event", sorted(event_ids - metadata_ids)),
    ):
        for source_id in ids:
            candidates = []
            if source_id.isdigit():
                number = int(source_id)
                numeric_meta = [x for x in metadata_only if x.isdigit()]
                candidates = sorted(
                    (abs(number - int(candidate)), candidate)
                    for candidate in numeric_meta
                )[:5]
            unmatched_rows.append(
                {
                    "source": source,
                    "unmatched_id": source_id,
                    "nearest_metadata_only_ids": "|".join(
                        f"{candidate}(delta={delta})"
                        for delta, candidate in candidates
                    ),
                    "status": "diagnostic_only_no_mapping_inferred",
                }
            )
    unmatched = pd.DataFrame(
        unmatched_rows,
        columns=[
            "source",
            "unmatched_id",
            "nearest_metadata_only_ids",
            "status",
        ],
    )

    exact_raw = presence.loc[presence["exact_cross_source_raw_eligible"]]
    exact_event = presence.loc[presence["exact_cross_source_event_complete"]]
    summary_rows = [
        ("participants_csv_ids", len(metadata_ids)),
        ("stimuli_csv_ids", len(stimulus_ids)),
        ("raw_file_ids", len(raw_ids)),
        ("event_ids", len(event_ids)),
        ("event_ids_with_8_tasks", int((event_counts == 8).sum())),
        ("exact_raw_metadata_ids", len(exact_raw)),
        ("exact_event_metadata_ids", len(exact_event)),
        (
            "exact_raw_prompt",
            int(exact_raw["experiment_condition"].eq("Prompt").sum()),
        ),
        (
            "exact_raw_non_prompt",
            int(exact_raw["experiment_condition"].eq("Non-prompt").sum()),
        ),
        ("raw_ids_absent_participants_csv", len(raw_ids - metadata_ids)),
        ("event_ids_absent_participants_csv", len(event_ids - metadata_ids)),
        ("event_ids_without_raw_file", len(event_ids - raw_ids)),
        ("raw_ids_without_event_files", len(raw_ids - event_ids)),
        (
            "raw_filename_embedded_id_mismatches",
            int((~raw["filename_matches_embedded"]).sum()),
        ),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    return presence, unmatched, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_identity_audit"),
    )
    args = parser.parse_args()

    presence, unmatched, summary = audit_identity(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    presence.to_csv(args.output_dir / "id_source_presence.csv", index=False)
    unmatched.to_csv(args.output_dir / "unmatched_id_candidates.csv", index=False)
    summary.to_csv(args.output_dir / "identity_summary.csv", index=False)

    print(summary.to_string(index=False))
    if len(unmatched):
        print("\nUnmatched IDs (no repair inferred):")
        print(unmatched.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
