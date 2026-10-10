#!/usr/bin/env python3
"""Audit the Emoji Keyboard Gaze Search archives before scientific analysis.

This script is research-only. It inventories the Zenodo raw/processed ZIPs,
compares task coverage across task type/layout cells, and refuses to infer that
processed files imply corresponding raw sample availability.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path

RAW_RE = re.compile(
    r"^(?P<task_type>Scenario-based Selection|Target Search)/"
    r"(?P<layout>android|ios)/(?P<participant>\d+)/extracted_data\.xlsx$"
)
SCROLL_RE = re.compile(
    r"^(?P<task_type>Scenario-based Selection|Target Search)/"
    r"(?P<layout>android|ios)-scroll/(?P<participant>\d+)/(?P<task>[1-6])\.txt$"
)
PROCESSED_RE = re.compile(
    r"^(?P<task_type>Scenario-based Selection|Target Search)_"
    r"(?P<layout>android|ios)_p(?P<participant>\d+)_task(?P<task>[1-6])\.csv$"
)


def _members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return [name.lstrip("./") for name in archive.namelist() if not name.endswith("/")]


def audit(raw_zip: Path, processed_zip: Path) -> tuple[list[dict[str, object]], list[str]]:
    raw_members = _members(raw_zip)
    processed_members = _members(processed_zip)

    raw_workbooks: set[tuple[str, str, str]] = set()
    scroll_tasks: set[tuple[str, str, str, int]] = set()
    processed_tasks: set[tuple[str, str, str, int]] = set()
    unmatched_raw: list[str] = []

    for name in raw_members:
        match = RAW_RE.match(name)
        if match:
            raw_workbooks.add(
                (match["task_type"], match["layout"], match["participant"])
            )
            continue
        match = SCROLL_RE.match(name)
        if match:
            scroll_tasks.add(
                (
                    match["task_type"],
                    match["layout"],
                    match["participant"],
                    int(match["task"]),
                )
            )
            continue
        unmatched_raw.append(name)

    for name in processed_members:
        match = PROCESSED_RE.match(Path(name).name)
        if match:
            processed_tasks.add(
                (
                    match["task_type"],
                    match["layout"],
                    match["participant"],
                    int(match["task"]),
                )
            )

    cells = [
        ("Scenario-based Selection", "android"),
        ("Scenario-based Selection", "ios"),
        ("Target Search", "android"),
        ("Target Search", "ios"),
    ]
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for task_type, layout in cells:
        raw_participants = sorted(
            p for task_name, layout_name, p in raw_workbooks if task_name == task_type and layout_name == layout
        )
        scroll_participants = sorted(
            {p for task_name, layout_name, p, _ in scroll_tasks if task_name == task_type and layout_name == layout}
        )
        processed_participants = sorted(
            {p for task_name, layout_name, p, _ in processed_tasks if task_name == task_type and layout_name == layout}
        )
        processed_count = sum(
            1 for task_name, layout_name, _, _ in processed_tasks if task_name == task_type and layout_name == layout
        )
        scroll_count = sum(
            1 for task_name, layout_name, _, _ in scroll_tasks if task_name == task_type and layout_name == layout
        )
        missing_processed = []
        for participant in processed_participants:
            present = {
                task
                for task_name, layout_name, p, task in processed_tasks
                if task_name == task_type and layout_name == layout and p == participant
            }
            for task in range(1, 7):
                if task not in present:
                    missing_processed.append(f"p{participant}:task{task}")

        rows.append(
            {
                "task_type": task_type,
                "layout": layout,
                "raw_workbook_participants": len(raw_participants),
                "scroll_participants": len(scroll_participants),
                "processed_participants": len(processed_participants),
                "scroll_task_files": scroll_count,
                "processed_task_files": processed_count,
                "raw_participant_ids": ",".join(raw_participants),
                "processed_participant_ids": ",".join(processed_participants),
                "missing_processed_tasks": ",".join(missing_processed),
            }
        )

        if processed_participants and not raw_participants:
            warnings.append(
                f"{task_type}/{layout}: processed task files exist but no raw "
                "extracted_data.xlsx participant directory is present."
            )
        raw_set = set(raw_participants)
        processed_set = set(processed_participants)
        if raw_set != processed_set:
            warnings.append(
                f"{task_type}/{layout}: raw/processed participant sets differ; "
                f"raw-only={sorted(raw_set - processed_set)}, "
                f"processed-only={sorted(processed_set - raw_set)}."
            )

    shortcut_like = sorted(
        name for name in unmatched_raw if name.lower().endswith((".lnk", ".url"))
    )
    if shortcut_like:
        warnings.append(
            "Raw archive contains shortcut/link files that are not scientific data: "
            + ", ".join(shortcut_like)
        )

    return rows, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_zip", type=Path)
    parser.add_argument("processed_zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("emoji_dataset_inventory.csv"))
    args = parser.parse_args()

    rows, warnings = audit(args.raw_zip, args.processed_zip)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"{row['task_type']} / {row['layout']}: "
            f"raw participants={row['raw_workbook_participants']}, "
            f"processed participants={row['processed_participants']}, "
            f"processed tasks={row['processed_task_files']}"
        )
    if warnings:
        print("\nFeasibility warnings:")
        for warning in warnings:
            print(f"- {warning}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
