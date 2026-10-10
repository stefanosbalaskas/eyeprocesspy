#!/usr/bin/env python3
"""Audit identifiability of SRL Prompt and material-type contrasts.

This audit uses design metadata only. It does not read gaze outcomes.

The central question is whether stimulus modality varies within a stimulus/topic
across participants. If every Task_n has only one modality, material type is
nested in task identity and a pure modality main effect is not separately
identifiable from task/content effects once task identity is controlled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PARTICIPANT_REQUIRED = {"part_ID", "experiment_condition"}
STIMULI_REQUIRED = {"part_ID", "stimulus_name", "stimulus_type"}
VALID_CONDITIONS = {"Prompt", "Non-prompt"}
VALID_TYPES = {"Text", "Multimedia"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _stem(value: object) -> str:
    text = _clean(value)
    return Path(text).stem if text else ""


def _require(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}.")


def audit_identifiability(
    participants: pd.DataFrame,
    stimuli: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return task structure, estimand status, and design issues."""
    _require(participants, PARTICIPANT_REQUIRED, "participants")
    _require(stimuli, STIMULI_REQUIRED, "stimuli")

    p = participants.copy()
    p["participant_id"] = p["part_ID"].map(_clean)
    p["experiment_condition"] = p["experiment_condition"].map(_clean)

    s = stimuli.copy()
    s["participant_id"] = s["part_ID"].map(_clean)
    s["stimulus_name_clean"] = s["stimulus_name"].map(_stem)
    s["stimulus_type_clean"] = s["stimulus_type"].map(_clean)

    issues: list[dict[str, object]] = []

    duplicate_p = p["participant_id"].duplicated(keep=False)
    if duplicate_p.any():
        for participant in sorted(p.loc[duplicate_p, "participant_id"].unique()):
            issues.append(
                {
                    "severity": "error",
                    "scope": "participants",
                    "message": f"Duplicate participant row: {participant}.",
                }
            )

    unknown_conditions = sorted(
        set(p["experiment_condition"]) - VALID_CONDITIONS - {""}
    )
    if unknown_conditions:
        issues.append(
            {
                "severity": "error",
                "scope": "participants",
                "message": f"Unknown experiment conditions: {unknown_conditions}.",
            }
        )

    unknown_types = sorted(
        set(s["stimulus_type_clean"]) - VALID_TYPES - {""}
    )
    if unknown_types:
        issues.append(
            {
                "severity": "error",
                "scope": "stimuli",
                "message": f"Unknown stimulus types: {unknown_types}.",
            }
        )

    condition_map = p.set_index("participant_id")["experiment_condition"]
    s["experiment_condition"] = s["participant_id"].map(condition_map)

    task_rows: list[dict[str, object]] = []
    for task, z in s.groupby("stimulus_name_clean", sort=True, dropna=False):
        types = sorted(set(z["stimulus_type_clean"]) - {""})
        conditions = sorted(set(z["experiment_condition"].dropna()) - {""})
        task_rows.append(
            {
                "stimulus_name": task,
                "participants": z["participant_id"].nunique(),
                "stimulus_type_levels": "|".join(types),
                "n_stimulus_type_levels": len(types),
                "condition_levels": "|".join(conditions),
                "n_condition_levels": len(conditions),
                "modality_varies_within_task": len(types) > 1,
                "prompt_varies_within_task": len(conditions) > 1,
            }
        )
    task_structure = pd.DataFrame(task_rows)

    nonempty_tasks = task_structure.loc[task_structure["stimulus_name"].ne("")]
    modality_crossed = bool(
        len(nonempty_tasks)
        and nonempty_tasks["modality_varies_within_task"].all()
    )
    modality_nested = bool(
        len(nonempty_tasks)
        and nonempty_tasks["n_stimulus_type_levels"].eq(1).all()
    )
    prompt_crossed = bool(
        len(nonempty_tasks)
        and nonempty_tasks["prompt_varies_within_task"].all()
    )

    estimands = pd.DataFrame(
        [
            {
                "estimand": "Prompt_vs_Non-prompt",
                "design_status": (
                    "randomized_between_participants_and_crossed_with_tasks"
                    if prompt_crossed
                    else "requires_design_review"
                ),
                "primary_eligible": prompt_crossed,
                "interpretation": (
                    "Prompt is the randomized between-participant manipulation; "
                    "task identity can be controlled without absorbing the prompt contrast."
                ),
            },
            {
                "estimand": "Multimedia_vs_Text",
                "design_status": (
                    "crossed_with_task_identity"
                    if modality_crossed
                    else "nested_in_task_identity"
                    if modality_nested
                    else "mixed_or_incomplete_mapping"
                ),
                "primary_eligible": modality_crossed,
                "interpretation": (
                    "A pure modality effect is separately identifiable from task/content "
                    "only if modality varies within task identity across participants. "
                    "If modality is fixed by Task_n, the contrast is a difference between "
                    "the released text-task and multimedia-task sets, not an isolated "
                    "causal format effect."
                ),
            },
        ]
    )

    if modality_nested:
        issues.append(
            {
                "severity": "warning",
                "scope": "identifiability",
                "message": (
                    "Stimulus modality is nested in task identity. Do not describe "
                    "Multimedia-vs-Text as a pure causal format effect after controlling "
                    "for Task_n; the modality main effect is aliased with task/content."
                ),
            }
        )
    if not prompt_crossed:
        issues.append(
            {
                "severity": "error",
                "scope": "identifiability",
                "message": (
                    "Prompt and Non-prompt are not both represented within every task. "
                    "The planned primary prompt contrast requires review."
                ),
            }
        )

    issue_frame = pd.DataFrame(
        issues,
        columns=["severity", "scope", "message"],
    )
    return task_structure, estimands, issue_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("participants_csv", type=Path)
    parser.add_argument("stimuli_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_identifiability_audit"),
    )
    args = parser.parse_args()

    task_structure, estimands, issues = audit_identifiability(
        _read(args.participants_csv),
        _read(args.stimuli_csv),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_structure.to_csv(args.output_dir / "task_modality_structure.csv", index=False)
    estimands.to_csv(args.output_dir / "estimand_identifiability.csv", index=False)
    issues.to_csv(args.output_dir / "identifiability_issues.csv", index=False)

    print(estimands.to_string(index=False))
    if len(issues):
        print("\nDesign issues:")
        print(issues.to_string(index=False))
    if not issues.empty and issues["severity"].eq("error").any():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
