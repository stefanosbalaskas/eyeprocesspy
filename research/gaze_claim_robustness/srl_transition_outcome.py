#!/usr/bin/env python3
"""Derive the prespecified SRL between-AOI transition-count outcome.

A transition is counted from the ordered fixation catalogue when two directly
adjacent fixation events are both assigned to AOIs and their AOI identifiers
differ. Unassigned fixations break the chain.

This distinction prevents:
A -> unassigned fixation -> B
from being silently collapsed into an A -> B transition.

Trials with no fixation evidence or no AOI-assigned fixations remain explicit
non-evaluable rows rather than being coerced to zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import eyeprocesspy as ep


def derive_transition_counts(dataset: ep.EyeDataset) -> pd.DataFrame:
    """Return one audited transition-count row per canonical trial interval."""
    if not ep.is_eye_dataset(dataset):
        raise TypeError("dataset must be an EyeDataset.")

    intervals = dataset["intervals"].copy()
    trials = intervals.loc[intervals["interval_type"].eq("trial")].copy()
    if trials.empty:
        raise ValueError("Trial intervals are required.")

    episodes = dataset["episodes"].copy()
    fixations = episodes.loc[episodes["episode_type"].eq("fixation")].copy()
    if "aoi_id" not in fixations.columns:
        raise ValueError("Fixation episodes must contain an aoi_id column.")

    rows: list[dict[str, object]] = []
    for _, trial in trials.iterrows():
        recording_id = trial["recording_id"]
        trial_id = trial["trial_id"]
        z = fixations.loc[
            fixations["recording_id"].eq(recording_id)
            & fixations["trial_id"].eq(trial_id)
        ].copy()
        z = z.sort_values(["start_time", "end_time", "episode_id"], kind="stable")

        total = len(z)
        assigned_mask = z["aoi_id"].notna()
        assigned = int(assigned_mask.sum())
        unassigned = int(total - assigned)

        transition_count: float = np.nan
        n_adjacent_pairs = max(total - 1, 0)
        n_evaluable_pairs = 0
        status = "ok"

        if total == 0:
            status = "no_fixations"
        elif assigned == 0:
            status = "no_aoi_assigned_fixations"
        else:
            aois = z["aoi_id"].astype("string").reset_index(drop=True)
            transition_count = 0.0
            if len(aois) >= 2:
                left = aois.iloc[:-1].reset_index(drop=True)
                right = aois.iloc[1:].reset_index(drop=True)
                evaluable = left.notna() & right.notna()
                n_evaluable_pairs = int(evaluable.sum())
                changed = evaluable & left.ne(right)
                transition_count = float(changed.sum())

        rows.append(
            {
                "recording_id": recording_id,
                "participant_id": trial.get("participant_id", pd.NA),
                "trial_id": trial_id,
                "stimulus_id": trial.get("stimulus_id", pd.NA),
                "condition_id": trial.get("condition_id", pd.NA),
                "experiment_condition": trial.get(
                    "experiment_condition",
                    pd.NA,
                ),
                "stimulus_type": trial.get("stimulus_type", pd.NA),
                "transition_count": transition_count,
                "status": status,
                "n_fixations_total": total,
                "n_fixations_assigned": assigned,
                "n_fixations_unassigned": unassigned,
                "n_adjacent_fixation_pairs": n_adjacent_pairs,
                "n_evaluable_adjacent_pairs": n_evaluable_pairs,
                "assigned_fixation_fraction": (
                    assigned / total if total else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def transition_sequence_audit(dataset: ep.EyeDataset) -> pd.DataFrame:
    """Return fixation adjacency details used to compute transition counts."""
    if not ep.is_eye_dataset(dataset):
        raise TypeError("dataset must be an EyeDataset.")

    fixations = dataset["episodes"].loc[
        dataset["episodes"]["episode_type"].eq("fixation")
    ].copy()
    if fixations.empty:
        return pd.DataFrame(
            columns=[
                "recording_id",
                "trial_id",
                "from_episode_id",
                "to_episode_id",
                "from_aoi_id",
                "to_aoi_id",
                "evaluable",
                "between_aoi_transition",
            ]
        )

    rows: list[dict[str, object]] = []
    for (recording_id, trial_id), z in fixations.groupby(
        ["recording_id", "trial_id"],
        sort=False,
        dropna=False,
        observed=True,
    ):
        z = z.sort_values(["start_time", "end_time", "episode_id"], kind="stable")
        z = z.reset_index(drop=True)
        for i in range(max(len(z) - 1, 0)):
            left = z.iloc[i]
            right = z.iloc[i + 1]
            left_aoi = left["aoi_id"]
            right_aoi = right["aoi_id"]
            evaluable = pd.notna(left_aoi) and pd.notna(right_aoi)
            transition = bool(evaluable and str(left_aoi) != str(right_aoi))
            rows.append(
                {
                    "recording_id": recording_id,
                    "trial_id": trial_id,
                    "from_episode_id": left["episode_id"],
                    "to_episode_id": right["episode_id"],
                    "from_aoi_id": left_aoi,
                    "to_aoi_id": right_aoi,
                    "evaluable": evaluable,
                    "between_aoi_transition": transition,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_pickle",
        type=Path,
        help=(
            "Research convenience only: a trusted local pickle containing an "
            "EyeDataset with fixation episodes and AOI assignments."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("srl_transition_counts.csv"),
    )
    args = parser.parse_args()

    dataset = pd.read_pickle(args.dataset_pickle)
    outcome = derive_transition_counts(dataset)
    outcome.to_csv(args.output, index=False)
    print(outcome.to_string(index=False))

    if outcome["status"].ne("ok").any():
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
