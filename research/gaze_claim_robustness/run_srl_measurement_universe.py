#!/usr/bin/env python3
"""Execute the frozen SRL raw-gaze measurement universe.

This script computes only measurement outcomes. It does not fit the focal
Prompt effect.

For each exact raw+metadata participant it evaluates:
3 detectors x 2 eyes x 3 viewing-distance assumptions x 2 AOI geometries
= 36 measurement branches per learning slide.

With 82 exact participants x 8 slides, the complete planned output contains
23,616 participant x slide x measurement rows. Cohort and quality decisions
are applied later at model time, so gaze detection is not duplicated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import srl_adapter
import srl_aoi_geometry
import srl_coordinate_branches
import srl_transition_outcome

import eyeprocesspy as ep

TASKS = tuple(f"Task_{i}" for i in range(1, 9))
EYES = ("left", "right")
DISTANCES_CM = (60.0, 65.0, 70.0)
AOI_CONVENTIONS = ("exact_quarters", "reported_area_center_seam")
DETECTOR_IDS = ("ivt_30_100_simple", "ivt_40_50_simple", "idt_1_100")


def frozen_detector_specs() -> tuple[Any, ...]:
    """Return the three open detector specifications in the primary universe."""
    return (
        ep.define_event_detector_spec(
            "ivt_30_100_simple",
            "ivt",
            velocity_threshold=30.0,
            minimum_duration_ms=100.0,
            maximum_gap_ms=75.0,
            sampling_rate=250.0,
            coordinate_unit="degrees",
            merge_rule="none",
            implementation="eyeprocesspy",
            parameters={"include_saccades": False},
        ),
        ep.define_event_detector_spec(
            "ivt_40_50_simple",
            "ivt",
            velocity_threshold=40.0,
            minimum_duration_ms=50.0,
            maximum_gap_ms=75.0,
            sampling_rate=250.0,
            coordinate_unit="degrees",
            merge_rule="none",
            implementation="eyeprocesspy",
            parameters={"include_saccades": False},
        ),
        ep.define_event_detector_spec(
            "idt_1_100",
            "idt",
            dispersion_threshold=1.0,
            minimum_duration_ms=100.0,
            sampling_rate=250.0,
            coordinate_unit="degrees",
            merge_rule="none",
            implementation="eyeprocesspy",
        ),
    )


def combine_trial_datasets(
    trials: list[ep.EyeDataset],
    *,
    participant_id: str,
    eye: str,
    source_file: Path,
) -> ep.EyeDataset:
    """Combine already-mapped Task datasets without duplicating raw frames."""
    if not trials:
        raise ValueError("At least one trial dataset is required.")
    if eye not in EYES:
        raise ValueError("eye must be left or right.")

    def cat(name: str) -> pd.DataFrame:
        frames = [trial[name] for trial in trials if not trial[name].empty]
        return (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )

    coordinate_spaces = cat("coordinate_spaces")
    if not coordinate_spaces.empty:
        coordinate_spaces = coordinate_spaces.drop_duplicates(
            subset=["coordinate_space_id"],
            keep="first",
        ).reset_index(drop=True)

    out = ep.new_eye_dataset(
        recordings=cat("recordings"),
        streams=cat("streams"),
        gaze_samples=cat("gaze_samples"),
        eye_samples=cat("eye_samples"),
        intervals=cat("intervals"),
        coordinate_spaces=coordinate_spaces,
        raw=[],
        vendor_metadata={
            "source_dataset": "Juřík et al. SRL eye-tracking dataset",
            "dataset_doi": "10.6084/m9.figshare.28304069",
            "participant_id": str(participant_id),
            "source_eye": eye,
            "combined_learning_tasks": len(trials),
            "binocular_fusion": "none",
            "interpolation": "none",
            "smoothing": "none",
        },
        validate=True,
    )
    return ep.add_provenance(
        out,
        "combine_srl_learning_trials",
        "gaze_samples",
        (
            f"participant={participant_id};eye={eye};"
            f"n_learning_tasks={len(trials)};"
            "raw_frames_not_duplicated=true"
        ),
        source_files=str(source_file),
        software="eyeprocesspy research runner",
        reversible=True,
    )


def _metadata_lookup(
    participants: pd.DataFrame,
    stimuli: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    p = participants.copy()
    p["participant_id"] = p["part_ID"].map(lambda x: str(x).strip())
    condition = (
        p.drop_duplicates("participant_id")
        .set_index("participant_id")["experiment_condition"]
        .astype(str)
        .to_dict()
    )

    s = stimuli.copy()
    s["participant_id"] = s["part_ID"].map(lambda x: str(x).strip())
    s["stimulus_id"] = s["stimulus_name"].map(
        lambda x: Path(str(x).strip()).stem
    )
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in s.iterrows():
        participant_id = str(row["participant_id"])
        stimulus_id = str(row["stimulus_id"])
        lookup[(participant_id, stimulus_id)] = {
            "experiment_condition": condition.get(participant_id, ""),
            "stimulus_type": row.get("stimulus_type", pd.NA),
            "tracking_ratio": row.get("tracking_ratio", np.nan),
            "stimulus_time": row.get("stimulus_time", np.nan),
        }
    return lookup


def _placeholder_rows(
    *,
    participant_id: str,
    metadata: dict[tuple[str, str], dict[str, object]],
    detector_id: str,
    eye: str,
    viewing_distance_cm: float,
    aoi_convention: str,
    failed_tasks: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, reason in failed_tasks.items():
        meta = metadata.get((participant_id, task), {})
        rows.append(
            {
                "participant_id": participant_id,
                "trial_id": f"{participant_id}_{task}",
                "stimulus_id": task,
                "condition_id": (
                    f"{meta.get('experiment_condition', '')}|"
                    f"{meta.get('stimulus_type', '')}"
                ),
                "experiment_condition": meta.get("experiment_condition", pd.NA),
                "stimulus_type": meta.get("stimulus_type", pd.NA),
                "transition_count": np.nan,
                "status": reason,
                "n_fixations_total": np.nan,
                "n_fixations_assigned": np.nan,
                "n_fixations_unassigned": np.nan,
                "n_adjacent_fixation_pairs": np.nan,
                "n_evaluable_adjacent_pairs": np.nan,
                "assigned_fixation_fraction": np.nan,
                "n_negative_gap_pairs": np.nan,
                "median_inter_fixation_gap_ms": np.nan,
                "max_inter_fixation_gap_ms": np.nan,
                "outcome_operationalization": (
                    "adjacent_aoi_assigned_fixation_change"
                ),
                "metadata_tracking_ratio_percent": meta.get(
                    "tracking_ratio",
                    np.nan,
                ),
                "metadata_stimulus_time_seconds": meta.get(
                    "stimulus_time",
                    np.nan,
                ),
                "detector_id": detector_id,
                "eye": eye,
                "viewing_distance_cm": viewing_distance_cm,
                "aoi_convention": aoi_convention,
            }
        )
    return rows


def run_participant(
    participant_id: str,
    *,
    raw_file: Path,
    participants_csv: Path,
    stimuli_csv: Path,
    participants: pd.DataFrame,
    stimuli: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all 36 measurement branches for one exact-cohort participant."""
    metadata = _metadata_lookup(participants, stimuli)
    result_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    specs = frozen_detector_specs()

    for eye in EYES:
        mapped: list[ep.EyeDataset] = []
        adapter_failures: dict[str, str] = {}

        for task in TASKS:
            try:
                mapped.append(
                    srl_adapter.load_srl_trial(
                        raw_file,
                        participants_csv,
                        stimuli_csv,
                        stimulus_name=task,
                        eye=eye,
                    )
                )
            except Exception as exc:
                adapter_failures[task] = (
                    f"adapter_failed:{type(exc).__name__}:{exc}"
                )
                failure_rows.append(
                    {
                        "participant_id": participant_id,
                        "eye": eye,
                        "viewing_distance_cm": pd.NA,
                        "detector_id": pd.NA,
                        "aoi_convention": pd.NA,
                        "stimulus_id": task,
                        "stage": "adapter",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        if not mapped:
            for distance in DISTANCES_CM:
                for detector_id in DETECTOR_IDS:
                    for aoi in AOI_CONVENTIONS:
                        result_rows.extend(
                            _placeholder_rows(
                                participant_id=participant_id,
                                metadata=metadata,
                                detector_id=detector_id,
                                eye=eye,
                                viewing_distance_cm=distance,
                                aoi_convention=aoi,
                                failed_tasks={
                                    task: adapter_failures.get(
                                        task,
                                        "adapter_failed:no_evaluable_task",
                                    )
                                    for task in TASKS
                                },
                            )
                        )
            continue

        combined = combine_trial_datasets(
            mapped,
            participant_id=participant_id,
            eye=eye,
            source_file=raw_file,
        )
        mapped_tasks = {
            str(value)
            for value in combined["intervals"]["stimulus_id"].dropna().unique()
        }

        for distance in DISTANCES_CM:
            angular = srl_coordinate_branches.convert_srl_pixels_to_degrees(
                combined,
                viewing_distance_cm=distance,
            )
            detected = ep.run_detector_multiverse(
                angular,
                specs,
                continue_on_error=True,
            )

            for _, row in detected.status.iterrows():
                status_rows.append(
                    {
                        "participant_id": participant_id,
                        "eye": eye,
                        "viewing_distance_cm": distance,
                        **row.to_dict(),
                    }
                )
            if not detected.failures.empty:
                for _, row in detected.failures.iterrows():
                    failure_rows.append(
                        {
                            "participant_id": participant_id,
                            "eye": eye,
                            "viewing_distance_cm": distance,
                            "aoi_convention": pd.NA,
                            "stimulus_id": pd.NA,
                            **row.to_dict(),
                        }
                    )
            if not detected.warnings.empty:
                for _, row in detected.warnings.iterrows():
                    warning_rows.append(
                        {
                            "participant_id": participant_id,
                            "eye": eye,
                            "viewing_distance_cm": distance,
                            **row.to_dict(),
                        }
                    )

            for spec in specs:
                detector_id = spec.detector_id
                branch = detected.branches.get(detector_id)

                for aoi in AOI_CONVENTIONS:
                    failed_tasks = dict(adapter_failures)

                    if branch is None:
                        for task in mapped_tasks:
                            failed_tasks[task] = "detector_failed"
                        result_rows.extend(
                            _placeholder_rows(
                                participant_id=participant_id,
                                metadata=metadata,
                                detector_id=detector_id,
                                eye=eye,
                                viewing_distance_cm=distance,
                                aoi_convention=aoi,
                                failed_tasks=failed_tasks,
                            )
                        )
                        continue

                    try:
                        registered = (
                            srl_aoi_geometry.register_srl_quartile_aois_many(
                                branch,
                                stimulus_ids=sorted(mapped_tasks),
                                convention=aoi,
                            )
                        )
                        assigned = ep.assign_aois(
                            registered,
                            component="episodes",
                            overlap="first",
                            overwrite=True,
                        )
                        outcome = (
                            srl_transition_outcome.derive_transition_counts(
                                assigned
                            )
                        )
                        outcome["detector_id"] = detector_id
                        outcome["eye"] = eye
                        outcome["viewing_distance_cm"] = distance
                        outcome["aoi_convention"] = aoi
                        outcome["detector_spec_hash"] = spec.fingerprint

                        for index in outcome.index:
                            task = str(outcome.at[index, "stimulus_id"])
                            meta = metadata.get((participant_id, task), {})
                            outcome.at[
                                index,
                                "metadata_tracking_ratio_percent",
                            ] = meta.get("tracking_ratio", np.nan)
                            outcome.at[
                                index,
                                "metadata_stimulus_time_seconds",
                            ] = meta.get("stimulus_time", np.nan)

                        result_rows.extend(outcome.to_dict(orient="records"))

                        if failed_tasks:
                            result_rows.extend(
                                _placeholder_rows(
                                    participant_id=participant_id,
                                    metadata=metadata,
                                    detector_id=detector_id,
                                    eye=eye,
                                    viewing_distance_cm=distance,
                                    aoi_convention=aoi,
                                    failed_tasks=failed_tasks,
                                )
                            )
                    except Exception as exc:
                        failure_rows.append(
                            {
                                "participant_id": participant_id,
                                "eye": eye,
                                "viewing_distance_cm": distance,
                                "detector_id": detector_id,
                                "aoi_convention": aoi,
                                "stimulus_id": pd.NA,
                                "stage": "aoi_or_outcome",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        all_failed = {
                            task: (
                                failed_tasks.get(task)
                                or f"aoi_or_outcome_failed:{type(exc).__name__}:{exc}"
                            )
                            for task in TASKS
                        }
                        result_rows.extend(
                            _placeholder_rows(
                                participant_id=participant_id,
                                metadata=metadata,
                                detector_id=detector_id,
                                eye=eye,
                                viewing_distance_cm=distance,
                                aoi_convention=aoi,
                                failed_tasks=all_failed,
                            )
                        )

    results = pd.DataFrame(result_rows)
    statuses = pd.DataFrame(status_rows)
    failures = pd.DataFrame(failure_rows)
    warnings = pd.DataFrame(warning_rows)

    expected = len(TASKS) * len(EYES) * len(DISTANCES_CM) * len(DETECTOR_IDS) * len(
        AOI_CONVENTIONS
    )
    if len(results) != expected:
        raise RuntimeError(
            f"Participant {participant_id}: expected {expected} measurement rows; "
            f"generated {len(results)}."
        )

    key = [
        "participant_id",
        "stimulus_id",
        "detector_id",
        "eye",
        "viewing_distance_cm",
        "aoi_convention",
    ]
    if results.duplicated(key).any():
        duplicate = results.loc[results.duplicated(key, keep=False), key]
        raise RuntimeError(
            "Measurement result contains duplicate branch keys: "
            + duplicate.head(10).astype(str).agg("|".join, axis=1).str.cat(sep=", ")
        )

    return results, statuses, failures, warnings


def exact_cohort_ids(identity_presence_csv: Path) -> list[str]:
    """Return only exact cross-source raw-eligible IDs."""
    presence = pd.read_csv(identity_presence_csv)
    required = {"participant_id", "exact_cross_source_raw_eligible"}
    missing = required - set(presence.columns)
    if missing:
        raise ValueError(
            "Identity presence table missing: " + ", ".join(sorted(missing))
        )
    ids = (
        presence.loc[
            presence["exact_cross_source_raw_eligible"].astype(bool),
            "participant_id",
        ]
        .astype(str)
        .tolist()
    )
    ids = sorted(
        ids,
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )
    if len(ids) != 82:
        raise ValueError(f"Expected 82 exact raw participants; found {len(ids)}.")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("identity_presence_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_measurement_universe"),
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        default=0,
        help="Research smoke-test limit; 0 means all exact participants.",
    )
    args = parser.parse_args()

    participants_matches = sorted(args.dataset_root.rglob("participants.csv"))
    stimuli_matches = sorted(args.dataset_root.rglob("stimuli.csv"))
    raw_dirs = sorted(
        path for path in args.dataset_root.rglob("ET_data_raw") if path.is_dir()
    )
    if len(participants_matches) != 1 or len(stimuli_matches) != 1 or len(raw_dirs) != 1:
        raise ValueError(
            "Expected exactly one participants.csv, stimuli.csv, and ET_data_raw directory."
        )

    participants_csv = participants_matches[0]
    stimuli_csv = stimuli_matches[0]
    raw_dir = raw_dirs[0]
    participants = pd.read_csv(participants_csv)
    stimuli = pd.read_csv(stimuli_csv)

    ids = exact_cohort_ids(args.identity_presence_csv)
    if args.max_participants < 0:
        raise ValueError("max_participants must be >= 0.")
    if args.max_participants:
        ids = ids[: args.max_participants]

    result_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    warning_frames: list[pd.DataFrame] = []

    for index, participant_id in enumerate(ids, start=1):
        raw_file = raw_dir / f"ET_data_raw_{participant_id}.txt"
        if not raw_file.exists():
            raise FileNotFoundError(raw_file)

        results, statuses, failures, detector_warnings = run_participant(
            participant_id,
            raw_file=raw_file,
            participants_csv=participants_csv,
            stimuli_csv=stimuli_csv,
            participants=participants,
            stimuli=stimuli,
        )
        result_frames.append(results)
        if not statuses.empty:
            status_frames.append(statuses)
        if not failures.empty:
            failure_frames.append(failures)
        if not detector_warnings.empty:
            warning_frames.append(detector_warnings)

        print(
            f"[{index}/{len(ids)}] participant={participant_id}; "
            f"measurement_rows={len(results)}; failures={len(failures)}; "
            f"warnings={len(detector_warnings)}",
            flush=True,
        )

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    statuses = (
        pd.concat(status_frames, ignore_index=True, sort=False)
        if status_frames
        else pd.DataFrame()
    )
    failures = (
        pd.concat(failure_frames, ignore_index=True, sort=False)
        if failure_frames
        else pd.DataFrame()
    )
    detector_warnings = (
        pd.concat(warning_frames, ignore_index=True, sort=False)
        if warning_frames
        else pd.DataFrame()
    )

    expected = len(ids) * 8 * 36
    if len(results) != expected:
        raise RuntimeError(
            f"Expected {expected} total measurement rows; generated {len(results)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        args.output_dir / "srl_measurement_counts.csv",
        index=False,
    )
    statuses.to_csv(
        args.output_dir / "srl_detector_status.csv",
        index=False,
    )
    failures.to_csv(
        args.output_dir / "srl_measurement_failures.csv",
        index=False,
    )
    detector_warnings.to_csv(
        args.output_dir / "srl_detector_warnings.csv",
        index=False,
    )

    print(
        f"Completed {len(ids)} participants; "
        f"rows={len(results)}; "
        f"non_ok={int((~results['status'].eq('ok')).sum())}; "
        f"failure_records={len(failures)}; "
        f"warning_records={len(detector_warnings)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
