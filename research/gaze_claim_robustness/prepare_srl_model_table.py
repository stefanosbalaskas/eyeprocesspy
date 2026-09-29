#!/usr/bin/env python3
"""Prepare the SRL primary model table without fitting the focal model.

The function keeps every planned trial visible. Rows that cannot enter the
transition-rate model receive an explicit status/reason rather than being
silently dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "participant_id",
    "stimulus_id",
    "experiment_condition",
    "transition_count",
    "status",
}
STIMULI_REQUIRED = {
    "part_ID",
    "stimulus_name",
    "stimulus_type",
    "stimulus_time",
}


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


def prepare_model_table(
    outcome: pd.DataFrame,
    stimuli: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join outcome and exposure metadata, preserving non-evaluable rows."""
    _require(outcome, REQUIRED, "outcome")
    _require(stimuli, STIMULI_REQUIRED, "stimuli")

    out = outcome.copy()
    out["participant_id"] = out["participant_id"].map(_clean)
    out["stimulus_id"] = out["stimulus_id"].map(_stem)

    meta = stimuli.copy()
    meta["participant_id"] = meta["part_ID"].map(_clean)
    meta["stimulus_id"] = meta["stimulus_name"].map(_stem)
    duplicate = meta.duplicated(["participant_id", "stimulus_id"], keep=False)
    if duplicate.any():
        pairs = (
            meta.loc[duplicate, ["participant_id", "stimulus_id"]]
            .drop_duplicates()
            .astype(str)
            .agg("|".join, axis=1)
            .tolist()
        )
        raise ValueError(
            "stimuli metadata contains duplicate participant × stimulus rows: "
            + ", ".join(pairs)
        )

    joined = out.merge(
        meta[
            [
                "participant_id",
                "stimulus_id",
                "stimulus_type",
                "stimulus_time",
            ]
        ],
        on=["participant_id", "stimulus_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_metadata"),
    )

    exposure = pd.to_numeric(joined["stimulus_time"], errors="coerce")
    count = pd.to_numeric(joined["transition_count"], errors="coerce")
    source_ok = joined["status"].eq("ok")
    meta_ok = joined["_merge"].eq("both")
    exposure_ok = np.isfinite(exposure) & exposure.gt(0)
    count_ok = np.isfinite(count) & count.ge(0)

    model_status: list[str] = []
    for i in range(len(joined)):
        reasons: list[str] = []
        if not bool(source_ok.iloc[i]):
            reasons.append(f"outcome_status={joined.iloc[i]['status']}")
        if not bool(meta_ok.iloc[i]):
            reasons.append("missing_stimulus_metadata")
        if not bool(exposure_ok.iloc[i]):
            reasons.append("invalid_or_nonpositive_exposure")
        if not bool(count_ok.iloc[i]):
            reasons.append("invalid_transition_count")
        model_status.append("ok" if not reasons else "|".join(reasons))

    joined["model_status"] = model_status
    joined["model_evaluable"] = joined["model_status"].eq("ok")
    joined["exposure_seconds"] = exposure
    joined["log_exposure"] = np.where(
        joined["model_evaluable"],
        np.log(exposure),
        np.nan,
    )
    joined["prompt_indicator"] = joined["experiment_condition"].map(
        {"Non-prompt": 0.0, "Prompt": 1.0}
    )
    bad_condition = joined["prompt_indicator"].isna()
    if bad_condition.any():
        joined.loc[bad_condition, "model_evaluable"] = False
        joined.loc[bad_condition, "model_status"] = joined.loc[
            bad_condition, "model_status"
        ].map(
            lambda value: (
                "invalid_experiment_condition"
                if value == "ok"
                else value + "|invalid_experiment_condition"
            )
        )
        joined.loc[bad_condition, "log_exposure"] = np.nan

    audit = (
        joined.groupby("model_status", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["model_status"], kind="stable")
        .reset_index(drop=True)
    )

    joined = joined.drop(columns=["_merge"])
    return joined, audit


def primary_analysis_rows(model_table: pd.DataFrame) -> pd.DataFrame:
    """Return only explicitly evaluable rows for the eventual fitting engine."""
    if "model_evaluable" not in model_table:
        raise ValueError("model_table must contain model_evaluable.")
    return model_table.loc[model_table["model_evaluable"].astype(bool)].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("outcome_csv", type=Path)
    parser.add_argument("stimuli_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_model_table"),
    )
    args = parser.parse_args()

    table, audit = prepare_model_table(
        pd.read_csv(args.outcome_csv),
        pd.read_csv(args.stimuli_csv),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "srl_model_table.csv", index=False)
    audit.to_csv(args.output_dir / "srl_model_row_audit.csv", index=False)

    print(audit.to_string(index=False))
    print(
        f"Evaluable rows: {int(table['model_evaluable'].sum())} / {len(table)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
