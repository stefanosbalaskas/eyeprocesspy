#!/usr/bin/env python3
"""Bridge the frozen SRL glmmTMB result into the gaze-robustness contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "model_id",
    "term",
    "estimand_id",
    "estimate_log_rate_ratio",
    "SE",
    "CI_lower_log",
    "CI_upper_log",
    "converged",
    "n_rows",
}
EXPECTED_TERM = "prompt_indicator"
EXPECTED_ESTIMAND = "prompt_transition_rate_ratio"


def _bool_strict(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    key = str(value).strip().lower()
    if key in {"true", "1"}:
        return True
    if key in {"false", "0"}:
        return False
    raise ValueError(f"Cannot interpret converged={value!r} as boolean.")


def read_primary_glmm_result(path: Path) -> dict[str, object]:
    """Read exactly one frozen Prompt coefficient from the R estimator."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(
            "Primary GLMM result is missing required columns: "
            + ", ".join(missing)
        )
    if len(frame) != 1:
        raise ValueError(
            f"Primary GLMM result must contain exactly one row; found {len(frame)}."
        )

    row = frame.iloc[0]
    if str(row["term"]).strip() != EXPECTED_TERM:
        raise ValueError(
            f"Expected term={EXPECTED_TERM!r}; found {row['term']!r}."
        )
    if str(row["estimand_id"]).strip() != EXPECTED_ESTIMAND:
        raise ValueError(
            f"Expected estimand_id={EXPECTED_ESTIMAND!r}; "
            f"found {row['estimand_id']!r}."
        )

    numeric = {}
    for source, target in (
        ("estimate_log_rate_ratio", "estimate"),
        ("SE", "SE"),
        ("CI_lower_log", "CI_lower"),
        ("CI_upper_log", "CI_upper"),
        ("n_rows", "N"),
    ):
        value = pd.to_numeric(pd.Series([row[source]]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            raise ValueError(f"{source} must be finite.")
        numeric[target] = float(value)

    if numeric["N"] < 1 or numeric["N"] != float(int(numeric["N"])):
        raise ValueError("n_rows must be a positive integer.")

    return {
        **numeric,
        "N": int(numeric["N"]),
        "converged": _bool_strict(row["converged"]),
        "model_id": str(row["model_id"]).strip(),
        "term": EXPECTED_TERM,
        "estimand_id": EXPECTED_ESTIMAND,
        "effect_scale": "log_rate_ratio",
        "rate_ratio": float(np.exp(numeric["estimate"])),
        "CI_lower_rate_ratio": float(np.exp(numeric["CI_lower"])),
        "CI_upper_rate_ratio": float(np.exp(numeric["CI_upper"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coefficient_csv", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = read_primary_glmm_result(args.coefficient_csv)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(pd.DataFrame([result]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
