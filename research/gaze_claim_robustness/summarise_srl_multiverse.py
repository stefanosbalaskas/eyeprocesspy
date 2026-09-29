#!/usr/bin/env python3
"""Summarize SRL multiverse results without automatic robustness labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DECISIONS = (
    "detector_id",
    "eye",
    "viewing_distance_cm",
    "aoi_convention",
    "quality_rule",
    "cohort_id",
)


def summarize_results(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "universe_id",
        "status",
        "estimate",
        "SE",
        "CI_lower",
        "CI_upper",
        "rate_ratio",
        *DECISIONS,
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError("Results missing columns: " + ", ".join(missing))

    ok = results.loc[results["status"].eq("ok")].copy()
    estimate = pd.to_numeric(ok["estimate"], errors="coerce")
    rr = pd.to_numeric(ok["rate_ratio"], errors="coerce")
    finite = np.isfinite(estimate)
    ok = ok.loc[finite].copy()
    estimate = pd.to_numeric(ok["estimate"], errors="coerce")
    rr = pd.to_numeric(ok["rate_ratio"], errors="coerce")

    summary = pd.DataFrame(
        [
            {
                "planned_specifications": len(results),
                "successful_specifications": len(ok),
                "non_ok_specifications": int(
                    (~results["status"].eq("ok")).sum()
                ),
                "positive_log_rate_ratio_specifications": int(
                    (estimate > 0).sum()
                ),
                "negative_log_rate_ratio_specifications": int(
                    (estimate < 0).sum()
                ),
                "zero_log_rate_ratio_specifications": int(
                    np.isclose(estimate, 0.0, atol=1e-12).sum()
                ),
                "positive_direction_proportion": (
                    float((estimate > 0).mean()) if len(ok) else np.nan
                ),
                "median_log_rate_ratio": (
                    float(estimate.median()) if len(ok) else np.nan
                ),
                "minimum_log_rate_ratio": (
                    float(estimate.min()) if len(ok) else np.nan
                ),
                "maximum_log_rate_ratio": (
                    float(estimate.max()) if len(ok) else np.nan
                ),
                "median_rate_ratio": (
                    float(rr.median()) if len(ok) else np.nan
                ),
                "minimum_rate_ratio": (
                    float(rr.min()) if len(ok) else np.nan
                ),
                "maximum_rate_ratio": (
                    float(rr.max()) if len(ok) else np.nan
                ),
            }
        ]
    )

    rows: list[dict[str, object]] = []
    for decision in DECISIONS:
        for level, group in ok.groupby(decision, dropna=False, sort=True):
            values = pd.to_numeric(group["estimate"], errors="coerce")
            rows.append(
                {
                    "decision": decision,
                    "level": level,
                    "n_specifications": len(group),
                    "mean_log_rate_ratio": float(values.mean()),
                    "median_log_rate_ratio": float(values.median()),
                    "minimum_log_rate_ratio": float(values.min()),
                    "maximum_log_rate_ratio": float(values.max()),
                }
            )
    decision = pd.DataFrame(rows)

    ranges: list[dict[str, object]] = []
    for name, group in decision.groupby("decision", sort=True):
        means = pd.to_numeric(group["mean_log_rate_ratio"], errors="coerce")
        medians = pd.to_numeric(group["median_log_rate_ratio"], errors="coerce")
        ranges.append(
            {
                "decision": name,
                "levels": len(group),
                "marginal_mean_range": float(means.max() - means.min()),
                "marginal_median_range": float(
                    medians.max() - medians.min()
                ),
            }
        )
    decision_ranges = pd.DataFrame(ranges)
    decision = decision.merge(
        decision_ranges,
        on="decision",
        how="left",
        validate="many_to_one",
    )
    return summary, decision


def make_specification_curve(
    results: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    data = results.loc[results["status"].eq("ok")].copy()
    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data["CI_lower"] = pd.to_numeric(data["CI_lower"], errors="coerce")
    data["CI_upper"] = pd.to_numeric(data["CI_upper"], errors="coerce")
    data = data.loc[np.isfinite(data["estimate"])].copy()
    data = data.sort_values(["estimate", "universe_id"], kind="stable")
    data = data.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(data))
    estimate = data["estimate"].to_numpy(dtype=float)
    lower = data["CI_lower"].to_numpy(dtype=float)
    upper = data["CI_upper"].to_numpy(dtype=float)
    finite_ci = np.isfinite(lower) & np.isfinite(upper)

    if finite_ci.any():
        ax.vlines(
            x[finite_ci],
            lower[finite_ci],
            upper[finite_ci],
            linewidth=0.7,
        )
    ax.scatter(x, estimate, s=12)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Frozen specification ordered by Prompt log-rate ratio")
    ax.set_ylabel("Prompt vs Non-prompt log transition-rate ratio")
    ax.set_title("SRL gaze-claim measurement robustness")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("srl_multiverse_summary"),
    )
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    summary, decisions = summarize_results(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "srl_claim_summary.csv", index=False)
    decisions.to_csv(
        args.output_dir / "srl_decision_sensitivity.csv",
        index=False,
    )
    make_specification_curve(
        results,
        args.output_dir / "srl_specification_curve.png",
    )

    row = summary.iloc[0]
    report = [
        "# SRL Prompt transition-rate robustness",
        "",
        f"- planned specifications: {int(row['planned_specifications'])}",
        f"- successful specifications: {int(row['successful_specifications'])}",
        f"- non-ok specifications: {int(row['non_ok_specifications'])}",
        (
            "- positive Prompt log-rate-ratio proportion: "
            f"{row['positive_direction_proportion']:.6f}"
        ),
        (
            "- log-rate-ratio median/range: "
            f"{row['median_log_rate_ratio']:.6f} "
            f"[{row['minimum_log_rate_ratio']:.6f}, "
            f"{row['maximum_log_rate_ratio']:.6f}]"
        ),
        (
            "- rate-ratio median/range: "
            f"{row['median_rate_ratio']:.6f} "
            f"[{row['minimum_rate_ratio']:.6f}, "
            f"{row['maximum_rate_ratio']:.6f}]"
        ),
        "",
        (
            "These are descriptive stability quantities within the frozen "
            "measurement universe. They are not probabilities that the "
            "scientific claim is true and no automatic robust/fragile label "
            "is assigned."
        ),
    ]
    (args.output_dir / "SUMMARY.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
