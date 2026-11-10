#!/usr/bin/env python3
"""Summarize the frozen MCFW detector-validation universe descriptively."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAIR_METRICS = (
    "agreement_proportion",
    "fixation_jaccard",
    "cohen_kappa",
    "fixation_count_absolute_difference",
    "fixation_count_symmetric_relative_difference",
    "fixation_rate_absolute_difference",
    "fixation_rate_symmetric_relative_difference",
    "median_duration_absolute_difference_ms",
    "median_duration_symmetric_relative_difference",
    "fixation_time_proportion_absolute_difference",
    "fixation_time_proportion_symmetric_relative_difference",
)


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {
            "n_finite": 0,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
        }
    return {
        "n_finite": int(len(finite)),
        "median": float(finite.median()),
        "q25": float(finite.quantile(0.25)),
        "q75": float(finite.quantile(0.75)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def summarize_pair_metrics(
    pairs: pd.DataFrame,
    *,
    group_by: list[str],
) -> pd.DataFrame:
    """Return long-form distribution summaries for successful pair rows."""
    ok = pairs.loc[pairs["status"].eq("ok")].copy()
    rows: list[dict[str, object]] = []
    for keys, group in ok.groupby(group_by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        common = dict(zip(group_by, keys, strict=True))
        for metric in PAIR_METRICS:
            rows.append(
                {
                    **common,
                    "metric": metric,
                    **_quantiles(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def quality_association(pairs: pd.DataFrame) -> pd.DataFrame:
    """Describe usable-gaze association with detector disagreement.

    Spearman rho is reported without p-value voting. The association is
    descriptive and does not imply that data quality causally changes detector
    disagreement.
    """
    ok = pairs.loc[pairs["status"].eq("ok")].copy()
    ok["agreement_disagreement"] = 1.0 - pd.to_numeric(
        ok["agreement_proportion"], errors="coerce"
    )
    ok["jaccard_disagreement"] = 1.0 - pd.to_numeric(
        ok["fixation_jaccard"], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    grouping_sets = [
        ["detector_a", "detector_b", "eye"],
        ["detector_a", "detector_b", "eye", "trial_family"],
    ]
    for group_by in grouping_sets:
        for keys, group in ok.groupby(group_by, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            common = dict(zip(group_by, keys, strict=True))
            for outcome in (
                "agreement_disagreement",
                "jaccard_disagreement",
                "fixation_count_symmetric_relative_difference",
                "fixation_rate_symmetric_relative_difference",
                "median_duration_symmetric_relative_difference",
                "fixation_time_proportion_symmetric_relative_difference",
            ):
                x = pd.to_numeric(
                    group["analysis_usable_fraction"], errors="coerce"
                ).to_numpy(float)
                y = pd.to_numeric(group[outcome], errors="coerce").to_numpy(float)
                keep = np.isfinite(x) & np.isfinite(y)
                n = int(keep.sum())
                rho = (
                    float(spearmanr(x[keep], y[keep]).statistic)
                    if n >= 3
                    else np.nan
                )
                rows.append(
                    {
                        **common,
                        "grouping": "+".join(group_by),
                        "outcome": outcome,
                        "n": n,
                        "spearman_rho_usable_fraction_vs_disagreement": rho,
                    }
                )
    return pd.DataFrame(rows)


def status_summary(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize planned versus evaluable detector and pairwise rows."""
    rows = []
    for table_name, frame in (("event_summary", events), ("pairwise", pairs)):
        counts = frame["status"].value_counts(dropna=False)
        rows.append(
            {
                "table": table_name,
                "planned_rows": int(len(frame)),
                "ok_rows": int(frame["status"].eq("ok").sum()),
                "non_ok_rows": int((~frame["status"].eq("ok")).sum()),
                "ok_proportion": (
                    float(frame["status"].eq("ok").mean()) if len(frame) else np.nan
                ),
                "status_counts": ";".join(
                    f"{index}={int(value)}"
                    for index, value in counts.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def family_profile(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return compact pair-by-eye-by-context medians for paper-facing inspection."""
    ok = pairs.loc[pairs["status"].eq("ok")].copy()
    metrics = [
        "agreement_proportion",
        "fixation_jaccard",
        "cohen_kappa",
        "fixation_count_symmetric_relative_difference",
        "fixation_rate_symmetric_relative_difference",
        "median_duration_symmetric_relative_difference",
        "fixation_time_proportion_symmetric_relative_difference",
        "analysis_usable_fraction",
    ]
    aggregations = {metric: "median" for metric in metrics}
    out = (
        ok.groupby(
            ["detector_a", "detector_b", "eye", "trial_family"],
            dropna=False,
            sort=True,
        )
        .agg(
            rows=("trial", "size"),
            participants=("participant", "nunique"),
            **{
                f"median_{metric}": (metric, function)
                for metric, function in aggregations.items()
            },
        )
        .reset_index()
    )
    return out


def write_summary_markdown(
    path: Path,
    status: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    pair_row = status.loc[status["table"].eq("pairwise")].iloc[0]
    ok = pairs.loc[pairs["status"].eq("ok")].copy()

    lines = [
        "# MCFW-Gaze detector-validation summary",
        "",
        "This artifact is an independent measurement-generalization analysis.",
        "It does not create or test an HCI treatment effect and it does not select",
        "a scientifically correct detector.",
        "",
        f"- planned pairwise rows: {int(pair_row['planned_rows'])}",
        f"- successful pairwise rows: {int(pair_row['ok_rows'])}",
        f"- non-ok pairwise rows retained: {int(pair_row['non_ok_rows'])}",
    ]
    if not ok.empty:
        lines.extend(
            [
                (
                    "- median sample-level agreement across evaluable rows: "
                    f"{pd.to_numeric(ok['agreement_proportion'], errors='coerce').median():.6f}"
                ),
                (
                    "- median fixation-state Jaccard across evaluable rows: "
                    f"{pd.to_numeric(ok['fixation_jaccard'], errors='coerce').median():.6f}"
                ),
                (
                    "- median Cohen kappa across evaluable rows: "
                    f"{pd.to_numeric(ok['cohen_kappa'], errors='coerce').median():.6f}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "All distributions and quality associations are descriptive.",
            "No p-value voting or automatic robust/fragile label is applied.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.validation_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(
        args.validation_dir / "mcfw_detector_event_summaries.csv"
    )
    pairs = pd.read_csv(
        args.validation_dir / "mcfw_detector_pairwise_agreement.csv"
    )

    status = status_summary(events, pairs)
    overall = summarize_pair_metrics(
        pairs,
        group_by=["detector_a", "detector_b", "eye"],
    )
    context = summarize_pair_metrics(
        pairs,
        group_by=["detector_a", "detector_b", "eye", "trial_family"],
    )
    quality = quality_association(pairs)
    profile = family_profile(pairs)

    status.to_csv(output_dir / "mcfw_validation_status.csv", index=False)
    overall.to_csv(output_dir / "mcfw_pairwise_overall.csv", index=False)
    context.to_csv(output_dir / "mcfw_pairwise_by_context.csv", index=False)
    quality.to_csv(output_dir / "mcfw_quality_association.csv", index=False)
    profile.to_csv(output_dir / "mcfw_context_profile.csv", index=False)
    write_summary_markdown(
        output_dir / "SUMMARY.md",
        status,
        pairs,
    )

    print(status.to_string(index=False))
    print()
    print(profile.to_string(index=False, max_rows=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
