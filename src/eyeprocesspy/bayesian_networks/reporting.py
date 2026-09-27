"""Reproducible Markdown reporting for Bayesian-network analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    BayesianNetworkComparisonResult,
    BayesianNetworkResult,
    BayesianNetworkStabilityResult,
    BayesianNetworkValidationResult,
)


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    columns = [str(c) for c in table.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in table.itertuples(index=False, name=None):
        body.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep, *body])


_CAUSAL_BOUNDARY = (
    "A learned directed edge represents a fitted probabilistic dependency/orientation under the "
    "specified data, algorithm, and constraints; it is not, by itself, evidence of a causal effect."
)


def report_bayesian_network(
    result: BayesianNetworkResult,
    *,
    stability: BayesianNetworkStabilityResult | None = None,
    validation: BayesianNetworkValidationResult | None = None,
    sensitivity: BayesianNetworkComparisonResult | None = None,
    path: str | Path | None = None,
) -> str:
    """Create a compact reproducibility-oriented Markdown report."""
    data = result.data_spec
    lines: list[str] = [
        "# Bayesian-network analysis report",
        "",
        "## Scientific boundary",
        "",
        _CAUSAL_BOUNDARY,
        "",
        "## Data",
        "",
        f"- Rows: **{len(data.data)}**",
        f"- Observation level: **{data.observation_level}**",
        f"- Structure nodes: **{len(result.nodes)}**",
        f"- Model family: **{result.model_family}**",
        f"- Participant identifier: **{data.participant_id or 'not declared'}**",
        "",
        "## Structure",
        "",
        f"- Backend: **{result.backend}**",
        f"- Algorithm: **{result.structure_algorithm or 'not recorded'}**",
        f"- Score: **{result.score or 'not applicable'}**",
        f"- Parameter estimator: **{result.parameter_estimator or 'not fitted'}**",
        f"- Learned/fixed edges: **{len(result.edges)}**",
        "",
    ]
    if result.edges:
        lines.extend(["| Source | Target |", "|---|---|"])
        lines.extend(f"| {a} | {b} |" for a, b in result.edges)
        lines.append("")
    if stability is not None:
        lines.extend(
            [
                "## Bootstrap stability",
                "",
                f"- Bootstrap replications: **{stability.n_boot}**",
                f"- Successful fits: **{stability.successful_fits}**",
                f"- Failed fits retained in denominator: **{stability.failed_fits}**",
                f"- Resampling unit: **{stability.resample_by or 'row'}**",
                "",
                (
                    _markdown_table(stability.edge_table)
                    if not stability.edge_table.empty
                    else "No edges were recovered."
                ),
                "",
            ]
        )
    if validation is not None:
        mean_ll = validation.summary.get("mean_log_likelihood", float("nan"))
        lines.extend(
            [
                "## Validation",
                "",
                f"- Method: **{validation.method}**",
                f"- Group column: **{validation.group_column or 'not declared'}**",
                f"- Mean held-out log likelihood per row: **{mean_ll:.4f}**",
                "",
            ]
        )
    if sensitivity is not None:
        lines.extend(
            [
                "## Sensitivity",
                "",
                (
                    _markdown_table(sensitivity.edge_table)
                    if not sensitivity.edge_table.empty
                    else "No edges were recovered across specifications."
                ),
                "",
            ]
        )
    if result.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
        lines.append("")
    provenance: dict[str, Any] = {
        "model": result.provenance,
        "source": data.provenance,
    }
    lines.extend(
        [
            "## Provenance",
            "",
            "~~~text",
            repr(provenance),
            "~~~",
            "",
        ]
    )
    report = "\n".join(lines)
    if path is not None:
        Path(path).write_text(report, encoding="utf-8")
    return report
