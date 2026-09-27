"""Scanpath metric comparison and representation-sensitivity analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _sequence(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(">") if part.strip()]
    try:
        return [str(v) for v in value if not pd.isna(v)]
    except TypeError as exc:
        raise EyeProcessValidationError("Scanpath must be a string or sequence.") from exc


def levenshtein_scanpath(a: Any, b: Any, *, normalize: bool = True) -> float:
    """Compute edit distance between AOI sequences."""
    left, right = _sequence(a), _sequence(b)
    previous = np.arange(len(right) + 1, dtype=float)
    for i, token in enumerate(left, start=1):
        current = np.empty(len(right) + 1, dtype=float)
        current[0] = i
        for j, other in enumerate(right, start=1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (token != other))
        previous = current
    distance = float(previous[-1])
    denominator = max(len(left), len(right))
    return distance / denominator if normalize and denominator else distance


def _transitions(sequence: list[str]) -> dict[tuple[str, str], float]:
    pairs = list(zip(sequence[:-1], sequence[1:], strict=True))
    counts: dict[tuple[str, str], float] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0.0) + 1.0
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def transition_js_distance(a: Any, b: Any) -> float:
    """Jensen-Shannon distance between AOI transition distributions."""
    left, right = _transitions(_sequence(a)), _transitions(_sequence(b))
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.asarray([left.get(k, 0.0) for k in keys], float)
    q = np.asarray([right.get(k, 0.0) for k in keys], float)
    return float(jensenshannon(p, q, base=2.0))


def ngram_jaccard_similarity(a: Any, b: Any, *, n: int = 2) -> float:
    """Jaccard similarity for ordered AOI n-grams."""
    n = int(n)
    if n < 1:
        raise EyeProcessValidationError("n must be positive.")
    def grams(seq: list[str]) -> set[tuple[str, ...]]:
        return {tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1))}
    ga, gb = grams(_sequence(a)), grams(_sequence(b))
    union = ga | gb
    return len(ga & gb) / len(union) if union else 1.0


def dynamic_time_warping_distance(
    reference: Any,
    candidate: Any,
    *,
    columns: tuple[str, ...] = ("x", "y"),
    normalize: bool = True,
) -> float:
    """Dynamic-time-warping distance for geometric/temporal scanpath representations."""
    a, b = pd.DataFrame(reference).copy(), pd.DataFrame(candidate).copy()
    missing = [c for c in columns if c not in a or c not in b]
    if missing:
        raise EyeProcessValidationError(f"Trajectory data are missing: {', '.join(missing)}.")
    aa = a[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bb = b[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all() or not len(aa) or not len(bb):
        raise EyeProcessValidationError("Trajectory values must be complete, finite, and non-empty.")
    cost = np.full((len(aa) + 1, len(bb) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(aa) + 1):
        for j in range(1, len(bb) + 1):
            local = float(np.linalg.norm(aa[i - 1] - bb[j - 1]))
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    value = float(cost[-1, -1])
    return value / (len(aa) + len(bb)) if normalize else value


def compare_scanpath_metrics(
    reference: Any,
    candidate: Any,
    *,
    reference_trajectory: Any | None = None,
    candidate_trajectory: Any | None = None,
    trajectory_columns: tuple[str, ...] = ("x", "y"),
) -> pd.DataFrame:
    """Return complementary sequence, transition, n-gram, and optional trajectory metrics."""
    rows = [
        {"metric": "normalized_levenshtein", "value": levenshtein_scanpath(reference, candidate), "higher_is_more_similar": False},
        {"metric": "transition_js_distance", "value": transition_js_distance(reference, candidate), "higher_is_more_similar": False},
        {"metric": "bigram_jaccard", "value": ngram_jaccard_similarity(reference, candidate, n=2), "higher_is_more_similar": True},
    ]
    if reference_trajectory is not None or candidate_trajectory is not None:
        if reference_trajectory is None or candidate_trajectory is None:
            raise EyeProcessValidationError("Both trajectories must be supplied together.")
        rows.append({
            "metric": "trajectory_dtw",
            "value": dynamic_time_warping_distance(reference_trajectory, candidate_trajectory, columns=trajectory_columns),
            "higher_is_more_similar": False,
        })
    return pd.DataFrame(rows)


def scanpath_metric_sensitivity(
    reference: Any,
    candidates: Mapping[str, Any],
) -> EyeResult:
    """Show how candidate ordering changes across defensible scanpath metrics."""
    if not candidates:
        raise EyeProcessValidationError("candidates must contain at least one named scanpath.")
    rows = []
    for name, candidate in candidates.items():
        table = compare_scanpath_metrics(reference, candidate)
        for _, row in table.iterrows():
            rows.append({"candidate": str(name), **row.to_dict()})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("metric", sort=False)["value"].rank(
        ascending=out.groupby("metric", sort=False)["higher_is_more_similar"].transform("first").map({True: False, False: True}),
        method="min",
    )
    return EyeResult(
        {
            "table": out,
            "caveat": "No scanpath metric is treated as universally privileged; metric dependence is an analysis result.",
        },
        eyeprocess_class="eye_scanpath_metric_sensitivity",
    )


def plot_scanpath_sensitivity(result: Any, ax: Any = None) -> Any:
    """Plot candidate metric ranks to expose metric-dependent conclusions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(result, "eyeprocess_class", None) != "eye_scanpath_metric_sensitivity":
        raise EyeProcessValidationError("result must be an eye_scanpath_metric_sensitivity.")
    table = result["table"]
    pivot = table.pivot(index="candidate", columns="metric", values="rank")
    axis = plt.subplots()[1] if ax is None else ax
    for column in pivot:
        axis.plot(pivot.index.astype(str), pivot[column], marker="o", label=column)
    axis.invert_yaxis()
    axis.set_ylabel("Rank (1 = most similar)")
    axis.set_title("Scanpath metric sensitivity")
    axis.legend()
    axis.eyeprocess_plot_data = table.copy()
    return axis


__all__ = [
    "compare_scanpath_metrics",
    "dynamic_time_warping_distance",
    "levenshtein_scanpath",
    "ngram_jaccard_similarity",
    "plot_scanpath_sensitivity",
    "scanpath_metric_sensitivity",
    "transition_js_distance",
]
