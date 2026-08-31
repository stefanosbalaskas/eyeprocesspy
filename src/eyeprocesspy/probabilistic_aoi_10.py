"""Probabilistic AOI assignment and uncertainty propagation.

Source-ports the nine public contracts from frozen
``R/031-probabilistic-aoi.R`` in eyeprocess 0.11.1.

The deterministic AOI geometry, softmax membership, ambiguity, entropy,
separation audit, and fuzzy-transition mathematics follow the frozen R source.
Monte Carlo propagation uses NumPy's explicit seeded generator; statistical and
structural semantics are preserved, but the random-number stream is not
represented as byte-identical to R's ``set.seed()`` implementation.

Matplotlib is loaded lazily so the base eyeprocesspy wheel remains
dependency-light.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessBackendError, EyeProcessValidationError
from .irt import EyeResult

__all__ = [
    "assign_aois_probabilistic",
    "audit_aoi_separation",
    "summarise_aoi_membership",
    "propagate_aoi_uncertainty",
    "plot_aoi_probability_map",
    "plot_aoi_boundary_risk",
    "plot_probabilistic_scanpath",
    "plot_fuzzy_transition_matrix",
    "plot_aoi_metric_uncertainty",
]


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def _assert_data_frame(x: Any, *, name: str = "x", min_rows: int = 1) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise EyeProcessValidationError(f"`{name}` must be a data frame.")
    if len(x) < int(min_rows):
        raise EyeProcessValidationError(f"`{name}` must contain at least {int(min_rows)} row(s).")
    return x


def _require_class(x: Any, cls: str, *, name: str = "x") -> None:
    if getattr(x, "eyeprocess_class", None) != cls:
        raise EyeProcessValidationError(f"`{name}` must be an `{cls}` object.")


def _first_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    tried = ", ".join(map(str, candidates))
    raise EyeProcessValidationError(f"Could not identify {label}. Tried: {tried}.")


def _aoi_columns(aois: pd.DataFrame) -> dict[str, str]:
    return {
        "id": _first_column(
            aois,
            ("aoi_id", "aoi", "name", "label"),
            label="AOI identifier",
        ),
        "xmin": _first_column(
            aois,
            ("xmin", "x_min", "left"),
            label="AOI xmin",
        ),
        "xmax": _first_column(
            aois,
            ("xmax", "x_max", "right"),
            label="AOI xmax",
        ),
        "ymin": _first_column(
            aois,
            ("ymin", "y_min", "top"),
            label="AOI ymin",
        ),
        "ymax": _first_column(
            aois,
            ("ymax", "y_max", "bottom"),
            label="AOI ymax",
        ),
    }


def _numeric(value: Any) -> np.ndarray:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
    return pd.to_numeric(
        pd.Series(value),
        errors="coerce",
    ).to_numpy(dtype=float)


def _safe_mean(value: Any) -> float:
    array = _numeric(value)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else np.nan


def _safe_sd(value: Any) -> float:
    array = _numeric(value)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size >= 2 else np.nan


def _safe_quantile(value: Any, probability: float) -> float:
    array = _numeric(value)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan
    # R stats::quantile(type = 8) corresponds to NumPy's median_unbiased.
    return float(
        np.quantile(
            array,
            float(probability),
            method="median_unbiased",
        )
    )


def _entropy(probabilities: Any) -> float:
    p = np.asarray(probabilities, dtype=float).ravel()
    p = p[np.isfinite(p) & (p > 0)]
    if not p.size:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _softmax(logits: Any) -> np.ndarray:
    """Frozen ``.mi_softmax`` equivalent used by probabilistic AOI assignment."""
    matrix = np.asarray(logits, dtype=float)
    if matrix.ndim != 2:
        raise EyeProcessValidationError("`logits` must be a two-dimensional matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.empty(matrix.shape, dtype=float)

    row_max = np.zeros(matrix.shape[0], dtype=float)
    for index, row in enumerate(matrix):
        finite = row[np.isfinite(row)]
        row_max[index] = float(finite.max()) if finite.size else 0.0

    shifted = matrix - row_max[:, None]
    finite = np.isfinite(shifted)
    shifted = shifted.copy()
    shifted[finite & (shifted > 700)] = 700
    shifted[finite & (shifted < -700)] = -700

    with np.errstate(over="ignore", invalid="ignore"):
        values = np.exp(shifted)
        totals = np.nansum(values, axis=1)
    invalid_total = ~np.isfinite(totals) | (totals <= 0)
    totals[invalid_total] = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        return values / totals[:, None]


def _signed_rectangle_margin(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    inside_x = (x >= xmin) & (x <= xmax)
    inside_y = (y >= ymin) & (y <= ymax)
    inside = inside_x & inside_y

    dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
    dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
    outside_distance = np.sqrt(dx**2 + dy**2)

    inside_margin = np.minimum.reduce(
        [
            x - xmin,
            xmax - x,
            y - ymin,
            ymax - y,
        ]
    )
    return np.where(inside, inside_margin, -outside_distance)


def _resolve_two_scale(value: Any, fallback: float) -> np.ndarray:
    if value is None:
        return np.repeat(float(fallback), 2)

    values = _numeric(np.atleast_1d(value))
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return np.repeat(float(fallback), 2)
    if values.size == 1:
        return np.repeat(float(values[0]), 2)
    return values[:2].astype(float)


def _resolve_bias(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(2, dtype=float)

    values = _numeric(np.atleast_1d(value))
    values = values[np.isfinite(values)]
    if not values.size:
        return np.zeros(2, dtype=float)
    if values.size == 1:
        return np.repeat(float(values[0]), 2)
    return values[:2].astype(float)


def assign_aois_probabilistic(
    x,
    aois,
    error_model=("empirical", "gaussian", "ellipse"),
    accuracy=None,
    precision=None,
    x_col=None,
    y_col=None,
    id_cols=None,
):
    """Assign gaze samples to rectangular AOIs probabilistically."""
    x = _assert_data_frame(x, name="x")
    aois = _assert_data_frame(aois, name="aois")

    if isinstance(error_model, (tuple, list)):
        error_model = error_model[0] if error_model else "empirical"
    error_model = str(error_model)
    if error_model not in {"empirical", "gaussian", "ellipse"}:
        raise EyeProcessValidationError("`error_model` must be 'empirical', 'gaussian', or 'ellipse'.")

    x_col = x_col or _first_column(
        x,
        ("x", "gaze_x", "x_norm", "x_px"),
        label="gaze x coordinate",
    )
    y_col = y_col or _first_column(
        x,
        ("y", "gaze_y", "y_norm", "y_px"),
        label="gaze y coordinate",
    )
    missing = [column for column in (x_col, y_col) if column not in x.columns]
    if missing:
        raise EyeProcessValidationError("`x` is missing required column(s): " + ", ".join(missing) + ".")

    columns = _aoi_columns(aois)
    gx = _numeric(x[x_col])
    gy = _numeric(x[y_col])
    nonfinite = ~np.isfinite(gx) | ~np.isfinite(gy)
    if nonfinite.any():
        warnings.warn(
            "Non-finite gaze coordinates receive outside-AOI probability one.",
            RuntimeWarning,
            stacklevel=2,
        )

    xmin = _numeric(aois[columns["xmin"]])
    xmax = _numeric(aois[columns["xmax"]])
    ymin = _numeric(aois[columns["ymin"]])
    ymax = _numeric(aois[columns["ymax"]])

    widths = np.abs(xmax - xmin)
    heights = np.abs(ymax - ymin)
    finite_dimensions = np.concatenate([widths, heights])
    finite_dimensions = finite_dimensions[np.isfinite(finite_dimensions)]
    median_dimension = float(np.median(finite_dimensions)) if finite_dimensions.size else np.nan
    fallback = max(1e-6, median_dimension / 8.0)

    spread = _resolve_two_scale(precision, fallback)
    bias = _resolve_bias(accuracy)
    gx_adjusted = gx - bias[0]
    gy_adjusted = gy - bias[1]

    logits = np.full((len(x), len(aois) + 1), np.nan, dtype=float)
    aoi_names = aois[columns["id"]].astype("string").astype(str).tolist()

    for j in range(len(aois)):
        margin = _signed_rectangle_margin(
            gx_adjusted,
            gy_adjusted,
            float(xmin[j]),
            float(xmax[j]),
            float(ymin[j]),
            float(ymax[j]),
        )
        if error_model == "ellipse":
            local_scale = float(np.sqrt(spread[0] * spread[1]))
        else:
            local_scale = float(np.mean(spread))
        logits[:, j] = margin / max(local_scale, 1e-8)

    logits[:, -1] = 0.0
    probability_values = _softmax(logits)
    probability_values[nonfinite, :] = 0.0
    probability_values[nonfinite, -1] = 1.0

    probability_columns = aoi_names + ["outside"]
    probabilities = pd.DataFrame(
        probability_values,
        columns=probability_columns,
        index=x.index,
    )

    requested_ids = [] if id_cols is None else list(id_cols)
    retained: list[str] = []
    for column in requested_ids + [x_col, y_col]:
        if column in x.columns and column not in retained:
            retained.append(column)

    wide = pd.concat(
        [
            x.loc[:, retained].reset_index(drop=True),
            probabilities.reset_index(drop=True),
        ],
        axis=1,
    )

    sample_id = np.arange(1, len(x) + 1, dtype=int)
    k = len(probability_columns)
    membership = pd.DataFrame(
        {
            "sample_id": np.repeat(sample_id, k),
            "aoi": np.tile(np.asarray(probability_columns, dtype=object), len(x)),
            "probability": probability_values.reshape(-1),
        }
    )

    valid_id_cols = [column for column in requested_ids if column in x.columns]
    if valid_id_cols:
        repeated_ids = x.loc[:, valid_id_cols].iloc[np.repeat(np.arange(len(x)), k)].reset_index(drop=True)
        membership = pd.concat([repeated_ids, membership], axis=1)

    max_index = np.argmax(probability_values, axis=1)
    maximum = probability_values[np.arange(len(x)), max_index]
    classification = pd.DataFrame(
        {
            "sample_id": sample_id,
            "most_likely_aoi": [probability_columns[index] for index in max_index],
            "maximum_probability": maximum,
            "ambiguity": 1.0 - maximum,
            "membership_entropy": [_entropy(row) for row in probability_values],
        }
    )

    return _result(
        "eye_probabilistic_aoi",
        data=x.copy(),
        aois=aois.copy(),
        probabilities=probabilities,
        membership=membership,
        wide=wide,
        classification=classification,
        coordinate_columns={"x": x_col, "y": y_col},
        error_model=error_model,
        spread=spread,
        accuracy=bias,
        status="Probabilistic AOI membership estimated.",
    )


def audit_aoi_separation(x=None, aois=None):
    """Audit pairwise AOI overlap, touching boundaries, and separation."""
    probability_summary = None

    if getattr(x, "eyeprocess_class", None) == "eye_probabilistic_aoi":
        aois = x["aois"]
        classification = x["classification"]
        probability_summary = pd.DataFrame(
            [
                {
                    "mean_maximum_probability": _safe_mean(classification["maximum_probability"]),
                    "mean_ambiguity": _safe_mean(classification["ambiguity"]),
                    "mean_membership_entropy": _safe_mean(classification["membership_entropy"]),
                }
            ]
        )

    if aois is None:
        raise EyeProcessValidationError("Supply `aois` or an `eye_probabilistic_aoi` object.")

    aois = _assert_data_frame(aois, name="aois")
    columns = _aoi_columns(aois)
    ids = aois[columns["id"]].astype("string").astype(str).tolist()

    xmin = _numeric(aois[columns["xmin"]])
    xmax = _numeric(aois[columns["xmax"]])
    ymin = _numeric(aois[columns["ymin"]])
    ymax = _numeric(aois[columns["ymax"]])

    rows: list[dict[str, Any]] = []
    for i in range(len(aois) - 1):
        for j in range(i + 1, len(aois)):
            overlap_width = max(
                0.0,
                min(xmax[i], xmax[j]) - max(xmin[i], xmin[j]),
            )
            overlap_height = max(
                0.0,
                min(ymax[i], ymax[j]) - max(ymin[i], ymin[j]),
            )
            overlap_area = float(overlap_width * overlap_height)

            horizontal_gap = max(
                0.0,
                max(xmin[i], xmin[j]) - min(xmax[i], xmax[j]),
            )
            vertical_gap = max(
                0.0,
                max(ymin[i], ymin[j]) - min(ymax[i], ymax[j]),
            )
            gap = float(np.sqrt(horizontal_gap**2 + vertical_gap**2))

            if overlap_area > 0:
                status = "overlap"
            elif gap == 0:
                status = "touching"
            else:
                status = "separated"

            rows.append(
                {
                    "aoi_1": ids[i],
                    "aoi_2": ids[j],
                    "overlap_area": overlap_area,
                    "boundary_gap": gap,
                    "separation_status": status,
                }
            )

    pairwise = pd.DataFrame(
        rows,
        columns=[
            "aoi_1",
            "aoi_2",
            "overlap_area",
            "boundary_gap",
            "separation_status",
        ],
    )

    if len(pairwise):
        summary = pd.DataFrame(
            [
                {
                    "pairs": len(pairwise),
                    "overlapping_pairs": int(pairwise["separation_status"].eq("overlap").sum()),
                    "touching_pairs": int(pairwise["separation_status"].eq("touching").sum()),
                    "minimum_gap": float(pairwise["boundary_gap"].min()),
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "pairs": 0,
                    "overlapping_pairs": 0,
                    "touching_pairs": 0,
                    "minimum_gap": np.nan,
                }
            ]
        )

    return _result(
        "eye_aoi_separation_audit",
        pairwise=pairwise,
        probability_summary=probability_summary,
        summary=summary,
        status="AOI separation audit completed.",
    )


def summarise_aoi_membership(x, by=None):
    """Summarise mean AOI membership probabilities."""
    _require_class(x, "eye_probabilistic_aoi")
    membership = x["membership"].copy()

    if by is None:
        groups = ["aoi"]
    else:
        requested = [by] if isinstance(by, str) else list(by)
        valid = [column for column in requested if column in membership.columns]
        groups = valid + ["aoi"]

    return (
        membership.groupby(groups, as_index=False, sort=True)["probability"]
        .mean()
        .rename(columns={"probability": "mean_probability"})
    )


def _uncertain_aoi_metrics(
    labels: np.ndarray,
    data: pd.DataFrame,
    time_col: str | None,
    duration_col: str | None,
    aoi_levels: list[str],
) -> dict[str, float]:
    if time_col is not None and time_col in data.columns:
        time = _numeric(data[time_col])
    else:
        time = np.arange(1, len(labels) + 1, dtype=float)

    if duration_col is not None and duration_col in data.columns:
        duration = _numeric(data[duration_col])
    else:
        duration = np.ones(len(labels), dtype=float)

    aoi_names = [aoi for aoi in aoi_levels if aoi != "outside"]
    values: dict[str, float] = {}

    finite_time = time[np.isfinite(time)]
    overall_start = float(finite_time.min()) if finite_time.size else np.nan

    for aoi in aoi_names:
        mask = labels == aoi
        values[f"dwell__{aoi}"] = float(np.nansum(np.where(mask, duration, np.nan)))

        hit_times = time[mask & np.isfinite(time)]
        values[f"ttff__{aoi}"] = (
            float(hit_times.min() - overall_start) if hit_times.size and np.isfinite(overall_start) else np.nan
        )

    values["transitions"] = float(np.sum(labels[1:] != labels[:-1]) if len(labels) > 1 else 0)

    counts = np.asarray(
        [np.sum(labels == level) for level in aoi_levels],
        dtype=float,
    )
    proportions = counts / len(labels)
    values["entropy"] = _entropy(proportions)
    return values


def propagate_aoi_uncertainty(
    x,
    metrics=("dwell", "ttff", "transitions", "entropy"),
    draws=500,
    time_col=None,
    duration_col=None,
    seed=20260807,
):
    """Propagate probabilistic AOI membership to process metrics by Monte Carlo."""
    _require_class(x, "eye_probabilistic_aoi")

    try:
        draws = int(draws)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EyeProcessValidationError("`draws` must be at least 2.") from exc
    if draws < 2:
        raise EyeProcessValidationError("`draws` must be at least 2.")

    requested = [metrics] if isinstance(metrics, str) else list(metrics)
    supported = ["dwell", "ttff", "transitions", "entropy"]
    selected = [metric for metric in requested if metric in supported]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise EyeProcessValidationError("No supported metrics were selected.")

    probabilities = pd.DataFrame(x["probabilities"]).copy()
    probability_values = probabilities.to_numpy(dtype=float)
    labels = list(map(str, probabilities.columns))
    data = x["data"]

    rng = np.random.default_rng(seed)
    draw_rows: list[dict[str, float]] = []

    for draw in range(1, draws + 1):
        sampled = np.asarray(
            [rng.choice(labels, p=probability_values[row]) for row in range(len(probability_values))],
            dtype=object,
        )
        values = _uncertain_aoi_metrics(
            sampled,
            data,
            time_col,
            duration_col,
            labels,
        )

        retained: dict[str, float] = {"draw": draw}
        for name, value in values.items():
            keep = (
                ("dwell" in selected and name.startswith("dwell__"))
                or ("ttff" in selected and name.startswith("ttff__"))
                or ("transitions" in selected and name == "transitions")
                or ("entropy" in selected and name == "entropy")
            )
            if keep:
                retained[name] = value
        draw_rows.append(retained)

    draw_table = pd.DataFrame(draw_rows)
    metric_names = [column for column in draw_table.columns if column != "draw"]

    summary_rows = []
    for metric in metric_names:
        values = draw_table[metric]
        summary_rows.append(
            {
                "metric": metric,
                "mean": _safe_mean(values),
                "sd": _safe_sd(values),
                "lower": _safe_quantile(values, 0.025),
                "median": _safe_quantile(values, 0.5),
                "upper": _safe_quantile(values, 0.975),
            }
        )

    return _result(
        "eye_aoi_uncertainty",
        source=x,
        draws=draw_table,
        summary=pd.DataFrame(summary_rows),
        metrics=selected,
        seed=seed,
        status="AOI metric uncertainty propagated by Monte Carlo sampling.",
    )


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EyeProcessBackendError(
            "Probabilistic AOI plotting requires matplotlib. Install `eyeprocesspy[plots]` or the development extras."
        ) from exc
    return plt


def _axis(ax=None):
    if ax is not None:
        return ax
    return _get_plt().subplots()[1]


def _empty_plot(message: str, title: str, ax=None):
    axis = _axis(ax)
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.eyeprocess_plot_data = pd.DataFrame()
    return axis


def plot_aoi_probability_map(x, ax=None, **kwargs):
    """Plot gaze samples with AOI rectangles and membership certainty."""
    del kwargs
    _require_class(x, "eye_probabilistic_aoi")
    from matplotlib.patches import Rectangle

    coordinates = x["coordinate_columns"]
    gx = _numeric(x["data"][coordinates["x"]])
    gy = _numeric(x["data"][coordinates["y"]])
    classification = x["classification"]
    maximum = _numeric(classification["maximum_probability"])

    axis = _axis(ax)
    sizes = 20.0 * (0.5 + 1.5 * maximum) ** 2
    axis.scatter(gx, gy, s=sizes)

    aois = x["aois"]
    columns = _aoi_columns(aois)
    for row in aois.itertuples(index=False):
        record = row._asdict()
        xmin = float(pd.to_numeric(record[columns["xmin"]], errors="coerce"))
        xmax = float(pd.to_numeric(record[columns["xmax"]], errors="coerce"))
        ymin = float(pd.to_numeric(record[columns["ymin"]], errors="coerce"))
        ymax = float(pd.to_numeric(record[columns["ymax"]], errors="coerce"))
        label = str(record[columns["id"]])
        axis.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                linewidth=2,
            )
        )
        axis.text((xmin + xmax) / 2, (ymin + ymax) / 2, label, ha="center")

    axis.set(
        xlabel=coordinates["x"],
        ylabel=coordinates["y"],
        title="Probabilistic AOI assignment",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.eyeprocess_plot_data = classification.copy()
    return axis


def plot_aoi_boundary_risk(x, ax=None, **kwargs):
    """Plot mean assignment ambiguity or AOI-pair overlap area."""
    del kwargs
    cls = getattr(x, "eyeprocess_class", None)
    axis = _axis(ax)

    if cls == "eye_probabilistic_aoi":
        data = (
            x["classification"]
            .groupby(
                "most_likely_aoi",
                as_index=False,
                sort=True,
            )["ambiguity"]
            .mean()
        )
        axis.bar(
            np.arange(len(data)),
            data["ambiguity"].to_numpy(dtype=float),
        )
        axis.set_xticks(
            np.arange(len(data)),
            labels=data["most_likely_aoi"].astype(str).tolist(),
            rotation=90,
        )
        axis.set(
            ylabel="Mean assignment ambiguity",
            title="AOI boundary risk",
        )
    elif cls == "eye_aoi_separation_audit":
        data = x["pairwise"].copy()
        if data.empty:
            return _empty_plot(
                "No AOI pairs available.",
                "AOI separation audit",
                ax,
            )
        labels = data["aoi_1"].astype(str) + " - " + data["aoi_2"].astype(str)
        axis.bar(
            np.arange(len(data)),
            data["overlap_area"].to_numpy(dtype=float),
        )
        axis.set_xticks(
            np.arange(len(data)),
            labels=labels.tolist(),
            rotation=90,
        )
        axis.set(
            ylabel="Overlap area",
            title="AOI separation audit",
        )
    else:
        raise EyeProcessValidationError("`x` must be an `eye_probabilistic_aoi` or `eye_aoi_separation_audit` object.")

    axis.eyeprocess_plot_data = data
    return axis


def plot_probabilistic_scanpath(x, ax=None, **kwargs):
    """Plot the gaze trajectory underlying a probabilistic AOI fit."""
    del kwargs
    _require_class(x, "eye_probabilistic_aoi")

    coordinates = x["coordinate_columns"]
    data = x["data"]
    gx = _numeric(data[coordinates["x"]])
    gy = _numeric(data[coordinates["y"]])

    axis = _axis(ax)
    axis.plot(gx, gy, marker="o")
    axis.set(
        xlabel=coordinates["x"],
        ylabel=coordinates["y"],
        title="Probabilistic scanpath",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.eyeprocess_plot_data = data.copy()
    return axis


def plot_fuzzy_transition_matrix(x, ax=None, **kwargs):
    """Plot expected adjacent-state transitions under AOI uncertainty."""
    del kwargs
    _require_class(x, "eye_aoi_uncertainty")

    probabilities = pd.DataFrame(x["source"]["probabilities"]).copy()
    p = probabilities.to_numpy(dtype=float)
    transition = np.zeros(
        (probabilities.shape[1], probabilities.shape[1]),
        dtype=float,
    )
    for index in range(max(0, len(p) - 1)):
        transition += np.outer(p[index], p[index + 1])

    matrix = pd.DataFrame(
        transition,
        index=probabilities.columns,
        columns=probabilities.columns,
    )

    axis = _axis(ax)
    axis.imshow(transition, origin="lower", aspect="auto")
    ticks = np.arange(len(probabilities.columns))
    labels = list(map(str, probabilities.columns))
    axis.set_xticks(ticks, labels=labels, rotation=90)
    axis.set_yticks(ticks, labels=labels)
    axis.set(
        xlabel="From AOI",
        ylabel="To AOI",
        title="Fuzzy transition matrix",
    )
    axis.eyeprocess_plot_data = matrix
    axis.eyeprocess_plot_matrix = transition
    return axis


def plot_aoi_metric_uncertainty(x, ax=None, **kwargs):
    """Plot Monte Carlo distributions for propagated AOI process metrics."""
    del kwargs
    _require_class(x, "eye_aoi_uncertainty")

    draws = x["draws"].copy()
    metrics = [column for column in draws.columns if column != "draw"]
    if not metrics:
        return _empty_plot(
            "No propagated AOI metrics available.",
            "AOI metric uncertainty",
            ax,
        )

    values = [pd.to_numeric(draws[metric], errors="coerce").dropna().to_numpy(dtype=float) for metric in metrics]
    axis = _axis(ax)
    axis.boxplot(values, tick_labels=metrics)
    axis.tick_params(axis="x", rotation=90)
    axis.set(
        ylabel="Metric value",
        title="AOI metric uncertainty",
    )
    axis.eyeprocess_plot_data = draws
    return axis
