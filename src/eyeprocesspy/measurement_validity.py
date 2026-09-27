"""Measurement-validity extensions for eye-tracking data.

The functions in this module complement the existing calibration-drift and
probabilistic-AOI APIs. Corrections are never applied implicitly: raw values are
preserved and every correction requires explicit target-referenced evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .irt import EyeResult


def _frame(value: Any, name: str = "data") -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _require(data: pd.DataFrame, columns: Sequence[str], name: str = "data") -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise EyeProcessValidationError(
            f"{name} is missing required column(s): {', '.join(missing)}."
        )


def _numeric(series: pd.Series) -> np.ndarray:
    return np.asarray(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float), dtype=float)


def _result(cls: str, **kwargs: Any) -> EyeResult:
    return EyeResult(kwargs, eyeprocess_class=cls)


def compute_spatial_error_field(
    validation: Any,
    *,
    gaze_x: str = "gaze_x",
    gaze_y: str = "gaze_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
    target_id: str | None = None,
    minimum_samples_per_target: int = 3,
) -> EyeResult:
    """Estimate target-local systematic gaze error without inventing missing targets."""
    data = _frame(validation, "validation")
    required = [gaze_x, gaze_y, target_x, target_y]
    if target_id is not None:
        required.append(target_id)
    _require(data, required, "validation")
    minimum_samples_per_target = int(minimum_samples_per_target)
    if minimum_samples_per_target < 1:
        raise EyeProcessValidationError("minimum_samples_per_target must be positive.")

    gx, gy = _numeric(data[gaze_x]), _numeric(data[gaze_y])
    tx, ty = _numeric(data[target_x]), _numeric(data[target_y])
    ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(tx) & np.isfinite(ty)
    if not ok.any():
        raise EyeProcessValidationError("No complete target-referenced validation samples.")

    work = pd.DataFrame(
        {
            "target_x": tx[ok],
            "target_y": ty[ok],
            "error_x": gx[ok] - tx[ok],
            "error_y": gy[ok] - ty[ok],
        }
    )
    if target_id is None:
        work["target_id"] = [
            f"{x:.12g}|{y:.12g}" for x, y in zip(work.target_x, work.target_y, strict=True)
        ]
    else:
        work["target_id"] = data.loc[ok, target_id].astype(str).to_numpy()

    rows: list[dict[str, Any]] = []
    for key, group in work.groupby("target_id", sort=False, dropna=False):
        ex = group.error_x.to_numpy(float)
        ey = group.error_y.to_numpy(float)
        radial = np.hypot(ex, ey)
        rows.append(
            {
                "target_id": key,
                "target_x": float(group.target_x.mean()),
                "target_y": float(group.target_y.mean()),
                "n": int(len(group)),
                "bias_x": float(ex.mean()),
                "bias_y": float(ey.mean()),
                "mean_radial_error": float(radial.mean()),
                "rms_radial_error": float(np.sqrt(np.mean(radial**2))),
                "eligible": bool(len(group) >= minimum_samples_per_target),
            }
        )
    field = pd.DataFrame(rows)
    eligible = field.loc[field.eligible].copy()
    if eligible.empty:
        raise EyeProcessValidationError(
            "No target has enough complete samples for a spatial error field."
        )
    return _result(
        "eye_spatial_error_field",
        field=field,
        eligible_field=eligible,
        sample_errors=work,
        minimum_samples_per_target=minimum_samples_per_target,
        coordinate_units="input_coordinate_units",
        caveat=(
            "This field estimates systematic error at observed validation targets. "
            "Interpolation away from those targets is an explicit modelling assumption."
        ),
    )


def _idw_bias(x: float, y: float, field: pd.DataFrame, power: float) -> tuple[float, float]:
    distances = np.hypot(field.target_x.to_numpy(float) - x, field.target_y.to_numpy(float) - y)
    exact = np.flatnonzero(distances == 0)
    if exact.size:
        row = field.iloc[int(exact[0])]
        return float(row.bias_x), float(row.bias_y)
    weights = 1.0 / np.power(distances, power)
    weights = weights / weights.sum()
    return (
        float(np.sum(weights * field.bias_x.to_numpy(float))),
        float(np.sum(weights * field.bias_y.to_numpy(float))),
    )


def correct_gaze_with_spatial_error(
    data: Any,
    field: Any,
    *,
    x: str = "gaze_x",
    y: str = "gaze_y",
    method: str = "idw",
    power: float = 2.0,
    strength: float = 1.0,
    suffix: str = "_corrected",
) -> pd.DataFrame:
    """Apply an explicit spatial-bias correction while preserving raw coordinates."""
    frame = _frame(data)
    _require(frame, [x, y])
    if not isinstance(field, Mapping) or getattr(field, "eyeprocess_class", None) != "eye_spatial_error_field":
        raise EyeProcessValidationError("field must be an eye_spatial_error_field.")
    method = str(method).lower()
    if method not in {"idw", "nearest"}:
        raise EyeProcessValidationError("method must be 'idw' or 'nearest'.")
    power, strength = float(power), float(strength)
    if power <= 0 or not np.isfinite(power):
        raise EyeProcessValidationError("power must be finite and positive.")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise EyeProcessValidationError("strength must lie in [0, 1].")

    spatial = field["eligible_field"]
    gx, gy = _numeric(frame[x]), _numeric(frame[y])
    bx = np.full(len(frame), np.nan)
    by = np.full(len(frame), np.nan)
    for i, (xx, yy) in enumerate(zip(gx, gy, strict=True)):
        if not np.isfinite(xx) or not np.isfinite(yy):
            continue
        if method == "nearest":
            distances = np.hypot(
                spatial.target_x.to_numpy(float) - xx,
                spatial.target_y.to_numpy(float) - yy,
            )
            row = spatial.iloc[int(np.argmin(distances))]
            bx[i], by[i] = float(row.bias_x), float(row.bias_y)
        else:
            bx[i], by[i] = _idw_bias(xx, yy, spatial, power)

    out = frame.copy()
    out[f"{x}_correction"] = strength * bx
    out[f"{y}_correction"] = strength * by
    out[f"{x}{suffix}"] = gx - strength * bx
    out[f"{y}{suffix}"] = gy - strength * by
    out.attrs["eyeprocess_class"] = "eye_spatial_error_corrected"
    out.attrs["eyeprocess_correction"] = {
        "method": method,
        "power": power,
        "strength": strength,
        "source": "target_referenced_spatial_error_field",
    }
    return out


def fit_pupil_size_artifact(
    validation: Any,
    *,
    pupil: str = "pupil",
    gaze_x: str = "gaze_x",
    gaze_y: str = "gaze_y",
    target_x: str = "target_x",
    target_y: str = "target_y",
    by: str | None = None,
    minimum_samples: int = 8,
) -> EyeResult:
    """Estimate pupil-size-related apparent gaze displacement from fixed-target data."""
    data = _frame(validation, "validation")
    required = [pupil, gaze_x, gaze_y, target_x, target_y] + ([] if by is None else [by])
    _require(data, required, "validation")
    minimum_samples = int(minimum_samples)
    if minimum_samples < 3:
        raise EyeProcessValidationError("minimum_samples must be at least 3.")

    rows: list[dict[str, Any]] = []
    grouped = [(None, data)] if by is None else list(data.groupby(by, sort=False, dropna=False))
    for key, group in grouped:
        p = _numeric(group[pupil])
        ex = _numeric(group[gaze_x]) - _numeric(group[target_x])
        ey = _numeric(group[gaze_y]) - _numeric(group[target_y])
        ok = np.isfinite(p) & np.isfinite(ex) & np.isfinite(ey)
        if int(ok.sum()) < minimum_samples:
            rows.append({"group": key, "n": int(ok.sum()), "status": "insufficient_data"})
            continue
        center = float(np.mean(p[ok]))
        pc = p[ok] - center
        design = np.column_stack([np.ones(ok.sum()), pc])
        bx = np.linalg.lstsq(design, ex[ok], rcond=None)[0]
        by_fit = np.linalg.lstsq(design, ey[ok], rcond=None)[0]
        rows.append(
            {
                "group": key,
                "n": int(ok.sum()),
                "pupil_center": center,
                "intercept_x": float(bx[0]),
                "slope_x": float(bx[1]),
                "intercept_y": float(by_fit[0]),
                "slope_y": float(by_fit[1]),
                "status": "estimated",
            }
        )
    model = pd.DataFrame(rows)
    if not (model.status == "estimated").any():
        raise EyeProcessValidationError("No group has enough target-referenced samples.")
    return _result(
        "eye_pupil_size_artifact_model",
        table=model,
        pupil=pupil,
        by=by,
        caveat=(
            "The regression is target-referenced. Applying it to free viewing assumes the "
            "estimated pupil-related displacement transfers to the analysed recording."
        ),
    )


def correct_pupil_size_artifact(
    data: Any,
    model: Any,
    *,
    pupil: str | None = None,
    gaze_x: str = "gaze_x",
    gaze_y: str = "gaze_y",
    by: str | None = None,
    suffix: str = "_pupil_corrected",
) -> pd.DataFrame:
    """Apply an explicit pupil-size artefact model without overwriting raw gaze."""
    frame = _frame(data)
    if not isinstance(model, Mapping) or getattr(model, "eyeprocess_class", None) != "eye_pupil_size_artifact_model":
        raise EyeProcessValidationError("model must be an eye_pupil_size_artifact_model.")
    pupil = str(model["pupil"] if pupil is None else pupil)
    by = model["by"] if by is None else by
    required = [pupil, gaze_x, gaze_y] + ([] if by is None else [by])
    _require(frame, required)
    table = model["table"].loc[lambda z: z.status == "estimated"].copy()
    lookup = {row.group: row for _, row in table.iterrows()}

    out = frame.copy()
    correction_x = np.full(len(out), np.nan)
    correction_y = np.full(len(out), np.nan)
    for i, row in out.reset_index(drop=True).iterrows():
        key = None if by is None else row[by]
        fit = lookup.get(key)
        pv = pd.to_numeric(pd.Series([row[pupil]]), errors="coerce").iloc[0]
        if fit is None or not np.isfinite(pv):
            continue
        correction_x[i] = float(fit.slope_x) * (float(pv) - float(fit.pupil_center))
        correction_y[i] = float(fit.slope_y) * (float(pv) - float(fit.pupil_center))
    gx, gy = _numeric(out[gaze_x]), _numeric(out[gaze_y])
    out[f"{gaze_x}_pupil_artifact"] = correction_x
    out[f"{gaze_y}_pupil_artifact"] = correction_y
    out[f"{gaze_x}{suffix}"] = gx - correction_x
    out[f"{gaze_y}{suffix}"] = gy - correction_y
    out.attrs["eyeprocess_class"] = "eye_pupil_size_artifact_corrected"
    return out


def audit_pupil_preprocessing(
    data: Any,
    *,
    pupil: str = "pupil",
    time: str = "timestamp_seconds",
    valid: str | None = None,
    blink: str | None = None,
    interpolated: str | None = None,
    by: str | None = None,
) -> EyeResult:
    """Produce a transparent pupil preprocessing/QC ledger."""
    frame = _frame(data)
    required = [pupil, time] + [v for v in (valid, blink, interpolated, by) if v is not None]
    _require(frame, required)
    groups = [(None, frame)] if by is None else list(frame.groupby(by, sort=False, dropna=False))
    rows = []
    warnings = []
    for key, group in groups:
        p = _numeric(group[pupil])
        t = _numeric(group[time])
        finite = np.isfinite(p)
        valid_mask = finite if valid is None else finite & group[valid].astype("boolean").fillna(False).to_numpy(bool)
        blink_mask = np.zeros(len(group), bool) if blink is None else group[blink].astype("boolean").fillna(False).to_numpy(bool)
        interp_mask = np.zeros(len(group), bool) if interpolated is None else group[interpolated].astype("boolean").fillna(False).to_numpy(bool)
        monotonic = bool(np.all(np.diff(t[np.isfinite(t)]) > 0)) if np.isfinite(t).sum() > 1 else True
        if not monotonic:
            warnings.append(f"Non-monotonic time detected for group {key!r}.")
        rows.append(
            {
                "group": key,
                "n_samples": int(len(group)),
                "finite_pupil_fraction": float(finite.mean()) if len(group) else np.nan,
                "valid_fraction": float(valid_mask.mean()) if len(group) else np.nan,
                "blink_fraction": float(blink_mask.mean()) if len(group) else np.nan,
                "interpolated_fraction": float(interp_mask.mean()) if len(group) else np.nan,
                "median_pupil": float(np.nanmedian(p)) if finite.any() else np.nan,
                "time_strictly_increasing": monotonic,
            }
        )
    return _result(
        "eye_pupil_preprocessing_audit",
        table=pd.DataFrame(rows),
        warnings=tuple(warnings),
        caveat=(
            "This audit reports preprocessing consequences. It does not choose blink, "
            "interpolation, baseline, filtering, or exclusion rules for the analyst."
        ),
    )


def plot_spatial_error_field(field: Any, ax: Any = None) -> Any:
    """Plot target-local bias vectors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(field, "eyeprocess_class", None) != "eye_spatial_error_field":
        raise EyeProcessValidationError("field must be an eye_spatial_error_field.")
    axis = plt.subplots()[1] if ax is None else ax
    table = field["eligible_field"]
    axis.scatter(table.target_x, table.target_y)
    axis.quiver(table.target_x, table.target_y, table.bias_x, table.bias_y, angles="xy", scale_units="xy", scale=1)
    axis.set_title("Spatial calibration error field")
    axis.set_xlabel("Target x")
    axis.set_ylabel("Target y")
    setattr(axis, "eyeprocess_plot_data", table.copy())
    return axis


def plot_pupil_size_artifact(model: Any, ax: Any = None) -> Any:
    """Plot pupil-artifact slope vectors by fitted group."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install eyeprocesspy[plots] for plotting.") from exc
    if getattr(model, "eyeprocess_class", None) != "eye_pupil_size_artifact_model":
        raise EyeProcessValidationError("model must be an eye_pupil_size_artifact_model.")
    axis = plt.subplots()[1] if ax is None else ax
    table = model["table"].loc[lambda z: z.status == "estimated"].reset_index(drop=True)
    axis.scatter(table.slope_x, table.slope_y)
    for i, row in table.iterrows():
        axis.annotate(str(row.group), (row.slope_x, row.slope_y))
    axis.axhline(0, linewidth=1)
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("Horizontal error / pupil unit")
    axis.set_ylabel("Vertical error / pupil unit")
    axis.set_title("Pupil-size artefact")
    setattr(axis, "eyeprocess_plot_data", table.copy())
    return axis


__all__ = [
    "audit_pupil_preprocessing",
    "compute_spatial_error_field",
    "correct_gaze_with_spatial_error",
    "correct_pupil_size_artifact",
    "fit_pupil_size_artifact",
    "plot_pupil_size_artifact",
    "plot_spatial_error_field",
]
