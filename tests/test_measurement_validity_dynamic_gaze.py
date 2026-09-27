from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep


def _validation() -> pd.DataFrame:
    rows = []
    for target_id, tx, ty in [("A", 0.0, 0.0), ("B", 1.0, 1.0), ("C", 2.0, 0.0)]:
        for k in range(5):
            pupil = 3.0 + 0.1 * k
            rows.append(
                {
                    "target_id": target_id,
                    "target_x": tx,
                    "target_y": ty,
                    "gaze_x": tx + 0.10 + 0.04 * (pupil - 3.2),
                    "gaze_y": ty - 0.05 - 0.02 * (pupil - 3.2),
                    "pupil": pupil,
                    "timestamp_seconds": float(len(rows)),
                    "valid": True,
                    "blink": k == 0,
                    "interpolated": k == 1,
                    "session": "s1" if target_id != "C" else "s2",
                }
            )
    return pd.DataFrame(rows)


def test_spatial_error_field_correction_and_plots():
    data = _validation()
    field = ep.compute_spatial_error_field(data, target_id="target_id", minimum_samples_per_target=3)
    assert field.eyeprocess_class == "eye_spatial_error_field"
    assert len(field.eligible_field) == 3

    anonymous = ep.compute_spatial_error_field(data.drop(columns="target_id"))
    assert len(anonymous.field) == 3

    corrected = ep.correct_gaze_with_spatial_error(data, field, method="idw")
    nearest = ep.correct_gaze_with_spatial_error(data, field, method="nearest", strength=0.5)
    assert "gaze_x_corrected" in corrected
    assert np.isfinite(nearest.gaze_y_corrected).all()

    missing = data.iloc[:2].copy()
    missing.loc[0, "gaze_x"] = np.nan
    corrected_missing = ep.correct_gaze_with_spatial_error(missing, field)
    assert np.isnan(corrected_missing.loc[0, "gaze_x_corrected"])

    ax = ep.plot_spatial_error_field(field)
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(ep.EyeProcessValidationError):
        ep.compute_spatial_error_field(data, minimum_samples_per_target=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.compute_spatial_error_field(data.assign(gaze_x=np.nan))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.compute_spatial_error_field(data.iloc[:2], target_id="target_id", minimum_samples_per_target=3)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.correct_gaze_with_spatial_error(data, {}, method="idw")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.correct_gaze_with_spatial_error(data, field, method="bad")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.correct_gaze_with_spatial_error(data, field, power=0)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.correct_gaze_with_spatial_error(data, field, strength=2)


def test_pupil_artifact_and_preprocessing_audit():
    data = _validation()
    model = ep.fit_pupil_size_artifact(data)
    grouped = ep.fit_pupil_size_artifact(data, by="session", minimum_samples=5)
    assert model.eyeprocess_class == "eye_pupil_size_artifact_model"
    assert (grouped.table.status == "estimated").all()

    corrected = ep.correct_pupil_size_artifact(data, model)
    assert "gaze_x_pupil_corrected" in corrected

    unknown_group = data.copy()
    unknown_group["session"] = "unknown"
    corrected_unknown = ep.correct_pupil_size_artifact(unknown_group, grouped, by="session")
    assert corrected_unknown.gaze_x_pupil_artifact.isna().all()

    audit = ep.audit_pupil_preprocessing(
        data,
        valid="valid",
        blink="blink",
        interpolated="interpolated",
        by="session",
    )
    assert audit.eyeprocess_class == "eye_pupil_preprocessing_audit"
    assert audit.table.valid_fraction.eq(1.0).all()

    reversed_time = data.copy()
    reversed_time.loc[2, "timestamp_seconds"] = -1
    warned = ep.audit_pupil_preprocessing(reversed_time)
    assert warned.warnings

    no_pupil = data.copy()
    no_pupil["pupil"] = np.nan
    no_pupil_audit = ep.audit_pupil_preprocessing(no_pupil)
    assert np.isnan(no_pupil_audit.table.median_pupil.iloc[0])

    ax = ep.plot_pupil_size_artifact(model)
    assert hasattr(ax, "eyeprocess_plot_data")

    with pytest.raises(ep.EyeProcessValidationError):
        ep.fit_pupil_size_artifact(data, minimum_samples=2)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.fit_pupil_size_artifact(data.iloc[:2], minimum_samples=3)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.correct_pupil_size_artifact(data, {})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_pupil_size_artifact({})
    with pytest.raises(ep.EyeProcessValidationError):
        ep.plot_spatial_error_field({})


def _dynamic_rectangles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"aoi": "target", "timestamp": 0.0, "shape": "rectangle", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            {"aoi": "target", "timestamp": 10.0, "shape": "rectangle", "x_min": 1.0, "x_max": 2.0, "y_min": 0.0, "y_max": 1.0},
            {"aoi": "overlap", "timestamp": 0.0, "shape": "rectangle", "x_min": 0.4, "x_max": 0.8, "y_min": 0.4, "y_max": 0.8},
            {"aoi": "overlap", "timestamp": 10.0, "shape": "rectangle", "x_min": 1.4, "x_max": 1.8, "y_min": 0.4, "y_max": 0.8},
        ]
    )


def test_dynamic_aoi_assignment_shapes_sensitivity_and_guards():
    rectangles = _dynamic_rectangles()
    spec = ep.validate_dynamic_aoi_spec(rectangles)
    samples = pd.DataFrame(
        {
            "timestamp": [0.0, 5.0, 5.0, 20.0, np.nan],
            "gaze_x": [0.2, 1.0, 1.1, 2.0, 0.5],
            "gaze_y": [0.2, 0.5, 2.0, 0.5, 0.5],
        }
    )
    result = ep.assign_dynamic_aoi(samples, spec, interpolation="linear", overlap="ambiguous")
    assert set(result.assignments.status) >= {"assigned", "outside", "no_active_geometry", "missing_sample"}
    first = ep.assign_dynamic_aoi(samples.iloc[:2], rectangles, interpolation="step", overlap="first", lag_tolerance=1)
    assert len(first.assignments) == 2
    all_hits = ep.assign_dynamic_aoi(
        pd.DataFrame({"timestamp": [5.0], "gaze_x": [1.0], "gaze_y": [0.6]}),
        rectangles,
        overlap="all",
    )
    assert all_hits.assignments.status.iloc[0] in {"assigned", "overlap_all"}
    coverage = ep.audit_dynamic_aoi_coverage(result)
    assert np.isclose(coverage.fraction.sum(), 1)
    sensitivity = ep.dynamic_aoi_sensitivity(samples.iloc[:3], rectangles, lag_tolerances=(0, 1))
    assert len(sensitivity) == 4
    ax = ep.plot_dynamic_aoi_alignment(result)
    assert hasattr(ax, "eyeprocess_plot_data")

    polygon = pd.DataFrame(
        [
            {"aoi": "poly", "timestamp": 0.0, "shape": "polygon", "vertices": [(0, 0), (1, 0), (0, 1)]},
            {"aoi": "poly", "timestamp": 10.0, "shape": "polygon", "vertices": [(1, 0), (2, 0), (1, 1)]},
        ]
    )
    poly_result = ep.assign_dynamic_aoi(
        pd.DataFrame({"timestamp": [5.0], "gaze_x": [0.7], "gaze_y": [0.2]}),
        polygon,
    )
    assert poly_result.assignments.n_hits.iloc[0] == 1

    mask = pd.DataFrame(
        [
            {"aoi": "mask", "timestamp": 0.0, "shape": "mask", "mask": np.ones((2, 2), bool), "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
            {"aoi": "mask", "timestamp": 10.0, "shape": "mask", "mask": np.ones((2, 2), bool), "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        ]
    )
    assert ep.assign_dynamic_aoi(
        pd.DataFrame({"timestamp": [1.0], "gaze_x": [0.5], "gaze_y": [0.5]}),
        mask,
        interpolation="step",
    ).assignments.aoi.iloc[0] == "mask"
    with pytest.raises(ep.EyeProcessValidationError):
        ep.assign_dynamic_aoi(
            pd.DataFrame({"timestamp": [5.0], "gaze_x": [0.5], "gaze_y": [0.5]}),
            mask,
            interpolation="linear",
        )

    with pytest.raises(ep.EyeProcessValidationError):
        ep.validate_dynamic_aoi_spec(rectangles.assign(shape="bad"))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.validate_dynamic_aoi_spec(rectangles.assign(timestamp=np.nan))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.validate_dynamic_aoi_spec(pd.concat([rectangles.iloc[[0]], rectangles.iloc[[0]]], ignore_index=True))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.validate_dynamic_aoi_spec(rectangles.assign(x_min=2, x_max=1))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.validate_dynamic_aoi_spec(pd.DataFrame([{"aoi":"p","timestamp":0,"shape":"polygon","vertices":"bad"}]))
    with pytest.raises(ep.EyeProcessValidationError):
        ep.assign_dynamic_aoi(samples, rectangles, interpolation="bad")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.assign_dynamic_aoi(samples, rectangles, overlap="bad")
    with pytest.raises(ep.EyeProcessValidationError):
        ep.assign_dynamic_aoi(samples, rectangles, lag_tolerance=-1)
    with pytest.raises(ep.EyeProcessValidationError):
        ep.audit_dynamic_aoi_coverage({})
