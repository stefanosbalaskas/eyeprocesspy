from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.semantic_validation_07 as sv


def _close(ax):
    if ax is not None and hasattr(ax, "figure"):
        plt.close(ax.figure)


def test_private_dataframe_correlation_rank_and_alignment_residual_paths():
    frame = pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]})
    assert sv._df({"samples": frame}, "x").equals(frame)
    assert sv._df({"samples": None, "data": frame}, "x").equals(frame)
    assert sv._df({"samples": None, "data": None, "gaze": frame}, "x").equals(frame)

    assert math.isnan(sv._safe_cor(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert math.isnan(sv._safe_cor(np.ones(3), np.arange(3.0)))
    assert sv._safe_cor(np.arange(3.0), np.arange(3.0)) == pytest.approx(1.0)
    assert math.isnan(sv._rank("NOT-A-FIDELITY-LEVEL"))

    source = pd.DataFrame({"id": [1, 2, 3], "block": ["a", "a", "b"], "x": [1, 2, 3]})
    reordered = source.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(ep.EyeProcessValidationError, match="allow_row_reorder"):
        sv._align(source, reordered, "id", allow_row_reorder=False)

    dup = pd.DataFrame({"id": [1, 1, 2], "x": [1, 2, 3]})
    with pytest.raises(ep.EyeProcessValidationError, match="uniquely identify"):
        sv._align(dup, dup.copy(), "id")

    partial = pd.DataFrame({"id": [2, 3, 4], "block": ["a", "b", "c"], "x": [2, 3, 4]})
    aligned = sv._align(source, partial, ["id", "block"])
    assert aligned["matched_n"] == 2
    assert aligned["alignment"] == "id+block"


def test_semantic_spec_and_field_report_missing_unsupported_and_ambiguous_paths():
    with pytest.raises(ep.EyeProcessValidationError, match="finite and non-negative"):
        ep.semantic_fidelity_spec(timestamp_tolerance=np.nan)
    with pytest.raises(ep.EyeProcessValidationError, match="finite and non-negative"):
        ep.semantic_fidelity_spec(coordinate_tolerance=-1)
    with pytest.raises(ep.EyeProcessValidationError, match="correlation_floor"):
        ep.semantic_fidelity_spec(correlation_floor=2)

    source = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "num": [np.nan, np.nan, np.nan],
            "text": ["a", "b", None],
        }
    )
    target = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "other": [10, 20, 30],
            "text": ["a", "x", None],
        }
    )
    report = ep.field_fidelity_report(
        source,
        target,
        fields=["absent_source", "num", "text"],
        mapping={"num": "absent_target", "text": "text"},
        key="id",
    )
    statuses = dict(zip(report.fields["field"], report.fields["status"]))
    assert statuses["absent_source"] == "MISSING"
    assert statuses["num"] == "UNSUPPORTED"
    assert statuses["text"] == "AMBIGUOUS"

    nan_target = pd.DataFrame({"id": [1, 2, 3], "num": [np.nan, np.nan, np.nan]})
    ambiguous = ep.field_fidelity_report(source[["id", "num"]], nan_target, fields="num", key="id")
    assert ambiguous.fields.loc[0, "status"] == "AMBIGUOUS"


def test_timestamp_fidelity_units_empty_offset_slope_and_ambiguous_paths():
    d = pd.DataFrame({"timestamp": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ep.EyeProcessValidationError, match="Timestamp units"):
        ep.timestamp_fidelity_audit(d, d, source_unit="minutes")

    empty = pd.DataFrame({"timestamp": [np.nan, np.nan]})
    out = ep.timestamp_fidelity_audit(empty, empty)
    assert out.status == "AMBIGUOUS"
    assert out.matched_n == 0
    assert math.isnan(out.offset_seconds)

    shifted = pd.DataFrame({"timestamp": d.timestamp + 0.25})
    out = ep.timestamp_fidelity_audit(d, shifted, tolerance=1e-12)
    assert out.status == "SEMANTICALLY_EQUIVALENT"
    assert out.offset_seconds == pytest.approx(0.25)

    a = pd.DataFrame({"timestamp": [-1.0, 0.0, 1.0]})
    b = pd.DataFrame({"timestamp": [-0.9, 0.2, 1.1]})
    out = ep.timestamp_fidelity_audit(a, b, tolerance=1e-12)
    assert out.status == "SEMANTICALLY_EQUIVALENT"
    assert out.affine_slope == pytest.approx(1.0)

    descending = pd.DataFrame({"timestamp": [3.0, 2.0, 0.0, -2.0]})
    out = ep.timestamp_fidelity_audit(d, descending, tolerance=1e-12)
    assert out.status == "AMBIGUOUS"
    assert out.roundtrip_monotonic is False


def test_coordinate_fidelity_empty_short_transformed_and_ambiguous_paths():
    empty = pd.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]})
    out = ep.coordinate_fidelity_audit(empty, empty)
    assert out.status == "AMBIGUOUS"
    assert math.isnan(out.x["error"])

    short_a = pd.DataFrame({"x": [1.0], "y": [2.0]})
    short_b = pd.DataFrame({"x": [2.0], "y": [4.0]})
    out = ep.coordinate_fidelity_audit(short_a, short_b)
    assert out.status == "AMBIGUOUS"
    assert math.isnan(out.x["slope"])

    source = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0, 4.0]})
    transformed = pd.DataFrame({"x": source.x * 2 + 1, "y": source.y * 3 - 2})
    out = ep.coordinate_fidelity_audit(source, transformed, tolerance=1e-10)
    assert out.status == "COORDINATE_TRANSFORMED"

    scrambled = pd.DataFrame({"x": [0.0, 5.0, 1.0, 9.0], "y": [9.0, 1.0, 5.0, 0.0]})
    assert ep.coordinate_fidelity_audit(source, scrambled, tolerance=1e-12).status == "AMBIGUOUS"


def test_pupil_eye_and_event_semantic_residual_paths():
    src = pd.DataFrame({"pupil_size": [1.0, 2.0, 3.0, 4.0]})
    scaled = pd.DataFrame({"pupil_size": [2.0, 4.0, 6.0, 8.0]})
    out = ep.pupil_unit_fidelity_audit(src, scaled, tolerance=1e-10)
    assert out.status == "UNIT_TRANSFORMED"
    assert out.estimated_scale_ratio == pytest.approx(2.0)

    zeros = pd.DataFrame({"pupil_size": [0.0, 0.0, np.nan]})
    assert ep.pupil_unit_fidelity_audit(zeros, zeros).status == "LOSSLESS"

    eye_a = pd.DataFrame({"eye": ["L", "right_eye", "both", None]})
    eye_b = pd.DataFrame({"eye": ["left", "R", "cyclopean", None]})
    assert ep.eye_stream_fidelity_audit(eye_a, eye_b).status == "LOSSLESS"
    eye_b.loc[1, "eye"] = "left"
    assert ep.eye_stream_fidelity_audit(eye_a, eye_b).status == "AMBIGUOUS"

    events = pd.DataFrame({"event": ["a", "b"], "timestamp": [0.0, 1.0]})
    assert ep.event_semantics_audit(events, events, time=None).status == "LOSSLESS"
    longer = pd.DataFrame({"event": ["a", "b", "c"], "timestamp": [0.0, 1.0, 2.0]})
    assert ep.event_semantics_audit(longer, events, time=None).status == "SEMANTICALLY_EQUIVALENT"
    bad = events.copy()
    bad.loc[1, "event"] = "x"
    assert ep.event_semantics_audit(events, bad).status == "AMBIGUOUS"

    nan_time = pd.DataFrame({"event": ["a"], "timestamp": [np.nan]})
    assert ep.event_semantics_audit(nan_time, nan_time).status == "LOSSLESS"


def test_hed_and_bids_structural_boundary_paths():
    hed = pd.DataFrame({"HED": ["(A)", ")A(", "(A", "", None]})
    out = ep.validate_hed_event_semantics(hed)
    assert out["structurally_valid"].tolist() == [True, False, False, False, False]

    data = pd.DataFrame(
        {
            "timestamp": [0.0],
            "x_coordinate": [0.1],
            "y_coordinate": [0.2],
            "pupil_size": [3.0],
        }
    )
    with pytest.raises(ep.EyeProcessValidationError, match="metadata must be a mapping"):
        ep.validate_bids_eye_semantics(data, [])

    meta = {
        "PhysioType": "eyetrack",
        "RecordedEye": "cyclopean",
        "SampleCoordinateSystem": "gaze-on-screen",
        "pupil_size": {},
    }
    missing = ep.validate_bids_eye_semantics(data, meta)
    assert missing.valid is False
    assert not missing.checks.loc[
        missing.checks["check"].isin(["gaze_on_screen_stimulus_metadata", "pupil_units_described"]),
        "pass",
    ].any()

    events_meta = {
        "StimulusPresentation": {
            "ScreenDistance": 60,
            "ScreenOrigin": "upper-left",
            "ScreenResolution": [1920, 1080],
            "ScreenSize": [53, 30],
        }
    }
    meta["pupil_size"] = {"Units": "mm"}
    complete = ep.validate_bids_eye_semantics(data, meta, events_meta)
    assert complete.valid is True


def test_semantic_roundtrip_attempt_failures_components_and_invalid_loss_map():
    d = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    rt = ep.semantic_roundtrip_audit(
        d,
        d.copy(),
        key="id",
        fields=["value"],
        pupil={"source_pupil": "missing_pupil"},
        eye={"source_eye": "missing_eye"},
        source_events=pd.DataFrame({"bad": [1]}),
        roundtrip_events=pd.DataFrame({"bad": [1]}),
    )
    assert rt.overall == "AMBIGUOUS"
    assert rt.component_status["timestamp"] == "UNSUPPORTED"
    assert rt.component_status["coordinates"] == "UNSUPPORTED"
    assert rt.component_status["pupil"] == "UNSUPPORTED"
    assert rt.component_status["eye"] == "UNSUPPORTED"
    assert rt.component_status["events"] == "UNSUPPORTED"

    with pytest.raises(ep.EyeProcessValidationError, match="eye_semantic_roundtrip"):
        ep.semantic_loss_map({})


def test_compatibility_evidence_mapping_defaults_invalid_levels_and_guards():
    comp = pd.DataFrame(
        {
            "ecosystem": ["A", "B"],
            "device": ["one", "two"],
            "supported": [True, True],
        }
    )
    declared = ep.compatibility_evidence_matrix({"matrix": comp})
    assert declared["detailed_evidence_level"].eq("declared").all()
    assert not declared["semantic_roundtrip_validated"].any()

    bad_ev = pd.DataFrame(
        {"ecosystem": ["A"], "device": ["one"], "evidence_level": ["impossible-level"]}
    )
    with pytest.raises(ep.EyeProcessValidationError, match="Unknown evidence_level"):
        ep.compatibility_evidence_matrix(comp, bad_ev)

    ev = pd.DataFrame(
        {
            "ecosystem": ["A", "A"],
            "device": ["one", "one"],
            "evidence_level": ["synthetic-fixture", "vendor-example"],
        }
    )
    out = ep.compatibility_evidence_matrix(comp, ev)
    row_a = out.loc[out.ecosystem.eq("A")].iloc[0]
    row_b = out.loc[out.ecosystem.eq("B")].iloc[0]
    assert row_a["detailed_evidence_level"] == "vendor-example"
    assert row_a["evidence_cases"] == 2
    assert bool(row_a["semantic_roundtrip_validated"]) is False
    assert row_b["detailed_evidence_level"] == "declared"
    assert row_b["evidence_cases"] == 0

    with pytest.raises(ep.EyeProcessValidationError, match="ecosystem and device"):
        ep.compatibility_evidence_matrix(
            pd.DataFrame({"other": [1]}),
            pd.DataFrame(
                {"ecosystem": ["A"], "device": ["one"], "evidence_level": ["declared"]}
            ),
        )


def test_vendor_timestamp_semantics_all_vendor_families_and_guards():
    d = pd.DataFrame(
        {
            "device": [0.0, 2.0, 1.0],
            "system": [10.0, 11.0, 12.0],
            "media": [0.0, 0.1, 0.2],
        }
    )
    with pytest.raises(ep.EyeProcessValidationError, match="vendor must"):
        ep.validate_vendor_timestamp_semantics(d, "")

    tobii = ep.validate_vendor_timestamp_semantics(
        d, "Tobii Pro", device_time="device", system_time="system"
    )
    assert len(tobii.clocks) == 2
    assert bool(tobii.clocks.loc[tobii.clocks.clock.eq("device"), "monotonic"].iloc[0]) is False

    pupil = ep.validate_vendor_timestamp_semantics(d, "Pupil Labs", device_time="device")
    assert pupil["pass"] is True

    gaze = ep.validate_vendor_timestamp_semantics(
        d, "Gazepoint GP3", device_time="device", media_time="media"
    )
    assert gaze["pass"] is True
    assert set(gaze.clocks.clock) == {"native", "media"}

    generic = ep.validate_vendor_timestamp_semantics(
        d, "OtherVendor", device_time=None, system_time="system"
    )
    assert generic["pass"] is False


def test_semantic_plot_guards_supplied_axes_and_compatibility_numeric_labels():
    with pytest.raises(ep.EyeProcessValidationError, match="eye_semantic_roundtrip"):
        ep.plot_eye_semantic_roundtrip({})

    d = pd.DataFrame({"id": [1, 2, 3], "timestamp": [0.0, 1.0, 2.0], "x": [1, 2, 3], "y": [2, 3, 4]})
    rt = ep.semantic_roundtrip_audit(d, d, key="id")
    fig, ax = plt.subplots()
    assert ep.plot_eye_semantic_roundtrip(rt, ax=ax) is ax
    _close(ax)

    evidence = pd.DataFrame({"detailed_evidence_level": ["declared", "vendor-example"]})
    fig, ax = plt.subplots()
    returned = ep.plot_eye_compatibility_evidence_matrix(evidence, ax=ax)
    assert returned is ax
    assert len(returned.gp3_data) == 2
    _close(returned)
