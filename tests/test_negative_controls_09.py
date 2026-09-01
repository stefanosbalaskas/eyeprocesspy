from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "audit_temporal_leakage",
    "negative_control_concordance",
    "outcome_blind_feature_audit",
    "placebo_window_audit",
    "process_feature_time_provenance",
    "process_negative_control_permute",
    "process_negative_control_shift",
    "process_null_benchmark",
    "run_process_negative_controls",
    "summarise_process_negative_controls",
    "validate_feature_availability",
]


def test_public_r077_exports_are_callable():
    assert len(TARGETS) == 11
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_leakage_and_negative_controls_are_explicit():
    provenance = ep.process_feature_time_provenance(
        ["pre", "post"],
        [10, 30],
        [20, 20],
    )
    audit = ep.audit_temporal_leakage(provenance)
    assert audit["eyeprocess_class"] == "eye_temporal_leakage_audit"
    assert audit["n_flagged"] == 1

    rng = np.random.default_rng(9)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=80),
            "y": rng.normal(size=80),
        }
    )

    def analysis(frame):
        return float(
            np.polyfit(
                frame["x"].to_numpy(dtype=float),
                frame["y"].to_numpy(dtype=float),
                1,
            )[0]
        )

    controls = ep.run_process_negative_controls(
        data,
        "y",
        analysis,
        replications=4,
    )
    assert controls["eyeprocess_class"] == "eye_process_negative_controls"
    assert len(controls["results"]) == 8

    benchmark = ep.process_null_benchmark(
        analysis(data),
        controls,
    )
    assert np.isfinite(benchmark["two_sided_tail"])


def test_frozen_negative_control_inputs_fail_safely():
    with pytest.raises(ep.EyeProcessValidationError, match="seed"):
        ep.process_negative_control_permute(
            pd.DataFrame({"y": [1, 2, 3]}),
            "y",
            seed=-1,
        )

    with pytest.raises(ep.EyeProcessValidationError, match="lag"):
        ep.process_negative_control_shift(
            pd.DataFrame({"y": [1, 2, 3]}),
            "y",
            lag=np.nan,
        )

    with pytest.raises(ep.EyeProcessValidationError, match="shift_lags"):
        ep.run_process_negative_controls(
            pd.DataFrame({"y": [1, 2, 3]}),
            "y",
            lambda frame: float(frame["y"].mean()),
            controls="shift",
            shift_lags=[0],
        )

    provenance = ep.process_feature_time_provenance(["x"], [1], [2])
    with pytest.raises(ep.EyeProcessValidationError, match="tolerance"):
        ep.audit_temporal_leakage(provenance, tolerance=-1)


def test_provenance_recycling_and_boundary_rules():
    provenance = ep.process_feature_time_provenance(
        ["a", "b"],
        [10],
        [20, 15],
        source=["raw"],
        transformation=["none", "smooth"],
    )
    np.testing.assert_allclose(
        provenance["lead"].to_numpy(dtype=float),
        [10.0, 5.0],
    )
    assert provenance["available_before_outcome"].tolist() == [True, True]

    boundary = ep.process_feature_time_provenance(
        ["equal", "near", "late"],
        [20, 20.5, 22],
        [20, 20, 20],
    )
    allowed = ep.audit_temporal_leakage(
        boundary,
        allow_equal=True,
        tolerance=1.0,
    )
    assert allowed["detail"]["leakage_flag"].tolist() == [False, False, True]

    strict = ep.audit_temporal_leakage(
        boundary,
        allow_equal=False,
        tolerance=1.0,
    )
    assert strict["detail"]["leakage_flag"].tolist() == [True, True, True]


def test_feature_availability_and_outcome_blind_audit():
    provenance = ep.process_feature_time_provenance(
        ["a", "b"],
        [10, 30],
        [40, 40],
    )
    scalar = ep.validate_feature_availability(provenance, 20)
    assert scalar["available"].tolist() == [True, False]

    named = ep.validate_feature_availability(
        provenance,
        {"a": 9, "b": 30},
    )
    assert named["available"].tolist() == [False, True]

    with pytest.raises(ep.EyeProcessValidationError, match="named"):
        ep.validate_feature_availability(provenance, [10, 30])

    data = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
    passed = ep.outcome_blind_feature_audit(
        data,
        "y",
        lambda frame: frame[["x"]].copy(),
    )
    assert passed["status"] == "pass"
    assert passed["input_columns"] == ["x"]

    failed = ep.outcome_blind_feature_audit(
        data,
        "y",
        lambda frame: frame["y"],
    )
    assert failed["status"] == "failed"
    assert failed["feature_result"] is None


def test_permutation_and_shift_contracts():
    data = pd.DataFrame(
        {
            "group": ["a"] * 4 + ["b"] * 4,
            "y": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )
    permuted = ep.process_negative_control_permute(
        data,
        "y",
        seed=3,
        within=["group"],
    )
    for group in ["a", "b"]:
        assert sorted(permuted.loc[permuted["group"] == group, "y"].tolist()) == sorted(
            data.loc[data["group"] == group, "y"].tolist()
        )

    grouped = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "y": [1, 2, 3, 10, 20, 30],
        }
    )
    positive = ep.process_negative_control_shift(
        grouped,
        "y",
        lag=1,
        by=["g"],
    )
    assert pd.isna(positive.loc[0, "y"])
    assert positive.loc[1, "y"] == 1
    assert positive.loc[2, "y"] == 2
    assert pd.isna(positive.loc[3, "y"])
    assert positive.loc[4, "y"] == 10
    assert positive.loc[5, "y"] == 20

    negative = ep.process_negative_control_shift(
        pd.DataFrame({"y": [1, 2, 3]}),
        "y",
        lag=-1,
    )
    assert negative["y"].tolist()[:2] == [2, 3]
    assert pd.isna(negative.loc[2, "y"])

    exhausted = ep.process_negative_control_shift(
        pd.DataFrame({"y": [1, 2, 3]}),
        "y",
        lag=3,
    )
    assert exhausted["y"].isna().all()


def test_placebo_summary_and_null_summary_contracts():
    data = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "time": [-1, 0, -1, 0],
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )
    placebo = ep.placebo_window_audit(
        data,
        "time",
        "value",
        [-1, 0],
        expected=2.0,
        by=["group"],
    )
    np.testing.assert_allclose(
        placebo["mean"].to_numpy(dtype=float),
        [2.0, 3.0],
    )
    assert placebo.attrs["placebo_window"] == [-1.0, 0.0]

    controls = {
        "results": pd.DataFrame(
            {
                "control": [
                    "permutation",
                    "permutation",
                    "shift",
                    "shift",
                ],
                "effect": [0.01, -0.01, 0.02, -0.02],
            }
        ),
        "eyeprocess_class": "eye_process_negative_controls",
    }
    summary = ep.summarise_process_negative_controls(
        controls,
        threshold=0.015,
    )
    assert summary["n_finite"].tolist() == [2, 2]

    benchmark = ep.process_null_benchmark(0.02, controls)
    assert benchmark["n_null"] == 4
    assert benchmark["percentile"] == pytest.approx(1.0)
    assert benchmark["two_sided_tail"] == pytest.approx(0.5)

    concordance = ep.negative_control_concordance(
        controls,
        tolerance=0.05,
    )
    assert concordance["all_within_tolerance"] is True


def test_failed_analysis_is_retained_as_control_row():
    data = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    def fails(_):
        raise RuntimeError("synthetic failure")

    controls = ep.run_process_negative_controls(
        data,
        "y",
        fails,
        controls="permutation",
        replications=2,
    )
    assert len(controls["results"]) == 2
    assert controls["results"]["effect"].isna().all()
    assert controls["results"]["error"].str.contains("synthetic failure").all()


def test_validation_boundaries_are_explicit():
    with pytest.raises(ep.EyeProcessValidationError, match="At least one"):
        ep.process_feature_time_provenance(
            [],
            [],
            [],
            source=[],
            transformation=[],
        )

    with pytest.raises(ep.EyeProcessValidationError, match="feature names"):
        ep.process_feature_time_provenance([""], [1], [2])

    with pytest.raises(ep.EyeProcessValidationError, match="finite"):
        ep.process_feature_time_provenance(["x"], [np.nan], [2])

    with pytest.raises(ep.EyeProcessValidationError, match="window"):
        ep.placebo_window_audit(
            pd.DataFrame({"time": [0], "value": [1.0]}),
            "time",
            "value",
            [0],
        )

    with pytest.raises(ep.EyeProcessValidationError, match="replications"):
        ep.run_process_negative_controls(
            pd.DataFrame({"y": [1, 2]}),
            "y",
            lambda frame: 0.0,
            replications=0,
        )
