from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "expand_eyeprocess_stress_evidence_plan",
    "eyeprocess_negative_control_evidence_plan",
    "eyeprocess_reliability_evidence_plan",
    "eyeprocess_stress_evidence_plan",
    "eyeprocess_validation_claim_matrix",
    "eyeprocess_validation_evidence_manifest",
    "eyeprocess_validation_readiness",
    "eyeprocess_validation_release_gate",
    "freeze_eyeprocess_validation_evidence",
    "read_eyeprocess_validation_evidence",
    "run_eyeprocess_stress_evidence",
    "summarise_eyeprocess_stress_evidence",
    "verify_eyeprocess_validation_evidence",
    "write_eyeprocess_validation_evidence",
]


def _small_stress_plan(seed=11):
    return ep.eyeprocess_stress_evidence_plan(
        missing_gaze=[0.0, 0.1],
        pupil_dropout=0.0,
        calibration_offset=0.0,
        sampling_jitter=0.0,
        aoi_label_noise=0.0,
        device_shift=0.0,
        trial_imbalance=0.0,
        seed=seed,
    )


def test_public_r091_exports_are_callable():
    assert len(TARGETS) == 14
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_stress_reliability_and_negative_control_plans_expand():
    plan = _small_stress_plan(seed=4)
    grid = ep.expand_eyeprocess_stress_evidence_plan(plan)

    assert plan["eyeprocess_class"] == "eye_stress_evidence_plan"
    assert len(grid) == 8
    assert grid["scenario_id"].tolist() == [f"STRESS{index:03d}" for index in range(1, 9)]
    assert grid.loc[0, "corruption"] == "missing_gaze"
    assert grid.loc[1, "severity"] == pytest.approx(0.1)
    assert grid.loc[0, "seed"] == ep.eyeprocess_validation_seed(
        4,
        1,
    )

    reliability = ep.eyeprocess_reliability_evidence_plan(
        bootstrap=10,
    )
    negative = ep.eyeprocess_negative_control_evidence_plan(
        replications=10,
    )
    assert reliability["eyeprocess_class"] == "eye_reliability_evidence_plan"
    assert negative["eyeprocess_class"] == "eye_negative_control_evidence_plan"
    assert "construct validity" in reliability["guardrail"]
    assert "participant state" in negative["guardrail"]


def test_stress_plan_validation_matches_frozen_guardrails():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="missing_gaze",
    ):
        ep.eyeprocess_stress_evidence_plan(
            missing_gaze=1.0,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="calibration_offset",
    ):
        ep.eyeprocess_stress_evidence_plan(
            calibration_offset=-0.1,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="seed",
    ):
        ep.eyeprocess_stress_evidence_plan(seed=0)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="unsupported reliability",
    ):
        ep.eyeprocess_reliability_evidence_plan(
            metrics=["unknown"],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="unsupported negative",
    ):
        ep.eyeprocess_negative_control_evidence_plan(
            controls=["unknown"],
        )


def test_claim_matrix_scalar_default_and_recycling_contract():
    one = ep.eyeprocess_validation_claim_matrix(
        "C1",
        "claim",
        "E1",
        "test",
    )
    assert len(one) == 1
    assert one.loc[0, "status"] == "qualified"

    recycled = ep.eyeprocess_validation_claim_matrix(
        ["C1", "C2"],
        ["a", "b"],
        "E",
        "test",
        status="supported",
    )
    assert recycled["evidence_id"].tolist() == ["E", "E"]
    assert recycled["status"].tolist() == [
        "supported",
        "supported",
    ]

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="common maximum length",
    ):
        ep.eyeprocess_validation_claim_matrix(
            ["C1", "C2"],
            ["a", "b", "c"],
            "E",
            "test",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="claim_id",
    ):
        ep.eyeprocess_validation_claim_matrix(
            ["C1", "C1"],
            ["a", "b"],
            ["E1", "E2"],
            "test",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="invalid claim status",
    ):
        ep.eyeprocess_validation_claim_matrix(
            "C1",
            "a",
            "E1",
            "test",
            status="overstated",
        )


def test_frozen_evidence_detects_tampering():
    claims = ep.eyeprocess_validation_claim_matrix(
        "C1",
        "software behavior is reproducible",
        "E1",
        "test",
        "supported",
    )
    frozen = ep.freeze_eyeprocess_validation_evidence(
        design=pd.DataFrame({"id": [1]}),
        recovery=pd.DataFrame({"x": [1]}),
        stress=pd.DataFrame({"x": [1]}),
        reliability=pd.DataFrame({"x": [1]}),
        negative_controls=pd.DataFrame({"x": [1]}),
        claims=claims,
        provenance={"commit": "abc"},
        source_commit="abc",
    )
    assert frozen["eyeprocess_class"] == "eye_validation_evidence_freeze"
    assert ep.verify_eyeprocess_validation_evidence(frozen)

    bad = copy.deepcopy(frozen)
    bad["components"]["recovery"].loc[0, "x"] = 2
    assert not ep.verify_eyeprocess_validation_evidence(bad)


def test_executed_stress_evidence_is_deterministic_and_isolated():
    data = pd.DataFrame(
        {
            "x": np.arange(1, 41, dtype=float),
            "keep": True,
        }
    )
    plan = _small_stress_plan(seed=11)

    def corruptor(frame, severity, seed):
        del seed
        frame["x"] = frame["x"] + severity
        return frame

    corruptors = {
        name: corruptor
        for name in [
            "missing_gaze",
            "pupil_dropout",
            "calibration_offset",
            "sampling_jitter",
            "aoi_label_noise",
            "device_shift",
            "trial_imbalance",
        ]
    }

    def metric(frame):
        return {"mean_x": float(frame["x"].mean())}

    first = ep.run_eyeprocess_stress_evidence(
        data,
        plan,
        corruptors,
        metric,
    )
    second = ep.run_eyeprocess_stress_evidence(
        data,
        plan,
        corruptors,
        metric,
    )

    assert first["eyeprocess_class"] == "eye_stress_evidence_result"
    pd.testing.assert_frame_equal(
        first["results"],
        second["results"],
    )
    assert len(first["failures"]) == 0
    assert data["x"].iloc[0] == pytest.approx(1.0)

    baseline = data["x"].mean()
    row = first["results"].query("corruption == 'missing_gaze' and severity == 0.1").iloc[0]
    assert row["baseline"] == pytest.approx(baseline)
    assert row["value"] == pytest.approx(baseline + 0.1)
    assert row["delta"] == pytest.approx(0.1)
    assert row["relative_change"] == pytest.approx(0.1 / abs(baseline))

    summary = ep.summarise_eyeprocess_stress_evidence(first)
    assert len(summary) > 0
    missing = summary.query("corruption == 'missing_gaze' and metric == 'mean_x'").iloc[0]
    assert missing["n_scenarios"] == 2
    assert missing["min_severity"] == pytest.approx(0.0)
    assert missing["max_severity"] == pytest.approx(0.1)
    assert missing["mean_delta"] == pytest.approx(0.05)
    assert missing["max_abs_delta"] == pytest.approx(0.1)


def test_stress_executor_captures_corruptor_and_metric_failures():
    plan = _small_stress_plan(seed=19)

    def identity(frame, severity, seed):
        del severity, seed
        return frame

    def broken(frame, severity, seed):
        del frame, severity, seed
        raise RuntimeError("synthetic corruption failure")

    corruptors = {
        name: identity
        for name in [
            "missing_gaze",
            "pupil_dropout",
            "calibration_offset",
            "sampling_jitter",
            "aoi_label_noise",
            "device_shift",
            "trial_imbalance",
        ]
    }
    corruptors["device_shift"] = broken

    result = ep.run_eyeprocess_stress_evidence(
        pd.DataFrame({"x": [1.0, 2.0]}),
        plan,
        corruptors,
        lambda frame: {"mean_x": frame["x"].mean()},
    )
    failures = result["failures"]
    assert not failures.empty
    assert set(
        failures.loc[
            failures["corruption"] == "device_shift",
            "error",
        ]
    ) == {"synthetic corruption failure"}

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Missing corruptors",
    ):
        ep.run_eyeprocess_stress_evidence(
            pd.DataFrame({"x": [1.0]}),
            plan,
            {"missing_gaze": identity},
            lambda frame: {"mean_x": frame["x"].mean()},
        )


def test_evidence_manifest_hashes_files_and_objects(tmp_path):
    evidence_file = tmp_path / "evidence.txt"
    evidence_file.write_text(
        "eyeprocess validation evidence\n",
        encoding="utf-8",
    )
    expected_md5 = hashlib.md5(evidence_file.read_bytes()).hexdigest()

    manifest = ep.eyeprocess_validation_evidence_manifest(
        files=[evidence_file],
        objects={"table": pd.DataFrame({"x": [1, 2]})},
        source_commit="abc",
        label="frozen",
    )

    assert manifest["eyeprocess_class"] == "eye_validation_evidence_manifest"
    assert manifest["files"].loc[0, "md5"] == expected_md5
    assert manifest["objects"].loc[0, "name"] == "table"
    assert len(manifest["objects"].loc[0, "hash"]) == 64
    assert manifest["source_commit"] == "abc"

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="Evidence files not found",
    ):
        ep.eyeprocess_validation_evidence_manifest(
            files=[tmp_path / "missing.txt"],
        )


def test_readiness_and_release_gate_are_conservative():
    claims = ep.eyeprocess_validation_claim_matrix(
        "C1",
        "claim",
        "E1",
        "test",
        "supported",
    )
    frozen = ep.freeze_eyeprocess_validation_evidence(
        design={"id": 1},
        recovery={"rmse": 0.1},
        stress={"stable": True},
        reliability={"icc": 0.9},
        negative_controls={"pass": True},
        claims=claims,
        provenance={"commit": "abc"},
        source_commit="abc",
    )
    readiness = ep.eyeprocess_validation_readiness(frozen)
    assert readiness["ready"] is True
    assert readiness["hash_valid"] is True
    assert readiness["table"]["satisfied"].all()

    acceptance = pd.DataFrame(
        {
            "pass": [True, True],
        }
    )
    gate = ep.eyeprocess_validation_release_gate(
        readiness,
        acceptance=acceptance,
    )
    assert gate["pass"] is True
    assert "software-release readiness only" in gate["interpretation"]

    failed_acceptance = pd.DataFrame(
        {
            "pass": [True, False],
        }
    )
    failed_gate = ep.eyeprocess_validation_release_gate(
        readiness,
        acceptance=failed_acceptance,
    )
    assert failed_gate["pass"] is False
    assert failed_gate["acceptance"] is False

    incomplete = ep.freeze_eyeprocess_validation_evidence(
        design={"id": 1},
    )
    not_ready = ep.eyeprocess_validation_readiness(incomplete)
    assert not not_ready["ready"]
    assert not ep.eyeprocess_validation_release_gate(
        not_ready,
    )["pass"]


def test_json_freeze_roundtrip_and_native_rds_gate(tmp_path):
    frozen = ep.freeze_eyeprocess_validation_evidence(
        design=pd.DataFrame({"id": [1, 2]}),
        recovery=pd.DataFrame({"rmse": [0.1]}),
        source_commit="abc",
    )
    path = tmp_path / "freeze.json"
    returned = ep.write_eyeprocess_validation_evidence(
        frozen,
        path,
    )
    assert returned == path.resolve().as_posix()

    restored = ep.read_eyeprocess_validation_evidence(path)
    assert ep.verify_eyeprocess_validation_evidence(restored)
    pd.testing.assert_frame_equal(
        restored["components"]["design"],
        frozen["components"]["design"],
        check_dtype=False,
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.write_eyeprocess_validation_evidence(
            frozen,
            tmp_path / "freeze.rds",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.read_eyeprocess_validation_evidence(
            tmp_path / "freeze.rds",
        )


def test_write_rejects_tampered_freeze(tmp_path):
    frozen = ep.freeze_eyeprocess_validation_evidence(
        design={"id": 1},
    )
    frozen["components"]["design"]["id"] = 2

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="hash verification failed",
    ):
        ep.write_eyeprocess_validation_evidence(
            frozen,
            tmp_path / "bad.json",
        )
