from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "evaluate_validation_acceptance",
    "expand_eyeprocess_validation_plan",
    "eyeprocess_validation_evidence_grade",
    "eyeprocess_validation_plan",
    "eyeprocess_validation_seed",
    "read_validation_scenario_manifest",
    "summarise_validation_acceptance",
    "validate_eyeprocess_validation_plan",
    "validation_acceptance_matrix",
    "validation_acceptance_rule",
    "validation_mcse_profile",
    "validation_replication_budget",
    "validation_scenario_manifest",
    "write_validation_scenario_manifest",
]


def test_public_r082_exports_are_callable():
    assert len(TARGETS) == 14
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_validation_plan_is_deterministic_and_auditable():
    plan = ep.eyeprocess_validation_plan(
        sample_size=100,
        n_items=6,
        missing_rate=[0.0, 0.1],
        noise_level="reference",
        specification=["correct", "misspecified"],
        replications=3,
        seed=123,
    )

    assert plan["eyeprocess_class"] == "eye_validation_evidence_plan"
    assert ep.validate_eyeprocess_validation_plan(plan) is True

    first = ep.expand_eyeprocess_validation_plan(plan)
    second = ep.expand_eyeprocess_validation_plan(plan)
    pd.testing.assert_frame_equal(first, second)
    assert first.attrs == second.attrs
    assert len(first) > 1
    assert (first["scenario_seed"] > 0).all()
    assert first.loc[0, "scenario_seed"] == 104853

    # R expand.grid() varies the first supplied dimension fastest.
    assert first.loc[0, "family"] == "recovery"
    assert first.loc[1, "family"] == "sbc"
    assert first.loc[4, "family"] == "negative_control"
    assert first.loc[5, "family"] == "recovery"


def test_validation_plan_rejects_invalid_design_values():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="families",
    ):
        ep.eyeprocess_validation_plan(
            families=["unknown"],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="sample_size",
    ):
        ep.eyeprocess_validation_plan(
            sample_size=10,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="n_items",
    ):
        ep.eyeprocess_validation_plan(
            n_items=2,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="missing_rate",
    ):
        ep.eyeprocess_validation_plan(
            missing_rate=1.0,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="specification",
    ):
        ep.eyeprocess_validation_plan(
            specification=["wrong"],
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="replications",
    ):
        ep.eyeprocess_validation_plan(
            replications=0,
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="seed",
    ):
        ep.eyeprocess_validation_plan(
            seed=0,
        )


def test_validation_seed_matches_frozen_formula_and_stream_contract():
    assert ep.eyeprocess_validation_seed(123, 1) == 104853
    assert (
        ep.eyeprocess_validation_seed(
            123,
            1,
            stream=2,
        )
        == 106871
    )

    value = ep.eyeprocess_validation_seed(
        2147483640,
        100000,
        stream=15,
    )
    assert 1 <= value <= 2147483646

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="positive scalar integers",
    ):
        ep.eyeprocess_validation_seed(0, 1)


def test_acceptance_rules_cover_all_frozen_directions():
    max_rule = ep.validation_acceptance_rule(
        "rmse",
        "max",
        0.2,
    )
    min_rule = ep.validation_acceptance_rule(
        "coverage",
        "min",
        0.9,
    )
    between_rule = ep.validation_acceptance_rule(
        "bias",
        "between",
        -0.05,
        upper=0.05,
    )
    equals_rule = ep.validation_acceptance_rule(
        "nominal",
        "equals",
        0.95,
        tolerance=0.01,
    )

    assert ep.evaluate_validation_acceptance(
        0.15,
        max_rule,
    )
    assert ep.evaluate_validation_acceptance(
        0.92,
        min_rule,
    )
    assert ep.evaluate_validation_acceptance(
        0.01,
        between_rule,
    )
    assert ep.evaluate_validation_acceptance(
        0.955,
        equals_rule,
    )
    assert pd.isna(
        ep.evaluate_validation_acceptance(
            np.nan,
            max_rule,
        )
    )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="between rules",
    ):
        ep.validation_acceptance_rule(
            "bias",
            "between",
            1.0,
            upper=0.0,
        )


def test_acceptance_matrix_and_summary_follow_frozen_contract():
    summary = pd.DataFrame(
        {
            "id": ["A", "B"],
            "rmse": [0.15, 0.25],
            "coverage": [0.94, 0.88],
        }
    )
    rules = {
        "rmse": ep.validation_acceptance_rule(
            "rmse",
            "max",
            0.2,
        ),
        "coverage": ep.validation_acceptance_rule(
            "coverage",
            "min",
            0.9,
        ),
    }

    matrix = ep.validation_acceptance_matrix(
        summary,
        rules,
        id_cols="id",
    )
    assert len(matrix) == 4
    assert matrix["rule_id"].tolist() == [
        "rmse",
        "coverage",
        "rmse",
        "coverage",
    ]
    assert matrix["pass"].tolist() == [
        True,
        True,
        False,
        False,
    ]

    overall = ep.summarise_validation_acceptance(matrix)
    assert overall.loc[0, "n"] == 4
    assert overall.loc[0, "n_evaluable"] == 4
    assert overall.loc[0, "n_pass"] == 2
    assert overall.loc[0, "pass_fraction"] == pytest.approx(0.5)

    grouped = ep.summarise_validation_acceptance(
        matrix,
        by="id",
    )
    assert grouped["id"].tolist() == ["A", "B"]
    assert grouped["pass_fraction"].tolist() == [1.0, 0.0]


def test_mcse_profile_and_replication_budget():
    values = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "value": [1, 2, 3, 4, 2, 2, 2, 2],
        }
    )

    overall = ep.validation_mcse_profile(
        values,
        "value",
    )
    expected_sd = np.std(values["value"], ddof=1)
    assert overall.loc[0, "n"] == 8
    assert overall.loc[0, "mean"] == pytest.approx(values["value"].mean())
    assert overall.loc[0, "sd"] == pytest.approx(expected_sd)
    assert overall.loc[0, "mcse_mean"] == pytest.approx(expected_sd / np.sqrt(8))

    grouped = ep.validation_mcse_profile(
        values,
        "value",
        by="group",
    )
    assert grouped["group"].tolist() == ["A", "B"]
    assert grouped.loc[1, "sd"] == pytest.approx(0.0)
    assert grouped.loc[1, "mcse_mean"] == pytest.approx(0.0)

    assert (
        ep.validation_replication_budget(
            0.5,
            0.05,
        )
        == 100
    )
    assert (
        ep.validation_replication_budget(
            0.0,
            0.05,
            minimum=25,
        )
        == 25
    )
    assert (
        ep.validation_replication_budget(
            10.0,
            0.01,
            maximum=500,
        )
        == 500
    )


def test_scenario_manifest_json_roundtrip_and_rds_gate(tmp_path):
    plan = ep.eyeprocess_validation_plan(
        families=["recovery", "stress"],
        sample_size=100,
        n_items=6,
        missing_rate=0.0,
        noise_level="reference",
        specification="correct",
        replications=2,
        seed=123,
        label="frozen-test",
    )
    manifest = ep.validation_scenario_manifest(
        plan,
        source_commit="abc123",
        generated_at=datetime(
            2026,
            8,
            11,
            12,
            30,
            tzinfo=timezone.utc,
        ),
    )

    assert manifest["eyeprocess_class"] == "eye_validation_scenario_manifest"
    assert manifest["label"] == "frozen-test"
    assert manifest["source_commit"] == "abc123"
    assert manifest["generated_at"] == ("2026-08-11 12:30:00 UTC")
    assert manifest["scientific_scope"] == ("software validation; not construct-validity evidence")
    assert len(manifest["scenarios"]) == 2

    path = tmp_path / "validation-manifest.json"
    returned = ep.write_validation_scenario_manifest(
        manifest,
        path,
    )
    assert returned == path.resolve().as_posix()

    restored = ep.read_validation_scenario_manifest(path)
    assert restored["eyeprocess_class"] == "eye_validation_scenario_manifest"
    assert restored["plan_hash"] == manifest["plan_hash"]
    pd.testing.assert_frame_equal(
        restored["scenarios"],
        manifest["scenarios"],
        check_dtype=False,
    )
    assert restored["scenarios"].attrs == (manifest["scenarios"].attrs)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.write_validation_scenario_manifest(
            manifest,
            tmp_path / "validation-manifest.rds",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.read_validation_scenario_manifest(
            tmp_path / "validation-manifest.rds",
        )


def test_validation_evidence_grade_contract():
    complete = ep.eyeprocess_validation_evidence_grade(
        [
            "design",
            "execution",
            "summary",
            "provenance",
            "hash",
        ]
    )
    assert complete["grade"] == "complete"
    assert complete["coverage"] == pytest.approx(1.0)

    partial = ep.eyeprocess_validation_evidence_grade(["design", "execution", "summary"])
    assert partial["grade"] == "partial"
    assert partial["coverage"] == pytest.approx(0.6)

    insufficient = ep.eyeprocess_validation_evidence_grade(["design"])
    assert insufficient["grade"] == "insufficient"
    assert insufficient["coverage"] == pytest.approx(0.2)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="required",
    ):
        ep.eyeprocess_validation_evidence_grade(
            [],
            required=[],
        )


def test_structural_validation_requires_plan_class():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_validation_evidence_plan",
    ):
        ep.validate_eyeprocess_validation_plan({})

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="rule must be created",
    ):
        ep.evaluate_validation_acceptance(
            0.1,
            {},
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="data.frame",
    ):
        ep.validation_mcse_profile(
            {"value": [1, 2]},
            "value",
        )
