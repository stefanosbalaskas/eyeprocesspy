from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "eyeprocess_irt_engine_evidence_table",
    "eyeprocess_irt_precision_evidence_table",
    "eyeprocess_negative_control_evidence_table",
    "eyeprocess_recovery_evidence_table",
    "eyeprocess_reliability_evidence_table",
    "eyeprocess_sbc_evidence_table",
    "eyeprocess_stress_evidence_table",
    "eyeprocess_validation_atlas_gaps",
    "eyeprocess_validation_evidence_atlas",
    "eyeprocess_validation_evidence_index",
    "freeze_eyeprocess_validation_atlas",
    "verify_eyeprocess_validation_atlas",
    "write_eyeprocess_validation_report",
]


def _claims(status=("supported", "qualified")):
    return ep.eyeprocess_validation_claim_matrix(
        ["C1", "C2"],
        ["deterministic", "stress behavior"],
        ["E1", "E2"],
        ["test", "stress"],
        list(status),
    )


def _complete_atlas():
    frame = pd.DataFrame({"x": [1.0]})
    return ep.eyeprocess_validation_evidence_atlas(
        _claims(),
        recovery=frame,
        sbc=frame,
        stress=frame,
        reliability=frame,
        negative_controls=frame,
        irt=frame,
        provenance={"hash": "x"},
        artifacts=pd.DataFrame({"path": ["x"]}),
    )


def test_public_r094_exports_are_callable():
    assert len(TARGETS) == 13
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_paper_ready_recovery_table_selects_frozen_columns():
    source = pd.DataFrame(
        {
            "scenario_id": ["S1"],
            "parameter": ["a"],
            "n": [100],
            "bias": [0.123456],
            "rmse": [0.234567],
            "mae": [0.111119],
            "coverage": [0.949999],
            "other": [999],
        }
    )
    table = ep.eyeprocess_recovery_evidence_table(
        source,
        digits=4,
    )
    assert table.columns.tolist() == [
        "scenario_id",
        "parameter",
        "n",
        "bias",
        "rmse",
        "mae",
        "coverage",
    ]
    assert table.loc[0, "bias"] == pytest.approx(0.1235)
    assert table.loc[0, "rmse"] == pytest.approx(0.2346)
    assert "other" not in table

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="recognized recovery",
    ):
        ep.eyeprocess_recovery_evidence_table(pd.DataFrame({"x": [1]}))


def test_sbc_stress_reliability_and_negative_tables_round():
    sbc = ep.eyeprocess_sbc_evidence_table(
        pd.DataFrame(
            {
                "n_ranks": [100],
                "ecdf_max_deviation": [0.123456],
            }
        ),
        digits=3,
    )
    assert sbc.loc[0, "ecdf_max_deviation"] == pytest.approx(0.123)

    stress = ep.eyeprocess_stress_evidence_table(
        pd.DataFrame(
            {
                "severity": [0.123456],
                "delta": [0.987654],
            }
        ),
        digits=2,
    )
    assert stress.loc[0, "severity"] == pytest.approx(0.12)
    assert stress.loc[0, "delta"] == pytest.approx(0.99)

    reliability = ep.eyeprocess_reliability_evidence_table(
        pd.DataFrame(
            {
                "measure": ["dwell"],
                "icc_a1": [0.876543],
            }
        ),
        digits=4,
    )
    assert reliability.loc[0, "icc_a1"] == pytest.approx(0.8765)

    negative = ep.eyeprocess_negative_control_evidence_table(
        pd.DataFrame(
            {
                "control": ["permutation"],
                "effect": [0.012345],
            }
        ),
        digits=4,
    )
    assert negative.loc[0, "effect"] == pytest.approx(0.0123)


def test_irt_precision_table_calls_existing_native_math_surface():
    items = pd.DataFrame(
        {
            "item_id": ["I1", "I2", "I3"],
            "a": [1.0, 1.2, 0.8],
            "b": [-0.5, 0.0, 0.5],
            "c": [0.0, 0.0, 0.0],
            "d": [1.0, 1.0, 1.0],
        }
    )
    table = ep.eyeprocess_irt_precision_evidence_table(
        items,
        theta=[-1, 0, 1],
        digits=4,
    )
    assert table.columns.tolist() == [
        "theta",
        "information",
        "conditional_sem",
    ]
    assert table["theta"].tolist() == [-1.0, 0.0, 1.0]
    assert np.isfinite(table["information"]).all()
    assert (table["information"] > 0).all()
    assert np.isfinite(table["conditional_sem"]).all()


def test_irt_engine_table_preserves_exact_engine_gating():
    table = ep.eyeprocess_irt_engine_evidence_table()
    assert {
        "engine",
        "capability",
        "package",
        "available",
        "policy",
    }.issubset(table.columns)

    unavailable = table.loc[~table["available"].astype(bool)]
    assert (unavailable["policy"] == "gated; no substitute estimator").all()


def test_validation_evidence_index_hashes_and_classifies(tmp_path):
    figure = tmp_path / "figure-one.png"
    table = tmp_path / "results_summary.csv"
    provenance = tmp_path / "manifest.json"
    artifact = tmp_path / "notes.txt"
    nested = tmp_path / "nested"
    nested.mkdir()
    hidden_nested = nested / "plot-secondary.svg"

    figure.write_bytes(b"figure")
    table.write_text("x\n1\n", encoding="utf-8")
    provenance.write_text("{}", encoding="utf-8")
    artifact.write_text("notes", encoding="utf-8")
    hidden_nested.write_text("<svg/>", encoding="utf-8")

    index = ep.eyeprocess_validation_evidence_index(
        tmp_path,
        recursive=True,
    )
    assert len(index) == 5
    roles = dict(zip(index["path"], index["role"]))
    assert roles["figure-one.png"] == "figure"
    assert roles["results_summary.csv"] == "table"
    assert roles["manifest.json"] == "provenance"
    assert roles["notes.txt"] == "artifact"
    assert roles["nested/plot-secondary.svg"] == "figure"

    row = index.loc[index["path"] == "notes.txt"].iloc[0]
    assert row["extension"] == "txt"
    assert row["bytes"] == len(b"notes")
    assert row["hash"] == hashlib.md5(b"notes").hexdigest()

    flat = ep.eyeprocess_validation_evidence_index(
        tmp_path,
        recursive=False,
    )
    assert "nested/plot-secondary.svg" not in set(flat["path"])


def test_validation_atlas_freezes_verifies_and_is_complete():
    atlas = _complete_atlas()
    assert atlas["eyeprocess_class"] == "eye_validation_evidence_atlas"
    assert atlas["coverage"] == pytest.approx(1.0)
    assert atlas["component_status"]["present"].all()
    assert "construct validity" in atlas["guardrail"]

    gaps = ep.eyeprocess_validation_atlas_gaps(atlas)
    assert gaps["missing_components"] == []
    assert gaps["unresolved_claims"].empty
    assert gaps["complete"] is True

    frozen = ep.freeze_eyeprocess_validation_atlas(
        atlas,
        metadata={"commit": "abc"},
    )
    assert frozen["eyeprocess_class"] == "eye_validation_atlas_freeze"
    assert frozen["payload"]["version"] == "0.11.1"
    assert ep.verify_eyeprocess_validation_atlas(frozen)

    changed = copy.deepcopy(frozen)
    changed["payload"]["metadata"]["commit"] = "def"
    assert not ep.verify_eyeprocess_validation_atlas(changed)


def test_validation_atlas_gaps_are_descriptive_not_certification():
    atlas = ep.eyeprocess_validation_evidence_atlas(
        _claims(status=("supported", "pending")),
        recovery=pd.DataFrame({"x": [1]}),
    )
    gaps = ep.eyeprocess_validation_atlas_gaps(atlas)

    assert "sbc" in gaps["missing_components"]
    assert gaps["unresolved_claims"]["claim_id"].tolist() == ["C2"]
    assert gaps["complete"] is False
    assert atlas["coverage"] == pytest.approx(1 / 8)


def test_validation_atlas_requires_claim_contract():
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="missing required columns",
    ):
        ep.eyeprocess_validation_evidence_atlas(pd.DataFrame({"claim": ["x"]}))

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="created by",
    ):
        ep.eyeprocess_validation_atlas_gaps({})

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_validation_evidence_atlas",
    ):
        ep.freeze_eyeprocess_validation_atlas({})

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_validation_atlas_freeze",
    ):
        ep.verify_eyeprocess_validation_atlas({})


def test_validation_report_matches_frozen_sections(tmp_path):
    atlas = _complete_atlas()
    path = tmp_path / "validation-report.md"

    returned = ep.write_eyeprocess_validation_report(
        atlas,
        path,
        title="Frozen validation report",
    )
    assert returned == path.resolve().as_posix()

    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Frozen validation report\n")
    assert "Atlas hash: `" in text
    assert "## Evidence components" in text
    assert "- recovery: present" in text
    assert "## Claim status" in text
    assert "`C1` - supported: deterministic" in text
    assert "## Interpretation guardrail" in text
    assert "does not establish substantive construct validity" in text
    assert "Missing components: none" in text
