from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "freeze_software_paper_evidence",
    "paper_reproducibility_manifest",
    "software_paper_claim_matrix",
    "software_paper_coverage",
    "software_paper_evidence_bundle",
    "software_paper_gap_analysis",
    "software_paper_readiness",
    "software_paper_validation_table",
    "write_software_paper_evidence",
]


def _ready_bundle():
    fingerprint = ep.eye_reproducibility_fingerprint(
        data=[1, 2, 3],
        result={"effect": 0.2},
    )
    claims = ep.software_paper_claim_matrix(
        ["Claim one", "Claim two"],
        evidence_id=["E1", "E2"],
        evidence_type=["validation", "benchmark"],
        status=["supported", "qualified"],
        scope=["synthetic data", "benchmark fixture"],
        source=["tests", "benchmark"],
    )
    return ep.software_paper_evidence_bundle(
        claims=claims,
        validation=pd.DataFrame({"id": ["E1"]}),
        examples=["example-1"],
        articles=["article-1"],
        benchmarks=pd.DataFrame({"id": ["E2"]}),
        reproducibility=fingerprint,
        metadata={"version": "0.9"},
    )


def test_public_r079_exports_are_callable():
    assert len(TARGETS) == 9
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_frozen_claim_matrix_recycles_and_validates():
    claims = ep.software_paper_claim_matrix(
        ["A", "B"],
        evidence_id="E1",
        evidence_type=["test", "benchmark"],
        status=["supported", "qualified"],
    )

    assert claims["claim_id"].tolist() == [
        "CL001",
        "CL002",
    ]
    assert claims["evidence_id"].tolist() == [
        "E1",
        "E1",
    ]
    assert claims["status"].tolist() == [
        "supported",
        "qualified",
    ]

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="claim",
    ):
        ep.software_paper_claim_matrix([])

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="status",
    ):
        ep.software_paper_claim_matrix(
            "A",
            status="unknown",
        )


def test_bundle_hash_coverage_and_readiness_contracts():
    bundle = _ready_bundle()

    assert bundle["eyeprocess_class"] == "eye_software_paper_evidence"
    assert bundle["schema_version"] == ("eyeprocess-paper-evidence-0.9")
    assert isinstance(bundle["bundle_hash"], str)
    assert len(bundle["bundle_hash"]) == 64

    coverage = ep.software_paper_coverage(bundle)
    row = coverage.iloc[0]
    assert row["n_claims"] == 2
    assert row["n_covered"] == 2
    assert row["coverage"] == pytest.approx(1.0)
    assert row["n_pending"] == 0
    assert row["n_unsupported"] == 0

    readiness = ep.software_paper_readiness(bundle)
    assert readiness["eyeprocess_class"] == "eye_software_paper_readiness"
    assert readiness["ready"] is True
    assert readiness["checks"]["satisfied"].all()


def test_incomplete_bundle_and_gap_analysis():
    claims = ep.software_paper_claim_matrix(
        ["Supported", "Pending"],
        evidence_id=["E1", None],
        status=["supported", "pending"],
    )
    bundle = ep.software_paper_evidence_bundle(
        claims=claims,
    )

    readiness = ep.software_paper_readiness(bundle)
    assert readiness["ready"] is False

    gaps = ep.software_paper_gap_analysis(bundle)
    requirements = set(gaps["requirement_gaps"]["requirement"])
    assert {
        "claims",
        "validation",
        "reproducibility",
        "examples",
        "articles",
    }.issubset(requirements)
    assert gaps["claim_gaps"]["claim"].tolist() == ["Pending"]

    relaxed = ep.software_paper_readiness(
        bundle,
        required_statuses=(
            "supported",
            "pending",
        ),
        require_validation=False,
        require_reproducibility=False,
        require_examples=False,
        require_articles=False,
    )
    assert relaxed["ready"] is True

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="required_statuses",
    ):
        ep.software_paper_readiness(
            bundle,
            required_statuses=[],
        )


def test_validation_table_dataframe_and_named_evidence_mapping():
    source = pd.DataFrame(
        {
            "metric": ["coverage"],
            "value": [0.95],
        }
    )
    returned = ep.software_paper_validation_table(source)
    pd.testing.assert_frame_equal(returned, source)

    mapped = ep.software_paper_validation_table(
        {
            "recovery": pd.DataFrame(
                {
                    "metric": ["bias"],
                    "value": [0.01],
                }
            )
        }
    )
    assert isinstance(mapped, pd.DataFrame)
    assert len(mapped) >= 1


def test_freeze_json_is_real_and_rds_is_explicitly_gated(tmp_path):
    bundle = _ready_bundle()
    path = tmp_path / "evidence.json"

    manifest = ep.freeze_software_paper_evidence(
        bundle,
        path,
    )
    assert path.exists()
    assert manifest.loc[0, "path"] == (path.resolve().as_posix())
    assert len(manifest.loc[0, "hash"]) == 32

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ("eyeprocess-paper-evidence-0.9")
    assert payload["bundle_hash"] == bundle["bundle_hash"]

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="R-specific",
    ):
        ep.freeze_software_paper_evidence(
            bundle,
            tmp_path / "evidence.rds",
        )


def test_human_readable_report_preserves_source_disclaimer(tmp_path):
    bundle = _ready_bundle()
    path = tmp_path / "evidence.md"

    returned = ep.write_software_paper_evidence(
        bundle,
        path,
    )
    assert Path(returned) == path

    text = path.read_text(encoding="utf-8")
    assert "# eyeprocess software-paper evidence bundle" in text
    assert "Descriptive readiness: **PASS**" in text
    assert "## Requirement audit" in text
    assert "## Claim coverage" in text
    assert "- Coverage: 1.0000" in text
    assert "does not establish external validity" in text
    assert "predict journal acceptance" in text


def test_paper_reproducibility_manifest_hashes_existing_files(tmp_path):
    bundle = _ready_bundle()

    manuscript = tmp_path / "paper.md"
    figure = tmp_path / "figure.txt"
    table = tmp_path / "table.csv"

    manuscript.write_text("# Paper\n", encoding="utf-8")
    figure.write_text("figure", encoding="utf-8")
    table.write_text("x\n1\n", encoding="utf-8")

    manifest = ep.paper_reproducibility_manifest(
        bundle,
        manuscript=manuscript,
        figures=[figure],
        tables=[table],
    )

    assert manifest["evidence_hash"] == bundle["bundle_hash"]
    assert len(manifest["files"]) == 3
    assert set(manifest["files"]["algorithm"]) == {"md5"}
    assert manifest["reproducibility"] is bundle["reproducibility"]

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="must exist",
    ):
        ep.paper_reproducibility_manifest(
            bundle,
            manuscript=tmp_path / "missing.md",
        )


def test_contracts_reject_non_bundle_inputs(tmp_path):
    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_software_paper_evidence",
    ):
        ep.software_paper_readiness({})

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_software_paper_evidence",
    ):
        ep.freeze_software_paper_evidence(
            {},
            tmp_path / "x.json",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_software_paper_evidence",
    ):
        ep.write_software_paper_evidence(
            {},
            tmp_path / "x.md",
        )

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="eye_software_paper_evidence",
    ):
        ep.paper_reproducibility_manifest({})
