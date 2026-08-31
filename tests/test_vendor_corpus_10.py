from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "audit_roundtrip_loss",
    "audit_vendor_field_coverage",
    "build_compatibility_matrix",
    "compare_vendor_semantics",
    "fingerprint_validation_case",
    "init_vendor_corpus",
    "promote_vendor_support",
    "read_vendor_registry",
    "redact_validation_case",
    "register_validation_case",
    "register_vendor_semantics",
    "write_vendor_case_report",
    "write_vendor_registry",
]


def test_public_r025_exports_are_callable():
    assert len(TARGETS) == 13
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_corpus_registry_and_case_retain_version_specific_evidence(tmp_path):
    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    assert (corpus / "vendor-cases.csv").exists()
    assert (corpus / "vendor-semantics.csv").exists()
    assert (corpus / "README.md").exists()

    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame({"time": [1, 2, 3], "x": [0.1, 0.2, 0.3]}).to_csv(
        source / "export.csv",
        index=False,
    )

    case = ep.register_validation_case(
        corpus,
        source,
        vendor="Tobii",
        case_id="tobii-demo",
        support_level="fixture-tested",
        device_model="synthetic",
        software_name="Tobii Pro Lab",
        software_version="test",
        sampling_rate_hz=60,
        coordinate_system="normalized",
        timebase="seconds",
        event_semantics="trial markers",
        ocular_structure="binocular",
        missingness_convention="NA",
        vendor_fixations=True,
        independent_source=False,
        licence_reviewed=True,
    )
    assert case.eyeprocess_class == "eye_validation_case"

    registry = ep.read_vendor_registry(corpus)
    assert registry["vendor"].tolist() == ["tobii"]
    assert registry["support_level"].tolist() == ["fixture-tested"]
    assert registry["software_version"].tolist() == ["test"]
    assert registry["sampling_rate_hz"].iloc[0] == pytest.approx(60)

    fingerprint = ep.fingerprint_validation_case(source)
    assert fingerprint.eyeprocess_class == "eye_validation_case_fingerprint"
    assert fingerprint["case_fingerprint"].str.startswith("case-").all()
    assert fingerprint["md5"].str.len().eq(32).all()
    assert fingerprint["sha256"].str.len().eq(64).all()

    path = ep.write_vendor_registry(registry, corpus)
    assert Path(path).exists()


def test_empirical_registration_requires_independent_licensed_evidence(tmp_path):
    corpus = ep.init_vendor_corpus(tmp_path / "corpus")
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1]}).to_csv(source, index=False)

    with pytest.raises(
        ep.EyeProcessValidationError,
        match="independent-source",
    ):
        ep.register_validation_case(
            corpus,
            source,
            vendor="Gazepoint",
            device_model="GP3",
            software_name="Analysis",
            software_version="7.2",
            support_level="empirically-validated",
            independent_source=False,
            licence_reviewed=True,
        )


def test_fingerprint_hidden_file_policy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "visible.txt").write_text("visible", encoding="utf-8")
    (source / ".hidden.txt").write_text("hidden", encoding="utf-8")

    ordinary = ep.fingerprint_validation_case(source)
    hidden = ep.fingerprint_validation_case(source, include_hidden=True)

    assert ordinary["relative_path"].tolist() == ["visible.txt"]
    assert set(hidden["relative_path"]) == {".hidden.txt", "visible.txt"}


def test_redaction_pseudonymizes_ids_drops_pii_and_excludes_binary(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "participant_id": ["A", "B"],
            "recording_id": ["R1", "R2"],
            "name": ["Alice", "Bob"],
            "comment": ["keep", "keep too"],
        }
    ).to_csv(source / "export.csv", index=False)
    (source / "video.bin").write_bytes(b"\x00\x01secret")

    output = tmp_path / "redacted"
    result = ep.redact_validation_case(
        source,
        output,
        salt="project-secret",
    )
    assert result.eyeprocess_class == "eye_redaction_result"

    redacted = pd.read_csv(output / "export.csv")
    assert "name" not in redacted
    assert redacted["participant_id"].str.startswith("ID").all()
    assert redacted["recording_id"].str.startswith("ID").all()
    assert not (output / "video.bin").exists()

    manifest = result["manifest"].set_index("relative_path")
    assert manifest.loc["export.csv", "status"] == "redacted"
    assert manifest.loc["video.bin", "status"] == "excluded"
    assert (output / "redaction-manifest.csv").exists()

    second = ep.redact_validation_case(
        source,
        tmp_path / "redacted-2",
        salt="project-secret",
    )
    again = pd.read_csv(Path(second["output_path"]) / "export.csv")
    assert redacted["participant_id"].tolist() == again["participant_id"].tolist()


def test_vendor_semantics_update_and_maximum_loss_risk(tmp_path):
    corpus = ep.init_vendor_corpus(tmp_path / "corpus")
    ep.register_vendor_semantics(
        corpus,
        "Tobii",
        "x",
        "horizontal gaze",
        "gaze_samples",
        "x",
        transformation="identity",
        loss_risk="none",
    )
    ep.register_vendor_semantics(
        corpus,
        "Gazepoint",
        "BPOGX",
        "best point of gaze x",
        "gaze_samples",
        "x",
        transformation="rename",
        loss_risk="low",
    )

    comparison = ep.compare_vendor_semantics(corpus)
    assert comparison.eyeprocess_class == "eye_vendor_semantic_comparison"
    assert comparison["vendors"].iloc[0] == 2
    assert comparison["maximum_loss_risk"].iloc[0] == "low"

    semantics = ep.register_vendor_semantics(
        corpus,
        "Tobii",
        "x",
        "horizontal gaze updated",
        "gaze_samples",
        "x",
        transformation="scale",
        loss_risk="moderate",
    )
    assert len(semantics) == 2
    updated = ep.compare_vendor_semantics(corpus)
    assert updated["maximum_loss_risk"].iloc[0] == "moderate"


def test_roundtrip_loss_audit_detects_loss_and_numeric_difference():
    source = ep.new_eye_dataset(
        recordings=pd.DataFrame({"recording_id": ["r1"], "participant_id": ["p1"]}),
        gaze_samples=pd.DataFrame(
            {
                "recording_id": ["r1", "r1"],
                "sample_id": ["s1", "s2"],
                "x": [0.1, 0.2],
                "y": [0.3, 0.4],
            }
        ),
        validate=False,
    )
    exact = source.copy()
    lossless = ep.audit_roundtrip_loss(
        source,
        exact,
        tables=["recordings", "gaze_samples"],
    )
    assert lossless.eyeprocess_class == "eye_roundtrip_loss_audit"
    assert lossless["summary"]["status"].eq("lossless").all()

    changed = source.copy()
    changed["gaze_samples"] = changed["gaze_samples"].copy()
    changed["gaze_samples"].loc[1, "x"] = 0.8
    audit = ep.audit_roundtrip_loss(
        source,
        changed,
        tables=["gaze_samples"],
        tolerance=1e-8,
    )
    row = audit["summary"].iloc[0]
    assert row["status"] == "review"
    assert row["maximum_mismatch_rate"] > 0
    detail = audit["details"].loc[audit["details"]["column"].eq("x")].iloc[0]
    assert detail["max_absolute_difference"] == pytest.approx(0.6)


def test_vendor_field_coverage_is_vendor_by_required_field():
    semantics = pd.DataFrame(
        {
            "vendor": ["tobii", "gazepoint", "gazepoint"],
            "canonical_table": ["gaze_samples"] * 3,
            "canonical_field": ["x", "x", "y"],
        }
    )
    required = pd.DataFrame(
        {
            "canonical_table": ["gaze_samples", "gaze_samples"],
            "canonical_field": ["x", "y"],
        }
    )
    coverage = ep.audit_vendor_field_coverage(semantics, required)
    tobii = coverage.loc[coverage["vendor"].eq("tobii")]
    assert tobii["supported"].tolist() == [True, False]


def test_promotion_requires_passing_validation_and_empirical_evidence(tmp_path):
    corpus = ep.init_vendor_corpus(tmp_path / "corpus")
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1]}).to_csv(source, index=False)

    ep.register_validation_case(
        corpus,
        source,
        vendor="Gazepoint",
        case_id="gp3",
        device_model="GP3",
        software_name="Analysis",
        software_version="7.2",
        independent_source=True,
        licence_reviewed=True,
    )
    with pytest.raises(ep.EyeProcessValidationError, match="passing"):
        ep.promote_vendor_support(
            corpus,
            "gp3",
            validation=False,
            reviewer="reviewer-1",
        )

    promoted = ep.promote_vendor_support(
        corpus,
        "gp3",
        level="empirically-validated",
        validation=True,
        reviewer="reviewer-1",
        notes="real export reviewed",
    )
    assert promoted["support_level"].iloc[0] == "empirically-validated"
    assert promoted["status"].iloc[0] == "validated"


def test_compatibility_matrix_separates_declared_fixture_and_empirical_evidence():
    registry = pd.DataFrame(
        {
            "case_id": ["g1", "g2", "t1"],
            "vendor": ["gazepoint", "gazepoint", "tobii"],
            "support_level": [
                "empirically-validated",
                "empirically-validated",
                "fixture-tested",
            ],
            "independent_source": [True, True, False],
            "licence_reviewed": [True, True, True],
            "status": ["validated", "validated", "validated"],
            "device_model": ["GP3", "GP3", "demo"],
            "software_version": ["7.2", "7.2", "x"],
        }
    )
    matrix = ep.build_compatibility_matrix(
        registry,
        required_vendors=["gazepoint", "tobii"],
        min_empirical_cases=2,
    )
    gp = matrix.loc[matrix["vendor"].eq("gazepoint")].iloc[0]
    tobii = matrix.loc[matrix["vendor"].eq("tobii")].iloc[0]
    assert bool(gp["production_claim_allowed"])
    assert not bool(tobii["production_claim_allowed"])
    assert gp["empirical_cases"] == 2
    assert tobii["fixture_tested_cases"] == 1


def test_vendor_case_report_preserves_claim_boundary(tmp_path):
    corpus = ep.init_vendor_corpus(tmp_path / "corpus")
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1]}).to_csv(source, index=False)
    ep.register_validation_case(
        corpus,
        source,
        vendor="Tobii",
        case_id="t1",
        device_model="demo",
        software_name="Tobii Pro Lab",
        software_version="test",
        support_level="fixture-tested",
        independent_source=False,
        licence_reviewed=True,
    )
    report_path = Path(
        ep.write_vendor_case_report(
            corpus,
            "t1",
            tmp_path / "report.md",
            validation={"status": "pass"},
        )
    )
    report = report_path.read_text(encoding="utf-8")
    assert "## Registration" in report
    assert "## Source fingerprint" in report
    assert "## Claim boundary" in report
    assert "Fixture tests are not independent empirical validation" in report
