from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep
import eyeprocesspy.vendor_corpus_10 as vc
from eyeprocesspy.exceptions import EyeProcessValidationError


def _source_csv(path: Path, **columns) -> Path:
    frame = pd.DataFrame(columns or {"x": [1.0, 2.0], "participant_id": ["A", "B"]})
    frame.to_csv(path, index=False)
    return path


def _register_basic(corpus: Path, source: Path, case_id: str = "case-1", **kwargs):
    return ep.register_validation_case(
        corpus,
        source,
        vendor=kwargs.pop("vendor", "Tobii"),
        case_id=case_id,
        device_model=kwargs.pop("device_model", "demo"),
        software_name=kwargs.pop("software_name", "Pro Lab"),
        software_version=kwargs.pop("software_version", "1.0"),
        **kwargs,
    )


def _eye_dataset():
    return ep.new_eye_dataset(
        recordings=pd.DataFrame(
            {"recording_id": ["r1"], "participant_id": ["p1"], "label": ["A"]}
        ),
        gaze_samples=pd.DataFrame(
            {
                "recording_id": ["r1", "r1"],
                "sample_id": ["s1", "s2"],
                "x": [0.1, 0.2],
                "flag": [True, False],
                "label": ["left", "right"],
            }
        ),
        validate=False,
    )


def test_vendor_private_mapping_argument_and_character_guards():
    obj = vc._EyeDict(answer=42)
    assert obj.answer == 42
    with pytest.raises(AttributeError, match="missing"):
        _ = obj.missing

    assert vc._vendor("Tobii Pro Lab") == "tobii"
    assert vc._vendor("Pupil Neon") == "pupillabs"
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        vc._vendor(None)

    assert vc._match_arg(("copy", "reference"), ["reference", "copy"], "mode") == "copy"
    with pytest.raises(EyeProcessValidationError, match="mode"):
        vc._match_arg([], ["reference", "copy"], "mode")

    assert vc._r_character(None) is pd.NA
    assert vc._r_character(pd.NA) is pd.NA
    assert vc._r_character(np.nan) is pd.NA
    assert vc._r_character(True) == "TRUE"
    assert vc._r_character(np.bool_(False)) == "FALSE"
    assert vc._r_character("x") == "x"
    assert vc._r_character(np.array([1, 2])).startswith("[")

    boolean = vc._coerce_bool_series(pd.Series([True, False], dtype=bool))
    assert boolean.tolist() == [True, False]
    mapped = vc._coerce_bool_series(pd.Series(["yes", "0", "unknown", None]))
    assert mapped.iloc[0] == True
    assert mapped.iloc[1] == False
    assert pd.isna(mapped.iloc[2])
    assert pd.isna(mapped.iloc[3])


def test_corpus_init_overwrite_registry_read_and_write_guards(tmp_path):
    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    assert Path(ep.init_vendor_corpus(corpus)) == corpus

    marker = corpus / "marker.txt"
    marker.write_text("old", encoding="utf-8")
    rebuilt = Path(ep.init_vendor_corpus(corpus, overwrite=True))
    assert rebuilt == corpus
    assert not marker.exists()

    file_target = tmp_path / "was-a-file"
    file_target.write_text("x", encoding="utf-8")
    converted = Path(ep.init_vendor_corpus(file_target, overwrite=True))
    assert converted.is_dir()

    missing = tmp_path / "missing-corpus"
    with pytest.raises(EyeProcessValidationError, match="registry is missing"):
        ep.read_vendor_registry(missing)

    empty_corpus = Path(ep.init_vendor_corpus(tmp_path / "empty-corpus"))
    (empty_corpus / "vendor-cases.csv").write_text("", encoding="utf-8")
    registry = ep.read_vendor_registry(empty_corpus)
    assert list(registry.columns) == vc.REGISTRY_COLUMNS
    assert registry.empty

    (empty_corpus / "vendor-cases.csv").write_text(
        "case_id,independent_source,licence_reviewed,redistribution_allowed\n"
        "x,yes,no,1\n",
        encoding="utf-8",
    )
    registry = ep.read_vendor_registry(empty_corpus)
    assert bool(registry.loc[0, "independent_source"])
    assert not bool(registry.loc[0, "licence_reviewed"])
    assert bool(registry.loc[0, "redistribution_allowed"])
    assert "vendor" in registry

    with pytest.raises(EyeProcessValidationError, match="data frame"):
        ep.write_vendor_registry({}, empty_corpus)
    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        ep.write_vendor_registry(pd.DataFrame(), tmp_path / "nowhere")

    minimal = pd.DataFrame({"case_id": ["only"]})
    path = Path(ep.write_vendor_registry(minimal, empty_corpus))
    assert path.exists()
    assert list(ep.read_vendor_registry(empty_corpus).columns) == vc.REGISTRY_COLUMNS


def test_case_file_iteration_fingerprint_and_copy_residuals(tmp_path):
    single = _source_csv(tmp_path / "one.csv")
    assert vc._iter_case_files(single, include_hidden=False) == [single.resolve()]

    with pytest.raises(EyeProcessValidationError, match="does not exist"):
        ep.fingerprint_validation_case(tmp_path / "absent")
    with pytest.raises(EyeProcessValidationError, match="Unsupported hash"):
        ep.fingerprint_validation_case(single, algorithms="definitely-not-a-hash")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EyeProcessValidationError, match="No files"):
        ep.fingerprint_validation_case(empty)

    md5_only = ep.fingerprint_validation_case(single, algorithms="md5")
    assert "sha256" not in md5_only
    assert md5_only.relative_path.tolist() == ["one.csv"]

    target = tmp_path / "copy.csv"
    assert vc._copy_case(single, target, "reference") == single.resolve()
    copied = vc._copy_case(single, target, "copy")
    assert copied.exists()
    with pytest.raises(EyeProcessValidationError, match="already exists"):
        vc._copy_case(single, target, "copy")

    source_dir = tmp_path / "dir-source"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("a", encoding="utf-8")
    directory_copy = vc._copy_case(source_dir, tmp_path / "dir-copy", "copy")
    assert (directory_copy / "a.txt").exists()


def test_registration_guards_auto_id_copy_and_duplicate(tmp_path):
    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    source = _source_csv(tmp_path / "source.csv")

    with pytest.raises(EyeProcessValidationError, match="source does not exist"):
        _register_basic(corpus, tmp_path / "missing.csv")
    with pytest.raises(EyeProcessValidationError, match="Vendor must be non-empty"):
        _register_basic(corpus, source, vendor="")
    with pytest.raises(EyeProcessValidationError, match="support_level"):
        _register_basic(corpus, source, support_level="bad")
    with pytest.raises(EyeProcessValidationError, match="mode"):
        _register_basic(corpus, source, mode="bad")
    with pytest.raises(EyeProcessValidationError, match="required"):
        _register_basic(corpus, source, device_model="")
    with pytest.raises(EyeProcessValidationError, match="sampling_rate_hz"):
        _register_basic(corpus, source, sampling_rate_hz=-1)
    with pytest.raises(EyeProcessValidationError, match="independent-source"):
        _register_basic(
            corpus,
            source,
            support_level="empirically-validated",
            independent_source=True,
            licence_reviewed=False,
        )

    auto = ep.register_validation_case(
        corpus,
        source,
        vendor="Gazepoint Analysis",
        device_model="GP3",
        software_name="Analysis",
        software_version="7.2",
        mode="copy",
        hardware_version=True,
        notes=False,
    )
    assert auto.case_id.iloc[0].startswith("gazepoint-")
    assert Path(auto.source_path.iloc[0]).exists()
    assert auto.hardware_version.iloc[0] == "TRUE"
    assert auto.notes.iloc[0] == "FALSE"

    with pytest.raises(EyeProcessValidationError, match="already exists"):
        ep.register_validation_case(
            corpus,
            source,
            vendor="Gazepoint",
            case_id=auto.case_id.iloc[0],
            device_model="GP3",
            software_name="Analysis",
            software_version="7.2",
        )


def test_redaction_guards_text_redactor_tsv_copy_and_overwrite(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(EyeProcessValidationError, match="source does not exist"):
        ep.redact_validation_case(missing, tmp_path / "out", salt="s")

    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "participant_id": ["A", "B"],
            "email": ["a@example.com", "b@example.com"],
            "comment": ["Alpha", "Beta"],
        }
    ).to_csv(source / "a.csv", index=False)
    pd.DataFrame({"subject": ["S1"], "text": ["Keep"]}).to_csv(
        source / "b.tsv", sep="\t", index=False
    )
    (source / "blob.bin").write_bytes(b"abc")

    with pytest.raises(EyeProcessValidationError, match="must not be inside"):
        ep.redact_validation_case(source, source / "nested", salt="s")
    with pytest.raises(EyeProcessValidationError, match="non-empty"):
        ep.redact_validation_case(source, tmp_path / "nosalt", salt="")

    output = tmp_path / "redacted"
    output.mkdir()
    with pytest.raises(EyeProcessValidationError, match="already exists"):
        ep.redact_validation_case(source, output, salt="s")

    output_file = tmp_path / "redacted-file"
    output_file.write_text("old", encoding="utf-8")
    result = ep.redact_validation_case(
        source,
        output_file,
        salt="s",
        overwrite=True,
        copy_non_tabular=True,
        text_redactor=lambda series, column: series.astype("string").str.lower(),
    )
    assert (Path(result.output_path) / "blob.bin").exists()
    manifest = result.manifest.set_index("relative_path")
    assert manifest.loc["blob.bin", "status"] == "copied-unchanged"
    csv = pd.read_csv(Path(result.output_path) / "a.csv")
    assert "email" not in csv
    assert csv.comment.tolist() == ["alpha", "beta"]
    tsv = pd.read_csv(Path(result.output_path) / "b.tsv", sep="\t")
    assert tsv.subject.str.upper().str.startswith("ID").all()

    second = tmp_path / "replace-dir"
    second.mkdir()
    (second / "stale.txt").write_text("stale", encoding="utf-8")
    ep.redact_validation_case(source, second, salt="s2", overwrite=True)
    assert not (second / "stale.txt").exists()


def test_redaction_read_error_is_retained_in_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    original = vc._read_delimited

    def fail_one(path):
        if Path(path).name == "bad.csv":
            raise ValueError("forced read failure")
        return original(path)

    monkeypatch.setattr(vc, "_read_delimited", fail_one)
    result = ep.redact_validation_case(source, tmp_path / "out", salt="salt")
    status = result.manifest.set_index("relative_path").loc["bad.csv", "status"]
    assert status.startswith("read-error:")


def test_semantics_empty_invalid_filter_unknown_risk_and_update(tmp_path):
    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    (corpus / "vendor-semantics.csv").write_text("", encoding="utf-8")
    empty = vc._read_semantics(corpus)
    assert empty.empty
    assert list(empty.columns) == vc.SEMANTICS_COLUMNS

    with pytest.raises(EyeProcessValidationError, match="loss_risk"):
        ep.register_vendor_semantics(
            corpus, "Tobii", "x", "x", "gaze_samples", "x", loss_risk="bad"
        )

    one = ep.register_vendor_semantics(
        corpus,
        "Tobii Pro Lab",
        "x",
        "horizontal",
        "gaze_samples",
        "x",
        transformation="identity",
        loss_risk="none",
    )
    assert len(one) == 1

    with pytest.raises(EyeProcessValidationError, match="corpus path"):
        ep.compare_vendor_semantics(object())
    with pytest.raises(EyeProcessValidationError, match="incomplete"):
        ep.compare_vendor_semantics(pd.DataFrame({"vendor": ["tobii"]}))

    filtered = ep.compare_vendor_semantics(one, vendors="Gazepoint")
    assert filtered.empty

    unknown = one.copy()
    unknown.loc[0, "loss_risk"] = "mystery"
    unknown.loc[0, "transformation"] = pd.NA
    comparison = ep.compare_vendor_semantics(unknown)
    assert pd.isna(comparison.maximum_loss_risk.iloc[0])
    assert comparison.transformations.iloc[0] == ""


def test_roundtrip_private_comparison_and_validation_guards():
    a = pd.DataFrame({"x": [1.0, np.nan], "label": ["a", "b"], "flag": [True, False]})
    b = pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "c"], "flag": [True, True]})

    assert vc._with_occurrence(a, []).equals(a)
    empty = vc._compare_columns(pd.DataFrame({"id": [1]}), pd.DataFrame({"id": [1]}), ["id"], 0)
    assert empty.empty

    detail = vc._compare_columns(a, b, [], 0)
    assert set(detail.column) == {"x", "label", "flag"}
    assert detail.loc[detail.column.eq("label"), "mismatch_rate"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(detail.loc[detail.column.eq("flag"), "max_absolute_difference"].iloc[0])

    source = _eye_dataset()
    with pytest.raises(EyeProcessValidationError, match="eye_dataset"):
        ep.audit_roundtrip_loss(source, {})
    with pytest.raises(EyeProcessValidationError, match="tolerance"):
        ep.audit_roundtrip_loss(source, source.copy(), tolerance="bad")
    with pytest.raises(EyeProcessValidationError, match="tolerance"):
        ep.audit_roundtrip_loss(source, source.copy(), tolerance=-1)

    only = ep.audit_roundtrip_loss(source, source.copy(), tables="gaze_samples")
    assert only.summary.table.tolist() == ["gaze_samples"]
    none = ep.audit_roundtrip_loss(source, source.copy(), tables=["not_canonical"])
    assert none.summary.empty


def test_roundtrip_detects_column_and_row_loss_and_no_shared_columns():
    source = _eye_dataset()
    changed = source.copy()
    changed["gaze_samples"] = changed["gaze_samples"].iloc[[0]].drop(
        columns=["label"]
    )
    audit = ep.audit_roundtrip_loss(source, changed, tables=["gaze_samples"])
    row = audit.summary.iloc[0]
    assert row.status == "review"
    assert row.row_difference == -1
    assert "label" in row.missing_source_columns

    no_shared = vc._compare_columns(
        pd.DataFrame({"id": [1], "a": [1]}),
        pd.DataFrame({"id": [1], "b": [1]}),
        ["id"],
        0,
    )
    assert no_shared.empty


def test_vendor_field_coverage_input_and_required_field_guards(tmp_path):
    with pytest.raises(EyeProcessValidationError, match="data frames"):
        ep.audit_vendor_field_coverage({}, pd.DataFrame())
    with pytest.raises(EyeProcessValidationError, match="canonical table"):
        ep.audit_vendor_field_coverage(
            pd.DataFrame({"vendor": ["tobii"]}),
            pd.DataFrame({"canonical_table": ["gaze_samples"]}),
        )

    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    ep.register_vendor_semantics(
        corpus, "Tobii", "x", "x", "gaze_samples", "x", loss_risk="none"
    )
    required = pd.DataFrame(
        {"canonical_table": ["gaze_samples"], "canonical_field": ["x"]}
    )
    coverage = ep.audit_vendor_field_coverage(corpus, required)
    assert coverage.supported.tolist() == [True]


def test_validation_pass_and_promotion_guard_matrix(tmp_path):
    assert vc._validation_pass(True)
    assert not vc._validation_pass(False)
    assert vc._validation_pass({"status": "PASSED"})
    assert vc._validation_pass(SimpleNamespace(status="success"))
    assert not vc._validation_pass({"other": "x"})
    assert not vc._validation_pass(SimpleNamespace(status="failed"))

    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    source = _source_csv(tmp_path / "source.csv")
    _register_basic(
        corpus,
        source,
        case_id="case",
        independent_source=False,
        licence_reviewed=True,
        notes="existing",
    )

    with pytest.raises(EyeProcessValidationError, match="reviewer"):
        ep.promote_vendor_support(corpus, "case", validation=True, reviewer=" ")
    with pytest.raises(EyeProcessValidationError, match="Unknown case"):
        ep.promote_vendor_support(corpus, "absent", validation=True, reviewer="r")

    fixture = ep.promote_vendor_support(
        corpus,
        "case",
        level="fixture-tested",
        validation={"status": "pass"},
        reviewer="r1",
        notes="fixture",
    )
    assert "existing" in fixture.notes.iloc[0]
    assert "fixture" in fixture.notes.iloc[0]

    with pytest.raises(EyeProcessValidationError, match="independent-source"):
        ep.promote_vendor_support(
            corpus,
            "case",
            level="empirically-validated",
            validation=SimpleNamespace(status="pass"),
            reviewer="r2",
        )


def test_compatibility_matrix_invalid_minimum_string_alias_and_missing_evidence():
    with pytest.raises(EyeProcessValidationError, match="corpus path or registry"):
        ep.build_compatibility_matrix(object())
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        ep.build_compatibility_matrix(pd.DataFrame(), min_empirical_cases="bad")
    with pytest.raises(EyeProcessValidationError, match="positive integer"):
        ep.build_compatibility_matrix(pd.DataFrame(), min_empirical_cases=0)

    registry = pd.DataFrame(
        {
            "vendor": ["tobii", "custom"],
            "support_level": ["declared", "unknown"],
            "device_model": ["T1", "C1"],
            "software_version": ["1", "2"],
        }
    )
    matrix = ep.build_compatibility_matrix(
        registry,
        required_vendors="Tobii Pro Lab",
        min_empirical_cases=1,
    )
    assert matrix.vendor.tolist() == ["tobii", "custom"]
    tobii = matrix.loc[matrix.vendor.eq("tobii")].iloc[0]
    custom = matrix.loc[matrix.vendor.eq("custom")].iloc[0]
    assert tobii.highest_support == "declared"
    assert custom.highest_support == "none"
    assert not bool(tobii.production_claim_allowed)
    assert not bool(custom.production_claim_allowed)


def test_markdown_and_case_report_missing_fingerprint_object_validation_and_roundtrip(tmp_path):
    assert vc._markdown_table(pd.DataFrame()) == "_No rows available._"
    rendered = vc._markdown_table(pd.DataFrame({"a": [1, pd.NA], "b": ["x", "y"]}))
    assert "| a | b |" in rendered
    assert "|  | y |" in rendered

    corpus = Path(ep.init_vendor_corpus(tmp_path / "corpus"))
    source = _source_csv(tmp_path / "source.csv")
    _register_basic(corpus, source, case_id="report")
    (corpus / "fingerprints" / "report.csv").unlink()

    with pytest.raises(EyeProcessValidationError, match="Unknown case"):
        ep.write_vendor_case_report(corpus, "absent", tmp_path / "missing.md")

    roundtrip = vc.EyeRoundtripLossAudit(
        summary=pd.DataFrame({"table": ["gaze_samples"], "status": ["lossless"]}),
        details=pd.DataFrame(),
        tolerance=1e-8,
    )
    path = Path(
        ep.write_vendor_case_report(
            corpus,
            "report",
            tmp_path / "nested" / "report.md",
            validation=SimpleNamespace(status="passed"),
            roundtrip=roundtrip,
        )
    )
    text = path.read_text(encoding="utf-8")
    assert "## Validation" in text
    assert "SimpleNamespace" in text
    assert "## Round-trip loss" in text
    assert "_No rows available._" in text
