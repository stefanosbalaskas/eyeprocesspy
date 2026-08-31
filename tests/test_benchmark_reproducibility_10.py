from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import eyeprocesspy as ep

TARGETS = [
    "eyeprocess_benchmark_study",
    "read_benchmark_table",
    "benchmark_expected_outputs",
    "import_benchmark_study",
    "validate_benchmark_study",
    "run_benchmark_reproduction",
    "write_benchmark_data_dictionary",
    "package_reproducibility_manifest",
    "verify_reproducibility_manifest",
    "write_software_paper_reproduction",
    "audit_benchmark_release",
]


def test_r029_public_exports_are_callable():
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_bundled_benchmark_contains_all_multimodal_layers():
    study = ep.eyeprocess_benchmark_study()
    assert study.eyeprocess_class == "eye_benchmark_study"
    required = {
        "participants",
        "items",
        "responses",
        "gaze_samples",
        "events",
        "aoi_definitions",
        "pupil_samples",
        "quality",
        "provenance",
    }
    assert required.issubset(set(study["manifest"]["table"].astype(str)))


def test_benchmark_fingerprints_and_relations_validate():
    validation = ep.validate_benchmark_study()
    assert validation.eyeprocess_class == "eye_benchmark_validation"
    assert validation["valid"] is True
    assert bool(validation["files"]["exists"].all())
    assert bool(validation["files"]["bytes_match"].all())
    assert bool(validation["files"]["hash_match"].all())
    assert bool(validation["relations"]["passed"].all())


def test_benchmark_logical_columns_have_stable_boolean_types():
    study = ep.eyeprocess_benchmark_study()
    gaze = ep.read_benchmark_table(study, "gaze_samples")
    pupil = ep.read_benchmark_table(study, "pupil_samples")
    for frame, column in [(gaze, "valid"), (pupil, "blink"), (pupil, "valid")]:
        assert pd.api.types.is_bool_dtype(frame[column].dtype)
        assert not bool(frame[column].isna().any())
    assert float(gaze["valid"].mean()) == pytest.approx(0.96375, abs=1e-10)


def test_benchmark_expected_outputs_reproduce_exactly():
    result = ep.run_benchmark_reproduction()
    assert result.eyeprocess_class == "eye_benchmark_reproduction"
    assert result["passed"] is True
    assert bool(result["comparison"]["passed"].all())


def test_import_benchmark_prefers_canonical_dataset_when_available():
    imported = ep.import_benchmark_study()
    assert ep.is_eye_dataset(imported) or getattr(imported, "eyeprocess_class", None) == "eye_benchmark_tables"


def test_data_dictionary_and_release_audit(tmp_path: Path):
    dictionary = Path(ep.write_benchmark_data_dictionary(path=tmp_path / "dictionary.md"))
    assert dictionary.is_file()
    text = dictionary.read_text(encoding="utf-8")
    assert "synthetic" in text.lower()
    audit = ep.audit_benchmark_release()
    assert audit["ready"] is True
    assert bool(audit["findings"]["passed"].all())


def test_reproducibility_manifest_detects_change(tmp_path: Path):
    target = tmp_path / "payload.txt"
    target.write_text("alpha\n", encoding="utf-8")
    manifest = ep.package_reproducibility_manifest(tmp_path)
    first = ep.verify_reproducibility_manifest(manifest)
    assert bool(first["unchanged"].all())
    target.write_text("beta\n", encoding="utf-8")
    second = ep.verify_reproducibility_manifest(manifest)
    assert not bool(second["unchanged"].all())


def test_software_paper_scaffold_is_self_contained(tmp_path: Path):
    directory = tmp_path / "software-paper"
    manifest = ep.write_software_paper_reproduction(directory)
    assert (directory / "scripts" / "run_reproduction.py").is_file()
    assert (directory / "README.md").is_file()
    assert manifest.eyeprocess_class == "eye_reproducibility_manifest"
    assert bool(ep.verify_reproducibility_manifest(manifest)["unchanged"].all())
    with pytest.raises(Exception):
        ep.write_software_paper_reproduction(directory)
