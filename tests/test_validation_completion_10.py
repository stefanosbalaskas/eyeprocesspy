from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eyeprocesspy as ep

FIX = Path(__file__).parent / "fixtures" / "gazepoint"

TARGETS = [
    "preprocessing_multiverse",
    "benchmark_eyeprocess",
    "reporting_guideline_audit",
    "write_reporting_guideline_report",
    "create_public_benchmark",
    "write_software_paper_scaffold",
    "run_eyeprocess_validation_program",
]


def _eye():
    return ep.read_gazepoint_folder(FIX, recording_id="R1", quiet=True)


def _corpus():
    vendors = ["gazepoint", "tobii", "pupillabs", "eyelink", "smi"]
    summary = pd.DataFrame(
        {
            "case_id": [f"{vendor}-{case}" for vendor in vendors for case in (1, 2)],
            "vendor": [vendor for vendor in vendors for _ in (1, 2)],
            "status": "pass",
            "software_version": "1.0",
            "device_model": "device",
            "independent_source": True,
            "licence_reviewed": True,
        }
    )
    return {"summary": summary, "manifest": summary.copy(), "status": "pass"}


def test_public_r021_completion_exports():
    assert len(TARGETS) == 7
    assert all(callable(getattr(ep, name, None)) for name in TARGETS)


def test_preprocessing_multiverse_preserves_results_and_failure_rows():
    result = ep.preprocessing_multiverse(
        np.arange(1, 6),
        {"a": 1, "b": 2, "bad": "bad"},
        transform=lambda x, spec: x * spec if isinstance(spec, int) else (_ for _ in ()).throw(ValueError("bad spec")),
        analyse=lambda x: pd.DataFrame({"estimate": [float(np.mean(x))]}),
    )
    assert result.eyeprocess_class == "eye_multiverse"
    assert set(result["results"].specification) == {"a", "b", "bad"}
    assert result["results"].loc[result["results"].specification.eq("bad"), "error"].notna().all()


def test_benchmark_contract():
    result = ep.benchmark_eyeprocess(lambda: sum(range(101)), iterations=2, label="smoke")
    assert result.attrs["eyeprocess_class"] == "eye_benchmark"
    assert list(result.columns) == [
        "label",
        "iteration",
        "elapsed_seconds",
        "result_size_bytes",
    ]
    assert len(result) == 2
    assert (result.elapsed_seconds >= 0).all()
    assert (result.result_size_bytes > 0).all()


def test_reporting_audit_and_report(tmp_path):
    audit = ep.reporting_guideline_audit(_eye())
    assert audit.attrs["eyeprocess_class"] == "eye_reporting_audit"
    assert len(audit) == 12
    assert set(["section", "item", "covered", "status"]).issubset(audit.columns)
    assert audit.loc[audit.section.eq("interpretation"), "covered"].all()

    path = Path(ep.write_reporting_guideline_report(audit, tmp_path / "audit.md"))
    assert path.exists()
    assert "Eye-tracking reporting-guideline audit" in path.read_text(encoding="utf-8")


def test_public_benchmark_is_deidentified_canonical_bundle(tmp_path):
    path = Path(
        ep.create_public_benchmark(
            _eye(),
            tmp_path / "benchmark",
            max_participants=2,
            include_samples=False,
            overwrite=True,
        )
    )
    assert path.is_dir()
    assert (path / "manifest.json").exists()
    assert (path / "reporting-guideline-audit.csv").exists()

    roundtrip = ep.import_canonical(path)
    assert roundtrip["gaze_samples"].empty
    assert roundtrip["eye_samples"].empty
    assert roundtrip["biometrics"].empty


def test_software_paper_scaffold(tmp_path):
    path = Path(ep.write_software_paper_scaffold(tmp_path / "paper.Rmd"))
    text = path.read_text(encoding="utf-8")
    assert "# Validation programme" in text
    assert "# Limitations and responsible interpretation" in text
    assert "Raven reproduction only after data/licensing review" in text


def test_complete_validation_program_writes_frozen_outputs_without_fake_rds(tmp_path):
    output = tmp_path / "programme"
    result = ep.run_eyeprocess_validation_program(
        _corpus(),
        output,
        benchmark_jobs={"smoke": lambda: sum(range(101))},
        reporting_dataset=_eye(),
        overwrite=True,
    )

    assert result.eyeprocess_class == "eye_validation_program"
    required = [
        "vendor-validation.md",
        "benchmarks.csv",
        "reporting-guideline-audit.csv",
        "reporting-guideline-audit.md",
        "advanced-model-evidence.csv",
        "advanced-model-evidence.md",
        "eyeprocess-software-paper.Rmd",
        "validation-program.json",
        "serialization-boundary.md",
        "plots/vendor-pass-rate.png",
        "plots/benchmarks.png",
        "plots/reporting-guideline-coverage.png",
        "plots/advanced-model-evidence.png",
    ]
    assert all((output / name).exists() for name in required)
    assert not list(output.rglob("*.rds"))


def test_output_overwrite_guard(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ep.EyeProcessValidationError, match="already exists"):
        ep.run_eyeprocess_validation_program(_corpus(), output)
