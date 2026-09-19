# ruff: noqa: I001
from __future__ import annotations

import re
from pathlib import Path

import eyeprocesspy as ep


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"


def test_detector_multiverse_documentation_surface_exists():
    pages = [
        "guides/event-detector-multiverse.md",
        "guides/event-detection-analytical-choice.md",
        "guides/building-detector-multiverse.md",
        "guides/comparing-event-catalogues.md",
        "guides/detector-aoi-features.md",
        "guides/detector-statistical-inference.md",
        "guides/reporting-detector-sensitivity.md",
        "guides/detector-multiverse-decision-guide.md",
        "guides/detector-multiverse-visual-atlas.md",
        "guides/detector-multiverse-reporting-template.md",
        "guides/detector-multiverse-troubleshooting.md",
        "examples/detector-multiverse-disclosure.md",
        "examples/detector-multiverse-failure-clinic.md",
        "reference/detector-multiverse.md",
    ]
    for page in pages:
        assert (DOCS / page).exists(), page


def test_detector_multiverse_reference_only_names_public_api():
    text = (DOCS / "reference/detector-multiverse.md").read_text(encoding="utf-8")
    names = re.findall(r"::: eyeprocesspy\.([A-Za-z0-9_]+)", text)
    assert len(names) >= 20
    assert all(callable(getattr(ep, name, None)) for name in names)


def test_new_documentation_internal_links_resolve():
    pages = list((DOCS / "guides").glob("*detector*.md")) + [
        DOCS / "examples/detector-multiverse-disclosure.md",
        DOCS / "examples/detector-multiverse-failure-clinic.md",
    ]
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target_path = target.split("#", 1)[0]
            resolved = (page.parent / target_path).resolve()
            assert resolved.exists(), f"{page}: {target}"


def test_worked_example_is_runnable_and_records_optional_backend_failure():
    import importlib.util

    path = ROOT / "examples" / "detector_multiverse_worked.py"
    spec = importlib.util.spec_from_file_location("detector_worked", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _, result, inference, summaries = module.main(n_participants=4)
    assert not summaries["event_summary"].empty
    assert not summaries["feature_sensitivity"].empty
    assert "remodnav" in set(result.status.detector_id)
    assert not inference.coefficients.empty


def test_method_landing_page_keeps_interpretation_limitations_and_reporting_handoff():
    text = (DOCS / "guides/event-detector-multiverse.md").read_text(encoding="utf-8")
    assert "## Interpretation" in text
    assert "## Limitations" in text
    assert "## Reporting and API links" in text
    assert "reporting-detector-sensitivity.md" in text
    assert "../reference/detector-multiverse.md" in text


def test_generated_report_uses_planned_detector_denominator():
    text = (DOCS / "assets" / "detector-multiverse" / "detector-multiverse-report.md").read_text(
        encoding="utf-8"
    )
    assert "six detector specifications were planned" in text
    assert "planned-specification convergence rate is 5/6 (0.833)" in text


def test_detector_failure_clinic_is_runnable():
    import importlib.util

    path = ROOT / "examples" / "detector_multiverse_failure_clinic.py"
    spec = importlib.util.spec_from_file_location("detector_failure_clinic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.main()
    assert set(result) == {
        "detection_status",
        "detection_failures",
        "model_input_audit",
        "model_input_warnings",
        "invalid_callback_failures",
    }
    audit = result["model_input_audit"]
    ivt = audit[audit.detector_id.eq("ivt30")].iloc[0]
    assert ivt.quality_excluded_rows == 1
    assert ivt.outcome_missing_rows == 1
    assert ivt.status == "modelled"
    invalid = result["invalid_callback_failures"]
    message = invalid[invalid.detector_id.eq("ivt30")].iloc[0].error
    assert "at most one row per coefficient term" in message


def test_detector_guidance_documents_input_audit_and_failure_rules():
    inference = (DOCS / "guides/detector-statistical-inference.md").read_text(encoding="utf-8")
    troubleshooting = (DOCS / "guides/detector-multiverse-troubleshooting.md").read_text(
        encoding="utf-8"
    )
    reporting = (DOCS / "guides/detector-multiverse-reporting-template.md").read_text(
        encoding="utf-8"
    )
    assert "input_audit" in inference
    assert "outcome_missing_rows" in inference
    assert "no_model_data" in troubleshooting
    assert "planned-specification convergence rate" in reporting


def test_detector_visual_atlas_and_assets_are_retained():
    atlas = DOCS / "guides" / "detector-multiverse-visual-atlas.md"
    text = atlas.read_text(encoding="utf-8")
    assets = [
        "detector-agreement.svg",
        "disclosure-dwell-by-detector.svg",
        "condition-coefficient-stability.svg",
        "robustness-denominator.svg",
        "model-input-audit.svg",
    ]
    for asset in assets:
        assert (DOCS / "assets" / "detector-multiverse" / asset).exists(), asset
        assert asset in text
    assert "5/6" in text
    assert "16 propagated rows" in text
    assert "missing outcomes are never recoded as zero" in text


def test_global_gallery_exposes_detector_accountability_visuals():
    text = (DOCS / "gallery.md").read_text(encoding="utf-8")
    assert "## Detector sensitivity and inference accountability" in text
    assert "robustness-denominator.svg" in text
    assert "model-input-audit.svg" in text
