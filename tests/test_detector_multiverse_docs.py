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
        "examples/detector-multiverse-disclosure.md",
        "reference/detector-multiverse.md",
    ]
    for page in pages:
        assert (DOCS / page).exists(), page


def test_detector_multiverse_reference_only_names_public_api():
    text = (DOCS / "reference/detector-multiverse.md").read_text()
    names = re.findall(r"::: eyeprocesspy\.([A-Za-z0-9_]+)", text)
    assert len(names) >= 20
    assert all(callable(getattr(ep, name, None)) for name in names)


def test_new_documentation_internal_links_resolve():
    pages = list((DOCS / "guides").glob("*detector*.md")) + [
        DOCS / "examples/detector-multiverse-disclosure.md"
    ]
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)")
    for page in pages:
        text = page.read_text()
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
