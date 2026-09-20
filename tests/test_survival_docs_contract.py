from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), f"required survival documentation asset is missing: {relative}"
    return path.read_text(encoding="utf-8")


def test_survival_site_navigation_contract() -> None:
    nav = _read("mkdocs.yml")
    required = (
        "methods/gaze-survival/index.md",
        "methods/gaze-survival/decision-guide.md",
        "methods/gaze-survival/reporting.md",
        "methods/gaze-survival/reporting-template.md",
        "methods/gaze-survival/troubleshooting.md",
        "methods/gaze-survival/reproducibility-checklist.md",
        "examples/gaze-survival-analysis.md",
        "examples/gaze-verification-survival.md",
        "reference/gaze-survival.md",
    )
    for item in required:
        assert item in nav, f"survival navigation entry disappeared: {item}"


def test_survival_discovery_contract() -> None:
    examples = _read("docs/examples/index.md")
    method = _read("docs/methods/gaze-survival/index.md")
    readme = _read("README.md")
    changelog = _read("CHANGELOG.md")

    assert "worked_gaze_survival_analysis.py" in examples
    assert "worked_gaze_verification_survival.py" in examples
    assert "Evidence-verification survival" in examples
    assert "Troubleshooting clinic" in method
    assert "Reproducibility checklist" in method
    assert "gaze-verification-survival" in readme
    assert "reproducibility-checklist" in readme
    assert "censored gaze-latency" in changelog.lower()


def test_survival_required_assets_exist() -> None:
    required = (
        "src/eyeprocesspy/survival.py",
        "tests/test_survival.py",
        "examples/worked_gaze_survival_analysis.py",
        "examples/worked_gaze_verification_survival.py",
        "docs/guides/gaze-survival-analysis.md",
        "docs/methods/gaze-survival/decision-guide.md",
        "docs/methods/gaze-survival/reporting.md",
        "docs/methods/gaze-survival/reporting-template.md",
        "docs/methods/gaze-survival/troubleshooting.md",
        "docs/methods/gaze-survival/reproducibility-checklist.md",
        "docs/examples/gaze-survival-analysis.md",
        "docs/examples/gaze-verification-survival.md",
        "docs/reference/gaze-survival.md",
        "docs/assets/gaze-survival/km-disclosure.svg",
        "docs/assets/gaze-survival/km-evidence-verification.svg",
        "docs/assets/gaze-survival/event-incidence-verification.svg",
        "docs/assets/gaze-survival/censoring-audit-verification.svg",
    )
    for item in required:
        assert (ROOT / item).is_file(), f"required survival asset is missing: {item}"


def test_recent_method_visual_discovery_contract() -> None:
    gallery = _read("docs/gallery.md")
    home = _read("docs/index.md")

    survival_assets = (
        "gaze-survival/km-disclosure.svg",
        "gaze-survival/km-evidence-verification.svg",
        "gaze-survival/event-incidence-verification.svg",
        "gaze-survival/censoring-audit-verification.svg",
    )
    for asset in survival_assets:
        assert asset in gallery, f"survival gallery visual disappeared: {asset}"

    recent_method_assets = (
        "aoi-uncertainty/robustness-surface.svg",
        "detector-multiverse/condition-coefficient-stability.svg",
        "data-quality-accuracy-precision.svg",
    )
    for asset in recent_method_assets:
        assert asset in gallery or asset in home, (
            f"recent-method visual disappeared from discovery surfaces: {asset}"
        )
