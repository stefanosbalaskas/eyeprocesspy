from pathlib import Path

import eyeprocesspy as ep


def test_data_quality_plot_gallery_site_contract():
    root = Path(__file__).resolve().parents[1]
    mkdocs = (root / "mkdocs.yml").read_text(encoding="utf-8")
    guide = (root / "docs/guides/data-quality.md").read_text(encoding="utf-8")
    worked = (root / "docs/examples/data-quality-validation.md").read_text(encoding="utf-8")
    gallery_path = root / "docs/examples/data-quality-plot-gallery.md"
    gallery = gallery_path.read_text(encoding="utf-8")

    assert "examples/data-quality-plot-gallery.md" in mkdocs
    assert "data-quality-plot-gallery.md" in guide
    assert "data-quality-plot-gallery.md" in worked

    for asset in (
        "docs/assets/data-quality-accuracy-precision.svg",
        "docs/assets/data-quality-dashboard.svg",
        "docs/assets/data-quality-sampling.svg",
    ):
        assert (root / asset).is_file()

    for symbol in (
        "plot_gaze_accuracy",
        "plot_gaze_precision",
        "plot_bcea",
        "plot_sampling_intervals",
        "plot_gaze_quality_dashboard",
        "create_gaze_quality_report",
        "report_gaze_quality",
    ):
        assert symbol in gallery
        assert hasattr(ep, symbol)

    assert "../reference/data-quality.md" in gallery
    assert "../guides/data-quality-reporting.md" in gallery

    # Prevent escaped-newline artifacts from being reintroduced into key pages.
    assert "\\n" not in guide
    assert "\\n" not in worked
