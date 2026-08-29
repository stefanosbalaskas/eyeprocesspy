from pathlib import Path

ARTICLES = [
    "m2-ppc-negative-controls-0-10.md",
    "m2-process-information-ablation-0-10.md",
    "m2-recovery-identifiability-0-10.md",
    "m2-three-way-reference-model-0-10.md",
    "m3-device-transport-sensor-value-0-10.md",
    "m3-four-channel-reference-model-0-10.md",
    "m3-functional-pupil-bridge-0-10.md",
    "m3-process-information-ablation-0-10.md",
    "m3-pupil-confounds-measurement-0-10.md",
    "m3-recovery-missingness-identifiability-0-10.md",
    "m4-identifiability-state-uncertainty-0-11.md",
    "m4-negative-controls-state-count-0-11.md",
    "m4-process-information-ablation-0-11.md",
    "m4-recovery-validation-0-11.md",
    "m4-reference-model-fitting-0-11.md",
    "m4-trait-conditioned-process-states-0-11.md",
    "manual-multimodal-backends-0-10.md",
    "process-information-and-ablation-0-10.md",
    "pupil-measurement-boundaries-0-10.md",
    "unified-multimodal-process-irt-0-10.md",
]


def test_staged_multimodal_articles_exist_and_are_python_native():
    root = Path(__file__).resolve().parents[1] / "docs" / "articles"
    assert len(ARTICLES) == 20
    for name in ARTICLES:
        path = root / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert len(text) > 250, name
        assert "eyeprocesspy" in text, name
        assert "```{r" not in text, name
        assert "library(eyeprocess)" not in text, name
