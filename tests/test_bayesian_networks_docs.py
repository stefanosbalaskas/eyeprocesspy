from pathlib import Path


def test_bayesian_network_documentation_surface():
    root = Path(__file__).resolve().parents[1]
    required = [
        "docs/guides/bayesian-networks/index.md",
        "docs/guides/bayesian-networks/data-contract.md",
        "docs/guides/bayesian-networks/structure-learning.md",
        "docs/guides/bayesian-networks/stability-validation.md",
        "docs/guides/bayesian-networks/sensitivity.md",
        "docs/guides/bayesian-networks/reporting.md",
        "docs/examples/bayesian-network-multimodal.md",
        "docs/examples/bayesian-network-sensitivity.md",
        "docs/reference/bayesian-networks.md",
        "docs/assets/bayesian-networks/bn-multimodal-dag.svg",
        "docs/assets/bayesian-networks/bn-posterior-update.svg",
        "docs/assets/bayesian-networks/bn-edge-stability.svg",
        "docs/assets/bayesian-networks/bn-detector-robustness.svg",
        "examples/bayesian_network_multimodal.py",
    ]
    for relative in required:
        path = root / relative
        assert path.exists(), relative
        assert path.stat().st_size > 100, relative

    overview = (
        root / "docs/guides/bayesian-networks/index.md"
    ).read_text(encoding="utf-8")
    assert "not, by itself, evidence of a causal effect" in overview
    assert "does not" in overview
    assert "silently discretize" in overview
