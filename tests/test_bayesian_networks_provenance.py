from importlib import metadata

import eyeprocesspy.bayesian_networks.learning as learning_module


def test_version_returns_none_when_distribution_metadata_is_absent(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(learning_module.metadata, "version", missing)
    assert learning_module._version("not-installed") is None
