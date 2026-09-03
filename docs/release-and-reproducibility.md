# Release and reproducibility

## Report both versions

For reproducibility, report both:

- the installed `eyeprocesspy` version; and
- `eyeprocesspy.__r_reference_version__`.

For release 0.1.0 the frozen R reference is 0.11.1.

## Release gate

The public release is built only after the following checks are green:

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest
uv run pytest --cov=eyeprocesspy --cov-branch --cov-report=term-missing --cov-fail-under=100
uv build
uv run python -m twine check dist/*
```

CI additionally verifies the frozen R oracle and runs the Python test matrix on Ubuntu, Windows and macOS with Python 3.11–3.14.

For `v0.1.0`, the controlling release evidence is **1,458 passing tests**, **23,085 / 23,085 statements**, and **9,680 / 9,680 branches** covered.

## Published artifacts

Release `0.1.0` is available through three coordinated publication surfaces:

- **GitHub Release:** `https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.1.0`
- **PyPI:** `https://pypi.org/project/eyeprocesspy/0.1.0/`
- **Zenodo archive:** `https://doi.org/10.5281/zenodo.22285167`

The release contains both a source distribution (`eyeprocesspy-0.1.0.tar.gz`) and a universal Python wheel (`eyeprocesspy-0.1.0-py3-none-any.whl`). The GitHub release is the source-control release of record, PyPI provides the installable distribution, and Zenodo provides the immutable research-software archive and DOI.

## PyPI

The repository uses GitHub Actions OpenID Connect / PyPI Trusted Publishing. No long-lived PyPI API token is stored in the repository.

Install the release with:

```bash
pip install eyeprocesspy
```

or pin the exact initial release:

```bash
pip install eyeprocesspy==0.1.0
```

## Zenodo

Zenodo archived `v0.1.0` and minted the release DOI **10.5281/zenodo.22285167**. The DOI is recorded in `CITATION.cff` and surfaced from the README. `.zenodo.json` remains the repository-side metadata source for future GitHub-to-Zenodo releases.

## Integrity

The package contains validation-manifest and freeze utilities for study-level evidence. Native RDS remains R-specific; Python persistence formats are labelled honestly and are not disguised as RDS.
