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

## Artifacts

A release produces both:

- a source distribution (`.tar.gz`); and
- a universal Python wheel (`.whl`).

The GitHub release is the source-control release of record. PyPI publishes the installable package. Zenodo archival is used for the immutable research-software record and DOI after the GitHub release has been archived.

## PyPI

The repository uses GitHub Actions OpenID Connect / PyPI Trusted Publishing. No long-lived PyPI API token is stored in the repository.

## Zenodo

`.zenodo.json` and `CITATION.cff` contain release metadata. The project does not publish a placeholder DOI. The DOI is added to the README and citation metadata only after Zenodo has created a real record.

## Integrity

The package contains validation-manifest and freeze utilities for study-level evidence. Native RDS remains R-specific; Python persistence formats are labelled honestly and are not disguised as RDS.
