# Release and reproducibility

## Report both versions

For reproducibility, report both:

- the installed `eyeprocesspy` version; and
- `eyeprocesspy.__r_reference_version__`.

For release 0.2.2 the frozen R reference remains **0.11.1**.

## Release gate

The public release is built only after the following controls are green:

~~~bash
uv sync --frozen --extra dev --extra narrow
uv run --frozen ruff check .
uv run --frozen mypy src/eyeprocesspy
uv run --frozen pytest --cov=eyeprocesspy --cov-branch --cov-report=term-missing --cov-fail-under=100
uv run --frozen python tools/deep_parity_audit.py --release-gate
uv run --frozen python tools/api_documentation_audit.py
uv run --frozen mkdocs build --strict
uv build --no-sources
uv run --frozen python -m twine check --strict dist/*
~~~

CI additionally verifies the frozen R oracle, audits published dependencies, clean-installs both wheel and sdist, and runs Python 3.11–3.14 across Ubuntu, Windows, and macOS.

For the measurement-validity, dynamic-gaze, and completed Bayesian-network robustness tranche carried into 0.2.2, the controlling qualification is **1,892 passing tests**, **36,556 / 36,556 statements**, and **12,630 / 12,630 branches** covered.

## Published artifacts

Release `0.2.2` is coordinated across:

- **GitHub Release:** `https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.2.2`
- **PyPI:** `https://pypi.org/project/eyeprocesspy/0.2.2/`
- **Zenodo metadata:** repository metadata is updated to 0.2.2; a release DOI is reported only after Zenodo mints it.

The release contains both a source distribution (`eyeprocesspy-0.2.2.tar.gz`) and a universal Python wheel (`eyeprocesspy-0.2.2-py3-none-any.whl`). GitHub is the source-control release of record and PyPI provides the installable distribution.

## PyPI

The repository uses GitHub Actions OpenID Connect / PyPI Trusted Publishing. No long-lived PyPI API token is stored in the repository.

Install the latest release with:

~~~bash
pip install eyeprocesspy
~~~

or pin this release:

~~~bash
pip install eyeprocesspy==0.2.2
~~~

## Integrity

Release artifacts include SHA-256 evidence and GitHub build-provenance attestation. The package also retains validation-manifest and freeze utilities for study-level evidence. Native RDS remains R-specific; Python persistence formats are labelled honestly and are not disguised as RDS.

## Bayesian-network boundary

The 0.2.2 Bayesian-network layer consumes already validated features. It does not silently alter event detection, AOIs, pupil/physiology preprocessing, missingness, or discretization. Learned directed edges are probabilistic graph orientations under the specified data and constraints; they are not automatically causal effects.
