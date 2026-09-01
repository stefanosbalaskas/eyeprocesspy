# eyeprocesspy

[![CI](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/eyeprocesspy.svg)](https://pypi.org/project/eyeprocesspy/)
[![Python](https://img.shields.io/pypi/pyversions/eyeprocesspy.svg)](https://pypi.org/project/eyeprocesspy/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://stefanosbalaskas.github.io/eyeprocesspy/)

**Vendor-neutral Python infrastructure for eye-tracking, pupillometry, biometrics, psychometrics, and multimodal process data.**

`eyeprocesspy` is the Python port of the frozen R package **`eyeprocess` 0.11.1**. It provides a common data model, import/harmonization tools, quality-control and governance workflows, gaze and pupil processing, multimodal measurement tools, psychometric and IRT infrastructure, validation/evidence tooling, plotting, reproducibility utilities, and controlled interoperability with external scientific backends.

## Frozen reference and parity

| Contract | Frozen reference |
| --- | ---: |
| R package | `eyeprocess` 0.11.1 |
| Frozen R commit | `d867555eecae46f262843501c07074cebe1f7aa9` |
| Public R exports | **1,182** |
| Python API exports implemented | **1,182 / 1,182** |
| Public API remaining | **0** |
| R articles/vignettes | 88 |
| R testthat files | 113 |
| R Stan programs | 13 |

The API surface reached 1,182/1,182 at commit `d1d38d6db8cb49ca6ec47b610b528422946a55be`, with the hosted Linux/macOS/Windows Python 3.11–3.14 matrix and the frozen-R oracle smoke test green. The release branch additionally audits numerical/oracle parity, documentation coverage, plot/data contracts, and package-wide test coverage before the `v0.1.0` release is published.

See [`parity/PARITY_MATRIX.csv`](parity/PARITY_MATRIX.csv) for the function-level ledger and [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) for the release-readiness view.

## Installation

### PyPI

```bash
python -m pip install eyeprocesspy
```

or with `uv`:

```bash
uv add eyeprocesspy
```

### Development checkout

```bash
git clone https://github.com/stefanosbalaskas/eyeprocesspy.git
cd eyeprocesspy
uv sync --extra dev
uv run pytest
```

Optional dependency groups are available for Arrow storage, Stan, Bayesian backends, physiology, streaming, gaze tooling, psychometrics, plotting, machine learning, and documentation.

## Minimal workflow

```python
import eyeprocesspy as ep

# Inspect the canonical schema.
schema = ep.eye_schema()

# Read a supported export through the vendor-neutral adapter layer.
data = ep.read_eye_export("path/to/export")

# Audit the imported dataset before analysis.
report = ep.analysis_readiness(data)
```

For Gazepoint exports, IRT workflows, pupil preprocessing, multimodal measurement, validation evidence, reproducibility, storage/interoperability, and plotting examples, see the [documentation site](https://stefanosbalaskas.github.io/eyeprocesspy/) and the source articles under [`docs/articles/`](docs/articles/).

## Scientific commitments

`eyeprocesspy` follows the frozen R package's core rules:

- harmonize semantics rather than only renaming columns;
- retain native timing/source information and make transformations explicit;
- preserve provenance and validation evidence;
- never silently interpolate, resample, exclude, or infer scientific meaning;
- distinguish software-validation evidence from construct validity;
- gate unavailable exact backends instead of substituting a different estimator;
- never disguise another serialization format as native RDS;
- keep optional scientific backends lazy so the base package remains lightweight.

Some R-specific engines, random-number streams, object serialization/hashing, platform timings, and external backend outputs cannot be byte-identical in Python. Those cases are marked explicitly as `python_reference_differs` in the parity ledger and require a documented blocker plus a Python conformance test rather than a fabricated equality claim.

## Verification

The release gate runs:

```bash
uv run ruff check src tests
uv run pytest
uv run pytest --cov=eyeprocesspy --cov-branch --cov-report=term-missing --cov-fail-under=100
uv build
python -m twine check dist/*
```

CI also installs and verifies the frozen R 0.11.1 oracle and tests Python 3.11–3.14 on Ubuntu, Windows, and macOS.

## Documentation

- Documentation: https://stefanosbalaskas.github.io/eyeprocesspy/
- Source: https://github.com/stefanosbalaskas/eyeprocesspy
- Issues: https://github.com/stefanosbalaskas/eyeprocesspy/issues
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The repository also ships Zenodo metadata in [`.zenodo.json`](.zenodo.json). A Zenodo DOI is added only after an actual archived release exists; no placeholder DOI is used.

## License

MIT. See [`LICENSE`](LICENSE).
