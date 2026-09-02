# eyeprocesspy

[![CI](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml)
[![Documentation](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml)
[![Deep parity audit](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue)](https://www.python.org/)
[![R reference 0.11.1](https://img.shields.io/badge/R%20reference-0.11.1-276DC3)](docs/parity-and-validation.md)

**Reproducible Python infrastructure for eye-tracking, pupillometry, process data, psychometrics, and multimodal behavioral measurement.**

`eyeprocesspy` is the Python companion and deep-parity port of the R package **eyeprocess**, with the frozen R **0.11.1** release used as the scientific reference. It brings vendor import, data contracts, preprocessing, gaze/AOI analysis, pupil workflows, process measurement, IRT, validation, reproducibility, and reporting into one auditable Python package.

> **Release-candidate status:** the `release/0.1.0-deep-parity` branch is being hardened against a true 100% statement-and-branch coverage gate before release. No reduced coverage threshold is used.

## Why eyeprocesspy?

- **One coherent data model** for recordings, gaze samples, fixations, events, AOIs, pupil signals, responses, and provenance.
- **Scientific parity first:** 1,182 frozen APIs are resolved against eyeprocess 0.11.1, with documented differences where Python and R ecosystems cannot be identical.
- **Beyond descriptive eye-tracking:** process psychometrics, IRT, sequence/process models, calibration uncertainty, reliability, leakage-aware validation, and multimodal measurement are first-class workflows.
- **Reproducibility by construction:** provenance, benchmark studies, validation evidence, manifests, software-paper evidence, and deterministic release audits are built into the package.
- **Cross-platform release discipline:** the release matrix covers Ubuntu, macOS, and Windows on Python 3.11, 3.12, 3.13, and 3.14, plus a frozen-R oracle and clean-wheel validation.

## At a glance

| Release dimension | Current deep-parity state |
| --- | ---: |
| Frozen public APIs resolved | **1,182 / 1,182** |
| Frozen R reference | **eyeprocess 0.11.1** |
| Frozen articles linked | **88 / 88** |
| Python versions in CI | **3.11–3.14** |
| Operating systems in CI | **Ubuntu / macOS / Windows** |
| P4 numerical parity debt marked `not_started` | **0** |
| P6 plot parity debt marked `not_started` | **0** |

## Installation

For the current release candidate, install directly from the deep-parity branch:

```bash
pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
```

For local development:

```bash
git clone https://github.com/stefanosbalaskas/eyeprocesspy.git
cd eyeprocesspy
git checkout release/0.1.0-deep-parity
python -m pip install -e ".[dev,docs]"
```

The final PyPI command will be documented after the release evidence gate is fully green.

## 30-second reproducible workflow

The bundled benchmark is the fastest way to verify an installation and exercise the package without external data:

```python
import eyeprocesspy as ep

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

print(audit["valid"])
print(data)
```

For real exports, use the vendor-aware import surface and continue in the same canonical data model:

```python
import eyeprocesspy as ep

eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)
```

## What is included?

| Area | Representative capabilities |
| --- | --- |
| **Import & canonicalization** | Generic and vendor-aware readers, Gazepoint workflows, schema validation, coordinate systems, event/timebase handling |
| **Gaze & AOIs** | Fixations, saccades, dwell, transitions, probabilistic/compositional AOIs, scanpaths, recurrence, spatial/process features |
| **Pupil & physiology-facing process analysis** | Baseline correction, pupil features, functional pupil models, registration, missingness, uncertainty, sensitivity workflows |
| **Psychometrics & IRT** | Binary/polytomous IRT utilities, process-informed models, adaptive/scoring diagnostics, DIF, validation, multidimensional and advanced process models |
| **Measurement intelligence** | Reliability, calibration uncertainty, cross-device linking, process norms, fairness, item-bank optimization, leakage-aware validation |
| **Reproducibility & validation** | Benchmarks, recovery/SBC/stress evidence, provenance, reproducibility manifests, validation atlases, software-paper evidence |
| **Plots & reporting** | Scientific diagnostic plots, validation visualizations, evidence/reporting helpers, publication-oriented reproducibility outputs |

## Scientific boundary

`eyeprocesspy` provides **measurement and analysis infrastructure**. A computed metric is not automatically a validated psychological construct, diagnosis, or causal explanation. Reliability does not establish construct validity; prediction does not establish causation; and biometrics should be interpreted only within an appropriate study design, measurement model, and ethical framework.

This boundary is explicit throughout the package through validation guardrails, caveats, provenance, and evidence objects.

## Documentation

- **Package website:** https://stefanosbalaskas.github.io/eyeprocesspy/
- [Getting started](docs/getting-started.md)
- [Parity and validation](docs/parity-and-validation.md)
- [Release and reproducibility](docs/release-and-reproducibility.md)
- [API reference](docs/reference/)
- [Articles](docs/articles/)

## Deep-parity release discipline

The release branch is not considered complete merely because tests pass. The release evidence gate requires:

1. the complete pytest suite to pass;
2. **100% statement coverage**;
3. **100% branch coverage**;
4. the full Ubuntu/macOS/Windows × Python 3.11–3.14 matrix to pass;
5. Ruff to pass;
6. a clean wheel to build, install, and import;
7. the frozen R 0.11.1 oracle to pass; and
8. the documentation site to build strictly.

The deep-parity audit intentionally remains red until the coverage conditions are genuinely satisfied.

## Relationship to the R package

The Python implementation is designed from the frozen eyeprocess 0.11.1 source, API, tests, articles, examples, and expected scientific behavior rather than from function names alone. Where an R dependency or runtime behavior has no faithful Python equivalent, the parity ledger records the difference instead of silently substituting a different estimator.

## Contributing

Issues and pull requests that improve scientific correctness, parity, validation, documentation, interoperability, or reproducibility are welcome. For changes affecting numerical behavior, include a focused test and describe whether the behavior matches, intentionally differs from, or extends the frozen R reference.

## Citation

A formal citation for `eyeprocesspy` will be added with the first archival software release. Until then, cite the repository and the exact version/commit used in your analysis so that the computational environment is reproducible.
