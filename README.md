<p align="center">
  <img src="https://raw.githubusercontent.com/stefanosbalaskas/gpbiometricspy/main/docs/assets/python-suite-logo.png" width="270" alt="Python Suite research packages logo">
</p>

<h1 align="center">eyeprocesspy</h1>

<p align="center">
  <strong>Reproducible Python infrastructure for eye-tracking, pupillometry, AOIs, process data, psychometrics, and multimodal behavioral measurement.</strong>
</p>

<p align="center">
  <a href="https://stefanosbalaskas.github.io/eyeprocesspy/">Documentation</a> ·
  <a href="https://pypi.org/project/eyeprocesspy/">PyPI</a> ·
  <a href="https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.1.0">Release v0.1.0</a> ·
  <a href="https://doi.org/10.5281/zenodo.22285167">Zenodo DOI</a> ·
  <a href="https://stefanosbalaskas.github.io/eyeprocesspy/articles/">88 workflows</a>
</p>

<p align="center">
  <a href="https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml"><img alt="Documentation" src="https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml"><img alt="Deep parity" src="https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/eyeprocesspy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/eyeprocesspy.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.22285167"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.22285167.svg"></a>
  <img alt="Python 3.11–3.14" src="https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue">
  <img alt="Frozen API 1182/1182" src="https://img.shields.io/badge/frozen%20API-1182%20%2F%201182-success">
  <img alt="Coverage 100%" src="https://img.shields.io/badge/statements%20%2B%20branches-100%25-success">
</p>

`eyeprocesspy` is the Python companion and deep-parity port of the R package **eyeprocess**, with frozen **eyeprocess 0.11.1** as the scientific reference. It provides one governed analytical surface for vendor import, canonical data contracts, preprocessing, gaze and AOI analysis, pupillometry, process measurement, psychometrics/IRT, multimodal workflows, validation, provenance, plotting, and reproducible reporting.

> **Published release — 0.1.0.** The controlling release gate passed with **1,458 tests**, **23,085 / 23,085 statements**, and **9,680 / 9,680 branches** covered. The frozen public API is **1,182 / 1,182**, all **88 / 88** workflow articles are linked, the Ubuntu/macOS/Windows × Python 3.11–3.14 matrix is green, and the release is archived at **DOI 10.5281/zenodo.22285167**.

## Install

```bash
pip install eyeprocesspy
```

Pin the initial scientific release when exact reproducibility matters:

```bash
pip install eyeprocesspy==0.1.0
```

For source installation from the immutable release tag:

```bash
pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@v0.1.0"
```

Windows users can also use the hardened installer described in the [manual-install guide](https://stefanosbalaskas.github.io/eyeprocesspy/manual-install/). It has been exercised on a real Windows/Python 3.11.9 installation with the recommended extras and a clean `pip check`.

## Release evidence

| Dimension | Verified state |
| --- | ---: |
| Frozen R public APIs resolved | **1,182 / 1,182** |
| Frozen R reference | **0.11.1** |
| Workflow articles linked | **88 / 88** |
| Deep-parity tests | **1,458 passed** |
| Statement coverage | **23,085 / 23,085 (100%)** |
| Branch coverage | **9,680 / 9,680 (100%)** |
| Numerical `not_started` debt | **0** |
| Plot `not_started` debt | **0** |
| CI matrix | **Ubuntu / macOS / Windows × Python 3.11–3.14** |
| PyPI | **eyeprocesspy 0.1.0** |
| Zenodo | **10.5281/zenodo.22285167** |

See [`RELEASE_VALIDATION.md`](RELEASE_VALIDATION.md), [`TEST_SUMMARY.md`](TEST_SUMMARY.md), and the [parity & validation guide](https://stefanosbalaskas.github.io/eyeprocesspy/parity-and-validation/) for the underlying evidence.

## What it covers

| Research layer | Representative capabilities |
| --- | --- |
| **Import & canonicalization** | Vendor/generic readers, Gazepoint workflows, schema validation, coordinates, events, timebase, file pairing |
| **Gaze & AOIs** | Fixations, saccades, dwell, scanpaths, transitions, entropy, recurrence, hard/probabilistic/compositional AOIs |
| **Pupil & multimodal** | Baselines, pupil features, missingness, functional pupil, synchronized streams, staged multimodal models |
| **Psychometrics & IRT** | Information, fit, DIF/DTF, process-informed and dynamic IRT, reliability, linking, norms, diagnostics |
| **Measurement intelligence** | Calibration uncertainty, reliability, process guardrails, fairness, item-bank optimization |
| **Validation** | Recovery, SBC-style evidence, stress tests, negative controls, grouped/leakage-aware validation, evidence atlases |
| **Reproducibility** | Benchmarks, manifests, provenance, frozen-R oracle, release audits, software-paper evidence |
| **Plots & reporting** | Publication-oriented gaze, AOI, pupil, quality, IRT, process and validation graphics |

## 30-second reproducible check

```python
import eyeprocesspy as ep

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

print(audit["valid"])
print(data)
```

For a real export:

```python
import eyeprocesspy as ep

eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)
```

## See the package in action

| Gaze trace | Probabilistic AOI |
| --- | --- |
| ![Gaze trace](docs/assets/gallery/gaze-trace.svg) | ![Probabilistic AOI](docs/assets/gallery/probabilistic-aoi.svg) |

| Pupil time series | Process reliability |
| --- | --- |
| ![Pupil time series](docs/assets/gallery/pupil-timeseries.svg) | ![Process reliability](docs/assets/gallery/process-reliability.svg) |

[Open the complete visual gallery →](https://stefanosbalaskas.github.io/eyeprocesspy/gallery/)

## Choose your path

- **New to the package:** [Getting started](https://stefanosbalaskas.github.io/eyeprocesspy/getting-started/) → [Worked examples](https://stefanosbalaskas.github.io/eyeprocesspy/examples/) → [Cookbook](https://stefanosbalaskas.github.io/eyeprocesspy/cookbook/)
- **Eye-tracking / AOI work:** [End-to-end eye-tracking](https://stefanosbalaskas.github.io/eyeprocesspy/guides/end-to-end-eye-tracking/) · [Gazepoint import & QC](https://stefanosbalaskas.github.io/eyeprocesspy/guides/gazepoint-import-qc/)
- **Pupillometry / multimodal:** [Pupillometry guide](https://stefanosbalaskas.github.io/eyeprocesspy/guides/pupillometry/) · [Quality & uncertainty](https://stefanosbalaskas.github.io/eyeprocesspy/guides/process-quality-uncertainty/)
- **Psychometrics / IRT:** [Psychometrics & IRT](https://stefanosbalaskas.github.io/eyeprocesspy/guides/psychometrics-irt/) · [IRT example](https://stefanosbalaskas.github.io/eyeprocesspy/examples/irt-diagnostics/)
- **Full workflows:** [Featured workflows](https://stefanosbalaskas.github.io/eyeprocesspy/articles/featured-workflows/) · [88-article library](https://stefanosbalaskas.github.io/eyeprocesspy/articles/)
- **Lookup:** [API reference](https://stefanosbalaskas.github.io/eyeprocesspy/reference/api/) · [Plotting reference](https://stefanosbalaskas.github.io/eyeprocesspy/reference/plotting/) · [FAQ](https://stefanosbalaskas.github.io/eyeprocesspy/faq/)

## Scientific boundary

`eyeprocesspy` provides **measurement and analysis infrastructure**. A metric is not automatically a validated psychological construct, diagnosis, or causal explanation. Reliability does not establish construct validity; prediction does not establish causation; probabilistic AOI membership represents modeled coordinate uncertainty rather than probability of attention; and gaze, pupil, biometric, and process measures require an appropriate design, measurement model, and ethical interpretation.

## Citation

If you use `eyeprocesspy 0.1.0`, cite the archived software release:

> Balaskas, S. (2026). *eyeprocesspy: Vendor-neutral Python infrastructure for eye-tracking and multimodal process data* (Version 0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22285167

For reproducibility, also report `eyeprocesspy.__version__` and the frozen R reference exposed as `eyeprocesspy.__r_reference_version__`.

## Relationship to R eyeprocess

The Python package is developed against the frozen `eyeprocess 0.11.1` reference. API, articles, data, plots, numerical evidence, backends, and unavoidable language-specific divergences are tracked explicitly. Python-native extensions are kept distinct from reference parity so they do not masquerade as R-equivalent behavior.

## License

MIT. See [`LICENSE`](LICENSE).
