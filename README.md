<p align="center">
  <img src="https://raw.githubusercontent.com/stefanosbalaskas/eyeprocesspy/main/docs/assets/python-suite-logo.png" width="270" alt="Python Suite research packages logo">
</p>

<h1 align="center">eyeprocesspy</h1>

<p align="center">
  <strong>Reproducible Python infrastructure for eye-tracking, pupillometry, AOIs, process data, psychometrics, and multimodal behavioral measurement.</strong>
</p>

<p align="center">
  <a href="https://stefanosbalaskas.github.io/eyeprocesspy/">Documentation</a> ·
  <a href="https://stefanosbalaskas.github.io/eyeprocesspy/getting-started/">Getting started</a> ·
  <a href="https://stefanosbalaskas.github.io/eyeprocesspy/gallery/">Visual gallery</a> ·
  <a href="https://github.com/stefanosbalaskas/eyeprocesspy/tree/release/0.1.0-deep-parity">Release-candidate branch</a>
</p>

[![CI](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml)
[![Documentation](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml/badge.svg?branch=release%2F0.1.0-deep-parity)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml)
[![Deep parity](https://img.shields.io/badge/frozen%20API-1182%20%2F%201182-success)](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/IMPLEMENTATION_STATUS.md)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue)](https://www.python.org/)
[![R reference](https://img.shields.io/badge/R%20reference-0.11.1-276DC3)](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/docs/parity-and-validation.md)

`eyeprocesspy` is the Python companion and deep-parity port of the R package **eyeprocess**, with frozen **eyeprocess 0.11.1** as the scientific reference. It brings vendor import, canonical data contracts, preprocessing, gaze/AOI analysis, pupil workflows, process measurement, IRT, validation, scientific plots, provenance, and reporting into one auditable package.

> **0.1.0 release candidate.** API parity is complete, but publication remains intentionally gated on the final deep-parity evidence, coverage, CI, artifact, and documentation checks. A green-looking badge is not substituted for scientific validation.

## Release-candidate snapshot

| Dimension | Current state |
| --- | ---: |
| Frozen R public APIs resolved | **1,182 / 1,182** |
| Frozen R reference | **0.11.1** |
| Frozen workflow articles linked | **88 / 88** |
| P4 numerical `not_started` debt | **0** |
| P6 plot `not_started` debt | **0** |
| CI target | **Ubuntu / macOS / Windows × Python 3.11–3.14** |
| Public release | **Not yet tagged or published** |

The active release-hardening work lives on [`release/0.1.0-deep-parity`](https://github.com/stefanosbalaskas/eyeprocesspy/tree/release/0.1.0-deep-parity). The release gate requires the full pytest suite, **100% statement coverage**, **100% branch coverage**, Ruff, clean wheel/sdist install checks, frozen-R oracle validation, and a strict documentation build.

## Windows installation: now verified on a real machine

The hardened manual installer has been exercised successfully with **Python 3.11.9** on Windows. Core import verification passed, the recommended extras (`plots`, `psychometrics`, `ml`, `arrow`, `physio`, `docs`) installed successfully, and `python -m pip check` reported **No broken requirements found**.

From the extracted manual-install bundle:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_eyeprocesspy.ps1 -WithAllRecommended
```

The installer does **not** require the Windows `py` launcher. It can detect supported Python installations directly and also accepts an explicit interpreter path.

[Read the complete Windows/manual-install guide →](https://stefanosbalaskas.github.io/eyeprocesspy/manual-install/)

## Why eyeprocesspy?

- **One coherent data model** for recordings, gaze samples, eye/pupil samples, fixations and episodes, events, intervals, AOIs, responses, features, quality, and provenance.
- **Scientific parity first:** frozen APIs are resolved explicitly, with governed records for unavoidable cross-language differences rather than fabricated equality.
- **Process data as first-class evidence:** scanpaths, transitions, temporal structure, uncertainty, reliability, and psychometrics are part of the same analytical surface.
- **Measurement guardrails:** calibration uncertainty, quality, reliability, DIF/fairness, and process metrics retain explicit interpretation boundaries.
- **Reproducibility by construction:** deterministic benchmarks, provenance, validation evidence, software-paper evidence, and release audits are built in.
- **Broad plotting surface:** gaze, AOI, pupil, quality, IRT, process-measurement, validation, and model-diagnostic graphics are supported through Matplotlib-oriented workflows.

## Capability map

| Area | Representative capabilities |
| --- | --- |
| **Import & canonicalization** | Generic/vendor-aware readers, Gazepoint workflows, schema validation, coordinates, events/timebase, file pairing |
| **Preprocessing & gaze** | Fixations, saccades, dwell, scanpaths, transitions, entropy, recurrence, spatial/process features |
| **AOI uncertainty** | Hard, probabilistic and compositional AOIs; calibration-error propagation and sensitivity |
| **Pupil & multimodal analysis** | Baselines, pupil features, functional pupil, missingness, synchronized streams, staged multimodal models |
| **Psychometrics & IRT** | Foundations, scoring, fit, Q3, DIF/DTF, process-informed/dynamic/advanced IRT, diagnostics |
| **Measurement intelligence** | Reliability, calibration uncertainty, process guardrails, linking, norms, fairness, item-bank optimization |
| **Validation** | Recovery, SBC-style evidence, stress tests, negative controls, grouped/leakage-aware validation, evidence atlases |
| **Reproducibility** | Bundled benchmarks, provenance, manifests, frozen-R oracle, software-paper and release evidence |
| **Plots & reporting** | Publication-oriented plots, validation visualizations, scientific evidence/reporting helpers |

## See it in action

The documentation includes package-generated examples for gaze traces, scanpaths, pupil time series, AOI dwell, probabilistic AOIs, calibration error, process reliability, sampling irregularity, IRT information, item fit, DIF, and transition structure.

**[Open the visual gallery →](https://stefanosbalaskas.github.io/eyeprocesspy/gallery/)**

**[Open worked examples →](https://stefanosbalaskas.github.io/eyeprocesspy/examples/)**

**[Browse the 88-article workflow library →](https://stefanosbalaskas.github.io/eyeprocesspy/articles/)**

## Scientific boundary

`eyeprocesspy` provides **measurement and analysis infrastructure**. A metric is not automatically a validated psychological construct, diagnosis, or causal explanation. Reliability does not establish construct validity; prediction does not establish causation; probabilistic AOI membership reflects modeled coordinate uncertainty rather than probability of attention; and gaze, pupil, and biometric measures require an appropriate design, measurement model, and ethical interpretation.

## Relationship to R eyeprocess

The Python package is developed against the frozen `eyeprocess 0.11.1` reference. API, articles, data, plots, backends, numerical evidence, and unavoidable language-specific divergences are tracked explicitly. Python-native extensions are separated from reference parity so they do not masquerade as R-equivalent behavior.

## Release discipline

`eyeprocesspy 0.1.0` remains deliberately unreleased while the deep-parity gate is open. GitHub Release, PyPI publication, and archival actions should occur only after the controlling release head satisfies the declared evidence requirements.

## License

See [`LICENSE`](LICENSE).
