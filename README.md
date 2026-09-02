# eyeprocesspy

[![CI](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml)
[![Documentation](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml)
[![Deep parity audit](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml/badge.svg)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue)](https://www.python.org/)
[![R reference 0.11.1](https://img.shields.io/badge/R%20reference-0.11.1-276DC3)](docs/parity-and-validation.md)

**Reproducible Python infrastructure for eye-tracking, pupillometry, process data, psychometrics, and multimodal behavioral measurement.**

`eyeprocesspy` is the Python companion and deep-parity port of the R package **eyeprocess**, with frozen **eyeprocess 0.11.1** as the scientific reference. It joins vendor import, canonical data contracts, preprocessing, gaze/AOI analysis, pupil workflows, process measurement, IRT, validation, reproducibility, scientific plots, and reporting in one auditable package.

> **Release-candidate status:** `release/0.1.0-deep-parity` is being hardened against a genuine 100% statement-and-branch coverage gate. The threshold is not reduced to manufacture a green release.

## Install

### Manual Windows bundle — recommended for the current release candidate

Extract the manual-install ZIP and run inside that directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_eyeprocesspy.ps1
```

To install the wheel together with the recommended plotting, psychometrics, machine-learning and Arrow backends:

```powershell
.\install_eyeprocesspy.ps1 -WithAllRecommended
```

You can also select a supported interpreter and only the extras you need:

```powershell
.\install_eyeprocesspy.ps1 -PythonVersion 3.12 -WithPlots -WithPsychometrics
```

Successful verification reports `eyeprocesspy 0.1.0` and frozen R reference `0.11.1`. The installer source is versioned in [`scripts/install_eyeprocesspy.ps1`](scripts/install_eyeprocesspy.ps1), with an additional benchmark smoke check in [`scripts/verify_manual_install.py`](scripts/verify_manual_install.py).

Or install the canonical wheel directly:

```powershell
python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
```

> **Windows download note:** if a browser renames the wheel to `eyeprocesspy-0.1.0-py3-none-any (1).whl`, rename it back to the canonical filename before calling `pip`. The inserted ` (1)` makes the filename fail wheel-tag parsing even though the wheel itself can be valid.

The CI workflow publishes a fresh `eyeprocesspy-manual-install-<commit>` artifact containing the wheel and source distribution **after** a clean install/import check.

### Directly from the release branch

```bash
pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
```

For plotting examples, install Matplotlib through the plotting extra or your environment's equivalent.

## Why eyeprocesspy?

- **One coherent data model** for recordings, gaze samples, eye/pupil samples, fixations and other episodes, events, intervals, AOIs, responses, features, quality and provenance.
- **Scientific parity first:** **1,182 / 1,182** frozen APIs are resolved against eyeprocess 0.11.1, with explicit records for unavoidable cross-language differences.
- **Process data, not only summaries:** scanpaths, transitions, temporal structure, uncertainty, reliability and psychometrics are first-class analytical objects.
- **Measurement guardrails:** calibration uncertainty, quality, reliability, DIF/fairness and process metrics carry explicit interpretation boundaries.
- **Reproducibility by construction:** deterministic benchmarks, provenance, validation evidence, software-paper evidence and release audits are built in.
- **Broad scientific plotting surface:** gaze, AOI, pupil, quality, IRT, process-measurement, validation and model-diagnostic graphics use Matplotlib and retain underlying plot data where applicable.

## Visual tour

| Gaze trace | Scanpath |
| --- | --- |
| ![Gaze trace](docs/assets/gallery/gaze-trace.svg) | ![Scanpath](docs/assets/gallery/scanpath.svg) |

| Pupil time series | Probabilistic AOI membership |
| --- | --- |
| ![Pupil time series](docs/assets/gallery/pupil-timeseries.svg) | ![Probabilistic AOI](docs/assets/gallery/probabilistic-aoi.svg) |

| Process reliability | IRT information |
| --- | --- |
| ![Process reliability](docs/assets/gallery/process-reliability.svg) | ![IRT information](docs/assets/gallery/irt-information.svg) |

**[Open the complete visual gallery →](docs/gallery.md)** — 15 package-generated previews paired with deterministic scripts that render through the public plotting API.

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

## Capability map

| Area | Representative capabilities |
| --- | --- |
| **Import & canonicalization** | Generic/vendor-aware readers, Gazepoint workflows, schema validation, coordinates, events/timebase, file pairing |
| **Preprocessing & gaze** | Fixations, saccades, dwell, scanpaths, transitions, entropy, recurrence, spatial/process features |
| **AOI uncertainty** | Hard, probabilistic and compositional AOI workflows; calibration-error propagation and sensitivity |
| **Pupil & multimodal process analysis** | Baselines, pupil features, functional pupil, missingness, synchronized streams, staged multimodal models |
| **Psychometrics & IRT** | Foundations, scoring, fit, Q3, DIF/DTF, process-informed/dynamic/advanced IRT, Bayesian/3PL diagnostics |
| **Measurement intelligence** | Reliability, calibration uncertainty, process registry/guardrails, linking, norms, fairness, item-bank optimization |
| **Validation** | Recovery, SBC-style evidence, stress tests, negative controls, grouped/leakage-aware validation, evidence atlases |
| **Reproducibility** | Bundled benchmarks, provenance, manifests, software-paper evidence, frozen-R oracle and release audits |
| **Plots & reporting** | Publication-oriented Matplotlib plots, validation visualizations, scientific evidence/reporting helpers |

## Worked workflows

Four focused examples have been validated against the CI-built `0.1.0` wheel and can be run without private data:

```bash
python examples/complete_workflow.py
python examples/calibration_probabilistic_aoi.py
python examples/process_reliability.py
python examples/irt_diagnostics.py
```

| Workflow | What it demonstrates |
| --- | --- |
| [Core gaze/AOI/provenance](docs/examples/core-workflow.md) | canonical `EyeDataset`, validation, scanpaths, transitions, entropy, plots and provenance |
| [Calibration uncertainty & probabilistic AOIs](docs/examples/calibration-probabilistic-aoi.md) | empirical calibration error, uncertainty ellipse and boundary-sensitive AOI assignment |
| [Process reliability](docs/examples/process-reliability.md) | ICC, Bland–Altman agreement and temporal stability |
| [IRT diagnostics](docs/examples/irt-diagnostics.md) | information/SEM, item fit and DIF plotting |

For shorter copy/paste tasks, use the **[Cookbook](docs/cookbook.md)**. For broader walkthroughs, use the **[Python-native guides](docs/guides/)** and **[88-article workflow library](docs/articles/)**.

## Gallery generators

```bash
python examples/core_gallery.py
python examples/advanced_gallery.py
```

These generate the documentation gallery and demonstrate canonical datasets, gaze/AOI/pupil plots, process reliability, calibration uncertainty, probabilistic AOIs, sampling irregularity and IRT diagnostics.

## Deep-parity state

| Dimension | State |
| --- | ---: |
| Frozen public APIs resolved | **1,182 / 1,182** |
| Frozen R reference | **0.11.1** |
| Frozen articles linked | **88 / 88** |
| P4 numerical `not_started` debt | **0** |
| P6 plot `not_started` debt | **0** |
| CI matrix | **Ubuntu / macOS / Windows × Python 3.11–3.14** |

The release evidence gate requires the full pytest suite, **100% statement coverage**, **100% branch coverage**, Ruff, a clean wheel install/import, the frozen-R oracle, and a strict documentation build.

## Documentation

- **Website:** https://stefanosbalaskas.github.io/eyeprocesspy/
- [Getting started](docs/getting-started.md)
- [Manual installation](docs/manual-install.md)
- [Runnable examples](docs/examples/index.md)
- [Practical cookbook](docs/cookbook.md)
- [Visual gallery](docs/gallery.md)
- [Python-native guides](docs/guides/)
- [88-article workflow library](docs/articles/)
- [Plotting and API reference](docs/reference/)
- [FAQ](docs/faq.md)
- [Parity and validation](docs/parity-and-validation.md)
- [Release and reproducibility](docs/release-and-reproducibility.md)

## Scientific boundary

`eyeprocesspy` provides **measurement and analysis infrastructure**. A metric is not automatically a validated psychological construct, diagnosis, or causal explanation. Reliability does not establish construct validity; prediction does not establish causation; probabilistic AOI membership reflects modeled coordinate uncertainty rather than probability of attention; and gaze/pupil/biometric measures require an appropriate design, measurement model and ethical interpretation.

## Relationship to the R package

The Python implementation is designed from the frozen eyeprocess 0.11.1 source, public API, tests, articles, examples and expected scientific behavior rather than from function names alone. Where an R dependency/runtime has no faithful Python equivalent, the parity ledger records that difference instead of silently substituting a different estimator.

## Contributing and citation

Issues and pull requests that improve scientific correctness, parity, validation, documentation, interoperability or reproducibility are welcome. For numerical changes, include a focused test and state whether the behavior matches, intentionally differs from, or extends the frozen R reference.

A formal citation will accompany the first archival software release. Until then, cite the repository and exact commit used so the computational environment is reproducible.
