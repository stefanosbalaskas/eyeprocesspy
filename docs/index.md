# eyeprocesspy

**Reproducible Python infrastructure for eye-tracking, pupillometry, process data, psychometrics, and multimodal behavioral measurement.**

`eyeprocesspy` is the Python companion and deep-parity port of the R package **eyeprocess**, using the frozen **0.11.1** R release as its scientific reference. It is designed for researchers who need more than a collection of gaze utilities: one package connects import, canonical data contracts, preprocessing, gaze/AOI and pupil analysis, process measurement, psychometrics, validation, reproducibility, and publication-ready evidence.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Explore the API](reference/index.md){ .md-button }
[Parity & validation](parity-and-validation.md){ .md-button }

## What makes it different

<div class="grid cards" markdown>

-   :material-eye: **Eye-tracking as process data**

    Fixations, saccades, AOIs, scanpaths, transitions, recurrence, temporal structure, and uncertainty are treated as analyzable behavioral processes rather than isolated summary metrics.

-   :material-chart-bell-curve-cumulative: **Psychometrics and measurement**

    IRT, process-informed measurement, DIF, conditional norms, reliability, calibration uncertainty, cross-device linking, and validation live in the same analysis ecosystem.

-   :material-shield-check: **Reproducibility first**

    Provenance, deterministic benchmarks, validation evidence, manifests, software-paper evidence, and explicit guardrails make analysis decisions auditable.

-   :material-language-python: **Deep Python parity**

    The frozen reference contains **1,182 resolved APIs** and **88 linked articles**, with explicit parity records where R and Python cannot be scientifically identical.

</div>

## Install the release candidate

The release candidate is currently hardened on `release/0.1.0-deep-parity`:

```bash
pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
```

The final PyPI installation command will be promoted here only after the complete release evidence gate is green.

## Verify your installation in 30 seconds

The package ships a benchmark study so a clean installation can be checked without downloading external data:

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

| Workflow | What eyeprocesspy provides |
| --- | --- |
| **Import & data contracts** | Vendor/generic readers, Gazepoint workflows, canonical schema, coordinate spaces, timebase and event handling |
| **Preprocessing & features** | Fixation/saccade workflows, pupil preprocessing, gaze/AOI metrics, temporal and sequence features |
| **Advanced process analysis** | Probabilistic/compositional AOIs, scanpaths, recurrence, process episodes, functional pupil, missingness and uncertainty |
| **Psychometrics & IRT** | IRT foundations, diagnostics, scoring, DIF, adaptive/process-informed models, multidimensional and advanced measurement tools |
| **Measurement intelligence** | Reliability, calibration uncertainty, device linking, fairness, process norms, item-bank optimization, grouped validation |
| **Validation & reproducibility** | Recovery, SBC, stress tests, negative controls, benchmarks, evidence atlases, provenance and reproducibility manifests |
| **Communication** | Scientific plots, reporting helpers, software-paper evidence and release-validation artifacts |

## Deep-parity release state

| Dimension | State |
| --- | ---: |
| Frozen APIs resolved | **1,182 / 1,182** |
| R reference | **0.11.1** |
| Frozen articles linked | **88 / 88** |
| P4 numerical `not_started` debt | **0** |
| P6 plot `not_started` debt | **0** |
| CI matrix | **Ubuntu / macOS / Windows × Python 3.11–3.14** |

The deep-parity gate is intentionally strict: a release requires the complete tests, **100% statement coverage**, **100% branch coverage**, Ruff, clean-wheel installation, the frozen-R oracle, and a strict documentation build. The project does not lower the threshold to manufacture a green badge.

## Scientific boundary

`eyeprocesspy` provides measurement and analysis infrastructure; it does not turn a process metric into a validated psychological construct by itself. Reliability is not construct validity, prediction is not causation, and biometric/process measures require a defensible study design, measurement model, and ethical interpretation.

The package therefore exposes caveats, validation objects, provenance, uncertainty and evidence structures alongside analytical results.

## Start here

- **New to the package?** Read [Getting started](getting-started.md).
- **Evaluating scientific fidelity?** Read [Parity and validation](parity-and-validation.md).
- **Reproducing or auditing a release?** Read [Release and reproducibility](release-and-reproducibility.md).
- **Looking for functions?** Browse the [API reference](reference/index.md).
- **Looking for complete workflows?** Browse the [articles](articles/).

## Project links

- [GitHub repository](https://github.com/stefanosbalaskas/eyeprocesspy)
- [Release branch](https://github.com/stefanosbalaskas/eyeprocesspy/tree/release/0.1.0-deep-parity)
- [Deep-parity audit](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml)
