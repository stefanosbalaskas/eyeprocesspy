<p align="center">
  <img src="https://raw.githubusercontent.com/stefanosbalaskas/eyeprocesspy/main/docs/assets/python-suite-logo.png" width="270" alt="Python Suite research packages logo">
</p>

<h1 align="center">eyeprocesspy</h1>

<p align="center">
  <strong>Reproducible Python infrastructure for eye-tracking, pupillometry, AOIs, process data, psychometrics, and multimodal behavioral measurement.</strong>
</p>

<p align="center">
  <a href="https://stefanosbalaskas.github.io/eyeprocesspy/">Documentation</a> ·
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/gallery.md">Visual gallery</a> ·
  <a href="docs/articles/index.md">88 workflow articles</a>
</p>

[![CI](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml/badge.svg?branch=release%2F0.1.0-deep-parity)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/ci.yml)
[![Documentation](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml/badge.svg?branch=release%2F0.1.0-deep-parity)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/docs.yml)
[![Deep parity audit](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml/badge.svg?branch=release%2F0.1.0-deep-parity)](https://github.com/stefanosbalaskas/eyeprocesspy/actions/workflows/deep-parity-audit.yml)
[![Frozen API](https://img.shields.io/badge/frozen%20API-1182%20%2F%201182-success)](IMPLEMENTATION_STATUS.md)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue)](https://www.python.org/)
[![R reference 0.11.1](https://img.shields.io/badge/R%20reference-0.11.1-276DC3)](docs/parity-and-validation.md)

`eyeprocesspy` is the Python companion and deep-parity port of the R package **eyeprocess**, with frozen **eyeprocess 0.11.1** as the scientific reference. It brings vendor import, canonical data contracts, preprocessing, gaze/AOI analysis, pupil workflows, process measurement, IRT, validation, scientific plots, provenance, and reporting into one auditable package.

> **Release-candidate status:** API parity is complete, while `release/0.1.0-deep-parity` remains deliberately gated on the final evidence, coverage, CI, artifact, and documentation checks. The coverage threshold is not reduced to manufacture a green release.

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

The release evidence gate requires the full pytest suite, **100% statement coverage**, **100% branch coverage**, Ruff, clean wheel/sdist install checks, the frozen-R oracle, and a strict documentation build.

## Windows manual installation — verified

The hardened installer has now been exercised successfully on a real Windows installation with **Python 3.11.9**. Package verification passed, the recommended extras installed, and `python -m pip check` reported **No broken requirements found**.

Extract the manual-install ZIP and either double-click:

```text
RUN_INSTALL_RECOMMENDED.cmd
```

or run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_eyeprocesspy.ps1 -WithAllRecommended
```

The installer does **not** require the Windows `py` launcher. It detects supported Python 3.11–3.14 installations through available launchers, active virtual/Conda environments, common Conda paths, and standard per-user Python locations. An interpreter can also be supplied explicitly:

```powershell
.\install_eyeprocesspy.ps1 -PythonCommand "C:\Path\To\python.exe" -WithAllRecommended
```

The recommended bundle installs `plots`, `psychometrics`, `ml`, `arrow`, `physio`, and `docs`. See the full [manual-install guide](docs/manual-install.md).

Direct wheel installation remains available:

```powershell
python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
```

Or install directly from this branch:

```bash
pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
```

## Why eyeprocesspy?

- **One coherent data model** for recordings, gaze samples, eye/pupil samples, fixations and episodes, events, intervals, AOIs, responses, features, quality, and provenance.
- **Scientific parity first:** **1,182 / 1,182** frozen APIs are resolved against eyeprocess 0.11.1, with governed records for unavoidable cross-language differences.
- **Process data as first-class evidence:** scanpaths, transitions, temporal structure, uncertainty, reliability, and psychometrics live in the same analytical surface.
- **Measurement guardrails:** calibration uncertainty, quality, reliability, DIF/fairness, and process metrics retain explicit interpretation boundaries.
- **Reproducibility by construction:** deterministic benchmarks, provenance, validation evidence, software-paper evidence, and release audits are built in.
- **Broad scientific plotting surface:** gaze, AOI, pupil, quality, IRT, process-measurement, validation, and model-diagnostic graphics are supported through Matplotlib-oriented workflows.

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

**[Open the complete visual gallery →](docs/gallery.md)**

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
| **AOI uncertainty** | Hard, probabilistic and compositional AOIs; calibration-error propagation and sensitivity |
| **Pupil & multimodal analysis** | Baselines, pupil features, functional pupil, missingness, synchronized streams, staged multimodal models |
| **Psychometrics & IRT** | Foundations, scoring, fit, Q3, DIF/DTF, process-informed/dynamic/advanced IRT, diagnostics |
| **Measurement intelligence** | Reliability, calibration uncertainty, process guardrails, linking, norms, fairness, item-bank optimization |
| **Validation** | Recovery, SBC-style evidence, stress tests, negative controls, grouped/leakage-aware validation, evidence atlases |
| **Reproducibility** | Bundled benchmarks, provenance, manifests, frozen-R oracle, software-paper and release evidence |
| **Plots & reporting** | Publication-oriented plots, validation visualizations, scientific evidence/reporting helpers |

## Tested workflows

```bash
python examples/complete_workflow.py
python examples/calibration_probabilistic_aoi.py
python examples/process_reliability.py
python examples/irt_diagnostics.py
```

| Workflow | What it demonstrates |
| --- | --- |
| [Core gaze/AOI/provenance](docs/examples/core-workflow.md) | canonical dataset, validation, scanpaths, transitions, entropy, plots and provenance |
| [Calibration uncertainty & probabilistic AOIs](docs/examples/calibration-probabilistic-aoi.md) | empirical calibration error, uncertainty ellipse and boundary-sensitive AOI assignment |
| [Process reliability](docs/examples/process-reliability.md) | ICC, Bland–Altman agreement and temporal stability |
| [IRT diagnostics](docs/examples/irt-diagnostics.md) | information/SEM, item fit and DIF plotting |

For shorter tasks use the **[Cookbook](docs/cookbook.md)**; for broader walkthroughs use the **[Python-native guides](docs/guides/)** and **[88-article workflow library](docs/articles/)**.

## Documentation

- **Website:** https://stefanosbalaskas.github.io/eyeprocesspy/
- [Getting started](docs/getting-started.md)
- [Manual installation](docs/manual-install.md)
- [Runnable examples](docs/examples/index.md)
- [Practical cookbook](docs/cookbook.md)
- [Visual gallery](docs/gallery.md)
- [Python-native guides](docs/guides/)
- [88-article workflow library](docs/articles/)
- [API and plotting reference](docs/reference/)
- [FAQ](docs/faq.md)
- [Parity and validation](docs/parity-and-validation.md)
- [Release and reproducibility](docs/release-and-reproducibility.md)

## Scientific boundary

`eyeprocesspy` provides **measurement and analysis infrastructure**. A metric is not automatically a validated psychological construct, diagnosis, or causal explanation. Reliability does not establish construct validity; prediction does not establish causation; probabilistic AOI membership reflects modeled coordinate uncertainty rather than probability of attention; and gaze, pupil, and biometric measures require an appropriate design, measurement model, and ethical interpretation.

## Relationship to R eyeprocess

The Python package is developed against the frozen `eyeprocess 0.11.1` reference. API, articles, data, plots, backends, numerical evidence, and unavoidable language-specific divergences are tracked explicitly. Python-native extensions are separated from reference parity so they do not masquerade as R-equivalent behavior.

## Release discipline

`eyeprocesspy 0.1.0` remains deliberately unreleased while the deep-parity gate is open. GitHub Release, PyPI publication, and archival actions should occur only after the controlling release head satisfies the declared evidence requirements.

## License

See [`LICENSE`](LICENSE).
