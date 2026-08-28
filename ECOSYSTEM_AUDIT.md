# Python ecosystem audit — initial architecture freeze

Audit date: 2026-08-29. This document guides optional backends and independent validation; it does **not** replace the frozen R 0.11.1 public API.

| Package / ecosystem | Current evidence | Decision for eyeprocesspy |
|---|---|---|
| pymovements | Active eye-movement processing package; PyPI shows 0.27.1 released 2026-07-09. Provides dataset handling, preprocessing, event detection and plotting. | **Adopt as optional interoperability + independent validation backend after parity.** Do not replace canonical eyeprocesspy readers/events API. |
| REMoDNaV | Mature velocity-based event detector for natural viewing; latest PyPI release is older (~2023). | **Optional detector/validation backend.** Pin/test carefully; no core dependency. |
| multimatch-gaze | Python reimplementation of MultiMatch scanpath similarity. | **Optional scanpath interoperability backend.** Validate output contract before exposing extension. |
| Pupil Labs Neon/Core APIs | Official docs expose Python-oriented recording/realtime access and synchronization/trigger workflows. | **Use official APIs for post-parity live/recording extensions.** Frozen R readers are ported independently first. |
| NeuroKit2 | PyPI 0.2.13 (2026-03-02), Python >=3.10; classifiers include 3.14. ECG/EDA/PPG etc. | **Strong optional physiology validation/interoperability backend.** |
| BioSPPy | PyPI 2.2.4 (2025-11-07); ECG/EDA/PPG/HRV, feature extraction and signal quality. | **Optional physiology interoperability/benchmark backend.** |
| CmdStanPy | Official Stan Python interface; docs 1.3.0; lightweight Python layer around CmdStan. | **Preferred canonical backend for the 13 frozen Stan programs.** Keep optional. |
| PyMC | PyPI 6.3.1 released 2026-08-16; Python >=3.12 with 3.14 classifier. | **Python-native Bayesian extension/backend after canonical Stan parity.** Never call PyMC rewrites algorithmically identical without validation. |
| ArviZ | Current 1.x diagnostics/inference-data ecosystem. | **Preferred posterior diagnostics/interop layer for Python-native probabilistic workflows.** |
| GIRTH | IRT estimation package, but PyPI release is ~4.8 years old. | **Reference/optional validation only.** Do not make core psychometric parity depend on it. |
| catsim | Current docs 0.21.0; CAT simulation, 1PL/2PL/3PL utilities; tested through Python 3.12 in its docs. | **Optional CAT comparison backend.** Keep Python 3.13/3.14 compatibility isolated until verified. |
| pylsl | Python interface to Lab Streaming Layer; current PyPI project active, may require liblsl on non-Windows/some configurations. | **Optional streaming extra.** Explicit runtime capability checks required. |
| pyxdf | PyPI 1.17.5 released 2026-06-15, Python >=3.10. | **Preferred XDF import/export interoperability backend.** |
| MNE-BIDS | Docs 0.19.0; robust BIDS read/write ecosystem for MNE-supported modalities. | **Optional BIDS/electrophysiology interoperability.** Eye-tracking BIDS contract must follow the current BIDS spec, not MNE assumptions alone. |
| PyArrow | PyPI 25.0.1 released 2026-08-10; supports Python 3.10–3.14. | **Preferred optional Arrow/Parquet backend.** Reproduce R codec-fallback semantics and schema metadata explicitly. |

## Core dependency policy

The parity core remains conservative: NumPy + pandas plus the Python standard library. Heavy modelling, storage, streaming, physiology and vendor SDKs stay behind extras.

## Compatibility policy

The core targets Python 3.11–3.14. Optional dependencies that lag Python 3.14 are isolated in dedicated CI lanes rather than lowering the package-wide ceiling.

## Naming check

An exact-name web/PyPI search did not surface an established `eyeprocesspy` project during this audit, and the connected GitHub account currently has no `stefanosbalaskas/eyeprocesspy` repository. Re-check PyPI immediately before the first public publication.

## Official/current sources consulted

- https://pymovements.readthedocs.io/
- https://pypi.org/project/pymovements/
- https://pypi.org/project/remodnav/
- https://github.com/adswa/multimatch_gaze
- https://docs.pupil-labs.com/neon/real-time-api/
- https://docs.pupil-labs.com/core/developer/
- https://pypi.org/project/neurokit2/
- https://pypi.org/project/biosppy/
- https://mc-stan.org/cmdstanpy/
- https://pypi.org/project/pymc/
- https://python.arviz.org/
- https://pypi.org/project/girth/
- https://douglasrizzo.com.br/catsim/
- https://pypi.org/project/pylsl/
- https://pypi.org/project/pyxdf/
- https://mne.tools/mne-bids/stable/
- https://arrow.apache.org/docs/python/
