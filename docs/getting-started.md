# Getting started

`eyeprocesspy` connects vendor import, canonical eye-tracking data contracts, preprocessing, gaze/AOI and pupil workflows, process measurement, psychometrics, validation, reproducibility, and scientific plotting.

## Install the current release candidate

Until the first archival/PyPI release is approved, use the CI-tested wheel/manual bundle or the deep-parity branch.

=== "Windows manual bundle"

    Extract the manual-install bundle, open PowerShell in that folder, and run:

    ```powershell
    Set-ExecutionPolicy -Scope Process Bypass
    .\install_eyeprocesspy.ps1
    ```

    The verified Windows path reports:

    ```text
    eyeprocesspy: 0.1.0
    R reference: 0.11.1
    ```

    See [Manual installation](manual-install.md) for troubleshooting and optional dependencies.

=== "Canonical wheel"

    ```powershell
    python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
    ```

    If your browser renamed the file to `eyeprocesspy-0.1.0-py3-none-any (1).whl`, rename it back first. The inserted ` (1)` makes the filename invalid for wheel-tag parsing.

=== "Release branch"

    ```bash
    pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
    ```

=== "Development checkout"

    ```bash
    git clone https://github.com/stefanosbalaskas/eyeprocesspy.git
    cd eyeprocesspy
    git checkout release/0.1.0-deep-parity
    python -m pip install -e ".[dev,docs]"
    ```

## Import and verify versions

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)
```

The Python `0.1.0` release candidate is tied to frozen R **eyeprocess 0.11.1** as its scientific reference.

## Verify the installation without external data

```python
study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

print(audit["valid"])
print(data)
```

The deterministic benchmark is the fastest installation and reproducibility check.

## Canonical data model

```python
schema = ep.eye_schema()
```

`eyeprocesspy` uses a vendor-neutral `EyeDataset` so imported files can retain source fields and provenance while downstream analyses operate on explicit semantic mappings. Canonical components cover recordings, streams, gaze samples, eye/pupil samples, episodes, events, intervals, responses, coordinate spaces, AOIs, biometrics, features, quality, and provenance.

## Import a supported export

```python
eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)
print(issues)
```

For Gazepoint data, dedicated helpers cover gaze, fixations, events, biometrics, file pairing, validation, media/trial workflows, and real-export handling.

## Audit measurement conditions before feature extraction

```python
readiness = ep.analysis_readiness(eye)
rates = ep.audit_sampling_rate(eye)
missing = ep.audit_missingness(eye)
spaces = ep.audit_coordinate_spaces(eye)
manifest = ep.provenance_manifest(eye)
```

Schema validity is not scientific validity. Sampling, calibration, missingness, time alignment, coordinates, preprocessing, feature level, and measurement assumptions remain part of the research design.

## Make your first plots

```python
ax = ep.plot_eye_trace(eye, trial_id="T1")
ax.figure.tight_layout()

ax = ep.plot_gaze_heatmap(eye, trial_id="T1")
ax.figure.tight_layout()

ax = ep.plot_pupil_timeseries(eye, trial_id="T1")
ax.figure.tight_layout()
```

Browse the [15-figure gallery](gallery.md) and [plotting reference](reference/plotting.md) for more examples.

## Advanced analysis routes

- **Gaze/AOI process structure:** scanpaths, transitions, entropy, recurrence, probabilistic/compositional AOIs.
- **Pupillometry:** baseline correction, pupil features, functional pupil, missingness, and synchronized-process workflows.
- **Measurement quality:** calibration error, sampling irregularity, reliability, data quality, and process-measure guardrails.
- **Psychometrics:** IRT foundations, fit, score uncertainty, DIF/DTF, process-informed, dynamic, and advanced models.
- **Validation:** recovery, SBC-style evidence, stress tests, negative controls, grouped/leakage-aware validation, and evidence atlases.
- **Reproducibility:** benchmarks, provenance, manifests, software-paper evidence, and frozen-R parity audits.

## Choose a complete workflow

- [End-to-end eye-tracking](guides/end-to-end-eye-tracking.md)
- [Gazepoint import and QC](guides/gazepoint-import-qc.md)
- [Pupillometry](guides/pupillometry.md)
- [Process quality and calibration uncertainty](guides/process-quality-uncertainty.md)
- [Psychometrics and IRT](guides/psychometrics-irt.md)
- [Reproducibility and release evidence](guides/reproducibility-release-evidence.md)
- [88-article parity library](articles/index.md)

## Optional backends and parity discipline

Install only the scientific backends required by your workflow. Unavailable exact R engines remain explicitly gated: `eyeprocesspy` does not silently replace an unavailable estimator with a different model and call it parity.

!!! warning "Interpretation boundary"
    Gaze, pupil, biometric, and psychometric outputs are measurement evidence, not automatic psychological labels. Use validation, uncertainty, provenance, and an appropriate study design when making substantive claims.
