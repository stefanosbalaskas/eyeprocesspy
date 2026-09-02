# Getting started

## Install the current release candidate

Until the first archival/PyPI release is approved, use the CI-tested wheel or the deep-parity branch rather than an unversioned package-index install.

=== "Manual wheel"

    Download the `eyeprocesspy-manual-install-<commit>` CI artifact, extract it, and install the wheel:

    ```powershell
    py -3 -m pip install .\eyeprocesspy-0.1.0-py3-none-any.whl
    py -3 -c "import eyeprocesspy as ep; print(ep.__version__, ep.__r_reference_version__)"
    ```

=== "Release branch"

    ```bash
    pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
    ```

=== "Development checkout"

    ```bash
    git clone https://github.com/stefanosbalaskas/eyeprocesspy.git
    cd eyeprocesspy
    git checkout release/0.1.0-deep-parity
    uv sync --extra dev
    uv run pytest
    ```

For plotting workflows, install the plotting dependencies through the package extra or development environment.

## Import and verify versions

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)
```

The Python 0.1.0 release candidate is tied to frozen R `eyeprocess` **0.11.1** as its scientific reference.

## Verify the installation without external data

```python
study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

print(audit["valid"])
print(data)
```

The deterministic benchmark is the fastest installation and reproducibility check.

## Canonical schema

```python
schema = ep.eye_schema()
```

`eyeprocesspy` uses a vendor-neutral `EyeDataset` so imported files can retain their source fields and provenance while downstream analyses operate on explicit semantic mappings. Canonical components cover recordings, streams, gaze samples, eye/pupil samples, episodes, events, intervals, responses, coordinate spaces, AOIs, biometrics, features, quality and provenance.

## Import a supported export

```python
data = ep.read_eye_export("path/to/export", vendor="auto")
issues = ep.validate_eye_dataset(data)
```

For Gazepoint data, dedicated helpers cover gaze, fixations, events, biometrics, file pairing, validation, media/trial workflows and real-export handling.

## Start with a visual workflow

```python
import matplotlib.pyplot as plt

ax = ep.plot_eye_trace(data, trial_id="trial-01")
plt.show()

ax = ep.plot_scanpath(data, trial_id="trial-01")
plt.show()
```

Then explore the [visual gallery](gallery.md) and [runnable examples](examples/index.md).

## Audit before analysis

Quality and governance functions are designed to make transformations, missingness, sampling assumptions, coordinate conversions, exclusions and feature levels visible rather than silently changing the data.

```python
readiness = ep.analysis_readiness(data)
manifest = ep.provenance_manifest(data)
```

## Advanced analysis routes

- **Gaze/AOI process structure:** scanpaths, transitions, entropy, recurrence, probabilistic/compositional AOIs.
- **Pupillometry:** baseline correction, pupil features, functional pupil and missingness workflows.
- **Measurement quality:** calibration error, sampling irregularity, reliability and process-measure guardrails.
- **Psychometrics:** IRT foundations, fit, score uncertainty, DIF/DTF, process-informed/dynamic/advanced models.
- **Validation:** recovery, SBC-style evidence, stress tests, negative controls and grouped/leakage-aware validation.
- **Reproducibility:** benchmarks, provenance, manifests, software-paper evidence and frozen-R parity audits.

Use the [featured workflow map](articles/featured-workflows.md) to choose a scientific route across the full article library.

## Optional backends and parity discipline

Install only the scientific backends required by your workflow. Unavailable exact R engines remain explicitly gated: `eyeprocesspy` does not silently replace an unavailable estimator with a different model and call it parity.

!!! warning "Interpretation boundary"
    Gaze, pupil, biometric and psychometric outputs are measurement evidence, not automatic psychological labels. Use validation, uncertainty, provenance and an appropriate study design when making substantive claims.
