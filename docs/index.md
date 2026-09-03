<div class="ep-hero" markdown>

<div class="ep-hero-copy" markdown>
<div class="ep-kicker">Scientific Python infrastructure for behavioral process data</div>

# eyeprocesspy

**Eye-tracking, pupillometry, AOIs, process measurement, psychometrics, validation, and reproducibility in one auditable package.**

`eyeprocesspy` is the Python companion and deep-parity port of **eyeprocess**, using frozen R **0.11.1** as its scientific reference. It is designed for researchers who need a governed path from raw exports to validated measurement evidence—not a loose collection of gaze utilities.

<div class="ep-actions" markdown>
[Get started](getting-started.md){ .md-button .md-button--primary }
[Manual install](manual-install.md){ .md-button }
[Worked examples](examples/index.md){ .md-button }
[Visual gallery](gallery.md){ .md-button }
</div>

</div>

<div class="ep-hero-mark" markdown>
<img src="https://raw.githubusercontent.com/stefanosbalaskas/eyeprocesspy/main/docs/assets/python-suite-logo.png" alt="Python Suite research packages logo">
</div>

</div>

<div class="ep-status-banner" markdown>
<strong>0.1.0 release candidate</strong> · Frozen API parity is complete. Public release remains gated on the final deep-parity evidence, 100% statement/branch coverage, CI, artifact, and strict documentation checks.
</div>

<div class="ep-stat-grid" markdown>
<div class="ep-stat"><strong>1,182 / 1,182</strong>resolved frozen APIs</div>
<div class="ep-stat"><strong>88 / 88</strong>linked workflow articles</div>
<div class="ep-stat"><strong>3.11–3.14</strong>supported CI interpreters</div>
<div class="ep-stat"><strong>3 OS</strong>Ubuntu · macOS · Windows</div>
</div>

## Start with evidence, not just an API

<div class="ep-grid" markdown>
<div class="ep-card ep-card-accent" markdown>

### :material-laptop: Windows install verified

The hardened manual installer has been exercised successfully with **Python 3.11.9** on Windows. Core verification passed, recommended extras installed, and `pip check` reported no broken requirements.

[Install on Windows](manual-install.md)

</div>
<div class="ep-card" markdown>

### :material-check-decagram: Frozen API resolved

All **1,182 / 1,182** public APIs in the frozen R 0.11.1 reference are resolved in the parity ledger. Cross-language differences remain explicit rather than being hidden behind names alone.

[Inspect parity & validation](parity-and-validation.md)

</div>
<div class="ep-card" markdown>

### :material-chart-box-outline: Process measurement

Move beyond endpoint summaries with scanpaths, transitions, temporal structure, uncertainty, reliability, calibration quality, IRT, DIF, and process-informed measurement.

[Browse scientific guides](guides/index.md)

</div>
<div class="ep-card" markdown>

### :material-shield-search: Release evidence

Deterministic benchmarks, provenance, negative controls, validation atlases, the frozen-R oracle, clean-wheel checks, and strict docs builds are part of the release discipline.

[Release & reproducibility](release-and-reproducibility.md)

</div>
</div>

## See the package in action

<div class="ep-gallery" markdown>

<figure><img src="assets/gallery/gaze-trace.svg" alt="Gaze trace"><figcaption>Canonical gaze trace with validity-aware sample selection.</figcaption></figure>
<figure><img src="assets/gallery/pupil-timeseries.svg" alt="Pupil time series"><figcaption>Eye-specific pupil streams aligned on a shared timebase.</figcaption></figure>
<figure><img src="assets/gallery/probabilistic-aoi.svg" alt="Probabilistic AOI membership"><figcaption>AOI membership after propagating empirical calibration uncertainty.</figcaption></figure>
<figure><img src="assets/gallery/process-reliability.svg" alt="Process reliability"><figcaption>Repeated-measure reliability with Bland–Altman evidence.</figcaption></figure>

</div>

[Explore all package-generated figures →](gallery.md)

## Four tested workflows

<div class="ep-grid" markdown>
<div class="ep-card" markdown>

### Core gaze, AOI & provenance
Validate a canonical dataset, recover scanpaths and transitions, compute entropy, plot the process, and retain provenance.

[Open worked example](examples/core-workflow.md)

</div>
<div class="ep-card" markdown>

### Calibration uncertainty → AOIs
Estimate empirical calibration error, propagate coordinate uncertainty, and inspect boundary-sensitive AOI assignments.

[Open worked example](examples/calibration-probabilistic-aoi.md)

</div>
<div class="ep-card" markdown>

### Process-measure reliability
Estimate ICC, Bland–Altman agreement, and temporal stability while keeping reliability distinct from validity.

[Open worked example](examples/process-reliability.md)

</div>
<div class="ep-card" markdown>

### IRT diagnostics
Visualize information/SEM, item fit, and DIF with links into the wider process-psychometrics surface.

[Open worked example](examples/irt-diagnostics.md)

</div>
</div>

All four examples are deterministic and require no private data. For compact snippets, use the [Cookbook](cookbook.md); for larger workflows, browse the [88-article library](articles/index.md).

## Install the release candidate

=== "Windows recommended"

    Extract the manual-install bundle, then run:

    ```powershell
    Set-ExecutionPolicy -Scope Process Bypass
    .\install_eyeprocesspy.ps1 -WithAllRecommended
    ```

    The installer does **not** require the Windows `py` launcher and can target Python 3.11–3.14. See [Manual installation](manual-install.md).

=== "Canonical wheel"

    ```powershell
    python -m pip install --upgrade .\eyeprocesspy-0.1.0-py3-none-any.whl
    ```

=== "Release branch"

    ```bash
    pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@release/0.1.0-deep-parity"
    ```

## Verify in 30 seconds

```python
import eyeprocesspy as ep

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

print(audit["valid"])
print(data)
```

## One package, multiple research layers

<div class="grid cards" markdown>

-   :material-database-import: **Import → canonical data**

    Vendor/generic readers, Gazepoint workflows, schema validation, coordinate spaces, timebase/event handling, and provenance.

-   :material-eye: **Gaze → process structure**

    Fixations, saccades, AOIs, dwell, scanpaths, transitions, entropy, recurrence, context, and uncertainty.

-   :material-chart-line: **Pupil → multimodal measurement**

    Baseline correction, pupil features, functional pupil models, missingness, synchronized streams, and process quality.

-   :material-chart-bell-curve-cumulative: **Features → psychometrics**

    IRT, process-informed measurement, DIF, conditional norms, reliability, calibration uncertainty, cross-device linking, and validation.

-   :material-shield-check: **Models → validation evidence**

    Recovery, SBC-style evidence, stress tests, negative controls, leakage-aware validation, and evidence atlases.

-   :material-source-branch: **Analysis → reproducibility**

    Deterministic benchmarks, manifests, software-paper evidence, frozen-R oracle checks, and release audits.

</div>

## Choose a scientific guide

<div class="ep-grid" markdown>
<div class="ep-card" markdown>

### Eye-tracking end to end
Import, schema validation, sampling/coordinate audits, gaze/AOI structure, plots, and provenance.

[Open guide](guides/end-to-end-eye-tracking.md)

</div>
<div class="ep-card" markdown>

### Gazepoint import & QC
Profile exports, pair gaze/biometric files, parse events, and validate the canonical result.

[Open guide](guides/gazepoint-import-qc.md)

</div>
<div class="ep-card" markdown>

### Pupillometry
Baseline, missingness, event alignment, functional trajectories, and reporting choices.

[Open guide](guides/pupillometry.md)

</div>
<div class="ep-card" markdown>

### Quality & uncertainty
Reliability, sampling irregularity, calibration-error propagation, and probabilistic AOIs.

[Open guide](guides/process-quality-uncertainty.md)

</div>
<div class="ep-card" markdown>

### Psychometrics & IRT
Information, fit, DIF, process-informed measurement, uncertainty, and validation.

[Open guide](guides/psychometrics-irt.md)

</div>
<div class="ep-card" markdown>

### Reproducibility
Benchmarks, provenance, parity, validation evidence, and release auditing.

[Open guide](guides/reproducibility-release-evidence.md)

</div>
</div>

## Deep-parity evidence

| Dimension | Current state |
| --- | ---: |
| Frozen APIs resolved | **1,182 / 1,182** |
| Frozen R reference | **0.11.1** |
| Frozen articles linked | **88 / 88** |
| P4 numerical `not_started` debt | **0** |
| P6 plot `not_started` debt | **0** |
| Release CI target | **Ubuntu / macOS / Windows × Python 3.11–3.14** |

The deep-parity gate remains intentionally strict: full tests, **100% statements**, **100% branches**, Ruff, clean-wheel verification, frozen-R oracle, and strict docs build.

## Find what you need

- **Start working:** [Getting started](getting-started.md) → [Worked examples](examples/index.md) → [Cookbook](cookbook.md)
- **Install without PyPI:** [Manual installation](manual-install.md)
- **Choose by research question:** [Python-native guides](guides/index.md)
- **Visual capabilities:** [15-figure gallery](gallery.md) → [Plotting reference](reference/plotting.md)
- **Full scientific workflows:** [Featured workflow map](articles/featured-workflows.md) → [88-article library](articles/index.md)
- **API lookup:** [API and plotting reference](reference/index.md)
- **Troubleshooting:** [FAQ](faq.md)
- **Scientific fidelity:** [Parity and validation](parity-and-validation.md)
- **Release audit:** [Release and reproducibility](release-and-reproducibility.md)

!!! warning "Interpretation boundary"
    `eyeprocesspy` provides measurement and analysis infrastructure. Reliability is not construct validity, prediction is not causation, probabilistic AOI membership is modeled coordinate uncertainty rather than probability of attention, and gaze/pupil/process metrics should not be interpreted as psychological states without external validity evidence and an appropriate study design.
