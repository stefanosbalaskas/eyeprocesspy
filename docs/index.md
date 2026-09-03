---
hide:
  - toc
---

<div class="ep-hero" markdown>

<div class="ep-hero-copy" markdown>
<div class="ep-kicker">Scientific Python infrastructure for behavioral process data</div>

# eyeprocesspy

**From raw eye-tracking and multimodal exports to validated, reproducible measurement evidence.**

`eyeprocesspy` is the Python companion and deep-parity port of **eyeprocess**, with frozen R **0.11.1** as the scientific reference. It brings import, preprocessing, gaze/AOI analysis, pupillometry, process measurement, psychometrics/IRT, validation, plotting, provenance, and reporting into one governed research-software surface.

<div class="ep-actions" markdown>
[Install from PyPI](#install){ .md-button .md-button--primary }
[Start here](getting-started.md){ .md-button }
[Worked examples](examples/index.md){ .md-button }
[API reference](reference/api.md){ .md-button }
</div>

<div class="ep-install-inline" markdown>
`pip install eyeprocesspy`
</div>

</div>

<div class="ep-hero-mark" markdown>
<img src="https://raw.githubusercontent.com/stefanosbalaskas/gpbiometricspy/main/docs/assets/python-suite-logo.png" alt="Python Suite research packages logo">
</div>

</div>

<div class="ep-release-bar" markdown>
<span class="ep-release-label">Published · v0.1.0</span>
<a href="https://pypi.org/project/eyeprocesspy/">PyPI</a>
<span>·</span>
<a href="https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.1.0">GitHub Release</a>
<span>·</span>
<a href="https://doi.org/10.5281/zenodo.22285167">Zenodo DOI 10.5281/zenodo.22285167</a>
</div>

<div class="ep-stat-grid" markdown>
<div class="ep-stat"><strong>1,182 / 1,182</strong><span>frozen APIs resolved</span></div>
<div class="ep-stat"><strong>1,458</strong><span>release tests passed</span></div>
<div class="ep-stat"><strong>100%</strong><span>statement coverage</span></div>
<div class="ep-stat"><strong>100%</strong><span>branch coverage</span></div>
<div class="ep-stat"><strong>88 / 88</strong><span>workflow articles linked</span></div>
<div class="ep-stat"><strong>12 lanes</strong><span>3 OS × Python 3.11–3.14</span></div>
</div>

## From raw data to defensible evidence

<div class="ep-grid ep-grid-3" markdown>
<div class="ep-card ep-card-accent" markdown>

### :material-database-import: Import & canonicalize

Bring vendor and generic exports into one canonical model for recordings, gaze, pupil, events, intervals, AOIs, responses, features, quality, and provenance.

[Start an eye-tracking workflow](guides/end-to-end-eye-tracking.md)

</div>
<div class="ep-card" markdown>

### :material-eye: Measure behavior

Recover fixations, saccades, dwell, scanpaths, transitions, entropy, recurrence, pupil features, uncertainty-aware AOIs, and multimodal process structure.

[Browse scientific guides](guides/index.md)

</div>
<div class="ep-card" markdown>

### :material-shield-check: Validate & reproduce

Connect reliability, calibration uncertainty, IRT, DIF, recovery, stress tests, negative controls, provenance, and release evidence without hiding cross-language differences.

[Inspect parity & validation](parity-and-validation.md)

</div>
</div>

## Install

=== "PyPI"

    ```bash
    pip install eyeprocesspy
    ```

=== "Reproducible 0.1.0"

    ```bash
    pip install eyeprocesspy==0.1.0
    ```

=== "Windows installer"

    ```powershell
    Set-ExecutionPolicy -Scope Process Bypass
    .\install_eyeprocesspy.ps1 -WithAllRecommended
    ```

    The hardened installer has been exercised successfully on Windows with **Python 3.11.9**, including recommended extras and a clean `pip check`. See [Manual installation](manual-install.md).

=== "Source tag"

    ```bash
    pip install "git+https://github.com/stefanosbalaskas/eyeprocesspy.git@v0.1.0"
    ```

## See the package in action

<div class="ep-gallery" markdown>

<figure><img src="assets/gallery/gaze-trace.svg" alt="Gaze trace"><figcaption>Canonical gaze trace with validity-aware sample selection.</figcaption></figure>
<figure><img src="assets/gallery/pupil-timeseries.svg" alt="Pupil time series"><figcaption>Eye-specific pupil streams aligned on a shared timebase.</figcaption></figure>
<figure><img src="assets/gallery/probabilistic-aoi.svg" alt="Probabilistic AOI membership"><figcaption>AOI membership after propagating empirical calibration uncertainty.</figcaption></figure>
<figure><img src="assets/gallery/process-reliability.svg" alt="Process reliability"><figcaption>Repeated-measure reliability with Bland–Altman evidence.</figcaption></figure>

</div>

<div class="ep-centered-link" markdown>
[Explore all package-generated figures →](gallery.md)
</div>

## Choose a tested workflow

<div class="ep-grid" markdown>
<div class="ep-card" markdown>

### Core gaze, AOI & provenance
Validate a canonical dataset, recover scanpaths and transitions, compute entropy, plot the process, and retain provenance.

[Open workflow](examples/core-workflow.md)

</div>
<div class="ep-card" markdown>

### Calibration uncertainty → AOIs
Estimate empirical calibration error, propagate coordinate uncertainty, and inspect boundary-sensitive AOI assignments.

[Open workflow](examples/calibration-probabilistic-aoi.md)

</div>
<div class="ep-card" markdown>

### Process-measure reliability
Estimate ICC, Bland–Altman agreement, and temporal stability while keeping reliability distinct from validity.

[Open workflow](examples/process-reliability.md)

</div>
<div class="ep-card" markdown>

### IRT diagnostics
Visualize information/SEM, item fit, and DIF with links into the wider process-psychometrics surface.

[Open workflow](examples/irt-diagnostics.md)

</div>
</div>

All four examples are deterministic and require no private data. For compact patterns, use the [Cookbook](cookbook.md); for larger analyses, browse the [88-workflow library](articles/index.md).

## Scientific capability map

<div class="grid cards" markdown>

-   :material-database-import: **Import → canonical data**

    Vendor/generic readers, Gazepoint workflows, schema validation, coordinate spaces, timebase/event handling, file pairing, and provenance.

-   :material-eye: **Gaze → process structure**

    Fixations, saccades, AOIs, dwell, scanpaths, transitions, entropy, recurrence, context, and uncertainty.

-   :material-chart-line: **Pupil → multimodal measurement**

    Baseline correction, pupil features, functional pupil models, missingness, synchronized streams, and process quality.

-   :material-chart-bell-curve-cumulative: **Features → psychometrics**

    IRT, process-informed measurement, DIF/DTF, conditional norms, reliability, calibration uncertainty, linking, and diagnostics.

-   :material-flask-outline: **Models → validation evidence**

    Recovery, SBC-style evidence, stress tests, negative controls, grouped/leakage-aware validation, and evidence atlases.

-   :material-source-branch: **Analysis → reproducibility**

    Deterministic benchmarks, manifests, provenance, frozen-R oracle checks, software-paper evidence, and release audits.

</div>

## Release assurance

<div class="ep-assurance" markdown>
<div markdown>

### Exact deep-parity gate

The published `0.1.0` release passed **1,458 tests**, **23,085 / 23,085 statements**, and **9,680 / 9,680 branches**. P4 numerical and P6 plot `not_started` debt are both **zero**.

</div>
<div markdown>

### Cross-platform release

The release matrix covers **Ubuntu, macOS, and Windows** across **Python 3.11–3.14**, with Ruff, a frozen R 0.11.1 oracle, clean-wheel install/import, strict documentation, PyPI Trusted Publishing, provenance attestation, and Zenodo archival.

</div>
</div>

<div class="ep-actions ep-actions-secondary" markdown>
[Release validation](https://github.com/stefanosbalaskas/eyeprocesspy/blob/main/RELEASE_VALIDATION.md){ .md-button }
[Reproducibility guide](release-and-reproducibility.md){ .md-button }
[GitHub Release](https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.1.0){ .md-button }
[Zenodo record](https://doi.org/10.5281/zenodo.22285167){ .md-button }
</div>

## Verify in 30 seconds

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

## Find what you need

- **Get productive:** [Getting started](getting-started.md) · [Examples](examples/index.md) · [Cookbook](cookbook.md)
- **Eye-tracking & AOIs:** [End to end](guides/end-to-end-eye-tracking.md) · [Gazepoint import & QC](guides/gazepoint-import-qc.md)
- **Pupil & uncertainty:** [Pupillometry](guides/pupillometry.md) · [Quality & uncertainty](guides/process-quality-uncertainty.md)
- **Psychometrics:** [Psychometrics & IRT](guides/psychometrics-irt.md) · [IRT diagnostics](examples/irt-diagnostics.md)
- **Workflow depth:** [Featured workflows](articles/featured-workflows.md) · [88-article library](articles/index.md)
- **Lookup & help:** [API reference](reference/api.md) · [Plotting reference](reference/plotting.md) · [FAQ](faq.md)

## Cite the release

> Balaskas, S. (2026). *eyeprocesspy: Vendor-neutral Python infrastructure for eye-tracking and multimodal process data* (Version 0.1.0) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.22285167>

For reproducibility, report both `eyeprocesspy.__version__` and `eyeprocesspy.__r_reference_version__`.

!!! warning "Interpretation boundary"
    `eyeprocesspy` provides measurement and analysis infrastructure. Reliability is not construct validity, prediction is not causation, probabilistic AOI membership is modeled coordinate uncertainty rather than probability of attention, and gaze/pupil/process metrics should not be interpreted as psychological states without external validity evidence and an appropriate study design.
