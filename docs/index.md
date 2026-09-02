<div class="ep-hero" markdown>

<div class="ep-kicker">Scientific Python infrastructure for behavioral process data</div>

# eyeprocesspy

**Eye-tracking, pupillometry, AOIs, process measurement, psychometrics, validation and reproducibility in one auditable package.**

`eyeprocesspy` is the Python companion and deep-parity port of **eyeprocess**, using frozen R **0.11.1** as its scientific reference. The package is designed for researchers who need a governed workflow from raw exports to validated measurement evidence—not a loose collection of gaze utilities.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Runnable examples](examples/index.md){ .md-button }
[Visual gallery](gallery.md){ .md-button }
[Browse 88 articles](articles/index.md){ .md-button }

</div>

## See the package in action

<div class="ep-gallery" markdown>

<figure><img src="assets/gallery/gaze-trace.svg" alt="Gaze trace"><figcaption>Canonical gaze trace with validity-aware sample selection.</figcaption></figure>
<figure><img src="assets/gallery/pupil-timeseries.svg" alt="Pupil time series"><figcaption>Eye-specific pupil streams aligned on a shared timebase.</figcaption></figure>
<figure><img src="assets/gallery/probabilistic-aoi.svg" alt="Probabilistic AOI membership"><figcaption>AOI membership after propagating empirical calibration uncertainty.</figcaption></figure>
<figure><img src="assets/gallery/process-reliability.svg" alt="Process reliability"><figcaption>Repeated-measure reliability with Bland–Altman evidence.</figcaption></figure>

</div>

[Explore all package-generated figures →](gallery.md)

## One package, multiple research layers

<div class="grid cards" markdown>

-   :material-database-import: **Import → canonical data**

    Vendor/generic readers, Gazepoint workflows, schema validation, coordinate spaces, timebase/event handling and provenance.

-   :material-eye: **Gaze → process structure**

    Fixations, saccades, AOIs, dwell, scanpaths, transitions, entropy, recurrence, context and uncertainty.

-   :material-chart-line: **Pupil → multimodal measurement**

    Baseline correction, pupil features, functional pupil models, missingness, synchronized streams and process quality.

-   :material-chart-bell-curve-cumulative: **Features → psychometrics**

    IRT, process-informed measurement, DIF, conditional norms, reliability, calibration uncertainty, cross-device linking and validation live in the same analysis ecosystem.

-   :material-shield-check: **Models → validation evidence**

    Recovery, SBC-style evidence, stress tests, negative controls, leakage-aware validation and evidence atlases.

-   :material-source-branch: **Analysis → reproducibility**

    Deterministic benchmarks, manifests, software-paper evidence, frozen-R oracle checks and release audits.

</div>

## Install the release candidate

=== "Manual wheel"

    ```powershell
    py -3 -m pip install .\eyeprocesspy-0.1.0-py3-none-any.whl
    py -3 -c "import eyeprocesspy as ep; print(ep.__version__, ep.__r_reference_version__)"
    ```

    CI publishes the tested wheel and source distribution as `eyeprocesspy-manual-install-<commit>` after a clean install/import check.

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

## Deep-parity evidence

| Dimension | Current state |
| --- | ---: |
| Frozen APIs resolved | **1,182 / 1,182** |
| Frozen R reference | **0.11.1** |
| Frozen articles linked | **88 / 88** |
| P4 numerical `not_started` debt | **0** |
| P6 plot `not_started` debt | **0** |
| Release CI | **Ubuntu / macOS / Windows × Python 3.11–3.14** |

The deep-parity gate remains intentionally strict: full tests, **100% statements**, **100% branches**, Ruff, clean-wheel verification, frozen-R oracle and strict docs build.

## Choose your route

- **First analysis:** [Getting started](getting-started.md) → [Runnable examples](examples/index.md)
- **Visual capabilities:** [Gallery](gallery.md)
- **Full scientific workflows:** [Featured workflow map](articles/featured-workflows.md) → [88-article library](articles/index.md)
- **API lookup:** [Reference](reference/index.md)
- **Scientific fidelity:** [Parity and validation](parity-and-validation.md)
- **Release audit:** [Release and reproducibility](release-and-reproducibility.md)

!!! warning "Interpretation boundary"
    `eyeprocesspy` provides measurement and analysis infrastructure. Reliability is not construct validity, prediction is not causation, and gaze/pupil/process metrics should not be interpreted as psychological states without external validity evidence and an appropriate study design.
