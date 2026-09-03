---
hide:
  - toc
---

<div class="ep-landing-hero" markdown>

<span class="ep-kicker">Scientific Python for eye-tracking + behavioral process measurement</span>

# Import. Measure. Model. Validate.

**eyeprocesspy** brings the frozen **eyeprocess R 0.11.1 scientific surface** into Python for eye-tracking, pupillometry, AOIs, scanpaths, behavioral process data, psychometrics/IRT, quality control, uncertainty, validation, reporting, and reproducible research.

<div class="ep-actions">
<a class="md-button md-button--primary" href="getting-started/">Get started</a>
<a class="md-button" href="guides/">Choose a workflow</a>
<a class="md-button" href="gallery/">See real plots</a>
</div>

</div>

<div class="ep-status-grid">
<div><span class="ep-status-value">0.1.0</span><span class="ep-status-label">stable release</span></div>
<div><span class="ep-status-value">R 0.11.1</span><span class="ep-status-label">frozen scientific reference</span></div>
<div><span class="ep-status-value">1,182 / 1,182</span><span class="ep-status-label">frozen APIs resolved</span></div>
<div><span class="ep-status-value">100%</span><span class="ep-status-label">statements + branches</span></div>
<div><span class="ep-status-value">12 lanes</span><span class="ep-status-label">3 OS × Python 3.11–3.14</span></div>
</div>

!!! info "Published release"
    `eyeprocesspy` 0.1.0 is published on PyPI and archived at DOI [10.5281/zenodo.22285167](https://doi.org/10.5281/zenodo.22285167). The release evidence records **1,458 tests**, **23,085 / 23,085 statements**, and **9,680 / 9,680 branches** at the frozen release gate.

## Install and move

=== "Stable"

    ```bash
    python -m pip install eyeprocesspy
    ```

=== "Exact 0.1.0"

    ```bash
    python -m pip install eyeprocesspy==0.1.0
    ```

=== "Source tag"

    ```bash
    python -m pip install \
      "eyeprocesspy @ git+https://github.com/stefanosbalaskas/eyeprocesspy.git@v0.1.0"
    ```

```python
import eyeprocesspy as ep

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

assert audit["valid"]
```

<div class="ep-callout-line">
<strong>Known-good benchmark:</strong> validate the package and data surface before moving to study exports.
</div>

## Choose your workflow

<div class="ep-card-grid">

<a class="ep-card ep-card-link" href="guides/end-to-end-eye-tracking/">
<span class="ep-card-icon">◎</span>
<h3>Eye-tracking end to end</h3>
<p>Move from imported samples through validity checks, fixations, saccades, AOIs, dwell, scanpaths, transitions, and process summaries.</p>
<span class="ep-card-cta">Open workflow →</span>
</a>

<a class="ep-card ep-card-link" href="guides/gazepoint-import-qc/">
<span class="ep-card-icon">⇥</span>
<h3>Import + quality control</h3>
<p>Read Gazepoint and generic exports, normalize schemas and coordinates, inspect timing, validity, screen bounds, missingness, and provenance.</p>
<span class="ep-card-cta">Open workflow →</span>
</a>

<a class="ep-card ep-card-link" href="guides/pupillometry/">
<span class="ep-card-icon">◉</span>
<h3>Pupillometry</h3>
<p>Work with eye-specific pupil streams, baseline correction, missingness, temporal features, functional representations, and aligned process data.</p>
<span class="ep-card-cta">Open workflow →</span>
</a>

<a class="ep-card ep-card-link" href="guides/process-quality-uncertainty/">
<span class="ep-card-icon">±</span>
<h3>Quality + uncertainty</h3>
<p>Propagate calibration and coordinate uncertainty, audit measurement quality, inspect boundary-sensitive AOIs, and keep uncertainty explicit.</p>
<span class="ep-card-cta">Open workflow →</span>
</a>

<a class="ep-card ep-card-link" href="guides/psychometrics-irt/">
<span class="ep-card-icon">θ</span>
<h3>Psychometrics + IRT</h3>
<p>Connect process measures to reliability, information, SEM, item fit, DIF/DTF, linking, diagnostics, and process-informed measurement.</p>
<span class="ep-card-cta">Open workflow →</span>
</a>

<a class="ep-card ep-card-link" href="guides/reproducibility-release-evidence/">
<span class="ep-card-icon">✓</span>
<h3>Validation + reproducibility</h3>
<p>Use recovery, stress tests, negative controls, frozen-R checks, provenance, release evidence, and auditable reporting boundaries.</p>
<span class="ep-card-cta">Open workflow →</span>
</a>

</div>

## Built for research you can audit

<div class="ep-pillar-grid">
<div class="ep-pillar">
<h3>Parity with provenance</h3>
<p>The frozen R 0.11.1 reference remains explicit alongside the Python implementation, with all 1,182 registered APIs resolved rather than silently dropped.</p>
<a href="parity-and-validation/">Inspect parity →</a>
</div>
<div class="ep-pillar">
<h3>Validation as a first-class output</h3>
<p>The published release closed at 100% statement and branch coverage, with 1,458 tests and a 12-lane Ubuntu/macOS/Windows × Python 3.11–3.14 matrix.</p>
<a href="release-and-reproducibility/">See validation →</a>
</div>
<div class="ep-pillar">
<h3>Conservative interpretation</h3>
<p>Reliability, prediction, gaze, pupil, AOI, and process metrics remain separated from unsupported claims about cognition, emotion, intention, or causal effects.</p>
<a href="faq/">Read boundaries →</a>
</div>
</div>

## From raw export to defensible result

```text
Vendor / generic eye-tracking exports
  → schema + coordinate + timing audit
  → validity / missingness / calibration QC
  → fixation / saccade / pupil preprocessing
  → AOI / dwell / scanpath / transition structure
  → process features + psychometric models
  → uncertainty + reliability + validation evidence
  → plots + reports + provenance
```

<p class="ep-center-link"><a class="md-button md-button--primary" href="guides/">Explore the scientific guides</a></p>

## Generated by the package, not drawn for the website

<div class="ep-feature-gallery">
<figure class="ep-feature-gallery-main">
<img src="assets/gallery/gaze-trace.svg" alt="Canonical gaze trace generated by eyeprocesspy">
<figcaption><strong>Canonical gaze trace.</strong> Validity-aware spatial process inspection from the package workflow.</figcaption>
</figure>
<figure>
<img src="assets/gallery/probabilistic-aoi.svg" alt="Probabilistic AOI membership generated by eyeprocesspy">
<figcaption><strong>Probabilistic AOIs.</strong> Coordinate uncertainty propagated into AOI membership.</figcaption>
</figure>
<figure>
<img src="assets/gallery/process-reliability.svg" alt="Process reliability visualization generated by eyeprocesspy">
<figcaption><strong>Process reliability.</strong> Agreement evidence for repeated behavioral measurements.</figcaption>
</figure>
</div>

The gallery uses package-generated scientific outputs rather than decorative website graphics. Browse the [complete plot gallery](gallery.md).

## Documentation paths

<div class="ep-mini-grid">
<a href="getting-started/">5-minute start</a>
<a href="guides/">Scientific guides</a>
<a href="examples/">Worked examples</a>
<a href="articles/">88 workflows</a>
<a href="reference/">API reference</a>
<a href="parity-and-validation/">Validation + parity</a>
</div>

## Citation

Balaskas, S. (2026). *eyeprocesspy: Vendor-neutral Python infrastructure for eye-tracking and multimodal process data* (Version 0.1.0) [Computer software]. Zenodo. [10.5281/zenodo.22285167](https://doi.org/10.5281/zenodo.22285167)

!!! warning "Interpretation boundary"
    `eyeprocesspy` provides measurement and analysis infrastructure. Reliability is not construct validity, prediction is not causation, probabilistic AOI membership represents modeled coordinate uncertainty rather than probability of attention, and gaze/pupil/process metrics should not be interpreted as psychological states without external validity evidence and an appropriate study design.
