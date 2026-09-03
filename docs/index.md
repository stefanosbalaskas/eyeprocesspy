---
hide:
  - toc
---

<div class="ep-home" markdown>

<section class="ep-home-hero" markdown>
<div class="ep-home-hero__copy" markdown>

<div class="ep-eyebrow">eyeprocesspy · v0.1.0 · scientific Python</div>

# From raw gaze to defensible evidence.

`eyeprocesspy` is vendor-neutral infrastructure for **eye-tracking, pupillometry, AOIs, behavioral process data, psychometrics, validation, and reproducible research**. It turns heterogeneous exports into an auditable measurement workflow without hiding the assumptions between import and inference.

<div class="ep-hero-actions">
<a class="ep-cta ep-cta--primary" href="getting-started/">Get started</a>
<a class="ep-cta ep-cta--secondary" href="examples/">See examples</a>
<a class="ep-cta ep-cta--ghost" href="reference/api/">API reference</a>
</div>

<div class="ep-command">
<span class="ep-command__prompt">$</span>
<code>pip install eyeprocesspy</code>
<a href="https://pypi.org/project/eyeprocesspy/" aria-label="Open eyeprocesspy on PyPI">PyPI ↗</a>
</div>

<div class="ep-release-links">
<a href="https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.1.0">GitHub Release</a>
<span>·</span>
<a href="https://doi.org/10.5281/zenodo.22285167">DOI 10.5281/zenodo.22285167</a>
</div>

</div>

<div class="ep-home-hero__visual">
<div class="ep-visual-topline"><span class="ep-live-dot"></span> validated process surface</div>
<div class="ep-visual-main">
<div class="ep-visual-label"><span>GAZE</span><span>canonical trace</span></div>
<img src="assets/gallery/gaze-trace.svg" alt="Example canonical gaze trace produced by eyeprocesspy">
</div>
<div class="ep-visual-mini-grid">
<div class="ep-visual-mini">
<div class="ep-visual-label"><span>PUPIL</span><span>time series</span></div>
<img src="assets/gallery/pupil-timeseries.svg" alt="Example pupil time series produced by eyeprocesspy">
</div>
<div class="ep-visual-mini">
<div class="ep-visual-label"><span>AOI</span><span>uncertainty</span></div>
<img src="assets/gallery/probabilistic-aoi.svg" alt="Example probabilistic AOI visualization produced by eyeprocesspy">
</div>
</div>
</div>
</section>

<div class="ep-proof-rail">
<div><strong>1,182 / 1,182</strong><span>frozen APIs resolved</span></div>
<div><strong>1,458</strong><span>release tests passed</span></div>
<div><strong>100% + 100%</strong><span>statements + branches</span></div>
<div><strong>12 lanes</strong><span>3 OS × Python 3.11–3.14</span></div>
<div><strong>88 / 88</strong><span>workflow articles linked</span></div>
</div>

<section class="ep-section ep-section--intro" markdown>
<div class="ep-section-head" markdown>
<div class="ep-section-kicker">One research surface</div>
## Everything between the export and the evidence
<p>Instead of stitching together disconnected scripts, use one governed data model across ingestion, measurement, modeling, validation, and provenance.</p>
</div>

<div class="ep-capability-grid">
<a class="ep-capability" href="guides/end-to-end-eye-tracking/">
<span class="ep-capability__num">01</span>
<strong>Import & harmonize</strong>
<p>Vendor and generic readers, canonical schemas, coordinates, events, intervals, AOIs, responses, quality, and provenance.</p>
<span class="ep-capability__link">Eye-tracking workflow →</span>
</a>
<a class="ep-capability" href="guides/process-quality-uncertainty/">
<span class="ep-capability__num">02</span>
<strong>Quality before metrics</strong>
<p>Sampling integrity, calibration uncertainty, missingness, screen bounds, validity, anomaly checks, and governed preflight.</p>
<span class="ep-capability__link">Quality & uncertainty →</span>
</a>
<a class="ep-capability" href="examples/core-workflow/">
<span class="ep-capability__num">03</span>
<strong>Gaze & AOI process</strong>
<p>Fixations, saccades, dwell, scanpaths, transitions, entropy, recurrence, context, and uncertainty-aware AOI assignment.</p>
<span class="ep-capability__link">Core workflow →</span>
</a>
<a class="ep-capability" href="guides/pupillometry/">
<span class="ep-capability__num">04</span>
<strong>Pupil & multimodal data</strong>
<p>Eye-specific pupil streams, baseline correction, functional pupil representations, synchronized channels, and process features.</p>
<span class="ep-capability__link">Pupillometry guide →</span>
</a>
<a class="ep-capability" href="guides/psychometrics-irt/">
<span class="ep-capability__num">05</span>
<strong>Psychometrics & IRT</strong>
<p>IRT, process-informed measurement, DIF/DTF, reliability, linking, diagnostics, uncertainty, and advanced process models.</p>
<span class="ep-capability__link">Psychometrics & IRT →</span>
</a>
<a class="ep-capability" href="parity-and-validation/">
<span class="ep-capability__num">06</span>
<strong>Validate & reproduce</strong>
<p>Recovery, stress tests, negative controls, leakage-aware validation, evidence atlases, frozen-R checks, and release audits.</p>
<span class="ep-capability__link">Validation evidence →</span>
</a>
</div>
</section>

<section class="ep-section ep-section--pipeline" markdown>
<div class="ep-section-head ep-section-head--light" markdown>
<div class="ep-section-kicker">A measurement pipeline, not a bag of functions</div>
## Keep the scientific chain intact
<p>Every stage is designed to preserve the information needed to inspect the next one.</p>
</div>

<div class="ep-pipeline">
<div class="ep-pipeline-step"><span>01</span><strong>INGEST</strong><p>Read exports and normalize schema, coordinates, time, events, AOIs, and metadata.</p></div>
<div class="ep-pipeline-step"><span>02</span><strong>MEASURE</strong><p>Recover gaze, pupil, spatial, temporal, sequence, and multimodal process features.</p></div>
<div class="ep-pipeline-step"><span>03</span><strong>MODEL</strong><p>Fit reliability, psychometric, IRT, process, uncertainty, and diagnostic models.</p></div>
<div class="ep-pipeline-step"><span>04</span><strong>VERIFY</strong><p>Attach quality checks, negative controls, recovery evidence, provenance, and release state.</p></div>
</div>
</section>

<section class="ep-section" markdown>
<div class="ep-section-head" markdown>
<div class="ep-section-kicker">See the process</div>
## Figures that expose what the pipeline is doing
<p>The visual layer is part of the audit trail: inspect trajectories, uncertainty, reliability, temporal behavior, and model diagnostics rather than relying on opaque summary tables.</p>
</div>

<div class="ep-showcase">
<a class="ep-showcase-card ep-showcase-card--wide" href="gallery/">
<div class="ep-showcase-meta"><span>01 · SPATIAL PROCESS</span><strong>Scanpaths & trajectories</strong></div>
<img src="assets/gallery/scanpath.svg" alt="Scanpath visualization generated by eyeprocesspy">
</a>
<a class="ep-showcase-card" href="gallery/">
<div class="ep-showcase-meta"><span>02 · UNCERTAINTY</span><strong>Probabilistic AOIs</strong></div>
<img src="assets/gallery/probabilistic-aoi.svg" alt="Probabilistic AOI visualization generated by eyeprocesspy">
</a>
<a class="ep-showcase-card" href="gallery/">
<div class="ep-showcase-meta"><span>03 · RELIABILITY</span><strong>Agreement evidence</strong></div>
<img src="assets/gallery/process-reliability.svg" alt="Process reliability visualization generated by eyeprocesspy">
</a>
</div>
<div class="ep-inline-link"><a href="gallery/">Explore the complete figure gallery →</a></div>
</section>

<section class="ep-section ep-trust" markdown>
<div class="ep-trust-copy" markdown>
<div class="ep-section-kicker">Validation is a first-class output</div>
## Evidence, not a confidence badge

The published `0.1.0` release was frozen only after the deep-parity gate reached **23,085 / 23,085 statements** and **9,680 / 9,680 branches**, with **1,458 tests** passing. The release matrix covers Ubuntu, macOS, and Windows across Python 3.11–3.14 and retains the frozen R 0.11.1 oracle as a scientific reference.

<div class="ep-trust-actions">
<a href="https://github.com/stefanosbalaskas/eyeprocesspy/blob/main/RELEASE_VALIDATION.md">Release validation ↗</a>
<a href="release-and-reproducibility/">Reproducibility guide →</a>
</div>
</div>

<div class="ep-trust-proof">
<div><span>API parity</span><strong>1,182 / 1,182</strong><em>resolved</em></div>
<div><span>Statement coverage</span><strong>100.000%</strong><em>23,085 / 23,085</em></div>
<div><span>Branch coverage</span><strong>100.000%</strong><em>9,680 / 9,680</em></div>
<div><span>Release archive</span><strong>Zenodo DOI</strong><em>10.5281/zenodo.22285167</em></div>
</div>
</section>

<section class="ep-section ep-start" markdown>
<div class="ep-start-copy" markdown>
<div class="ep-section-kicker">Start with a known-good surface</div>
## Install, validate, then analyze
<p>The benchmark study ships with the package so you can verify the installation and the workflow before touching study data.</p>

<div class="ep-start-links">
<a href="getting-started/">Getting started →</a>
<a href="examples/">Worked examples →</a>
<a href="cookbook/">Cookbook →</a>
</div>
</div>

<div class="ep-code-card" markdown>
<div class="ep-code-card__title"><span>quickstart.py</span><span>v0.1.0</span></div>

```python
import eyeprocesspy as ep

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
data = ep.import_benchmark_study(study)

assert audit["valid"]
print(data)
```

</div>
</section>

<section class="ep-section ep-release" markdown>
<div class="ep-release-copy" markdown>
<div class="ep-section-kicker">Published software</div>
## Reproducible by version, release, and DOI
<p>Install the current release from PyPI, pin the exact version for a study, and cite the archived Zenodo record.</p>
</div>
<div class="ep-release-install">
<span>stable</span>
<code>pip install eyeprocesspy==0.1.0</code>
</div>
<div class="ep-release-buttons">
<a href="https://pypi.org/project/eyeprocesspy/">PyPI ↗</a>
<a href="https://github.com/stefanosbalaskas/eyeprocesspy/releases/tag/v0.1.0">GitHub Release ↗</a>
<a href="https://doi.org/10.5281/zenodo.22285167">Zenodo ↗</a>
</div>
</section>

<section class="ep-citation" markdown>
<div>
<span class="ep-citation__label">CITE</span>
<p><strong>Balaskas, S. (2026).</strong> <em>eyeprocesspy: Vendor-neutral Python infrastructure for eye-tracking and multimodal process data</em> (Version 0.1.0) [Computer software]. Zenodo.</p>
</div>
<a href="https://doi.org/10.5281/zenodo.22285167">10.5281/zenodo.22285167 ↗</a>
</section>

<div class="ep-boundary"><strong>Interpretation boundary.</strong> `eyeprocesspy` provides measurement and analysis infrastructure. Reliability is not construct validity, prediction is not causation, probabilistic AOI membership represents modeled coordinate uncertainty rather than probability of attention, and gaze/pupil/process metrics should not be interpreted as psychological states without external validity evidence and an appropriate study design.</div>

</div>
