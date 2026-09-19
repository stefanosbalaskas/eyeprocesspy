# Visual gallery

The previews on this page summarize deterministic `eyeprocesspy 0.1.0` example outputs. Run `python examples/core_gallery.py` and `python examples/advanced_gallery.py` to generate the full Matplotlib figures directly through the package plotting API. The scripts were validated against the CI-built wheel and use no private participant data.

## Gaze, fixation and AOI workflows

<div class="ep-gallery" markdown>

<figure>
  <img src="../assets/gallery/gaze-trace.svg" alt="Gaze trace">
  <figcaption><strong>Gaze trace.</strong> Ordered valid gaze samples in canonical coordinates.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/fixations.svg" alt="Fixation plot">
  <figcaption><strong>Fixations.</strong> Duration-scaled fixation centroids.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/scanpath.svg" alt="Scanpath plot">
  <figcaption><strong>Scanpath.</strong> Ordered AOI visits with spatial trajectory.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/gaze-heatmap.svg" alt="Gaze density heatmap">
  <figcaption><strong>Gaze density.</strong> Two-dimensional histogram over gaze coordinates.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/aoi-dwell.svg" alt="AOI dwell plot">
  <figcaption><strong>AOI dwell.</strong> AOI-level dwell features carried in the canonical feature table.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/transition-matrix.svg" alt="AOI transition matrix">
  <figcaption><strong>Transition matrix.</strong> Row-normalized transitions derived from scanpath episodes.</figcaption>
</figure>

</div>

## AOI uncertainty and robustness

These diagnostics come from the synthetic AOI perturbation workflow. Read them together: geometry explains **what changed**, assignment stability quantifies **how much changed**, coefficient stability shows **whether inference moved**, and the robustness surface shows **where sensitivity concentrates across two-dimensional boundary changes**.

<div class="ep-gallery" markdown>

<figure>
  <img src="../assets/aoi-uncertainty/geometry-perturbation.svg" alt="Nominal and perturbed AOI geometry with reassigned fixations">
  <figcaption><strong>Geometry perturbation.</strong> Compare nominal and perturbed boundaries and inspect observations that cross an AOI decision boundary.</figcaption>
</figure>

<figure>
  <img src="../assets/aoi-uncertainty/assignment-stability.svg" alt="AOI assignment stability across perturbation branches">
  <figcaption><strong>Assignment stability.</strong> Summarize unchanged, newly assigned, lost, and reassigned observations across declared branches.</figcaption>
</figure>

<figure>
  <img src="../assets/aoi-uncertainty/coefficient-stability.svg" alt="Coefficient stability across AOI perturbation branches">
  <figcaption><strong>Coefficient stability.</strong> Inspect effect magnitude and intervals across evaluable branches instead of reducing robustness to significance counts.</figcaption>
</figure>

<figure>
  <img src="../assets/aoi-uncertainty/robustness-surface.svg" alt="Two-dimensional AOI robustness surface">
  <figcaption><strong>Robustness surface.</strong> Locate regions of higher or lower stability when horizontal and vertical geometry changes are varied jointly.</figcaption>
</figure>

</div>

[Open the AOI uncertainty workflow →](guides/aoi-uncertainty/)

## Time-to-event gaze workflows

<div class="ep-gallery" markdown>

<figure>
  <img src="../assets/gaze-survival/km-disclosure.svg" alt="Kaplan–Meier gaze-latency curves">
  <figcaption><strong>Censored gaze latency.</strong> Kaplan–Meier curves retain valid trials that end before the target AOI is inspected.</figcaption>
</figure>

</div>

[Open the survival-analysis method →](methods/gaze-survival/index.md)

## Pupil and data-quality workflows

<div class="ep-gallery" markdown>

<figure>
  <img src="../assets/gallery/pupil-timeseries.svg" alt="Pupil time series">
  <figcaption><strong>Pupil time series.</strong> Eye-specific pupil streams on a shared timebase.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/sampling-irregularity.svg" alt="Sampling irregularity audit">
  <figcaption><strong>Sampling irregularity.</strong> Effective sampling diagnostics with an explicit review threshold.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/calibration-error.svg" alt="Calibration error model">
  <figcaption><strong>Calibration error.</strong> Empirical horizontal/vertical error cloud used to propagate gaze uncertainty.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/probabilistic-aoi.svg" alt="Probabilistic AOI assignment">
  <figcaption><strong>Probabilistic AOIs.</strong> Membership probabilities after calibration uncertainty is propagated.</figcaption>
</figure>

</div>

## Psychometrics and measurement

<div class="ep-gallery" markdown>

<figure>
  <img src="../assets/gallery/process-reliability.svg" alt="Process reliability Bland Altman plot">
  <figcaption><strong>Process reliability.</strong> Bland–Altman diagnostics for repeated process measures.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/irt-information.svg" alt="IRT information curve">
  <figcaption><strong>IRT information.</strong> Test information across the latent continuum.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/irt-item-fit.svg" alt="IRT item fit">
  <figcaption><strong>IRT item fit.</strong> Item-level fit diagnostics against the reference line.</figcaption>
</figure>

<figure>
  <img src="../assets/gallery/irt-dif.svg" alt="IRT differential item functioning curve">
  <figcaption><strong>DIF curve.</strong> Signed focal-reference probability difference across theta.</figcaption>
</figure>

</div>

## Dataset overview

![Canonical eyeprocess dataset overview](assets/gallery/eye-overview.svg)

The plotting functions retain the underlying plot data on the Matplotlib axes where relevant (`ax.eyeprocess_plot_data`, and for matrix plots `ax.eyeprocess_plot_matrix`). This makes the visual output auditable and allows publication styling without losing the numerical payload.

!!! note "Scientific interpretation"
    The gallery demonstrates computational capabilities. A gaze, pupil, reliability, or psychometric statistic is not automatically a validated psychological construct. Use the package's evidence, provenance, and guardrail layers alongside an appropriate study design.
