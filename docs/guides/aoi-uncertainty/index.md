# AOI perturbation and uncertainty analysis

AOI boundaries are an **analytical specification**, not an error-free property of a stimulus. This workflow asks whether a substantive result changes under small, defensible changes to AOI geometry.

> **Core question:** Does the conclusion depend materially on plausible AOI boundary choices?

## When to use

Use AOI perturbation analysis when gaze or fixation coordinates cluster near AOI borders, AOIs are small relative to spatial error, adjacent AOIs are close, multiple boundary codings are defensible, or conclusions depend strongly on dwell, counts, first fixation, or inspection.

Do **not** use it to search for a favorable AOI, replace calibration/coordinate-quality checks, or rescue an implausible layout. Define the perturbation envelope from measurement and design knowledge rather than from whichever branch produces the preferred result.

## Visual orientation

![AOI perturbation geometry and reassigned fixations](../../assets/aoi-uncertainty/geometry-perturbation.svg)

The full [worked example](../../examples/aoi-perturbation-sensitivity.md) also shows assignment stability, coefficient trajectories, and the two-dimensional robustness surface.

## Four diagnostics, four questions

<div class="ep-gallery" markdown>

<figure>
  <img src="../../assets/aoi-uncertainty/geometry-perturbation.svg" alt="AOI geometry perturbation example">
  <figcaption><strong>1 · Geometry.</strong> Which observations cross a decision boundary when the AOI changes?</figcaption>
</figure>

<figure>
  <img src="../../assets/aoi-uncertainty/assignment-stability.svg" alt="AOI assignment stability example">
  <figcaption><strong>2 · Assignment.</strong> How much of the measurement mapping changes?</figcaption>
</figure>

<figure>
  <img src="../../assets/aoi-uncertainty/coefficient-stability.svg" alt="AOI coefficient stability example">
  <figcaption><strong>3 · Model.</strong> Does the same prespecified model tell a materially different story?</figcaption>
</figure>

<figure>
  <img src="../../assets/aoi-uncertainty/robustness-surface.svg" alt="AOI robustness surface example">
  <figcaption><strong>4 · Surface.</strong> Where does sensitivity concentrate when geometry changes jointly along two axes?</figcaption>
</figure>

</div>

Use these plots as diagnostics, not as decoration. A visually stable coefficient does not erase assignment instability, changing model N, failed branches, or an implausible perturbation plan.

## Method guides

- [AOI Geometry Is an Analytical Assumption](geometry-assumption.md)
- [AOI Perturbation Sensitivity Analysis](perturbation-sensitivity.md)
- [Pixels versus Degrees of Visual Angle](pixels-versus-degrees.md)
- [Interpreting AOI Assignment Stability](assignment-stability.md)
- [Propagating AOI Uncertainty Into Statistical Models](model-propagation.md)
- [AOI Robustness Decision Clinic](decision-clinic.md)
- [AOI Sensitivity Analysis Plan](analysis-plan.md)
- [AOI Perturbation Troubleshooting](troubleshooting.md)
- [Reporting AOI Robustness](reporting.md)
- [AOI Robustness Reporting Bundle](reporting-bundle.md)
- [R/Python Scientific Parity](r-python-parity.md)

## Scientific pipeline

```text
canonical AOIs
→ declared perturbation grid
→ remap samples/fixations
→ recompute features
→ compare assignments
→ optionally refit the same model
→ quantify stability and failures
→ visualize and report
```

## Geometry operations

The public API supports dilation, erosion, x/y translation, seeded uniform jitter, anisotropic expansion, pixel margins, degree-of-visual-angle margins, rectangular AOIs, and polygonal AOIs.

Convex polygons support true geometric dilation/erosion. Concave polygon buffering is rejected rather than silently convexified or approximated. Translation and anisotropic expansion remain available for polygons.

```python
grid = ep.create_aoi_perturbation_grid(
    dilations=[0.25, 0.50, 1.00],
    erosions=[0.25],
    translations_x=[0.50],
    translations_y=[0.50],
    unit="deg",
    screen_width_px=1024,
    screen_height_px=768,
    viewing_distance=60,
    physical_screen_size=(53.1, 29.9),
    include_baseline=True,
)
```

### Screen-edge behavior

Screen boundaries are never handled silently. Choose `warn`, `clip`, `error`, or `allow`. A perturbation that collapses geometry is recorded as a failed branch.

### Random jitter

Jitter uses a reproducible seed stored with the perturbation specification. The same geometry and seed reproduce the same perturbation.

## Pixels versus degrees of visual angle

Pixel values are display coordinates; degrees of visual angle depend on screen geometry and viewing distance. Degree-based perturbations therefore require either explicit degrees-per-pixel values or screen width/height in pixels, physical screen dimensions, and viewing distance.

```python
deg = ep.convert_aoi_margin_to_degrees(
    (20, 20),
    screen_width_px=1920,
    screen_height_px=1080,
    viewing_distance=60,
    physical_screen_size=(53.1, 29.9),
)
```

Horizontal and vertical degrees-per-pixel are preserved separately. The workflow never silently changes units.

## Assignment stability

For every branch, the nominal and perturbed assignments are compared. The result includes:

- proportion unchanged;
- proportion newly assigned;
- proportion lost;
- proportion reassigned between AOIs;
- AOI-to-AOI reassignment matrices;
- AOI-level stability;
- optional participant-level stability;
- optional trial-level stability.

Missing coordinates remain missing and are excluded from comparable-assignment denominators. They are never converted to zero or outside-AOI observations.

Overlapping membership defaults to `__ambiguous__`. Analysts can explicitly request all memberships or an error, but the package does not silently choose the first AOI.

`estimate_fixation_assignment_probability()` returns an empirical assignment **frequency** across the declared perturbation set. It is not a posterior probability of true AOI membership.

## Recompute features

`recompute_aoi_features()` preserves the complete declared AOI universe within observed participant/trial groups. Set `observation_level="fixation"` or `"sample"`: universal observation counts/timing are retained, while fixation- and sample-specific fields are populated only at the matching level. Structural zero cells are distinguished from groups with no valid assignment opportunity.

## Propagate geometry uncertainty into models

The package does not choose a statistical estimator. Supply an explicit model callback. Each successful callback must return at least:

```text
term
estimate
SE
CI_low
CI_high
p_value
model_converged
N
```

The workflow adds `perturbation_id` and coefficient direction. Callback failures are written to the failure table. Rows with `model_converged=False` remain visible and are not treated as valid converged estimates.

Prefer summaries of coefficient direction, effect magnitude, confidence-interval width, convergence, and assignment stability. Do not reduce robustness to significant/not-significant counts.

## Interpretation

A high same-sign frequency or high unchanged-assignment proportion means the result was stable **within the declared perturbation set**. It does not estimate the probability that the scientific conclusion is true.

Low stability can indicate edge-heavy fixation clouds, neighboring AOIs that are too close, or unusually influential nominal geometry. It does not identify a single scientifically correct AOI.

## Reporting guidance

Report the nominal AOI source, coordinate system, perturbation operations/values, pixel-versus-degree units, screen/viewing geometry when needed, overlap policy, screen-boundary policy, jitter seed, assignment level, recomputed features, model specification, and handling of failed/non-converged branches.

A compact reporting pattern is:

> We evaluated AOI-boundary sensitivity by rerunning assignment, feature extraction, and the prespecified model across the nominal geometry and defensible perturbations. We report unchanged/reassigned observations, coefficient ranges and uncertainty intervals, and model convergence across branches. Stability frequencies are interpreted as descriptive robustness to this perturbation set, not probabilities that the scientific conclusion is true.

Use `report_aoi_sensitivity()` to generate a compact Markdown summary, then edit it to match the exact study design.

## Limitations

- Perturbation results are conditional on the chosen perturbation envelope.
- Geometry sensitivity does not replace calibration-error or event-detector sensitivity analysis.
- Exact concave polygon buffering requires a dedicated geometry engine.
- Degree conversion is only as accurate as the physical screen and viewing-distance information supplied.
- Correlated analytical choices should be investigated with an explicit multiverse rather than interpreted as independent replications.

## Next steps

Start with the [worked filled plan](../../examples/aoi-analysis-plan-worked.md), then run the [worked synthetic advertising example](../../examples/aoi-perturbation-sensitivity.md), use the [worked failure clinic](../../examples/aoi-perturbation-failure-clinic.md) to inspect non-evaluable branches, package the evidence with the [worked reporting bundle](../../examples/aoi-reporting-bundle.md), then use the focused [API reference](../../reference/aoi-perturbation.md).