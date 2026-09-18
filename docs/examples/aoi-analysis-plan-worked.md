# Worked example: filling an AOI sensitivity analysis plan

This synthetic example shows how to translate a study design into a fixed AOI sensitivity plan **before** inspecting perturbation-specific results.

## Study setup

Assume a five-region digital-advertising stimulus with `headline`, `image`, `claim`, `disclosure`, and `cta` AOIs. Fixation centroids are analyzed on a 1920 × 1080 display with known physical size and a controlled 60 cm viewing distance.

The primary outcome is disclosure dwell. The primary model is fixed before the AOI robustness analysis.

## Filled plan

```yaml
analysis_id: disclosure-dwell-aoi-robustness
nominal_geometry:
  source: stimulus-layout-v3
  representation: rectangle
  coordinate_space: pixels
observation_level: fixation
perturbations:
  unit: deg
  dilation: [0.25, 0.50, 1.00]
  erosion: [0.25]
  translation_x: [0.50]
  translation_y: [0.50]
overlap_policy: ambiguous
boundary_policy: allow
features:
  duration_column: duration
  time_column: time
model:
  estimator: fixed-disclosure-dwell-callback
  convergence_required: true
failure_handling:
  retain_geometry_failures: true
  retain_callback_failures: true
  retain_nonconverged_rows: true
report:
  assignment_stability: true
  coefficient_range: true
  confidence_intervals: true
  convergence: true
  model_N_range: true
```

## Why these choices are defensible

The perturbation values are defined in degrees because viewing geometry is known and the intended uncertainty is spatial rather than tied to one display resolution. The nominal AOIs remain the reference. Overlap defaults to explicit ambiguity, so newly overlapping regions are not resolved by AOI order.

The same statistical model is run in every evaluable branch. Failed geometry and callback branches remain in the audit denominator.

## API mapping

| Plan field | API location |
| --- | --- |
| perturbations | `create_aoi_perturbation_grid()` |
| units/display geometry | `create_aoi_perturbation_grid()` |
| overlap policy | `run_aoi_sensitivity_analysis()` |
| observation level | `run_aoi_sensitivity_analysis()` |
| feature columns | `run_aoi_sensitivity_analysis()` |
| model callback | `run_aoi_sensitivity_analysis()` |
| assignment summaries | `estimate_aoi_assignment_stability()` |
| inference summaries | `assess_aoi_inference_stability()` |
| compact report | `report_aoi_sensitivity()` |

## Planned interpretation language

Before seeing the results, write the interpretation rule:

> We will describe the result as stable within the declared AOI perturbation envelope when assignment changes, coefficient magnitude/direction, uncertainty intervals, convergence, and model N do not indicate a material dependence on a small subset of defensible boundary choices. Robustness frequencies will be interpreted descriptively and not as probabilities that the scientific conclusion is true.

## Example result patterns

| Observed pattern | Planned interpretation |
| --- | --- |
| High assignment stability + stable estimate/interval/N | stable within declared envelope |
| Low assignment stability + stable estimate/interval/N | mapping-sensitive but model-level conclusion stable |
| High assignment stability + unstable estimate or N | inspect influential groups/case loss before concluding robustness |
| Low assignment stability + unstable estimate | materially geometry-sensitive |
| Failed/non-converged branches | partially non-evaluable; keep failures visible |

## Reporting example

> The AOI perturbation plan was fixed before branch-specific results were inspected. The nominal geometry was evaluated together with prespecified dilation, erosion, and translation branches in degrees of visual angle. We report assignment changes, coefficient ranges and confidence intervals, convergence, model-N variation, and all failed branches. The sensitivity summary is interpreted as robustness to this declared set of AOI alternatives rather than as a probability that the nominal AOIs are correct.

This is a synthetic reporting template, not an empirical finding.

Continue to the [full analysis-plan template](../guides/aoi-uncertainty/analysis-plan.md), [worked sensitivity example](aoi-perturbation-sensitivity.md), [failure clinic](aoi-perturbation-failure-clinic.md), and [API reference](../reference/aoi-perturbation.md).
