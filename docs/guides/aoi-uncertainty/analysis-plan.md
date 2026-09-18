# AOI sensitivity analysis plan

Use this template **before inspecting branch-specific results**. Its purpose is to make the perturbation envelope, denominator rules, and statistical model explicit before robustness results can influence those choices.

## Minimum plan

Record the following items:

| Decision | Required specification |
| --- | --- |
| Nominal AOIs | source/version, rectangle or polygon representation, coordinate space |
| Observation level | `fixation` or `sample` |
| Perturbation grid | operations, exact values, and branch IDs |
| Units | pixels or degrees of visual angle |
| Display geometry | resolution, physical size, viewing distance when degrees are used |
| Overlap policy | usually `ambiguous` unless another rule is substantively justified |
| Screen-boundary policy | `warn`, `clip`, `error`, or `allow` |
| Randomness | jitter seed if random perturbations are used |
| Feature recomputation | dwell/count/first observation/inspection definitions |
| Missingness | zero-versus-missing rules and valid-observation denominator |
| Model callback | estimator, family/link, covariates, random effects, contrasts |
| Model validity | convergence and finite-output requirements |
| Failure handling | how failed geometry/callback branches remain in the audit |
| Reporting | assignment, coefficient, interval, convergence, and model-`N` summaries |

## Machine-readable template

A copyable YAML template is available at
[`aoi-analysis-plan-template.yml`](../../assets/aoi-uncertainty/aoi-analysis-plan-template.yml).

```yaml
analysis_id: aoi-sensitivity-primary
observation_level: fixation
perturbations:
  unit: deg
  dilation: [0.25, 0.50, 1.00]
  erosion: [0.25]
  translation_x: [0.50]
  translation_y: [0.50]
overlap_policy: ambiguous
boundary_policy: allow
model:
  estimator: prespecified-callback
  convergence_required: true
report:
  assignment_stability: true
  model_N_range: true
  failed_branches: true
```

The package does not read this file automatically; it is a reproducibility aid. The executable analysis should still declare the same settings in code.

## Translate the plan into the API

```python
grid = ep.create_aoi_perturbation_grid(
    dilations=[0.25, 0.50, 1.00],
    erosions=[0.25],
    translations_x=[0.50],
    translations_y=[0.50],
    unit="deg",
    screen_width_px=1920,
    screen_height_px=1080,
    viewing_distance=60,
    physical_screen_size=(53.1, 29.9),
    boundary_policy="allow",
)

result = ep.run_aoi_sensitivity_analysis(
    fixations,
    aois,
    grid,
    x_col="x",
    y_col="y",
    observation_id_col="obs",
    participant_col="participant",
    trial_col="trial",
    duration_col="duration",
    time_col="time",
    observation_level="fixation",
    overlap_policy="ambiguous",
    model_callback=prespecified_model,
)
```

## Rules for amendments

If the plan changes after results are available, preserve both versions and state:

1. what changed;
2. why it changed;
3. whether the change was motivated by a scientific/design issue or by the observed result;
4. which conclusions depend on the amended specification.

Do not silently remove a failed branch, alter the perturbation envelope, or change the estimator after seeing which branch gives the preferred result.

## Interpretation rule

The analysis plan defines the **scope of the robustness claim**. A high unchanged-assignment or same-sign frequency means stability within that declared scope only. It is not a probability that the AOI geometry or scientific conclusion is true.

Continue to the [worked filled plan](../../examples/aoi-analysis-plan-worked.md), [decision clinic](decision-clinic.md), [troubleshooting guide](troubleshooting.md), or [API reference](../../reference/aoi-perturbation.md).
