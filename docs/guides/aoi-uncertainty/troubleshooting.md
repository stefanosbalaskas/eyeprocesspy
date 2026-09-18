# AOI perturbation troubleshooting

Use this guide when a sensitivity run contains unexpected ambiguity, failed branches, missing feature cells, non-convergence, or changing model sample sizes. Diagnose the earliest failing layer first: **geometry → assignment → feature recomputation → model callback**.

## Triage table

| Symptom | Likely layer | Inspect | Do not do |
| --- | --- | --- | --- |
| Geometry branch is `failed` | Geometry | perturbation audit, AOI size, boundary policy | Delete the branch after seeing results |
| Many `__ambiguous__` assignments | Geometry/assignment | overlap audit, reassignment matrix, geometry plot | Silently choose the first AOI |
| Many `__outside__` assignments | Assignment/layout | units, screen transform, AOI position | Recode outside as zero exposure without justification |
| Count is `0` | Feature recomputation | `n_valid_observations` and AOI level | Treat zero as missing |
| Count is missing | Missingness | coordinates and `n_missing_observations` | Coerce missing to zero |
| Callback appears in `failures` | Model | callback exception and required columns | Treat the branch as a valid null fit |
| `model_converged=False` | Model | estimator diagnostics and model `N` | Reclassify it as converged |
| Model `N` changes | Model/data | assignment loss and complete-case rules | Report only coefficient direction |
| Baseline unavailable | Geometry/design | nominal AOI validity | Use a perturbed branch as the reference silently |
| Degree conversion fails | Units | display geometry/viewing distance | Guess viewing geometry |

## Recommended debugging order

1. Validate nominal AOIs with `validate_aoi_geometry()`.
2. Inspect `result["grid_result"]["audit"]`.
3. Plot the problematic geometry.
4. Inspect reassignment matrices and assignment stability.
5. Audit zero-versus-missing cells and observation level.
6. Inspect callback failures, convergence, and model `N`.
7. Only then interpret coefficient stability.

## Geometry failures

A failed perturbation remains in the planned audit denominator.

```python
audit = result["grid_result"]["audit"]
print(audit[["perturbation_id", "status", "message"]])
```

If a perturbation is scientifically implausible, revise and document the **analysis plan**; do not remove a failed branch simply because it is inconvenient for the result.

## Assignment ambiguity

The default overlap behavior is explicit `__ambiguous__`.

```python
ax = ep.plot_aoi_perturbations(
    result,
    perturbation_id="dilate_1_px",
    data=observations,
    x_col="x",
    y_col="y",
)
```

A new ambiguous assignment usually means the perturbation made AOIs overlap at that observation. It does not authorize arbitrary AOI selection.

## Zero versus missing

- `observation_count = 0`: valid observations existed but none entered that AOI.
- Missing count: no valid assignment opportunity existed.
- `fixation_count` and `sample_count` populate only at the matching `observation_level`.

Use `n_valid_observations` and `n_missing_observations` to audit denominators.

## Model failures

Callback exceptions remain in `result["failures"]` with `stage="model"`. A returned `model_converged=False` row remains visible but is excluded from converged-effect summaries.

```python
print(result["failures"])
print(result["models"][["perturbation_id", "term", "model_converged", "N"]])
```

Converged callback rows require finite estimates, standard errors, confidence limits, and positive integer-valued `N`.

## Reporting problematic branches

> Of four planned AOI branches, three produced valid geometries and one erosion collapsed an AOI. The failed branch remained in the perturbation audit. Among completed branches, one model callback failed and one returned a non-converged fit; these were retained in the audit trail and excluded from converged-effect summaries. Assignment ambiguity was reported explicitly rather than resolved by AOI order.

This is a reporting template, not an empirical result.

## Limitations

Troubleshooting identifies where a branch became non-evaluable; it cannot establish the substantively correct AOI. Geometry sensitivity also does not replace calibration checks, event-detector sensitivity, or estimator-specific diagnostics.

Run the [worked failure clinic](../../examples/aoi-perturbation-failure-clinic.md), then return to the [decision clinic](decision-clinic.md) or [API reference](../../reference/aoi-perturbation.md).
