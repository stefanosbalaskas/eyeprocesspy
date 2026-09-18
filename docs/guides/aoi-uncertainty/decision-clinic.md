# AOI robustness decision clinic

This page helps analysts choose a defensible perturbation envelope, diagnose unstable branches, and report robustness without turning sensitivity frequencies into probabilities.

## Before running the grid

1. Confirm coordinate units and screen geometry.
2. State whether rows are fixations or samples.
3. Freeze the nominal AOIs and statistical model.
4. Choose perturbation magnitudes from spatial accuracy, display geometry, stimulus layout, or a preregistered design rationale.
5. Decide overlap and screen-edge behavior before looking at results.

## Diagnostic patterns

| Pattern | Interpretation | Action |
| --- | --- | --- |
| High assignment stability + stable coefficient | Conclusion is insensitive within the declared envelope | Report the envelope and stability summaries; do not call it proof of correctness |
| Low assignment stability + stable coefficient | Feature mapping changes but the model-level conclusion is insensitive | Report both layers and inspect reassignment matrices |
| High assignment stability + unstable coefficient | A small set of observations/groups may be influential | Inspect group/AOI summaries, model N, intervals, and influential design cells |
| Low assignment stability + unstable coefficient | Conclusion depends materially on geometry | Treat the nominal result as geometry-sensitive and avoid a single-branch robustness claim |
| Failed geometry/model branches | The sensitivity design is partly non-evaluable | Retain failed branches, explain why they failed, and do not shrink the denominator silently |

## Choosing perturbation magnitudes

Use the smallest set that represents scientifically plausible uncertainty. Prefer degrees of visual angle when viewing geometry is known and cross-display comparability matters; use pixels when the design itself is pixel-defined or physical geometry is unavailable. Avoid result-driven searches for a margin that restores significance.

## Worked interpretation

Suppose seven planned branches all complete, unchanged assignment ranges from 0.956 to 1.000, the target coefficient stays positive, confidence intervals overlap, all models converge, and model N is constant. The appropriate interpretation is that the result is stable within the declared geometry envelope. It is not a 95.6–100% probability that the AOIs or conclusion are true.

## Reporting checklist

Report the nominal AOI source, observation level, perturbation operations and units, screen/viewing geometry, overlap and boundary policies, seed for random perturbations, assignment denominators, feature definitions, model callback/estimator, failed branches, convergence, model N range, coefficient range/intervals, and assignment stability.

## API links

- [`create_aoi_perturbation_grid()`](../../reference/aoi-perturbation.md) — declare the sensitivity set.
- [`run_aoi_sensitivity_analysis()`](../../reference/aoi-perturbation.md) — rerun assignment, features, and optional models.
- [`estimate_aoi_assignment_stability()`](../../reference/aoi-perturbation.md) — inspect assignment robustness.
- [`assess_aoi_inference_stability()`](../../reference/aoi-perturbation.md) — summarize coefficient, convergence, and model-N stability.
- [`report_aoi_sensitivity()`](../../reference/aoi-perturbation.md) — generate a compact reporting draft.
