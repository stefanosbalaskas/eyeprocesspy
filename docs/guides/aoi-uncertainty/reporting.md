# Reporting AOI Robustness

A reproducible report states what was perturbed, why those changes were defensible, and which assignment/model features remained stable.

## Minimum methods information

Report the nominal AOI source, geometry representation, coordinate system, perturbation operations and values, pixel/degree units, screen/viewing geometry when needed, overlap policy, screen-edge policy, jitter seed, assignment level, recomputed features, model specification, and failure handling.

## Minimum results information

Report assignment stability and reassignment patterns together with coefficient direction, magnitude, intervals, and convergence. Include failed branches rather than silently changing the denominator.

## Suggested language

> We reran AOI assignment, feature extraction, and the prespecified model across the nominal geometry and defensible perturbations. We report reassignment patterns, coefficient ranges and intervals, and convergence across branches. Stability frequencies are interpreted as descriptive robustness to this perturbation set, not probabilities that the scientific conclusion is true.

## Avoid

Do not describe a 90% same-sign frequency as a 90% probability that the finding is true, and do not select the AOI branch that maximizes statistical significance.

Use `report_aoi_sensitivity()` for a compact starting report and the [worked example](../../examples/aoi-perturbation-sensitivity.md) for an end-to-end pattern.