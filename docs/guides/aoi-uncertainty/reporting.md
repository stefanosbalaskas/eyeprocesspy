# Reporting AOI Robustness

A reproducible report states **what was perturbed, why those changes were defensible, which rows were analyzed, and which assignment/model quantities remained stable**.

## Before results are inspected

Use the [AOI sensitivity analysis plan](analysis-plan.md) to freeze the perturbation envelope, denominator rules, model callback, failure handling, and reporting fields before branch-specific results are examined. Any later amendment should preserve the original plan and state what changed and why.

## Methods checklist

Report:

- source and version of the nominal AOIs;
- rectangle/polygon representation and coordinate system;
- whether rows are samples or fixations;
- perturbation operations and exact values;
- pixels versus degrees of visual angle;
- display resolution, physical display size, and viewing distance for degree-based margins;
- overlap and screen-boundary policies;
- random seed for jitter;
- feature definitions and missing-data rules;
- the fixed statistical estimator/model specification;
- how callback failures and non-convergence were retained.

## Results checklist

Report assignment and inference layers together:

- comparable observations and unchanged/new/lost/reassigned proportions;
- important AOI-to-AOI reassignment patterns;
- participant/trial/AOI heterogeneity when relevant;
- coefficient median/range and confidence intervals;
- converged/attempted branches;
- model `N` median/range;
- geometry/model failures and their reasons.

Do not silently remove failed branches from the planned denominator.

## Manuscript-ready example

> We evaluated AOI-boundary sensitivity by rerunning assignment, feature extraction, and the prespecified model across the nominal geometry and a defensible perturbation set. Rows were analyzed at the fixation level. We report unchanged, newly assigned, lost, and reassigned observations; coefficient ranges and confidence intervals; model convergence; and model-N variation across branches. Failed geometry or model branches remained in the audit trail. Stability frequencies are interpreted as descriptive robustness to the declared perturbation set, not probabilities that the AOI definition or substantive conclusion is true.

A results sentence can be equally explicit:

> Across the seven planned branches, unchanged assignment ranged from 0.956 to 1.000; all model fits converged, model N was constant, and the target coefficient remained positive with overlapping uncertainty intervals. These results indicate stability within the declared perturbation envelope, not certainty that the nominal AOI boundaries are correct.

The numerical values above are an **illustrative synthetic reporting example**, not empirical evidence.

## Avoid

Do not describe a 90% same-sign frequency as a 90% probability that a finding is true. Do not choose the branch with the smallest p-value, quietly reduce the perturbation set after failures, or reinterpret missing observations as zero exposure.

Use `report_aoi_sensitivity()` for a compact starting draft, the [worked example](../../examples/aoi-perturbation-sensitivity.md) for end-to-end code, and the [AOI perturbation API](../../reference/aoi-perturbation.md) for function signatures.

## Reporting bundle

For manuscript supplements, reviewer responses, internal handoff, or replication packages, use the [AOI robustness reporting bundle](reporting-bundle.md) to keep the declared plan, branch audit, assignment summaries, model outputs, failures, provenance, narrative report, and diagnostic figures together. The [worked bundle](../../examples/aoi-reporting-bundle.md) also demonstrates a deterministic SHA-256 manifest without adding a public API.
