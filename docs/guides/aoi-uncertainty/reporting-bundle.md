# AOI robustness reporting bundle

A reporting bundle turns one sensitivity result into a **reviewable evidence package** rather than a single summary paragraph. The bundle should preserve the analysis plan, branch audit, assignment summaries, model outputs, failures, provenance, report text, and the figures used to interpret robustness.

## Recommended file set

| File | Purpose |
| --- | --- |
| `analysis-plan.yml` | prespecified perturbation/model/failure plan |
| `perturbation-audit.csv` | planned branches, completion status, geometry failures |
| `assignment-stability.csv` | unchanged/new/lost/reassigned summaries |
| `assignment-table.csv` | observation-level branch assignments |
| `assignment-frequency.csv` | empirical assignment frequencies across declared branches |
| `model-results.csv` | branch-level coefficient, interval, convergence, and N |
| `inference-stability.csv` | term-level robustness summary |
| `failures.csv` | retained geometry/model failures |
| `provenance.json` | source/AOI hashes, preprocessing, detector, model, software, caveat |
| `report.md` | compact manuscript-oriented narrative |
| `aoi-*.svg` | geometry, assignment, and coefficient diagnostics |
| `manifest.json` | stable paths, roles, byte sizes, and SHA-256 hashes |

## Why bundle the evidence?

A coefficient trajectory alone does not show whether a branch failed, whether model N changed, whether an assignment became ambiguous, or whether the analysis plan changed after results were inspected. The bundle keeps those layers together.

Use the bundle for:

- manuscript supplements;
- reviewer/editor responses;
- internal analysis handoff;
- release evidence;
- replication packages.

## Deterministic manifest

The worked Python example writes `manifest.json` without timestamps. Each file except the manifest itself has a stable role, byte count, and SHA-256 hash. Re-running the same writer on identical inputs therefore yields the same manifest.

The manifest is **not** a scientific validity certificate. It verifies bundle identity and file integrity; interpretation still depends on the declared AOI envelope, data quality, estimator, and study design.

## Reviewer-facing reading order

1. Read `analysis-plan.yml` to see what was fixed before branch results.
2. Inspect `perturbation-audit.csv` and `failures.csv` for non-evaluable branches.
3. Inspect `assignment-stability.csv` and the geometry/assignment figures.
4. Inspect `model-results.csv` and `inference-stability.csv`, including model N.
5. Read `provenance.json` for preprocessing, detector, model, and software context.
6. Use `report.md` as the narrative summary, not as a substitute for the tables above.

## Interpretation

A complete, internally consistent bundle supports a claim such as “stable within the declared perturbation envelope.” It does **not** establish that the nominal AOI boundaries are objectively correct, that calibration error is negligible, or that the statistical conclusion is true with some probability.

## Limitations

- File hashes prove identity, not scientific correctness.
- A bundle cannot rescue an implausible perturbation plan.
- Geometry sensitivity does not replace detector, calibration, missingness, or model-diagnostic sensitivity.
- Large observation-level tables may need privacy review before external sharing.
- If the analysis plan changes, preserve both plan versions and document the amendment.

Continue to the [worked reporting bundle](../../examples/aoi-reporting-bundle.md), [analysis plan](analysis-plan.md), [reporting guidance](reporting.md), [troubleshooting](troubleshooting.md), and [API reference](../../reference/aoi-perturbation.md).
