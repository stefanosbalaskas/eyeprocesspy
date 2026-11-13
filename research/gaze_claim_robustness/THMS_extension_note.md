# THMS: post-freeze measurement-reporting extension

## Provenance and scope

This journal-specific extension derives **descriptive comparisons only** from the
existing 144-row `results/srl_paper_figure_snapshot.csv`. No participant data,
detector settings, AOI definitions, statistical models, or frozen primary results
were changed. It is **post-freeze**, not preregistered, and must not be described
as an independent empirical replication or new confirmatory test.

The CHI-audited research head remains `aab7213da03a6383ddfa6cc74db38adcaac0c1e7`.
This THMS extension is on a separate branch.

## Reporting-view comparison

The illustrative reference uses I-VT 40°/s and 50-ms minimum duration, left
eye, **65-cm midpoint assumption**, exact-quarter AOIs, released-sample quality,
and the 82-person exact-match cohort. Its universe ID is `srl_u057`. This
combination is a **post-freeze illustration**, not a chosen best detector,
primary preregistered analysis, true observed viewing distance, or ground truth.

| View | Branches | Positive/negative estimates | Median RR | RR range | Wald 95% CI includes null |
|---|---:|---:|---:|---:|---:|
| Illustrative single pipeline | 1 | 1/0 | 1.035459 | 1.035459 | 1/1 |
| Only detector varied, other settings fixed | 3 | 1/2 | 0.988163 | 0.973073–1.035459 | 3/3 |
| Full frozen cross-stage universe | 144 | 92/52 | 1.034604 | 0.943965–1.257701 | 144/144 |

**Important negative finding:** detector-only variation already produces both
positive and negative point estimates. It is therefore *incorrect* to claim that
only the full cross-stage analysis reveals a sign change.

The full cross-stage analysis additionally reveals a wider range and the
detector × visual-angle-geometry structure. The original frozen decision-summary
tables identify viewing geometry as the largest marginal descriptive shift.
These nested reporting views are not statistically independent replications,
and their branch frequencies are not probabilities of an effect.

## Sample support

The four cohort/quality combinations have these **model-evaluable trial rows**
(after source integrity auditing):

| Cohort | Released-sample quality | Trial tracking ratio ≥80% |
|---|---:|---:|
| 82 exact-match participants | 647 | 634 |
| 77 nominal-rate participants | 608 | 595 |

The measurement ledger records 23,616 planned trial/specification rows, of
which 23,292 have status `ok`; 324 are not evaluable because nine underlying
participant-by-task timestamp streams are not strictly increasing. They were
not repaired or imputed. This is distinct from 144/144 successfully fitted
model branches.

## Protocol for a new application

1. **Source audit:** obtain measurement units, timebase, calibration/display
   geometry, sample validity, tasks, and participant identifiers.
2. **Claim contract:** identify exactly one scientifically meaningful outcome,
   focal contrast, exposure definition, estimand, model family, and population.
3. **Admissible choices:** freeze only scientifically justified detector, geometry,
   AOI, eye, and other measurement alternatives; mark cohort and quality
   variations as sample-support sensitivities.
4. **Complete ledger:** assign an ID to every planned branch before result
   inspection; record execution status, trial/participant support, errors,
   estimate, and model-conditional interval.
5. **Report dimensions separately:** point-estimate direction and magnitude,
   interval crossing, evaluability, sample support, stage-specific patterns.
6. **Independent check:** test event-representation agreement on a separate
   source where appropriate; never call detector consensus accuracy without
   an independent ground truth.

This is an **evaluation protocol**, not a new estimator or guarantee of better
system performance.

## Independent evidence and limits

The separate MCFW-Gaze corpus comprises 15 participants and 4,695 trials;
28,164/28,170 detector summaries and 27,660/28,170 detector-pair rows were
evaluable. Overall median raw agreement = 0.7068 and positive-class
Jaccard = 0.2724. These nested trial comparisons must not be interpreted as
independent participant replications. MCFW lacks a directly comparable SRL
Prompt contrast and fixation ground-truth labels.

## Reproducibility

- Code: `research/gaze_claim_robustness/thms_frozen_reporting_comparison.py`
- Frozen source: `results/srl_paper_figure_snapshot.csv`
- New derived table: `results/thms_reporting_comparison.csv`
- New three-branch detail: `results/thms_detector_only_snapshot.csv`
- Original source and model contracts are unmodified.

## Eligibility of a second end-to-end dataset

A second genuine claim-level case would need public raw samples, original
condition assignment/task structure, screen/AOI geometry, a source-justified
behavioral contrast, and a frozen analysis plan before viewing resulting
coefficients. The existing dataset feasibility audit identifies limitations in
public UEyes event-level exports, dynamic web-AOI registration, and restricted
datasets. No additional dataset is called a validated replication here.
