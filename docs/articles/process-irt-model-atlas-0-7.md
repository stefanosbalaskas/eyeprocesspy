# Process-IRT Model Atlas: What to Fit, What to Validate, What Not to Claim

## Purpose

The process-IRT layer is organized by **measurement question**, not estimator novelty. Eye-tracking, pupil, response time, omissions, sequences, and response combinations become measurement channels only when their role and validation evidence are declared.

## Core model families

| Question | Python API | Current scientific status |
|---|---|---|
| Do response, time, and gaze share structure? | `fit_joint_gaze_rt_irt()` | reference architecture; Python estimator differs from R lme4 |
| Do graded scores and time/process co-vary? | `fit_joint_graded_rt_process_irt()` | experimental |
| Which option was selected and inspected? | `fit_nominal_gaze_irt()` | experimental two-stage |
| Are omissions and not-reached items distinct time processes? | `fit_omission_survival_irt()` | experimental reference |
| Are process measures transportable across facets? | `fit_manyfacet_process_irt()` | reference architecture |
| Does the process change within session? | `fit_changepoint_multimodal_irt()` | experimental reference |
| Does a bounded process measure pile up at limits? | `fit_censored_normal_process_irt()` | conditional calibration |
| Are multiple selected options informative? | `fit_multiple_response_process_irt()` | reference/external-gated |
| Is there residual response/process local dependence? | `audit_process_local_dependence()` | diagnostic |
| Do revisits/RT/gaze add collateral CDM evidence? | `fit_revisit_process_cdm()` | experimental adapter |

## A process channel must earn its place

Use `process_channel_ablation()` with an out-of-sample evaluator. In-sample fit improvement alone is not enough to justify a gaze/pupil/process channel.

## Missingness: separate exposure from response

`classify_item_missingness()` distinguishes not-reached, reached-but-not-inspected, inspected omission, started-unanswered, and answered observations. Missingness classification is evidence about the response process, not a behavioral diagnosis.

## Cross-device measurement is an estimand

Use `fit_manyfacet_process_irt()`, `audit_process_measurement_invariance()`, and `cross_device_process_equating_audit()` alongside semantic/unit/coordinate audits and held-device validation. A small device variance component alone does not establish interchangeability.

## Latent distribution and IRF stress tests

The broader stress-testing and flexible-IRF families are later source tranches. Exact GPIRT/dynamic GPIRT/flow-MIRT/continuous-time estimators remain behind explicit gates until validated implementations are available.

## Promotion is evidence-based

Retain recovery, bias/RMSE, interval coverage, convergence/failure classification, misspecification, preprocessing sensitivity, grouped/external validation, and—for Bayesian models—SBC/PPC evidence before promotion.
