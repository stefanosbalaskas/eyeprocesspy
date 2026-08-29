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


## Requested API completion and semantic validation

The 0.7 completion layer is now explicit in Python. `fit_gaze_informed_missingness_irt()` is a two-part conditional diagnostic with an explicitly labelled smoothed person-score proxy when theta is not supplied. `fit_event_time_irt()` is a theta-conditioned Cox-reference diagnostic, not a full joint continuous-time latent-trait estimator. `device_facet_effects()`, `session_facet_effects()`, and `algorithm_facet_effects()` expose named facet evidence from `fit_manyfacet_process_irt()`.

For model-development validation, use `simulate_from_model()`, `extract_parameter_truth()`, and `fit_validation_replicate()` so recovery rows retain scenario, engine, convergence and truth metadata. For latent-distribution sensitivity, `audit_latent_distribution()`, `compare_latent_distribution_models()`, and `latent_distribution_stress_test()` separate descriptive distribution diagnostics from changes to the latent distribution inside a fitted IRT model.

Interoperability claims should be paired with the semantic-fidelity programme in `validation-evidence-0-7.md`, including vendor contracts, event/HED round trips, BIDS callback audits and cross-version adapter regression.

## Promotion is evidence-based

Retain recovery, bias/RMSE, interval coverage, convergence/failure classification, misspecification, preprocessing sensitivity, grouped/external validation, and—for Bayesian models—SBC/PPC evidence before promotion.
