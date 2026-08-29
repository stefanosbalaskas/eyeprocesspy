# Validation Before Promotion: Recovery, SBC, PPC, and Transportability

This article is the Python counterpart of the frozen `advanced-model-validation-0-7` vignette. The central rule is unchanged: advanced process-IRT models require a common evidence programme before promotion.

## A common validation contract

`irt_validation_spec()` records the model identifier, planned replications, parameter families, recovery metrics, grouped transport validation and evidence thresholds. It is an evidence contract, not an estimator.

A validation programme should ask whether parameters can be recovered, whether bias/RMSE/coverage are acceptable, how often fitting fails, whether parameters are empirically identifiable, whether Bayesian inference is calibrated, whether posterior predictive replicas reproduce scientifically relevant features, whether results survive misspecification and preprocessing changes, and whether measurement transports to held-out devices, sessions, sites and items.

## Recovery tables

Use `as_irt_recovery_results()` and `summarize_parameter_recovery()` to obtain bias, absolute bias, RMSE, MAE, coverage, interval width and convergence/failure rates. The corresponding audits—`audit_bias()`, `audit_rmse()`, `audit_coverage()`, `audit_interval_width()`, `audit_convergence()` and `audit_identifiability()`—preserve explicit thresholds. `validation_mcse()` and `recommended_validation_replications()` quantify Monte Carlo uncertainty rather than treating a simulation count as automatically sufficient.

## Prior and posterior SBC

`run_sbc()` implements callback-driven prior simulation-based calibration and stores normalized ranks plus classified failures. `audit_sbc()` provides a coarse rank-uniformity screen. SBC validates the inference algorithm under the generator; it does not establish empirical model fit or construct validity.

Posterior SBC is deliberately separate. `posterior_sbc_contract()` requires a model-specific conditional self-consistency callback and `run_posterior_sbc()` executes it. Ordinary prior SBC is not relabelled as posterior SBC.

## Posterior predictive checks

`posterior_predictive_discrepancies()` compares user-selected discrepancies between observed and replicated datasets and reports lower, upper and two-sided tail probabilities. Discrepancies should be chosen because failure would matter scientifically—for example omission rates, tail fixation counts, transition entropy, pupil timing or accuracy by item difficulty.

## Misspecification

`stress_test_misspecification()` is the common runner. Specialized helpers construct scenario grids for latent-distribution misspecification, local dependence, speededness, missingness and preprocessing variants. Errors are retained and classified through `validation_failure_taxonomy()` rather than discarded.

## External and grouped validation

`external_validate_irt()` tests a completely held-out dataset. `leave_device_out_validation()`, `leave_session_out_validation()`, `leave_site_out_validation()` and `leave_item_out_validation()` create grouped holdout folds. `audit_measurement_transportability()` summarizes between-group stability instead of reporting only a pooled score.

## Incremental information and negative controls

`audit_channel_incremental_information()` asks whether a process channel improves held-out performance relative to a baseline. `negative_control_process_test()` permutes named process variables, optionally within strata, and compares the observed evaluation score against the permutation distribution. This is intended to guard against crediting high-dimensional gaze/process channels merely because they improve in-sample fit.

## Evidence grade

`grade_model_evidence()` summarizes supplied recovery, SBC, PPC, external-validation and semantic-roundtrip evidence against the declared contract. Its grade is a governance summary only. It is not construct validity, causal evidence or independent replication.
