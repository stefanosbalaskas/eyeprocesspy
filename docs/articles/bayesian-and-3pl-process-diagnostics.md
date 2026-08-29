# Bayesian and 3PL Process Diagnostics

## Scope

The frozen R package provides two separate diagnostic layers: Bayesian `brms` diagnostics (LOO, posterior convergence/effective sample size, optional Bayes factors) and a standard `mirt` 3PL calibration aligned descriptively with item-level process summaries.

`eyeprocesspy` preserves those backend identities. `bayesian_process_diagnostics_dashboard()` and `fit_gaze_anchored_3pl_audit()` therefore raise actionable backend errors when the exact R-specific estimator is unavailable instead of substituting another Bayesian or 3PL implementation.

`bayesian_process_diagnostic_flags()` and `audit_3pl_process_signatures()` are estimator-independent post-fit diagnostics and remain executable when supplied the corresponding validated result contract.

A 3PL lower asymptote is an **item psychometric parameter**, not a participant-level guessing label. Correlations or review flags involving TTFF, dwell, pupil, RT, or accuracy are descriptive response-process evidence and require independent validation before interpretation in terms of rapid responding, guessing, effort, engagement, or strategy.
