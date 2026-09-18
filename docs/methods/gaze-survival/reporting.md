---
title: Reporting gaze-latency survival models
---

# Reporting and limitations

## Minimum reporting set

Report the participant count, trial count, observed-event count, censored-trial count and percentage, review-required rows, target event, target AOI, time origin, observation-window definition, event detector, fixation-duration rule where relevant, gaze-quality rule, repeated-participant structure, estimator family, effect estimate with 95% CI, diagnostics, and sensitivity branches.

`report_gaze_survival_model()` returns these fields together with model and software provenance. `summarise_gaze_censoring()` should normally be reported by experimental condition as well as overall.

## Interpretation templates

For Cox models, an exponentiated coefficient is a **hazard ratio**. It describes the relative instantaneous target-event rate among trials still at risk; it is not a ratio of mean TTFF values.

For AFT models, an exponentiated coefficient is a **time ratio** under the fitted distribution. Values above one indicate multiplicatively longer event times and values below one indicate shorter event times, conditional on the model specification.

## Example manuscript wording

> Gaze latency was analysed as a right-censored time-to-event outcome. Trials in which the disclosure AOI was not fixated before the valid trial window ended were retained as censored observations; trials with unresolved gaze validity were not reclassified as censoring. We fitted a Cox proportional-hazards model with participant-clustered standard errors and reported hazard ratios with 95% confidence intervals. Proportional-hazards diagnostics and sensitivity analyses across pre-specified AOI/event definitions were examined. A Weibull AFT specification was reported as an alternative estimand and interpreted using time ratios.

Adapt this text to the actual detector, AOI, quality threshold, time origin, and repeated-effects structure used in the study.

## Important limitations

- A missing event is not evidence of censoring without a known usable risk window.
- Censoring mechanisms should be scientifically defensible; informative dropout can bias standard survival estimators.
- Very sparse events can make Cox and AFT estimates unstable even when software converges.
- Time-zero events require a pre-specified rule grounded in sampling resolution and the definition of trial onset.
- Python cluster-robust Cox and R latent frailty Cox are different estimands and uncertainty structures; numerical identity is not claimed.
- Competing risks require a separate competing-risks framework rather than interpreting `1 - KM` as a cause-specific cumulative incidence function.