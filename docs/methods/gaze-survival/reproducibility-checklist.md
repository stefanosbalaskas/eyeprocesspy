---
title: Reproducibility checklist for censored gaze latency
---

# Reproducibility checklist

Use this checklist before preregistration, manuscript submission, or freezing a gaze-latency analysis. It is designed to make censoring, event construction, estimator choice, sensitivity analysis, and software provenance auditable.

## Design and estimand

- State the target event before inspecting condition effects.
- Name the target AOI and version or geometry specification.
- State the time origin and trial observation-window rule.
- State whether the estimand is survival probability, hazard ratio, time ratio, or a latent-frailty contrast.
- Distinguish first-event analysis from recurrent-event or competing-risk questions.

## Event construction

- Record the event detector and its parameters.
- Record any minimum fixation-duration rule.
- Record how consecutive samples/fixations are collapsed into AOI visits.
- Record the join key used to connect events to trials.
- Verify event time is non-negative and does not exceed the censoring limit.
- Review time-zero events explicitly.

## Censoring and quality

- Report observed events, censored trials, and censoring percentage overall and by condition.
- Confirm that a censored trial has a complete usable observation window.
- Keep incomplete or unresolved trials as review-required rather than ordinary censoring.
- Record gaze-quality thresholds and the number of rows they affect.
- Quantify whether quality failure or censoring differs by condition.
- Discuss whether censoring could be informative.

## Repeated observations and model family

- State the participant identifier used for repeated trials.
- Name clustered Cox versus latent frailty explicitly.
- Name the AFT family explicitly; do not rely on an implicit default.
- Report the effect measure as a hazard ratio or time ratio, not merely a coefficient.
- Do not rank Cox partial likelihood, frailty penalized likelihood, and AFT full likelihood by a single AIC/BIC table.

## Diagnostics

- Inspect event sparsity and extreme censoring.
- Inspect proportional-hazards diagnostics for Cox models.
- Record convergence warnings and reject failed/non-finite fits.
- For AFT models, justify the parametric family and include a planned family sensitivity check where appropriate.
- For R frailty Cox, report the corresponding marginal Cox PH diagnostic rather than applying `cox.zph()` to the `coxme` fit.

## Sensitivity analysis

Pre-specify which of these will be varied:

- AOI geometry or margin;
- event detector;
- minimum fixation duration;
- time origin;
- quality threshold;
- clustered versus frailty Cox where scientifically relevant;
- Weibull versus log-normal AFT;
- transparent review/exclusion rules.

Rebuild the survival table when the event/AOI definition changes. Do not relabel the same prepared table and call it a new scientific specification.

## Reporting bundle

Archive or report:

- participant/trial counts;
- event/censor/review counts;
- target event, AOI, and time origin;
- observation-window and quality rules;
- detector and fixation settings;
- model formula and repeated-observation structure;
- effect estimates with confidence intervals;
- diagnostics;
- sensitivity results;
- package and backend versions;
- random seed for synthetic or stochastic procedures;
- source-data and preprocessing provenance.

## Software provenance

For Python, record `eyeprocesspy`, `statsmodels`, `lifelines`, Patsy, NumPy, pandas, and SciPy versions where relevant. For R, record `eyeprocess`, `survival`, and `coxme` versions where relevant. If Gazepoint adapters are used, also record `gp3tools`.

## Final freeze questions

Before freezing the analysis, confirm:

1. Would a reader know exactly why every non-event trial is censored rather than missing?
2. Would a reader know exactly how the event timestamp was constructed?
3. Would a reader know whether the reported effect is marginal, conditional/frailty, or parametric-time based?
4. Would rerunning a defensible AOI/detector/quality alternative change the substantive conclusion?
5. Are all review-required rows and convergence/diagnostic failures still visible in the audit trail?

## Related pages

- [Decision guide](decision-guide.md)
- [Troubleshooting clinic](troubleshooting.md)
- [Reporting template](reporting-template.md)
- [Disclosure-inspection example](../../examples/gaze-survival-analysis.md)
- [Evidence-verification example](../../examples/gaze-verification-survival.md)
- [API reference](../../reference/gaze-survival.md)
