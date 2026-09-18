---
title: Decision guide for censored gaze latency
---

# Decision guide

Use this page when the analysis question is clear but you are unsure whether a trial should be treated as an observed event, a right-censored observation, a review case, or a different modelling problem.

## 1. Is there a pre-defined event?

Examples include first fixation on a disclosure, first AOI entry into evidence, first revisit, first transition into a target AOI, or a substantively defined disengagement event.

If the event is not defined before looking at the outcomes, stop and define the event rule first. Do not choose the event detector, AOI, or fixation-duration threshold from whichever version produces the clearest effect.

## 2. Is the observation window known and usable?

| Trial state | Survival treatment | Why |
| --- | --- | --- |
| Event observed inside a valid window | Observed event | Event latency is known |
| Event absent when a complete valid window ends | Right censored | Risk time is known up to the censoring time |
| Window incomplete or event status unknowable | Review required | Ordinary censoring is not justified |
| Gaze fails the declared quality rule | Review or explicit sensitivity branch | Quality failure is not the same as no event |
| Event timestamp falls after the censoring limit | Contract error | The trial/event definitions are inconsistent |

A missing event time is never sufficient evidence for censoring on its own.

## 3. Which model answers the scientific question?

| Scientific question | Preferred descriptive/model family | Primary estimand |
| --- | --- | --- |
| What share of trials remains uninspected over time? | Kaplan–Meier | Survival probability |
| How does a covariate alter the instantaneous event rate? | Cox PH | Hazard ratio |
| How does a covariate multiply event time? | Weibull/log-normal AFT | Time ratio |
| How should repeated trials be handled in Python? | Clustered Cox | Marginal HR with participant-clustered uncertainty |
| How should latent participant heterogeneity be modelled in R? | Gaussian frailty Cox via `coxme` | Conditional HR with participant random effect |

Do not interpret clustered Cox as a latent frailty model. Python deliberately rejects `structure="frailty"` rather than silently substituting clustered standard errors.

## 4. Which diagnostics are required?

For Cox models, inspect proportional-hazards diagnostics, event counts, censoring by condition, convergence, and sensitivity to AOI/event definitions. For AFT models, inspect the plausibility of the chosen parametric family and compare Weibull versus log-normal only as an explicitly planned sensitivity analysis.

If events are extremely sparse or censoring is extreme, convergence alone is not evidence that the estimates are scientifically stable.

## 5. When is this workflow not enough?

Use a different framework when:

- multiple event types genuinely compete;
- the target can recur and the scientific estimand concerns recurrent-event intensity rather than first occurrence;
- trial termination is plausibly informative and cannot be defended as ordinary right censoring;
- the observation window is unknown;
- the scientific question concerns the full gaze trajectory rather than time to a target event.

## Minimal analysis checklist

Before fitting a model, confirm that you can state all of the following:

- target event;
- target AOI;
- time origin;
- observation-window rule;
- event detector;
- fixation-duration rule where applicable;
- quality threshold;
- event and censoring counts by condition;
- repeated-participant structure;
- estimator family;
- diagnostic plan;
- sensitivity branches.

## Related pages

- [Method overview](index.md)
- [Methodological guide](../../guides/gaze-survival-analysis.md)
- [Worked disclosure-inspection example](../../examples/gaze-survival-analysis.md)
- [Reporting and limitations](reporting.md)
- [Copyable reporting template](reporting-template.md)
- [API reference](../../reference/gaze-survival.md)
