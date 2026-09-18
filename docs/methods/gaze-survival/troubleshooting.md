---
title: Troubleshooting censored gaze-latency analysis
---

# Troubleshooting common failures

This page is intentionally conservative. A failure in validation or model fitting is a signal to resolve the scientific/data contract, not a reason to silently recode or drop rows.

## Missing event time

A missing event time does **not** automatically mean right censoring. Confirm that the trial has a complete usable observation window and that the target event was genuinely absent. Otherwise keep the row as review-required.

## Event occurs after the censoring limit

Treat this as a contract error. Check the time origin, event timestamp units, trial-window boundaries, and event/trial join key. Do not truncate the event to the censoring time.

## Target is already fixated at time zero

Review the scientific time origin and the display/sampling resolution. AFT models require strictly positive analysis times. Do not add an arbitrary epsilon unless that rule was pre-specified and scientifically justified.

## Extreme censoring or very sparse events

Convergence does not guarantee stable inference. Report event and censoring counts by condition, simplify the model only for a defensible scientific reason, and consider whether the estimand is supported by the available information.

## Proportional-hazards diagnostic is flagged

Do not ignore the diagnostic. Inspect time-varying effects, consider a pre-specified alternative estimand/model, and report the diagnostic. A flagged PH check is not automatically repaired by switching to an AFT model after seeing the result.

## Python frailty request fails

This is expected. `eyeprocesspy` does not silently approximate latent participant frailty with clustered uncertainty. Use `structure="cluster_robust"` for the marginal repeated-trial Cox model, or use R `eyeprocess` with `coxme` when the scientific model requires latent participant frailty.

## Cox and AFT model-comparison table says information criteria are not comparable

This is deliberate. Ordinary Cox uses a partial likelihood, `coxme` frailty uses a penalized likelihood, and AFT uses a full likelihood. Interpret each estimand and its diagnostics rather than ranking all families by AIC/BIC.

## AOI sensitivity changes the conclusion

Report it. AOI geometry is part of the analytical specification. Rebuild the event/censor table under each defensible geometry; do not only relabel metadata on the same table in a real sensitivity analysis.

## Poor gaze quality differs by condition

Do not convert low-quality trials into ordinary censoring. Quantify the imbalance, keep review/exclusion rules explicit, and run a documented sensitivity analysis under defensible quality thresholds.

## Competing events are present

The single-event `1 - KM` plot is not a competing-risks cumulative incidence function. Use a dedicated competing-risks framework when mutually exclusive causes/events compete.

## Minimum failure report

When an analysis cannot proceed, preserve:

- participant/trial identifier;
- event definition and AOI;
- time origin and observation window;
- event/censor/review state;
- validation message;
- gaze-quality fields;
- preprocessing/event-detector/AOI specification;
- requested estimator family;
- software/backend versions.

## Related pages

- [Decision guide](decision-guide.md)
- [Methodological guide](../../guides/gaze-survival-analysis.md)
- [Disclosure-inspection example](../../examples/gaze-survival-analysis.md)
- [Evidence-verification example](../../examples/gaze-verification-survival.md)
- [Reporting template](reporting-template.md)
- [API reference](../../reference/gaze-survival.md)
