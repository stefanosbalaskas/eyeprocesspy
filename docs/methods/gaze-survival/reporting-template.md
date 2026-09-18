---
title: Reporting template for gaze-latency survival analysis
---

# Copyable reporting template

This page turns the package outputs into a concise manuscript-ready reporting structure. Replace every bracketed field with values from the actual analysis.

## Methods template

> Gaze latency was analysed as a right-censored time-to-event outcome. The target event was [EVENT] within the [TARGET AOI] region, measured from [TIME ORIGIN]. Trials remained under observation until [OBSERVATION-WINDOW RULE]. Trials in which the target event did not occur before the end of a complete usable observation window were retained as right-censored observations; trials with unresolved event status or gaze quality below [QUALITY RULE] were not recoded as censoring. Events were defined using [DETECTOR/FIXATION RULE]. We fitted [MODEL FAMILY] with [REPEATED-PARTICIPANT STRUCTURE] and reported [HAZARD RATIOS / TIME RATIOS] with 95% confidence intervals. We examined [PH / PARAMETRIC] diagnostics and repeated the analysis under the pre-specified sensitivity specifications [LIST].

## Results template

> The analysis included [N PARTICIPANTS] participants and [N TRIALS] trials. The target event was observed in [N EVENTS] trials; [N CENSORED] trials ([CENSORING %]%) were right censored, and [N REVIEW] trials required review/exclusion under the pre-specified quality rules. Relative to [REFERENCE CONDITION], [CONDITION] was associated with a [HR OR TIME RATIO] of [ESTIMATE], 95% CI [LOW, HIGH], [TEST STATISTIC / p VALUE]. Proportional-hazards diagnostics were [RESULT], and the primary interpretation was [INTERPRETATION]. The substantive conclusion was [ROBUST / QUALIFIED / SENSITIVE] across the pre-specified AOI, event-definition, quality-threshold, and estimator sensitivity analyses.

## Interpretation reminders

For Cox models, a hazard ratio above one means that the target event occurs at a higher instantaneous rate among trials still at risk. It does **not** mean that mean TTFF is smaller by the same percentage.

For AFT models, a time ratio above one indicates multiplicatively longer event times under the fitted parametric distribution; a value below one indicates shorter event times.

Clustered Cox and latent frailty Cox are not interchangeable. The former changes the uncertainty structure around a marginal Cox model; the latter adds a latent participant-level random effect.

## Minimum table fields

A compact manuscript table should contain:

| Field | Recommended content |
| --- | --- |
| Outcome | Event type + target AOI |
| Time origin | Trial/stimulus/event origin |
| Participants / trials | Counts entering the survival table |
| Events / censored | Counts and censoring percentage |
| Review-required rows | Count and handling rule |
| Model | Cox / clustered Cox / frailty Cox / Weibull AFT / log-normal AFT |
| Repeated structure | Participant cluster or frailty |
| Effect | HR or time ratio |
| Uncertainty | 95% CI |
| Test | Statistic and p value where appropriate |
| Diagnostics | PH or parametric-family checks |
| Sensitivity | AOI/event/quality/model branches |
| Software | Package/backend versions |

## Package helpers

Use these functions together rather than reporting a model coefficient in isolation:

```python
from eyeprocesspy.survival import (
    summarise_gaze_censoring,
    tidy_gaze_survival_model,
    check_gaze_proportional_hazards,
    report_gaze_survival_model,
)

print(summarise_gaze_censoring(data, by="condition"))
print(tidy_gaze_survival_model(model))
print(check_gaze_proportional_hazards(model))
print(report_gaze_survival_model(model))
```

For AFT models, replace the PH diagnostic with the pre-specified parametric-family assessment and sensitivity comparison.

## Required limitations statement

At minimum, disclose whether censoring could be informative, whether event counts were sparse, whether time-zero events occurred, whether the result depended on AOI or detector definitions, and whether the repeated-participant structure changes interpretation.

## Related pages

- [Decision guide](decision-guide.md)
- [Reporting and limitations](reporting.md)
- [Methodological guide](../../guides/gaze-survival-analysis.md)
- [Worked example](../../examples/gaze-survival-analysis.md)
- [API reference](../../reference/gaze-survival.md)
