---
title: Worked example — disclosure inspection
---

# Worked example: disclosure inspection

This example begins with **synthetic trial windows plus fixation rows** rather than a pre-censored table. The target is a disclosure AOI. Some trials never inspect it before the valid trial window ends and therefore become right-censored observations.

![Kaplan–Meier curves from the worked example](../assets/gaze-survival/km-disclosure.svg)

## 1. Generate synthetic trial/event inputs

```python
from eyeprocesspy.survival import simulate_gaze_survival_inputs

raw = simulate_gaze_survival_inputs(
    "disclosure",
    seed=20260918,
    n_participants=36,
    trials_per_participant=3,
)
```

The synthetic event table intentionally omits `participant_id`, mirroring canonical episode workflows where `recording_id + trial_id` identifies the event stream.

## 2. Construct event and censoring rows

```python
from eyeprocesspy.survival import prepare_gaze_survival_data

data = prepare_gaze_survival_data(
    raw["trials"],
    raw["events"],
    target_aoi="disclosure",
    event_type="first_fixation",
    condition_col="condition_id",
    min_valid_fraction=0.90,
    event_detector="synthetic_truth",
    aoi_specification="fixed synthetic disclosure AOI",
    quality_rules={"minimum_fixation_ms": 80, "valid_fraction_min": 0.90},
)
```

No target event inside a complete usable trial produces `event_observed = 0` and `analysis_time = censor_time`. Unusable or incomplete trials would instead retain an unknown event status plus `review_required = True`.

## 3. Audit censoring

```python
from eyeprocesspy.survival import summarise_gaze_censoring, validate_gaze_survival_data

print(validate_gaze_survival_data(data, raise_on_error=False))
print(summarise_gaze_censoring(data, by="condition"))
```

Report the censoring proportion by condition; a condition-specific difference in usable follow-up or event occurrence can itself be substantively important.

## 4. Plot Kaplan–Meier curves

```python
from eyeprocesspy.survival import plot_gaze_survival_curve

ax = plot_gaze_survival_curve(data, group="condition")
ax.set_title("Time to first disclosure fixation")
```

Lower survival at a given time means a larger share of trials has already inspected the target by that point.

## 5. Fit repeated-participant Cox and AFT models

```python
from eyeprocesspy.survival import fit_gaze_aft_model, fit_gaze_mixed_cox_model

cox = fit_gaze_mixed_cox_model(
    data,
    "C(condition)",
    participant_col="participant_id",
    structure="cluster_robust",
)
weibull = fit_gaze_aft_model(data, "C(condition)", distribution="weibull")
lognormal = fit_gaze_aft_model(data, "C(condition)", distribution="lognormal")
```

The Cox model reports hazard ratios; the AFT models report time ratios. They should be interpreted as different estimands rather than ranked automatically.

## 6. Diagnose and compare transparently

```python
from eyeprocesspy.survival import (
    check_gaze_proportional_hazards,
    compare_gaze_survival_models,
    tidy_gaze_survival_model,
)

print(tidy_gaze_survival_model(cox))
print(tidy_gaze_survival_model(weibull))
print(check_gaze_proportional_hazards(cox))
print(compare_gaze_survival_models(cox, weibull, lognormal))
```

The comparison table includes a likelihood-basis field and a flag showing whether information criteria are directly comparable. Cox-versus-AFT comparisons are deliberately marked non-comparable on AIC/BIC grounds.

## 7. Run explicit sensitivity branches

```python
from eyeprocesspy.survival import compare_gaze_survival_specifications

expanded = data.copy()
expanded["aoi_specification"] = "synthetic expanded disclosure AOI"

sensitivity = compare_gaze_survival_specifications(
    {"primary": data, "expanded_aoi_demo": expanded},
    "C(condition)",
    model_families=["cox_cluster_robust", "aft_weibull"],
)
```

In a real study, rebuild `expanded` from the alternative AOI geometry or detector; do not merely change its label. The example relabels it only to demonstrate the output contract without fabricating a second scientific truth.

## 8. Produce manuscript-ready output

```python
from eyeprocesspy.survival import report_gaze_survival_model

report = report_gaze_survival_model(cox)
print(report["N_participants"])
print(report["N_censored_trials"])
print(report["effects"])
```

The repository script `examples/worked_gaze_survival_analysis.py` executes the entire workflow and writes censoring, effect, diagnostic, sensitivity, report, and SVG figure files.

## Failure case: unusable gaze is not censoring

```python
bad = raw["trials"].copy()
bad.loc[0, "valid_data_fraction"] = 0.20

reviewed = prepare_gaze_survival_data(
    bad,
    raw["events"],
    target_aoi="disclosure",
    min_valid_fraction=0.90,
)

assert reviewed.loc[0, "review_required"]
assert reviewed.loc[0, "event_observed"] != 0
```

That row must be resolved or handled in an explicit sensitivity/exclusion branch before inferential model fitting.