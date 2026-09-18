---
title: Survival analysis for gaze latency
---

# Survival analysis for gaze latency

## Why TTFF is a censored-data problem

Time to first fixation (TTFF), first AOI entry, evidence inspection, revisit, transition latency, and substantively defined disengagement are time-to-event outcomes. A valid trial that ends before the target event contributes information up to its observation limit and should remain as a right-censored observation. Dropping those trials conditions the latency distribution on eventual target inspection.

`eyeprocesspy.survival` therefore distinguishes three states:

1. **event observed** — the qualifying target event occurred inside the usable risk window;
2. **right censored** — the target event was not observed before a complete usable observation window ended;
3. **review required** — event status is unknown because the gaze record or observation window is incomplete or unusable.

Only the second state is ordinary right censoring.

## Canonical survival table

The one-row-per-participant-trial contract contains:

```text
participant_id      trial_id            stimulus_id
condition           target_aoi          time_origin
event_time          censor_time         analysis_time
event_observed      event_type          n_valid_samples
valid_data_fraction trial_duration
```

Governance columns additionally preserve `censor_reason`, `analysis_eligible`, `review_required`, observation-end reason, event join key, event detector, AOI specification, quality rules, preprocessing specification, source identity, and software versions.

## Preparing from canonical trials and events

```python
from eyeprocesspy.survival import (
    prepare_gaze_survival_data,
    simulate_gaze_survival_inputs,
)

raw = simulate_gaze_survival_inputs("disclosure", seed=20260918)

data = prepare_gaze_survival_data(
    raw["trials"],
    raw["events"],
    target_aoi="disclosure",
    event_type="first_fixation",
    condition_col="condition_id",
    min_valid_fraction=0.90,
    time_origin="trial_start",
    event_detector="synthetic_truth",
    aoi_specification="fixed synthetic disclosure AOI",
)
```

Canonical `eyeprocess` episode tables can be matched by `participant_id + trial_id` when participant identity is present, or by `recording_id + trial_id` when participant identity is not stored on each event row. Trial-only matching is accepted only when trial IDs are globally unique. Ambiguous identity is an error, not an inferred join.

A non-default origin must be explicit. For example, if latency begins at stimulus onset rather than trial onset, pass `time_origin="stimulus_onset"` together with `time_origin_col="stimulus_onset"`.

## Worked variant: evidence verification

The same contract can model time to first inspection of a source/evidence AOI. The synthetic verification design has repeated trials under `standard` and `evidence_prompt` conditions; some valid trials never enter the evidence AOI and therefore remain right-censored.

```python
raw_verify = simulate_gaze_survival_inputs(
    "verification",
    seed=20260918,
    n_participants=36,
    trials_per_participant=3,
)

verification = prepare_gaze_survival_data(
    raw_verify["trials"],
    raw_verify["events"],
    target_aoi="source_evidence",
    event_type="first_aoi_entry",
    condition_col="condition_id",
    min_valid_fraction=0.90,
    time_origin="trial_start",
    event_detector="synthetic_truth",
    aoi_specification="fixed synthetic source/evidence AOI",
)

print(summarise_gaze_censoring(verification, by="condition"))
```

This is not a binary “looked/did not look” analysis: trials that end without evidence inspection contribute their observed risk time instead of being discarded.

## Event definitions

`event_type` supports first fixation, first AOI entry, first evidence inspection, first revisit, first transition into the target, and disengagement. Revisit/transition/disengagement are treated as **visit-level** concepts: consecutive identical AOI labels are collapsed before those events are identified, so two successive fixations inside the same visit do not create a false revisit.

### Censoring decision guide

| Trial state | `event_observed` | `analysis_time` | Model eligible? | Interpretation |
| --- | ---: | ---: | --- | --- |
| Target event occurs inside a usable window | 1 | Event latency | Yes | Observed event |
| Target never occurs before a complete usable window ends | 0 | Censoring time | Yes | Right censored |
| Window incomplete or event status unknowable | NA | NA | No | Review required |
| Gaze quality fails the declared rule | NA | NA | No | Review/exclusion branch |
| Event occurs after declared censoring limit | Invalid | Invalid | No | Data-contract error |

The package never converts the last three cases into ordinary censoring automatically.

## Validation and safeguards

`validate_gaze_survival_data()` rejects duplicated participant-trial rows, invalid event indicators, negative/impossible times, observed events without event times, events after the censoring limit, inconsistent analysis times, missing risk windows on analyzable rows, and invalid gaze-quality fractions. It flags time-zero events, zero-follow-up censoring, sparse events, extreme censoring, and unresolved review rows.

Model fitting refuses review-required rows. Resolve them explicitly or construct a documented exclusion/sensitivity branch; they are never silently dropped by the estimator.

## Descriptive survival curves

```python
from eyeprocesspy.survival import estimate_gaze_survival, summarise_gaze_censoring

print(summarise_gaze_censoring(data, by="condition"))
km = estimate_gaze_survival(data, group="condition")
```

The Kaplan–Meier curve estimates the probability that the target event has **not yet** occurred. The corresponding `1 - KM` plot is useful for a single target event under ordinary right censoring. Do not call it a competing-risks cumulative incidence function when multiple event types compete.

## Cox versus AFT models

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

Cox coefficients are reported as hazard ratios. A hazard ratio above one means a higher instantaneous target-event rate among trials still at risk; it is not a ratio of mean latency. AFT coefficients are reported as time ratios and describe multiplicative changes in event time under the fitted distribution.

### Backend and parity contract

Python Cox models delegate estimation to `statsmodels.PHReg`. Python Weibull and log-normal AFT models delegate to `lifelines` (`WeibullAFTFitter` and `LogNormalAFTFitter`). R `eyeprocess` uses `survival::coxph`, `coxme::coxme` for true participant frailty, and `survival::survreg` for AFT. The public scientific contract—censoring semantics, explicit estimator choice, effect-measure meaning, provenance, diagnostics, and output fields—is aligned across languages. Exact coefficient identity is not promised when the statistical backends use different parameterizations or numerical optimizers.

Model family is always explicit. `compare_gaze_survival_models()` reports fit summaries but marks Cox partial-likelihood and AFT full-likelihood information criteria as non-comparable across families rather than ranking models automatically.

## Repeated observations and participant effects

Python `fit_gaze_mixed_cox_model(..., structure="cluster_robust")` uses the dependency-aware standard errors supported by `statsmodels.PHReg.fit(groups=...)`. This changes uncertainty for correlated trials but not fitted values. A request for `structure="frailty"` fails explicitly because the Python backend does not currently expose the same latent random-effect estimator.

R `eyeprocess` exposes true participant frailty through `coxme` and clustered Cox through `survival::coxph`. These are distinct estimators and are documented as such.

## Diagnostics

```python
from eyeprocesspy.survival import check_gaze_proportional_hazards

ph = check_gaze_proportional_hazards(cox)
print(ph)
```

The Python PH diagnostic examines time trends in Schoenfeld residuals and is a scientific-contract analogue to R `survival::cox.zph`, not a claim of numerical identity. Examine it together with event counts, censoring, influential observations, detector/AOI sensitivity, and convergence status.

AFT models require strictly positive analysis time. A target already fixated at time zero should trigger review of the time origin and sampling-resolution rule, not an automatic numerical offset.

## Sensitivity analysis

Rebuild the survival table across defensible AOI geometries, event detectors, trial origins, fixation-duration thresholds, and quality rules. Then fit explicitly named estimator families:

```python
from eyeprocesspy.survival import compare_gaze_survival_specifications

result = compare_gaze_survival_specifications(
    {
        "primary": data_primary,
        "expanded_aoi": data_expanded_aoi,
        "strict_quality": data_strict_quality,
    },
    "C(condition)",
    model_families=["cox_cluster_robust", "aft_weibull"],
)
```

Each output row preserves the named specification and relevant provenance. The function does not select a preferred detector, AOI, quality threshold, or estimator.

## Reporting

Use `report_gaze_survival_model()` together with `summarise_gaze_censoring()` and `tidy_gaze_survival_model()` to produce a compact manuscript bundle. The dedicated [reporting guide](../methods/gaze-survival/reporting.md) provides required fields, interpretation templates, and limitations.