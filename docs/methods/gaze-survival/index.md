---
title: Censored gaze-latency survival analysis
---

# Censored gaze-latency survival analysis

<div class="grid cards" markdown>

-   :material-timer-sand: **Keep never-inspected trials**

    A valid trial that ends before the target AOI is inspected remains informative as a right-censored observation. It is not dropped and it is not assigned a latency of zero.

-   :material-account-check: **Separate censoring from unusable gaze**

    Missing or incomplete gaze produces an explicit review state. `eyeprocesspy` does not infer ordinary censoring merely from a missing event time.

-   :material-account-multiple: **Respect repeated participants**

    Python exposes participant-clustered Cox uncertainty. The R package additionally exposes latent participant frailty through the specialist `coxme` engine.

-   :material-source-branch: **Audit specification sensitivity**

    AOI geometry, event detector, time origin, fixation threshold, quality rule, and model family remain explicit analysis branches with preserved provenance.

</div>

![Kaplan–Meier curves for the synthetic disclosure-inspection example](../../assets/gaze-survival/km-disclosure.svg)

## When to use it

Use this pathway when the outcome is genuinely **time to an event**: first fixation, first AOI entry, first evidence inspection, first revisit, first transition into a target AOI, or a substantively defined disengagement latency. It is especially important when a non-trivial number of trials end before that event occurs.

A typical workflow is:

```text
canonical trials + fixation/AOI-visit events
→ define target event and time origin
→ construct event/censoring rows
→ validate observation windows and quality
→ Kaplan–Meier description
→ explicitly requested Cox and/or AFT model
→ repeated-participant treatment
→ diagnostics
→ specification sensitivity
→ manuscript-ready report
```

## When not to use it

Do not use this workflow when event status cannot be determined, the observation window is unknown, competing event types require a competing-risks estimand, or the scientific question concerns the full continuous gaze trajectory rather than event latency. Informative trial termination also requires substantive sensitivity analysis; ordinary right censoring is not automatically justified simply because the recording ended.

## Model choices

| Question | Model | Primary interpretation |
| --- | --- | --- |
| What proportion remains uninspected over time? | Kaplan–Meier | Survival probability |
| How do covariates change the instantaneous event rate? | Cox PH | Hazard ratio |
| How do covariates multiply event time? | Weibull/log-normal AFT | Time ratio |
| How do I account for repeated trials in Python? | Clustered Cox | Marginal HR with cluster-robust SE |
| How do I estimate latent participant frailty in R? | `coxme` mixed Cox | Conditional HR with random participant effect |

Cox and AFT estimates answer different questions. The package therefore summarizes them side by side but does not declare a winner. Cox partial-likelihood and AFT full-likelihood AIC/BIC are marked as non-comparable across families.

## Backend map

| Scientific task | Python backend | R backend | Contract note |
| --- | --- | --- | --- |
| Cox PH | `statsmodels.PHReg` | `survival::coxph` | Semantic parity; backend diagnostics can differ numerically |
| Repeated-trial marginal Cox | `PHReg.fit(groups=...)` | `coxph(..., cluster=...)` | Participant dependence is handled without pretending to be frailty |
| Participant frailty Cox | Not currently exposed; request fails | `coxme::coxme` | No silent substitution in Python |
| Weibull AFT | `lifelines.WeibullAFTFitter` | `survival::survreg(dist = "weibull")` | Time-ratio estimand aligned; parameterization may differ |
| Log-normal AFT | `lifelines.LogNormalAFTFitter` | `survival::survreg(dist = "lognormal")` | Time-ratio estimand aligned; parameterization may differ |

## Install the modelling backends

```bash
pip install "eyeprocesspy[survival]"
```

The extra installs the tested `statsmodels`, `lifelines`, Patsy, and plotting dependencies used by the modelling and diagnostic examples.

## Start here

- [Methodological guide](../../guides/gaze-survival-analysis.md)
- [Decision guide](decision-guide.md)
- [Runnable disclosure-inspection example](../../examples/gaze-survival-analysis.md)
- [Runnable evidence-verification example](../../examples/gaze-verification-survival.md)
- [Reporting and limitations](reporting.md)
- [Copyable reporting template](reporting-template.md)
- [Troubleshooting clinic](troubleshooting.md)
- [API reference](../../reference/gaze-survival.md)