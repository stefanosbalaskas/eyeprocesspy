# Trial-level mediation readiness checklist

Use this checklist **before** handing a prepared dataset to a mediation estimator. It is intentionally conservative: a checked box means the design/data support the requested analysis step, not that a causal mediation claim has been established.

## Design and timing

- [ ] The exposure precedes the gaze/process mediator in the trial sequence.
- [ ] The mediator precedes the behavioral outcome, or the scientific interpretation is explicitly non-causal.
- [ ] The participant and trial identifiers uniquely identify rows.
- [ ] The exposure varies at the level claimed in the analysis (within participant, between participant, or both).
- [ ] Any serial mediator ordering is justified by the task design rather than chosen post hoc.

## Observation semantics

- [ ] A genuine observed zero is distinguishable from an unobserved mediator.
- [ ] Poor-quality gaze is represented separately from missing gaze.
- [ ] Missing behavioral responses remain missing; they are not recoded as negative/zero outcomes.
- [ ] Quality thresholds and masking rules were declared before model fitting.
- [ ] No trial was silently deleted during preparation.

## Within/between decomposition

Inspect `levels`, `variance`, and `trial_counts` from the prepared object.

- [ ] `X_within` has variation when a within-participant exposure path is requested.
- [ ] `X_between` has variation before an `a_between` or `cprime_between` path is requested.
- [ ] `M_within` has variation before a within mediator→outcome path is requested.
- [ ] `M_between` has variation before a between mediator→outcome path is requested.
- [ ] Singleton participants are retained and flagged rather than silently removed.
- [ ] Unbalanced trial counts are summarized and considered in model support.

A perfectly balanced within-subject manipulation can make `X_between` constant. That is an expected design property, not a reason to force a between-person coefficient into the model.

## Likelihood support

Preparation does not choose the estimator. Before fitting, confirm that the observed mediator/outcome support is compatible with the intended likelihood:

| Observed variable | Typical candidate family | Important guardrail |
| --- | --- | --- |
| roughly symmetric continuous measure | Gaussian | inspect tails and scale |
| strictly positive duration | lognormal or Gamma | exact zeros are outside support |
| proportion strictly inside (0,1) | beta | exact 0/1 need another model or justified transformation |
| inspection indicator | Bernoulli | define success explicitly |
| count | Poisson / negative binomial | inspect overdispersion |
| binary decision | Bernoulli/logit | indirect coefficient product is on link scale |

## Provenance handoff

The prepared object should retain, where applicable:

- source identity or fingerprint;
- preprocessing specification;
- event detector specification;
- AOI definition/version;
- quality rule;
- decomposition convention;
- missingness/observation-state semantics;
- software/package version.

The downstream Bayesian package should consume these prepared columns and provenance rather than reimplementing the decomposition.

## Stop conditions

Do not proceed to substantive mediation interpretation when the requested path has no observed design support, the mediator/outcome ordering is incompatible with the proposed mechanism, or missingness/quality decisions have not been made explicit. A model that can be fit computationally is not automatically a scientifically identified mediation model.

## Related API

- `prepare_multilevel_mediation_data()` — canonical preparation.
- `validate_multilevel_mediation_data()` — design and schema checks.
- `identify_mediation_levels()` — within/between support classification.
- `summarise_within_between_variance()` — decomposition support.
- `audit_mediation_missingness()` — missing/zero/quality audit.
- `check_mediation_trial_counts()` — participant trial-support audit.
- `add_multilevel_mediation_component()` — additional serial mediator/moderator preparation.
