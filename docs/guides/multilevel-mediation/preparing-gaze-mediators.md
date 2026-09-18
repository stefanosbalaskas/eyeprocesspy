# Preparing trial-level gaze mediators

## Recommended sequence

1. Reconstruct trials and AOIs using the usual eyeprocesspy pipeline.
2. Apply or record the chosen event detector and preprocessing specification.
3. Derive the trial-level mediator, such as source dwell, first-entry latency, revisit count, or inspection indicator.
4. Attach a trial-level quality measure where available.
5. Call `prepare_multilevel_mediation_data()`.
6. Inspect `levels`, `variance`, `missingness`, and `trial_counts` before any Bayesian fit.

## Zero is not missing

A zero dwell time can mean the AOI was validly observable but never fixated. A missing dwell value can mean the trial was not observed, synchronization failed, the gaze stream was unusable, or the mediator could not be computed. These states are not interchangeable.

```python
prepared.data[
    [
        "source_dwell",
        "mediation_mediator_observed",
        "mediation_mediator_true_zero",
        "mediation_poor_quality",
        "mediation_mediator_state",
    ]
]
```

## Quality policy

`quality_action="flag"` is the default. It preserves the observed mediator and marks the trial as ineligible under the declared rule. `quality_action="mask_mediator"` is available only when the analyst explicitly wants poor-quality mediator values replaced by missing before decomposition.

No option silently deletes the trial.

## Non-Gaussian mediators

Preparation does not transform a skewed or bounded gaze mediator merely to satisfy a Gaussian model. Keep the scientifically meaningful scale and choose an appropriate family later in gp3bayespy. Positive durations can use lognormal or Gamma models; binary inspection can use Bernoulli; count mediators can use Poisson or negative binomial where justified.
