# Gaze-informed diffusion-IRT modelling

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/gaze-diffusion-irt.Rmd`.

## Confirmatory parameter mapping

Gaze features must be assigned to theoretically defensible diffusion parameters before fitting. A feature cannot be placed simultaneously on drift, boundary, non-decision time, and starting bias in a confirmatory specification.

```python
import eyeprocesspy as ep

spec = ep.gaze_diffusion_spec(
    response="score",
    response_time="response_time",
    drift_features=("evidence_dwell_balance", "verification_transitions"),
    boundary_features=("warning_dwell",),
    nondecision_features=("first_fixation_latency",),
    starting_features=("initial_option_bias",),
    censor_column="rt_censoring",
    contaminant=True,
    engine="stan",
)

prepared = ep.prepare_gaze_diffusion_data(trials, spec)
fit = ep.fit_gaze_diffusion_irt(trials, spec, seed=42)
```

The Stan engine uses a Wiener first-passage likelihood for observed responses, mirrored parameters for the lower boundary, censoring contributions, person/item heterogeneity, and an optional contaminant mixture. Backend availability remains explicit.

## Identification and posterior checks

```python
ep.extract_diffusion_parameters(fit)
ep.diffusion_parameter_diagnostics(fit, correlation_threshold=0.85)
ep.diffusion_posterior_predictive(fit)
ep.compare_diffusion_accuracy_rt(fit)
```

Generated predictive response times are diagnostic simulations; likelihood-based inference remains tied to the declared diffusion model.

## Simulation programme

```python
programme = ep.diffusion_identification_study(
    conditions={
        "n_person": (50, 150, 500),
        "n_item": (10, 30),
        "gaze_effect": (0.0, 0.20, 0.40),
        "contaminant_fraction": (0.0, 0.05),
    },
    replications=200,
)
```

Promotion requires identification, parameter recovery, interval coverage, contaminant and censoring sensitivity, grouped validation, comparison with conventional accuracy–RT models, and empirical reproduction.
