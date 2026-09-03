# Dynamic IRTree and transition-model hardening

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/dynamic-irtree-hardening.Rmd`.

## Scope

The dynamic-state layer models transitions among explicitly declared AOI or process states. Observed states may be used directly, or an optional hidden-state model may separate noisy observations from latent states. A hidden state is not automatically a cognitive state; substantive interpretation requires theory and external validation.

## Simulation and observed-state models

```python
import pandas as pd
import eyeprocesspy as ep

sim = ep.simulate_dynamic_irtree_data(
    n_person=100,
    n_item=20,
    transitions_per_trial=10,
    state_misclassification=0.05,
    missing_state=0.10,
    seed=42,
)

spec = ep.dynamic_irtree_spec(
    engine="multinomial",
    include_person=True,
    include_item=True,
    condition_columns="condition",
    transition_predictors=("time_gap", "score"),
    structural_zeros=pd.DataFrame({"from": ["submit"], "to": ["prompt"]}),
)

fit = ep.fit_dynamic_irtree(sim.transitions, spec)
ep.decode_dynamic_states(fit)
ep.transition_residual_diagnostics(fit)
```

`dynamic_transition_design()` exposes the exact design matrix, transition mask, state coding, scaling, participant/item indices, and uncertainty weights before estimation.

## Hidden states with Stan

```python
hidden_spec = ep.dynamic_irtree_spec(
    engine="stan",
    hidden_states=3,
    missing_state="marginalize",
    person_effect="random",
    item_effect="random",
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
)

hidden_fit = ep.fit_dynamic_irtree(sim.transitions, hidden_spec, seed=42)
probability = ep.decode_dynamic_states(hidden_fit, method="probability")
```

The hidden engine uses a forward algorithm and estimates an emission matrix. Returned probabilities are filtered state probabilities, not claims about named cognition. If the Stan backend is unavailable or not validated in the current environment, the package raises or returns an explicit backend gate rather than silently substituting another estimator.

## Model comparison and recovery

```python
baseline = ep.fit_dynamic_irtree(
    sim.transitions,
    ep.dynamic_irtree_spec(engine="baseline"),
)
multinomial = ep.fit_dynamic_irtree(
    sim.transitions,
    ep.dynamic_irtree_spec(engine="multinomial"),
)
comparison = ep.compare_dynamic_transition_models(
    {"baseline": baseline, "multinomial": multinomial}
)

programme = ep.dynamic_irtree_recovery(
    grid=pd.DataFrame(
        {
            "state_misclassification": [0.0, 0.05, 0.15, 0.0, 0.05, 0.15],
            "missing_state": [0.0, 0.0, 0.0, 0.10, 0.10, 0.10],
        }
    ),
    replications=200,
)
```

Promotion requires recovery, coverage, state-error sensitivity, misspecification studies, grouped validation, engine comparison, and empirical reproduction.
