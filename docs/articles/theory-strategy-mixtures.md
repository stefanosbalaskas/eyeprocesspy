# Theory-constrained strategy mixtures

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/theory-strategy-mixtures.Rmd`.

## Prespecification before fitting

A strategy mixture is appropriate only when theory defines distinguishable process signatures before estimation. Data-derived classes must not be named after cognition merely because their means differ.

```python
import eyeprocesspy as ep

spec = ep.theory_strategy_spec(
    strategies={
        "analytic": {
            "prompt_dwell": 1.0,
            "evidence_dwell": 1.0,
            "option_switches": 0.5,
        },
        "heuristic": {
            "prompt_dwell": -0.5,
            "evidence_dwell": -0.8,
            "option_switches": -0.2,
        },
    },
    response="score",
    participant="participant_id",
    item="item_id",
    condition="condition",
    item_availability=availability,
    engine="stan",
    anchor_strength=3,
)

fit = ep.fit_theory_strategy_irt(trials, spec, seed=42)
```

## Classification uncertainty

```python
probability = ep.strategy_posterior_probabilities(fit)
ep.strategy_classification_uncertainty(fit, threshold=0.70)
ep.strategy_label_switching_diagnostics(fit)
```

Posterior probabilities and entropy are primary outputs. Modal assignment alone conceals uncertainty.

## Sensitivity and competing heterogeneity

```python
sensitivity = ep.strategy_aoi_sensitivity(
    {
        "primary_aoi": trials_primary,
        "expanded_aoi": trials_expanded,
    },
    spec,
    seed=42,
)

ep.compare_strategy_heterogeneity(fit)
```

The package compares discrete-mixture explanations with continuous process heterogeneity and requires external strategy manipulations before substantive class labels can be promoted. A mixture component is not a cognitive strategy merely because its process means resemble a verbal theory label.
