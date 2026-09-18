# Propagating AOI Uncertainty Into Statistical Models

The package can rerun the same model after each AOI perturbation through an explicit callback. It never chooses the estimator.

## Callback contract

Each successful branch returns at least `term`, `estimate`, `SE`, `CI_low`, `CI_high`, `p_value`, `model_converged`, and `N`. The framework adds `perturbation_id` and coefficient direction.

## Convergence and failures

A callback error is retained in the failure table. A row with `model_converged=False` remains visible but is excluded from converged-effect summaries. Neither is reclassified as a valid result.

## What to summarize

Prefer coefficient direction, median/range of estimates, interval widths, convergence frequency, and assignment stability. Do not reduce the analysis to how many branches cross a p-value threshold.

```python
inference = ep.assess_aoi_inference_stability(result, term="condition")
```

The same-sign proportion is descriptive sensitivity evidence, not the probability that an effect exists.

Continue to [Reporting AOI Robustness](reporting.md).