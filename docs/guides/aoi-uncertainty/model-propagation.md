# Propagating AOI Uncertainty Into Statistical Models

The package can rerun the **same prespecified model** after each AOI perturbation through an explicit callback. It never chooses the estimator, family, covariates, random-effects structure, link, or exclusion rule for you.

## Callback contract

Each successful branch returns at least `term`, `estimate`, `SE`, `CI_low`, `CI_high`, `p_value`, `model_converged`, and `N`. `eyeprocesspy` adds `perturbation_id` and coefficient direction.

The callback contract is strict:

- `term` must be non-empty;
- `model_converged` must be boolean, `0`/`1`, or missing—strings such as `"yes"` are rejected;
- numeric fields must be numeric or missing;
- converged rows require finite `estimate`, `SE`, `CI_low`, `CI_high`, and a positive integer-valued `N`;
- a callback exception is retained as a model-stage failure.

This prevents a malformed or failed fit from being silently summarized as a valid branch.

## What to summarize

Use `assess_aoi_inference_stability()` to report more than direction alone:

```python
inference = ep.assess_aoi_inference_stability(result, term="condition")
```

For each term, inspect:

- number and proportion of converged branches;
- same-sign frequency relative to the converged baseline;
- median and range of coefficient estimates;
- median confidence-interval width;
- median and range of model `N`.

A stable coefficient with changing `N` is not the same analytical result as a stable coefficient with constant `N`. Report branch-specific case loss when it occurs.

## Do not use significance counts as the robustness result

A branch crossing `p = .05` can reflect interval width, model `N`, or numerical variation without a meaningful reversal of the estimated effect. Prefer magnitude, direction, uncertainty intervals, convergence, and assignment stability.

The same-sign proportion is descriptive sensitivity evidence, **not** the probability that an effect exists.

Continue to the [AOI robustness decision clinic](decision-clinic.md) and [Reporting AOI Robustness](reporting.md).
