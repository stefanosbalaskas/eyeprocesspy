# Propagating Detector Choice to Statistical Inference

The inference stage applies **the same declared model specification** to every detector branch.

```python
model_spec = {
    "engine": "statsmodels_ols",
    "formula": "dwell_time_ms ~ C(condition_id) + C(participant_id)",
    "outcome": "dwell_time_ms",
    "aoi_id": "disclosure",
}

inference = ep.run_detector_inference_multiverse(
    result,
    model_spec,
    minimum_valid_fraction=0.5,
)
```

## Explicit engines only

The core does not silently choose a model. Supported routes are:

- `statsmodels_ols` for an explicit OLS formula;
- `statsmodels_mixedlm` for an explicit mixed model with an explicit `groups` column and optional `re_formula`;
- `callback` for a user-supplied specialist estimator returning the documented tidy contract.

Use specialist statistical packages for models that belong there. The callback route exists so detector propagation does not force a simplified internal estimator.

## Missing data

Formula engines are configured with `missing="raise"`. Missing predictors therefore cause an explicit branch failure instead of an unnoticed reduction in N. If complete-case analysis is intended, construct that analysis dataset explicitly before fitting and report the rule.

### Model-input audit

Outcome and quality attrition are also explicit. `run_detector_inference_multiverse()` returns `inference.input_audit`, one row per planned detector specification, with:

- `input_rows`: all propagated feature rows for that detector;
- `aoi_selected_rows`: rows remaining after an explicit `aoi_id` selection;
- `quality_excluded_rows`: rows removed by the declared `minimum_valid_fraction`;
- `outcome_missing_rows`: rows with a non-finite model outcome;
- `model_rows_used`: rows handed to the estimator;
- `status`: `modelled`, `failed`, or `no_model_data`.

Quality and non-finite-outcome exclusions also create `model_input` warning rows. The same audit counts are copied onto coefficient rows when a model is fitted and onto failure rows when fitting cannot proceed. A trial can therefore leave the model only with a recorded reason; missing outcomes are never silently converted to zero or silently discarded.

## Convergence

Each coefficient row stores `converged`, warnings and N. A non-converged branch is retained for diagnosis, but `assess_detector_inference_stability()` excludes it from coefficient-stability calculations.

## Stability is more than p-values

```python
stability = ep.assess_detector_inference_stability(
    inference,
    term="C(condition_id)[T.disclosure]",
    substantive_threshold=100,
    direction="above",
)
```

The summary includes the coefficient distribution, estimate range, same-sign proportion, common CI overlap, convergence rate, and—only when supplied explicitly—a substantive-threshold stability measure.

A p-value is still stored because it is part of many fitted-model summaries, but the robustness framework does not reduce scientific stability to a count of significant branches.

Next: [Reporting Detector Sensitivity](reporting-detector-sensitivity.md).
