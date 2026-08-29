# Item seeding, accessibility review, and presentation fairness

## Experimental item-parameter seeding

`fit_item_parameter_seed_model(..., engine="lm")` predicts cold-start item difficulty and discrimination from calibrated item-design/process features. The predictions are explicitly labelled non-operational.

```python
seed = ep.fit_item_parameter_seed_model(
    calibrated_items,
    predictors=["visual_density", "word_count"],
    engine="lm",
)
priors = ep.predict_item_parameter_priors(seed, candidate_items)
audit = ep.audit_candidate_item_bank(seed, candidate_items)
```

Predictions are screening priors only. They do not replace expert content review, accessibility/bias review, pilot testing, or formal IRT calibration. The optional R `ranger` implementation remains an explicit backend boundary rather than being silently replaced.

Presentation/accessibility workflows elsewhere in `eyeprocesspy` retain the same governance rule as the frozen R article: presentation patterns must not be converted into diagnoses such as dyslexia, ADHD, neurodivergence, or visual impairment.
