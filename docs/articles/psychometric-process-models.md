# Psychometric and Process Models

> Python counterpart of the frozen `eyeprocess` 0.11.1 vignette `psychometric-process-models.Rmd`.

`eyeprocesspy` keeps the original separation between response measurement and process measurements. The legacy modelling API is retained because it connects canonical eye-process datasets to familiar IRT, response-time, DIF and process-informed regression workflows while making approximation and backend identity explicit.

## Response and response-time matrices

```python
import eyeprocesspy as ep

x = ep.simulate_eye_dataset(n_person=20, n_item=8, seed=1)
Y = ep.response_matrix(x)
RT = ep.response_time_matrix(x, log_transform=True)
aligned = ep.align_response_matrices(Y, RT)
```

Rows and columns retain participant and item identifiers. Duplicate participant-item observations are rejected by default rather than silently aggregated.

## Baseline IRT and response-time models

The transparent Python reference path for `fit_irt()` is `engine="rasch_glm"`, matching the frozen R approximation with participant and item fixed effects. The exact R `mirt` and `TAM` engines remain explicit backend gates. Likewise, `fit_accuracy_rt(engine="two_stage")` provides the frozen two-stage approximation while the `LNIRT` route is not silently replaced.

```python
rasch = ep.fit_irt(x, engine="rasch_glm")
items = ep.item_parameters(rasch)
persons = ep.person_scores(rasch)
fit = ep.model_fit_statistics(rasch)

two_stage = ep.fit_accuracy_rt(x, engine="two_stage")
```

## Process-informed IRT

`process_irt_spec()` records which gaze, pupil and biometric features enter a process-informed response model. These are **associational measurement covariates**, not automatic psychological constructs.

```python
spec = ep.process_irt_spec(
    gaze_features=["gaze_feature"],
    pupil_features=["pupil_feature"],
    estimand="association",
)
# fit = ep.fit_process_irt(x, spec, engine="glm")
```

The Python GLM path represents participant and item effects as fixed categorical effects, exactly reflecting the approximation warning in the frozen R implementation. `lme4` and `brms` model identities remain explicit gates.

## DIF, process factors and descriptive strategy mixtures

The legacy layer also retains logistic item-level DIF, PCA-based shared process factors and exploratory k-means strategy mixtures. Strategy classes are descriptive clusters; they must not be named as cognitive strategies without external validation.

## Missing process measurements

`model_missing_process()` models whether a process feature is observed, and `sensitivity_missing_process()` compares complete-case and median-indicator analyses. Missingness analysis does not prove that missingness is ignorable.

## Experimental status

These interfaces preserve the frozen package's experimental status. Gaze, pupil, RT and derived process states are observations or statistical summaries. They do not, by themselves, identify attention, cognitive load, effort, strategy, guessing, comprehension, emotion or misconduct.
