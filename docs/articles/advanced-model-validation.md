# Advanced model programme and validation

The advanced functions are model families with explicit validation obligations. Execution alone does not make them confirmatory.

## Dynamic gaze-state IRTree baseline

`dynamic_irtree_spec()` and the dynamic-state engine preserve observed transition order, item/person structure and uncertainty. AOI states remain observational unless externally validated as substantive states.

## Functional pupil-informed IRT

```python
spec = ep.functional_pupil_irt_spec(df=5, engine="two_stage_glm")
pupil_fit = ep.fit_joint_functional_pupil_irt(dataset, spec)
```

The functional-pupil engine is a physiological measurement model; its basis coefficients are not automatic cognitive-load indicators.

## Theory-defined strategies and gaze diffusion

Theory-defined strategy mixtures and gaze-diffusion models retain their explicit validation and construct-naming boundaries. Descriptive clusters or latent states are not automatically psychological strategies.

## Monte Carlo design

```python
grid = ep.advanced_validation_grid(quick=True)
scenario = grid.iloc[0].to_dict()
simulation = ep.simulate_advanced_process_data(**scenario, seed=20260804)
```

The design varies sample size, item count, ability-speed correlation, process effects, feature reliability, process missingness, AOI-state error, pupil autocorrelation, luminance confounding, DIF and local dependence. Production work should use the full design or a preregistered subset, sufficient replications, interval coverage, expected-failure scenarios and grouped person/item validation.

## Evidence promotion gate

A model is promoted only when the declared evidence programme is actually satisfied: recovery, calibration, misspecification, grouped validation, engine-equivalence where relevant, empirical reproduction and sensitivity analysis. Python-only unit tests do not substitute for this evidence.
