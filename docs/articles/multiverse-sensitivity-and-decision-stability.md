# Sensitivity, specification curves, and decision stability

Reasonable preprocessing and modelling choices can change numerical results. `eyeprocesspy` therefore represents a defensible specification set explicitly and evaluates each branch under the same extraction contract.

```python
import eyeprocesspy as ep

grid = ep.process_sensitivity_grid(
    fixation_min_ms=(60, 80, 100),
    pupil_interpolation=("none", "linear", "spline"),
    max_invalid_trial=(.2, .3, .4),
)
result = ep.run_process_sensitivity(data, grid, analysis_fun, extract_fun)
summary = ep.summarise_process_sensitivity(result, p_value="p_value")
stability = ep.decision_stability(result, p_value="p_value")
leverage = ep.sensitivity_decision_leverage(result)
ax = ep.plot_eye_process_sensitivity(result, lower="lower", upper="upper")
```

A stable declared multiverse does not validate specifications that were never included. Stability thresholds are reporting conventions, not universal validity cutoffs. `analysis_decision_entropy()` and `decision_space_coverage()` make the declared decision space and its evaluated fraction explicit.
