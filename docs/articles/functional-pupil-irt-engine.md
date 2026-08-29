# Functional pupil-IRT modelling

## Measurement stance

Pupil diameter is treated as a physiological time series. It is **not automatically labelled cognitive load, effort, surprise, or arousal**. `eyeprocesspy` preserves preprocessing choices, baseline uncertainty and optional nuisance adjustment before any substantive interpretation.

## Specification

```python
import eyeprocesspy as ep

spec = ep.functional_pupil_irt_spec(
    df=6,
    basis="natural_spline",
    response="score",
    engine="stan",
    alignment="event",
    event_time_column="event_time",
    latency_ms=200,
    baseline_window=(-500, 0),
    baseline_method="subtract",
    time_window=(-200, 2000),
    luminance_column="luminance",
    gaze_x_column="x",
    gaze_y_column="y",
    blink_column="blink",
    interpolated_column="interpolated",
    max_interpolated_fraction=0.20,
    ar1=True,
    participant_effect=True,
    item_effect=True,
)

prepared = ep.prepare_functional_pupil_data(pupil_trials, spec)
basis = ep.functional_pupil_basis(prepared, df=6)
fit = ep.fit_joint_functional_pupil_irt(pupil_trials, spec, seed=42)
```

The canonical Stan route uses the bundled `functional_pupil_irt.stan` program through CmdStanPy. It retains shared person/item effects, functional basis terms, optional nuisance channels and AR(1) structure. Exact R `lme4` and `brms` routes remain explicit backend boundaries rather than silent substitutions.

## Diagnostics and scalar comparisons

```python
parameters = ep.extract_functional_pupil_parameters(fit)
diagnostics = ep.functional_pupil_diagnostics(fit)
comparison = ep.compare_functional_scalar_models(fit)
```

Peak, AUC and mean-pupil summaries remain transparent baselines. Functional summaries are not assumed to be superior simply because they are higher-dimensional.

## Preprocessing sensitivity

```python
grid = ep.pupil_preprocessing_grid(
    baseline_windows=((-500, 0), (-200, 0)),
    latency_ms=(100, 200, 300),
    basis_df=(4, 6, 8),
    baseline_methods=("subtract", "percent"),
    max_interpolated_fraction=(0.10, 0.20),
)

sensitivity = ep.pupil_preprocessing_sensitivity(
    pupil_trials,
    grid=grid,
    base_spec=ep.functional_pupil_irt_spec(engine="two_stage_glm"),
)
ax = ep.plot_eye_functional_pupil_sensitivity(sensitivity)
```

Promotion requires recovery under autocorrelation, luminance confounding, blink/interpolation variation, baseline uncertainty and external experimental validation.
