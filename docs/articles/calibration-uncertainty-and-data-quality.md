# Calibration uncertainty and eye-tracking data quality

The frozen 0.9 measurement-quality programme makes calibration/validation error, successive-sample precision, effective sampling frequency, irregular sampling, and data loss visible rather than hiding them inside preprocessing. `eyeprocesspy` preserves that contract.

```python
import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(9)
target_x = np.repeat([.2, .5, .8], 10)
target_y = np.repeat([.2, .5, .8], 10)
cal = pd.DataFrame({
    "target_x": target_x,
    "target_y": target_y,
    "gaze_x": target_x + rng.normal(0, .01, 30),
    "gaze_y": target_y + rng.normal(0, .01, 30),
})
model = ep.calibration_error_model(cal)
ellipse = ep.gaze_uncertainty_ellipse(model)
ax = ep.plot_eye_calibration_error_model(model)
```

The empirical model is intentionally expressed in the same coordinate system as the downstream AOIs. It summarizes observed acquisition error; it is not a universal statement about tracker accuracy.

## Sampling and quality reporting

```python
samples = pd.DataFrame({
    "timestamp_ms": np.arange(0, 600, 10),
    "gaze_x": rng.normal(size=60),
    "gaze_y": rng.normal(size=60),
    "valid": pd.Series([True] * 58 + [False, True], dtype="boolean"),
})
quality = ep.gaze_data_quality_profile(samples, valid="valid")
report = ep.data_quality_reporting_table(quality)
```

Quality thresholds remain workflow-specific. A quality metric describes measurement conditions and should not become an automatic participant label.

## Propagating calibration uncertainty into AOIs

`propagate_calibration_uncertainty()` generates spatial uncertainty draws from the empirical calibration-error distribution. `probabilistic_aoi_assignment()` then evaluates rectangular AOI membership under those draws.

```python
aois = pd.DataFrame({
    "aoi": ["left", "right"],
    "x_min": [0.0, .5], "x_max": [.499, 1.0],
    "y_min": [0.0, 0.0], "y_max": [1.0, 1.0],
})
probabilistic = ep.probabilistic_aoi_assignment(
    samples.iloc[:5], aois, model, draws=100, seed=4
)
ax = ep.plot_eye_probabilistic_aoi_assignment(probabilistic)
```

These are **probabilities of spatial AOI membership under the fitted error model**. They are not probabilities of psychological attention, engagement, comprehension, or intent.

See the executable counterpart in `examples/irt_calibration_quality_09.py`.
