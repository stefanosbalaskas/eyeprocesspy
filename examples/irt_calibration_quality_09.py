"""Calibration uncertainty and gaze data-quality example for eyeprocesspy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(9)
target_x = np.repeat([.2, .5, .8], 10)
target_y = np.repeat([.2, .5, .8], 10)
calibration = pd.DataFrame({
    "target_x": target_x,
    "target_y": target_y,
    "gaze_x": target_x + rng.normal(0, .01, len(target_x)),
    "gaze_y": target_y + rng.normal(0, .01, len(target_y)),
})
model = ep.calibration_error_model(calibration)
ellipse = ep.gaze_uncertainty_ellipse(model)

samples = pd.DataFrame({
    "timestamp_ms": np.arange(0, 600, 10),
    "gaze_x": np.clip(.5 + rng.normal(0, .08, 60), 0, 1),
    "gaze_y": np.clip(.5 + rng.normal(0, .08, 60), 0, 1),
    "valid": pd.Series([True] * 58 + [False, True], dtype="boolean"),
})
quality = ep.gaze_data_quality_profile(samples, valid="valid")

aois = pd.DataFrame({
    "aoi": ["left", "right"],
    "x_min": [0.0, .5], "x_max": [.499, 1.0],
    "y_min": [0.0, 0.0], "y_max": [1.0, 1.0],
})
probabilistic = ep.probabilistic_aoi_assignment(
    samples.iloc[:5], aois, model, draws=100, seed=4, min_probability=.5,
)
ax = ep.plot_eye_probabilistic_aoi_assignment(probabilistic)

assert len(ellipse) == 1
assert quality.table.effective_hz.iloc[0] > 0
assert len(probabilistic.assignments) == 5
assert ax.eyeprocess_plot_matrix.shape == (2, 5)
