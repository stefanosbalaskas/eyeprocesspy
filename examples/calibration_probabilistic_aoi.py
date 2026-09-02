"""Calibration-error propagation and probabilistic AOI assignment."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("workflow-output")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260903)

targets = np.repeat(
    np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.8, 0.8]]),
    10,
    axis=0,
)
errors = rng.multivariate_normal(
    [0.012, -0.008],
    [[0.00055, 0.00008], [0.00008, 0.00042]],
    size=len(targets),
)
calibration = pd.DataFrame(
    {
        "target_x": targets[:, 0],
        "target_y": targets[:, 1],
        "gaze_x": targets[:, 0] + errors[:, 0],
        "gaze_y": targets[:, 1] + errors[:, 1],
    }
)

model = ep.calibration_error_model(calibration)
ellipse = ep.gaze_uncertainty_ellipse(model, level=0.95)
print("calibration metrics\n", model["metrics"].to_string(index=False))
print("\n95% uncertainty ellipse\n", ellipse.to_string(index=False))

points = pd.DataFrame(
    {
        "gaze_x": [0.21, 0.28, 0.46, 0.52, 0.60, 0.73, 0.51],
        "gaze_y": [0.28, 0.32, 0.31, 0.35, 0.37, 0.32, 0.70],
    }
)
aois = pd.DataFrame(
    [
        {"aoi": "Headline", "x_min": 0.12, "x_max": 0.36, "y_min": 0.18, "y_max": 0.42},
        {"aoi": "Evidence", "x_min": 0.42, "x_max": 0.80, "y_min": 0.20, "y_max": 0.45},
        {"aoi": "CTA", "x_min": 0.38, "x_max": 0.62, "y_min": 0.58, "y_max": 0.82},
    ]
)

assignment = ep.probabilistic_aoi_assignment(
    points,
    aois,
    model,
    draws=400,
    seed=9,
    min_probability=0.50,
)
print("\nassignments\n", assignment["assignments"].to_string(index=False))

for name, axis in {
    "calibration-error": ep.plot_eye_calibration_error_model(model),
    "probabilistic-aoi": ep.plot_eye_probabilistic_aoi_assignment(assignment),
}.items():
    axis.figure.set_size_inches(7.2, 4.5)
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(axis.figure)

print("\nImportant: membership probabilities quantify propagated calibration uncertainty; they are not probabilities of psychological attention.")
