"""Deterministic synthetic demonstration of measurement-validity additions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("docs/assets/measurement-validity")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260927)

rows = []
for target_id, tx, ty in [("A", 0.0, 0.0), ("B", 1.0, 1.0), ("C", 2.0, 0.0)]:
    for k in range(20):
        pupil = 3.2 + 0.025 * k
        rows.append(
            {
                "target_id": target_id,
                "target_x": tx,
                "target_y": ty,
                "pupil": pupil,
                "gaze_x": tx + 0.10 + 0.05 * (pupil - 3.45) + rng.normal(0, 0.015),
                "gaze_y": ty - 0.05 - 0.03 * (pupil - 3.45) + rng.normal(0, 0.015),
            }
        )
validation = pd.DataFrame(rows)
field = ep.compute_spatial_error_field(validation, target_id="target_id")
ax = ep.plot_spatial_error_field(field)
ax.figure.tight_layout()
ax.figure.savefig(OUT / "spatial-error-field.svg")
plt.close(ax.figure)

dynamic = pd.DataFrame(
    [
        {"aoi": "target", "timestamp": 0.0, "shape": "rectangle", "x_min": 0.0, "x_max": 0.5, "y_min": 0.0, "y_max": 0.5},
        {"aoi": "target", "timestamp": 1.0, "shape": "rectangle", "x_min": 1.0, "x_max": 1.5, "y_min": 0.0, "y_max": 0.5},
    ]
)
samples = pd.DataFrame({"timestamp": np.linspace(0, 1, 80), "gaze_x": np.linspace(0.15, 1.35, 80), "gaze_y": 0.25})
assignment = ep.assign_dynamic_aoi(samples, dynamic)
ax = ep.plot_dynamic_aoi_alignment(assignment)
ax.figure.tight_layout()
ax.figure.savefig(OUT / "dynamic-aoi-alignment.svg")
plt.close(ax.figure)

known = ep.simulate_known_gaze_process(n_cycles=2, sampling_hz=300, seed=20260927)
pursuit = ep.detect_smooth_pursuits(
    known,
    method="directional",
    minimum_velocity=1.0,
    maximum_velocity=40.0,
    maximum_direction_change_deg=90,
    minimum_duration_ms=10,
)
ax = ep.plot_pursuit_velocity(pursuit)
ax.figure.tight_layout()
ax.figure.savefig(OUT / "pursuit-classification.svg")
plt.close(ax.figure)


def evaluator(frame: pd.DataFrame) -> dict[str, float]:
    x = pd.to_numeric(frame["gaze_x_deg"], errors="coerce")
    clean = pd.to_numeric(frame.get("gaze_x_deg_clean", x), errors="coerce")
    return {"mean_absolute_x_error": float(np.nanmean(np.abs(x - clean)))}


stress = ep.run_gaze_stress_suite(
    known,
    evaluator,
    corruption="drift",
    severities=(0, 0.25, 0.5, 0.75, 1.0),
)
ax = ep.plot_gaze_stress(stress, metric="mean_absolute_x_error")
ax.figure.tight_layout()
ax.figure.savefig(OUT / "gaze-stress.svg")
plt.close(ax.figure)
