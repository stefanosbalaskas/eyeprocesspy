import numpy as np
import pandas as pd

import eyeprocesspy as ep

rng = np.random.default_rng(7)
d = pd.DataFrame({"dwell": rng.normal(1000, 100, 60), "pupil": rng.normal(0.1, 0.03, 60)})
u = ep.estimate_process_uncertainty(
    d, ep.process_uncertainty_spec(draws=50), metrics=["dwell", "pupil"]
)
print(ep.uncertainty_budget(u).head())
ref = pd.DataFrame({"target_x": rng.random(50), "target_y": rng.random(50)})
obs = ref.assign(x=lambda z: z.target_x + 0.04, y=lambda z: z.target_y - 0.02, time=np.arange(50))
drift = ep.detect_calibration_drift(obs, window=10, x_col="x", y_col="y", time_col="time")
model = ep.fit_offline_recalibration(drift, method="translation")
print(
    ep.audit_recalibration(
        obs, ep.apply_offline_recalibration(obs, model, x_col="x", y_col="y")
    ).summary
)
