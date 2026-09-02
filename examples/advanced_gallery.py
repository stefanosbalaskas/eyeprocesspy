"""Generate advanced eyeprocesspy reliability, calibration, AOI and IRT plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("gallery-output")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260903)


def save(ax, name: str) -> None:
    ax.figure.set_size_inches(7.2, 4.5)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(ax.figure)
    print(f"saved {OUT / f'{name}.svg'}")


# Repeated-measure process reliability.
persons = [f"P{i:02d}" for i in range(1, 31)]
latent = rng.normal(0, 1, len(persons))
rows = []
for pid, base in zip(persons, latent):
    rows.append({"person": pid, "session": "S1", "dwell_score": base + rng.normal(0, 0.28)})
    rows.append({"person": pid, "session": "S2", "dwell_score": base + 0.08 + rng.normal(0, 0.28)})
reliability_data = pd.DataFrame(rows)
profile = ep.process_reliability_profile(reliability_data, "person", "session", "dwell_score")
save(ep.plot_eye_process_reliability_profile(profile), "process-reliability")

# Calibration error and propagated uncertainty.
targets = np.repeat(np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.2, 0.8], [0.8, 0.8]]), 8, axis=0)
errors = rng.multivariate_normal([0.012, -0.008], [[0.00055, 0.00008], [0.00008, 0.00042]], size=len(targets))
calibration = pd.DataFrame(
    {
        "target_x": targets[:, 0],
        "target_y": targets[:, 1],
        "gaze_x": targets[:, 0] + errors[:, 0],
        "gaze_y": targets[:, 1] + errors[:, 1],
    }
)
model = ep.calibration_error_model(calibration)
save(ep.plot_eye_calibration_error_model(model), "calibration-error")

points = pd.DataFrame(
    {
        "gaze_x": [0.21, 0.27, 0.47, 0.51, 0.58, 0.72, 0.76, 0.45, 0.49, 0.54, 0.29, 0.67],
        "gaze_y": [0.27, 0.33, 0.30, 0.34, 0.35, 0.31, 0.36, 0.68, 0.72, 0.66, 0.29, 0.33],
    }
)
aois = pd.DataFrame(
    [
        {"aoi": "Headline", "x_min": 0.12, "x_max": 0.36, "y_min": 0.18, "y_max": 0.42},
        {"aoi": "Evidence", "x_min": 0.42, "x_max": 0.80, "y_min": 0.20, "y_max": 0.45},
        {"aoi": "CTA", "x_min": 0.38, "x_max": 0.62, "y_min": 0.58, "y_max": 0.82},
    ]
)
probabilistic = ep.probabilistic_aoi_assignment(points, aois, model, draws=180, seed=9, min_probability=0.45)
save(ep.plot_eye_probabilistic_aoi_assignment(probabilistic), "probabilistic-aoi")

# Timestamp irregularity audit.
timestamps = np.cumsum(np.r_[0, np.clip(rng.normal(16.67, 1.2, 239), 10, 25)])
samples = pd.DataFrame({"timestamp_ms": timestamps, "recording_id": "demo-001"})
audit = ep.audit_sampling_irregularity(samples, time="timestamp_ms", unit="ms", by="recording_id", cv_threshold=0.05)
save(ep.plot_eye_sampling_irregularity_audit(audit), "sampling-irregularity")

# IRT plotting surface using deterministic diagnostic tables.
theta = np.linspace(-3, 3, 121)
information = 4.8 * np.exp(-0.5 * (theta / 1.35) ** 2) + 0.35
irt_information = pd.DataFrame({"theta": theta, "information": information, "conditional_sem": 1 / np.sqrt(information)})
save(ep.plot_eye_irt_information_profile(irt_information), "irt-information")

item_fit = pd.DataFrame(
    {
        "item_id": [f"I{i}" for i in range(1, 13)],
        "infit": [0.92, 1.01, 0.97, 1.06, 1.11, 0.88, 1.03, 0.95, 1.08, 0.99, 1.14, 0.91],
        "outfit": [0.89, 1.04, 0.94, 1.10, 1.15, 0.84, 1.06, 0.92, 1.12, 0.97, 1.18, 0.88],
    }
)
save(ep.plot_eye_irt_item_fit(item_fit, statistic="infit"), "irt-item-fit")

signed_difference = 0.11 * np.exp(-0.5 * ((theta - 0.4) / 1.1) ** 2) - 0.045
save(ep.plot_eye_irt_dif_curve(pd.DataFrame({"theta": theta, "signed_difference": signed_difference})), "irt-dif")
