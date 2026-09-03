"""Deterministic IRT diagnostic plotting examples."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("workflow-output")
OUT.mkdir(exist_ok=True)
theta = np.linspace(-3, 3, 121)
information = 4.8 * np.exp(-0.5 * (theta / 1.35) ** 2) + 0.35
profile = pd.DataFrame(
    {
        "theta": theta,
        "information": information,
        "conditional_sem": 1.0 / np.sqrt(information),
    }
)
item_fit = pd.DataFrame(
    {
        "item_id": [f"I{i}" for i in range(1, 13)],
        "infit": [0.92, 1.01, 0.97, 1.06, 1.11, 0.88, 1.03, 0.95, 1.08, 0.99, 1.14, 0.91],
        "outfit": [0.89, 1.04, 0.94, 1.10, 1.15, 0.84, 1.06, 0.92, 1.12, 0.97, 1.18, 0.88],
    }
)
dif = pd.DataFrame(
    {
        "theta": theta,
        "signed_difference": 0.11 * np.exp(-0.5 * ((theta - 0.4) / 1.1) ** 2) - 0.045,
    }
)

plots = {
    "irt-information": ep.plot_eye_irt_information_profile(profile),
    "irt-item-fit": ep.plot_eye_irt_item_fit(item_fit, statistic="infit"),
    "irt-dif": ep.plot_eye_irt_dif_curve(dif),
}
for name, axis in plots.items():
    axis.figure.set_size_inches(7.2, 4.5)
    axis.figure.tight_layout()
    axis.figure.savefig(OUT / f"{name}.svg", format="svg", bbox_inches="tight")
    plt.close(axis.figure)

print(profile.head().to_string(index=False))
print("\nitem fit\n", item_fit.to_string(index=False))
print("\nDIF maximum absolute difference:", float(dif["signed_difference"].abs().max()))
print("saved figures to:", OUT.resolve())
