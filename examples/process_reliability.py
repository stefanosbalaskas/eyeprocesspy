"""Repeated-measure process reliability with ICC and Bland-Altman diagnostics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eyeprocesspy as ep

OUT = Path("workflow-output")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20260903)
persons = [f"P{i:02d}" for i in range(1, 41)]
latent = rng.normal(0, 1, len(persons))
rows = []
for pid, base in zip(persons, latent):
    rows.append({"person": pid, "session": "S1", "dwell_score": base + rng.normal(0, 0.25)})
    rows.append({"person": pid, "session": "S2", "dwell_score": base + 0.08 + rng.normal(0, 0.25)})
data = pd.DataFrame(rows)

profile = ep.process_reliability_profile(data, "person", "session", "dwell_score")
stability = ep.process_temporal_stability(data, "person", "session", "dwell_score")
print("ICC\n", profile["icc"].to_string(index=False))
print("\nBland-Altman\n", profile["bland_altman"]["summary"].to_string(index=False))
print("\nTemporal stability\n", stability.to_string(index=False))

axis = ep.plot_eye_process_reliability_profile(profile)
axis.figure.set_size_inches(7.2, 4.5)
axis.figure.tight_layout()
axis.figure.savefig(OUT / "process-reliability.svg", format="svg", bbox_inches="tight")
plt.close(axis.figure)

print("\nInterpretation boundary: reliability is design- and population-dependent and does not establish construct validity.")
