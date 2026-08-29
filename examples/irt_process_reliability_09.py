"""Process-measure registry and reliability example for eyeprocesspy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(9)
people = np.arange(1, 31)
trait = rng.normal(size=len(people))
rows = []
for i, person in enumerate(people):
    for session in (1, 2):
        rows.append({
            "person_id": person,
            "session": session,
            "dwell_time": 850 + 130 * trait[i] + rng.normal(scale=30),
        })
repeated = pd.DataFrame(rows)

registry = ep.process_measure_registry()
card = ep.process_measure_card("pupil_peak")
icc = ep.process_icc(repeated, "person_id", "session", "dwell_time")
profile = ep.process_reliability_profile(repeated, "person_id", "session", "dwell_time")
ax = ep.plot_eye_process_reliability_profile(profile, type="bland_altman")

assert "pupil_peak" in set(registry.name)
assert card.guardrail
assert np.isfinite(icc.icc_a1.iloc[0])
assert len(ax.eyeprocess_plot_data) == len(people)
