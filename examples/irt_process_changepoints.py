import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(4)
rows = []
for order in range(1, 31):
    shifted = order > 15
    rows.append({
        "participant_id": "p1",
        "item_order": order,
        "response": rng.binomial(1, .8 if not shifted else .3),
        "rt": rng.lognormal(1.0 if not shifted else 1.7, .12),
        "fixation_count": rng.poisson(3 if not shifted else 8),
    })
d = pd.DataFrame(rows)
cp = ep.detect_irt_changepoints(d, gaze="fixation_count", min_segment=5, min_delta_sic=0)
assert len(cp.results) == 1
