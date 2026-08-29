import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(8)
n = 50
d = pd.DataFrame({"person_id": [f"P{i}" for i in range(n)], "theta": rng.normal(size=n),
                  "accuracy": rng.random(n), "dwell": rng.normal(700, 70, n), "entropy": rng.random(n),
                  "pupil": rng.normal(size=n), "validity": rng.uniform(.8, 1, n), "criterion": rng.normal(size=n)})
blocks = ep.process_feature_blocks(d, {"Psychometric": ["theta", "accuracy"], "Gaze": ["dwell", "entropy"],
                                       "Pupil": ["pupil"], "Quality": ["validity"]}, id="person_id")
mb = ep.fit_multiblock_process_map(blocks, engine="pca_block_scaled")
assert len(ep.multiblock_person_coordinates(mb)) == n
profiles = ep.fit_process_profile_mixture(d, ["dwell", "entropy", "pupil"], k=3, engine="kmeans_reference")
assert len(ep.process_profile_probabilities(profiles)) == n
valid = ep.audit_process_external_validity(d, "criterion", ["theta", "pupil", "dwell"])
assert np.isfinite(valid.incremental_r2)
