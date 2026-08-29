"""Pre-action and decision-process feature example."""
import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(12)
rows = []
for person in ["P1", "P2"]:
    for trial in ["T1", "T2"]:
        for sample in range(1, 51):
            rows.append((person, trial, sample))
d = pd.DataFrame(rows, columns=["person_id", "trial_id", "sample"])
d["time_ms"] = d["sample"] * 20
d["response_time_ms"] = 1000
d["aoi"] = np.resize(["target", "distractor", "button", "text"], len(d))
d["pupil_bc"] = rng.normal(size=len(d))
d["blink"] = False

pre = ep.preaction_process_features(d, windows_ms=[500, 1000])
proxy = ep.addm_glam_proxy_features(d)
assert len(pre.data) > 0 and len(proxy.features) > 0

stability_input = pd.DataFrame({
    "feature": ["pupil_mean", "valid_gaze", "aoi_entropy"] * 3,
    "split": np.repeat([1, 2, 3], 3),
    "importance": rng.random(9),
})
stability = ep.process_feature_stability(stability_input, top_n=2)
assert "feature_family" in stability
