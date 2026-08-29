import numpy as np
import pandas as pd
import eyeprocesspy as ep

X = np.ones((20, 5))
models = [ep.fit_kde_latent_distribution_irt(X),
          ep.fit_persistence_gaze_diffusion_irt(pd.DataFrame({"y": range(5)})),
          ep.fit_nonignorable_missing_irt(pd.DataFrame({"y": range(5)}))]
assert all(m.status == "gated" for m in models)
d = pd.DataFrame({"person_id": [f"P{i}" for i in range(12)], "fold": np.repeat([1, 2, 3], 4), "x": range(12)})
contract = ep.prepare_structured_unstructured_process_features(d, fold="fold")
assert "training fold only" in contract.contract["leakage_rule"]
