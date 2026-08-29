"""Requested 0.7 IRT API-completion example."""
import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(7)
data = pd.DataFrame([
    (f"P{p}", f"I{i}") for p in range(1, 13) for i in range(1, 6)
], columns=["participant_id", "item_id"])
data["gaze_exposure"] = rng.exponential(size=len(data))
data["response"] = rng.binomial(1, 0.65, size=len(data)).astype(float)
data.loc[np.arange(2, len(data), 11), "response"] = np.nan

missingness = ep.fit_gaze_informed_missingness_irt(data)
assert missingness.status == "reference-diagnostic"
assert missingness.theta_source == "smoothed-person-score-proxy"

latent = rng.normal(size=100)
audit = ep.audit_latent_distribution(latent)
comparison = ep.compare_latent_distribution_models(latent)
assert np.isfinite(audit.loc[0, "normal_qq_correlation"])
assert len(comparison.comparison) == 3

sim = ep.simulate_from_model(lambda: {"data": [1, 2], "truth": {"mu": 0.5}})
truth = ep.extract_parameter_truth(sim)
assert truth.loc[0, "truth"] == 0.5
