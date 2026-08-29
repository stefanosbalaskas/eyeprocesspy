"""M3 four-channel pupil-aware measurement workflow."""
import numpy as np
import eyeprocesspy as ep

sim = ep.simulate_multimodal_m3(
    n_person=40, n_item=8, pupil_signal="informative",
    pupil_missingness="none", dropout=(0, 0, 0, 0), seed=20260815,
)
ident = ep.audit_multimodal_m3_identifiability(sim)
assert ident.supported
controls = ep.multimodal_m3_negative_controls(sim, seed=99)
assert "pupil_phase_randomized" in controls.datasets

d = sim.data.copy()
d["trajectory_score"] = np.linspace(-1, 1, len(d))
bridge = ep.multimodal_m3_functional_bridge(
    d, "trajectory_score", provenance="predeclared functional pupil score"
)
assert bridge.representation == "functional_score"
assert "does not claim" in bridge.boundary
