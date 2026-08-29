"""Response-anchored process channel ablation example."""
import eyeprocesspy as ep
sim = ep.simulate_multimodal_irt(n_person=30, n_item=8, seed=7)
abl = ep.ablate_multimodal_channels(sim.measurement)
assert "response" in abl.scenarios
assert "response+rt+gaze+pupil" in abl.scenarios
