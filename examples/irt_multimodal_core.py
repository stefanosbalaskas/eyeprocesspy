"""Unified multimodal IRT measurement contract and process-information example."""
import numpy as np
import eyeprocesspy as ep

sim = ep.simulate_multimodal_irt(n_person=30, n_item=8, seed=42)
audit = ep.audit_multimodal_measurement(sim.measurement)
assert audit.valid
validation = ep.validate_multimodal_irt(sim)
assert validation.valid

rng = np.random.default_rng(42)
baseline = rng.normal(size=(1000, 2))
augmented = rng.normal(scale=.8, size=(1000, 2))
information = ep.process_information(baseline, augmented)
assert (information.relative_variance_reduction > 0).all()
ablation = ep.ablate_multimodal_channels(sim.measurement)
assert "response+rt+gaze+pupil" in ablation.scenarios
