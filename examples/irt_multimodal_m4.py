"""M4 latent response-process state evidence-gating workflow."""
import eyeprocesspy as ep

sim = ep.simulate_multimodal_m4(n_person=30, n_item=10, n_states=2, seed=20260820)
states = ep.multimodal_m4_state_diagnostics(sim)
probability_cols = [c for c in states.probability if c.startswith("state_") and c.endswith("_probability")]
assert len(probability_cols) == 2

spec = ep.multimodal_m4_spec(
    n_states=2, transition_structure="markov", trait_conditioning=("theta", "tau")
)
audit = ep.audit_multimodal_m4_identifiability(sim, spec=spec, include_posterior=False)
assert audit.overall == "REVIEW"
controls = ep.multimodal_m4_negative_controls(sim, run=False)
assert not controls.executed
recovery = ep.multimodal_m4_recovery()
assert not recovery.executed
