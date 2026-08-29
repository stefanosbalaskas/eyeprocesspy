"""M2 response + response-time + gaze reference workflow."""
import eyeprocesspy as ep

sim = ep.simulate_multimodal_m2(n_person=40, n_item=8, seed=20260814)
spec = ep.multimodal_m2_spec()
assert spec.model == "M2"
ident = ep.audit_multimodal_m2_identifiability(sim)
assert ident.supported
controls = ep.multimodal_m2_negative_controls(sim, seed=99)
assert len(controls.datasets) == 4
recovery = ep.multimodal_m2_recovery(n_rep=3, n_person=40, n_item=8)
assert not recovery.executed
# fit_multimodal_m2() intentionally requires the canonical CmdStan backend.
