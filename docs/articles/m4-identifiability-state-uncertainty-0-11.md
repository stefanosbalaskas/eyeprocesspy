# Identifiability, State Separation, and Label Uncertainty

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

State labels are anchored through ordered RT deviations only as an identification convention. This ordering does not order psychological meaning.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m4(30,8,scenario="weak",seed=20260820)
audit = ep.audit_multimodal_m4_identifiability(sim, include_posterior=False)
states = ep.multimodal_m4_state_diagnostics(sim)
```

Entropy, occupancy, transition structure, emission separation and contextual associations should be inspected together before substantive interpretation.

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
