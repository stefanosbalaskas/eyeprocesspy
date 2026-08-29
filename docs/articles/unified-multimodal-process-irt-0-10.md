# Unified Multimodal Process-IRT

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

The unified 0.10 contract brings response, RT, gaze and pupil into one vendor-neutral measurement object with explicit keys, availability, missingness, provenance and optional time-series payloads.

```python
import eyeprocesspy as ep
sim=ep.simulate_multimodal_irt(30,8,seed=42)
audit=ep.audit_multimodal_measurement(sim.measurement)
assert audit.valid
```

The architecture harmonizes measurement semantics without assigning gaze or pupil observations an automatic psychological interpretation.

## Source structure retained

- **Objective**
- **Native visualization**
- **Interpretation boundary**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
