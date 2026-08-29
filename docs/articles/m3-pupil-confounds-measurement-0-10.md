# M3 pupil measurement: confounds, quality, and missingness

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

M3 keeps baseline, luminance, gaze position, quality, blink, interpolation and time-on-task nuisance measurements explicit. Missing nuisance values on observed-pupil rows are rejected rather than silently imputed.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m3(40,8,pupil_missingness="quality",seed=11)
audit = ep.audit_multimodal_m3_identifiability(sim)
```

Pupil responsivity is a statistical measurement dimension. Any substantive psychological interpretation requires independent theory and validation.

## Source structure retained

- **Pupil is a measurement, not a label**
- **Inspect the contract**
- **Missingness stress is part of the model evidence**
- **Device and session effects**
- **Practical rule**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
