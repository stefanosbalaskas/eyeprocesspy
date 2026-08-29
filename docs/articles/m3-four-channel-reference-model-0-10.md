# M3: Four-channel response, RT, gaze, and pupil measurement

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

M3 adds a neutral pupil process channel to the M2 architecture. The reference pupil likelihood is a trial-level Gaussian summary with explicit nuisance adjustment, not a cognitive-load model.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m3(40, 8, pupil_missingness="none", dropout=(0,0,0,0), seed=20260815)
audit = ep.audit_multimodal_m3_identifiability(sim)
assert audit.supported
```

Canonical posterior fitting remains CmdStanPy/CmdStan-only. M3 is incremental to M0–M2 and should be evaluated by ablation, negative controls, recovery, and missingness stress.

## Source structure retained

- **Purpose**
- **Simulate and audit before fitting**
- **Fit with CmdStanR**
- **Relationship to M0-M2**
- **Evidence boundary**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
