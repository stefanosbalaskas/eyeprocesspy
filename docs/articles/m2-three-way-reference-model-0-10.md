# M2 Three-Way Reference Model: Response, RT, and Gaze

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

M2 joins a Rasch scored-response channel, lognormal response time, and negative-binomial fixation count. The Python port preserves the Man–Harring–Zhan reference DOI and the rule that gaze is a process measurement rather than an automatic psychological label.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m2(n_person=40, n_item=8, seed=20260814)
spec = ep.multimodal_m2_spec()
audit = ep.audit_multimodal_m2_identifiability(sim)
assert audit.supported
```

`fit_multimodal_m2()` is deliberately CmdStanPy/CmdStan-only because the frozen R reference is Stan-based; no fallback estimator is substituted.

## Source structure retained

- **Scope**
- **Canonical specification**
- **Simulation**
- **Structural audit**
- **CmdStan fit**
- **Identification**
- **Missingness**
- **Interpretation boundary**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
