# Specifying and Fitting the M4 Reference Model

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

The M4 reference model uses a marginalized finite-state Markov process over explicitly ordered participant/session sequences. K=1 is the formal null and K=2 is the conservative default.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m4(30,8,seed=20260820)
audit = ep.audit_multimodal_m4_identifiability(sim, include_posterior=False)
```

Canonical posterior inference requires CmdStanPy/CmdStan. Trait-conditioned Markov fitting is additionally REVIEW-gated until the declared validation evidence is satisfied.

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
