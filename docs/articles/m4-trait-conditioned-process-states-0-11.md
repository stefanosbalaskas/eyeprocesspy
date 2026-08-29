# Trait-Conditioned Latent Response-Process States

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

M4 extends M3 with a sequential latent response-process state. The state is a statistical configuration of process measurements, not a psychological construct. The scored-response Rasch equation remains state-independent; states shift RT, gaze, and pupil channels only.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m4(30,8,seed=20260820)
spec = ep.multimodal_m4_spec(n_states=2, trait_conditioning=("theta","tau"))
```

Posterior state probabilities—not hard MAP labels—are the primary inferential object.

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
