# M2 Posterior Predictive Checks and Negative Controls

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

Evidence is channel-specific. Response, RT, and gaze should be checked separately, while within-item shuffles break person-process alignment without changing within-item marginal values.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m2(40, 8, seed=23)
ppc = ep.multimodal_m2_ppc(sim)
controls = ep.multimodal_m2_negative_controls(sim, seed=99)
```

A negative-control failure is evidence against interpreting apparent process gains as person-aligned measurement information.

## Source structure retained

- **Channel-specific model checks**
- **Alignment negative controls**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
