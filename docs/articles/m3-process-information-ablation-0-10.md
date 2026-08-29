# M3 process information: ablation, redundancy, and sensor value

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

The full response-anchored M3 lattice distinguishes RT, gaze, and pupil contributions rather than treating “more sensors” as automatically better. The current Python interface prepares the M0–M3 scenarios and reports design-level pupil availability/cost while posterior comparisons remain backend-explicit.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m3(40,8,seed=9)
abl = ep.multimodal_m3_ablation(sim)
info = ep.multimodal_m3_process_information(sim, pupil_cost=1.0)
```

## Source structure retained

- **Why eight models?**
- **Fit the ablation lattice**
- **Incremental pupil evidence**
- **Redundancy and complementarity**
- **Sensor value and channel conflict**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
