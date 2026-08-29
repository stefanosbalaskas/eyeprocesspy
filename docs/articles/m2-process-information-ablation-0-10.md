# M2 Process Information and Channel Ablation

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

M2 process information is evaluated relative to the common scored-response target rather than by assuming that every additional channel is useful. Response-only, response+RT, and response+RT+gaze scenarios are retained explicitly.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m2(40, 8, seed=5)
abl = ep.multimodal_m2_ablation(sim)
info = ep.multimodal_m2_process_information(sim)
```

Information gains need not be additive: RT and gaze can be redundant, complementary, or weakly informative.

## Source structure retained

- **A common response target**
- **Two complementary quantities**
- **Non-additivity**
- **Interpretation**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
