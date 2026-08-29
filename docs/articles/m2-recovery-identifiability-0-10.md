# M2 Recovery and Identifiability

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

Recovery is a prerequisite for promoting the M2 model beyond software demonstration. The default Python recovery interface returns a deterministic design and does not silently start expensive posterior sampling.

```python
import eyeprocesspy as ep
rec = ep.multimodal_m2_recovery(n_rep=10, n_person=100, n_item=10)
assert not rec.executed
```

Recovery does not establish construct validity or external validity; it tests whether the declared data-generating parameters can be recovered under known synthetic truth.

## Source structure retained

- **Why recovery precedes promotion**
- **Recovery experiment**
- **What recovery does not show**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
