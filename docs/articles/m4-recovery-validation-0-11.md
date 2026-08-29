# Recovery and Validation of Latent Response-Process States

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

The frozen M4 recovery programme is bounded to clear K=2, weak K=2, true K=1, trait-conditioned K=2, and nuisance/confounded structure. The default is inert.

```python
import eyeprocesspy as ep
rec = ep.multimodal_m4_recovery()
assert not rec.executed
assert len(rec.design) == 5
```

When posterior fits are available, state labels must be aligned before recovery evaluation and probability calibration is preferred to MAP accuracy alone.

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
