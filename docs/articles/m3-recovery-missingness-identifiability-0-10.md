# M3 recovery, identifiability, and failure-case validation

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

M3 validation deliberately includes informative, weak, null, redundant and confounded pupil scenarios crossed with missingness stress.

```python
import eyeprocesspy as ep
rec = ep.multimodal_m3_recovery(reps=3)
assert not rec.executed
```

A scientifically useful negative result is one where the pipeline correctly refuses or weakens claims under null, confounded, poorly connected, or heavily missing designs.

## Source structure retained

- **Validation must include failure cases**
- **Recovery programme**
- **Identifiability boundaries**
- **What counts as a successful negative result?**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
