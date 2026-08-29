# Manual Installation of Multimodal Backends

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

The core package imports without Stan or Bayesian dependencies. Use `multimodal_backend_status()` to inspect optional backend availability. Canonical M2/M3/M4 fitting requires the `stan` extra and a working CmdStan installation.

```python
import eyeprocesspy as ep
print(ep.multimodal_backend_status())
```

Missing heavy backends do not break unrelated import, harmonization, simulation, audit or plotting workflows.

## Source structure retained

- **Backend status**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
