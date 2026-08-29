# External IRT engines and exact-method gating

This is the Python counterpart of the frozen R article **External IRT engines and exact-method gating**.

`eyeprocesspy` keeps estimator identity inspectable:

```python
import eyeprocesspy as ep
print(ep.eyeprocess_irt_engine_registry())
```

The frozen registry covers `mirt`, `TAM`, `GDINA`, `LNIRT`, `eRm`, `equateIRT`, `catR`, and `mirtCAT`. These names refer to the exact R engines used by `eyeprocess` 0.11.1. The Python parity layer does **not** treat similarly named Python packages as equivalent estimators.

When the exact frozen engine is unavailable, wrappers such as `fit_eyeprocess_mirt()` and `fit_eyeprocess_gdina()` return an `eye_gated_irt_engine` contract with `fit=None`. A wrong explicit engine raises immediately. This is a scientific-governance feature: estimator identity is part of the model specification.
