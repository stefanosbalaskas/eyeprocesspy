# Multidimensional, testlet, and latent-regression IRT

The frozen 0.9 IRT layer adds transparent design and diagnostic support for multidimensional IRT without pretending to duplicate mature estimators.

```python
import numpy as np
import eyeprocesspy as ep

L = np.array([[1,0],[1,0],[0,1],[0,1],[1,0],[0,1]], dtype=float)
spec = ep.eyeprocess_mirt_loading_spec(
    [f"I{i}" for i in range(1, 7)], L,
    ["accuracy", "process"], simple_structure=True,
)
print(ep.eyeprocess_mirt_loading_audit(spec, min_items_per_dimension=2))
print(ep.eyeprocess_mirt_information_matrix([0, 0], [1, .5]))
```

Testlet declarations, directional information, latent-regression design matrices, and identification audits are native. Exact multidimensional, bifactor/two-tier, multiple-group, mixed and polytomous estimation remains attached to the explicit `mirt`/`TAM` engine contracts where the R package requires them.
