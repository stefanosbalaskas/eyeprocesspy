# IRT linking, DIF, DTF, and invariance evidence

Scale linking and invariance are represented as evidence components rather than binary declarations of validity.

```python
import numpy as np
import pandas as pd
import eyeprocesspy as ep

ref = pd.DataFrame({
    "item_id": [f"I{i}" for i in range(1, 9)],
    "a": np.linspace(.8, 1.5, 8), "b": np.linspace(-1.5, 1.5, 8),
    "c": 0.0, "d": 1.0,
})
A, B = 1.2, -.3
focal = ref.copy()
focal["a"] = ref["a"] * A
focal["b"] = (ref["b"] - B) / A
link = ep.eyeprocess_irt_mean_sigma_link(ref, focal)
linked = ep.eyeprocess_irt_apply_link(focal, link)
```

Native utilities include mean-sigma and mean-mean linking, Stocking-Lord and Haebara objectives, anchor screening/purification, DIF/DTF effect curves, anchor-set stability, session/device drift, and descriptive process-channel concordance. For the mature `equateIRT` engine, exact-method gating is preserved rather than replaced by a simpler estimator.
