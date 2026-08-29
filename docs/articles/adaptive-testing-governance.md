# Adaptive testing design and governance

Native adaptive-design utilities are intentionally auditable: item-bank declaration, maximum-information selection, exposure summaries, content-balance audits, stopping rules and step-by-step adaptive traces.

```python
import numpy as np
import pandas as pd
import eyeprocesspy as ep

items = pd.DataFrame({
    "item_id": [f"I{i}" for i in range(1, 7)],
    "a": [.8,1,1.5,1.1,.9,1.2],
    "b": np.linspace(-1.5, 1.5, 6), "c": 0.0, "d": 1.0,
})
bank = ep.eyeprocess_irt_item_bank(items, content=["A","A","B","B","C","C"])
selection = ep.eyeprocess_irt_item_selection(bank, theta=0)
stopping = ep.eyeprocess_irt_stopping_rule(10, se=.25)
```

For mature CAT simulation and constrained/shadow-testing designs, the frozen R package delegates to `catR` and `mirtCAT`. The Python parity API preserves those boundaries and does not silently substitute a different CAT engine.
