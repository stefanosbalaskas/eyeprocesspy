# Visual-context and testlet IRT

Items displayed on the same screen, page, diagram, or stimulus can share presentation-context variance. `visual_context_registry()` records that design dependence explicitly rather than silently treating all items as independent.

```python
import eyeprocesspy as ep

registry = ep.visual_context_registry(item_metadata, item="item_id", context="screen_id")
```

The frozen R implementation of `fit_visual_context_irt()` uses `mirt` to fit a one-factor base model and a separate visual-context/testlet dimension. `eyeprocesspy` preserves that estimator identity: the exact fitting route is an explicit backend gate rather than a different estimator presented as equivalent. Registry construction and validation remain fully executable in Python.

A visual-context factor represents known shared presentation context. It is **not** automatically a second substantive ability, strategy, attention state, or cognitive trait.
