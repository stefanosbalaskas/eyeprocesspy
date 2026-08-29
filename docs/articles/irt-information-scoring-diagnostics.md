# IRT information, scoring, and diagnostics

This article is the Python counterpart of the frozen R article **IRT information, scoring, and diagnostics** in `eyeprocess` 0.11.1.

The native IRT layer provides transparent mathematical utilities for common dichotomous and polytomous families. Specialized estimators are never silently approximated when an exact external engine is required.

```python
import numpy as np
import pandas as pd
import eyeprocesspy as ep

items = pd.DataFrame({
    "item_id": [f"I{i}" for i in range(1, 9)],
    "a": np.linspace(.8, 1.5, 8),
    "b": np.linspace(-1.5, 1.5, 8),
    "c": 0.0,
    "d": 1.0,
})
theta = np.arange(-3, 3.01, .25)
info = ep.eyeprocess_irt_test_information(theta, items)
precision = ep.eyeprocess_irt_measurement_precision_profile(theta, items)
```

The same layer supplies EAP, MAP and ML ability scoring, conditional uncertainty, test and item information, bank targeting, residual fit, Q3/local-dependence summaries, and Infit/Outfit-style diagnostics. Person-fit quantities are model diagnostics; they must not be converted into claims about cheating, disengagement, diagnosis, or mental state.

For visual diagnostics use `plot_eye_irt_information_profile()` and `plot_eye_irt_q3_matrix()`. These plots describe the declared model and simulated/observed residual structure; they are not construct-validity evidence.
