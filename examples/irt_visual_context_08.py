import numpy as np
import pandas as pd
import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError

meta = pd.DataFrame({"item_id": [f"i{i}" for i in range(1, 9)],
                     "screen_id": ["screen_A"] * 4 + [f"unique_{i}" for i in range(5, 9)]})
registry = ep.visual_context_registry(meta, context="screen_id")
assert registry.mapping.shared_context.sum() == 4
X = pd.DataFrame(np.ones((120, 8)), columns=meta.item_id)
try:
    ep.fit_visual_context_irt(X, registry)
except EyeProcessBackendError:
    pass
