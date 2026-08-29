import numpy as np
import pandas as pd
import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError

rng = np.random.default_rng(13)
cls = pd.DataFrame({"person_id": [f"P{i}" for i in range(20)], "class": ["A"] * 10 + ["B"] * 10})
proc = pd.DataFrame({"person_id": np.repeat(cls.person_id, 2), "dwell": rng.normal(size=40), "pupil": rng.normal(size=40)})
align = ep.map_latent_classes_to_process_profiles(cls, proc, process_features=["dwell", "pupil"])
assert len(align.summary) == 2
imp = ep.biometric_imputation_sensitivity(pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]}), ["a", "b"], methods=[])
assert len(imp.results) == 0
try:
    ep.fit_mixture_irt_process_classes(np.ones((100, 6)))
except EyeProcessBackendError:
    pass
