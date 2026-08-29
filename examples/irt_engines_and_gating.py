import numpy as np
import eyeprocesspy as ep
reg=ep.eyeprocess_irt_engine_registry()
assert "mirt" in set(reg.engine)
g=ep.fit_eyeprocess_gdina(np.array([[1,0],[0,1]]),np.ones((2,1)))
assert g.eyeprocess_class=="eye_gated_irt_engine" and g.fit is None
