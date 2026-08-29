import numpy as np
import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError

try:
    ep.fit_gaze_anchored_3pl_audit(np.ones((120, 6)))
except EyeProcessBackendError:
    pass
try:
    ep.bayesian_process_diagnostics_dashboard(object())
except EyeProcessBackendError:
    pass
