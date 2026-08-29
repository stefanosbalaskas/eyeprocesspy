import numpy as np, pandas as pd
import eyeprocesspy as ep
rng=np.random.default_rng(8)
d=pd.DataFrame({'person_id':np.repeat([f'P{i}' for i in range(1,7)],5),'valid_gaze_prop':np.r_[np.repeat(.95,25),np.repeat(.55,5)],'valid_pupil_prop':rng.uniform(.8,.98,30),'missing_gaze':np.r_[np.zeros(25),np.ones(5)],'missing_pupil':0,'rt_ms':900,'blink_cluster_count':rng.poisson(1,30),'sampling_rate_hz':60})
a=ep.audit_biometric_preflight(d)
assert len(ep.preflight_decisions(a))==6
assert 'not be interpreted' in a.interpretation
