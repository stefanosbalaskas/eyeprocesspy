import numpy as np, pandas as pd
import eyeprocesspy as ep
rng=np.random.default_rng(10);t=np.arange(0,3000.1,20);k=ep.pupil_response_kernel(t-500)
d=pd.DataFrame({'person_id':'P1','trial_id':'T1','time_ms':t,'pupil_bc':.7*k+rng.normal(0,.02,len(t)),'event_ms':500})
f=ep.fit_pupil_event_deconvolution(d,events={'stimulus':'event_ms'})
assert ep.pupil_event_effects(f).beta__stimulus.iloc[0]>0
assert np.isfinite(ep.pupil_activity_index(d.pupil_bc,d.time_ms,50,method='frequency_contrast'))
