import numpy as np, pandas as pd, eyeprocesspy as ep
rng=np.random.default_rng(10); n=80
g=pd.DataFrame({'x':np.cumsum(rng.normal(0,.03,n)),'y':np.cumsum(rng.normal(0,.03,n)),'time':np.arange(n),'pupil':rng.normal(size=n),'duration':rng.exponential(200,n)})
r=ep.gaze_recurrence(g); print(r.summary)
fit=ep.fit_fixation_point_process(g,interaction='self_exciting',grid_size=8)
print(ep.diagnose_gaze_point_process(fit).summary)
