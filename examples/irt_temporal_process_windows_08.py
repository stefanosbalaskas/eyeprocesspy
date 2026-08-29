import numpy as np, pandas as pd
import eyeprocesspy as ep
rng=np.random.default_rng(9); rows=[]
for p in ['P1','P2']:
 for tr in ['T1','T2']:
  for s in range(60): rows.append((p,tr,s,s*50))
d=pd.DataFrame(rows,columns=['person_id','trial_id','sample','time_ms']);d['pupil_bc']=np.sin(d.time_ms/600)+rng.normal(0,.03,len(d));d['x']=rng.normal(500,20,len(d));d['y']=rng.normal(400,20,len(d));d['aoi']=np.resize(['target','text','button'],len(d));d['valid_gaze_prop']=.95;d['valid_pupil_prop']=.95;d['blink']=False;d['trackloss']=False
w=ep.extract_process_windows(d,spec=ep.process_window_spec(1000,500,0,3000))
assert ep.validate_process_windows(w).valid.iloc[0]
traj=ep.aoi_trajectory_features(d,degree=2)
assert len(traj.features)==4
