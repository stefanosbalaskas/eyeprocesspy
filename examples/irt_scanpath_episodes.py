import numpy as np, pandas as pd, eyeprocesspy as ep
paths={'A':['stem','evidence','options'],'B':['stem','options','evidence'],'C':['stem','evidence','options','evidence']}
rep=ep.representative_scanpath(paths,method='consensus',distance='edit'); print(rep.summary)
rng=np.random.default_rng(11); d=pd.DataFrame({'pupil':np.r_[rng.normal(0,1,40),rng.normal(2,1,40),rng.normal(-1,1,40)],'gaze_velocity':np.r_[rng.normal(1,1,40),rng.normal(3,1,40),rng.normal(.5,1,40)]})
ep_obj=ep.label_process_episodes(ep.segment_process_episodes(ep.detect_process_changepoints(d,['pupil','gaze_velocity'],window=8)))
print(ep_obj.summary)
