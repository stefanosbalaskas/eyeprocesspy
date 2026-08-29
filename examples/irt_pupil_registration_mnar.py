import numpy as np, pandas as pd, eyeprocesspy as ep
rng=np.random.default_rng(9); rows=[]
for i in range(8):
    t=np.linspace(0,2,40); rows.append(pd.DataFrame({'person_id':f'P{i}','time':t,'pupil':np.exp(-((t-(.8+i/100))**2)/.08)+rng.normal(0,.02,40)}))
reg=ep.register_pupil_curves(pd.concat(rows),'time','pupil')
print(ep.decompose_pupil_phase_amplitude(reg,components=2).scores.head())
values=rng.normal(size=50); values[rng.random(50)<.2]=np.nan
print(ep.sensitivity_mnar_process(ep.process_pattern_mixture(values,delta=[-1,0,1])).summary)
