import numpy as np, pandas as pd, eyeprocesspy as ep
rng=np.random.default_rng(8)
d=pd.DataFrame({'person':np.repeat([f'P{i}' for i in range(8)],10),'item':np.tile([f'I{i}' for i in range(10)],8),'session':np.tile(np.repeat([1,2],5),8),'device':np.tile(['A','B'],40),'metric':rng.normal(size=80)})
g=ep.fit_process_gstudy(d,'metric',facets=['person','item','session','device'])
print(ep.process_variance_components(g))
print(ep.design_process_dstudy(g,items=[5,10,20],sessions=[1,2]).design_grid)
