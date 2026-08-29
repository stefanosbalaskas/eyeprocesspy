import numpy as np, pandas as pd
import eyeprocesspy as ep
d=pd.MultiIndex.from_product([['i1','i2'],[1,2,3]],names=['item_id','deployment_batch']).to_frame(index=False)
d['irt_difficulty']=[0,0,.1,.1,.8,.1]; d['irt_discrimination']=1.; d['valid_gaze_prop']=.95
fit=ep.audit_process_drift(d,metrics=['irt_difficulty','irt_discrimination','valid_gaze_prop'])
assert len(fit.table)==2
assert 'do not prove' in fit.caveat
