import numpy as np, pandas as pd
import eyeprocesspy as ep
items=pd.DataFrame({"item_id":[f"I{i}" for i in range(1,9)],"a":np.linspace(.8,1.5,8),"b":np.linspace(-1.5,1.5,8),"c":0.,"d":1.})
theta=np.arange(-3,3.01,.25)
info=ep.eyeprocess_irt_test_information(theta,items)
assert info.attrs["eyeprocess_class"]=="eye_irt_information_profile"
score=ep.eyeprocess_irt_eap_score([1,1,1,1,0,0,0,0],items)
assert np.isfinite(score.estimate)
