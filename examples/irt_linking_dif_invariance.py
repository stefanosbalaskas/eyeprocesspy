import numpy as np, pandas as pd
import eyeprocesspy as ep
ref=pd.DataFrame({"item_id":[f"I{i}" for i in range(1,9)],"a":np.linspace(.8,1.5,8),"b":np.linspace(-1.5,1.5,8),"c":0.,"d":1.})
A,B=1.2,-.3; focal=ref.copy(); focal["a"]=ref.a*A; focal["b"]=(ref.b-B)/A
link=ep.eyeprocess_irt_mean_sigma_link(ref,focal)
assert np.isclose(link.A,A) and np.isclose(link.B,B)
