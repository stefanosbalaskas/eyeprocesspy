import numpy as np, pandas as pd
import eyeprocesspy as ep
items=pd.DataFrame({"item_id":[f"I{i}" for i in range(1,6)],"a":1.,"b":np.linspace(-1,1,5),"c":0.,"d":1.})
design=ep.eyeprocess_irt_recovery_design(sample_size=[50],n_items=[5],missing_rate=[0],testlet_sd=[0],replications=1,seed=1)
assert design.attrs["eyeprocess_class"]=="eye_irt_recovery_design"
sbc=ep.run_eyeprocess_irt_ability_sbc(items,replications=20,posterior_draws=19,theta_grid=np.linspace(-5,5,201),seed=19)
assert sbc.eyeprocess_class=="eye_irt_sbc_evidence"
