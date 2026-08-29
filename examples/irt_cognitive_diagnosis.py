import numpy as np
import eyeprocesspy as ep
Q=np.array([[1,0],[0,1],[1,1],[1,0]])
a=ep.eyeprocess_cdm_qmatrix_audit(Q); assert a.complete_identity_block
pr=ep.eyeprocess_cdm_attribute_profiles(2)[["A1","A2"]]
p=ep.eyeprocess_cdm_dina_probability(ep.eyeprocess_cdm_dina_ideal_response(Q,pr)); assert np.all((p>=0)&(p<=1))
