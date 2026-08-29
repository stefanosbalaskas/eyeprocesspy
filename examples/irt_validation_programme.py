"""Recovery, SBC, PPC and negative-control validation contracts."""
import numpy as np
import pandas as pd
import eyeprocesspy as ep

recovery=pd.DataFrame({
    'replicate':[1,2,3,4], 'parameter':['a']*4, 'truth':[1.0]*4,
    'estimate':[1.05,.95,1.02,.98], 'lower':[.5]*4, 'upper':[1.5]*4,
    'converged':[True]*4,
})
summary=ep.summarize_parameter_recovery(recovery)
assert summary.rmse.iloc[0] < .1
spec=ep.irt_validation_spec('demo',replications=4)
grade=ep.grade_model_evidence(summary,spec)
assert 'validation_evidence' in grade.grade

rng=np.random.default_rng(5)
data=pd.DataFrame({'y':rng.normal(size=30),'process':rng.normal(size=30),'person':np.repeat(np.arange(10),3)})
negative=ep.negative_control_process_test(data,'process',lambda d:float(d.y.corr(d.process)),within='person',permutations=20,seed=5)
assert 0 < negative.p_value <= 1
