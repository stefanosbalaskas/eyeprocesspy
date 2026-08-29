import numpy as np
import eyeprocesspy as ep
L=np.array([[1,0],[1,0],[0,1],[0,1],[1,0],[0,1]],float)
spec=ep.eyeprocess_mirt_loading_spec([f"I{i}" for i in range(1,7)],L,["accuracy","process"],simple_structure=True)
audit=ep.eyeprocess_mirt_loading_audit(spec,min_items_per_dimension=2)
assert audit.meets_minimum.all()
