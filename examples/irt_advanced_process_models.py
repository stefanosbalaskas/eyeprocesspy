"""Advanced process-IRT reference workflows."""
import numpy as np
import pandas as pd
import eyeprocesspy as ep

seqs=[['stem','A','stem','B'],['stem','B','B'],['A','B','A']]
emb=ep.process_sequence_embedding(seqs,n=(1,2),dimensions=2)
assert emb.shape==(3,2)

reference=pd.DataFrame({'a':[1,1.2,.9],'b':[-1,0,1]})
new=pd.DataFrame({'a':[.9,1.1,.8],'b':[-.8,.2,1.2]})
link=ep.equate_irt_scales(reference,new,method='mean-sigma')
assert np.isfinite(link.A) and np.isfinite(link.B)

bank=pd.DataFrame({'item_id':['i1','i2','i3'],'a':[1,1.2,.9],'b':[-.5,0,.5],
                   'process_information':[.1,.2,.15]})
cat=ep.simulate_process_cat(bank,true_theta=.4,n_items=3,seed=3)
assert len(cat)==3
