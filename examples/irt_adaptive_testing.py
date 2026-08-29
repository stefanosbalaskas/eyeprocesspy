import numpy as np, pandas as pd
import eyeprocesspy as ep
items=pd.DataFrame({"item_id":[f"I{i}" for i in range(1,7)],"a":[.8,1,1.5,1.1,.9,1.2],"b":np.linspace(-1.5,1.5,6),"c":0.,"d":1.})
bank=ep.eyeprocess_irt_item_bank(items,content=["A","A","B","B","C","C"])
sel=ep.eyeprocess_irt_item_selection(bank,theta=0); assert sel.selected in set(items.item_id)
assert ep.eyeprocess_irt_stopping_rule(10,se=.25)["stop"]
