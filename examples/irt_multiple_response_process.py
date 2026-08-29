import numpy as np
import pandas as pd
import eyeprocesspy as ep

long = pd.DataFrame({
    "participant_id": ["p1"] * 4 + ["p2"] * 4,
    "item_id": ["item1"] * 8,
    "option_id": ["A", "B", "C", "D"] * 2,
    "selected": [True, False, True, False, False, True, False, True],
})
encoded = ep.encode_response_combinations(long)
assert set(encoded["response_combination"]) == {"A|C", "B|D"}

rng = np.random.default_rng(7)
r = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("ABCD"))
p = r.to_numpy() + rng.normal(scale=.3, size=r.shape)
aud = ep.audit_process_local_dependence(r, p)
assert len(aud.pairs) == 6
