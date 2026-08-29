"""Executable functional pupil-IRT example."""
import numpy as np
import pandas as pd
import eyeprocesspy as ep

rows=[]
for pi in range(5):
    for ji in range(3):
        score=(pi+ji)%2
        for s in range(12):
            rows.append({
                "participant_id":f"P{pi+1}","item_id":f"I{ji+1}","trial_id":f"P{pi+1}-I{ji+1}",
                "time_ms":800+s*100,"event_time":1200,"pupil":3+0.02*s+0.08*score+0.01*pi,
                "score":score,"response_time":1.0+0.03*ji,
            })
data=pd.DataFrame(rows)
spec=ep.functional_pupil_irt_spec(df=3,engine="two_stage_glm",alignment="event",event_time_column="event_time",latency_ms=0,baseline_window=(-400,0),pupil_column="pupil",time_column="time_ms")
prepared=ep.prepare_functional_pupil_data(data,spec)
fit=ep.fit_joint_functional_pupil_irt(data,spec)
parameters=ep.extract_functional_pupil_parameters(fit)
diagnostics=ep.functional_pupil_diagnostics(fit)
assert len(prepared.trials)==15
assert len(parameters)>0
assert len(diagnostics.checks)>0
