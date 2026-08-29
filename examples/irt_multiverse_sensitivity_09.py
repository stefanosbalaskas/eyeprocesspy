"""Defensible multiverse and decision-stability example."""
import pandas as pd
import eyeprocesspy as ep

source = pd.DataFrame({"y": [1., 2., 3.]})
grid = ep.process_sensitivity_grid(method=("a", "b", "c"))
effects = {"a": .2, "b": .3, "c": .1}
result = ep.run_process_sensitivity(source, grid, lambda data, spec: effects[spec.method.iloc[0]])
summary = ep.summarise_process_sensitivity(result)
stability = ep.decision_stability(result)
assert summary.specifications.iloc[0] == 3
assert stability.stable_sign is True
