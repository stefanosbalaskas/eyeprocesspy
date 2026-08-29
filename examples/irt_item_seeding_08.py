import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(10)
d = pd.DataFrame({"visual_density": rng.random(30), "word_count": rng.uniform(20, 100, 30)})
d["irt_difficulty"] = .8 * d.visual_density + .01 * d.word_count + rng.normal(0, .1, 30)
d["irt_discrimination"] = 1.5 - .4 * d.visual_density + rng.normal(0, .1, 30)
fit = ep.fit_item_parameter_seed_model(d, predictors=["visual_density", "word_count"], engine="lm")
pred = ep.predict_item_parameter_priors(fit, d[["visual_density", "word_count"]].iloc[:3])
assert pred.operational_status.str.contains("not_operational").all()
audit = ep.audit_candidate_item_bank(fit, d[["visual_density", "word_count"]].iloc[:3])
assert "review_required" in audit.table
