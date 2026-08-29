import numpy as np
import pandas as pd
import eyeprocesspy as ep

rng = np.random.default_rng(11)
theta = np.linspace(-2, 2, 60)
x = pd.DataFrame({
    "item1": np.clip(.5 + .12 * theta + rng.normal(0, .08, 60), 0, 1),
    "item2": np.clip(.4 + .18 * theta + rng.normal(0, .10, 60), 0, 1),
})
fit = ep.fit_censored_normal_process_irt(x, theta)
pred = ep.predict_eye_censored_normal_process_irt(fit, theta=[-1, 0, 1])
assert pred.shape == (3, 2)
