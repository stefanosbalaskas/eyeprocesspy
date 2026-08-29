# M3 functional pupil bridge: from trajectories to joint measurement

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

The frozen M3 likelihood uses one pupil value per trial. `multimodal_m3_functional_bridge()` allows a preregistered trajectory-derived score to enter that scalar contract while retaining provenance.

```python
import numpy as np
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m3(25,6,seed=7)
d = sim.data.copy()
d["trajectory_score"] = np.linspace(-1,1,len(d))
bridge = ep.multimodal_m3_functional_bridge(d, "trajectory_score", provenance="predeclared trajectory basis")
```

The bridge does **not** claim that a functional score is equivalent to raw pupil diameter, nor that either measure denotes cognitive load, effort, or arousal.

## Source structure retained

- **Why the first M3 likelihood is scalar**
- **What the bridge does not do**
- **Future full functional likelihood**

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
