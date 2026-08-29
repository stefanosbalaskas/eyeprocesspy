# Cognitive diagnosis and Q-matrix governance

The cognitive-diagnosis layer supplies transparent Q-matrix audits, attribute-profile enumeration, and deterministic DINA ideal-response/probability calculations.

```python
import numpy as np
import eyeprocesspy as ep

Q = np.array([[1,0],[0,1],[1,1],[1,0]])
audit = ep.eyeprocess_cdm_qmatrix_audit(Q)
profiles = ep.eyeprocess_cdm_attribute_profiles(2)[["A1", "A2"]]
eta = ep.eyeprocess_cdm_dina_ideal_response(Q, profiles)
probability = ep.eyeprocess_cdm_dina_probability(eta)
```

These utilities do not replace full cognitive-diagnosis estimation, empirical Q-matrix validation, or model comparison. `fit_eyeprocess_gdina()` preserves the exact-engine boundary and gates cleanly when the frozen GDINA backend is unavailable. `plot_eye_cdm_qmatrix_audit()` visualizes declared attribute coverage; it does not establish substantive validity.
