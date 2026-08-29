# M3 Device Transport and Sensor Value

> Python counterpart of the frozen `eyeprocess` 0.11.1 article.

M3 evidence includes device/session transport rather than assuming that a pupil channel has the same measurement behavior across acquisition systems. The Python port combines the existing device-linking/equivalence tools with the M3 audit and process-information contracts.

```python
import eyeprocesspy as ep

sim = ep.simulate_multimodal_m3(40, 8, seed=20260815)
audit = ep.audit_multimodal_m3_identifiability(sim)
info = ep.multimodal_m3_process_information(sim)
```

Device effects, missingness, calibration and sensor costs are measurement issues. A sensor that improves an in-sample process fit is not automatically transportable or substantively valid.

## Interpretation boundary

Pupil and other sensor channels remain observations. Cross-device agreement and incremental measurement information must be demonstrated rather than inferred from the existence of a fitted model.
