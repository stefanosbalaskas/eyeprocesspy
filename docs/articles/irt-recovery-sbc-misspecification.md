# IRT recovery, SBC, and misspecification evidence

Parameter recovery and simulation-based calibration (SBC) answer different software-validation questions. Recovery asks whether an estimator reproduces known generating quantities under declared scenarios. SBC checks calibration of posterior computation under the declared generative model.

```python
import eyeprocesspy as ep

design = ep.eyeprocess_irt_recovery_design(
    sample_size=[250], n_items=[12],
    missing_rate=[0, .15], testlet_sd=[0, .35],
    replications=5, seed=20260811,
)
print(design.head())
print(ep.eyeprocess_irt_misspecification_suite())
```

Exact parameter-recovery fitting in the frozen reference requires `mirt`; therefore the Python core returns an explicit gated result when that exact engine is unavailable. Known-item ability SBC is executable natively and returns `eye_irt_sbc_evidence`. `plot_eye_irt_sbc_evidence()` visualizes the rank diagnostic. SBC is computational-calibration evidence, not evidence of empirical model adequacy or construct validity.
