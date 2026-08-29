# Negative Controls and State-Count Sensitivity

> Python counterpart of the frozen `eyeprocess` 0.11.1 article. The R 0.11.1 source remains the scientific reference.

A credible state model should weaken when temporal structure is destroyed or when apparent states are generated only by nuisance/device structure.

```python
import eyeprocesspy as ep
sim = ep.simulate_multimodal_m4(30,8,seed=20260820)
controls = ep.multimodal_m4_negative_controls(sim, seed=20260821)
sensitivity = ep.multimodal_m4_sensitivity(sim, n_states=(1,2,3))
```

Neither K nor substantive state meaning is selected automatically.

## Interpretation boundary

These workflows describe measurement, model checking, and software-validation evidence. Process measurements and latent states are not automatically psychological constructs.
