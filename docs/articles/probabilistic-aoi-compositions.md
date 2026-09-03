# Probabilistic AOIs and Compositional Attention

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/probabilistic-aoi-compositions.Rmd`.

This workflow replaces brittle inside/outside AOI assignment with probabilistic membership and then analyses AOI dwell allocation as a composition. AOI probabilities remain explicit, overlap and boundary ambiguity are audited, and downstream dwell, TTFF, transition, and entropy metrics can propagate assignment uncertainty through Monte Carlo draws.

```python
import pandas as pd
import eyeprocesspy as ep

aois = pd.DataFrame(
    {
        "aoi_id": ["prompt", "evidence", "options"],
        "xmin": [0.05, 0.35, 0.10],
        "xmax": [0.30, 0.90, 0.90],
        "ymin": [0.05, 0.05, 0.60],
        "ymax": [0.45, 0.45, 0.95],
    }
)

prob = ep.assign_aois_probabilistic(
    samples,
    aois,
    precision=(0.03, 0.04),
)
ep.plot_aoi_probability_map(prob)
ep.plot_aoi_boundary_risk(ep.audit_aoi_separation(prob))

uncertain_metrics = ep.propagate_aoi_uncertainty(
    prob,
    draws=500,
    time_col="time",
    duration_col="duration",
)
ep.plot_aoi_metric_uncertainty(uncertain_metrics)
```

For trial-level dwell totals, use log-ratio coordinates rather than entering several raw proportions from the same composition into an ordinary regression.

```python
composition = ep.derive_aoi_composition(
    trial_features,
    aois=("prompt_dwell", "evidence_dwell", "options_dwell"),
    id_cols=("person_id", "item_id", "condition"),
)

ilr = ep.transform_aoi_composition(composition, "ilr")
comparison = ep.compare_aoi_compositions(composition, group="condition")
ep.plot_aoi_ternary(composition)
ep.plot_aoi_variation_matrix(composition)
ep.plot_compositional_group_difference(comparison)
```

The probability model and zero-replacement method must be reported. Probabilistic assignment reduces false certainty; it does not remove calibration error, and compositional transformation does not create causal meaning for dwell allocation.
