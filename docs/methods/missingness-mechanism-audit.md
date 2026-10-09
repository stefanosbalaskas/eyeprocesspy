# Experimental missingness-mechanism audit (Python)

This development-only, module-scoped utility is not a new stable root export. It addresses the 20 September 2026 methods briefing without claiming to implement PMSI or identify an MCAR/MAR/MNAR mechanism.

```python
from eyeprocesspy.missingness_mechanism_audit import (
    profile_missingness_mechanism,
    compare_missingness_sensitivity,
)

profile = profile_missingness_mechanism(
    samples, observed_col="gaze_available",
    group_cols=("participant_id", "trial_id"), time_col="time_s",
)
scenario = compare_missingness_sensitivity(
    trials, outcome_col="pupil_change", observed_col="recorded",
    group_cols=("condition",), deltas=(-0.2, 0, 0.2),
)
```

## Claims and exclusions

- `summary` measures row availability by declared grouping; `gap_runs` reports consecutive **recorded missing rows**, not inferred absent samples.
- Timestamps must be finite and strictly increasing within each group. The tool does not guess native sampling rate from the device label.
- `scenario_mean` substitutes observed-group mean plus a user-supplied delta, in the outcome's units, for missing outcomes. It is descriptive pattern-mixture sensitivity, **not** an identified estimate of unseen physiology and not a replacement for a hierarchical outcome model.
- MCAR/MAR/MNAR is never inferred automatically; no automatic interpolation, sample exclusion, or claim of robustness.
- Subsequent qualification should compare gap profiles with controlled known-truth dropout and participant-clustered empirical data, then validate integration with inference.

See [mediation missingness and quality](../guides/multilevel-mediation/missingness-and-quality.md) and the [data-quality guide](../guides/data-quality.md).
