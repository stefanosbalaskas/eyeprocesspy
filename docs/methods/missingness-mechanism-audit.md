# Experimental missingness-mechanism audit (Python)

This source-checkout-only (not pip-installed) development, module-scoped utility is not a new stable root export. It addresses the 20 September 2026 methods briefing without claiming to implement PMSI or identify an MCAR/MAR/MNAR mechanism.

```python
from research.methods_briefing_2026.missingness_mechanism_audit import (
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


## Related: physical gaze-validation provenance (experimental)

The 17 September retinal-tracking paper motivates separating artificial-eye and human validation, relative trajectory precision, absolute gaze accuracy, and post-offset correction. This source-checkout-only helper **declares** reference metadata; it does not measure accuracy.

```python
from research.methods_briefing_2026.gaze_reference_provenance import (
    declare_gaze_validation_provenance,
)

reference = declare_gaze_validation_provenance(
    reference_type="human_validation_target",
    coordinate_space="absolute_gaze",
    reference_identifier="participant-9-point-2026",
    accuracy_deg=.65, precision_deg=.18,
    target_measurements_available=True,
)
```

Absolute accuracy requires a declared physical reference with actual target measurements. Relative-only evidence cannot be passed as absolute accuracy; precision is not interchangeable with accuracy. This complements the existing [spatial QC methods](../guides/data-quality.md), not a replacement for empirical calibration experiments.
