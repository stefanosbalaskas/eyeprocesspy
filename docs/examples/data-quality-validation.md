# Worked example: six ways validation can fail

This synthetic 9-point example is deliberately designed to prevent a common mistake: equating accuracy with precision.

```python
import eyeprocesspy as ep

validation = ep.simulate_gaze_quality_calibration(
    seed=20260918,
    samples_per_target=18,
    nominal_sampling_hz=60,
)

quality = ep.create_gaze_quality_report(
    validation,
    by=["profile", "target_id"],
    valid="valid",
    missing_reason="missing_reason",
    nominal_sampling_hz=60,
    bcea_probability=0.68,
)
```

The six profiles are:

| Profile | What it demonstrates |
| --- | --- |
| `good_accuracy_good_precision` | centered, stable gaze |
| `poor_accuracy_good_precision` | systematic calibration offset with stable gaze |
| `good_accuracy_poor_precision` | centered mean with spatial noise |
| `poor_accuracy_poor_precision` | offset and noise together |
| `irregular_sampling` | timebase instability despite available gaze coordinates |
| `missingness` | blink/tracker-invalid loss runs |

## A review rule is not an exclusion rule

```python
reviewed = ep.create_gaze_quality_report(
    validation,
    by=["profile", "target_id"],
    valid="valid",
    thresholds={
        "accuracy_mean": {"max": 1.0},
        "valid_sample_fraction": {"min": 0.80},
    },
)

reviewed[["profile", "target_id", "quality_flags", "review_required"]]
reviewed.attrs["gaze_quality_provenance"]["automatic_exclusion"]
# False
```

## Visual diagnostics

```python
fig = ep.plot_gaze_quality_dashboard(quality)
```

The dashboard is descriptive. Inspect the underlying rows before deciding whether any quality dimension matters for the planned analysis.

## Failure case: mixed targets

Precision should be calculated within stable-target periods. If you request an accuracy report across multiple targets in one group, the canonical report adds `mixed_accuracy_targets` to `quality_flags`, prompting you to reconsider the grouping rather than silently accepting a pooled value.

## Failure case: missing samples

RMS-S2S never joins the sample before a missing observation to the sample after it. This avoids treating a loss gap as one ordinary successive-sample displacement.

## Sensitivity pattern

A strong workflow reports results under the pre-specified quality rule and then repeats the key analysis under plausible alternative review/exclusion decisions. The quality subsystem preserves the metadata needed for that audit rather than performing deletion itself.
\n\n## Continue with sensitivity analysis\n\nAfter inspecting the baseline quality report, use the [quality-rule sensitivity example](data-quality-sensitivity.md) to compare defensible review/exclusion specifications without deleting observations inside the quality subsystem.\n

## Continue with visual diagnostics

Use the [plot gallery](data-quality-plot-gallery.md) for focused accuracy, RMS-S2S, BCEA, sampling-interval, and dashboard examples. Then use the [quality-rule sensitivity example](data-quality-sensitivity.md) to compare defensible downstream decisions without deleting observations inside the quality subsystem.
