# Worked example: sample-level AOI sensitivity

The AOI uncertainty engine supports both fixation-centroid and sample-level gaze input. This compact example shows the semantic difference.

```python
result = ep.run_aoi_sensitivity_analysis(
    samples,
    aois,
    grid,
    x_col="gaze_x",
    y_col="gaze_y",
    participant_col="participant",
    trial_col="trial",
    duration_col="sample_duration",
    time_col="time_seconds",
    observation_level="sample",
)
```

## What changes at sample level

The geometry and assignment calculations are identical. The feature labels are not:

- `observation_count` is always populated when the group has valid assignments;
- `sample_count` is populated for sample-level input;
- `fixation_count` remains missing;
- `first_observation` contains the first assigned sample time;
- `first_fixation` remains missing.

This prevents a high-frequency sample stream from being described as a sequence of fixations.

## Zero versus missing

If a participant/trial contains valid gaze samples but none enter the claim AOI, the claim row has `sample_count = 0`. If every coordinate in that participant/trial is missing, the count remains missing rather than becoming zero.

## When to prefer sample-level sensitivity

Use sample-level perturbation when your analysis is defined on raw/cleaned gaze samples or when you want to isolate geometry sensitivity from event-detector sensitivity. If your substantive variables are fixation dwell/count/first fixation, run the primary analysis at fixation level and examine detector uncertainty separately.

Continue to [Interpreting AOI Assignment Stability](../guides/aoi-uncertainty/assignment-stability.md), the [decision clinic](../guides/aoi-uncertainty/decision-clinic.md), and the [API reference](../reference/aoi-perturbation.md).
