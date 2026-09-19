# Data Quality API

This is the focused API entry point for standardized gaze measurement quality. The functions remain vendor-neutral; Gazepoint-specific adaptation belongs in `gp3tools`.

## Validation and canonical report

::: eyeprocesspy.validate_gaze_quality_inputs
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.create_gaze_quality_report
    options:
      show_root_heading: true
      show_source: false

## Accuracy and precision

::: eyeprocesspy.compute_gaze_accuracy
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.compute_rms_s2s
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.compute_gaze_sd_precision
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.compute_bcea
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.compute_gaze_precision
    options:
      show_root_heading: true
      show_source: false

## Sampling and data availability

::: eyeprocesspy.estimate_sampling_interval
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.estimate_sampling_jitter
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.estimate_effective_sampling_rate
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.compute_valid_sample_fraction
    options:
      show_root_heading: true
      show_source: false

::: eyeprocesspy.compute_gaze_data_loss
    options:
      show_root_heading: true
      show_source: false

## Interpretation and reporting

Use the [methodological guide](../guides/data-quality.md) for equations, units, appropriate use, and interpretation boundaries. The [worked validation example](../examples/data-quality-validation.md) demonstrates six synthetic quality profiles, while the [sensitivity example](../examples/data-quality-sensitivity.md) shows how to vary review rules without hiding exclusions. The [reporting guideline](../guides/data-quality-reporting.md) provides manuscript-ready items and limitations.

The broader package API also exposes `summarise_spatial_quality()`, `summarise_sampling_quality()`, session/condition comparison helpers, focused plots, `plot_gaze_quality_dashboard()`, `report_gaze_quality()`, and `simulate_gaze_quality_calibration()`.
