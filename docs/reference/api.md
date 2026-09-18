# API reference

The reference below is generated from the installed `eyeprocesspy` package. The function-level frozen-R mapping is maintained separately in `parity/PARITY_MATRIX.csv`.

## Standardized Data Quality

The vendor-neutral quality subsystem keeps measurement dimensions separate and exposes them through the public package namespace.

| Purpose | Public API |
| --- | --- |
| Input and unit validation | `validate_gaze_quality_inputs()` |
| Target-referenced accuracy | `compute_gaze_accuracy()` |
| Precision bundle | `compute_gaze_precision()` |
| RMS sample-to-sample precision | `compute_rms_s2s()` |
| Spatial SD precision | `compute_gaze_sd_precision()` |
| BCEA | `compute_bcea()` |
| Sampling intervals | `estimate_sampling_interval()` |
| Sampling jitter | `estimate_sampling_jitter()` |
| Effective sampling rate | `estimate_effective_sampling_rate()` |
| Valid-data fraction | `compute_valid_sample_fraction()` |
| Data loss and missing runs | `compute_gaze_data_loss()` |
| Spatial summary | `summarise_spatial_quality()` |
| Sampling summary | `summarise_sampling_quality()` |
| Canonical report | `create_gaze_quality_report()` |
| Session comparison | `compare_gaze_quality_sessions()` |
| Condition comparison | `compare_gaze_quality_conditions()` |
| Accuracy plot | `plot_gaze_accuracy()` |
| Precision plot | `plot_gaze_precision()` |
| BCEA plot | `plot_bcea()` |
| Sampling-interval plot | `plot_sampling_intervals()` |
| Quality dashboard | `plot_gaze_quality_dashboard()` |
| Manuscript reporting text | `report_gaze_quality()` |
| Reproducible 9-point fixture | `simulate_gaze_quality_calibration()` |

See the [Data Quality guide](../guides/data-quality.md), [worked 9-point validation](../examples/data-quality-validation.md), and [reporting guideline](../guides/data-quality-reporting.md) for equations, interpretation boundaries, failure cases, and manuscript guidance.

::: eyeprocesspy
    options:
      members: true
      inherited_members: false
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      members_order: source
      separate_signature: true
      show_signature_annotations: true
