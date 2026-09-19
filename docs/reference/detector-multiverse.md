# Detector multiverse API

## Specification and execution

::: eyeprocesspy.define_event_detector_spec

::: eyeprocesspy.validate_event_detector_spec

::: eyeprocesspy.create_detector_multiverse

::: eyeprocesspy.detect_events_with_spec

::: eyeprocesspy.run_detector_multiverse

::: eyeprocesspy.import_external_detector_events

## Event comparison

::: eyeprocesspy.match_detected_events

::: eyeprocesspy.compare_event_catalogues

::: eyeprocesspy.estimate_detector_agreement

::: eyeprocesspy.summarise_detector_events

::: eyeprocesspy.summarise_detector_disagreement

## AOI and feature propagation

::: eyeprocesspy.propagate_detector_to_aoi

::: eyeprocesspy.propagate_detector_to_features

## Inference robustness

::: eyeprocesspy.run_detector_inference_multiverse

::: eyeprocesspy.assess_detector_inference_stability

::: eyeprocesspy.summarise_detector_robustness

## Inference result audit contract

`run_detector_inference_multiverse()` returns a `DetectorInferenceResult`. In addition to coefficients, warnings, failures, the model specification, and feature fingerprint, `input_audit` provides one row per planned detector specification.

| Field | Meaning |
| --- | --- |
| `input_rows` | Propagated feature rows available for the detector before AOI/model filtering |
| `aoi_selected_rows` | Rows remaining after an explicit target-AOI restriction |
| `quality_excluded_rows` | Rows removed by the declared `minimum_valid_fraction` |
| `outcome_missing_rows` | Rows with a non-finite declared model outcome |
| `model_rows_used` | Rows supplied to the estimator |
| `status` | `modelled`, `failed`, or `no_model_data` |

These fields are additive audit metadata; they do not select an estimator, impute missing outcomes, or alter units.

## Visualisation and reporting

::: eyeprocesspy.plot_detector_event_timeline

::: eyeprocesspy.plot_detector_agreement

::: eyeprocesspy.plot_detector_feature_distributions

::: eyeprocesspy.plot_detector_coefficient_stability

::: eyeprocesspy.plot_detector_multiverse

::: eyeprocesspy.report_detector_multiverse

## Synthetic example data

::: eyeprocesspy.simulate_detector_multiverse_data
