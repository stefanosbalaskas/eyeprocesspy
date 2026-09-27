# Measurement validity and dynamic-gaze API

## Calibration and pupil artefact

- compute_spatial_error_field()
- correct_gaze_with_spatial_error()
- fit_pupil_size_artifact()
- correct_pupil_size_artifact()
- audit_pupil_preprocessing()
- plot_spatial_error_field()
- plot_pupil_size_artifact()

The established detect_calibration_drift(), fit_offline_recalibration(), apply_offline_recalibration(), and audit_recalibration() APIs remain the canonical drift/recalibration path.

## Dynamic AOIs

- validate_dynamic_aoi_spec()
- assign_dynamic_aoi()
- audit_dynamic_aoi_coverage()
- dynamic_aoi_sensitivity()
- plot_dynamic_aoi_alignment()

## Pursuits and microsaccades

- detect_events_ivvt()
- detect_events_directional()
- detect_smooth_pursuits()
- summarise_pursuits()
- compute_pursuit_gain()
- compute_pursuit_velocity_error()
- validate_pursuit_detection()
- detect_microsaccades()
- summarise_microsaccades()
- microsaccade_main_sequence()

## Detector benchmarking

- benchmark_event_detector()
- compare_event_detectors()
- event_boundary_error()
- event_confusion_matrix()
- detector_parameter_sensitivity()

## Naturalistic coordinates

- pixels_to_visual_angle()
- visual_angle_to_pixels()
- screen_gaze_to_head_vectors()
- align_head_pose_to_gaze()
- transform_head_gaze_to_world()
- transform_screen_gaze_to_world()
- propagate_coordinate_uncertainty()

## Scanpath sensitivity

- levenshtein_scanpath()
- transition_js_distance()
- ngram_jaccard_similarity()
- dynamic_time_warping_distance()
- compare_scanpath_metrics()
- scanpath_metric_sensitivity()

## Stress testing

- simulate_known_gaze_process()
- inject_spatial_drift()
- inject_spatial_noise()
- inject_blink_gaps()
- inject_event_misclassification()
- run_gaze_stress_suite()

These APIs are vendor-neutral. Vendor-specific parsing and convenience adapters remain outside the scientific core.
