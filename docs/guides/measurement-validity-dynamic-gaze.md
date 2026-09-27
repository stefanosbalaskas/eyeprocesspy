# Measurement validity and dynamic gaze

This workflow joins calibration error, pupil-size-related apparent gaze displacement, moving AOIs, pursuit and microsaccade classification, naturalistic coordinate transforms, scanpath metric dependence, and synthetic stress testing.

The design rule is simple: **measurement uncertainty is carried forward rather than silently repaired**. Every correction preserves raw values, every dynamic geometry choice is explicit, and failed or unsupported branches remain visible.

## Spatial calibration error fields

The established calibration and offline-recalibration API remains the canonical drift workflow. The new layer adds target-local spatial bias fields.

~~~python
import eyeprocesspy as ep

field = ep.compute_spatial_error_field(
    validation,
    target_id="target_id",
    minimum_samples_per_target=5,
)

corrected = ep.correct_gaze_with_spatial_error(
    samples,
    field,
    method="idw",
    power=2,
    strength=1,
)
~~~

Use a validation or recalibration recording with known target coordinates. Do not estimate a spatial error field from unconstrained free viewing. The IDW method interpolates observed target-local bias; nearest-neighbour correction is available as a sensitivity branch.

## Pupil-size artefact in reported gaze

~~~python
artifact = ep.fit_pupil_size_artifact(
    validation,
    pupil="pupil",
    gaze_x="gaze_x",
    gaze_y="gaze_y",
    target_x="target_x",
    target_y="target_y",
)

corrected = ep.correct_pupil_size_artifact(samples, artifact)
~~~

The model is target-referenced. It does not infer real gaze motion from pupil diameter.

## Dynamic AOIs

Dynamic AOIs are keyframed, vendor-neutral geometry. Rectangle and polygon coordinates may be linearly interpolated; all shapes support step interpolation. Masks are intentionally step-only.

~~~python
spec = ep.validate_dynamic_aoi_spec(dynamic_aois)

assigned = ep.assign_dynamic_aoi(
    samples,
    spec,
    interpolation="linear",
    lag_tolerance=0.020,
    overlap="ambiguous",
)

coverage = ep.audit_dynamic_aoi_coverage(assigned)
~~~

Assignment status distinguishes assigned, outside, ambiguous, overlap-priority/all, no-active-geometry, and missing-sample states. This prevents undefined media time from being silently coded as outside an AOI.

Use dynamic_aoi_sensitivity() to compare interpolation and lag-tolerance choices.

## Smooth pursuits and microsaccades

~~~python
pursuit = ep.detect_smooth_pursuits(
    samples_deg,
    method="directional",
    minimum_velocity=2,
    maximum_velocity=30,
)

micro = ep.detect_microsaccades(
    samples_deg,
    minimum_sampling_hz=200,
)
~~~

Microsaccade detection refuses recordings below the declared minimum sampling frequency. Pursuit thresholds have no universal validity and must be justified for the acquisition system and task. For moving targets, compute_pursuit_gain() and compute_pursuit_velocity_error() compare synchronized gaze and target velocity.

## Detector benchmarking

~~~python
benchmark = ep.benchmark_event_detector(
    truth_events,
    detected_events,
    minimum_iou=0.5,
)

comparison = ep.compare_event_detectors(
    truth_events,
    {
        "IVT": ivt_events,
        "IVVT": ivvt_events,
        "directional": directional_events,
    },
)
~~~

The benchmark separates event precision/recall/F1, temporal overlap, and onset/offset error. detector_parameter_sensitivity() keeps failed branches visible.

## Mobile and naturalistic coordinates

~~~python
head = ep.screen_gaze_to_head_vectors(
    gaze,
    screen_width_px=1920,
    screen_height_px=1080,
    screen_width_mm=530,
    screen_height_mm=300,
    viewing_distance_mm=600,
)

aligned = ep.align_head_pose_to_gaze(head, pose)
world = ep.transform_head_gaze_to_world(aligned)
~~~

Rotation uses spherical interpolation; position uses linear interpolation. Pose extrapolation is refused. propagate_coordinate_uncertainty() can propagate declared viewing-distance uncertainty.

## Pupil preprocessing audit

~~~python
audit = ep.audit_pupil_preprocessing(
    samples,
    pupil="pupil",
    valid="valid",
    blink="blink",
    interpolated="interpolated",
    by="participant_id",
)
~~~

The audit records finite and valid pupil fractions, blink/interpolation fractions, median pupil, time ordering, and warnings. It deliberately does not choose filtering, interpolation, baseline, or exclusion rules.

## Scanpath metric sensitivity

~~~python
sensitivity = ep.scanpath_metric_sensitivity(
    reference=["A", "B", "C"],
    candidates={
        "participant_1": ["A", "B", "C"],
        "participant_2": ["A", "C", "B"],
    },
)
~~~

The comparison includes normalized Levenshtein distance, transition Jensen-Shannon distance, bigram Jaccard similarity, and optional trajectory dynamic-time-warping distance.

## Eye-tracking-specific stress tests

~~~python
clean = ep.simulate_known_gaze_process()

stress = ep.run_gaze_stress_suite(
    clean,
    evaluator=my_metric_function,
    corruption="drift",
    severities=(0, 0.25, 0.5, 1.0),
)
~~~

Available corruption families include spatial drift, spatial noise, contiguous blink gaps, and event-label misclassification. Existing benchmark-stress utilities remain available for broader missingness, sampling jitter, device shift, pupil dropout, and AOI-label stress tests.

## Reporting checklist

Report the coordinate system and units, sampling rate, calibration/validation design, correction model and transfer assumption, AOI interpolation and overlap policy, detector thresholds, annotation benchmark if used, pose interpolation rule, scanpath metric set, stress-test corruption model, failed branches, and package version/provenance.

None of these utilities turns a measurement correction, AOI assignment, learned event, or transformed coordinate into a validated psychological construct by itself.
