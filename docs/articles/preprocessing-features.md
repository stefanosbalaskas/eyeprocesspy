# Preprocessing, AOIs, and Feature Engineering

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/preprocessing-features.Rmd`.

## Declare preprocessing

```python
import eyeprocesspy as ep

spec = ep.preprocess_spec(
    gaze_filter="median",
    gaze_window=5,
    pupil_interpolation="linear",
    pupil_max_gap_ms=150,
    pupil_filter="median",
    pupil_window=5,
    pupil_baseline="subtract",
    fixation_algorithm="ivt",
    fixation_parameters={"velocity_threshold": 30},
)

x = ep.preprocess_eye(x, spec)
```

Vendor-produced episodes are retained separately from package-derived episodes through source-algorithm, source-parameter, and derivation metadata. Preprocessing should never erase the distinction between native and derived events.

## Static and dynamic AOIs

```python
x = ep.register_aois(
    x,
    ep.new_aoi("prompt", x=0.00, y=0.00, width=0.45, height=1.00),
    ep.new_aoi("options", x=0.45, y=0.00, width=0.55, height=1.00),
)

x = ep.assign_aois(x, component="gaze_samples")
x = ep.build_aoi_visits(x)
```

A dynamic AOI is represented by one definition and multiple geometry records with explicit validity intervals or frame identifiers.

## Declared features

```python
fspec = ep.feature_spec(
    level="trial_aoi",
    include_post_response=False,
    response_time=True,
    biometrics=True,
)

x = ep.derive_all_features(x, fspec)
ep.features_wide(x)
ep.feature_dictionary(x)
```

## Sensitivity analysis

```python
ep.compare_preprocessing(
    x,
    {"raw": ep.preprocess_spec(), "filtered": spec},
)
ep.compare_aoi_definitions(
    x,
    {"primary": primary_aois, "alternative": alternative_aois},
)
ep.check_process_leakage(x)
ep.check_feature_level(x)
```

Feature engineering decisions are part of the scientific specification. Alternative filters, AOI definitions, post-response rules, and feature levels should be exposed through sensitivity analysis rather than selected after observing the preferred inferential result.
