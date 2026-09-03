# Getting Started with eyeprocesspy

`eyeprocesspy` harmonizes heterogeneous eye-tracking, pupil, event, response, and biometric streams without erasing their source semantics. The core object is a relational eye dataset, not a single wide data frame.

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/getting-started.Rmd`.

## Simulate a complete project

```python
import eyeprocesspy as ep

x = ep.simulate_eye_dataset(n_person=20, n_item=8, seed=42)
ep.validate_eye_dataset(x)
ep.provenance_manifest(x)
```

## Standard workflow

```python
spec = ep.preprocess_spec(
    gaze_filter="median",
    pupil_interpolation="linear",
    pupil_filter="median",
    fixation_algorithm="ivt",
)

x = ep.preprocess_eye(x, spec)
x = ep.build_aoi_visits(x)
x = ep.derive_all_features(x)

ep.analysis_readiness(x)
ep.feature_dictionary(x)
```

Preprocessing choices remain explicit. Interpolation, filtering, fixation detection, and exclusions should be selected deliberately rather than hidden in import or feature extraction.

## Inspect and visualize

```python
trial = x.intervals["trial_id"].iloc[0]

ep.plot_eye_overview(x)
ep.plot_scanpath(x, trial_id=trial)
ep.plot_pupil_timeseries(x, trial_id=trial)
ep.plot_transition_matrix(x)
```

Plots are views of the canonical data and process representations. They do not replace the underlying quality and provenance records.

## Persist the canonical representation

```python
ep.write_eye_dataset(x, "analysis/eye-dataset")
ep.export_canonical(x, "analysis/canonical-folder")
ep.report_eye_dataset(
    x,
    "analysis/eyeprocess-report.md",
    include_plots=True,
)
```

A reproducible analysis should preserve the canonical representation, the preprocessing specification, quality evidence, derived-feature definitions, and provenance required to reconstruct downstream results.
