# Runnable examples

This page provides short, copy-paste workflows that exercise distinct parts of `eyeprocesspy`. The full scripts used for the documentation gallery live in the repository `examples/` directory.

## 1. Verify the installation

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
print(audit["valid"])
```

## 2. Import and validate an export

```python
import eyeprocesspy as ep

eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)

if not issues.empty:
    print(issues)
```

The canonical `EyeDataset` keeps recordings, streams, gaze, eye samples, episodes, events, intervals, responses, coordinate spaces, AOIs, features, quality and provenance in explicit tables.

## 3. Scanpath and transition analysis

```python
sequence = ep.scanpath_sequence(
    eye,
    trial_id="trial-01",
    source="visits",
    collapse_consecutive=True,
)

matrix = ep.transition_matrix(
    eye,
    source="visits",
    normalize="row",
)

print(sequence)
print(matrix)
```

## 4. Plot gaze, fixations and pupil data

```python
import matplotlib.pyplot as plt
import eyeprocesspy as ep

ax = ep.plot_eye_trace(eye, trial_id="trial-01")
plt.show()

ax = ep.plot_scanpath(eye, trial_id="trial-01")
plt.show()

ax = ep.plot_pupil_timeseries(eye, trial_id="trial-01")
plt.show()
```

See the [visual gallery](../gallery.md) for package-generated examples.

## 5. Process-measure reliability

```python
import pandas as pd
import eyeprocesspy as ep

repeated = pd.DataFrame(
    {
        "person": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
        "session": ["S1", "S2"] * 4,
        "dwell_score": [0.10, 0.14, 0.60, 0.55, -0.30, -0.25, 1.10, 1.03],
    }
)

profile = ep.process_reliability_profile(
    repeated,
    person="person",
    session="session",
    measure="dwell_score",
)
print(profile["icc"])
print(profile["bland_altman"]["summary"])
```

Reliability is population- and design-dependent; it does not establish construct validity.

## 6. Calibration uncertainty and probabilistic AOIs

```python
model = ep.calibration_error_model(calibration_validation_data)
uncertainty = ep.gaze_uncertainty_ellipse(model, level=0.95)

assignment = ep.probabilistic_aoi_assignment(
    gaze_points,
    aois,
    model,
    draws=500,
    seed=1,
    min_probability=0.50,
)

print(uncertainty)
print(assignment["assignments"])
```

This workflow quantifies uncertainty due to the fitted calibration-error model; it does **not** estimate a posterior probability of psychological attention.

## 7. IRT diagnostic plotting

```python
profile = ep.eye_irt_information_profile(model)
ax = ep.plot_eye_irt_information_profile(profile)
```

The IRT surface also includes item/person fit, Q3/local dependence, score uncertainty, adaptive traces, DIF/DTF, recovery/SBC evidence, bank coverage and process-alignment diagnostics.

## 8. Provenance and reproducibility

```python
manifest = ep.provenance_manifest(eye)
print(manifest["schema_version"])
print(manifest["sources"])
print(manifest["validation"])
```

For release-level verification, use the deterministic benchmark, validation evidence, reproducibility manifest and software-paper evidence workflows documented in the article library.

## Full example programs

- `examples/core_gallery.py` — constructs and validates a complete synthetic `EyeDataset`, then renders eight core plots.
- `examples/advanced_gallery.py` — reliability, calibration uncertainty, probabilistic AOIs, sampling irregularity and IRT diagnostic plots.

Both are deterministic and use no private participant data.
