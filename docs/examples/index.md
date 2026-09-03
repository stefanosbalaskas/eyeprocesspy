# Runnable examples

`eyeprocesspy` ships deterministic examples that require no private participant data. The focused workflows below have been executed against the CI-built `0.1.0` wheel; the gallery generators produce the figures shown throughout this site.

<div class="grid cards" markdown>

-   :material-eye: **Core gaze, AOI and provenance**

    Validate a canonical `EyeDataset`, recover scanpaths and transitions, compute gaze entropy, render auditable plots and inspect provenance.

    [Open worked workflow](core-workflow.md)

-   :material-target: **Calibration uncertainty → probabilistic AOIs**

    Fit an empirical calibration-error model, propagate coordinate uncertainty and diagnose boundary-sensitive AOI assignments.

    [Open worked workflow](calibration-probabilistic-aoi.md)

-   :material-chart-timeline-variant: **Process-measure reliability**

    Estimate repeated-measure ICC, Bland–Altman agreement and temporal stability without confusing reliability with construct validity.

    [Open worked workflow](process-reliability.md)

-   :material-chart-bell-curve-cumulative: **IRT diagnostics**

    Inspect conditional information, item fit and DIF with publication-ready Matplotlib diagnostics.

    [Open worked workflow](irt-diagnostics.md)

</div>

For short task-oriented snippets, use the [Cookbook](../cookbook.md). For visual output, browse the [15-figure gallery](../gallery.md).

## Verify the installation

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)

study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
print(audit["valid"])
```

## Import and validate an export

```python
import eyeprocesspy as ep

eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)

if not issues.empty:
    print(issues)
```

The canonical `EyeDataset` keeps recordings, streams, gaze, eye samples, episodes, events, intervals, responses, coordinate spaces, AOIs, features, quality and provenance in explicit tables.

## Scanpath and transition analysis

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

entropy = ep.gaze_entropy(
    eye,
    level="trial",
    source="samples",
)
```

## Plot gaze, fixations and pupil data

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

The plotting surface preserves its numerical payload on the returned axes where relevant:

```python
plot_data = ax.eyeprocess_plot_data
```

Matrix plots additionally expose `ax.eyeprocess_plot_matrix`.

## Process-measure reliability

```python
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

## Calibration uncertainty and probabilistic AOIs

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
```

This workflow quantifies coordinate uncertainty under the fitted calibration-error model; it does **not** estimate a posterior probability of psychological attention.

## IRT diagnostic plotting

```python
ax = ep.plot_eye_irt_information_profile(information_profile)
ax = ep.plot_eye_irt_item_fit(item_fit, statistic="infit")
ax = ep.plot_eye_irt_dif_curve(dif_curve)
```

The wider IRT surface also includes person fit, Q3/local dependence, score uncertainty, adaptive traces, link stability, DTF, recovery/SBC evidence, bank coverage, prior sensitivity and process-alignment diagnostics.

## Provenance and reproducibility

```python
manifest = ep.provenance_manifest(eye)
print(manifest["schema_version"])
print(manifest["sources"])
print(manifest["validation"])
```

For release-level verification, use the deterministic benchmark, validation evidence, reproducibility manifest and software-paper evidence workflows documented in the article library.

## Executable programs

| Script | Purpose | Output |
| --- | --- | --- |
| `examples/complete_workflow.py` | Canonical dataset → validation → scanpath/transitions/entropy → plots → provenance | `workflow-output/*.svg` |
| `examples/calibration_probabilistic_aoi.py` | Calibration error → uncertainty ellipse → probabilistic AOI | `workflow-output/*.svg` |
| `examples/process_reliability.py` | ICC, Bland–Altman and temporal stability | `workflow-output/process-reliability.svg` |
| `examples/irt_diagnostics.py` | Information, item fit and DIF diagnostics | `workflow-output/*.svg` |
| `examples/core_gallery.py` | Eight core gaze/AOI/pupil plots | `gallery-output/*.svg` |
| `examples/advanced_gallery.py` | Reliability, uncertainty, quality and IRT plot families | `gallery-output/*.svg` |

All six are deterministic and use no private participant data.
