# Cookbook

Short recipes for common `eyeprocesspy` tasks. These snippets are intentionally small; use the linked worked examples and articles when the analysis decision itself needs justification.

## Import a vendor export

```python
import eyeprocesspy as ep

eye = ep.read_eye_export("participant_001.csv", vendor="auto")
issues = ep.validate_eye_dataset(eye)
```

## Discover supported adapters

```python
print(ep.supported_eye_formats())
```

## Inspect one canonical table

```python
gaze = ep.get_eye_table(eye, "gaze_samples")
```

## Validate before analysis

```python
issues = ep.validate_eye_dataset(eye)
errors = issues.loc[issues["severity"].eq("error")]
if not errors.empty:
    raise RuntimeError(errors.to_string(index=False))
```

## Extract a scanpath

```python
seq = ep.scanpath_sequence(
    eye,
    trial_id="T1",
    source="visits",
    collapse_consecutive=True,
)
```

## Build a normalized AOI transition matrix

```python
matrix = ep.transition_matrix(
    eye,
    normalize="row",
    source="visits",
)
```

## Compute gaze entropy

```python
entropy = ep.gaze_entropy(
    eye,
    level="trial",
    source="samples",
)
```

## Plot a gaze trace

```python
ax = ep.plot_eye_trace(eye, trial_id="T1")
ax.figure.savefig("gaze-trace.svg", bbox_inches="tight")
```

## Retrieve the data behind a plot

```python
plot_data = ax.eyeprocess_plot_data
```

For matrix plots:

```python
matrix = ax.eyeprocess_plot_matrix
```

## Plot fixations

```python
ax = ep.plot_fixations(eye, trial_id="T1")
```

## Plot a scanpath

```python
ax = ep.plot_scanpath(eye, trial_id="T1")
```

## Plot a gaze heatmap

```python
ax = ep.plot_gaze_heatmap(eye, trial_id="T1", bins=(50, 50))
```

## Plot pupil time series

```python
ax = ep.plot_pupil_timeseries(eye, trial_id="T1")
```

## Estimate effective sampling frequency

```python
quality = ep.effective_sampling_frequency(
    samples,
    time="timestamp_ms",
    unit="ms",
    by="recording_id",
)
```

## Audit sampling irregularity

```python
audit = ep.audit_sampling_irregularity(
    samples,
    time="timestamp_ms",
    unit="ms",
    by="recording_id",
    cv_threshold=0.05,
)
```

## Fit an empirical calibration-error model

```python
model = ep.calibration_error_model(calibration_validation_data)
ellipse = ep.gaze_uncertainty_ellipse(model, level=0.95)
```

## Propagate calibration uncertainty to AOIs

```python
assignment = ep.probabilistic_aoi_assignment(
    gaze_points,
    aois,
    model,
    draws=500,
    seed=1,
)
```

## Estimate repeated-measure reliability

```python
profile = ep.process_reliability_profile(
    repeated,
    person="person",
    session="session",
    measure="dwell_score",
)
```

## Inspect the process-measure registry

```python
registry = ep.process_measure_registry()
gaze_measures = ep.find_process_measures(registry, channel="gaze")
```

## Read a measure's guardrail card

```python
card = ep.process_measure_card("dwell_time")
print(card["interpretation"])
print(card["guardrail"])
```

## Plot an IRT information profile

```python
ax = ep.plot_eye_irt_information_profile(profile_table)
```

## Plot item fit

```python
ax = ep.plot_eye_irt_item_fit(item_fit, statistic="infit")
```

## Plot a DIF curve

```python
ax = ep.plot_eye_irt_dif_curve(dif_curve)
```

## Capture provenance

```python
manifest = ep.provenance_manifest(eye)
```

## Verify the bundled benchmark

```python
study = ep.eyeprocess_benchmark_study()
audit = ep.validate_benchmark_study(study)
print(audit["valid"])
```

## Manual-install wheel verification

```powershell
python -c "import eyeprocesspy as ep; print(ep.__version__, ep.__r_reference_version__)"
```

## Where to go next

- [Core gaze/AOI workflow](examples/core-workflow.md)
- [Calibration uncertainty and probabilistic AOIs](examples/calibration-probabilistic-aoi.md)
- [Process reliability](examples/process-reliability.md)
- [IRT diagnostics](examples/irt-diagnostics.md)
- [Visual gallery](gallery.md)
- [Article library](articles/index.md)
- [API reference](reference/index.md)
