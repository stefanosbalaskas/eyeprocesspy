# Core gaze, AOI and provenance workflow

This worked example shows the shortest defensible path from a vendor-neutral canonical dataset to validated gaze/AOI process summaries, auditable plots, and a provenance manifest.

![Gaze trace](../assets/gallery/gaze-trace.svg)

The complete executable program is [`examples/complete_workflow.py`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/examples/complete_workflow.py).

## 1. Construct or import a canonical dataset

In a real study, start with `read_eye_export()` / `read_eye_generic()` or a vendor-specific helper. The example constructs a small deterministic `EyeDataset` directly so the workflow can run without private files.

```python
import eyeprocesspy as ep

# imported_or_constructed_tables = ...
eye = ep.new_eye_dataset(
    recordings=recordings,
    streams=streams,
    gaze_samples=gaze,
    episodes=episodes,
    intervals=intervals,
    validate=False,
)
```

The canonical object keeps recordings, streams, gaze samples, eye samples, episodes, events, intervals, responses, coordinate spaces, AOIs, features, quality and provenance in explicit tables.

## 2. Validate before deriving features

```python
issues = ep.validate_eye_dataset(eye)
if not issues.empty:
    raise RuntimeError(issues.to_string(index=False))
```

Validation should happen before model fitting or feature aggregation so duplicate keys, orphan identifiers, malformed intervals and non-finite timestamps do not silently propagate.

## 3. Recover scanpaths and transitions

```python
sequence = ep.scanpath_sequence(
    eye,
    trial_id="T1",
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

`scanpath_sequence()` preserves ordered AOI states. `transition_matrix()` converts those states into a transition representation, and `gaze_entropy()` quantifies occupancy dispersion at the declared level.

## 4. Plot without losing the numerical payload

```python
ax = ep.plot_scanpath(eye, trial_id="T1")
plot_data = ax.eyeprocess_plot_data
```

![Scanpath](../assets/gallery/scanpath.svg)

Core plot helpers attach their underlying data to the returned Matplotlib axes. Matrix plots additionally expose `ax.eyeprocess_plot_matrix`. This makes later publication styling auditable rather than separating a figure from the data that produced it.

## 5. Capture provenance

```python
manifest = ep.provenance_manifest(eye)
print(manifest["schema_version"])
print(manifest["validation"])
```

A reproducible analysis should retain the exact package version/commit, input hashes where available, coordinate conventions, validation output, preprocessing decisions and feature/model specifications.

## What to report in a manuscript

At minimum, report the eye tracker/vendor and sampling configuration, the import path, validity rule, coordinate system, event/trial segmentation, fixation/AOI derivation rule, exclusion/QC criteria, exact feature definitions, and the `eyeprocesspy` version or commit.

!!! note "Interpretation boundary"
    Scanpaths, transitions, dwell, entropy and other gaze-process metrics are operational measures. Their psychological interpretation depends on the study design and external construct-validity evidence.
