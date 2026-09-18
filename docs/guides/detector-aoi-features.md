# Propagating Detector Choice to AOI Features

Detector branches must be remapped to the same AOI definitions before feature sensitivity can be assessed.

```python
result = ep.run_detector_multiverse(data, multiverse)
result = ep.propagate_detector_to_aoi(result, overlap="error")
result = ep.propagate_detector_to_features(result)
```

## AOI ambiguity

`overlap="error"` is the default. If an event centroid falls inside multiple AOIs, propagation stops for that detector branch and records the failure. An analyst may explicitly choose `first`, `smallest`, or `all`, but this is a scientific decision and should be reported.

## Trial preservation

The propagated feature table is constructed from explicit trial intervals × registered AOIs. This means a trial is not lost merely because a detector finds no target fixation.

For a trial with valid gaze but no target fixation:

- fixation count = `0`;
- dwell = `0`;
- mean fixation duration = missing;
- TTFF = missing, with `ttff_event_observed=False` and censor time retained;
- revisits = `0`.

For a trial with **no valid gaze**, count and dwell remain missing rather than becoming zero.

## Recomputed outputs

Each detector branch recomputes:

- fixation count;
- total AOI dwell;
- mean fixation duration;
- first fixation latency / TTFF;
- event-observed and censor-time fields for TTFF;
- revisits;
- incoming/outgoing AOI transition counts;
- scanpath sequence;
- pupil-within-fixation mean when pupil samples are available.

`feature_review_required` flags extremely low-validity trials for review without silently excluding them.

## Provenance

Feature rows carry source-data, preprocessing/provenance, detector, AOI, quality, and software hashes/identifiers where available. This supports a trace from the model row back to the analytical branch that produced it.

Next: [Propagating Detector Choice to Statistical Inference](detector-statistical-inference.md).
