# Comparing Event Catalogues

Different detectors rarely produce event tables with the same number of rows. Row 17 from one detector therefore should not be assumed to correspond to row 17 from another.

`match_detected_events()` uses one-to-one temporal matching within recording/trial/event type. Candidate pairs are prioritized by temporal intersection-over-union (IoU), with onset tolerance available for near-boundary cases.

```python
matches = ep.match_detected_events(
    events_a,
    events_b,
    event_type="fixation",
    onset_tolerance_ms=75,
    minimum_overlap=0.10,
)
```

The low `minimum_overlap` above is an example sensitivity setting, not a validation threshold. Studies with manually annotated ground truth may justifiably use a stricter criterion. For example, event-detection evaluations have used IoU-based event matching to penalize fragmentation and erroneous merging.

## Pairwise summaries

```python
comparison = ep.compare_event_catalogues(events_a, events_b)
agreement = ep.estimate_detector_agreement(multiverse_result)
```

Available summaries include matched-event precision, recall and F1, mean/median overlap, onset and offset differences, and duration differences. These metrics are most interpretable when one catalogue is a meaningful reference. In a detector-vs-detector multiverse with no ground truth, report them as **agreement**, not accuracy.

## Count and duration diagnostics

`summarise_detector_events()` reports fixation/saccade counts and fixation-duration summaries. `summarise_detector_disagreement()` adds unmatched-event and event-count differences.

Do not use these descriptive differences as the endpoint of the robustness analysis. Continue propagation into the measures and models used for the scientific claim.

Next: [Propagating Detector Choice to AOI Features](detector-aoi-features.md).
