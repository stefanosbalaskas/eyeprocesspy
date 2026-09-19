# Worked example: quality decisions as sensitivity analysis

A quality metric is evidence about the measurement process, not an automatic exclusion rule. This example shows how to keep the original report intact, define a transparent review rule, and compare downstream subsets explicitly.

## 1. Create a complete report

```python
import eyeprocesspy as ep

validation = ep.simulate_gaze_quality_calibration(
    seed=20260918,
    samples_per_target=12,
    nominal_sampling_hz=60,
)

quality = ep.create_gaze_quality_report(
    validation,
    by=["profile", "target_id"],
    valid="valid",
    missing_reason="missing_reason",
    nominal_sampling_hz=60,
)
```

Nothing has been excluded. The canonical report preserves every analysis unit and records `automatic_exclusion=False` in provenance.

## 2. Add a study-defined review rule

```python
reviewed = ep.create_gaze_quality_report(
    validation,
    by=["profile", "target_id"],
    valid="valid",
    missing_reason="missing_reason",
    nominal_sampling_hz=60,
    thresholds={
        "accuracy_mean": {"max": 1.0},
        "valid_sample_fraction": {"min": 0.80},
    },
)

reviewed[
    ["profile", "target_id", "accuracy_mean",
     "valid_sample_fraction", "quality_flags", "review_required"]
]
```

Threshold names must exist in the report, rule keys must be `min` and/or `max`, and values must be finite numeric. Malformed rules fail rather than being silently ignored.

## 3. Make the analysis decision explicitly

```python
review_subset = reviewed.loc[~reviewed["review_required"]].copy()
flagged_subset = reviewed.loc[reviewed["review_required"]].copy()
```

This filtering step is intentionally **outside** the quality-report constructor. The scientific record therefore distinguishes:

1. the observed quality data;
2. the study-defined review rule;
3. the analyst's eventual inclusion/exclusion decision.

## 4. Compare conclusions under plausible decisions

For a downstream trial-level model, preserve the report key and join quality metadata back to the model data. Then repeat the planned analysis under pre-specified alternatives such as:

- all eligible trials, with quality metrics used diagnostically;
- excluding only trials failing a strict pre-specified rule;
- stratifying by a quality flag;
- weighting by a defensible reliability quantity when the statistical model supports that interpretation.

Do **not** use an arbitrary post-hoc cutoff merely because it improves the substantive result.

## 5. Sampling-loss sensitivity

When nominal frequency is supplied, distinguish:

- `long_interval_count`: number of observed intervals exceeding the declared long-gap rule;
- `dropped_interval_count`: estimated missing nominal samples represented by those gaps.

The latter is an inference from timing, not a direct hardware packet-loss counter.

## Suggested manuscript wording

> Eye-tracking quality thresholds were specified as review criteria rather than automatic exclusion rules. Primary models retained all otherwise eligible trials and included quality diagnostics in model checking. Sensitivity analyses repeated the key models after excluding trials that exceeded the pre-specified accuracy or valid-data thresholds. Conclusions were compared across specifications, and the number of affected trials was reported.

## API links

- [Data Quality overview](../guides/data-quality.md)
- [Reporting guideline](../guides/data-quality-reporting.md)
- [9-point validation example](data-quality-validation.md)
- [Complete API reference](../reference/api.md)
