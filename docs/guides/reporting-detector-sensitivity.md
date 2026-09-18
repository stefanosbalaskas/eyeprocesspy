# Reporting Detector Sensitivity

A detector-sensitivity report should make the analytical decision space reproducible and show what changed downstream.

## Minimum reporting set

Report:

1. the primary detector and scientific rationale;
2. every detector family/specification in the sensitivity set;
3. sampling rate and coordinate units;
4. velocity/dispersion thresholds, minimum duration, maximum gap, smoothing/filtering, merge rules, and external implementation versions;
5. AOI specification and overlap rule;
6. data-quality rule(s) applied before modelling;
7. number of planned, successful, failed, and non-converged branches;
8. event-count/duration differences and temporal agreement metrics;
9. range/sensitivity of the AOI features supporting the scientific claim;
10. coefficient distribution, uncertainty and convergence across branches;
11. any predeclared substantive threshold used to interpret stability;
12. source/preprocessing/detector/AOI/quality/model/software provenance identifiers.

## Example wording

> We repeated the disclosure-dwell analysis across five successfully evaluated event-detection specifications spanning I-VT, I-DT, and an adaptive-velocity reference detector. REMoDNaV was also specified as an external branch and its availability/status was recorded rather than replaced by a surrogate. Detector choice changed event counts and the magnitude of AOI dwell to varying degrees. We therefore report the full coefficient range and confidence intervals across converged branches, together with sign and substantive-threshold stability, rather than a count of nominally significant models.

Adapt this text to the actual outputs. Do not report the example numbers from the documentation as if they came from an empirical dataset.

## What not to write

Avoid statements such as “the result was robust because 4/5 p-values were significant.” This conflates sampling uncertainty, detector sensitivity, effect direction and arbitrary alpha thresholds. Also avoid calling pairwise detector agreement “accuracy” unless one side is a justified reference standard.

## Automated Markdown report

```python
ep.report_detector_multiverse(
    result,
    inference=inference,
    term="C(condition_id)[T.disclosure]",
    substantive_threshold=100,
    path="detector-multiverse-report.md",
)
```

The generated report includes specification, event, feature, inference, failure, limitation and reporting sections.
