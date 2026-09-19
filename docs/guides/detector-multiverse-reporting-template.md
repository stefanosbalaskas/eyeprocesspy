# Detector multiverse reporting template

Replace bracketed fields with study-specific information. Do not copy synthetic example values into an empirical manuscript.

## Methods template

> Event detection was treated as an explicit measurement decision. The primary specification was [detector family/implementation and parameters], selected because [scientific rationale]. Sensitivity analyses evaluated [number] prespecified detector branches spanning [families/parameter ranges]. Gaze was sampled at [Hz] and analysed in [coordinate units]. All branches used the same [preprocessing], AOI definitions, overlap rule, quality rules, and downstream model. Detector failures were retained in the analysis audit rather than replaced by another algorithm. Model-input accounting recorded propagated rows, target-AOI rows, quality exclusions, non-finite outcomes, and rows supplied to the estimator for every branch.

## Results template

> Across [number planned] planned specifications, [number successful detection] produced valid event catalogues and [number term available] returned the focal coefficient. [Number converged] branches converged, corresponding to a planned-specification convergence rate of [rate]. Pairwise event agreement ranged from [range/metric], while [key feature] ranged from [range]. The focal coefficient ranged from [min] to [max], with [same-sign proportion] sharing the same direction. [If used: The prespecified substantive threshold of X was supported in Y% of converged branches.] Model-input audits showed [quality exclusions/outcome missingness summary]. [Describe failures/non-convergence explicitly.]

## Interpretation template

> The substantive pattern was [stable/sensitive] across the declared detector decision space. This supports [narrow claim] conditional on the evaluated preprocessing, AOIs, quality rules, detector specifications, and model. Detector agreement is not interpreted as event accuracy, and the sensitivity analysis does not establish that any detector is ground truth.

## Minimum table columns

For the detector table, report:

- detector ID/family/implementation/version;
- threshold and duration/gap/merge parameters;
- sampling rate and coordinate units;
- detection status and event count;
- model-input audit counts;
- model convergence/term availability;
- focal estimate, interval, and N when available.

For the robustness summary, report:

- planned specifications;
- term-available specifications;
- model-failure specifications;
- converged specifications and planned-denominator convergence rate;
- median/range of the estimate;
- same-sign proportion;
- CI overlap when estimable;
- prespecified substantive-threshold stability when used.

## Reproducible output

```python
text = ep.report_detector_multiverse(
    result,
    inference=inference,
    term="C(condition_id)[T.disclosure]",
    substantive_threshold=100,
    path="detector-multiverse-report.md",
)
```

The automated report is a starting point. Add the study-specific rationale for the detector set, preprocessing, AOIs, quality rule, model, and interpretation boundary.

## API and examples

[API reference](../reference/detector-multiverse.md) · [Disclosure worked example](../examples/detector-multiverse-disclosure.md) · [Failure clinic](../examples/detector-multiverse-failure-clinic.md) · [Reporting guidance](reporting-detector-sensitivity.md)
