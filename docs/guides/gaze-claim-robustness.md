# Gaze claim robustness

## Why this workflow exists

A gaze-based HCI conclusion is rarely a direct property of raw samples. It is produced through a measurement chain:

raw gaze → event detection → AOI assignment → quality handling → derived outcome → statistical estimate → HCI claim.

Each stage can contain more than one defensible analytical choice. The gaze-claim robustness workflow asks a narrow question:

> Would the focal HCI conclusion remain substantively similar under the other measurement specifications that were declared defensible before inspecting their results?

This is a sensitivity analysis. It does not identify a ground-truth detector, objectively correct AOI, or probability that a scientific claim is true.

## Scientific boundary

Multiverse analysis is not new to eye-movement research. Godwin, Lee, and Drieghe (2025) evaluated 1,890 reasonable cleaning and analysis pipelines for reading data and showed that a well-known effect was highly stable in direction while its estimated magnitude varied materially. Their study motivates transparent examination of analytical flexibility.

The present orchestration layer is narrower in implementation but broader in measurement scope: it is designed to connect already existing detector, AOI, quality, outcome, and model branches to the stability of one substantive HCI claim.

Reference:

- Godwin, H. J., Lee, C. E., & Drieghe, D. (2025). A multiverse analysis of cleaning and analyzing procedures of eye movement data during reading. Behavior Research Methods, 57, 164. https://doi.org/10.3758/s13428-025-02689-0

## One specification, one estimand

Do not mix estimand-changing alternatives in a single robustness denominator.

For example, these answer different questions:

- mean time to first fixation among trials where a fixation occurred;
- time to first fixation with valid non-fixated trials represented as right-censored observations.

If both are scientifically important, define two gaze robustness specifications and compare their conclusions explicitly.

This rule prevents a large multiverse from concealing a change in the scientific question.

## Declare the claim

~~~python
import eyeprocesspy as ep

claim = ep.define_gaze_claim_spec(
    "privacy_notice_attention",
    term="condition",
    estimand_id="disclosure_dwell_difference_ms",
    substantive_threshold=50.0,
    threshold_direction="above",
)
~~~

The substantive threshold is optional. When supplied, it must be chosen for the study before inspecting branch-specific results.

## Declare the measurement universe

~~~python
spec = ep.define_gaze_robustness_spec(
    claim,
    {
        "detector": ["ivt_30", "idt_reference", "adaptive_reference"],
        "aoi": ["nominal", "dilate_025deg", "erode_025deg"],
        "quality_rule": ["valid_070", "valid_080"],
    },
    primary_decisions={
        "detector": "ivt_30",
        "aoi": "nominal",
        "quality_rule": "valid_080",
    },
)
~~~

Every value in this grid needs a methodological justification. Do not add implausible branches to make a result appear unstable.

Use expand_gaze_robustness_spec(spec) to freeze and inspect the complete planned universe before evaluating outcomes.

## Connect existing engines

The new layer does not reimplement detector or AOI algorithms. A study-specific runner connects a declared branch to the package workflows already responsible for:

- detector specifications and event catalogues;
- AOI perturbation or probabilistic assignment;
- spatial/data-quality rules;
- feature derivation;
- the fixed focal statistical model.

The runner must return one focal coefficient:

~~~python
{
    "estimate": ...,
    "SE": ...,
    "CI_lower": ...,
    "CI_upper": ...,
    "converged": ...,
    "N": ...,
}
~~~

A one-row DataFrame is also accepted. Common column aliases from the existing detector/AOI inference surfaces are normalized.

## Preserve the planned denominator

~~~python
audit = ep.run_gaze_robustness_audit(spec, runner)
summary = ep.summarise_gaze_claim_robustness(audit)
~~~

Failed and non-converged branches remain visible. The summary reports both the number of evaluable universes and the denominator of all planned universes.

The workflow emphasizes:

- primary-estimate direction;
- same-direction stability;
- effect-size range;
- common uncertainty-interval overlap;
- model-N variation;
- a prespecified substantive threshold when available;
- explicit failures and non-convergence.

It does not summarize robustness by counting statistically significant results.

## Decision-family sensitivity

~~~python
ep.decompose_gaze_decision_sensitivity(audit)
~~~

This reports how much marginal branch means/medians shift across levels of each declared decision family.

Treat this as descriptive sensitivity accounting. It is not a causal attribution and is not a variance-explained decomposition.

## Reporting

~~~python
print(ep.report_gaze_claim_robustness(audit))
ax = ep.plot_gaze_specification_curve(audit)
~~~

Avoid automatic labels such as "robust", "moderately robust", or "fragile" unless the study protocol defined those classification thresholds before the multiverse results were inspected.

A defensible conclusion is conditional:

> The focal effect was stable within the declared detector, AOI, and quality-rule universe.

It is not:

> The effect is objectively robust.

## First HCI pilot

The intended first empirical pilot is an interface-search dataset with raw timestamped gaze coordinates so event detection can be rerun from the same sample stream. The pilot should freeze:

1. the focal HCI contrast;
2. the estimand;
3. the detector family and parameter envelope;
4. the AOI perturbation envelope;
5. quality rules;
6. the statistical model;
7. any substantive-effect threshold;

before inspecting the resulting specification curve.

The objective is not to manufacture reversals. A finding that conclusions remain stable across the declared universe is equally informative.
