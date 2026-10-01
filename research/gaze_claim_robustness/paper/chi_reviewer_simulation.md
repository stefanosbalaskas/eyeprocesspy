# CHI reviewer simulation — gaze-claim robustness paper

This is an author-facing review-control document. It evaluates the assembled manuscript against the current CHI methodology framing and the general review dimensions of originality, correctness, novelty, importance, and clarity. It is not a new analysis and does not modify either frozen empirical universe.

## Overall reading

The paper is strongest when read as a **methodology contribution about the measurement-to-claim mapping**, not as a detector-comparison paper, a software paper, or a generic multiverse paper. Its empirical evidence has two complementary roles: SRL demonstrates propagation from upstream measurement decisions to a downstream HCI coefficient, whereas MCFW-Gaze demonstrates that detector-dependent event representations persist in an independent raw-signal setting. The manuscript should preserve that division throughout review and revision.

The primary review risk is therefore conceptual positioning rather than technical qualification. A reviewer familiar with Godwin et al. (2025), Peelle and Van Engen (2021), or general multiverse methods may initially ask whether the work is only another sensitivity analysis. The manuscript must make the narrower extension immediately legible: cross-stage eye-tracking measurement decisions are declared before focal-result inspection, propagated through one claim contract, audited against a planned denominator including non-evaluable branches, and complemented by independent measurement-level validation.

## Review dimensions

### Originality

**Current strength.** The paper does not claim that multiverse analysis, detector disagreement, AOI sensitivity, or eye-tracking reporting is new. Instead, it treats the cross-stage mapping from recorded gaze to a downstream HCI claim as the methodological object. The distinction between measurement-definition alternatives, sample-definition sensitivities, and estimand-changing analyses is particularly useful because it prevents a nominally fixed coefficient formula from being mistaken for a formally identical target population.

**Likely reviewer question.** How is this materially different from prior eye-movement multiverse work, especially Godwin et al. (2025)?

**Defensible response already in the manuscript.** The difference is scope and unit of interpretation: the SRL case begins at sample-level gaze, crosses geometry, detector, eye, AOI, quality/cohort, and outcome construction, and ends at one fixed downstream HCI coefficient. The planned denominator and failure semantics are explicit, and detector behavior is then examined independently in MCFW-Gaze.

**Do not strengthen further by claiming a "first."** The contribution is an extension and synthesis with a different methodological object.

### Correctness

**Current strength.** The scientific contract is unusually explicit. Branch inclusion is frozen and source/literature-grounded; failed/non-evaluable branches remain visible; there is no fallback estimator; source defects are not silently repaired; and the paper distinguishes point-estimate sensitivity from interval-level stability. The MCFW case correctly avoids using inter-detector agreement as ground truth.

**Likely reviewer question.** Are the 144 branches really comparable if quality/cohort choices change sample support?

**Defensible response already in the manuscript.** No claim of strict formal population-estimand identity is made. Detector/eye/geometry/AOI are measurement-definition alternatives; quality/cohort are explicitly labelled sample-definition sensitivities. The common claim contract fixes the focal Prompt contrast, transition-rate semantics, Task adjustment, exposure definition, model family, and interpreted coefficient while reporting sample support separately.

**Likely reviewer question.** Why these three detectors and thresholds?

**Current defense.** The detector plan was frozen before focal-result inspection. I-VT 30°/s + 100 ms is a conventional fixed-threshold comparator; I-VT 40°/s + 50 ms is an SMI-context comparator; I-DT 1° + 100 ms provides a conceptually distinct dispersion comparator. None is treated as vendor-equivalent or universally correct. The manuscript now states this rationale directly; exact source tracing remains in the reproducibility package.

### Novelty

**Current strength.** The novelty argument is appropriately narrow. The method combines ideas that exist separately but asks a different robustness question: not merely whether preprocessing outputs differ, but whether defensible uncertainty across several measurement stages changes the HCI quantity that is interpreted.

**Primary risk.** A reviewer may see "multiverse + eye tracking" and stop reading before reaching the distinction.

**Author action.** Keep the introduction's contribution paragraph to three scientific contributions. Treat the reproducibility package as evidence infrastructure, not a fourth scientific contribution. Preserve the sentence that the intended contribution is the measurement-to-claim mapping.

### Importance

**Current strength.** The problem is broadly relevant to inferential eye-tracking HCI because many published constructs—fixations, AOI visits, transitions, dwell measures, gaze-derived process variables—are outputs of measurement pipelines rather than direct sensor observations. The workflow is selective rather than maximal: it asks researchers to identify uncertain, consequential assumptions rather than enumerate every imaginable preprocessing option.

**Likely reviewer question.** Does one SRL claim and one independent detector-validation dataset justify broader HCI relevance?

**Defensible response already in the manuscript.** The empirical cases demonstrate possibility and structure, not prevalence. The claimed generality belongs to the workflow, while the empirical scope is explicitly limited. The paper should not claim that most HCI eye-tracking findings are fragile or that every study requires a large Cartesian multiverse.

**Audience boundary.** The intended audience is HCI researchers deriving event-, AOI-, or process-level gaze measures for inferential claims. The paper does not position the workflow as a universal prescription for real-time gaze-control systems.

### Clarity of exposition

**Current strength.** The paper now presents a coherent sequence: problem → prior flexibility work → claim-centered contract → SRL propagation → MCFW measurement generalization → multidimensional interpretation. The three empirical figures correspond to the three most important result messages.

**Potential confusion to continue avoiding.**

- 92/144 is not a probability or invariant robustness percentage.
- sign instability is not the same as significance instability.
- all intervals including the null does not imply that measurement assumptions are irrelevant.
- MCFW is not a replication of the Prompt effect.
- inter-detector similarity does not identify a correct detector.
- sample-definition sensitivities are not presented as formally identical population estimands.

## Most likely substantive reviewer objections

### 1. "All 144 confidence intervals include the null, so what claim actually changes?"

This is the most serious substantive objection and should be answered rather than evaded. The paper does **not** demonstrate instability of a binary significance conclusion. It demonstrates that point-estimate direction and magnitude are sensitive while interval-level inference is stable. That coexistence is itself the methodological point: robustness is multidimensional. The paper should never imply that the SRL example shows a result switching from statistically established positive to statistically established negative.

No new analysis is needed to answer this objection. The existing Results and Discussion already state both properties separately.

### 2. "The Cartesian grid gives arbitrary weight to families with more levels."

Correct. The manuscript explicitly states that specification frequencies depend on grid granularity and therefore are descriptive inventory counts, not probabilities or invariant robustness scores. Family-level patterns are interpreted separately. This safeguard should remain prominent around the 92/144 result.

### 3. "Viewing distance is hypothetical rather than measured."

Correct and intentionally so. The public source reports approximately 60–70 cm without participant-specific measurements. The 60 and 70 cm branches are source bounds; 65 cm is an explicit midpoint sensitivity assumption. The paper uses the result to show what can happen when angular detector thresholds are derived from incompletely observed geometry, not to estimate participants' true distances.

### 4. "Why should the MCFW case count as validation without ground truth?"

It validates **measurement generalization**, not detector accuracy. Its purpose is to show whether representation differences among the frozen detector choices persist on different hardware, sampling rate, signal quality, and interaction contexts. The absence of universal fixation ground truth is acknowledged and prevents a detector-winner claim.

### 5. "Is this a software contribution disguised as methodology?"

The paper should continue to keep software machinery subordinate. The implementation is described as an orchestration layer that enforces the methodology. Function counts, package breadth, and product-style claims should stay out of the manuscript. Reproducibility artifacts support the method but are not the core contribution.

## Changes justified before submission

The following author-facing changes are justified and have been made:

1. reduce the headline contribution list from four items to **three scientific contributions**, with reproducibility infrastructure supporting them rather than competing with them;
2. state the intended HCI audience explicitly;
3. expose the pre-result rationale for the exact detector branches in the main Methods rather than leaving it only in the registry;
4. retain the existing safeguards around specification frequency, sample-definition sensitivity, detector correctness, and independent-validation scope.

## Changes not justified by this review

Do **not** add another primary dataset, detector family, AOI perturbation, quality threshold, statistical estimator, or outcome model merely to make the paper look larger. Doing so after observing the current results would weaken the pre-result/post-result boundary that is central to the contribution.

A conceptual workflow figure could improve skimmability, but it is optional rather than evidentially necessary. It should be added only if author inspection finds that the method remains difficult to understand from Sections 1 and 3; it must not replace empirical detail or trigger a new analysis path.

## Current submission posture

The paper is technically qualified and scientifically frozen. Remaining work is author judgment: confirm that the contribution framing is persuasive to an HCI methodology reviewer, verify the target-cycle call and anonymization rules immediately before submission, and select reviewer-expertise descriptors that emphasize HCI methods, eye tracking, quantitative measurement, reproducibility/sensitivity analysis, and empirical evaluation rather than software engineering.
