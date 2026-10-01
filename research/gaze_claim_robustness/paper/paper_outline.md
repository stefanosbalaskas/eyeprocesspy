# CHI paper scaffold — gaze-claim robustness

## Working title

**From Gaze Signals to HCI Claims: Propagating Eye-Tracking Measurement Uncertainty Through Analysis Pipelines**

Alternative shorter title:

**When Eye-Tracking Measurement Choices Change the HCI Claim**

## One-sentence thesis

Eye-tracking preprocessing is part of the measurement model: when several measurement choices are scientifically defensible, their uncertainty can propagate to substantive HCI estimates and should therefore be frozen, propagated, and reported at the level of the claim rather than hidden behind one pipeline.

## Research questions

**RQ1 — Claim propagation.** How much can the direction, magnitude, uncertainty, and evaluability of a fixed HCI estimand vary across a prespecified set of defensible eye-tracking measurement decisions?

**RQ2 — Sensitivity structure.** Which measurement-decision families account for the largest descriptive shifts in the focal estimate, and do different decisions interact rather than act independently?

**RQ3 — Independent measurement generalization.** Do detector-dependent event representations persist in an independent raw-gaze dataset collected with different hardware, sampling rate, and interaction contexts?

## Contribution claims

The paper should claim four contributions, without claiming that multiverse analysis itself is new:

1. **A claim-centered measurement-robustness workflow for eye-tracking HCI.** The workflow fixes the substantive estimand, declares a finite set of defensible measurement alternatives before result inspection, propagates each alternative through the same outcome/model contract, retains failed or non-evaluable branches in the denominator, and reports direction, magnitude, uncertainty, and evaluability separately.

2. **An end-to-end empirical demonstration of measurement-to-claim propagation.** In the SRL case, 144/144 prespecified models converged, yet point-estimate direction changed across the frozen measurement universe. Viewing-distance assumptions used in pixel-to-degree conversion produced the largest systematic shift and interacted with detector choice, while all 144 confidence intervals still included the null.

3. **Independent measurement-level generalization.** In MCFW-Gaze, detector-dependent fixation representations remained materially different across a separate 120 Hz Tobii dataset and six interaction contexts. Positive-class overlap and event summaries exposed substantially more disagreement than raw fixation/non-fixation agreement.

4. **A reproducible evidence package.** Decision registries, archive audits, frozen specification manifests, workflow provenance, exact artifact hashes, failure accounting, compact result snapshots, and deterministic publication figures preserve the distinction between pre-result specification and post-result interpretation.

## Abstract draft

Eye tracking is widely used in HCI to operationalize visual attention and interaction processes, yet substantive conclusions can depend on measurement decisions made before statistical modeling. We present a claim-centered workflow for propagating uncertainty in defensible eye-tracking measurement choices to a fixed HCI estimand. In a self-regulated-learning dataset, we froze a 144-specification universe spanning event detector, eye, viewing-distance assumption, AOI geometry, quality rule, and cohort definition, while holding the transition-rate estimand and negative-binomial mixed model fixed. All 144 models converged. Prompt-effect point estimates changed direction across specifications (92 positive; 52 negative), with viewing-distance assumptions producing the largest systematic shift, although every 95% confidence interval included the null. We then evaluated detector representation in an independent 120 Hz Tobii dataset across six interaction contexts. Median raw fixation/non-fixation agreement was 0.707, whereas median fixation-state Jaccard overlap was 0.272, revealing substantial event-representation differences hidden by shared non-fixation time. We argue that eye-tracking measurement assumptions should be treated as part of the inferential design: frozen before focal-result inspection, propagated to the substantive claim, and reported through multidimensional stability evidence rather than a single pipeline or binary robustness label.

## Planned paper structure

### 1. Introduction

Frame the problem at the HCI-claim level, not as a software or detector-comparison paper. The motivating gap is that eye-tracking studies often document a final detector/AOI/filtering configuration but do not show whether a substantive conclusion depends on other defensible measurement choices.

End the Introduction with the three RQs and four contribution claims above.

### 2. Related Work

Organize by methodological problem rather than tool:

- analytical flexibility and multiverse/sensitivity analysis;
- eye-tracking event-detection variability;
- AOI and coordinate-geometry uncertainty;
- data-quality decisions and missing gaze;
- reproducible eye-tracking/HCI measurement pipelines.

The novelty boundary should be explicit: prior work can study detector sensitivity, AOI sensitivity, temporal-window sensitivity, or multiverse analysis; this paper's intended contribution is **end-to-end propagation from a declared measurement-decision space to the stability of a prespecified substantive HCI claim**, with an independent raw-signal validation case.

### 3. Claim-Centered Measurement Robustness

Present the general workflow independently of either dataset:

1. define the substantive estimand;
2. separate estimand-preserving measurement decisions from estimand-changing alternatives;
3. freeze a defensible decision registry before focal-result inspection;
4. execute all planned specifications under the same outcome/model contract;
5. preserve failures/non-evaluable branches in the denominator;
6. summarize claim stability by direction, magnitude, uncertainty, evaluability, and decision-family structure;
7. validate general measurement behavior on an independent dataset where possible.

### 4. Empirical Cases and Results

#### 4.1 SRL complete-pipeline case

Dataset identity, archive audit, Prompt/Non-prompt contrast, transition-rate estimand, 144-specification construction, frozen NB2 GLMM, and complete result.

#### 4.2 MCFW-Gaze independent measurement case

Dataset provenance, timestamp and geometry gates, three frozen detectors, source-usable sample contract, pairwise fixation-state/event-summary outcomes, and six context families.

#### 4.3 Results

Use `results_draft.md` and Figures 1–3.

### 5. Discussion

Use `discussion_draft.md`. The Discussion should emphasize inference and measurement practice rather than recommending a detector.

### 6. Limitations and Future Work

Keep this focused:

- one substantive claim-propagation dataset;
- finite rather than exhaustive decision universe;
- no universal fixation ground truth in MCFW-Gaze;
- SRL viewing distances are sensitivity assumptions, not participant measurements;
- future work should test additional HCI estimands and measurement families under preregistered/frozen uncertainty sets.

### 7. Conclusion

Return to the thesis: a reproducible eye-tracking study should make consequential measurement uncertainty visible at the level of the conclusion, not only at the level of preprocessing documentation.

## Claims to avoid

Do not write that:

- multiverse analysis is new to eye tracking;
- the 60, 65, or 70 cm branch is the true SRL viewing distance;
- I-VT 40°/s or I-DT is the correct detector because they agree more closely;
- MCFW-Gaze replicates the SRL Prompt effect;
- 92/144 positive estimates imply a 63.9% probability that the Prompt effect is positive;
- all intervals including zero means measurement choices do not matter;
- the present two datasets establish prevalence of measurement fragility across HCI generally.
