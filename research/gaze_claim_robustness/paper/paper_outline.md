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

## Abstract source

The authoritative abstract is in `frontmatter_conclusion.md`. It is intentionally **not duplicated here**. The current version is 133 words and was compressed to satisfy the current CHI 2027 150-word abstract cap. See `chi_format_notes.md` for the dated format check.

## Paper structure

### 1. Introduction

Frame the problem at the HCI-claim level, not as a software or detector-comparison paper. End with the three RQs and contribution claims above. Source: `introduction_related_work.md`.

### 2. Related Work

Organize by methodological problem rather than tool:

- analytical flexibility and multiverse/specification analysis;
- eye-tracking temporal-window and cleaning/analysis multiverses;
- event-detection variability;
- AOI and coordinate-geometry uncertainty;
- data-quality decisions and missing gaze;
- reproducible eye-tracking/HCI measurement pipelines.

The novelty boundary must remain explicit: prior work already studies detector sensitivity, AOI sensitivity, temporal-window sensitivity, cleaning/analysis multiverses, and HCI multiverse tooling. The intended extension is **end-to-end propagation from a declared measurement-decision space to the stability of a prespecified substantive HCI claim**, with planned-denominator accountability and an independent raw-signal validation case.

Sources: `introduction_related_work.md`, `literature_positioning.csv`, `literature_references.bib`.

### 3. Claim-Centered Measurement Robustness

Present the general workflow independently of either dataset:

1. define the substantive estimand;
2. separate estimand-preserving measurement decisions from estimand-changing alternatives;
3. freeze a defensible decision registry before focal-result inspection;
4. execute all planned specifications under the same outcome/model contract;
5. preserve failures/non-evaluable branches in the denominator;
6. summarize claim stability by direction, magnitude, uncertainty, evaluability, and decision-family structure;
7. validate general measurement behavior on an independent dataset where possible.

Then document the SRL claim-propagation case and MCFW independent measurement-generalization case. Source: `methods_draft.md`.

### 4. Results

Use `results_draft.md` and the three deterministic empirical figures:

1. 144-specification SRL curve;
2. detector × viewing-distance SRL sensitivity;
3. MCFW detector-overlap profile across contexts.

### 5. Discussion

Use `discussion_draft.md`. Emphasize inference and measurement practice rather than recommending a detector.

### 6. Conclusion

Use `frontmatter_conclusion.md`. Return to the thesis that reproducible eye-tracking research should make consequential measurement uncertainty visible at the level of the conclusion, not only through documentation of one preprocessing pipeline.

## Claims to avoid

Do not write that:

- multiverse analysis is new to eye tracking or HCI;
- the 60, 65, or 70 cm SRL branch is the true participant viewing distance;
- I-VT 40°/s or I-DT is the correct detector because they agree more closely;
- MCFW-Gaze replicates the SRL Prompt effect;
- 92/144 positive estimates imply a 63.9% probability that the Prompt effect is positive;
- all intervals including zero means measurement choices do not matter;
- the two datasets establish prevalence of measurement fragility across HCI generally;
- decision-family mean shifts identify causal effects of preprocessing choices.

For line-by-line supported wording, use `claims_audit.md`.
