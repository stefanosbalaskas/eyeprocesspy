# Manuscript claim and overclaim audit

This file records the strongest wording currently supported by the frozen empirical evidence and literature-positioning matrix. It is a writing-control artifact, not a new analysis.

| Claim area | Supported wording | Evidence basis | Stronger wording to avoid |
|---|---|---|---|
| Methodological novelty | We **extend** multiverse/specification approaches with a claim-centered, cross-stage eye-tracking measurement workflow that preserves one estimand, freezes a finite measurement-decision universe, retains non-evaluable branches, and independently validates measurement representation. | `literature_positioning.csv`; `gaze_claim_robustness.py`; SRL/MCFW decision registries. | “First multiverse analysis in eye tracking”; “first sensitivity analysis for gaze”; “no prior work has connected preprocessing to conclusions.” |
| Eye-tracking multiverses | Prior eye-tracking work already shows that temporal-window, cleaning, and analysis choices can alter estimates and inference. | Peelle & Van Engen (2021); Godwin et al. (2025). | Any wording implying multiverse analysis is absent from eye-movement research. |
| Detector uncertainty | Reasonable event-detection algorithms can produce materially different representations of the same gaze stream; performance/orderings depend on event type, context, and evaluation criterion. | Andersson et al. (2017); MCFW frozen validation. | “I-VT 40°/s is correct”; “I-DT is ground truth”; “I-VT 30°/s is invalid.” |
| SRL point-estimate stability | Across the frozen 144-specification SRL universe, Prompt point estimates vary in direction and magnitude (92 positive; 52 negative; RR range 0.944–1.258). | Frozen SRL model artifact and compact result snapshot. | “The Prompt effect is positive in 63.9% of reality”; treating specification frequency as posterior probability. |
| SRL interval stability | All 144 Wald 95% confidence intervals include the no-effect value. | Frozen SRL model artifact. | “Measurement choices do not matter”; “the effect is proven null.” |
| Viewing-distance sensitivity | Within this SRL reconstruction, the assumed 60–70 cm viewing geometry used for pixel-to-degree conversion is the largest descriptive sensitivity family and interacts with detector choice. | Frozen SRL decision-sensitivity and detector×distance summaries. | “Viewing distance is always the most important eye-tracking decision”; “60/65/70 cm are participants' true distances.” |
| AOI sensitivity | The two source-compatible SRL AOI boundary conventions are nearly inert for the focal coefficient in this dataset. | Frozen SRL decision-sensitivity summary; source AOI evidence. | “AOI definitions do not matter in eye tracking”; generalizing this source-specific result beyond the tested geometry. |
| Claim propagation | The SRL case demonstrates that defensible upstream measurement choices **can** propagate to the direction and magnitude of a fixed downstream HCI coefficient. | Fixed estimand/model across the 144 frozen measurement specifications. | “Most HCI eye-tracking findings are fragile”; prevalence claims across HCI. |
| MCFW role | MCFW-Gaze provides independent **measurement generalization** across different hardware, nominal sampling rate, signal quality, and interaction contexts. | Frozen MCFW protocol/registry/result artifact. | “MCFW replicates the SRL Prompt effect”; “cross-study replication of the HCI treatment.” |
| MCFW agreement | Raw fixation/non-fixation agreement is moderately high while positive-class overlap is substantially lower, showing that overall agreement can obscure event-representation differences. | Median agreement 0.707; Jaccard 0.272; kappa 0.199 in the frozen MCFW validation. | “Jaccard proves one detector is more accurate”; accuracy claims without ground truth. |
| Data quality | In MCFW, higher analyzed-eye usable fractions are generally associated with lower positive-class/event-summary detector divergence. | Frozen Spearman quality-association summaries. | Causal wording that data loss *causes* detector disagreement; universal quality thresholds. |
| Reporting contribution | Reporting one chosen pipeline is necessary for reproducibility but does not quantify whether alternative defensible measurement choices would change the conclusion. | Dunn et al. reporting guideline plus multiverse literature; conceptual synthesis. | Suggesting existing reporting guidelines are inadequate or incorrect; they address a different layer of reproducibility. |
| Denominator accountability | Planned failures/non-evaluable branches are retained as evidence rather than disappearing from robustness summaries. | Framework implementation and both frozen protocols/registries. | Claiming this is universally required for every sensitivity analysis without qualification. |
| Causal language | Prompt is the randomized between-participant contrast in the released SRL design; our primary analysis estimates the declared transition-rate contrast under that design while blocking Task identity. | Source descriptor and frozen SRL protocol. | Broader causal claims about learning outcomes, mechanisms, or material type beyond the released design and estimand. |
| Generalizability | Two complementary cases support the **possibility and structure** of measurement-to-claim propagation and independent detector representation differences. | SRL + MCFW frozen results. | Prevalence estimates for HCI generally; claims that all gaze-based HCI results require large multiverses. |

## Required language checks before submission

Every manuscript revision should preserve the following distinctions:

1. **measurement sensitivity ≠ statistical significance instability**;
2. **specification frequency ≠ probability of a true effect**;
3. **detector agreement ≠ detector correctness**;
4. **measurement generalization ≠ treatment-effect replication**;
5. **source-compatible sensitivity assumptions ≠ measured participant parameters**;
6. **descriptive decision-family shifts ≠ causal attribution to one preprocessing choice**;
7. **transparent reporting of one pipeline ≠ robustness evidence across pipelines**;
8. **a finite defensible universe ≠ an exhaustive set of all imaginable analyses**.

## Submission-positioning sentence

A defensible concise positioning sentence is:

> Prior work has established analytical flexibility in eye-tracking windows, cleaning procedures, event detection, AOI construction, and data quality. We build on these strands by treating the mapping from a prespecified cross-stage measurement-decision space to a fixed downstream HCI claim as the object of robustness analysis, while preserving the planned denominator and validating measurement representation independently of the primary claim dataset.
