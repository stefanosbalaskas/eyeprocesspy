# SRL complete-pipeline pilot protocol

Status: pre-analysis design. No multiverse branch effects have been inspected.

## Dataset

Primary open Tier-3 dataset:

Juřík, V., Juhaňák, L., Ružičková, A., Dostálová, N., & Juříková, Z. (2025).
Experimental Dataset on Eye-tracking Activity During Self-Regulated Learning.
Scientific Data, 12, 967.
Dataset DOI: 10.6084/m9.figshare.28304069.

The released study has a 2 × 2 mixed design:

- Prompt versus Non-prompt between participants;
- Text versus Multimedia within participants;
- eight study materials per participant;
- SMI RED 250 acquisition at 250 Hz;
- 1600 × 900 stimulus display;
- sample-level left/right point-of-regard coordinates;
- released stimulus images;
- four content AOIs per learning slide in the original study.

## Why this is the primary demonstration

The outcome can be reconstructed from the sample stream rather than accepted from a vendor export.

That permits one coherent audit:

raw point-of-regard
→ event detector
→ independently declared AOI geometry
→ quality rule
→ between-AOI transition outcome
→ fixed repeated-measures model
→ focal HCI contrast.

The original event-level and AOI-labelled exports remain useful reference branches. They are not treated as ground truth.

## Related-work risk and novelty boundary

The Scientific Data descriptor cites an unpublished manuscript, *Gaze transitions as a mediator of the effect of metacognitive prompts on student learning: An eye-tracking experiment*. A broad public search did not identify a published version with that title as of 2026-09-29, but unpublished or later work may overlap substantively.

Therefore this pilot must not claim novelty from merely analyzing gaze transitions in this dataset. The intended contribution is measurement robustness: whether a prespecified HCI/learning conclusion survives defensible event-detection, viewing-geometry, AOI-boundary, quality, eye-representation, and modeling decisions.

## Primary HCI contrast

The primary scientific contrast is now the randomized manipulation:

**Prompt versus Non-prompt**

on between-AOI transition **rate** during the eight learning slides.

This is preferable to a Text-versus-Multimedia primary claim for identification. The published design states that presentation order was randomized, but the experiment contains eight fixed learning-material pages: four text and four multimedia pages. Only one released image exists for each `Task_1` to `Task_8`, and the paper does not describe alternate text/multimedia versions of the same optics topic. Therefore material type may be nested in topic/task identity rather than independently randomized within topic.

The archive-level `stimuli.csv` audit remains the definitive gate. Until it proves modality varies within task/topic identity across participants, Text versus Multimedia must not be described here as a pure causal format effect.

Prompt assignment, by contrast, was randomized between participants. The primary prompt contrast remains identifiable while controlling for the eight task identities.

The analysis does not prespecify that the prompt effect must be positive, negative, or statistically significant.

## Transition operationalization boundary

The source study defines a transition as a **saccade from one AOI to another**. The primary open/reproducible detector branches do not all expose identical proprietary saccade semantics, so the common primary outcome is operationalized more narrowly:

> A between-AOI transition occurs when two directly adjacent fixation episodes are both assigned to AOIs, are in valid temporal order, and their AOI identifiers differ.

An unassigned fixation breaks the chain. Thus:

```text
AOI A fixation → unassigned fixation → AOI B fixation
```

does **not** become an A→B transition.

If adjacent fixation episodes overlap in time, that trial branch is marked non-evaluable rather than forcing an ordering.

The released BeGaze saccade-defined transition catalogue remains a historical reference outside the primary 96-specification denominator. Agreement or disagreement between that vendor construct and the open adjacent-fixation construct is itself measurement evidence; the two are not declared algorithmically identical.

## Exposure-time decision

The primary estimand is a **transition-rate ratio**, not a raw-count difference.

The published AJET analysis reports that prompted participants spent more time studying the learning materials on average. Raw transition counts can therefore differ simply because participants had more observation time.

Operationally, the model uses:

- outcome: between-AOI transition count per participant × learning slide;
- exposure: observed `stimulus_time` for that slide;
- offset: `log(stimulus_time)`.

This preserves the auditable count outcome while estimating a rate. Trials with non-positive, missing, or internally inconsistent exposure time are non-evaluable and must remain visible in the branch audit.

Raw transition count without an exposure offset is an **estimand-changing sensitivity analysis** and must not be included in the primary transition-rate robustness denominator.

## Secondary contrasts

Subject to the structural and identifiability audits:

- Prompt-effect heterogeneity across the released Text-task versus Multimedia-task sets;
- the released Text-task versus Multimedia-task difference as a descriptive set contrast;
- raw transition count as a separate estimand;
- optional transition subclasses such as top↔bottom or text↔image transitions where geometry supports them.

If material type is nested in Task identity, neither its main effect nor Prompt × Material-type should be described as a pure causal modality effect. The interaction can at most describe whether the randomized prompt effect differs between the four task sets labelled Text and the four labelled Multimedia.

## Primary estimand

The intended primary estimand is the population-level **transition-rate ratio for Prompt versus Non-prompt**, averaged over the eight released learning tasks for the released target population.

The planned model structure is:

```text
transition_count
  ~ Prompt
  + Task identity
  + offset(log(stimulus_time))
  + participant random intercept
```

with a negative-binomial count family as the current primary candidate because transition counts are non-negative and overdispersion is plausible.

Task identity is included as a fixed blocking factor so content-specific baseline differences do not contaminate the randomized prompt contrast. Material type is not added as a main effect when it is perfectly nested in task identity because that parameter would be aliased with the task indicators.

The exact negative-binomial parameterization and fitting engine remain to be frozen after the structural count/exposure audit, but they must be chosen without reference to which measurement branches strengthen the prompt effect.

## Measurement decision families

### 1. Eye/event representation

The raw release provides separate left/right point-of-regard coordinates.

Possible branches may include separately justified left-eye, right-eye, and explicit binocular constructions, but no binocular averaging rule is assumed by default.

Any binocular rule must be specified as an analytical decision with its own missing-eye behavior.

### 1a. Pixel-to-visual-degree geometry

Angular detector thresholds require an explicit conversion. The source reports a 22-inch, 1600 × 900 display and an approximate 60–70 cm viewing distance, but no participant-specific distance.

The prespecified geometry sensitivity family is therefore:

- 60 cm: reported lower bound;
- 65 cm: labelled midpoint sensitivity branch;
- 70 cm: reported upper bound.

Assuming the physical panel aspect ratio matches the 1600:900 pixel aspect ratio, these correspond to approximately 34.40, 37.27, and 40.14 pixels per degree. The transform is reversible and retains the original pixel coordinates. The 65 cm branch must never be described as a measured participant distance.

### 2. Event detection

The SRL descriptor reports a BeGaze high-speed saccade-based configuration with:

- minimum saccade duration 22 ms;
- saccadic peak-velocity threshold **4 degrees/s**;
- minimum fixation duration 50 ms.

The value 4 degrees/s must be preserved exactly as reported. It is also a source discrepancy: multiple SMI BeGaze / RED250 reports use 40 degrees/s with the same 22 ms saccade and 50 ms fixation settings, and some describe 40 degrees/s as the default SMI implementation. The project must not silently correct 4 to 40 or claim that the descriptor's value is a standard I-VT threshold.

The released BeGaze event catalogue is therefore the faithful historical reference branch. Re-running eyeprocesspy I-VT at 4 degrees/s would not reproduce proprietary BeGaze High Speed Event Detection and is not an acceptable substitute.

The open comparator plan is now source-backed:

- threshold-matched simple I-VT at 30 degrees/s and 100 ms minimum fixation, explicitly without claiming equivalence to the full Tobii I-VT filter;
- threshold-matched simple I-VT at 40 degrees/s and 50 ms as an SMI-context sensitivity comparator, not a BeGaze reproduction;
- I-DT at 1 degree and 100 ms;
- REMoDNaV with published defaults, once the runtime version and dense-timebase checks are fixed;
- the package's robust-MAD adaptive detector remains an implementation diagnostic until its numerical parameters are frozen.

No detector may be added or removed because of its focal Prompt-versus-Non-prompt result.

### 3. AOI definition

The source now constrains the nominal AOI family more tightly than the initial audit assumed:

- four AOIs per learning slide;
- 1600 × 900 display;
- 359,550 px reported area for each AOI;
- usage notes explicitly describe transitions between different **quartiles of the learning slides**.

Therefore the nominal geometry family is frozen as **four slide quartiles** before focal effects are inspected.

The exact pixel boundary convention remains unresolved. Exact 1600 × 900 quarters are 800 × 450 = 360,000 px, whereas the reported area is exactly 799 × 450 = 359,550 px. This is consistent with a narrow seam/edge convention but does not identify its placement.

The analysis must therefore preserve all source-compatible boundary conventions until the released stimulus/archive metadata are inspected. Raw vendor AOI labels may be used as a reference-agreement audit, but not to manufacture the independent geometry or choose a boundary because it strengthens the focal effect.

After the nominal source-compatible set is frozen, a broader prespecified spatial perturbation envelope can quantify measurement uncertainty around the boundaries.

### 4. Data quality

The original study excluded recordings below an 80% tracking ratio and reports manual gaze-offset correction for correctable systematic distortion.

The historical 80% rule is a documented reference rule, not automatically the only defensible threshold.

The quality multiverse must be justified from methodological literature and frozen before focal branch effects are inspected.

No missing samples are converted to zero, and failed/insufficient-quality branches remain visible in the planned denominator.

### 5. Statistical model

The same model and focal contrast must be fitted across all estimand-preserving measurement branches.

The model must return:

- focal estimate;
- standard error;
- confidence or posterior interval;
- convergence/status;
- model N;
- any exclusion/attrition audit.

## Provenance limitation

The Scientific Data paper describes the release as providing original raw eye-tracking data, but it also reports manual BeGaze gaze-offset correction for correctable systematic distortion before final retention.

Until the archive documents whether the released point-of-regard coordinates precede or follow those corrections, this project must use the phrase sample-level raw export rather than claiming untouched sensor-native coordinates.

Consequently, the primary pilot can audit event-detection, AOI, quality-rule, and downstream inferential uncertainty. It must not claim to quantify uncertainty upstream of an unrecoverable manual offset correction.

## Required pre-analysis gates

Before computing the focal effect:

1. run `audit_srl_dataset.py` on the extracted Figshare release;
2. run `audit_srl_identifiability.py` and verify Prompt/Non-prompt is represented within every Task;
3. determine empirically whether `stimulus_type` varies within Task identity or is nested in it;
4. verify participant counts and each retained participant's eight-task structure;
5. verify raw sample columns, millisecond timestamps, and observed sampling intervals;
6. verify `stimulus_time` against raw trial duration and define the exposure discrepancy rule;
7. inspect all eight released Task images;
8. freeze source-compatible quartile AOI boundaries independently from focal outcomes;
9. document agreement/disagreement with released vendor AOI labels;
10. determine what gaze-offset-correction provenance is recoverable;
11. freeze detector specifications and the remaining gap/interpolation semantics;
12. freeze AOI perturbation envelope;
13. freeze quality-rule branches, including any recovery-calibrated threshold;
14. freeze the negative-binomial repeated-measures fitting engine and diagnostics;
15. record the complete planned universe before estimating branch-specific effects.

## Success criterion

The study is informative whether the focal Prompt-versus-Non-prompt transition-rate conclusion is stable or sensitive.

No branch will be added, removed, or reparameterized because it strengthens or weakens the effect.
