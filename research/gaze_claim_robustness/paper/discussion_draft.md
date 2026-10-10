# Discussion draft — CHI gaze-claim robustness study

## 5.1 Eye-tracking measurement choices can propagate into substantive HCI conclusions

The central result is not that one detector, viewing distance, eye, AOI convention, or quality rule is preferable. It is that choices commonly treated as preprocessing details can propagate into the quantity that an HCI study ultimately interprets. In the SRL case, the same prespecified Prompt/Non-prompt contrast moved from consistently positive estimates at 60 cm to mixed estimates at 65 cm and predominantly negative estimates at 70 cm. The change arose before any alteration to the statistical estimand or model family: it followed from a defensible change in the geometry used to transform gaze coordinates and, secondarily, from the event-detection rule applied to those coordinates.

This distinction matters for reproducibility. Two analysts can use the same raw recording, the same experimental contrast, the same outcome definition, and the same inferential model yet obtain meaningfully different point estimates because they instantiated the measurement pipeline differently. Reproducibility therefore requires more than reporting the final model formula. It also requires making consequential measurement assumptions explicit and, when more than one assumption is defensible, evaluating how conclusions behave across that uncertainty.

## 5.2 Robustness is multidimensional rather than a binary property

The SRL universe illustrates why a single robust/fragile label would discard important information. Direction was not stable: 92 specifications were positive and 52 negative. Magnitude also varied, with point-estimate rate ratios ranging from 0.944 to 1.258. Yet interval-level inference was completely stable in a different sense: every 95% Wald interval included the null. These are not contradictory findings. They describe different properties of the same result.

The 92/52 split is also not an invariant “robustness percentage.” In a Cartesian universe, branch frequencies depend partly on how many levels each decision family contributes, so changing grid granularity can change the fraction of positive specifications without changing any underlying fitted branch. We therefore use the count only to describe this frozen decision inventory and interpret family-level patterns separately.

For HCI robustness analyses, direction, magnitude, uncertainty, model evaluability, sample support, and the source of variation should therefore be reported separately. A specification frequency is descriptive evidence about a declared decision universe; it is not a posterior probability that an effect exists. Likewise, a branch crossing a conventional significance threshold does not by itself establish substantive fragility. The relevant question is what aspect of the scientific conclusion changes, under which defensible decisions, and by how much.

## 5.3 Viewing geometry is part of the measurement model

The largest SRL sensitivity arose from viewing distance because the detector thresholds were expressed in visual degrees while the released gaze coordinates were pixel based. Under that combination, viewing geometry is not merely display metadata. It helps determine the mapping from recorded position to angular displacement and therefore the velocity or dispersion presented to an event detector. Different plausible distances can consequently change event segmentation, AOI transition counts, and the downstream effect estimate.

This result should not be generalized into a claim that viewing distance will dominate every eye-tracking study. Its importance depends on the coordinate system, hardware, detector, task, and whether participant-specific geometry is measured. The practical implication is narrower: when angular quantities are derived from incompletely observed viewing geometry, that uncertainty should be represented explicitly rather than hidden inside a single conversion constant.

## 5.4 Detector disagreement is a representation problem, not simply an accuracy problem

The MCFW-Gaze validation clarifies the detector component of the SRL result. On independent 120 Hz Tobii data, the three frozen detector branches did not converge on a common fixation representation. I-VT 40°/s and I-DT were substantially closer to each other than either was to the I-VT 30°/s branch, and the latter produced a markedly sparser fixation sequence. The difference therefore concerns the representation of behavior itself, not only small disagreements at event boundaries.

The validation also shows why overall sample-level agreement can be misleading. A large fraction of an eye-tracking stream is often classified as non-fixation by both algorithms, so raw agreement can remain moderately high even when positive fixation overlap is low. Positive-class measures such as fixation-state Jaccard and downstream event summaries exposed disagreements that raw agreement obscured. For detector-sensitivity work, reporting only overall agreement can therefore create an overly reassuring picture of measurement equivalence.

At the same time, MCFW-Gaze does not provide a universal fixation ground truth. The analysis cannot establish that the closer I-VT 40°/s/I-DT pair is more scientifically correct, nor that the sparse I-VT 30°/s representation is incorrect. Its contribution is diagnostic: reasonable open algorithms can instantiate materially different event representations on the same source signal, and those differences persist across multiple interaction contexts.

## 5.5 Data quality changes how detector disagreement should be interpreted

Detector divergence generally decreased as the fraction of source-usable gaze increased, particularly for fixation-count and Jaccard-based disagreement. This suggests that some apparent method disagreement is amplified when the algorithms operate on less complete source signals. However, the relationship was not captured reliably by raw fixation/non-fixation agreement, again because the shared negative state dominates that denominator.

This observation supports a separation between two questions that are often collapsed: whether a recording is usable enough for a declared analysis, and whether two measurement algorithms produce equivalent event representations on the usable signal. Quality filtering cannot be assumed to resolve detector uncertainty, and detector sensitivity cannot substitute for a transparent quality account. Both belong in the measurement audit.

## 5.6 A claim-centered robustness workflow

The methodological contribution of this work is a claim-centered workflow for propagating measurement uncertainty through eye-tracking analysis. The workflow begins by defining a substantive claim contract and then constructs a finite decision universe around choices that are consequential, scientifically defensible, and auditable before the focal result is inspected. It distinguishes measurement-definition alternatives from sample-definition sensitivities and keeps analyses that change the estimand itself outside the primary claim-stability denominator. Each planned specification is carried through the declared focal outcome/model contract, with failed or non-evaluable branches retained rather than disappearing from the denominator. The result is summarized at the level of the scientific claim: direction, magnitude, uncertainty, evaluability, sample support, and the decision families associated with variation.

This differs from using a multiverse simply to enumerate preprocessing alternatives. The unit of interpretation is not the number of pipelines that produce a preferred threshold crossing. Instead, the analysis asks how a prespecified substantive conclusion behaves as measurement construction and explicitly declared sample-support assumptions vary. The independent MCFW-Gaze case then separates general measurement behavior from the idiosyncrasies of the primary HCI contrast.

For practice, the workflow implies five reporting commitments: state the assumptions that map raw gaze to the analyzed construct; distinguish measurement-definition alternatives, sample-definition sensitivities, and estimand-changing analyses; preserve the planned denominator including failures; report multiple dimensions of claim stability rather than a binary robustness verdict; and avoid selecting a pipeline retrospectively because it produces the most convenient conclusion.

## 5.7 Scope and limitations

Several boundaries constrain the present evidence. First, the SRL case is one substantive dataset and one focal transition-rate contrast. The result demonstrates the possibility and structure of measurement-to-claim propagation, not the prevalence of such sensitivity across HCI eye-tracking studies. Second, the three detector branches were intentionally limited to a small open comparison set that could be executed consistently on the released data. They do not exhaust defensible event-detection algorithms.

Third, the SRL viewing-distance branches represent plausible analytical assumptions rather than participant-specific measured distances. That is precisely why they are informative as a sensitivity dimension, but they should not be interpreted as evidence that any participant actually viewed the display at 60, 65, or 70 cm. Fourth, MCFW-Gaze provides independent raw-signal generalization but no universal fixation ground truth and no matched replication of the SRL interface contrast. The two cases therefore support complementary claims rather than direct replication.

Finally, the decision universe is intentionally finite. A robustness analysis can itself become researcher-flexible if alternatives are added after outcomes are inspected. The present workflow addresses this by freezing decision registries and execution rules before the relevant results are examined. Future applications should preserve the same separation between pre-result uncertainty specification and post-result interpretation.

## 5.8 Implications for HCI research practice

For studies in which eye-tracking measures are used to support claims about interface use, attention, search, learning, trust, or decision processes, the measurement pipeline should be treated as part of the inferential design. This does not imply that every study requires a large Cartesian multiverse. When a parameter is measured, theoretically fixed, or empirically inconsequential, a sensitivity branch may add little value. The stronger requirement is to identify assumptions that are both uncertain and capable of changing the operationalized behavior.

A practical workflow is therefore selective rather than maximal: document the raw-to-measure mapping; identify the few decisions with genuine scientific uncertainty; freeze plausible alternatives before inspecting the focal result; propagate those alternatives through the same declared focal contrast and outcome/model contract; label sample-definition sensitivities explicitly when they alter empirical support; and report where the conclusion changes. In this form, measurement robustness becomes a reproducibility tool for HCI rather than an exercise in generating many analyses.
