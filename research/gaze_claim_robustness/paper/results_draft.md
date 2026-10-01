# Results draft — CHI gaze-claim robustness study

## 4.1 Frozen SRL measurement universe

The primary SRL analysis evaluated 144 prespecified measurement specifications formed by crossing three event-detection branches, two eyes, three viewing-distance assumptions, two AOI geometries, two quality rules, and two cohort definitions. All 144 models converged with optimizer code 0 and positive-definite Hessians. Across this universe, 92/144 (63.9%) Prompt coefficients were positive and 52/144 (36.1%) were negative. The median Prompt log-rate ratio was 0.0340, corresponding to a median rate ratio (RR) of 1.0346. Point-estimate RRs ranged from 0.9440 to 1.2577. However, every Wald 95% confidence interval included the no-effect value (144/144; Figure 1). The frozen universe therefore exhibited sensitivity in point-estimate direction and magnitude without a corresponding change in interval-level inference.

## 4.2 Viewing geometry dominated SRL estimate sensitivity

The largest systematic shift was associated with the viewing-distance assumption used to convert pixels to degrees of visual angle. At 60 cm, all 48 specifications produced positive Prompt estimates and the median RR was 1.2131. At 65 cm, 28/48 estimates were positive and 20/48 negative, with a median RR of 1.0161. At 70 cm, 16/48 were positive and 32/48 negative, with a median RR of 0.9832. Thus, a defensible change in an unobserved viewing-geometry assumption was sufficient to move the median estimate from a positive association to approximately null and then slightly below the null.

Detector choice interacted with this geometry decision (Figure 2). At 65 cm, the I-VT 30°/s branch was negative in all 16 specifications (median RR = 0.9744), whereas the I-VT 40°/s branch was positive in all 16 (median RR = 1.0957); I-DT lay between them (12 positive, 4 negative; median RR = 1.0161). The separation persisted at 70 cm: the I-VT 30°/s branch remained negative in all 16 specifications (median RR = 0.9626), while I-VT 40°/s remained positive in 14/16 (median RR = 1.0532). AOI boundary convention was comparatively inert in this dataset, with a marginal mean log-rate-ratio difference of approximately 0.00015 between the two prespecified geometries. These patterns identify visual-angle geometry and detector choice as the principal sources of measurement sensitivity in the primary case rather than implying that any one branch is the correct analysis.

## 4.3 Independent MCFW-Gaze measurement generalization

The independent MCFW-Gaze validation tested whether detector-dependent representation differences persisted on a different eye tracker, nominal sampling rate, and set of interaction contexts. The frozen validation contained 28,170 planned detector-summary rows and the same number of detector-pair rows. Of these, 28,164 detector summaries (99.98%) and 27,660 detector-pair comparisons (98.19%) were evaluable. One source file with a non-increasing timestamp was retained as non-evaluable rather than repaired, and pairwise rows with no source-usable samples for the analyzed eye remained in the planned denominator.

Across evaluable detector-pair rows, the median sample-level fixation/non-fixation agreement was 0.7068. Positive-class overlap was substantially lower: the median fixation-state Jaccard index was 0.2724 and median Cohen's kappa was 0.1989. This difference indicates that raw state agreement can be dominated by shared non-fixation time and should not be interpreted as event-level equivalence.

The disagreement was structured rather than uniform across algorithms. I-VT 40°/s and I-DT were the closest pair, with median Jaccard values of 0.6144 for the left eye and 0.5344 for the right, and median kappa values of 0.4100 and 0.3942, respectively. I-VT 30°/s and I-DT were substantially farther apart (median Jaccard = 0.1623 left, 0.0968 right; kappa = 0.0874 left, 0.0665 right). I-VT 30°/s versus I-VT 40°/s was also weak-to-moderate (median Jaccard = 0.2432 left, 0.1737 right). The event summaries showed the same representation shift: median fixation counts were 3 and 2 per trial for I-VT 30°/s (left/right), 16 and 15 for I-VT 40°/s, and 21 and 22 for I-DT.

The relative detector pattern persisted across the six source-defined contexts (Figure 3). I-VT 40°/s versus I-DT remained the closest pair in natural-image viewing, gaze-pattern authentication, password entry, shopping, news browsing, and video viewing, whereas detector pairs involving I-VT 30°/s showed consistently lower fixation-state overlap. Context therefore modulated the magnitude of detector agreement but did not erase the broad ordering of measurement representations.

## 4.4 Data quality and detector disagreement

Higher analyzed-eye usable-gaze fractions were generally associated with smaller positive-class and event-summary divergence. For I-VT 40°/s versus I-DT, the Spearman correlation between usable fraction and fixation-count symmetric divergence was -0.559 for the left eye and -0.639 for the right; the corresponding correlations with Jaccard disagreement were -0.379 and -0.489. The same direction was present, though weaker, for the other detector pairs. In contrast, raw fixation/non-fixation agreement did not consistently improve with usable fraction, reinforcing that overall agreement is sensitive to the prevalence of the shared non-fixation state.

## 4.5 Cross-case result

The two empirical cases address different levels of the measurement problem. The SRL case demonstrates **claim propagation**: changing defensible measurement choices can alter the direction and magnitude of a substantive HCI effect estimate even when interval-level inference remains stable. The MCFW-Gaze case demonstrates **measurement generalization**: detector-dependent event representations persist in an independent dataset with different hardware, sampling rate, and interaction contexts. Together, the cases show why eye-tracking measurement decisions should be represented as explicit uncertainty in the analysis pipeline rather than treated as an invisible preprocessing constant.

## Figure captions

**Figure 1. SRL specification curve across the 144 frozen measurement specifications.** Points show the Prompt log-rate-ratio estimate and vertical lines show Wald 95% confidence intervals. Marker shape indicates the prespecified viewing-distance assumption used for pixel-to-degree conversion. Estimates span both sides of zero, but every interval includes zero.

**Figure 2. Joint sensitivity of the SRL Prompt estimate to detector and viewing-distance assumptions.** Points show the median Prompt rate ratio within each detector × viewing-distance cell; vertical lines show the interquartile range across eye, AOI, quality-rule, and cohort branches. The dashed horizontal line marks RR = 1.

**Figure 3. Independent MCFW-Gaze detector overlap across source-defined interaction contexts.** Lines show context-specific median fixation-state Jaccard overlap for each detector pair, separately for left and right eyes. I-VT 40°/s and I-DT are consistently closer than pairs involving I-VT 30°/s, although the magnitude of overlap varies by context.
