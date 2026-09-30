# Frozen MCFW-Gaze independent validation result

MCFW-Gaze is an independent **measurement-generalization** case. It does not
replicate the SRL Prompt effect, create an interface treatment contrast, or
identify a scientifically correct detector.

## Frozen execution

- analysis commit: `878e351a82a48665990955853c55b2315b10fd27`
- workflow run: `36776432517`
- workflow artifact: `11127976095`
- artifact SHA-256:
  `ed2af7a036d1fb9a62b5226b71b6f02d007f87a9d23b1970828f9c9742b93020`
- source archive: Zenodo record `20300972`
- source archive size: 1,015,028,669 bytes
- source archive MD5: `5e04a7b5e50f7508775e1daea527246b`
- source archive SHA-256:
  `c474c38a1b0e8e0a04a648d67e565c36b9c6b81b0e59e9360e84d42ebf14dc89`
- 4,695 raw trial files; 15 participants; 6,304,159 source samples
- frozen timestamp conversion: `1e-6` seconds per device timestamp unit
- nominal geometry: 1920 x 1080 px, approximately 310 x 175 mm,
  approximately 650 mm viewing distance
- detector input: contiguous source-usable runs; no interpolation or imputation
- separately published ambiguous quality-proportion fields remained quarantined

No execution/protocol file was changed after the validation result was observed.

## Planned denominator and evaluability

The frozen validation planned 28,170 detector-summary rows and 28,170
detector-pair rows.

- detector summaries: 28,164/28,170 evaluable (99.979%)
- pairwise comparisons: 27,660/28,170 evaluable (98.190%)
- 6 detector-summary/pair rows trace to one source file with a non-increasing
  device timestamp (`participant_005/news.tsv`, both eyes); it was retained as
  non-evaluable and was not reordered or repaired
- 504 pairwise rows were non-evaluable because the analyzed eye contained zero
  source-usable samples; those rows remain in the planned ledger
- the detector failure ledger contains 2 source-file/eye failures, both from the
  same non-increasing-timestamp file

## Overall detector representation

Across evaluable detector-pair rows, the pooled medians were:

- sample-level fixation/non-fixation agreement: 0.706767
- fixation-state Jaccard: 0.272358
- Cohen's kappa: 0.198911

Raw agreement therefore overstates event-level equivalence because long
non-fixation periods can agree even when fixation representations differ.

The pair structure is highly informative:

- I-VT 40 deg/s vs I-DT is the closest pair: median Jaccard 0.614 (left) and
  0.534 (right), with median kappa 0.410 and 0.394
- I-VT 30 deg/s vs I-DT is much farther apart: median Jaccard 0.162 (left) and
  0.097 (right), with median kappa 0.087 and 0.066
- I-VT 30 deg/s vs I-VT 40 deg/s is also weak-to-moderate: median Jaccard
  0.243 (left) and 0.174 (right)

The event summaries explain this separation. Median fixation counts per trial
were approximately 2-3 for I-VT 30 deg/s, 15-16 for I-VT 40 deg/s, and 21-22
for I-DT. Thus detector choice changes the substantive event representation,
not only boundary placement.

## Context and quality

The I-VT 40 deg/s vs I-DT relationship remains comparatively close across all
six source-defined contexts. Context-specific median Jaccard values range from
approximately 0.526 to 0.755. In contrast, comparisons involving I-VT 30 deg/s
remain substantially lower across contexts.

Higher analyzed-eye usable-gaze fractions are generally associated with
smaller positive-class/event-summary disagreement. For example, the Spearman
association between usable fraction and fixation-count symmetric divergence is
-0.559 (left) and -0.639 (right) for I-VT 40 deg/s vs I-DT, and -0.408/-0.454
for I-VT 30 deg/s vs I-DT. Similar negative associations occur for Jaccard
disagreement and fixation-time divergence.

The raw agreement proportion can show the opposite association for some pairs,
which is expected when shared non-fixation dominates the denominator. For the
paper, positive-class overlap and event-summary divergence should therefore be
treated as the more informative measurement-sensitivity quantities.

## Methodological interpretation

The independent validation supports a **measurement-generalization** result:
defensible open event-detection choices can produce materially different gaze
event representations on a different eye tracker, sampling rate, and task
ecology. The effect is structured rather than arbitrary: two detector
specifications are substantially closer to each other, while the lower
velocity-threshold I-VT branch yields a markedly sparser fixation
representation.

This complements, but does not replicate, the SRL claim-level result. SRL
shows that measurement decisions can propagate to the direction and magnitude
of an HCI effect estimate; MCFW-Gaze shows that detector-dependent
representation differences persist in an independent raw-signal setting.
