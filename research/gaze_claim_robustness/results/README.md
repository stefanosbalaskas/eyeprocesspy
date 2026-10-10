# Frozen SRL empirical result snapshot

This directory records the first fully frozen empirical output of the
cross-stage gaze-claim robustness pilot.

## Provenance

- measurement workflow run: `36683906120`
- measurement artifact:
  `srl-measurement-universe-cfd8f8f99c5d4fdc19399661e6ec0f24a0080bb8`
- model workflow run: `36708103076`
- model analysis head: `5fad0f192067ca74277c4b5f2bec462514395124`
- estimator: `glmmTMB 1.1.14`, NB2/log link
- planned model specifications: 144
- successful/converged specifications: 144
- optimizer convergence code: 0 for every specification
- positive-definite Hessian: yes for every specification

The complete workflow artifact remains the canonical machine output. The files
here are compact frozen summaries for the research record.

## Primary empirical pattern

Across the 144 prespecified specifications:

- 92 point estimates are positive and 52 are negative;
- the median Prompt log-rate ratio is 0.034019;
- the median rate ratio is 1.034604;
- point-estimate rate ratios range from 0.943965 to 1.257701;
- **all 144 Wald 95% confidence intervals include the null**.

The result therefore has two distinct stability properties that must not be
collapsed into one binary label:

1. **Point-estimate direction and magnitude are measurement-sensitive.**
2. **Interval-level inference is stable across the frozen universe in that no
   specification's 95% Wald interval excludes the no-effect value.**

This is descriptive robustness evidence, not a probability that the scientific
claim is true or false.

## Dominant measurement decision

The largest descriptive marginal shift is the assumed viewing distance used for
pixel-to-visual-degree conversion:

- 60 cm: 48/48 positive point estimates; median log-rate ratio 0.193215;
  median rate ratio 1.213144.
- 65 cm: 28/48 positive and 20/48 negative; median log-rate ratio 0.016007;
  median rate ratio 1.016136.
- 70 cm: 16/48 positive and 32/48 negative; median log-rate ratio -0.016952;
  median rate ratio 0.983191.

Detector choice is the second-largest decision family. The interaction is
substantive: at 65 and 70 cm the simple 30 deg/s I-VT comparator is uniformly
negative, whereas the 40 deg/s comparator is predominantly positive.

The AOI boundary convention is essentially inert in this pilot: the marginal
mean log-rate-ratio difference between the exact-quarter and
published-area-compatible centered-seam geometries is approximately 0.000148.

Quality-rule, eye, and cohort decisions shift estimates modestly relative to
viewing distance and detector choice.

## Non-evaluable measurement evidence

The frozen measurement table contains 23,616 planned rows. Of these, 23,292
have outcome status `ok`. The remaining 324 rows arise from nine
participant-by-Task trials whose released timestamps are not strictly
increasing. Those trials were retained as explicit non-evaluable evidence and
were not reordered, repaired, or converted to zero.

The four prespecified model-evaluable row counts are exactly:

- exact_raw82 / released_sample: 647
- exact_raw82 / trial_80_sensitivity: 634
- nominal250_77 / released_sample: 608
- nominal250_77 / trial_80_sensitivity: 595

## Independent MCFW-Gaze measurement validation

The independent MCFW-Gaze validation was executed only after its timestamp,
geometry, detector-input, quality, outcome, and reporting decisions were
frozen.

Provenance:

- analysis commit: `878e351a82a48665990955853c55b2315b10fd27`
- workflow run: `36776432517`
- workflow artifact: `11127976095`
- workflow artifact SHA-256:
  `ed2af7a036d1fb9a62b5226b71b6f02d007f87a9d23b1970828f9c9742b93020`
- planned detector summaries: 28,170; evaluable: 28,164
- planned detector-pair rows: 28,170; evaluable: 27,660

Across evaluable pairwise rows, median sample-level agreement was 0.706767,
but median fixation-state Jaccard was 0.272358 and median Cohen's kappa was
0.198911. This distinction matters because raw agreement can be dominated by
shared non-fixation time.

The detector differences are structured. I-VT 40 deg/s and I-DT are the
closest pair (median Jaccard 0.614 left / 0.534 right), whereas comparisons
involving I-VT 30 deg/s show substantially lower positive-class overlap. Median
fixation counts are approximately 2-3 per trial for I-VT 30 deg/s, 15-16 for
I-VT 40 deg/s, and 21-22 for I-DT.

Higher analyzed-eye usable-gaze fractions are generally associated with lower
positive-class and event-summary divergence. For I-VT 40 deg/s versus I-DT,
the usable-fraction association with fixation-count symmetric divergence is
rho=-0.559 for the left eye and rho=-0.639 for the right eye.

MCFW-Gaze is not treated as a replication of the SRL Prompt effect. Its role is
independent measurement generalization: detector-dependent event
representations persist across a different tracker, sampling rate, and task
ecology. The canonical full workflow artifact remains outside Git; only compact
summaries, provenance, and hashes are stored here.

