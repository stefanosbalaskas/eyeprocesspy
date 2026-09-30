# MCFW-Gaze independent raw-signal validation protocol

## Role in the gaze-claim robustness program

MCFW-Gaze is an **independent technical/generalization case**, not a second
substantive HCI treatment-effect study.

The SRL pilot asks whether a prespecified HCI/learning conclusion changes across
defensible measurement pipelines. MCFW-Gaze instead asks whether the
measurement-stage instability implicated by that pilot generalizes to a
different device, sampling rate, task ecology, and raw-data quality profile.

No artificial interface A/B contrast will be created.

## Source design fixed before detector results

The public MCFW-Gaze v3 record describes:

- Tobii Pro Fusion, nominal 120 Hz;
- binocular continuous raw gaze exports from Titta/PsychoPy;
- 1920 x 1080 display;
- physical display size approximately 31.0 x 17.5 cm;
- viewing distance approximately 65 cm;
- 15 participants;
- controlled natural-image, gaze-pattern authentication, and password tasks;
- 600 s naturalistic shopping, news-browsing, and video-viewing sessions.

The released contexts are treated as **context families**, not randomized
treatments.

The one-time archive audit completed before detector execution found:

- 4,695 raw trial files across 15 participants and all six planned context
  families;
- 6,304,159 source samples;
- zero raw-file audit failures;
- one file with a non-increasing device timestamp step;
- a median raw device timestamp increment of 8,333 units across the archive.

The source archive is therefore usable for the planned validation, while the
single non-monotonic file remains explicitly non-evaluable under the frozen
timestamp rule.

## Primary methodological question

> Does detector-dependent event representation remain materially variable in
> an independent 120 Hz Tobii dataset, and is the amount of detector
> disagreement associated with recorded gaze quality and interaction context?

This question is informative whether disagreement is large, small, uniform,
or context-dependent.

## Frozen detector set

Use the same three open detector comparators retained in the SRL primary
universe:

1. simple I-VT, 30 deg/s, 100 ms minimum fixation, explicit 75 ms run-gap;
2. simple I-VT, 40 deg/s, 50 ms minimum fixation, explicit 75 ms run-gap;
3. I-DT, 1 degree dispersion, 100 ms minimum fixation.

No detector may be added or removed based on the observed MCFW results.

The package's adaptive detector and any vendor event labels may be retained as
diagnostics/reference evidence outside the primary comparison.

## Eye representation

Left and right eyes are analyzed separately.

No implicit binocular averaging, best-eye selection, interpolation, smoothing,
or imputation is permitted in the primary comparison.

A binocular construction would be an estimand-changing extension and is not
part of the minimum validation universe.

## Timestamp provenance and frozen scale

The Tobii Pro SDK documents device and system timestamps in microseconds. The
archive-level median device timestamp step is 8,333 raw units, which is
consistent with the source-reported 120 Hz acquisition rate under a
microsecond scale.

The primary adapter therefore freezes:

```text
timestamp_scale_seconds = 1e-6
```

No timestamp scale is estimated from detector outcomes.

Files with non-finite or non-increasing timestamps remain non-evaluable; they
are not reordered or repaired.

## Angular geometry

The source-reported physical screen geometry and approximately 65 cm viewing
distance define the single nominal visual-angle conversion:

```text
screen = 1920 x 1080 px
physical display = approximately 310 x 175 mm
viewing distance = approximately 650 mm
```

Normalized display coordinates are converted axis-wise to signed visual
degrees about screen center using the reported horizontal and vertical physical
pixel densities. The source normalized coordinates are retained.

Unlike SRL, the MCFW primary validation does **not** introduce additional
viewing-distance branches. The source gives one approximate distance but no
participant-specific measurements or defensible uncertainty range.

## Shared detector-input eligibility

The raw archive supplies explicit validity and availability flags. For each
eye, a source sample is detector-usable only when:

- the eye-specific gaze point is marked valid;
- the gaze point is marked available;
- both normalized coordinates are finite;
- the timestamp is finite.

Unusable observations are **not** interpolated, imputed, reordered, or
converted into synthetic samples.

For detector execution only, usable observations are divided into maximal
contiguous runs. A new run starts after any unusable source sample or when the
observed timestamp gap exceeds 75 ms. This common pre-detector rule prevents
all three algorithms from bridging source-unusable intervals and avoids
algorithm-specific treatment of validity flags.

The untouched source timeline remains the denominator for quality accounting
and the reference timeline onto which detector events are mapped.

## Primary detector-disagreement outcomes

All outcomes are trial-level and are computed only after the source audit and
the above provenance decisions are frozen.

### 1. Pairwise sample-level fixation agreement

For each detector pair, eye, participant, and trial:

- fixation-state agreement proportion;
- fixation-state Jaccard index;
- Cohen's kappa where defined.

Detector events are mapped back onto the existing source timestamps. Pairwise
agreement is calculated only over source-usable samples for that eye. This
prevents invalid/unavailable observations from being counted as agreement
merely because both algorithms label them non-fixation.

If neither detector assigns any usable sample to fixation, fixation-state
Jaccard is undefined rather than automatically set to one. Cohen's kappa is
undefined when its expected-agreement denominator is zero.

### 2. Event-summary divergence

For each detector, eye, participant, and trial:

- fixation count;
- fixation rate per minute;
- median fixation duration;
- total fixation-time proportion.

For every detector pair and each quantity, report:

- absolute difference;
- symmetric relative difference

```text
2 * |A - B| / (|A| + |B|)
```

when the denominator is non-zero. This definition is invariant to detector
ordering and does not designate one detector as ground truth.

### 3. Context sensitivity

Summarize the above across the frozen source-defined context families:

- natural image;
- gaze-pattern authentication;
- password entry;
- web shopping;
- web news;
- web video.

Context is descriptive unless the source design supports a stronger repeated
comparison. No causal wording is allowed.

## Quality association

The primary quality covariates are source-native validity/availability measures:

- left-eye usable fraction;
- right-eye usable fraction;
- either-eye usable fraction;
- both-eyes usable fraction.

For the analyzed eye, the continuous detector-analysis usable fraction is also
retained.

Detector disagreement versus analyzed-eye usable fraction is summarized with
Spearman's rho, overall by detector pair/eye and again descriptively within
context family. No p-value threshold is used and the association is not
interpreted causally.

### Published quality-table semantic gate

The pre-analysis semantic audit found that the separately published quality
tables cannot be treated as conventional proportions without clarification:

- `prop_missing_data` contains 9,196 values outside [0, 1];
- `Prop_data_loss_left_eye` contains 9 values outside [0, 1];
- `Prop_data_loss_right_eye` contains 5 values outside [0, 1].

Those fields are therefore quarantined from primary filtering and quality
association. They may be reported as source-semantic anomalies, but they are
not clipped, rescaled, or reinterpreted after seeing detector results.

The primary quality analysis uses only the source-native sample-level
validity/availability flags whose operational meaning is explicit in the raw
files.

## Reporting

The MCFW result must report:

- all planned detector pairs;
- all source context families represented in the archive;
- trial/file failures and reasons;
- quality distributions;
- detector agreement/divergence distributions;
- context-specific profiles;
- eye-specific profiles;
- the full planned denominator, including non-evaluable rows.

No p < .05 voting and no automatic robust/fragile label.

## Contribution boundary

MCFW-Gaze can support the claim that detector/measurement sensitivity
generalizes across hardware and interaction contexts.

It cannot by itself validate the SRL Prompt effect or establish that a
particular detector is scientifically correct, because the dataset does not
provide a randomized interface treatment or fixation ground truth for all
contexts.
