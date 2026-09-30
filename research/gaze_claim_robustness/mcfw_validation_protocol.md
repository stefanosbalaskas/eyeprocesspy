# MCFW-Gaze independent raw-signal validation protocol

## Role in the gaze-claim robustness program

MCFW-Gaze is an **independent technical/generalization case**, not a second
substantive HCI treatment-effect study.

The SRL pilot asks whether a prespecified HCI/learning conclusion changes across
defensible measurement pipelines. MCFW-Gaze instead asks whether the
measurement-stage instability implicated by that pilot generalizes to a
different device, sampling rate, task ecology, and raw-data quality profile.

No artificial interface A/B contrast will be created.

## Source design fixed before audit results

The public MCFW-Gaze record describes:

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

## Timestamp and angular geometry gates

The native timestamp unit must be established from the released documentation
or archive metadata before detector execution.

The source-reported physical screen geometry and approximately 65 cm viewing
distance define the nominal pixel/normalized-coordinate to visual-angle
conversion.

Unlike SRL, the MCFW primary validation does **not** introduce additional
viewing-distance branches unless the archive or source documentation establishes
a defensible uncertainty range. A single approximate value must not be
misrepresented as participant-specific measured distance.

Files with non-finite or non-increasing timestamps remain non-evaluable; they
are not reordered or repaired.

## Primary detector-disagreement outcomes

All outcomes are trial-level and are computed only after the raw archive audit
passes.

### 1. Pairwise sample-level fixation agreement

For each detector pair, eye, participant, and trial:

- fixation-state agreement proportion;
- fixation-state Jaccard index;
- Cohen's kappa where defined.

The sample timeline is shared; detector outputs are mapped back to sample-level
fixation/non-fixation state without inventing samples.

### 2. Event-summary divergence

For each detector, eye, participant, and trial:

- fixation count;
- fixation rate per minute;
- median fixation duration;
- total fixation-time proportion.

For each detector pair, report absolute and relative differences in these
quantities.

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

Published trial/validation quality tables may be used only after their column
semantics pass the independent semantic audit.

The validation should ask whether detector disagreement increases as usable
gaze decreases. Quality variables are not thresholded merely because a
convenient cutoff exists.

## Reporting

The MCFW result must report:

- all planned detector pairs;
- all source context families represented in the archive;
- trial/file failures and reasons;
- quality distributions;
- detector agreement/divergence distributions;
- context-specific profiles;
- eye-specific profiles.

No p < .05 voting and no automatic robust/fragile label.

## Contribution boundary

MCFW-Gaze can support the claim that detector/measurement sensitivity
generalizes across hardware and interaction contexts.

It cannot by itself validate the SRL Prompt effect or establish that a
particular detector is scientifically correct, because the dataset does not
provide a randomized interface treatment or fixation ground truth for all
contexts.
