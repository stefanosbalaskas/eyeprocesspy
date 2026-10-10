# Frozen SRL release audit record

Status: **pre-results structural evidence**. No focal Prompt-versus-Non-prompt effect was fitted or inspected when this record was frozen.

## Source archive

- Dataset: Juřík et al. SRL eye-tracking release
- Dataset DOI: 10.6084/m9.figshare.28304069
- Figshare article ID: 28304069
- Figshare file ID: 54379415
- File name: `ET_data_new.zip`
- Published/observed size: 927,921,378 bytes
- Published/observed MD5: `ba5fcbc6ac4559c6529c62ddd74b4b8a`
- Observed SHA-256: `8906b3196faf65cb0bc401c0919a0fe075369c998ac431f67e46a39662fe7e1a`

The downloaded binary matched both the Figshare-reported byte size and MD5 before extraction.

## Successful readiness audit

GitHub Actions workflow: **SRL archive audit**, run #4  
Run ID: `36635156168`  
Evidence head: `4c8da6d05070ce88ac51983c84277858a5c76a79`

Result: **0 unresolved BLOCKER gates**.

The successful readiness table established:

- actual archive structure: pass;
- randomized Prompt estimand identifiability: pass;
- AOI geometry multiverse: frozen;
- I-VT gap semantics: frozen;
- released-cohort/timebase validation: pass;
- primary glmmTMB NB2 engine: frozen.

Manual gaze-offset-correction provenance remains an explicit **LIMITATION**, not a silently resolved fact. The analysis scope is therefore downstream of the released sample coordinates unless additional provenance becomes available.

## Cross-source identity findings

Released identity counts:

| Evidence source | Participant IDs |
| --- | ---: |
| `participants.csv` | 110 |
| `stimuli.csv` | 110 |
| raw participant files | 83 |
| event-data IDs | 84 |
| event IDs with all 8 Task files | 84 |
| exact raw + participant/stimulus metadata intersection | **82** |
| exact event + metadata intersection | 82 |

The exact raw cohort is balanced:

- Prompt: **41**
- Non-prompt: **41**

Unresolved release identities are not repaired by similarity or guessing:

- raw/event ID **945** is absent from `participants.csv`;
- event ID **494** is absent from `participants.csv` and has no raw file;
- all raw filenames agree with their embedded participant ID;
- no raw participant lacks event files.

Accordingly, the primary raw-sample cohort is `exact_raw82`, not the descriptor-level nominal count of 84.

## Task/modality identifiability

Prompt versus Non-prompt is crossed with all eight learning Tasks and remains eligible as the randomized primary contrast.

Material format is nested in Task identity in the released metadata:

| Task | Released material type |
| --- | --- |
| Task_1 | Text |
| Task_2 | Multimedia |
| Task_3 | Text |
| Task_4 | Multimedia |
| Task_5 | Text |
| Task_6 | Multimedia |
| Task_7 | Text |
| Task_8 | Multimedia |

Therefore Text-versus-Multimedia is **not** treated as a separately identified causal format effect after controlling Task identity.

## Task-level timebase findings

All 83 raw participant files contain all eight learning Task segments.

Observed participant-level sampling classification:

- all eight Tasks nominally ~250 Hz: **78/83**
- all eight Tasks nominally ~60 Hz: **5/83**
- all eight Tasks strictly dense/regular 250 Hz: **0/83**
- all eight Tasks losslessly regularizable to a strict 4 ms grid without a new missing-slot decision: **0/83**

The five ~60 Hz participants are:

- 236
- 324
- 342
- 355
- 464

Within the exact raw+metadata cohort, the nominal-250-Hz sensitivity cohort contains **77** participants:

- `exact_raw82`: 82 participants, 41 Prompt / 41 Non-prompt;
- `nominal250_77`: 77 exact participants whose eight learning Tasks are nominally ~250 Hz.

Nine of the 83 raw files contain one non-positive timestamp step somewhere in their eight Task segments. These are preserved as audit evidence rather than silently reordered or repaired.

## Detector consequence

The primary denominator excludes REMoDNaV 1.1.2 because the actual release does not satisfy its dense-regular-sampling requirement without introducing a new resampling/regularization decision.

REMoDNaV remains an optional diagnostic outside the primary denominator.

The primary detector set is therefore:

1. simple I-VT 30°/s, 100 ms minimum fixation, 75 ms run-gap;
2. simple I-VT 40°/s, 50 ms minimum fixation, 75 ms run-gap;
3. I-DT 1°, 100 ms minimum fixation.

The I-VT and I-DT implementations use observed timestamps for velocity/duration. Sampling-rate discrepancies are retained as detector warnings rather than silently resampled.

## Frozen primary universe after release audit

The real-release audit changed the structurally compatible primary denominator to:

```text
3 detector specifications
× 2 eyes
× 3 viewing-distance assumptions
× 2 AOI geometries
× 2 quality rules
× 2 cohort definitions
= 144 planned specifications
```

All 144 specifications have unique deterministic hashes.

This expansion was driven by source discrepancies discovered **before** focal Prompt effects were estimated: the exact 82-person cohort and the 77-person nominal-250-Hz sensitivity cohort.

## Analysis lock status at freeze

At this audit freeze:

- no Prompt model had been fitted;
- no specification curve existed;
- no detector/AOI/quality/cohort branch had been selected using the Prompt result;
- no branch had been removed because it weakened or reversed the focal result.

The next permitted stage is measurement-only computation of fixation/AOI transition outcomes across the frozen raw-data branches.
