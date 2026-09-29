# CHI pilot: gaze-claim robustness

Status: experimental research plan. This directory is not part of the stable public API and does not alter release semantics.

## Research question

Primary methodological question:

> To what extent do defensible eye-tracking measurement decisions change the direction, magnitude, uncertainty, or substantive interpretation of an HCI effect?

Secondary questions:

1. Which measurement stages contribute the largest descriptive shifts in the focal estimate?
2. Does sensitivity differ across HCI tasks, interface geometries, or data-quality conditions?
3. For gaze-latency claims, how do conclusions differ when valid non-events are modeled with an estimand that retains censoring rather than silently dropping non-fixated trials?

The third question is a separate estimand comparison. It must not be mixed into the same robustness denominator as event-only latency.

## Novelty boundary

This project must not claim that multiverse analysis is new to eye tracking.

Godwin, Lee, and Drieghe (2025) evaluated 1,890 defensible cleaning/analysis pipelines for reading eye movements and found a highly stable word-frequency effect whose estimated magnitude nevertheless varied materially across pipelines.

The intended contribution here is different:

- HCI rather than a single reading effect;
- several upstream measurement stages rather than one large cleaning/model grid alone;
- explicit propagation to a prespecified substantive HCI claim;
- a fixed planned denominator that retains failed/non-converged branches;
- explicit separation of estimand-preserving and estimand-changing alternatives;
- provenance linking the final coefficient to detector, AOI, quality, and model specifications.

Reference: https://doi.org/10.3758/s13428-025-02689-0

## Candidate dataset audit

### 1. Emoji Keyboard Gaze Search — preferred first feasibility pilot

Source: https://doi.org/10.5281/zenodo.21059646

Confirmed from the public record:

- two task types: Scenario-based Selection and Target Search;
- Android vertical and iOS horizontal layouts;
- raw workbooks with six task sheets;
- sample index and time from task onset;
- gaze coordinates in pixels and normalized screen units;
- angular velocity;
- existing I-VT event labels using a 30 degree/s threshold;
- separate scroll-position logs;
- 684 processed participant/task files.

Why it is attractive:

- direct interface/layout context;
- raw sample stream can support detector re-analysis;
- existing I-VT labels offer a useful reference branch without treating them as ground truth;
- repeated tasks/layouts may allow a clear HCI contrast after the participant/task design is confirmed.

Current gates before a scientific claim is frozen:

- recover or reconstruct the exact emoji/target geometry needed for AOIs;
- confirm participant overlap and assignment across Android/iOS and task types;
- verify task-target metadata and success/correctness information;
- estimate gaze sampling intervals from raw timestamps rather than assuming a rate;
- establish how scroll offsets map screen gaze to content coordinates;
- determine whether calibration/accuracy metadata exist.

The public landing record explicitly leaves the scroll-log sampling rate unspecified. No rate should be invented.

### 2. MCFW-Gaze — strong raw-signal validation candidate

Current public record reports:

- continuous raw gaze at 120 Hz from Tobii Pro Fusion;
- normalized display gaze, timestamps, pupil diameter and validity information;
- a 1920 x 1080 display with physical dimensions and approximate viewing distance;
- structured tasks plus 600-second real-web browsing sessions.

Strengths:

- unusually complete raw measurement information;
- known display/viewing geometry is valuable for spatial uncertainty work;
- good candidate for detector and quality sensitivity.

Limitation for the first claim:

- the available task structure is not automatically a clean experimental interface contrast. A focal HCI estimand must be justified rather than invented from convenience.

### 3. UEyes — strong CHI-context external dataset

Confirmed features:

- CHI 2023 dataset;
- 62 participants;
- 1,980 UI screenshots;
- webpage, desktop UI, mobile UI, and poster categories;
- released eye-tracker logs, image stimuli, scanpaths, saliency maps, and metadata.

Strength:

- unmistakably HCI-facing and broad across interface types.

Gate:

- inspect the raw GP3 log schema and determine whether the released logs expose the sample-level information required for detector re-analysis rather than only processed fixation information.

### 4. RecGaze — strong later interactive validation

Confirmed features:

- 87 users;
- 3,477 interactions;
- three movie-selection tasks and 40 carousel interfaces per user;
- fixation, click, cursor, and selection-explanation data.

The current non-public release states that raw gaze and pixel positions are included in the restricted version.

Strength:

- genuine interactive recommender interfaces with behavioral outcomes.

Gate:

- raw-gaze analysis depends on obtaining the appropriate dataset access. Do not substitute the public processed data for raw detector-level claims.


## Verified archive-structure finding

The Zenodo raw-archive preview currently shows raw `extracted_data.xlsx` participant directories for Scenario-based Selection/android, Scenario-based Selection/iOS, and Target Search/android. For Target Search/iOS, the raw preview instead exposes `Target Search.lnk` plus `ios-scroll/` logs; no corresponding raw iOS participant workbook directory is visible in the archive preview. The processed archive nevertheless contains `Target Search_ios_...` CSV files.

This mismatch is a hard feasibility warning. Processed Target Search/iOS files may be used only for downstream analyses supported by their released columns; they must not be treated as evidence that raw sample-level iOS Target Search data are available for detector re-analysis. Detector-level cross-layout claims therefore remain blocked until the original raw iOS workbooks are recovered or the archive is corrected by the dataset authors.

The research-only script `audit_emoji_keyboard_dataset.py` inventories both ZIP archives and writes a cell-by-cell manifest of raw workbooks, scroll logs, processed task files, missing tasks, participant mismatches, and shortcut/link artifacts. It returns a non-zero status when the archive is structurally unsuitable for the full detector-to-claim pilot.

## Dataset selection rule

A dataset can enter the complete detector-to-claim multiverse only if it provides, at minimum:

participant × trial × timestamp × gaze-x × gaze-y

plus enough task/stimulus information to reconstruct the focal outcome and AOIs.

A dataset that begins at processed fixations can still be useful for downstream AOI/model sensitivity, but it cannot be used as evidence about detector robustness.

## Validation strategy after feasibility audit

No currently verified open dataset should be forced to support every stage of the final paper.

### Tier 1 — raw-signal qualification: MCFW-Gaze

MCFW-Gaze is the strongest currently verified open source for detector and quality sensitivity. Version 3 provides continuous 120 Hz Tobii Pro Fusion TSV streams for 15 participants, normalized display gaze, binocular validity/availability fields, pupil and eye-openness measures, known display geometry, structured tasks, and long web-browsing sessions.

It should first be used to verify that the cross-stage runner can:

- ingest a real continuous gaze stream;
- preserve left/right validity and missingness;
- rerun defensible event detectors;
- propagate quality rules without silently dropping trials;
- retain failed/non-converged branches;
- reproduce deterministic universe manifests and hashes.

MCFW-Gaze should not be presented as an interface A/B experiment unless a focal contrast is supported by its study design. The initial use is methodological qualification, not a manufactured HCI effect.

The released `data_loss_per_trial.csv` also shows substantial trial-to-trial and eye-to-eye variation in invalid-sample fractions. That makes it suitable for testing quality-rule propagation, while any quality threshold used in a substantive multiverse must still be justified and frozen before inspecting focal effects.

### Tier 2 — downstream HCI qualification: UEyes

UEyes is strongly HCI-facing: 62 participants viewed 1,980 screenshots spanning webpage, desktop UI, mobile UI, and poster categories. It releases the screenshots, metadata, scanpaths and eyetracker logs.

However, its released processing code treats the logs as Gazepoint fixation exports: it filters valid `BPOG` coordinates and uses `FPOGD` fixation duration. The released file naming also uses `*_fixations.csv`.

Therefore UEyes is suitable for:

- AOI-definition sensitivity;
- feature/outcome sensitivity;
- time-window sensitivity;
- model-level conclusion stability across UI categories;

but not for claims about re-running fixation detectors from the original sample stream.

### Tier 3 — complete detector-to-claim HCI validation

The final paper still needs at least one dataset that combines:

1. continuous raw gaze;
2. reconstructable interface/stimulus geometry;
3. a defensible HCI manipulation or comparison;
4. participant/trial metadata;
5. enough information to reproduce the focal outcome.

The University of Seville infrared/webcam benchmark is especially attractive because its record explicitly describes raw gaze/UI logs, scenario definitions, and low- versus high-density static-form tasks. Its files are currently restricted, so it is an access candidate rather than current evidence.

The non-public RecGaze release is another strong later candidate because it adds raw gaze and pixel positions to genuinely interactive carousel-selection sessions, but it likewise requires authorized research access.

A split Tier-1/Tier-2 demonstration is useful for framework qualification, but it must not be misrepresented as full cross-stage validation. Full detector-to-claim propagation remains a required milestone for the final CHI study.

## First-pilot decision

Use Emoji Keyboard Gaze Search as the first feasibility target, but do not freeze the focal claim until the task-target and geometry metadata are recovered.

If those materials cannot support defensible AOIs and content-coordinate reconstruction, move the first full pipeline pilot to a dataset with complete raw measurement and stimulus geometry rather than weakening the research question.

## Prespecification template

Before evaluating branch-specific effects, freeze:

- dataset version and source hash;
- focal HCI contrast;
- participant/trial inclusion rules;
- one estimand per robustness specification;
- primary detector specification;
- defensible detector alternatives;
- nominal AOIs and perturbation envelope;
- coordinate units and display/viewing geometry where required;
- quality/calibration rules;
- treatment of missing gaze and valid non-events;
- outcome construction;
- fixed focal statistical model;
- focal coefficient term;
- substantive threshold, if scientifically justified;
- planned universe count;
- rules for branch failure and non-convergence.

## Output

Each evaluated universe should retain:

dataset, claim, estimand, universe id, detector specification, AOI specification,
quality specification, non-event specification where applicable, N, estimate,
SE, uncertainty interval, convergence status, failure reason, and provenance
hashes.

The primary paper should report the full planned denominator and should not
reduce robustness to the number of p-values below .05.
