# EMIP Route B — frozen public-mirror subset analysis protocol v1.0

**Frozen before computing gaze-derived outcomes or inspecting a model coefficient (13 November 2026).**
This is a NEW prospective, post-SRL validation demonstration, not an original EMIP
analysis, not a preregistered clinical experiment, and not the full EMIP release.

## Source and feasibility audit

- Primary source: Bednarik et al. (2020), EMIP: The Eye Movements in Programming Dataset, *Science of Computer Programming*, 198, 102520, DOI 10.1016/j.scico.2020.102520.
- Participant-level raw mirror: https://github.com/egeiran/Data-Preprocessing-EMIP/tree/main/emip_dataset/rawdata
- Original EMIP metadata and original source AOI coordinates: https://github.com/hiqb269/EMIP-Toolkit/tree/main/emtk/datasets/EMIP/EMIP-Toolkit-%20replication%20package/emip_dataset
- The mirror exposes **32 TSV participant recordings** and one Excel recording (ID 211).
  Only **31 Java-language TSV participants** form the *a priori* accessible,
  code-comparable cohort; one Scala participant (ID 7) and the separate-format
  Excel ID 211 are excluded before any result analysis. The 31 IDs are:
  `1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,20,21,22,23,24,25,26,28,30,31,33,34,37,39,41`.
- The original 216-participant full EMIP corpus **has not** been accessed.
  The 31-case mirror is a nonrandom convenience subset selected *only* on data
  access and language compatibility. No inferential generalization to all EMIP
  programmers will be claimed.
- Metadata-only cohort audit: source correctness field has both 0 and 1 for both
  Rectangle and Vehicle; trial order varies in the mirror. This was inspected
  before gaze outcomes. The source dataset's original research questions and
  ethics remain those of the source publication.
- Every inspected TSV source header reports nominal 250 Hz, screen calibration
  1920x1080 px, 344x194-mm physical display, 700-mm nominal head distance.
  Actual per-person head position was NOT measured here; header values are
  not substituted for individual fixation ground truth.
- The exact stimulus labels and code-message onset/offset markers are
  reconstructible from MSG entries for 32 TSV records. In particular code
  stimulus end is the next `mupliple_choice_{task}.jpg` MSG, excluding the
  response screen. `Trial` remains 1 and **must not** define task boundaries.
- Four Java image AOI CSVs exist for Rectangle/Vehicle and default/"2" screen:
  `rectangle_java.csv`, `rectangle_java2.csv`,
  `vehicle_java.csv`, `vehicle_java2.csv`.
  Each contains `level=line` and `level=element` pixel-coordinate boxes.

## Fixed research question and estimand

Among the accessible 31 Java-language participants, is task-comprehension
correctness (0 vs 1, as recorded in EMIP source metadata) associated with
the **observed rate of transitions among code-line areas of interest** during
stimulus inspection, after adjustment for task identity and within-person
presentation order?

Association only. Correctness is an observed response, **not randomized**.
This is **not** a causal effect of understanding code or of expertise.
The same source code stimulus may have multiple variants, and those differences
are not claimed to be experimentally isolated here.

## Outcome

For one participant × task × decision branch:
- Identify code display interval from exact source MSG timestamps.
- Read 250-Hz gaze samples within `[onset, offset)` only.
- Per eye, consider a gaze sample valid when eye-specific validity flag is
  exactly `1`, both point-of-regard pixel coordinates are finite and
  `0 < x <= 1920`, `0 < y <= 1080`.
- Never interpolate or reorder; gaps >100 ms terminate a run. Non-increasing
  timestamps are handled by splitting runs and counted as an audit reason.
- Detect fixation episodes using the declared detector, each within a valid
  contiguous run. A fixation centroid is assigned at most one code-line AOI.
- Count an AOI transition if **two temporally adjacent fixation episodes**
  have distinct assigned code-line AOIs. Any unassigned episode interrupts the
  transition chain; transitions are not bridged across invalid runs or task
  screens. Zero observed transitions remains a valid zero count.
- Offset exposure = `(source code stimulus offset − onset) / 1,000,000`
  seconds. Nonpositive exposure is non-evaluable, not zero.
- Trial support records number of samples, valid fraction, detected fixations,
  assigned fixations, and code exposure, alongside status/failure reason.

## Frozen processing decisions — 3 × 2 × 2 × 2 = 24 branches

**Event detector (3):**

1. `ivt30_100`: I-VT 30 deg/s instantaneous two-sample velocity and 100-ms
   minimum fixation; pixel displacement converted using 344×194 mm at nominal
   700-mm distance and 1920×1080 px.
2. `ivt40_50`: I-VT 40 deg/s and 50-ms minimum fixation, same geometry.
3. `idt25_50`: classic I-DT with dispersion defined by
   `(max(x)−min(x)) + (max(y)−min(y)) <= 25 px`, 50-ms minimum; the threshold
   and 50-ms duration are documented by the original EMIP Toolkit. Uses
   contiguous, valid samples only.

These are **algorithm and parameter comparisons**, not instrument accuracy
certifications.

**Eye (2):** independently left vs right. No undocumented binocular averaging.

**AOI representation (2):**
1. `line_bbox`: published source `level=line` rectangles.
2. `element_union`: published `level=element` rectangles; map each
   element to a unique original line by maximal nonzero vertical overlap,
   and map a fixation hit to that line. If an element maps ambiguously,
   reject that AOI source rather than assigning a line opportunistically.
   Overlap of multiple line hits also becomes unassigned.
No manual AOI expansion or gaze-derived offset correction.

**Quality rule (2):**
1. `source_valid`: each code interval with positive exposure and at least
   one usable valid sample; a trial with no detected fixation yields Y=0.
2. `valid80`: same plus eye-specific valid sample fraction >=0.80 for the
   code interval. Mark excluded trials explicitly as sample-support
   sensitivity; do **not** claim strict estimand preservation.

There are **24 planned specifications**; each is a fixed participant × task
ledger plus one model fit. Every branch remains in the denominator even on
error or non-evaluable status. Participants/trials may differ under `valid80`.

## Fixed statistical claim contract

For participant i and code task t (Rectangle or Vehicle):

`Y_it | X, T ~ Poisson(mu_it)`,
`log(mu_it) = alpha + beta_correct * correct_it +
               gamma_vehicle * I(task=Vehicle) +
               delta_second * I(presentation_order=2) + log(T_it)`.

- Primary focal estimand: `exp(beta_correct)`, a conditional *associational*
  transition-rate ratio adjusting for task identity and within-person order.
- Inference: Poisson log-link generalized linear model with participant-
  clustered sandwich (HC0 meat) covariance and finite-sample cluster factor
  `G/(G-1) * (N-1)/(N-p)` where G is participant clusters, N trial rows,
  p regressors; two-sided Wald 95% interval using normal critical value 1.96.
  This is a fixed working mean model with cluster-robust uncertainty,
  not a Poisson variance assumption or an NB2 random-intercept model.
- No effect-dependent estimator fallback. Matrix rank deficiency, insufficient
  participant clusters (<10), singular Hessian, insufficient outcome variation,
  or failed fitting marks a model **non-evaluable** with explicit reason.
- Prespecified fixed effects (including order) stay in all fits; do not change
  terms after observing coefficients. No significance-driven model selection.
- Estimate magnitudes and CI interpretation are *conditional on each branch*;
  specification percentages are not posterior effect probabilities.
- A fixed code path, deterministic output format, and independent validation
  checks are required before presenting any EMIP result in the THMS manuscript.

## Execution and disclosure

Implement and run in an isolated GitHub branch without adding raw participant
files to the repository. Store only compact participant-free result summaries,
branch-level coefficients, source-matching counts, source version/provenance and
reproducible code. Display the **31-case nonrandom-mirror** limitation
prominently if the third demonstration is eventually included.

**Do not alter the earlier 144-specification SRL or Tobii MCFW research freeze.**
