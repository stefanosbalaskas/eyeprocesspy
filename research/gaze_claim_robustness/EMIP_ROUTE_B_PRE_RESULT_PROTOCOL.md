# Route B / EMIP — prospective, source-gated claim contract (v0.1)

**Status: PRE-RESULT CANDIDATE PROTOCOL; NOT YET FROZEN; NO MODEL HAS BEEN FIT.**

This record is committed before inspecting any EMIP focal gaze-derived result. The
final executable registry and branch denominator require a successful raw-source
audit, AOI geometry check and participant/trial linkage check. A failed gate
means EMIP is **not** incorporated as a completed third study.

## Source evidence as of 13 November 2026

- Bednarik et al. (2020), *Science of Computer Programming*, 198, 102520,
  doi:10.1016/j.scico.2020.102520, describes 216 publicly analysed participants,
  two code-comprehension tasks (Rectangle and Vehicle), 250 Hz SMI RED250,
  1920 x 1080 display, and metadata with
  `correct_rectangle`, `correct_vehicle`, `stimulus_rectangle`,
  `stimulus_vehicle`, `order_rectangle`, `order_vehicle`,
  `experiment_language`, and expertise.
- Original paper describes *same order* for all, but current dataset website
  describes alternating order. Audit **actual metadata** before assuming order
  balance or applying an order adjustment.
- Website describes 269 recruited, 15 excluded, and 15% of data not published.
  The paper reports 216 released cases. Neither denominator is substituted
  for a verified count of actual accessible files.
- Source paper specifies CC-BY-NC-SA and ~560 MB ZIP; website summarizes
  "CC 4.0" more generally. Retain source attribution and do not redistribute
  participant raw streams in a public repository.
- The original article reports screen resolution, *not a verified physical
  display size or participant-specific viewing distance*. Do not invent
  visual-degree conversions.

Primary sources: https://www.emipws.org/dataset/ and
https://doi.org/10.1016/j.scico.2020.102520
Corrected-toolkit methods: https://doi.org/10.1145/3448018.3457425

## Focal research question — declared before inspecting focal gaze results

To what extent does a **gaze-derived rate of navigation among code-line AOIs**
differ between trials answered correctly and trials answered incorrectly,
after accounting for stimulus/task and repeated observations within persons,
and how does that contrast depend on defensible raw-gaze processing choices?

This is a *descriptive observational association* between recorded behavior
and subsequent task-response correctness, **not** the causal effect of correct
comprehension, expertise or a randomized interface intervention.

## Outcome contract

A trial corresponds to the actual participant × code stimulus for Rectangle
or Vehicle (excluding instruction/question/response display). A code-line
transition is an ordered change from one assigned line AOI to a different
line AOI between adjacent nonoverlapping fixation episodes within the trial.
An unassigned episode interrupts the sequence. An event overlapping the
question screen is not a code-trial event. No interpolation, timestamp repair,
implicit binocular averaging or AOI-offset correction is allowed without
an audit-supported, separately justified branch.

Trial exposure is the valid *stimulus-viewing time in seconds*, derived from
documented stimulus-onset and offset markers; its log is an offset. A trial with
non-positive or unidentifiable exposure is non-evaluable, **not zero**.

## Fixed focal model, subject to design identifiability gates

`Y_it ~ NB2(mu_it, theta)`
`log(mu_it) = alpha + beta_correct * correct_it + gamma_task[it] +
               delta_language[i] + eta_order[it] + log(T_it) + u_i`

- Focal coefficient: `beta_correct` on an exponentiated transition-rate-ratio
  scale; the tested null is RR=1.
- Person random intercept `u_i`; task and language as fixed effects.
- Include order fixed effect **if it varies and is full-rank**, determined from
  actual metadata *before inspecting fixation outcomes or coefficients*.
  A constant order term is structurally dropped, with reason logged.
- If the language factor has a single category, it is structurally dropped,
  with reason logged before outcome inspection.
- NB2/log link, Wald 95% model-conditional confidence interval, no post hoc
  model fallback selected based on effect sign/significance.
- Model-convergence failures stay in the denominator.
- Data with multiple versions of a participant×stimulus must be reconciled by
  a deterministic, source-supported rule before model fitting.

## Candidate decision families (not yet assigned a denominator)

1. **Fixation definition:** baseline pixel-coordinate I-DT (25 px dispersion;
   50 ms minimum) documented in the original EMTK parser, and an explicitly
   marked alternative dispersion threshold (provisional 35 px/50 ms, requiring
   methodological justification). Do not assert equal visual-degree thresholds.
2. **Eye selection:** left and right separately, if both usable coordinates and
   eye-specific validity fields are present.
3. **AOI mask representation:** source line-level rectangles; alternative
   source token rectangles aggregated by line **only if** both coordinate
   representations and alignment semantics are verified. Do not fabricate an
   AOI perturbation.
4. **Quality:** source-valid trials; source-valid with >=80% usable samples
   (sample-definition sensitivity, not strict estimand preservation).
5. **Cohort:** no alternative cohort specified at this stage. Additional
   subgroups cannot be added after looking at the focal estimate.
6. **Viewing geometry:** excluded unless independently documented physical
   display dimensions and applicable viewing distance make conversion
   justified. No invented size, distance, or visual degrees.

Once schema and AOI gates pass, record each branch's exact parameterization,
source rationale, stable ID and hash, and resulting finite denominator in a
separate frozen registry **before model fitting**. Branch deletions after
outcome inspection are prohibited.

## Non-result feasibility gates

- Accessible downloadable EMIP archive and SHA-256 of acquired version.
- Actual counts of publicly released distinct participant raw TSVs, metadata
  rows and matched IDs; no extrapolation from website denominators.
- Actual raw header schema, timestamp units/monotonicity and message sequence.
- Exact code-stimulus onset/offset parse for Rectangle/Vehicle and absence of
  question-screen contamination; verify order flags.
- Source AOI rectangles and coordinate system align with rendered stimuli.
- At least two interpretable code-line AOIs per trial in both code programs.
- Verified correctness and language metadata linked to participants, without
  inspecting association with gaze-derived outcomes during design freeze.
- Model identifiability from metadata alone; minimum usable cases and cells
  predeclared to avoid underpowered, degenerate or non-estimable contrast.
- No raw participant data or unlicensed stimulus copies added to public Git.

If any essential gate fails, **record a blocked/infeasible EMIP candidate** and
retain the two-study THMS manuscript unchanged. Alternative public datasets
may be screened, but no data-driven substitution is permitted.

## Reporting plan

Present independent empirical sources separately: SRL = original frozen
inferential case; MCFW-Gaze = representation-level validation; EMIP =
additional end-to-end claim case *only if completed and validated*. Label
new EMIP analysis prospective from this source-audit date, not originally
preregistered with SRL and not a causal experiment.

Results are acceptable even if all CIs include the null or signs do not vary.
Branch-frequency percentages are descriptions of the declared decision
universe, **not** probabilities of true effects.
