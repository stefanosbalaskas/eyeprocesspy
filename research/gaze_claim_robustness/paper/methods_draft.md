# Methods draft — CHI gaze-claim robustness study

## 3 Claim-Centered Measurement Robustness

### 3.1 Analysis object and robustness contract

We treat an eye-tracking result as the output of a measurement-to-claim pipeline rather than the direct consequence of a statistical model. The analysis object is one prespecified substantive claim under a **claim contract** fixing the focal contrast, outcome semantics, and interpreted model term, while recording any population or sample-support dimensions allowed to vary. A specification combines that contract with a finite pre-result decision grid and receives a stable universe identifier and specification hash.

The framework distinguishes three branch types. **Measurement-definition alternatives** change how the recorded signal becomes the analyzed construct while preserving the focal contrast and outcome meaning; in SRL these include detector, eye, visual-angle geometry, and AOI convention. **Sample-definition sensitivities** change which observations or participants support the estimate; here they are the quality and cohort branches. Because these can alter empirical support and the formal target population, we do not call them strictly estimand-preserving merely because the coefficient formula is unchanged. **Estimand-changing analyses** alter the scientific quantity itself and remain outside the primary denominator; removing the exposure offset, for example, changes a transition-rate ratio into a raw-count ratio.

A branch enters the declared universe only when its rationale is auditable before focal-result inspection: it must be grounded in source metadata, a documented ambiguity, or methodological precedent; operationally specified in advance; executable without outcome-dependent tuning, invented values, or silent source-data repair; and comparable under the same claim contract. Choices that fail these conditions remain reference-only, provenance-gated, optional-extension, or separate-estimand items.

The universe is frozen before branch-specific focal results are inspected. Branches cannot be added, removed, or reparameterized because they strengthen or weaken the focal result, and failed or non-evaluable branches remain in the denominator with reasons. We report direction, magnitude, uncertainty, model N, sample support, convergence/evaluability, failure reasons, and descriptive decision-family sensitivity rather than p<.05 branch counts. Specification frequencies describe the declared grid; they are neither effect probabilities nor invariant robustness scores. The primary object is therefore a **claim-stability universe**, not an assertion that every branch shares an identical formal population estimand.

The implementation is an orchestration layer, not a replacement detector, AOI engine, or estimator. It records and expands the frozen universe, calls prespecified measurement/model procedures, and standardizes branch-level accountability.

### 3.2 Primary case: self-regulated-learning gaze data

#### 3.2.1 Dataset and release audit

The primary claim-propagation case uses the open dataset accompanying Juřík et al. (2025), *Experimental Dataset on Eye-tracking Activity During Self-Regulated Learning*. The study crossed a between-participant Prompt versus Non-prompt manipulation with eight learning materials and recorded gaze with an SMI RED 250. The release contains sample-level left/right point-of-regard coordinates, millisecond timestamps, task/stimulus metadata, stimulus images, event exports, and AOI information.

We audited the public archive before defining the executable universe. The cross-source identity audit found 83 raw participant identifiers and 84 event-data identifiers, but 82 exact identities shared by raw gaze, `participants.csv`, and `stimuli.csv`. These 82 participants were balanced across Prompt and Non-prompt (41/41) and formed the inclusive cohort branch. No identifier repair was inferred.

Seventy-seven of the 82 exact participants had all eight learning tasks at the nominal approximately 250 Hz rate, whereas five exhibited approximately 60 Hz task streams. This motivated a prespecified cohort sensitivity between all exact participants (`exact_raw82`) and the 77 nominal-250-Hz participants (`nominal250_77`). Cohort membership was determined from source structure before model effects were inspected.

The source describes the public data as raw eye-tracking data but also reports manual BeGaze gaze-offset correction for correctable systematic distortion. Because the archive does not establish whether released coordinates precede or follow every such correction, we call them **sample-level raw exports** rather than untouched sensor-native measurements. Our analysis therefore audits uncertainty downstream of the released coordinates.

#### 3.2.2 Focal contrast and estimand

The focal scientific contrast is Prompt versus Non-prompt on the rate of transitions between AOIs during the eight learning slides. Prompt assignment is the between-participant manipulation; Task identity is a fixed blocking factor for stable page/content differences.

The outcome is between-AOI transition count per participant × learning slide. Because viewing time varies, the focal quantity within a fixed sample-definition branch is a **transition-rate ratio**, with observed `stimulus_time` entering as a log exposure offset. Raw transition count without this offset is an estimand-changing sensitivity outside the 144-specification primary denominator.

Quality and cohort branches can change the observations supporting the coefficient, so we do not claim strict target-population identity across them. Fixed throughout the primary universe are the Prompt/Non-prompt contrast, transition-rate semantics, Task adjustment, exposure definition, model family, and interpreted coefficient. Sample support and model N remain part of the robustness evidence.

Fixation episodes are ordered temporally. A transition is counted only when two directly adjacent fixations are both AOI-assigned, temporally ordered, and have different AOI identifiers. An unassigned fixation breaks the chain; AOI A → unassigned → AOI B therefore does not become an A→B transition. If adjacent fixations overlap, that trial branch is non-evaluable rather than arbitrarily ordered. The proprietary BeGaze transition catalogue remains a historical reference outside the primary denominator because its saccade-defined semantics are not assumed equivalent to the open adjacent-fixation-change construct.

#### 3.2.3 Frozen SRL measurement universe

The primary universe contains 144 planned specifications:

```text
3 event detectors
× 2 eyes
× 3 viewing-distance assumptions
× 2 AOI geometries
× 2 quality rules
× 2 cohort definitions
= 144 specifications
```

**Event detection.** Three open branches are executed consistently on the released samples: (1) simple I-VT at 30°/s with 100 ms minimum fixation and a 75 ms run-gap rule; (2) simple I-VT at 40°/s with 50 ms minimum fixation and the same gap rule; and (3) I-DT at 1° dispersion with 100 ms minimum fixation. Their rationale was frozen before focal-result inspection in a source-backed detector plan: the 30°/s + 100 ms branch is a conventional fixed-threshold comparator, the 40°/s + 50 ms branch provides an SMI-context comparator, and the 1° + 100 ms branch provides a conceptually distinct dispersion-threshold comparator. None is claimed to reproduce a proprietary vendor filter. Released BeGaze events remain reference evidence. REMoDNaV was excluded because these irregular time series would require an additional resampling/regularization decision; the package's adaptive robust-MAD detector lacked externally validated frozen parameters for this study. No interpolation or smoothing was performed at import.

**Eye representation.** Left and right eyes are analyzed separately. No implicit binocular averaging is introduced; binocular fusion would require an additional measurement and missing-eye policy.

**Visual-angle geometry.** Detector thresholds are in visual-angle units whereas released coordinates are pixels. The source reports a 22-inch, 1600 × 900 display and approximate 60–70 cm viewing distance but no participant-specific distance. Assuming physical aspect ratio matches pixel aspect ratio, we convert displacement under 60 cm (reported lower bound), 65 cm (explicit midpoint sensitivity), and 70 cm (reported upper bound). Original pixels are retained; 65 cm is not presented as a measured participant distance.

**AOI geometry.** The source reports four slide quartiles and an AOI area of 359,550 pixels. Exact 1600 × 900 quarters are 800 × 450 = 360,000 pixels, whereas 799 × 450 equals the reported area. We therefore freeze exact quarters and a 799 × 450 published-area-compatible geometry with a symmetric two-pixel vertical seam. Released vendor labels are reference evidence only.

**Quality rule.** The source documents an 80% tracking-ratio exclusion rule. We contrast the public released sample with a trial-level sensitivity retaining `stimuli.csv` tracking ratio of at least 80%. The latter is not claimed to reconstruct the historical recording-level exclusion process. Missing samples are never converted to zero. This is a sample-definition sensitivity because it changes trial support.

**Cohort.** We contrast the 82 exact raw+metadata identities with the 77-participant nominal-250-Hz subset. This is likewise a sample-definition sensitivity.

Together these branches define a prespecified **claim-stability universe** around a fixed focal contrast, rate definition, and model contract. Detector, eye, viewing geometry, and AOI are measurement-definition alternatives; quality and cohort are sample-definition sensitivities. Broader AOI perturbations, binocular fusion, recovery-calibrated quality thresholds, adaptive detector extensions, and raw-count models remain outside this primary universe.

#### 3.2.4 Statistical model and execution contract

Every evaluable SRL specification uses `glmmTMB` 1.1.14 with an NB2 negative-binomial distribution and log link:

```text
transition_count
  ~ prompt_indicator
  + factor(stimulus_id)
  + offset(log(stimulus_time))
  + (1 | participant_id)
```

The focal coefficient is Prompt minus Non-prompt on the log-rate scale; exponentiation yields the rate ratio. We require optimizer convergence code 0 and a positive-definite Hessian. There is no fallback estimator. Each branch records estimate, standard error, Wald interval, N, convergence/evaluability, specification decisions, and stable hash.

Measurement and modeling were executed in separate frozen workflows (runs `36683906120` and `36708103076`; model analysis commit `5fad0f19`). Full artifact identifiers and checksums are retained in the reproducibility package. No primary decision changed after the focal SRL result was observed.

### 3.3 Independent case: MCFW-Gaze measurement generalization

#### 3.3.1 Validation role and source audit

The second dataset is used for **measurement generalization**, not replication of the SRL Prompt contrast. MCFW-Gaze provides binocular gaze recorded with a Tobii Pro Fusion at nominal 120 Hz across natural-image viewing, gaze-pattern authentication, password entry, shopping, news browsing, and video viewing. The frozen archive contains 4,695 trial files from 15 participants and 6,304,159 raw samples.

A pre-detector audit verified the source binary and structural denominator. The Zenodo archive was checked against its published size/checksum and additionally hashed with SHA-256. All raw files were retained. One source file contained a non-increasing device timestamp and was declared non-evaluable rather than reordered or repaired.

Published fields labelled as missing-data or data-loss proportions contained values outside [0,1]. We quarantined those fields from primary filtering and quality-association analyses rather than clipping or reinterpreting them. Primary quality measures use the raw sample-level validity and availability flags.

#### 3.3.2 Timebase and visual-angle geometry

Tobii documentation records device timestamps in microseconds, and the archive-wide median timestamp increment is 8,333 units, consistent with 120 Hz under that scale. We therefore freeze `1e-6` seconds per timestamp unit; non-finite or non-increasing timestamps remain non-evaluable.

The public description reports a 1920 × 1080 display, approximately 310 × 175 mm, at approximately 650 mm viewing distance. We use that single source-reported nominal geometry rather than inventing a distance range. Normalized coordinates are converted axis-wise to signed degrees around screen center; original normalized coordinates are retained.

#### 3.3.3 Shared detector input and frozen detectors

The validation reuses the same three open detectors as SRL without outcome-based tuning. Eyes are analyzed separately. A sample is detector-usable only when the eye-specific point is valid and available and timestamp and coordinates are finite.

No interpolation, smoothing, sample creation, or binocular fusion is performed. Usable samples are divided into maximal contiguous runs, split after an unusable source sample or timestamp gap >75 ms. This prevents algorithms from bridging source-unusable intervals while preserving the untouched source timeline for quality accounting and event mapping.

#### 3.3.4 Detector agreement and event-summary divergence

Detector events are mapped back to source timestamps. For each participant, trial, eye, and detector pair, sample-level agreement is computed only on source-usable samples so shared missingness is not rewarded. We report fixation/non-fixation agreement, fixation-state Jaccard, and Cohen's kappa where defined. Jaccard is undefined when neither detector assigns any usable fixation sample; kappa is undefined when its expected-agreement denominator is zero.

For each detector we also summarize fixation count, fixation rate/minute, median fixation duration, and total fixation-time proportion. Pairwise divergence is reported as absolute difference and symmetric relative difference

```text
2 × |A - B| / (|A| + |B|)
```

when the denominator is nonzero, avoiding a ground-truth reference detector.

The six source-defined contexts are descriptive, not randomized treatments. We compute Spearman's rho between eye-specific usable fraction and disagreement outcomes overall for each detector pair/eye and descriptively within context. No p-value threshold or automatic robustness label is applied.

The frozen MCFW validation was executed at commit `878e351a` in workflow run `36776432517`. Full artifact identifiers and checksums are retained in the reproducibility package. The planned denominator is 28,170 detector-summary rows and 28,170 detector-pair rows; no execution/protocol decision changed after the detector result was observed.

### 3.4 Cross-case reporting and interpretation

The cases answer different questions and are never pooled. SRL tests claim propagation across a cross-stage claim-stability universe; MCFW-Gaze tests whether detector-dependent representation differences persist under different hardware, sampling rate, data quality, and interaction contexts.

Across both cases, failures and non-evaluable branches remain evidence. We do not silently drop them, convert missing outcomes to zero, select a detector based on agreement with another detector, or interpret the most frequent sign as an effect probability. Decision-family summaries are descriptive sensitivity accounting, not causal attribution to preprocessing choices.

### 3.5 Reproducibility and pre-result/post-result separation

Decision registries, source audits, detector/AOI/quality plans, claim definitions, model contracts, and planned denominators were committed before the corresponding focal outcomes were inspected. Executions used verified public archives in GitHub Actions. Canonical outputs are preserved by workflow run, commit, and checksums; compact snapshots are committed while large derived outputs remain represented by their canonical workflow artifacts.

The paper figure generator is presentation-only: it verifies frozen input hashes, refits no model, reruns no detector, and deterministically regenerates the publication figures. Thus figure production does not create a second analytical pathway.

Both datasets are previously released public research data; this work introduces no new participant recruitment or data collection. Ethical approval and consent procedures are those of the source studies.
