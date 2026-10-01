# Methods draft — CHI gaze-claim robustness study

## 3 Claim-Centered Measurement Robustness

### 3.1 Analysis object and robustness contract

We treat an eye-tracking result as the output of a measurement-to-claim pipeline rather than as the direct consequence of a statistical model. The analysis object is one prespecified substantive claim attached to one estimand. A robustness specification contains: (1) the focal claim and model term; (2) a finite decision grid of measurement alternatives judged defensible before focal-result inspection; (3) one declared primary value for each decision family; and (4) a deterministic manifest of all planned combinations. Every combination receives a stable universe identifier and specification hash.

The framework distinguishes **estimand-preserving measurement alternatives** from **estimand-changing analyses**. Alternatives may coexist in one robustness denominator only when they preserve the scientific quantity being estimated. For example, changing event detector, eye, visual-angle geometry, AOI boundary convention, quality rule, or cohort can alter the measured transition outcome while retaining the Prompt/Non-prompt transition-rate estimand. By contrast, removing the exposure-time offset changes the estimand from a transition-rate ratio to a raw-count ratio and is therefore analyzed separately rather than entering the same denominator.

The decision universe is frozen before branch-specific focal results are inspected. No branch can be added, removed, or reparameterized because it strengthens, weakens, or reverses the focal coefficient. Planned branches that fail, do not converge, or become non-evaluable remain represented in the denominator with an explicit reason. The primary summary is consequently not the percentage of specifications with p < .05. We report coefficient direction, effect magnitude, uncertainty intervals, model N, convergence/evaluability, failure reasons, and descriptive sensitivity associated with each decision family. Specification frequencies describe the declared decision universe and are not interpreted as probabilities that a claim is true.

The implementation is an orchestration layer rather than a replacement detector, AOI engine, or statistical estimator. It records the claim and decision universe, expands that universe deterministically, calls the prespecified measurement/model procedures, and standardizes branch-level accountability. This separation is intended to keep scientific choices explicit rather than embedding them in a generic robustness function.

### 3.2 Primary case: self-regulated-learning gaze data

#### 3.2.1 Dataset and release audit

The primary claim-propagation case uses the open dataset accompanying Juřík et al. (2025), *Experimental Dataset on Eye-tracking Activity During Self-Regulated Learning*. The released study crossed a between-participant Prompt versus Non-prompt manipulation with eight learning materials and recorded gaze with an SMI RED 250. The public release contains sample-level left- and right-eye point-of-regard coordinates, millisecond timestamps, task/stimulus metadata, released stimulus images, event exports, and AOI information.

Before defining the executable measurement universe, we audited the public archive rather than relying only on descriptor-level sample counts. The cross-source identity audit found 83 raw participant identifiers and 84 event-data identifiers, but 82 exact identities shared by the raw gaze files, `participants.csv`, and `stimuli.csv`. These 82 participants were balanced across Prompt and Non-prompt (41/41) and formed the inclusive cohort branch. No identifier repair was inferred for unmatched records.

The timebase audit also found a source-level sampling distinction. Seventy-seven of the 82 exact participants had all eight learning tasks at the nominal approximately 250 Hz rate, whereas five exact participants exhibited approximately 60 Hz task streams. This motivated a prespecified cohort sensitivity between all exact participants (`exact_raw82`) and the 77 exact participants with nominal approximately 250 Hz learning-task streams (`nominal250_77`). Cohort membership was determined from source structure before model effects were inspected.

The source describes the public data as raw eye-tracking data but also reports manual BeGaze gaze-offset correction for correctable systematic distortion. Because the archive does not establish whether the released point-of-regard coordinates precede or follow every such correction, we refer to them as **sample-level raw exports** rather than untouched sensor-native measurements. The present analysis therefore audits uncertainty downstream of the released coordinates; it does not claim to recover uncertainty associated with unreleased manual correction.

#### 3.2.2 Focal contrast and estimand

The focal scientific contrast is Prompt versus Non-prompt on the rate of transitions between AOIs during the eight learning slides. Prompt assignment is the between-participant manipulation used for the primary coefficient. Task identity is retained as a fixed blocking factor to account for stable differences among the eight learning pages.

The outcome is a count of between-AOI transitions per participant × learning slide. Because study time varies across trials and the source study reports differences in time spent on learning materials, the primary estimand is a **transition-rate ratio**, not a raw-count difference. Observed `stimulus_time` enters the model as a log exposure offset. Raw transition count without this offset is an estimand-changing sensitivity and is kept outside the 144-specification primary denominator.

A transition is operationalized consistently across the open detector branches. Fixation episodes are ordered temporally. A between-AOI transition is counted only when two directly adjacent fixation episodes are both AOI-assigned, are in valid temporal order, and have different AOI identifiers. An unassigned fixation breaks the chain; thus AOI A → unassigned → AOI B does not become an A→B transition. If adjacent fixation episodes overlap in time, the corresponding trial branch is marked non-evaluable rather than imposing an arbitrary order. The released proprietary BeGaze transition catalogue remains a historical reference outside the primary denominator because its saccade-defined semantics are not assumed to be algorithmically identical to the open adjacent-fixation-change construct.

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

**Event detection.** We use three open detector branches that can be executed consistently on the released samples: (1) simple I-VT with a 30°/s velocity threshold, 100 ms minimum fixation duration, and explicit 75 ms run-gap rule; (2) simple I-VT with a 40°/s threshold, 50 ms minimum fixation, and the same 75 ms run-gap rule; and (3) I-DT with a 1° dispersion threshold and 100 ms minimum fixation. The I-VT branches do not claim to reproduce complete proprietary vendor filters. The released BeGaze event catalogue remains reference evidence outside the Cartesian denominator. REMoDNaV and the package's adaptive robust-MAD detector were also excluded from the primary denominator: applying REMoDNaV to the irregular released time series would require an additional resampling/regularization decision, while the internal adaptive detector lacked externally validated frozen numerical parameters for this study. No interpolation or smoothing was performed at import.

**Eye representation.** Left and right eyes are analyzed separately. The release contains separate point-of-regard coordinates for both eyes, and the primary universe does not introduce implicit binocular averaging. A binocular fusion rule would require a new measurement definition and missing-eye policy and is therefore outside the primary denominator.

**Visual-angle geometry.** Detector thresholds are expressed in visual-angle units, whereas the released coordinates are in pixels. The source reports a 22-inch, 1600 × 900 display and an approximate viewing distance of 60–70 cm but no participant-specific distance. We assume that physical panel aspect ratio matches the 1600:900 pixel aspect ratio and convert pixel displacement to angular displacement under three prespecified distance branches: 60 cm (reported lower bound), 65 cm (explicit midpoint sensitivity value), and 70 cm (reported upper bound). The original pixel coordinates are retained. The 65 cm branch is not described as a measured participant distance.

**AOI geometry.** The source reports four AOIs corresponding to quartiles of each learning slide and an AOI area of 359,550 pixels. On a 1600 × 900 display, exact mathematical quarters are 800 × 450 = 360,000 pixels, whereas 799 × 450 equals 359,550 pixels. We therefore retain two source-compatible geometries: exact 800 × 450 quarters and a 799 × 450 published-area-compatible geometry with a symmetric two-pixel vertical seam. These alternatives were frozen before focal effects were inspected. Released vendor AOI labels are used only as reference evidence and do not determine which independent geometric branch is retained.

**Quality rule.** The source documents an 80% tracking-ratio exclusion rule. The primary universe contrasts (1) the public released sample with no additional tracking-ratio exclusion and (2) a trial-level sensitivity retaining trials whose `stimuli.csv` tracking ratio is at least 80%. The second branch is explicitly a trial-level sensitivity rather than a claim to reconstruct the source study's historical recording-level exclusion process. Missing samples are never converted to zero.

**Cohort.** The two cohort branches are all 82 exact raw+metadata identities and the 77-participant nominal-250-Hz subset described above.

These branches define measurement alternatives around one fixed estimand. Broader AOI perturbations, binocular fusion, recovery-calibrated quality thresholds, adaptive detector extensions, and raw-count models are intentionally kept outside the minimum primary universe because adding them would either introduce additional unvalidated choices or change the estimand.

#### 3.2.4 Statistical model and execution contract

Every evaluable SRL specification is fitted with the same negative-binomial mixed model using `glmmTMB` 1.1.14 and the NB2 variance parameterization with a log link:

```text
transition_count
  ~ prompt_indicator
  + factor(stimulus_id)
  + offset(log(stimulus_time))
  + (1 | participant_id)
```

The focal coefficient is Prompt minus Non-prompt on the log-rate scale. Exponentiation yields the transition-rate ratio. We require optimizer convergence code 0 and a positive-definite Hessian. There is no fallback estimator if the frozen backend fails; a failed fit would remain a failed planned specification. Each branch records the coefficient estimate, standard error, Wald interval, N, convergence status, specification decisions, and stable specification hash.

The measurement stage and model stage were executed in separate frozen workflows. The canonical measurement run is `36683906120`. The frozen model universe was executed at analysis commit `5fad0f192067ca74277c4b5f2bec462514395124` in workflow run `36708103076`. Exact SHA-256 hashes are retained for the measurement table, universe manifest, model result table, and complete workflow artifact. No primary decision was changed after the focal SRL result was observed.

### 3.3 Independent case: MCFW-Gaze measurement generalization

#### 3.3.1 Validation role and source audit

The second dataset is used for **measurement generalization**, not replication of the SRL Prompt contrast. MCFW-Gaze provides raw binocular gaze recorded with a Tobii Pro Fusion at a nominal 120 Hz across natural-image viewing, gaze-pattern authentication, password entry, shopping, news browsing, and video viewing. The frozen source archive contains 4,695 trial files from 15 participants and 6,304,159 raw samples.

Before detector execution, a one-time archive audit verified the source binary and structural denominator. The source ZIP from Zenodo record `20300972` was verified against its published size and checksum and additionally hashed with SHA-256. All raw files were retained in the audit. One source file contained a non-increasing device timestamp and was declared non-evaluable rather than reordered or repaired.

A separate semantic audit identified values outside [0,1] in published fields labelled as missing-data or data-loss proportions. Those fields were quarantined from primary filtering and quality-association analyses rather than clipped, rescaled, or reinterpreted after detector results were observed. Primary quality measures are derived directly from the sample-level validity and availability flags whose operational meaning is explicit in the raw files.

#### 3.3.2 Timebase and visual-angle geometry

Tobii documentation records device timestamps in microseconds, and the archive-wide median raw timestamp increment is 8,333 units, consistent with 120 Hz under a microsecond scale. We therefore freeze `1e-6` seconds per device timestamp unit. Files with non-finite or non-increasing timestamps remain non-evaluable.

The public MCFW description reports a 1920 × 1080 display with physical dimensions of approximately 310 × 175 mm and an approximately 650 mm viewing distance. We use that single source-reported nominal geometry rather than introducing an unsupported distance range. Normalized display coordinates are converted axis-wise to signed degrees around screen center using the separately implied horizontal and vertical physical pixel densities. Original normalized coordinates are retained for provenance.

#### 3.3.3 Shared detector input and frozen detectors

The validation reuses the same three open detectors used in the SRL universe rather than tuning a detector set to MCFW outcomes. Left and right eyes are analyzed separately. A sample is detector-usable only when the eye-specific gaze point is marked valid and available, the timestamp is finite, and both gaze coordinates are finite.

No interpolation, smoothing, sample creation, or binocular fusion is performed. For detector execution, usable samples are divided into maximal contiguous runs. A new run begins after an unusable source sample or when the observed timestamp gap exceeds 75 ms. This common pre-detector rule prevents algorithms from bridging source-unusable intervals while preserving the untouched source timeline for quality accounting and event-to-sample mapping.

#### 3.3.4 Detector agreement and event-summary divergence

Detector events are mapped back to the original source timestamps. For each participant, trial, eye, and detector pair, sample-level agreement is computed only across source-usable samples for that eye so shared missingness cannot be rewarded as detector agreement. We report fixation/non-fixation agreement proportion, fixation-state Jaccard overlap, and Cohen's kappa where defined. Jaccard is left undefined when neither detector assigns any usable sample to fixation; kappa is undefined when its expected-agreement denominator is zero.

We also compare scientifically interpretable event summaries for every detector: fixation count, fixation rate per minute, median fixation duration, and total fixation-time proportion. Pairwise divergence is represented by absolute difference and the symmetric relative difference

```text
2 × |A - B| / (|A| + |B|)
```

when the denominator is nonzero. The symmetric form does not privilege either detector as a reference or ground truth.

The six source-defined interaction contexts are summarized descriptively. They are not treated as randomized experimental treatments. To assess whether detector disagreement changes with source-signal completeness, we compute Spearman's rho between analyzed-eye usable fraction and disagreement outcomes overall for each detector pair/eye and descriptively within context. No p-value threshold or automatic robust/fragile label is applied.

The complete frozen MCFW validation was executed at commit `878e351a82a48665990955853c55b2315b10fd27` in workflow run `36776432517`. The canonical GitHub Actions artifact is `11127976095` with SHA-256 `ed2af7a036d1fb9a62b5226b71b6f02d007f87a9d23b1970828f9c9742b93020`. The planned denominator comprises 28,170 detector-summary rows and 28,170 detector-pair rows. Post-result execution changes are explicitly recorded as none.

### 3.4 Cross-case reporting and interpretation

The two cases answer different questions and are never pooled into one effect analysis. SRL is the complete **claim-propagation** case: it tests how a fixed substantive coefficient behaves across a prespecified cross-stage measurement universe. MCFW-Gaze is the independent **measurement-generalization** case: it tests whether detector-dependent representation differences persist under different hardware, sampling rate, data quality, and interaction contexts.

Across both cases, failures and non-evaluable branches are part of the evidence. We do not silently drop them from denominators, convert missing outcomes to zero, select a detector based on agreement with another detector, or infer that the most frequent sign across specifications is the probability of a true effect. Decision-family summaries are descriptive sensitivity accounting rather than causal attribution of estimate variation to individual preprocessing choices.

### 3.5 Reproducibility and pre-result/post-result separation

The research record preserves a strict boundary between specification and interpretation. Decision registries, source audits, detector/AOI/quality plans, estimand definitions, model contracts, and planned denominators were committed before the corresponding focal outcomes were inspected. Executions occurred in GitHub Actions against verified public-source archives. Canonical artifacts are identified by workflow run, commit SHA, file checksums, and workflow-artifact SHA-256 hashes. Compact result snapshots are committed to the repository, while large trial-level derived outputs remain represented by their canonical workflow artifacts and hashes.

The paper-facing figure generator is presentation-only. It verifies the exact SRL and MCFW artifact SHA-256 values before reading any result table, refits no model, reruns no detector, and uses deterministic SVG metadata/hashing. This allows figure regeneration without creating a second analytical pathway.

Because both datasets are previously released public research data, the present work introduces no new participant recruitment or data collection. Ethical approval and consent procedures are those of the original source studies; the present contribution is a secondary methodological analysis of their released data.
