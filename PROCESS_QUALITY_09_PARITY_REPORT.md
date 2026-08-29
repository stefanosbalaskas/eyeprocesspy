# Process reliability and calibration-quality parity report

Checkpoint: `eyeprocesspy 0.1.0.dev0` against frozen R `eyeprocess 0.11.1`.

## Scope

This tranche ports **30 frozen exported functions** from:

- `R/074-process-registry-reliability-0-9.R` — 15 process-measure registry and repeatability/reliability exports;
- `R/076-calibration-quality-uncertainty-0-9.R` — 15 calibration uncertainty, sampling and gaze-quality exports.

It also ports **six S3 plot counterparts** from `R/080-plots-0-9.R`:

- `plot_eye_process_reliability_profile()`;
- `plot_eye_calibration_error_model()`;
- `plot_eye_calibration_drift_profile()`;
- `plot_eye_data_quality_profile()`;
- `plot_eye_probabilistic_aoi_assignment()`;
- `plot_eye_sampling_irregularity_audit()`.

## Implemented measurement contracts

The process-measure registry preserves channel, unit, aggregation level, neutral interpretation, guardrail and lifecycle/status metadata. Registry validation rejects incomplete metadata and vector-valued fields where the frozen R API requires a scalar. The built-in registry keeps observed/process measures separate from unsupported psychological inference.

Reliability support includes odd/even and repeated-random split-half summaries, absolute-agreement ICC(A,1), Bland–Altman summaries, pairwise temporal stability and participant-resampling bootstrap uncertainty. Reliability remains explicitly separate from construct validity.

Calibration/quality support includes empirical target error, RMS successive-sample precision, effective sampling frequency, interval irregularity, bivariate empirical calibration-error covariance, uncertainty ellipses, Monte Carlo calibration-error propagation, probabilistic rectangular-AOI membership, hard/probabilistic comparison, deterministic calibration sensitivity grids, fixation-boundary distance, calibration drift and gaze data-quality reporting.

## Missing-value semantic correction

Python comparisons against `NaN` ordinarily produce `False`. The frozen R `aoi_membership_probability()` instead preserves `NA` when all uncertainty draws for a sample have missing gaze coordinates. The Python implementation explicitly uses nullable booleans so an all-missing sample produces an undefined AOI-membership probability rather than a false zero probability.

Likewise, a missing validity flag in `gaze_data_quality_profile()` is treated as invalid for that sample while still yielding a finite valid-fraction denominator, matching the frozen R contract.

## Scientific boundary

Calibration-error and probabilistic-AOI outputs characterize **spatial measurement uncertainty** under an empirical error model. They are not probabilities of attention, engagement, comprehension, intent, cognitive load or other psychological constructs. Reliability describes repeatability under a design/population and does not establish validity or diagnostic meaning.

## Documentation and examples

Python article counterparts:

- `docs/articles/process-measure-registry-and-reliability.md`;
- `docs/articles/calibration-uncertainty-and-data-quality.md`.

Executable examples:

- `examples/irt_process_reliability_09.py`;
- `examples/irt_calibration_quality_09.py`.

## Validation

- Focused process-quality tests: **9 passed**.
- Full package suite after this tranche: **167 passed**.
- Executable `irt_*.py` examples after this tranche: **32 passed** through the global example-smoke suite.
- Installed offline validation-wheel smoke: PASS.
- Canonical Stan resources in installed wheel: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `3689803ed8460c1256bcd4048994e5014556828a423b94fd953a19a6d543dc89`.

## Parity accounting after tranche

- Frozen exports implemented: **569 / 1,182**.
- P5 source-ported algorithms/contracts: **500**.
- P5 Python-reference/backend-different functions: **69**.
- P4 cross-language numerical parity: unchanged; extended R oracle remains pending.
- Verified plot-ledger entries: **61 / 341**.
- Complete article counterparts: **45 / 88**.
