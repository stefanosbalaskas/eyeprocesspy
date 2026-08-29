# Restored measurement/process parity report

Checkpoint reconstructed from the frozen `eyeprocess` 0.11.1 source after runtime compaction.

## Scope

This tranche restores the frozen exports from:

- `R/033-process-uncertainty.R`
- `R/034-calibration-recalibration.R`
- `R/035-process-reliability.R`
- `R/037-pupil-registration.R`
- `R/038-informative-missingness.R`
- `R/039-recurrence-analysis.R`
- `R/040-fixation-point-process.R`
- `R/041-representative-scanpaths.R`
- `R/042-process-episodes.R`
- `R/046-evidence-provenance-graph.R`
- `R/047-measurement-intelligence-adapters.R`

The restored surface contains **91 frozen exports** and **77 plot-ledger counterparts**.

## Validation

- 91/91 frozen exports resolve to public Python callables.
- The eleven frozen R test files are represented by reconstruction contract tests.
- Full Python regression is green in split batches: **178/178 collected tests pass**.
- Executable `irt_*.py` example suite: **38/38 pass**.
- Installed-wheel smoke: PASS.
- Wheel contains **13/13** canonical Stan programs.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `e9fe7b7834bc61f38a75d87b4cb3386514d5bcdb311db27a9908cb76117eda19`.

## Algorithmic boundary

Most functions are direct dependency-light translations of the frozen R algorithms. Four functions are conservatively marked `python_reference_differs` because the reconstructed Python implementation does not claim exact estimator identity with the frozen R path:

- `fit_offline_recalibration()` for robust affine/polynomial fitting;
- `fit_fixation_point_process()`;
- `predict_fixation_intensity()`;
- `diagnose_gaze_point_process()`.

No new P4 cross-language numerical parity is claimed without an R oracle.
