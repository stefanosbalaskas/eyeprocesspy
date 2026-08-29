# Functional pupil-IRT parity checkpoint

Frozen R reference: `eyeprocess 0.11.1`.

## Scope completed

- Final overriding public contracts from `R/026-functional-pupil-engine.R`: 10 exports.
- Advanced validation exports retained in `R/022-advanced-models-v2.R`: 2 exports.
- Total frozen exports in this tranche: **12/12**.
- S3 plot counterparts: `plot.eye_functional_pupil_irt`, `plot.eye_functional_pupil_diagnostics`, and `plot.eye_functional_pupil_sensitivity`.
- Python article counterparts: `functional-pupil-irt-engine`, `advanced-model-validation`, and `research-validation-program`.
- Executable examples: `irt_functional_pupil.py` and `irt_advanced_validation.py`.

## Source-manifest correction

`functional_pupil_irt_spec()` and `fit_joint_functional_pupil_irt()` are defined first in `R/022-advanced-models-v2.R` and then **redefined later** in `R/026-functional-pupil-engine.R`. R's source loading order means the 0.26 definitions are the operative public contracts. The frozen signature/API manifests were corrected to the later definitions and Rd usage, including the 34-argument final specification and the final `seed` argument for `fit_joint_functional_pupil_irt()`.

## Implemented scientific contracts

The Python layer now provides explicit event/trial alignment, physiological latency shifting, baseline-window uncertainty fields, quality filtering, optional nuisance residualization, natural/B-spline basis construction, trial-level functional coefficients, two-stage GLM fitting, the canonical CmdStanPy entry point for the bundled `functional_pupil_irt.stan`, parameter extraction, diagnostics, preprocessing sensitivity, scalar-vs-functional comparison, the declared advanced-validation design, and advanced process-data simulation.

Pupil trajectories remain physiological observations. The implementation never labels them automatically as cognitive load, effort, arousal, surprise or another named latent construct.

## Algorithmic boundary

Nine functions are recorded as source-ported algorithm/contracts. Three are conservatively marked `python_reference_differs`: the functional basis uses Patsy rather than R `splines`, the two-stage fit uses statsmodels for the GLM path, and functional-vs-scalar model comparison uses statsmodels/grouped Python folds. Exact R `lme4` and `brms` routes remain explicit backend gates. The canonical Stan model uses the frozen packaged Stan source through CmdStanPy.

No P4 numerical-parity credit is granted without the R oracle.

## Validation

- Functional-pupil focused tests: **7/7 passed**.
- Full package suite: **147/147 passed**.
- Installed validation-wheel smoke: PASS.
- Canonical Stan resources in wheel: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `ce76162b490a91ac5445638469bfa54c924cc085fa93c31a72c0c49b0ae6ae7c`.
