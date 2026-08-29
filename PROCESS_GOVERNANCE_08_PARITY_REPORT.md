# Process governance / windows / advanced pupil parity report

Frozen R reference: `eyeprocess 0.11.1`.

This tranche covers the frozen exported APIs from:

- `R/058-process-preflight-governance-0-8.R` — 13 exports
- `R/059-deployment-drift-0-8.R` — 8 exports
- `R/060-process-window-representations-0-8.R` — 10 exports
- `R/061-pupil-advanced-representations-0-8.R` — 19 exports
- five standalone exported plotting functions from `R/064-next-generation-plots-0-8.R`

Total newly ledgered frozen exports: **55**.

## Implemented contracts

The Python layer now provides executable contracts for preflight review/exclusion manifests, multivariate process anomaly review, presentation/accessibility sensitivity, deployment drift, temporal process-window extraction and sensitivity, AOI trajectories/growth curves, pupil frequency/activity features, event-kernel deconvolution, luminance/trial-order confound adjustment, fatigue/drift sensitivity, and auditable signal filtering.

Scientific review boundaries from the R implementation are retained: process-quality and presentation flags are review signals, not psychological, clinical, ability, misconduct, or causal labels.

## Backend identity

Five functions are conservatively marked `python_reference_differs` because the frozen R package can select optional R engines that have no algorithmically identical Python backend in this port:

- `fit_pupil_confound_model()` — R `mgcv` path; Python exposes the transparent LM reference path and explicitly gates `mgcv`.
- `audit_pupil_fatigue_drift()` — R `plm` path; Python exposes the fixed-effects LM reference path and explicitly gates `plm`.
- `filter_eye_signal()` and `filter_pupil_signal()` — R `robfilter` path; Python exposes the running-median reference and explicitly gates `robfilter`.
- `compare_signal_filters()` — only available Python reference filters are compared; unavailable `robfilter` is not impersonated.

All other new functions are marked `source_ported` at P5. Cross-language P4 numerical parity is **not** inferred from Python-only tests.

## Plot parity

Twenty plot-ledger rows are now verified:

- 15 S3-style Matplotlib counterparts for preflight, anomaly, accessibility/fairness, drift, process windows, AOI trajectories/growth curves, pupil frequency/stability/deconvolution/confounds/fatigue, and signal filtering;
- five standalone R/064 plotting exports: `plot_pupil_spectrum()`, `plot_pupil_band_power()`, `plot_pupil_activity_windows()`, `plot_pupil_activity_sensitivity()`, and `plot_process_window_sensitivity()`.

Each counterpart exposes `eyeprocess_plot_data` for data-layer smoke validation.

## Documentation and examples

Four additional frozen articles have Python counterparts:

- `process-preflight-and-anomaly-governance`
- `deployment-drift-monitoring`
- `temporal-process-representations`
- `advanced-pupillometry-representations`

The already-complete item-seeding/accessibility/presentation-fairness article also covers the presentation-sensitivity interfaces. Four new executable examples are included and the complete `irt_*.py` example smoke suite passes.

## Validation

- Focused R/058–061 Python contract tests: **15 passed**.
- Frozen argument-name signature audit for the 55-export tranche: **55/55 matched**, using deterministic R `...` → Python variadic mapping.
- Full collected Python regression surface: **197 tests**, all passing in deterministic split batches because the monolithic invocation exceeds this sandbox's process-time ceiling.
- Installed validation-wheel smoke: PASS.
- Canonical Stan resources in wheel: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `255045c596faaf0798e1bd83f06cea6e24b91e569a112cb301a5433a30641c87`.

Extended R-oracle numerical validation remains pending because `Rscript` is unavailable in this sandbox.
