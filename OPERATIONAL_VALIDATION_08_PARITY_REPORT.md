# Operational validation and process-decision parity report

Frozen R reference: `eyeprocess 0.11.1`.

This tranche covers:

- **9** frozen exports from `R/063-operational-validation-0-8.R`;
- **5** frozen exports from `R/066-process-decision-features-0-8.R`;
- the standalone exported `plot_process_feature_stability()` utility from `R/064-next-generation-plots-0-8.R`;
- four corresponding S3-style plot counterparts.

Total newly ledgered frozen exports: **15**.

## Streaming scoring boundary

`score_partial_response_pattern()` in the frozen R package is an exact `mirt::fscores()` adapter. `eyeprocesspy` does not silently replace it with a different Python estimator. The direct partial-score API therefore raises an explicit backend error when the R `mirt` engine is unavailable.

`score_response_stream()` preserves the cumulative response-pattern contract and records each observed-response step. When the exact engine is unavailable, `theta` and `theta_se` remain missing rather than being fabricated. `update_person_score()` preserves the same explicit backend boundary.

These three scoring functions are conservatively marked `python_reference_differs`.

## Validation evidence bundles

`collect_validation_evidence()`, `validation_bundle_manifest()`, `validation_report()`, and `write_validation_report()` port the frozen evidence inventory/reporting contracts and preserve the interpretation guardrails.

`export_validation_bundle()` is deliberately marked `python_reference_differs` for serialization: RDS is R-specific and the Python port does not imitate it with unsafe pickle serialization. Tabular evidence is written as CSV and other safe metadata/evidence representations as JSON/text. This is an explicit parity exception rather than a silent format substitution.

## Process-decision representations

The frozen source algorithms are ported for:

- `preaction_process_features()` — look-back AOI entropy/switches/proportions, pupil mean/slope and blink proportion;
- `addm_glam_proxy_features()` — aDDM/GLAM-inspired descriptive gaze-evidence proxies;
- `process_feature_family_registry()` and `assign_process_feature_family()` — first-match conservative interpretation families;
- `process_feature_stability()` — top-N selection rate and mean importance across repeated splits.

These are observed process representations/proxies, not fitted DDM/GLAM parameters, causal attention effects, or latent intentions.

## Plot and documentation parity

Five plot-ledger rows are verified:

- `plot.eye_streaming_score` → `plot_eye_streaming_score()`
- `plot.eye_validation_bundle` → `plot_eye_validation_bundle()`
- `plot.eye_preaction_process_features` → `plot_eye_preaction_process_features()`
- `plot.eye_decision_process_proxy` → `plot_eye_decision_process_proxy()`
- `plot_process_feature_stability()`

All Matplotlib counterparts expose `eyeprocess_plot_data` for data-layer assertions.

The frozen `streaming-scoring-and-validation-bundles` article now has a Python counterpart. The existing `process-decision-proxies-and-frontier-gates` counterpart was extended with the newly ported pre-action/proxy/stability workflows. Two executable examples were added.

## Validation

- Focused operational tests: **7 passed**.
- Complete executable `irt_*.py` example smoke suite: **44 passed**.
- Full collected Python regression surface: **206/206 passed** in deterministic split batches.
- Installed validation-wheel operational smoke: PASS.
- Canonical Stan resources: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `8b635350be8250ae0fda01e82a651917da840bc440fe7125a2945639ce076217`.

No P4 cross-language numerical parity is inferred from Python-only validation.
