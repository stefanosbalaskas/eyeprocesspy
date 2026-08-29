# Staged multimodal M0-M4 parity report

## Scope

This checkpoint covers the 43 frozen exports in the `eyeprocess` 0.11.1 staged multimodal measurement programme from the 0.10/0.11 source families. The scope includes generic multimodal measurement contracts and the M2, M3 and M4 model/evidence/recovery layers.

## API and algorithmic status

- Frozen exports in tranche: **43**
- Python callables present: **43/43**
- Frozen R argument-name signatures verified: **43/43**
- P1 API: implemented
- P2 structural: implemented initial
- P3 semantic: implemented initial
- P4 numerical: **not claimed**; R oracle pending
- P5 algorithmic: source-ported contracts/gates for all 43

Canonical M2/M3/M4 fitting remains CmdStan-backed. The Python port does not silently replace those models with an unrelated estimator when CmdStanPy/CmdStan is unavailable.

## Plot parity

The tranche adds **29** explicit Matplotlib counterparts for the frozen multimodal S3 plot families. Every plot smoke test asserts a returned Axes and an attached `gp3_data` data layer. Exact typography is not treated as the scientific parity criterion.

## Article and example parity

- Python counterparts for staged frozen articles: **20**
- Executable staged examples: **5**
- Article smoke verifies that the staged documents exist and contain Python-native material without R code chunks.

## M4 governance and evidence boundary

M4 remains REVIEW/evidence-gated. The port preserves:

- explicit sequence/order contracts and no silent sorting;
- person-specific initial and transition probabilities in synthetic truth;
- probability-bearing state diagnostics and secondary MAP labels;
- null/confounded K=1 scenarios;
- state-dependent missingness guardrails;
- negative-control, sensitivity and recovery design objects;
- the interpretation boundary that latent process states are statistical states, not automatic labels for strategy, attention, cognitive load, effort, guessing, misconduct or comprehension.

## Validation

- Focused staged suite: **11 passed**
- Complete package suite after integration: **130 passed**
- Installed validation-wheel smoke: PASS
- Canonical Stan resources in validation wheel: **13/13**
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`

Cross-language numerical validation remains pending because `Rscript` is unavailable in the current sandbox. GitHub CI is configured as the authoritative environment for the frozen R oracle and standard package build.
