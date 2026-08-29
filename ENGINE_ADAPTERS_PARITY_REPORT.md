# External engine adapters parity report

Checkpoint: `eyeprocesspy 0.1.0.dev0` against frozen R `eyeprocess 0.11.1`.

## Scope

This tranche ports **27 frozen exported functions** from the stable external-engine/model-contract and sequence-interoperability layers, plus the S3 plot counterpart for engine comparison.

Implemented frozen exports:

- stable API/model contracts: `eyeprocess_api_version()`, `object_schema()`, `validate_model_object()`, `upgrade_eyeprocess_model()`, `eyeprocess_deprecation()`;
- adapter registry/contracts: `external_model_engines()`, `engine_adapter_status()`, `fit_external_engine()`, `validate_engine_adapter()`, `compare_engine_adapters()`;
- convenience adapters: `fit_mirt_adapter()`, `fit_tam_adapter()`, `fit_brms_adapter()`, `fit_lnirt_adapter()`, `fit_traminer_adapter()`, `fit_seqhmm_adapter()`, `fit_gdina_adapter()`, `fit_openmx_adapter()`, `fit_diffirt_engine_adapter()`, `fit_eyetrackingr_adapter()`, `fit_pupillometryr_adapter()`;
- sequence bridges: `as_procdata_sequence()`, `as_traminer_sequence()`, `as_seqhmm_data()`;
- strict legacy adapters: `fit_diffirt_adapter()`, `fit_openmx_process_model()`;
- equivalence harness: `compare_model_engines()`.

Python plot counterpart:

- `plot_eye_engine_comparison()` for frozen `plot.eye_engine_comparison`.

## Runtime-definition correction

`fit_gdina_adapter()` is defined in both `R/020-interoperability-storage.R` and `R/028-api-storage-adapters.R`. The later `R/028` definition is the runtime definition in `eyeprocess 0.11.1`, and its Rd usage already documents:

`fit_gdina_adapter(data, Q, model = "GDINA", purpose = "cognitive diagnosis", ...)`

The frozen Python manifests were corrected to this final definition rather than retaining the shadowed earlier signature.

## Engine-identity policy

The stable R registry names exact R engines: `mirt`, `TAM`, `brms`, `LNIRT`, `GDINA`, `OpenMx`, `diffIRT`, `TraMineR`, `seqHMM`, `eyetrackingR`, and `PupillometryR`.

The pure-Python parity core does **not** treat related Python estimators as these exact engines. Therefore:

- exact-engine convenience wrappers return a structured `not_available` adapter result where the stable R adapter does so;
- the older strict GDINA/diffIRT/OpenMx paths raise an explicit backend error after the same pre-backend validation checks as R;
- no function silently chooses a replacement estimator;
- successful adapter availability is not treated as scientific validation.

This is why **14 functions are conservatively classified `python_reference_differs`** at P5. Their public/structural/semantic contracts are implemented, but exact R-engine algorithmic identity is not claimed.

## Validation

- New focused adapter tests: **8 passed**.
- Full local package suite after this tranche: **156 passed**.
- Executable `irt_*.py` examples: **30 passed** through the global example-smoke suite.
- Installed offline validation-wheel smoke: PASS.
- Canonical Stan resources in installed wheel: **13/13**.
- Canonical M4 Stan MD5: `c5af3e5d25ff63db42c58573eb42124b`.
- Validation-wheel SHA-256: `0ae7c56fa006c798e13f8adcf0f000aed57d5193da1fd29377ace8894d5a5365`.

## Parity accounting after tranche

- Frozen exports implemented: **539 / 1,182**.
- P5 source-ported algorithms/contracts: **470**.
- P5 Python-reference/backend-different functions: **69**.
- P4 numerical parity: unchanged; extended R oracle still pending.
- Verified plot-ledger entries: **55 / 341**.
- Complete article counterparts: **43 / 88**.

`compare_model_engines()` is source-ported as an executable backend-neutral equivalence harness and records heterogeneous engine failures instead of dropping them. Its Python plot counterpart attaches the exact plotted data frame to the Matplotlib axis for smoke/regression testing.
