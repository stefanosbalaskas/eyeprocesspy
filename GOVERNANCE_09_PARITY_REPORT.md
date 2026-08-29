# Governance 0.9 parity report

## Frozen R scope

This tranche ports the public contracts from the frozen `eyeprocess 0.11.1` sources:

- `R/069-validation-program-0-9.R`
- `R/070-governed-pipelines-0-9.R`
- `R/071-api-lifecycle-0-9.R`
- `R/072-sensitivity-multiverse-0-9.R`
- `R/073-decision-manifests-0-9.R`

The tranche contains **80 frozen exports**. All 80 resolve as Python callables and have translated contract/smoke coverage in `tests/test_governance_09.py`.

## Implemented capability families

- Empirical validation programme design, simulation, execution, frozen references, reference comparison and evidence matrices.
- Governed pipeline specifications, step/dependency graphs, execution/resume contracts, audit/report output and DOT/Mermaid/targets interoperability templates.
- API lifecycle registry, inventory, status, audits, diffs and lifecycle recommendations using the packaged frozen 1,182-row registry and 108-row module policy.
- Multiverse/sensitivity grids, execution, specification curves, sign/significance/threshold/rank stability, leverage, fragility and method comparison.
- Decision manifests, hashing/locking, comparison, read/write, provenance, blinded snapshots, decision entropy and decision-space coverage.

## Plot/documentation surface

Nine data-bearing Matplotlib counterparts were added for the frozen governance S3 plot families. Five frozen article counterparts and five executable examples were added. Plot objects expose `eyeprocess_plot_data` for smoke-testable scientific data.

## Algorithmic boundary

**76/80** functions are recorded as source-ported contracts/algorithms. Four remain deliberately marked `python_reference_differs` rather than claiming false R identity:

- `eye_api_inventory()` adapts namespace inventory to Python while retaining the frozen R reference inventory mode.
- `freeze_validation_reference()` uses Python serialization for local persisted references rather than RDS.
- `write_decision_manifest()` and `read_decision_manifest()` use JSON/Python serialization rather than native R serialization.

No P4 cross-language numerical parity is inferred from Python-only tests.

## CI portability repairs bundled with this checkpoint

The GitHub test environment exposed two dependency declarations and one pandas-3 compatibility issue in earlier implemented paths. This checkpoint therefore also records:

- `matplotlib>=3.9`, `patsy>=1.0`, and `statsmodels>=0.14` in the development test extra;
- a writable copy when converting dynamic-IRTree `time_gap` values to NumPy, avoiding pandas 3 read-only array mutation errors.

These are portability fixes to existing implementations, not new frozen-export parity claims.

## Checkpoint accounting

- Frozen exports implemented: **810 / 1,182**
- Source-ported algorithms/contracts: **724**
- Python-reference/backend-different algorithms: **86**
- Article counterparts complete: **61 / 88**
- Verified plot-ledger rows: **172 / 341**
- Executable `irt_*.py` examples: **49**
- Source regression: **219 / 219 passed** in deterministic split batches
