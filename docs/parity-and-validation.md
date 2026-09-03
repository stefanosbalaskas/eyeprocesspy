# Parity and validation

`eyeprocesspy` distinguishes several kinds of parity. A function name being present is necessary, but not sufficient, for a defensible scientific port.

## Frozen source

The scientific reference is R `eyeprocess` 0.11.1 at commit `d867555eecae46f262843501c07074cebe1f7aa9`. The frozen namespace contains 1,182 public exports.

## Parity ledger

`parity/PARITY_MATRIX.csv` tracks the public surface across seven evidence dimensions:

- `p1_api`: public API presence;
- `p2_structural`: structural/schema contract;
- `p3_semantic`: semantic behavior;
- `p4_numerical`: numerical/oracle evidence;
- `p5_algorithmic`: source-algorithm correspondence;
- `p6_plot`: plot/data contract where applicable;
- `p7_docs_examples`: documentation/example coverage.

The 0.1.0 release is not allowed to rely on `p1_api` alone. Release validation audits the remaining p3/p4/p6/p7 evidence and records legitimate cross-language differences explicitly.

## Numerical parity

Exact numerical equality is tested where the same deterministic mathematical contract can be represented in R and Python. Tolerances are declared by the relevant oracle test rather than chosen after inspecting a discrepancy.

Some values cannot be byte-identical across languages or environments. Examples include:

- native RDS serialization;
- R namespace-specific fitted model objects;
- R versus NumPy random-number streams;
- Python versus R object serialization hashes;
- wall-clock benchmark timings and memory estimates;
- renderer-specific pixel output.

These cases use the `python_reference_differs` status only when the blocker states why exact identity is inappropriate and the shared scientific contract is tested independently.

## Plot parity

Plot parity focuses on the scientific data contract: values, ordering, transformations, grouping, labels and matrices supplied to the renderer. Cross-platform pixel identity is not used as a scientific parity criterion. Render smoke tests ensure the Python plotting backend can materialize the result.

## External engines

An unavailable exact estimator is gated. A similarly named Python library is not automatically equivalent to an R package, and `eyeprocesspy` does not substitute one silently.

## Validation is not construct validity

Recovery, SBC, reliability, stress tests, negative controls, benchmark reproducibility and evidence freezes establish properties of software and declared measurement/statistical workflows. They do not certify that a psychological, behavioral or physiological construct is valid in a particular study.
