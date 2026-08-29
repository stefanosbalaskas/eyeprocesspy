# Release validation — development checkpoint 0.1.0.dev0

This is **not a formal parity release**.

## Frozen source

- R reference: `eyeprocess` 0.11.1
- SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Tag commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- Frozen scope: 1,182 exports / 435 S3 registrations / 88 articles / 113 testthat files / 13 Stan programs

## Current checkpoint

- **286** frozen exports have genuine Python implementations.
- No generated placeholder exports are counted.
- **72** local pytest tests pass.
- **11** Python article counterparts are complete in the parity ledger.
- **11** executable IRT examples pass.
- **44** explicit IRT/process plot counterparts are currently exported.
- M4 remains evidence-gated and its canonical Stan source MD5 is `c5af3e5d25ff63db42c58573eb42124b`.

## Parity caveats

- Cross-language numerical R-oracle validation is pending for the extended IRT/process surface.
- 13 process-model functions currently use documented Python reference estimators where the R implementation delegates to optional R-only engines; they are marked `python_reference_differs` rather than algorithmically identical.
- Full 1,182-export, 88-article and 435-S3 parity remains incomplete.
- Clean networked dependency installation and multi-platform CI must pass before a formal release.
