# Release validation — development checkpoint 0.1.0.dev0

This is **not a formal parity release**.

## Frozen source

- R reference: `eyeprocess` 0.11.1
- SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Tag commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- Frozen scope: 1,182 exports / 435 S3 registrations / 88 articles / 113 testthat files / 13 Stan programs

## Current checkpoint

- **349** frozen exports have genuine Python implementations.
- **85** local pytest tests pass.
- **12** Python article counterparts are complete.
- **13** executable IRT examples pass.
- **62** explicit plot counterparts are exported.
- Wheel build + installed-wheel smoke: PASS.
- Installed wheel contains **13/13** Stan resources.
- Wheel SHA-256: `0df0ee5098b85f7ca38ff3f14e4d77e3a69d7beaff29a10827f488cb90bdf681`.
- M4 remains evidence-gated; canonical Stan MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.

## Not yet a parity release because

- 833 frozen exports remain outside the implemented API.
- 76/88 article counterparts remain incomplete.
- Cross-language numerical validation is incomplete for the expanded statistical surface.
- 24 functions intentionally use Python reference estimators differing from R optional-engine algorithms and require backend-specific validation.
- Full multi-platform CI, documentation-site parity and optional-backend lanes are not yet frozen green.
