# Release validation — development checkpoint 0.1.0.dev0

This is **not a formal parity release**.

## Frozen source

- R reference: eyeprocess 0.11.1
- SHA-256: `fd2638d7ccf0c5dd5a18745aed59c2f647e142c1753a339a2ab8ca99c3fd5d0a`
- Tag commit: `d867555eecae46f262843501c07074cebe1f7aa9`
- 1,182 exports / 435 S3 / 88 articles / 113 testthat files / 13 Stan programs

## Development checkpoint

- 27 frozen exports have real initial implementations.
- No generated placeholder exports are counted.
- 13 local pytest tests pass.
- Wheel builds and contains all 13 Stan resources.
- M4 Stan MD5 remains `c5af3e5d25ff63db42c58573eb42124b`.

## Not yet validated

- Cross-language R oracle.
- CI matrix on Linux/Windows/macOS and Python 3.11–3.14.
- Optional backend lanes.
- Full 1,182-function parity.
- 88-article parity.
- Plot parity.
