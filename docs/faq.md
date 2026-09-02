# Frequently asked questions

## Is eyeprocesspy released on PyPI?

Not yet. Version `0.1.0` is a release candidate on `release/0.1.0-deep-parity`. Use the CI-tested manual bundle/wheel or install directly from the release branch until the archival release gate is complete.

## Does the Windows manual installer work?

Yes. The bundle has been successfully installed on Windows with Python 3.11.9 and verified as:

```text
eyeprocesspy: 0.1.0
R reference: 0.11.1
```

See [Manual installation](manual-install.md).

## Why did pip reject `eyeprocesspy-0.1.0-py3-none-any (1).whl`?

The browser-added ` (1)` breaks the standardized wheel filename. The wheel itself can still be valid. Rename it to:

```text
eyeprocesspy-0.1.0-py3-none-any.whl
```

or use the manual bundle, which preserves the canonical filename.

## Is the wheel Windows-specific?

No. `py3-none-any` indicates a pure-Python wheel without a platform-specific compiled wheel tag. Its dependencies may still install platform-specific binaries through their own packages.

## Why is the deep-parity badge red if the package tests pass?

The deep-parity workflow intentionally fails its **final release gate** until both statement and branch coverage are exactly 100%. A red deep-parity run can therefore coexist with a fully passing pytest suite, wheel check, frozen-R oracle, and documentation build.

Always inspect the workflow steps rather than interpreting the final red/green state alone.

## Does “deep parity” mean the Python source is a line-by-line translation of R?

No. Scientific parity concerns public contracts, semantics, numerical behavior, validation, plots, examples, and documented differences. Pythonic implementation details can differ when that does not change the scientific contract.

## What happens when an R backend has no faithful Python equivalent?

The difference is recorded explicitly. `eyeprocesspy` does not silently substitute a different estimator while presenting it as exact parity. Functions may raise a backend/parity error or expose a documented Python-reference implementation where appropriate.

## Is eyeprocesspy only for Gazepoint?

No. The package is vendor-neutral at its core. Gazepoint receives first-class support because the ecosystem includes dedicated import/validation workflows, but downstream analyses operate on the canonical `EyeDataset` model.

## Can I use vendor-derived fixations?

Yes, when they are imported into the canonical episode representation. You should document whether the analyzed fixation events came from the vendor or were derived by `eyeprocesspy`, because algorithm/settings differences can materially affect results.

## Are gaze coordinates assumed to be pixels?

No. Coordinate spaces are explicit. Register/audit the relevant coordinate space and convert only when the source/target geometry is known. Never infer pixels, normalized coordinates, or visual angle solely from the numerical range without source documentation.

## Does `validate_eye_dataset()` mean my data are scientifically valid?

No. It checks canonical/schema-level consistency and related contracts. Scientific readiness also requires appropriate checks of calibration, sampling, missingness, events, coordinates, task design, preprocessing, feature definitions, and study-specific validity.

## Does a gaze metric measure attention?

Not automatically. Fixation, dwell, transition, scanpath, pupil, and other process metrics are observations or derived measurements. Psychological interpretation requires a design and validation evidence that support the construct claim and address plausible alternatives.

## Does pupil dilation measure cognitive load?

Not by itself. Pupil size is affected by multiple physiological, optical, task, environmental, and cognitive factors. Baseline, luminance, timing, missingness, event structure, and competing explanations must be addressed.

## What does probabilistic AOI assignment mean?

In the calibration-uncertainty workflow, it represents the probability of AOI membership after propagating an empirical gaze-coordinate error model. It is **not** a probability that the participant psychologically attended to the AOI.

## Is reliability evidence enough to validate a process metric?

No. Reliability concerns repeatability/consistency under a design. Construct validity concerns what the measure means. A measure can be highly reliable and substantively invalid.

## Can I use process features in machine learning or psychometrics?

Yes, but define the measurement level and validation unit carefully. Random row-wise splitting can leak participant, item, stimulus, session, or device information. Use grouped/leakage-aware validation when generalization requires independence across those units.

## Can I use the plots in publications?

The plotting functions return standard Matplotlib axes, so figures can be refined and exported through ordinary Matplotlib workflows. Many plots also attach the underlying table/matrix to the returned axes for auditing.

See [Plotting reference](reference/plotting.md) and the [visual gallery](gallery.md).

## How can I reproduce the gallery?

Run:

```bash
python examples/core_gallery.py
python examples/advanced_gallery.py
```

The examples are deterministic and do not require private study data.

## Why are there so many articles?

The frozen reference package spans import, preprocessing, gaze, AOIs, pupillometry, process measurement, psychometrics, validation, reproducibility, governance, storage/interoperability, and advanced models. The 88 linked articles preserve that scientific workflow context.

Use the [Featured workflow map](articles/featured-workflows.md) rather than reading them alphabetically.

## Where should a new user start?

1. [Getting started](getting-started.md)
2. [End-to-end eye-tracking](guides/end-to-end-eye-tracking.md)
3. [Visual gallery](gallery.md)
4. [Runnable examples](examples/index.md)
5. the domain guide relevant to the study
6. [Parity and validation](parity-and-validation.md) when auditing scientific fidelity.
