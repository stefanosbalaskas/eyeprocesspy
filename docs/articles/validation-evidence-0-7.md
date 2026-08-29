# Independent Vendor Validation and Semantic Fidelity

## Why import success is not validation

`eyeprocesspy` treats vendor compatibility as an evidence claim. Reading a file without an exception establishes parser reachability only; it does not establish that timestamps, coordinate systems, eye identity, pupil units, event meanings, or missingness semantics survived harmonization.

Use `validation_evidence_levels()` to retain the frozen 0.7 evidence ladder: `declared`, `synthetic-fixture`, `vendor-example`, `independent-public-real`, `multisession-multidevice-real`, and `semantic-roundtrip-validated`.

## Public validation corpus

`public_validation_corpus()` exposes conservative registry metadata for Gazepoint, EyeLink, Tobii, and Pupil Labs validation sources. The package deliberately does not auto-download human-participant data. Review the source licence and terms before obtaining a corpus.

## Semantic round-trip contract

A strong interoperability test is:

```text
native vendor
  -> eyeprocesspy canonical
  -> interchange/BIDS
  -> eyeprocesspy canonical
  -> field-by-field semantic comparison
```

`field_fidelity_report()` and `semantic_roundtrip_audit()` classify preservation as `LOSSLESS`, `SEMANTICALLY_EQUIVALENT`, `UNIT_TRANSFORMED`, `COORDINATE_TRANSFORMED`, `DERIVED`, `INTENTIONALLY_DROPPED`, `UNSUPPORTED`, `AMBIGUOUS`, or `MISSING`. `semantic_loss_map()` converts the audit into a reviewable table and `plot_eye_semantic_roundtrip()` visualizes it.

## Timestamp, coordinate, pupil, and eye-stream semantics

Use `timestamp_fidelity_audit()`, `coordinate_fidelity_audit()`, `pupil_unit_fidelity_audit()`, and `eye_stream_fidelity_audit()` separately when the scientific meaning of one channel matters. Device and system clocks are not interchangeable merely because both are numeric. Likewise, an affine coordinate/unit conversion is acceptable only when it is detected and documented rather than silently applied.

`validate_vendor_timestamp_semantics()` records expected clock semantics for Gazepoint, Tobii, Pupil Labs/Neon, and generic vendor data without replacing one clock with another.

## BIDS and HED semantics

`validate_bids_eye_semantics()` provides the same lightweight structural checks as the frozen R vignette: required eye-tracking columns, `PhysioType = "eyetrack"`, `RecordedEye`, `SampleCoordinateSystem`, and gaze-on-screen presentation metadata where applicable. It is not a replacement for the official BIDS validator.

`event_semantics_audit()` checks event labels/timing, while `validate_hed_event_semantics()` performs package-level HED structural checks. `event_roundtrip_audit()` combines those checks when HED annotations are available.

## Callback round trips and adapter regression

`roundtrip_eye_bids()` intentionally uses exporter/importer callbacks so the semantic audit remains stable even when a storage adapter changes. `cross_version_adapter_regression()` compares two parser versions against the same fixture and reports `LOSSLESS`, `SEMANTICALLY_EQUIVALENT`, or `REGRESSION_OR_AMBIGUOUS`.

## Evidence matrix

`compatibility_evidence_matrix()` joins case-level evidence to an existing compatibility table. A vendor/device should be promoted only from retained evidence, not undocumented impressions.
