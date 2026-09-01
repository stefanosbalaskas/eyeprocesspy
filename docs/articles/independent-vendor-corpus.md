# Independent multi-vendor validation corpus

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/independent-vendor-corpus.Rmd`.

## Evidence levels

Vendor support is reported at three distinct levels:

1. **declared** — an adapter and semantic mapping exist;
2. **fixture-tested** — synthetic or developer fixtures pass;
3. **empirically-validated** — independent real exports with version metadata, licensing review, validation artifacts, and successful audits pass.

The third level cannot be created by software alone.

## Registering a case

```python
import eyeprocesspy as ep

corpus = ep.init_vendor_corpus("private/vendor-corpus")

ep.register_validation_case(
    corpus_path=corpus,
    source_path="private/exports/tobii-study-01",
    vendor="Tobii",
    device_model="Tobii Pro Spectrum",
    software_name="Tobii Pro Lab",
    software_version="1.x",
    export_profile="gaze data + events",
    sampling_rate_hz=300,
    coordinate_system="display area normalized",
    timebase="microseconds",
    event_semantics="stimulus and trial markers",
    ocular_structure="binocular",
    missingness_convention="vendor validity codes",
    vendor_fixations="exported fixation filter retained separately",
    package_transformations="canonical renaming and unit normalization",
    unsupported_fields="declared explicitly",
    independent_source=True,
    licence_reviewed=True,
    redistribution_allowed=False,
    support_level="empirically-validated",
    mode="reference",
)
```

## Fingerprinting, redaction, and semantics

```python
ep.fingerprint_validation_case("private/exports/tobii-study-01")
ep.redact_validation_case(
    corpus,
    "tobii-study-01",
    remove_columns=("participant_name",),
)
```

Semantic mappings should record the native field, native meaning, canonical table/field, unit, transformation, loss risk, and evidence case that supports the mapping. Cross-vendor comparison should operate on these explicit semantics rather than assuming similarly named fields are equivalent.

```python
ep.compare_vendor_semantics(corpus)
```

## Compatibility evidence

```python
ep.build_compatibility_matrix(corpus, min_empirical_cases=2)
ep.audit_vendor_field_coverage(corpus)
ep.audit_roundtrip_loss(original, roundtrip)
ep.write_vendor_case_report(corpus, "tobii-study-01")
```

A production compatibility claim should remain version-specific and identify unsupported semantics rather than presenting a binary vendor badge. Independent validation evidence, redistribution permissions, and source provenance remain separate from adapter implementation.
