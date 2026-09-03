# Multi-vendor empirical validation

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/multi-vendor-validation.Rmd`.

`eyeprocesspy` separates adapter availability, fixture testing, and empirical compatibility. A production claim requires independent real exports for each relevant software version and device family, together with explicit licence and redistribution review.

```python
import eyeprocesspy as ep

corpus = ep.validate_eye_corpus(
    "C:/private/eyeprocess-validation-corpus"
)
audit = ep.audit_vendor_validation(
    corpus,
    ep.vendor_validation_spec(
        required_vendors=("gazepoint", "tobii", "pupillabs", "eyelink", "smi"),
        min_cases_per_vendor=2,
        min_pass_rate=0.95,
        require_licence_reviewed=True,
    ),
)

ep.write_vendor_validation_report(
    audit,
    "validation/vendor-validation.md",
)
```

The audit should retain case-level failures, software/device metadata, semantic coverage, and licence status rather than collapsing everything into a binary vendor badge.

Raw proprietary exports should remain outside the public repository unless redistribution is clearly permitted. Share only manually reviewed, de-identified validation bundles and aggregate evidence. Fixture success is useful engineering evidence, but it must not be relabeled as independent empirical compatibility.
