# Validation evidence atlas and software-paper reporting

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/validation-evidence-atlas-software-paper.Rmd`.

The evidence atlas links declared claims to executed outputs, tables, figures, hashes, provenance, and limitations while preserving the distinction between software behavior and substantive validity.

```python
import eyeprocesspy as ep

claims = ep.eyeprocess_validation_claim_matrix(
    ("C1", "C2"),
    (
        "Scenario expansion is deterministic",
        "Exact engines are not silently substituted",
    ),
    ("E1", "E2"),
    ("test", "engine-contract"),
    ("supported", "qualified"),
)

atlas = ep.eyeprocess_validation_evidence_atlas(
    claims,
    recovery={"id": "demo"},
)
```

`freeze_eyeprocess_validation_atlas()` creates an integrity-checked snapshot. `write_eyeprocess_validation_report()` emits an archival Markdown report, while the table helpers convert recovery, SBC, stress, reliability, negative-control, IRT-precision, and external-engine status into compact reporting objects.

```python
frozen = ep.freeze_eyeprocess_validation_atlas(atlas)
assert ep.verify_eyeprocess_validation_atlas(frozen)

ep.write_eyeprocess_validation_report(
    frozen,
    "validation-evidence-report.md",
)
```

The atlas records the organization and provenance of evidence. A frozen or visually complete atlas does not convert software evidence into evidence for a substantive psychological interpretation.
