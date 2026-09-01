# Building a software-paper evidence bundle

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/software-paper-evidence-bundle.Rmd`.

The paper-evidence layer links manuscript claims to explicit evidence IDs, evidence types, qualifications, validation outputs, examples, articles, benchmarks, and reproducibility fingerprints.

```python
import eyeprocesspy as ep

claims = ep.software_paper_claim_matrix(
    claim=(
        "Pipeline transformations are traceable",
        "Recovery is characterized under declared simulation regimes",
    ),
    evidence_id=("PIPE-01", "VAL-01"),
    evidence_type=("integration test", "simulation"),
    status=("supported", "qualified"),
)

bundle = ep.software_paper_evidence_bundle(
    claims=claims,
    validation=validation,
    examples=examples,
    articles=articles,
    reproducibility=fingerprint,
)

readiness = ep.software_paper_readiness(bundle)
gaps = ep.software_paper_gap_analysis(bundle)
```

Readiness is a completeness audit against declared requirements. It is not a prediction of peer-review acceptance, scientific importance, or construct validity. Qualified and missing evidence should remain visible rather than being converted into binary success labels.
