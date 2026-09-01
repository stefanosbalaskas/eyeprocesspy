# Frozen validation evidence programme

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/frozen-validation-evidence-programme.Rmd`.

The validation programme separates **software-validation evidence** from substantive construct validity. A reproducible plan fixes scenario families, sample sizes, missingness, perturbations, replications, and seeds before results are interpreted.

```python
import eyeprocesspy as ep

plan = ep.eyeprocess_validation_plan(
    sample_size=(250, 750),
    n_items=(12, 24),
    missing_rate=(0.0, 0.15),
    specification=("correct", "misspecified"),
    replications=20,
    seed=20260811,
)
expanded = ep.expand_eyeprocess_validation_plan(plan)
```

Acceptance criteria are declared reporting contracts rather than universal scientific thresholds.

```python
rules = {
    "rmse": ep.validation_acceptance_rule("rmse", "max", 0.20),
    "failure": ep.validation_acceptance_rule("failure_rate", "max", 0.05),
}
```

Frozen evidence objects contain integrity hashes and source-commit metadata. A passing release gate means only that the declared software-validation criteria were satisfied.

```python
claim = ep.eyeprocess_validation_claim_matrix(
    "C1",
    "software behaviour is reproducible",
    "E1",
    "test",
    "supported",
)

freeze = ep.freeze_eyeprocess_validation_evidence(
    design={"id": 1},
    recovery={"x": 1},
    stress={"x": 1},
    reliability={"x": 1},
    negative_controls={"x": 1},
    claims=claim,
    provenance={"commit": "documentation-example"},
    source_commit="documentation-example",
)

assert ep.verify_eyeprocess_validation_evidence(freeze)
```

The frozen object is provenance-aware software evidence, not a validity assessment of a psychological construct.
