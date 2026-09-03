# Reproducibility fingerprints, provenance, and RO-Crate

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/reproducibility-prov-and-ro-crate.Rmd`.

The reproducibility layer consolidates hashes of data, decisions, model specifications, results, package versions, files, and the Python environment into a reproducibility fingerprint.

```python
import eyeprocesspy as ep

fingerprint = ep.eye_reproducibility_fingerprint(
    data=data,
    analysis_spec=spec,
    decisions=manifest,
    result=fit,
)
ep.write_reproducibility_fingerprint(
    fingerprint,
    "fingerprint.json",
)
assert ep.verify_reproducibility_fingerprint(fingerprint)
```

Lineage nodes and edges can use PROV-like relation labels and can be exported to compact machine-readable representations or Graphviz-style graphs. `export_ro_crate_metadata()` writes minimal RO-Crate metadata for interoperability.

```python
crate = ep.export_ro_crate_metadata(
    fingerprint,
    "ro-crate-metadata.json",
)
```

This is interoperability scaffolding, not a claim of full external conformance unless independent validation establishes that conformance. Fingerprints show whether declared inputs and analysis decisions reproduce; they do not establish that the scientific model is substantively correct.
