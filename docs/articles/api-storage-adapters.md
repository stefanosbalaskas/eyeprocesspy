# Stable APIs, scalable storage, and external adapters

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/api-storage-adapters.Rmd`.

## Contracts

```python
import eyeprocesspy as ep

ep.eyeprocess_api_version()
ep.object_schema("eye_dataset")
ep.object_schema("eyeprocess_model")
ep.validate_model_object(fit)
ep.upgrade_eye_dataset(old_data)
ep.upgrade_eyeprocess_model(old_fit)
```

Schemas lock required components, identifiers, return-value expectations, serialization compatibility, error classes, and scientific safeguards. `eyeprocess_deprecation()` records replacement and removal horizons rather than silently changing public behavior.

## Partitioned storage

```python
spec = ep.partition_eye_storage(
    by=("participant_id", "session_id", "recording_id"),
    format="parquet",
    compression="zstd",
    max_rows=1_000_000,
)

store = ep.write_partitioned_eye_storage(x, "analysis/store", spec)
subset = ep.query_eye_storage(
    store,
    table="gaze_samples",
    filters={"participant_id": ("P001", "P002")},
    columns=("participant_id", "recording_id", "time", "x", "y"),
)

ep.validate_eye_storage_metadata(store)
ep.detect_corrupt_partitions(store)
ep.storage_transaction_manifest(store)
```

Writes use staging followed by an atomic commit. Every partition retains row count, byte count, partition keys, and a fingerprint. Storage backends are implementation choices; they do not change the scientific meaning of the canonical records.

## Schema migration and benchmarks

```python
ep.migrate_eye_storage_schema(
    store,
    "analysis/store-v2",
    target_version="2.0.0",
)

ep.benchmark_eye_storage(
    x,
    formats=("pickle", "csv", "parquet"),
)
```

## External engines

```python
ep.external_model_engines()
ep.fit_mirt_adapter(
    response_matrix,
    model=1,
    purpose="unidimensional item calibration",
)
ep.fit_tam_adapter(
    response_matrix,
    purpose="Rasch sensitivity analysis",
)
ep.fit_brms_adapter(
    formula="score ~ dwell + (1|participant_id) + (1|item_id)",
    data=trials,
    purpose="Bayesian explanatory model",
)
ep.fit_lnirt_adapter(
    {"Y": response_matrix, "RT": rt_matrix},
    purpose="joint accuracy-RT comparison",
)
```

Every adapter reports an explicit outcome such as fitted, not available, or failed. Adapters do not install external software, choose a scientific model, or reinterpret backend outputs automatically. Backend differences remain visible in parity and provenance evidence.
