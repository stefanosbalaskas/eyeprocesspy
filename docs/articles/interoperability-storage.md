# Interoperability, Eye-Tracking-BIDS, and storage

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/interoperability-storage.Rmd`.

## Eye-Tracking-BIDS

```python
import eyeprocesspy as ep

ep.export_eye_bids(
    dataset,
    "bids-eye-study",
    task="reasoning",
    screen_distance_m=0.60,
    screen_size_m=(0.53, 0.30),
    overwrite=True,
)

roundtrip = ep.import_eye_bids("bids-eye-study")
ep.validate_eye_dataset(roundtrip)
```

The exporter keeps the physiological recording columns and their sidecar metadata explicit. Interoperability output does not erase the original source representation or provenance.

## Columnar storage

```python
handle = ep.write_eye_storage(
    dataset,
    "storage",
    format="parquet",
    overwrite=True,
)
restored = ep.collect_eye_storage(handle)
```

Storage format is an engineering choice. Canonical identifiers, units, clocks, quality fields, and provenance should survive round trips.

## Package ecosystems

```python
x = ep.as_eyeprocess_eyetools(
    external_object,
    mapping=declared_mapping,
)
proc = ep.as_procdata_sequence(x)
seqs = ep.as_traminer_sequence(x, create_object=True)
hmm_data = ep.as_seqhmm_data(x)
```

Adapters expose the assumptions needed to translate between ecosystems. They should not imply semantic equivalence when the external object lacks information required by the canonical model.
