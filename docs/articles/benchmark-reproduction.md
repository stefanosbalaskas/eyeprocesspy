# Public benchmark and software-paper reproduction

This article is the Python migration of the frozen `eyeprocess 0.11.1` vignette `vignettes/benchmark-reproduction.Rmd`.

## Bundled multimodal study

The package includes a compact, fully synthetic and openly redistributable benchmark containing participant and item metadata, binary responses, response times, recordings, gaze samples, events, AOIs, pupil trajectories, quality indicators, and provenance.

```python
import eyeprocesspy as ep

study = ep.eyeprocess_benchmark_study()
```

The benchmark tests contracts and reproducibility. It is **not** real participant data, vendor compatibility evidence, or scientific validation of advanced models.

## Integrity and expected outputs

```python
validation = ep.validate_benchmark_study(study)
assert validation.valid

reproduction = ep.run_benchmark_reproduction(study)
comparison = reproduction.comparison
```

Source assets are fingerprinted. Relational checks ensure that participants, items, trials, events, gaze, and pupil records agree. Expected scalar outputs use explicit numerical tolerances so reproducibility is tested as a scientific contract rather than a byte-for-byte accident.

## Data dictionary and reproduction directory

```python
ep.write_benchmark_data_dictionary(
    study,
    "benchmark-data-dictionary.md",
)
ep.write_software_paper_reproduction(
    "software-paper-reproduction",
    study,
)
```

The reproduction scaffold contains the benchmark data and the files needed to recreate the expected analysis outputs, together with a manifest of the environment and source evidence.

## Release audit

```python
release_audit = ep.audit_benchmark_release(study)
assert release_audit.ready
```

A public empirical benchmark can later be added under an independently reviewed licence. It must remain distinct from this synthetic test asset, and its existence must not be implied before such evidence is actually available.
