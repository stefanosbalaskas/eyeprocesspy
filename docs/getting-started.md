# Getting started

## Install

```bash
python -m pip install eyeprocesspy
```

or:

```bash
uv add eyeprocesspy
```

For a development checkout:

```bash
git clone https://github.com/stefanosbalaskas/eyeprocesspy.git
cd eyeprocesspy
uv sync --extra dev
uv run pytest
```

## Import

```python
import eyeprocesspy as ep

print(ep.__version__)
print(ep.__r_reference_version__)
```

The 0.1.0 Python release is tied to the frozen R `eyeprocess` 0.11.1 scientific reference.

## Canonical schema

```python
schema = ep.eye_schema()
```

The package uses a vendor-neutral representation so that imported files retain their source fields and provenance while downstream analyses operate on explicit semantic mappings.

## Import a supported export

```python
data = ep.read_eye_export("path/to/export")
```

For Gazepoint data, dedicated helpers are available for gaze, fixations, events, biometrics, file pairing, validation, and downstream media/trial workflows.

## Audit before analysis

```python
readiness = ep.analysis_readiness(data)
```

Quality and governance functions are designed to make transformations, missingness, sampling assumptions, coordinate conversions, exclusions and feature levels visible rather than silently changing the data.

## Optional backends

Install only the scientific backends you need, for example:

```bash
python -m pip install "eyeprocesspy[plots]"
python -m pip install "eyeprocesspy[psychometrics]"
python -m pip install "eyeprocesspy[stan]"
```

Unavailable exact R engines remain gated. `eyeprocesspy` does not silently replace an unavailable estimator with a different model and call it parity.

## Next steps

The `docs/articles/` directory contains source-ported workflows covering preprocessing, IRT, pupillometry, multimodal measurement, validation, reproducibility, negative controls, benchmarking, evidence governance, and advanced model families.
