# Research validation and software-paper programme

## Parameter recovery and interval coverage

Use the validation/recovery APIs with declared simulation truth and preserve failed replications. Coverage is a property of the full simulation programme, not a single successful fit.

## Grouped validation and leakage

Person- and item-grouped validation is required when process features can leak stable participant or item information into prediction. Crossed grouping should be used when both sources of dependence matter.

## Preprocessing multiverse

Preprocessing choices—especially pupil baselines, latency shifts, interpolation thresholds and basis dimension—must be treated as analysis decisions. `pupil_preprocessing_grid()` and `pupil_preprocessing_sensitivity()` make those decisions inspectable.

## Reproducible release assets

Public benchmarks, session metadata, parity manifests, validation summaries and plots should be archived alongside formal releases. Reproduction claims require exact public materials, scoring rules and published estimands.

## Explicit confirmatory gates

Simulation-based calibration, engine comparison and empirical reproduction are evidence-producing operations. Unrun simulations or inaccessible vendor corpora are never represented as completed evidence.

## Systematic advanced-model grid

```python
full_design = ep.advanced_validation_grid()
full_factorial = ep.advanced_validation_grid(full_factorial=True)
```

The default grid is a one-factor-at-a-time screening design. The Cartesian grid is intentionally large and should be executed only on declared computing infrastructure with preserved failures, summaries, diagnostics and provenance.
