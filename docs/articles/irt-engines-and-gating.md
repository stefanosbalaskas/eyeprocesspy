# External IRT engines and exact-method gating

This is the Python counterpart of the frozen R article **External IRT engines and exact-method gating**.

`eyeprocesspy` keeps estimator identity inspectable. The frozen `eyeprocess 0.11.1` ecosystem includes `mirt`, `TAM`, `GDINA`, `LNIRT`, `eRm`, `equateIRT`, `catR`, `mirtCAT`, and—through the stable external-adapter registry—`brms`, `OpenMx`, `diffIRT`, `TraMineR`, `seqHMM`, `eyetrackingR`, and `PupillometryR`.

```python
import eyeprocesspy as ep

print(ep.eyeprocess_irt_engine_registry())
print(ep.external_model_engines())
```

These names refer to the **exact R engines used by the frozen R package**. The Python parity layer does not treat a similarly named Python library, or a different estimator solving a related problem, as the same engine.

## Two complementary gating contracts

The 0.9 IRT-specific wrappers such as `fit_eyeprocess_mirt()` and `fit_eyeprocess_gdina()` return a gated IRT result when the exact frozen engine is unavailable. The later stable adapter API provides a broader three-state contract:

```python
responses = [[1, 0], [0, 1]]
result = ep.fit_external_engine(
    "mirt",
    responses,
    purpose="confirmatory IRT model",
)

print(result.status)  # fitted, not_available, or failed
print(ep.validate_engine_adapter(result).findings)
```

In a pure-Python installation the exact R engines are reported as `not_available`. This is deliberate: availability is not silently converted into a different model.

Convenience wrappers preserve the declared scientific purpose:

```python
mirt = ep.fit_mirt_adapter(responses, purpose="multidimensional IRT")
tam = ep.fit_tam_adapter(responses, purpose="Rasch sensitivity analysis")
lnirt = ep.fit_lnirt_adapter(responses, purpose="joint accuracy-response-time model")

print(ep.compare_engine_adapters(mirt, tam, lnirt))
```

The final frozen `fit_gdina_adapter()` is also preserved. For an `eye_dataset`, its Q-matrix is validated before the exact GDINA backend gate; for raw response data it follows the stable adapter-result contract. This matches the later `R/028` definition that overrides the earlier `R/020` helper in `eyeprocess 0.11.1`.

## Cross-engine equivalence is separate from adapter availability

`compare_model_engines()` is a backend-neutral validation harness. It runs named fitting functions supplied by the researcher, extracts named parameters, compares each engine with a declared reference, records fitting failures rather than dropping them, and evaluates a stated tolerance.

```python
import numpy as np
import pandas as pd

z = pd.DataFrame({"x": np.arange(8.0)})
z["y"] = 1 + 2 * z.x

comparison = ep.compare_model_engines(
    z,
    engines={
        "a": lambda d: np.polyfit(d.x, d.y, 1),
        "b": lambda d: np.polyfit(d.x, d.y + 1e-10, 1),
    },
    extractors=lambda fit: {"beta": float(fit[0])},
    reference="a",
    tolerance=1e-8,
)

ep.plot_eye_engine_comparison(comparison, parameter="beta")
```

Equivalence within a numerical tolerance is evidence about the selected estimand and fixture; it is not proof that two engines are generally interchangeable.

## Sequence and process-package bridges

The frozen interoperability layer is also represented:

- `as_procdata_sequence()` produces a package-neutral long action table;
- `as_traminer_sequence()` produces the wide state table and gates creation of a native TraMineR object;
- `as_seqhmm_data()` produces sequences, lengths, alphabet, and trial index metadata;
- strict legacy `fit_diffirt_adapter()` and `fit_openmx_process_model()` preserve their validation-before-backend-error behavior.

The governing principle is unchanged from R: **adapter existence, engine availability, successful fitting, numerical agreement, and scientific validation are different claims.**
