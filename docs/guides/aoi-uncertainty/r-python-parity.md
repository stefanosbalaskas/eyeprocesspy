# R/Python Scientific Parity for AOI Uncertainty

`eyeprocess` and `eyeprocesspy` share the same scientific contract for AOI perturbation and uncertainty analysis. Parity means the same analytical concepts, defaults, branch semantics, failure handling, and output meanings—not necessarily byte-for-byte numerical identity across different language backends.

## Contract shared across both packages

| Concept | R | Python | Contract |
| --- | --- | --- | --- |
| Baseline branch | `baseline` | `baseline` | Unchanged nominal geometry is explicit and retained |
| Outside label | `__outside__` | `__outside__` | No AOI contains the observation |
| Ambiguous label | `__ambiguous__` | `__ambiguous__` | More than one AOI contains the observation |
| Missing coordinates | `NA` | `None`/`NaN` | Preserved as missing, never converted to outside/zero |
| Units | `px`, `deg` | `px`, `deg` | Degree perturbations require explicit viewing geometry or degrees-per-pixel |
| Jitter | seeded | seeded | Deterministic within implementation and seed recorded |
| Geometry failure | audit row | audit row | Branch retained as failed, never silently dropped |
| Model failure | failure table | failure table | Callback error retained |
| Non-convergence | explicit flag | explicit flag | Not treated as a valid converged estimate |
| Stability frequency | descriptive | descriptive | Never interpreted as probability the conclusion is true |

## Frozen cross-language fixture

Both packages ship the same fixture for a two-AOI example. It checks exact baseline labels, dilation labels, outside membership, overlap ambiguity, missing-coordinate preservation, unchanged-assignment proportion, and newly-assigned proportion.

```text
observation_id,x,y,baseline,dilate_1_px
1,1,1,a,a
2,11,5,__outside__,__ambiguous__
3,13,5,b,b
4,10.5,5,__outside__,a
5,,,,
```

## Expected backend differences

Some implementation details are intentionally language-native:

- R uses base-R graphics; Python returns Matplotlib axes.
- Seeded jitter is reproducible **within each implementation**, but R and NumPy RNG streams are not required to generate identical coordinates from the same integer seed.
- Statistical callbacks are user supplied, so exact coefficient values depend on the estimator/backend chosen by the analyst.
- Floating-point results can differ at machine precision even when geometry semantics agree.

These differences must not alter labels, units, failure semantics, or interpretation rules.

## Reporting parity

Methods sections should describe the analytical contract—not package-specific implementation quirks. Report the nominal AOIs, perturbation grid, coordinate/viewing geometry, overlap policy, assignment level, recomputed features, model specification, convergence/failure handling, and sensitivity summaries in the same way regardless of language.

Return to the [AOI uncertainty handbook](index.md).