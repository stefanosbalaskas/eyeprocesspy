# Worked example: AOI perturbation failure clinic

This fully synthetic example is intentionally designed to **fail safely**. It demonstrates geometry collapse, overlap-induced ambiguity, model non-convergence, callback failure, and missing coordinates without private data.

The executable source is `examples/aoi_perturbation_failure_clinic.py`.

## Perturbation plan

Two AOIs are separated by a one-pixel gap. The grid contains the baseline, +1 px dilation, −6 px erosion, and +1 px horizontal translation.

```python
grid = ep.create_aoi_perturbation_grid(
    dilations=[1.0],
    erosions=[6.0],
    translations_x=[1.0],
    include_baseline=True,
    unit="px",
)
```

The dilation creates overlap, the erosion collapses the smaller AOI, and the translation branch deliberately triggers a synthetic model-callback error.

## Local execution result

| perturbation | status | interpretation |
| --- | --- | --- |
| `baseline` | completed | nominal geometry valid |
| `dilate_1_px` | completed | overlap creates explicit ambiguity |
| `erode_6_px` | failed | erosion collapses the `claim` AOI |
| `shift_x_1_px` | completed | geometry valid; model callback later fails |

The failed erosion remains in the four-branch perturbation audit.

## Model audit

The callback produces a converged baseline row, an explicit non-converged dilation row, and an exception for the translation branch. The converged-effect summary therefore reports **1/2 model rows converged**. The callback failure is not converted into a model row.

## Generated report

```text
Planned perturbations: 4; completed geometry branches: 3;
geometry failures: 1; model callback failures: 1.
Median unchanged AOI assignment across completed perturbations: 0.750.
```

The report explicitly states that these are descriptive robustness summaries and **not probabilities that the scientific conclusion is true**.

## Files produced

The example writes audit, failure, model, assignment, Markdown-report, and SVG-geometry files under the requested output directory. The outputs are CI-small and synthetic.

## Interpretation

A failed branch changes what can be evaluated and must stay visible. Non-convergence is not evidence of a null effect, and overlap ambiguity is not permission to pick the first AOI.

Continue with [AOI perturbation troubleshooting](../guides/aoi-uncertainty/troubleshooting.md), [Reporting AOI Robustness](../guides/aoi-uncertainty/reporting.md), and the [API reference](../reference/aoi-perturbation.md).
