# IRT diagnostics and measurement plots

`eyeprocesspy` includes an extensive IRT and process-psychometrics surface rather than treating eye-tracking features only as predictors. This worked page focuses on diagnostic visualization: information, item fit and DIF.

<div class="ep-gallery" markdown>

<figure>
  <img src="../../assets/gallery/irt-information.svg" alt="IRT information curve">
  <figcaption><strong>Information profile.</strong> Test information across the latent continuum.</figcaption>
</figure>

<figure>
  <img src="../../assets/gallery/irt-item-fit.svg" alt="IRT item fit">
  <figcaption><strong>Item fit.</strong> Item-level fit statistic relative to the reference value.</figcaption>
</figure>

<figure>
  <img src="../../assets/gallery/irt-dif.svg" alt="IRT DIF curve">
  <figcaption><strong>DIF curve.</strong> Signed focal-reference probability difference across theta.</figcaption>
</figure>

</div>

The executable example is [`examples/irt_diagnostics.py`](https://github.com/stefanosbalaskas/eyeprocesspy/blob/release/0.1.0-deep-parity/examples/irt_diagnostics.py).

## Information and conditional SEM

```python
ax = ep.plot_eye_irt_information_profile(information_profile)
ax_sem = ep.plot_eye_irt_information_profile(
    information_profile,
    show_sem=True,
)
```

Information is conditional on theta; a single global reliability number cannot substitute for an information profile when precision varies substantially across the latent continuum.

## Item fit

```python
ax = ep.plot_eye_irt_item_fit(item_fit, statistic="infit")
```

Use fit statistics as diagnostics rather than automatic item-deletion rules. Investigate content, local dependence, dimensionality and data quality before changing an assessment model.

## Differential item functioning

```python
ax = ep.plot_eye_irt_dif_curve(dif_curve)
```

DIF is a measurement-invariance diagnostic. Statistical DIF is not itself proof of unfairness; substantive interpretation requires the grouping variable, item content, model specification and potential impact on scores.

## Wider IRT surface

The package also exposes score uncertainty, Q3/local-dependence diagnostics, adaptive traces, link stability, DTF, recovery and SBC evidence, bank coverage, process alignment, sparse-design audits, prior sensitivity and advanced process-informed models. See the [IRT and psychometrics guide](../guides/psychometrics-irt.md) and the [API reference](../reference/index.md).
