# AOI Geometry Is an Analytical Assumption

AOI boundaries encode a substantive decision about where a meaningful visual region begins and ends. Even with a fixed stimulus, that decision can be uncertain.

## When to use this perspective

Treat geometry as a sensitivity dimension when fixations cluster near borders, AOIs are small relative to spatial error, neighboring regions nearly touch, or more than one boundary coding is scientifically defensible.

## When not to use perturbation

Do not use AOI perturbation to repair poor calibration, search for significance, or replace a preregistered primary AOI after seeing the result. The nominal geometry remains the reference; perturbations test dependence on it.

## Boundary ambiguity

`eyeprocesspy` does not silently choose the first AOI when regions overlap. The default assignment is `__ambiguous__`, making geometry conflicts visible in downstream summaries.

## Practical design rule

Define a perturbation envelope from the display, acquisition quality, and stimulus design. A small set such as ±0.25°, ±0.50°, and ±1.00° is easier to defend than a large result-driven search grid.

Continue to [AOI Perturbation Sensitivity Analysis](perturbation-sensitivity.md).