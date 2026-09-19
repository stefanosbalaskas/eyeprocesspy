# Event-detector multiverse and inference robustness

Event detection is not only a preprocessing detail. A detector turns a continuous gaze stream into a discrete event catalogue; that catalogue then determines AOI membership, dwell metrics, scanpaths, latency variables, and sometimes the statistical conclusion itself.

`eyeprocesspy` therefore treats detector choice as an explicit analytical specification:

```text
raw gaze
  -> detector specification
  -> event catalogue
  -> AOI assignment
  -> derived features
  -> statistical model
  -> robustness summary
```

The purpose of a detector multiverse is **not** to maximize the number of analyses or count how often `p < .05`. The scientific question is whether a conclusion that matters substantively is stable across a defensible set of event-detection choices.

## When to use

Use this workflow when event detection is upstream of substantive outcomes such as AOI dwell, fixation count, TTFF, revisits, transitions, scanpaths, or fixation-locked pupil summaries; when plausible detector thresholds are not uniquely determined; when comparing vendor events with research algorithms; or when reviewers need evidence that a result is not an artifact of one detector specification.

## When not to use

Do not build a large arbitrary grid merely because the software can do so. A multiverse is not a substitute for data-quality checks, calibration assessment, appropriate coordinate conversion, a justified primary detector, or external validation against human labels where such labels exist. For head-mounted data with substantial scene/head motion, simple screen-coordinate I-VT/I-DT specifications may be scientifically inappropriate without motion compensation.

## Safeguards

The detector-multiverse API follows several non-negotiable rules:

- sampling rate and threshold units are explicit;
- REMoDNaV is bridged when requested; a missing dependency never causes silent substitution;
- external detector callbacks are validated before their events enter the canonical schema;
- AOI overlap defaults to `overlap="error"`, so ambiguous assignments are not silently resolved;
- zero-event trials remain represented in the feature table;
- trials with no valid gaze are not converted to zero counts or zero dwell;
- inference requires an explicit statistical engine or callback;
- formula engines use missing-data failure rather than silently dropping model rows;
- non-converged branches are retained diagnostically and excluded from stability summaries;
- event, AOI, feature, and model outputs retain hashes for source data, preprocessing/provenance, detector, AOI, quality, and model specifications where relevant.

## Start here

1. [Why Event Detection Is an Analytical Choice](event-detection-analytical-choice.md)
2. [Building a Detector Multiverse](building-detector-multiverse.md)
3. [Comparing Event Catalogues](comparing-event-catalogues.md)
4. [Propagating Detector Choice to AOI Features](detector-aoi-features.md)
5. [Propagating Detector Choice to Statistical Inference](detector-statistical-inference.md)
6. [Reporting Detector Sensitivity](reporting-detector-sensitivity.md)
7. [Worked disclosure-dwell example](../examples/detector-multiverse-disclosure.md)
8. [API reference](../reference/detector-multiverse.md)

## Methodological grounding

Threshold-based I-VT and I-DT algorithms remain common, but detector performance depends on the signal, task, sampling characteristics, noise, and event types present. Recent methodological guidance emphasizes that detector choice should be aligned with the signal and research question rather than treated as universally interchangeable. REMoDNaV extends adaptive velocity-based classification for natural viewing and detects saccades, post-saccadic oscillations, fixations, and pursuit. Event-matching work also motivates temporal-overlap matching rather than comparing event rows by position.

References include Dar, Wagner, & Hanke (2021), *Behavior Research Methods*, DOI `10.3758/s13428-020-01428-x`; Drews & Dierkes (2024), *Behavior Research Methods*, DOI `10.3758/s13428-024-02360-0`; and the operationalization guidance in Hooge et al. (2025), *Behavior Research Methods*, DOI `10.3758/s13428-024-02590-2`.


## R/Python contract parity

The R and Python implementations share the same scientific concepts, detector-spec fields, failure semantics, event-matching fixture, feature names, and tidy inference outputs. The shared fixture is exercised in both test suites.

Backend names differ where the host ecosystems genuinely differ:

| Scientific role | Python | R |
| --- | --- | --- |
| ordinary linear model | `statsmodels_ols` | `stats_lm` |
| mixed model | `statsmodels_mixedlm` | `lme4_lmer` |
| specialist estimator | `callback` | `callback` |
| REMoDNaV integration | installed Python REMoDNaV package | installed REMoDNaV CLI |

These are implementation differences, not invitations to change the scientific model between branches or languages. Numerical identity is not required when backends differ; detector definitions, retained observations, model meaning, convergence status, provenance, and semantic output fields are required to align.

## Interpretation

Treat the multiverse as a **sensitivity analysis over defensible measurement choices**, not as a search for the detector that produces the preferred inferential result. Event-count disagreement shows that segmentation changes; pairwise temporal agreement describes catalogue similarity; neither quantity identifies a ground-truth detector without an external reference.

At the inference level, inspect the coefficient distribution, confidence intervals, sign stability, substantive-threshold stability, and the planned-specification convergence rate together. A stable coefficient across the declared branches supports robustness **conditional on that declared detector set**. It does not establish robustness to omitted preprocessing, AOI geometry, quality rules, outcome definitions, or model specifications.

## Limitations

- The multiverse is only as informative as the scientific justification for the included detector families and parameter ranges.
- Detector agreement is not accuracy unless one catalogue is independently justified as a reference standard.
- External backends such as REMoDNaV remain external dependencies; an unavailable backend is recorded as a failed branch rather than replaced by a surrogate.
- Coordinate units and sampling rate remain part of the detector contract. Pixel/normalized coordinates must not be silently treated as degrees.
- Non-convergence, missing requested coefficients, and model-stage failures reduce the planned-specification convergence rate and remain part of the scientific record.
- Stability of a detector-sensitive feature does not by itself establish construct validity or causal interpretation.

## Reporting and API links

Use [Reporting detector sensitivity](reporting-detector-sensitivity.md) for the minimum reporting set, denominator discipline, manuscript wording, and anti-patterns. The [worked disclosure example](../examples/detector-multiverse-disclosure.md) shows event-, feature-, and inference-level propagation with an explicit external-backend failure. Exact signatures and return contracts are collected in the [Detector multiverse API reference](../reference/detector-multiverse.md).

