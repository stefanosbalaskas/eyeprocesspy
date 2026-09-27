# Reporting Bayesian-network analyses

`report_bayesian_network()` generates a Markdown record that keeps modeling decisions and interpretation boundaries together.

~~~python
from eyeprocesspy.bayesian_networks import report_bayesian_network

report = report_bayesian_network(
    fitted,
    stability=stability,
    validation=cv,
    sensitivity=detector_comparison,
    path="bn-report.md",
)
~~~

The report includes:

- observation level, rows, node count, and model family;
- backend, structure algorithm, score, edges, and parameter estimator;
- bootstrap replications, successful/failed fits, edge/direction strengths;
- grouped validation evidence;
- detector/AOI sensitivity evidence when supplied;
- source/model provenance and warnings;
- the explicit non-causal interpretation boundary.

## Suggested methods wording

> A Bayesian network was fitted to prespecified trial-level multimodal features after gaze, pupil, and physiological preprocessing. Temporal/experimental constraints prevented edges that contradicted known measurement order. Structure stability was evaluated by participant-level bootstrap resampling, and predictive fit was assessed using participant-grouped cross-validation. Directed learned edges were interpreted as probabilistic graph orientations under the specified assumptions rather than as causal effects.

Adapt the statement to the actual algorithm, model family, preprocessing choices, and validation results.
