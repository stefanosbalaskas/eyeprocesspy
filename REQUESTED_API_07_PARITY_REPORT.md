# Requested API + semantic validation 0.7 parity report

Reference: frozen R `eyeprocess` 0.11.1.

## Scope

This tranche accounts for **37 frozen exports**:

- `R/048-semantic-validation-0-7.R`: **15 exports**
- `R/055-requested-api-completion-0-7.R`: **22 exports**
- four direct exported plot functions from R/055 and two S3 semantic plot counterparts are implemented and plot-data tested.

## Implemented capabilities

- semantic evidence ladder and semantic-fidelity specification;
- field, timestamp, coordinate, pupil-unit, eye-stream, event and HED semantic audits;
- BIDS eye-tracking structural semantic checks;
- semantic round-trip aggregation/loss maps and detailed compatibility-evidence matrices;
- vendor timestamp semantics and public validation-corpus registry;
- vendor schema contracts and versioned adapter-regression audits;
- callback-based BIDS round trips;
- gaze-informed two-part missingness/response diagnostic with explicit theta-proxy labelling;
- named device/session/algorithm facet extraction;
- process changepoint alias/plot, latent person-item explanations, and flexible-IRF uncertainty plot;
- latent-distribution audit/comparison/stress-test alias;
- theta-conditioned event-time IRT reference diagnostic;
- registered simulation/truth extraction and one-replicate recovery harness.

## Validation

- 37/37 frozen exports resolve to callables.
- 9 focused semantic/completion tests pass.
- Full package suite at tranche freeze: **94/94 tests pass**.
- Two new executable examples run in the global IRT example smoke suite.
- `validation-evidence-0-7` article parity is complete; the existing process-IRT atlas was updated with the completion layer.

## Algorithmic boundary

`fit_event_time_irt()` uses a dependency-light Cox partial-likelihood reference but does not reproduce R `survival::coxph(..., robust=TRUE, cluster=person)` robust clustered standard errors, so it is explicitly marked `python_reference_differs`. `plot_irf_uncertainty()` uses the fitted Python flexible-IRF reference and only propagates link-scale covariance when that covariance is available; it is also marked `python_reference_differs`. No cross-language numerical parity is claimed without the R oracle.
