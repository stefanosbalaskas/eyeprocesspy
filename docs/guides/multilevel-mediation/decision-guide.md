# Trial-level mediation decision guide

Use trial-level multilevel mediation when the scientific question concerns a **within-participant process** and the mediator is observed for repeated trials. Keep the trial as the observational unit and represent participant clustering explicitly.

## Choose the workflow

| Design question | Recommended action |
| --- | --- |
| X varies within participant and gaze varies across trials | Prepare `X_within`, `X_between`, `M_within`, and `M_between`; estimate the within indirect effect. |
| X is purely between participants | Do not describe a within indirect effect. Prepare with `require_within_x=False` only when the between-person design is intentional. |
| Every participant receives exactly the same treatment proportion | The between component of X may be constant. Do **not** estimate `a_between` or `cprime_between`; report them as not estimable. |
| Gaze is absent because tracking failed | Treat it as not observed, not zero. |
| Gaze was observed and dwell was genuinely 0 | Preserve zero and record the observed-zero state. |
| Poor-quality trial | Flag first. Mask only through an explicit quality rule; never silently discard. |
| Binary decision outcome | Use a Bernoulli/logit outcome model and report coefficient-product indirect effects on the linear-predictor scale. |
| Positive skewed dwell mediator | Consider lognormal or Gamma only when the observed support matches the family. |
| Proportion mediator in (0,1) | Beta can be appropriate; exact 0/1 values require a different model or explicit transformation justified by the design. |
| Serial chain is not temporally/design justified | Do not fit serial mediation merely because several variables are available. |

## Minimum evidence before fitting

1. Confirm participant/trial identifiers are unique.
2. Review within- and between-participant variance.
3. Review missingness, true zeros, and quality flags.
4. Confirm the causal/temporal ordering is defensible from the experiment.
5. Choose mediator/outcome likelihoods from their supports, not convenience.
6. Declare priors and the missingness policy before sampling.

The preparation layer never selects an estimator. `gp3bayes`/`gp3bayespy` owns Bayesian estimation; `eyeprocess`/`eyeprocesspy` owns decomposition and observation semantics.
