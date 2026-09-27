# Robustness, sensitivity, and calibration

A Bayesian-network analysis is not complete when one graph has been estimated. eyeprocesspy now exposes complementary robustness analyses for parameters, structure, preprocessing, measurement error, sample size, and predictive probabilities.

## CPT sensitivity

~~~python
from eyeprocesspy.bayesian_networks import cpt_sensitivity_analysis

cpt = cpt_sensitivity_analysis(
    fitted,
    node="trust",
    state="High",
    values=(0.20, 0.40, 0.60, 0.80),
    parent_configuration={"gaze": "High"},
    target="choice",
    evidence={"condition": "Explanation"},
)
~~~

Only the selected CPT column is changed. Remaining state probabilities are rescaled proportionally. The fitted model supplied by the analyst is never modified in place.

## Structural perturbation

~~~python
from eyeprocesspy.bayesian_networks import structural_perturbation_sensitivity

structural = structural_perturbation_sensitivity(
    fitted,
    target="choice",
    evidence={"condition": "Explanation"},
    include_delete=True,
    include_reverse=True,
    add_edges=[("pupil", "choice")],
)
~~~

Each admissible add/delete/reverse branch is refitted on the same data. Cyclic perturbations and fitting failures remain in provenance.

## Discretization sensitivity

~~~python
disc = discretization_sensitivity(
    continuous_spec,
    columns=["gaze", "pupil"],
    schemes={
        "tertiles": {"method": "quantile", "bins": 3},
        "equal_width": {"method": "width", "bins": 3},
        "declared_cuts": {
            "method": "cuts",
            "bins": [-10, -0.5, 0.5, 10],
        },
    },
)
~~~

No default discretization is silently selected. Missing values are not imputed.

## Measurement-noise sensitivity

~~~python
noise = measurement_noise_sensitivity(
    continuous_spec,
    columns=["gaze", "pupil"],
    noise_scales=(0, 0.05, 0.10, 0.20),
    repeats=10,
)
~~~

Noise scales are expressed relative to each selected variable's observed standard deviation. This is a perturbation analysis, not an estimate of instrument error unless those scales are externally justified.

## Sample-size stability curves

~~~python
curve = sample_size_stability_curve(
    bn_data,
    fractions=(0.25, 0.50, 0.75, 1.0),
    repeats=25,
    resample_by="participant_id",
)
~~~

For repeated-trial designs, use resample_by to preserve participant clustering. The curve compares each learned skeleton with the full-data learned skeleton using Jaccard agreement.

## Predictive calibration

~~~python
cal = predictive_calibration(
    fitted,
    target="choice",
    positive_state="Accept",
    n_bins=10,
)
~~~

The result contains row-level predicted probabilities, a reliability table, multiclass Brier score, and log loss.

!!! warning "Validation design"
    Calibration on the same observations used to fit the network is descriptive, not out-of-sample validation. Use held-out or grouped data when making predictive-generalization claims.

A robustness battery can combine algorithm/score comparisons, participant-level bootstrap stability, equivalence-class awareness, discretization sensitivity, measurement-noise perturbation, sample-size stability, CPT sensitivity, structural perturbation, grouped cross-validation, and probability calibration. Agreement across these axes supports robustness to the declared analysis choices; it does not prove causality or rule out latent confounding.
