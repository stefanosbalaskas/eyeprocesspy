"""Provenance-aware Bayesian networks for validated multimodal features.

This subpackage is deliberately downstream of gaze preprocessing, event
detection, AOI assignment, pupil cleaning, and feature construction. It wraps
established probabilistic-graphical-model engines rather than reimplementing
structure learning or inference algorithms.

A learned directed edge is not, by itself, evidence of a causal effect.
"""

from .constraints import define_bn_constraints, define_temporal_tiers
from .fitting import fit_bayesian_network
from .inference import predict_bayesian_network, query_bayesian_network
from .learning import learn_bayesian_network
from .plotting import (
    plot_bayesian_network,
    plot_bn_detector_robustness,
    plot_bn_direction_stability,
    plot_bn_edge_stability,
    plot_bn_posterior,
    plot_bn_validation,
)
from .prepare import define_bn_nodes, prepare_bayesian_network_data, validate_bayesian_network_data
from .reporting import report_bayesian_network
from .schema import (
    BayesianConstraintSpec,
    BayesianDataSpec,
    BayesianDataValidationResult,
    BayesianNetworkComparisonResult,
    BayesianNetworkQueryResult,
    BayesianNetworkResult,
    BayesianNetworkStabilityResult,
    BayesianNetworkValidationResult,
    BayesianNodeSpec,
)
from .sensitivity import (
    compare_bayesian_networks,
    compare_bn_across_aoi_specs,
    compare_bn_across_detectors,
)
from .stability import bootstrap_bn_structure
from .validation import (
    simulate_multimodal_bayesian_network_example,
    validate_bayesian_network,
    validate_bayesian_network_recovery,
)

__all__ = [
    "BayesianConstraintSpec",
    "BayesianDataSpec",
    "BayesianDataValidationResult",
    "BayesianNetworkComparisonResult",
    "BayesianNetworkQueryResult",
    "BayesianNetworkResult",
    "BayesianNetworkStabilityResult",
    "BayesianNetworkValidationResult",
    "BayesianNodeSpec",
    "bootstrap_bn_structure",
    "compare_bayesian_networks",
    "compare_bn_across_aoi_specs",
    "compare_bn_across_detectors",
    "define_bn_constraints",
    "define_bn_nodes",
    "define_temporal_tiers",
    "fit_bayesian_network",
    "learn_bayesian_network",
    "plot_bayesian_network",
    "plot_bn_detector_robustness",
    "plot_bn_direction_stability",
    "plot_bn_edge_stability",
    "plot_bn_posterior",
    "plot_bn_validation",
    "predict_bayesian_network",
    "prepare_bayesian_network_data",
    "query_bayesian_network",
    "report_bayesian_network",
    "simulate_multimodal_bayesian_network_example",
    "validate_bayesian_network",
    "validate_bayesian_network_data",
    "validate_bayesian_network_recovery",
]


from .robustness import (
    cpt_sensitivity_analysis,
    discretization_sensitivity,
    measurement_noise_sensitivity,
    plot_bn_cpt_sensitivity,
    plot_bn_predictive_calibration,
    plot_bn_sample_size_stability,
    predictive_calibration,
    sample_size_stability_curve,
    structural_perturbation_sensitivity,
)

__all__.extend([
    "cpt_sensitivity_analysis",
    "discretization_sensitivity",
    "measurement_noise_sensitivity",
    "plot_bn_cpt_sensitivity",
    "plot_bn_predictive_calibration",
    "plot_bn_sample_size_stability",
    "predictive_calibration",
    "sample_size_stability_curve",
    "structural_perturbation_sensitivity",
])
