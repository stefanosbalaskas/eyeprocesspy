# Bayesian-network API

The Bayesian-network surface is a **Python-native extension** to `eyeprocesspy`; it is not part of the frozen R parity ledger.

## Data contracts

::: eyeprocesspy.bayesian_networks.schema.BayesianNodeSpec

::: eyeprocesspy.bayesian_networks.schema.BayesianDataSpec

::: eyeprocesspy.bayesian_networks.schema.BayesianConstraintSpec

::: eyeprocesspy.bayesian_networks.schema.BayesianNetworkResult

::: eyeprocesspy.bayesian_networks.schema.BayesianNetworkQueryResult

::: eyeprocesspy.bayesian_networks.schema.BayesianNetworkStabilityResult

::: eyeprocesspy.bayesian_networks.schema.BayesianNetworkValidationResult

## Preparation and constraints

::: eyeprocesspy.bayesian_networks.prepare.define_bn_nodes

::: eyeprocesspy.bayesian_networks.prepare.prepare_bayesian_network_data

::: eyeprocesspy.bayesian_networks.prepare.validate_bayesian_network_data

::: eyeprocesspy.bayesian_networks.constraints.define_temporal_tiers

::: eyeprocesspy.bayesian_networks.constraints.define_bn_constraints

## Learning, fitting, inference

::: eyeprocesspy.bayesian_networks.learning.learn_bayesian_network

::: eyeprocesspy.bayesian_networks.fitting.fit_bayesian_network

::: eyeprocesspy.bayesian_networks.inference.query_bayesian_network

::: eyeprocesspy.bayesian_networks.inference.predict_bayesian_network

## Stability and validation

::: eyeprocesspy.bayesian_networks.stability.bootstrap_bn_structure

::: eyeprocesspy.bayesian_networks.validation.validate_bayesian_network

::: eyeprocesspy.bayesian_networks.validation.validate_bayesian_network_recovery

::: eyeprocesspy.bayesian_networks.validation.simulate_multimodal_bayesian_network_example

## Sensitivity

::: eyeprocesspy.bayesian_networks.sensitivity.compare_bayesian_networks

::: eyeprocesspy.bayesian_networks.sensitivity.compare_bn_across_detectors

::: eyeprocesspy.bayesian_networks.sensitivity.compare_bn_across_aoi_specs

## Plots and reports

::: eyeprocesspy.bayesian_networks.plotting.plot_bayesian_network

::: eyeprocesspy.bayesian_networks.plotting.plot_bn_edge_stability

::: eyeprocesspy.bayesian_networks.plotting.plot_bn_direction_stability

::: eyeprocesspy.bayesian_networks.plotting.plot_bn_posterior

::: eyeprocesspy.bayesian_networks.plotting.plot_bn_detector_robustness

::: eyeprocesspy.bayesian_networks.plotting.plot_bn_validation

::: eyeprocesspy.bayesian_networks.reporting.report_bayesian_network
