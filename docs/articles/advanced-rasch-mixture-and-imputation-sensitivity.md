# Advanced Rasch, mixture-IRT, and imputation sensitivity

This layer contains optional diagnostics and sensitivity adapters rather than replacements for the stable process-IRT core.

The exact frozen implementations of response-mixture IRT (`mirt`), nonparametric Rasch tests and stepwise item reduction (`eRm`), and process-informed Rasch trees (`psychotree`) remain explicit backend gates in Python. This prevents a superficially similar algorithm from being reported as parity.

`map_latent_classes_to_process_profiles()` is dependency-light and executable. It aggregates independent process features by person and class, but its caveat is substantive: class/process alignment **cannot prove** psychological strategy labels.

`biometric_imputation_sensitivity()` always reports missingness first. Without an exact requested imputation backend it does not silently fill values; completed biometric datasets remain sensitivity analyses rather than automatic replacements for the declared missingness strategy.
