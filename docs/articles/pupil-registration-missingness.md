# Pupil registration and informative missingness

Pupil trajectories can differ in both timing and amplitude. `register_pupil_curves()` aligns individual curves on a normalized time grid without discarding the original observations. `decompose_pupil_phase_amplitude()` then separates phase shift from amplitude variation using a principal-component representation of registered curves.

`fit_phase_amplitude_irt()` links those person-level phase/amplitude summaries to an outcome using the dependency-light reference model. It is a process-measurement bridge, not a claim that pupil components directly measure cognitive load or another mental state.

Missing pupil/process observations can be informative. `fit_process_observation_model()` models observation probability, `fit_joint_signal_missingness()` provides a transparent joint approximation, and `process_pattern_mixture()` / `sensitivity_mnar_process()` expose delta-based MNAR sensitivity and tipping points.
