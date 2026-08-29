"""Legacy/core IRT and process-model API example."""
import eyeprocesspy as ep

x = ep.simulate_eye_dataset(n_person=12, n_item=5, sampling_rate=10, samples_per_trial=5, seed=12)
Y = ep.response_matrix(x)
RT = ep.response_time_matrix(x, log_transform=True)
assert Y.shape == RT.shape == (12, 5)
model = ep.fit_irt(x, engine="rasch_glm")
assert model.engine == "rasch_glm"
assert not ep.model_fit_statistics(model).empty
proc = ep.simulate_process_irt(n_person=12, n_item=5, seed=13)
assert len(proc.data) == 60
