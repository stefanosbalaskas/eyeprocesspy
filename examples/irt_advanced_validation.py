"""Executable advanced-model validation design example."""
import eyeprocesspy as ep

grid=ep.advanced_validation_grid(quick=True)
scenario=grid.iloc[0].to_dict()
sim=ep.simulate_advanced_process_data(**scenario,seed=20260804)
assert len(grid)>1
assert len(sim.trials)==int(scenario["n_person"]*scenario["n_item"])
assert {"trials","states","pupil","truth"}<=set(sim)
