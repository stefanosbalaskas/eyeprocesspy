from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import eyeprocesspy as ep

EXPORTS=[
    'functional_pupil_irt_spec','fit_joint_functional_pupil_irt','advanced_validation_grid','simulate_advanced_process_data',
    'functional_pupil_basis','prepare_functional_pupil_data','fit_functional_pupil_stan','extract_functional_pupil_parameters',
    'functional_pupil_diagnostics','pupil_preprocessing_grid','pupil_preprocessing_sensitivity','compare_functional_scalar_models',
]


def _long_pupil(n_person=4,n_item=3,n_time=12):
    rows=[]
    for pi in range(n_person):
        for ji in range(n_item):
            score=int((pi+ji)%2)
            trial=f'P{pi+1}-I{ji+1}'
            for s in range(n_time):
                time=800+s*100
                rows.append({
                    'participant_id':f'P{pi+1}','item_id':f'I{ji+1}','trial_id':trial,
                    'time_ms':time,'event_time':1200,'pupil':3+0.02*s+0.08*score+0.02*pi,
                    'score':score,'response_time':1.0+.03*ji,
                })
    return pd.DataFrame(rows)


def test_exports_and_final_override_signatures():
    assert all(callable(getattr(ep,n,None)) for n in EXPORTS)
    sigs=json.loads((Path(__file__).resolve().parents[1]/'reference'/'R_SIGNATURES.json').read_text())
    for name in EXPORTS:
        expected=[a['name'] for a in sigs[name]['args']]
        expected=['kwargs' if n=='...' else n for n in expected]
        assert list(inspect.signature(getattr(ep,name)).parameters)==expected, name
    assert len(sigs['functional_pupil_irt_spec']['args'])==34
    assert sigs['functional_pupil_irt_spec']['source_file']=='R/026-functional-pupil-engine.R'


def test_spec_is_neutral_and_validation_matches_final_contract():
    spec=ep.functional_pupil_irt_spec(engine='two_stage_glm')
    assert spec.eyeprocess_class=='eye_functional_pupil_irt_spec'
    assert 'not automatic' in spec.interpretation
    with pytest.raises(ep.EyeProcessValidationError,match='event_time_column'):
        ep.functional_pupil_irt_spec(alignment='event')
    with pytest.raises(ep.EyeProcessValidationError,match='parallel_chains'):
        ep.functional_pupil_irt_spec(chains=4,parallel_chains=5)
    with pytest.raises(ep.EyeProcessValidationError,match='adapt_delta'):
        ep.functional_pupil_irt_spec(adapt_delta=1)


def test_basis_grid_and_advanced_validation_contracts():
    time=np.linspace(-.2,1.5,60)
    B=ep.functional_pupil_basis(time,df=6,degree=3)
    assert B.shape[0]==len(time) and B.shape[1]>=4 and np.isfinite(B.to_numpy()).all()
    grid=ep.pupil_preprocessing_grid(
        baseline_windows=((-200,0),(-100,0)),latency_ms=(100,200),basis_df=(4,6),
        baseline_methods=('subtract','percent'),max_interpolated_fraction=(.1,.2))
    assert len(grid)>=8 and len(grid.drop_duplicates())==len(grid)
    av=ep.advanced_validation_grid(quick=True)
    expected={'n_person','n_item','ability_speed_correlation','gaze_effect','feature_reliability','missing_process','state_misclassification','pupil_ar1','luminance_effect','dif_effect','local_dependence'}
    assert expected<=set(av) and 1<len(av)<30
    assert len(ep.advanced_validation_grid(quick=True,full_factorial=True))>len(av)


def test_event_alignment_baseline_uncertainty_and_two_stage_fit():
    d=_long_pupil(n_person=6,n_item=3,n_time=12)
    spec=ep.functional_pupil_irt_spec(
        df=3,alignment='event',event_time_column='event_time',latency_ms=0,
        baseline_window=(-400,0),min_baseline_samples=3,pupil_column='pupil',time_column='time_ms',engine='two_stage_glm')
    prepared=ep.prepare_functional_pupil_data(d,spec)
    assert prepared.eyeprocess_class=='eye_functional_pupil_data'
    assert {'baseline_n','baseline_se','baseline_valid'}<=set(prepared.data)
    assert prepared.data.baseline_valid.all()
    assert (prepared.data['.time']<0).any() and (prepared.data['.time']>0).any()
    fit=ep.fit_joint_functional_pupil_irt(d,spec,seed=1)
    assert fit.eyeprocess_class=='eye_functional_pupil_irt'
    assert len(fit.feature_names)>0
    pars=ep.extract_functional_pupil_parameters(fit)
    assert {'parameter','estimate','std_error','lower','upper'}<=set(pars)
    diag=ep.functional_pupil_diagnostics(fit)
    assert diag.eyeprocess_class=='eye_functional_pupil_diagnostics'
    assert {'check','pass'}<=set(diag.checks)


def test_functional_plots_and_sensitivity_prepare_only():
    d=_long_pupil(n_person=5,n_item=3,n_time=12)
    spec=ep.functional_pupil_irt_spec(df=3,alignment='event',event_time_column='event_time',latency_ms=0,baseline_window=(-400,0),pupil_column='pupil',time_column='time_ms',engine='two_stage_glm')
    fit=ep.fit_joint_functional_pupil_irt(d,spec)
    ax=ep.plot_eye_functional_pupil_irt(fit,type='coefficients')
    try: assert hasattr(ax,'gp3_data') and len(ax.gp3_data)>0
    finally: plt.close(ax.figure)
    diag=ep.functional_pupil_diagnostics(fit)
    ax=ep.plot_eye_functional_pupil_diagnostics(diag)
    try: assert hasattr(ax,'gp3_data')
    finally: plt.close(ax.figure)
    grid=ep.pupil_preprocessing_grid(baseline_windows=((-400,0),),latency_ms=(0,),basis_df=(3,4),baseline_methods=('subtract',),max_interpolated_fraction=(.2,))
    sens=ep.pupil_preprocessing_sensitivity(d,grid=grid,base_spec=spec,fit=False)
    assert len(sens.grid)==2 and len(sens.results)==2
    ax=ep.plot_eye_functional_pupil_sensitivity(sens)
    try: assert hasattr(ax,'gp3_data') and len(ax.gp3_data)==2
    finally: plt.close(ax.figure)


def test_advanced_simulator_layers_and_guards():
    sim=ep.simulate_advanced_process_data(n_person=20,n_item=8,n_time=12,n_states=4,ability_speed_correlation=.35,gaze_effect=.4,feature_reliability=.65,missing_process=.25,state_misclassification=.30,pupil_ar1=.5,luminance_effect=.4,dif_effect=.3,local_dependence=.2,seed=205)
    assert {'trials','states','pupil','truth'}<=set(sim)
    assert len(sim.trials)==160 and sim.trials.gaze_1.isna().any()
    assert (sim.states.state!=sim.states.true_state).any()
    assert {'score','response_time','gaze_1','gaze_2','strategy','group','dif_item','testlet'}<=set(sim.trials)
    assert {'ability_speed_correlation','gaze_effect','feature_reliability','pupil_ar1'}<=set(sim.truth)
    with pytest.raises(ep.EyeProcessValidationError,match='outside their valid ranges'):
        ep.simulate_advanced_process_data(feature_reliability=0)


def test_stan_engine_is_explicit_optional_backend():
    d=_long_pupil(n_person=4,n_item=3,n_time=12)
    spec=ep.functional_pupil_irt_spec(df=3,alignment='event',event_time_column='event_time',latency_ms=0,baseline_window=(-400,0),pupil_column='pupil',time_column='time_ms',engine='stan',chains=1,parallel_chains=1,iter_warmup=2,iter_sampling=2)
    prepared=ep.prepare_functional_pupil_data(d,spec)
    try:
        import cmdstanpy  # noqa: F401
    except Exception:
        with pytest.raises(ep.EyeProcessBackendError): ep.fit_functional_pupil_stan(prepared)
