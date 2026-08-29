import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import eyeprocesspy as ep


def trial_data(n=80,seed=1):
    r=np.random.default_rng(seed)
    return pd.DataFrame({'person_id':np.repeat([f'P{i}' for i in range(8)],10)[:n], 'item_id':np.tile([f'I{i}' for i in range(10)],8)[:n], 'person':np.repeat([f'P{i}' for i in range(8)],10)[:n], 'item':np.tile([f'I{i}' for i in range(10)],8)[:n], 'session':np.tile(np.repeat([1,2],5),8)[:n], 'device':np.tile(['A','B'],40)[:n], 'dwell_ms':r.normal(1000,120,n), 'pupil':r.normal(.1,.03,n), 'observed':r.binomial(1,.8,n)})

def gaze_data(n=100,seed=2):
    r=np.random.default_rng(seed);return pd.DataFrame({'x':np.cumsum(r.normal(0,.03,n))+.5,'y':np.cumsum(r.normal(0,.03,n))+.5,'time':np.arange(n),'pupil':r.normal(1,.1,n)})

def test_uncertainty_recalibration_gstudy():
    r=np.random.default_rng(1);d=pd.DataFrame({'dwell':r.normal(1000,100,50),'pupil':r.normal(.1,.03,50)})
    spec=ep.process_uncertainty_spec(source_sd={'calibration':3,'preprocessing':2},draws=30);fit=ep.estimate_process_uncertainty(d,spec,metrics=['dwell','pupil']);b=ep.uncertainty_budget(fit);assert abs(b.loc[b.metric=='dwell','variance_share'].sum()-1)<1e-8
    p=ep.propagate_process_uncertainty(fit,lambda x:x.dwell.mean(),draws=20);assert int(p.summary.draws.iloc[0])==20;assert ep.compare_uncertainty_budgets(first=fit,second=fit).eyeprocess_class=='eye_uncertainty_budget_comparison';assert ep.plot_uncertainty_waterfall(fit)
    ref=pd.DataFrame({'target_x':r.random(60),'target_y':r.random(60)});obs=ref.assign(x=lambda z:z.target_x+.05+r.normal(0,.005,60),y=lambda z:z.target_y-.03+r.normal(0,.005,60),time=np.arange(60))
    drift=ep.detect_calibration_drift(obs,window=10,x_col='x',y_col='y',time_col='time');m=ep.fit_offline_recalibration(drift,method='translation');corr=ep.apply_offline_recalibration(obs,m,x_col='x',y_col='y');a=ep.audit_recalibration(obs,corr);assert a.summary.after_rmse.iloc[0]<a.summary.before_rmse.iloc[0];assert ep.plot_calibration_vector_field(drift);assert ep.plot_recalibration_before_after(a)
    td=trial_data();gs=ep.fit_process_gstudy(td,'dwell_ms',facets=['person','item','session','device']);ds=ep.design_process_dstudy(gs,items=[5,10],sessions=[1,2]);assert ds.design_grid.absolute_dependability.between(0,1).all();aud=ep.audit_process_reliability(td,'dwell_ms');assert aud.eyeprocess_class=='eye_process_reliability_audit';assert ep.plot_dependability_surface(ds)

def test_pupil_registration_missingness():
    r=np.random.default_rng(1);rows=[]
    for i in range(12):
        t=np.linspace(0,2,50);rows.append(pd.DataFrame({'person_id':f'P{i+1}','time':t,'pupil':np.exp(-((t-(.8+(i+1)/100))**2)/.08)+r.normal(0,.02,50)}))
    d=pd.concat(rows,ignore_index=True);reg=ep.register_pupil_curves(d,'time','pupil');dec=ep.decompose_pupil_phase_amplitude(reg,components=2);assert {'phase_shift','amplitude_pc1'}.issubset(dec.scores.columns);resp=pd.DataFrame(r.binomial(1,.6,(12,5)),index=[f'P{i+1}' for i in range(12)]);fit=ep.fit_phase_amplitude_irt(resp,dec);assert fit.eyeprocess_class=='eye_phase_amplitude_irt';assert ep.plot_pupil_registration(reg)
    td=trial_data();obs=ep.fit_process_observation_model(td,'observed',['dwell_ms','pupil']);joint=ep.fit_joint_signal_missingness('pupil',obs,x=td,predictors=['dwell_ms']);assert joint.eyeprocess_class=='eye_joint_signal_missingness';vals=td.pupil.to_numpy().copy();vals[td.observed.eq(0)]=np.nan;s=ep.process_pattern_mixture(vals,delta=[-1,0,1]);assert len(s.table)==3;tip=ep.sensitivity_mnar_process(s);assert ep.plot_mnar_tipping_point(tip)

def test_recurrence_point_process():
    g=gaze_data(100);rec=ep.gaze_recurrence(g);assert 0<=rec.summary.recurrence_rate.iloc[0]<=1;cross=ep.cross_recurrence(g.x,g.pupil);assert cross.eyeprocess_class=='eye_cross_recurrence';win=ep.windowed_recurrence(rec,20,10);assert len(win.summary)>=4;assert ep.plot_recurrence_matrix(rec)
    x=np.arange(30)[:,None];y=np.sin(np.arange(30)/4)[:,None];cr=ep.cross_recurrence(x,y);assert cr.matrix.shape==(30,30)
    g['duration']=np.random.default_rng(3).exponential(200,100);fit=ep.fit_fixation_point_process(g,interaction='self_exciting',grid_size=8);pred=ep.predict_fixation_intensity(fit);assert (pred.predicted_intensity>=0).all();mark=ep.fit_marked_gaze_process(g,marks=['duration','pupil']);assert mark.eyeprocess_class=='eye_marked_gaze_process';diag=ep.diagnose_gaze_point_process(fit);assert diag.eyeprocess_class=='eye_gaze_point_process_diagnostics';assert ep.plot_fixation_intensity(fit)
    const=pd.DataFrame({'x':.5,'y':.5,'time':np.arange(30),'salience':np.linspace(0,1,30)});f2=ep.fit_fixation_point_process(const,spatial_covariates=['salience'],grid_size=5);assert 'salience' in f2.covariate_map.source.values;assert np.isfinite(ep.predict_fixation_intensity(f2).predicted_intensity).all()

def test_scanpath_episodes_evidence_adapters():
    paths={'A':['stem','evidence','options'],'B':['stem','options','evidence'],'C':['stem','evidence','options','evidence'],'D':['options','stem','evidence']};rep=ep.representative_scanpath(paths,method='consensus',distance='edit');assert len(ep.scanpath_dispersion(rep))==4;comp=ep.compare_scanpath_distributions(paths,['G1','G1','G2','G2'],distance='edit',permutations=19);assert 0<=comp.p_value<=1;boot=ep.bootstrap_representative_scanpath(rep,draws=10);assert boot.eyeprocess_class=='eye_scanpath_bootstrap';assert ep.plot_representative_scanpath(rep)
    r=np.random.default_rng(1);d=pd.DataFrame({'time':np.arange(120),'pupil':np.r_[r.normal(0,1,40),r.normal(2,1,40),r.normal(-1,1,40)],'gaze_velocity':np.r_[r.normal(1,1,40),r.normal(3,1,40),r.normal(.5,1,40)]});ch=ep.detect_process_changepoints(d,['pupil','gaze_velocity'],time_col='time',window=8);episodes=ep.label_process_episodes(ep.segment_process_episodes(ch));assert 'episode_label' in episodes.summary;cmp=ep.compare_episode_structure(episodes,np.resize(['A','B'],len(episodes.data)));assert cmp.eyeprocess_class=='eye_episode_comparison';assert ep.plot_process_episodes(episodes);assert len(ep.detect_process_changepoints(pd.DataFrame({'signal':np.r_[np.zeros(10),np.ones(11)]}),channels=['signal'],window=10).score)==21
    graph=ep.build_evidence_graph(raw_data=['pupil_samples','gaze_samples'],transformations=['blink_removal','baseline_correction'],metrics=['pupil_auc','aoi_entropy'],models=['functional_pupil_model'],diagnostics=['high_effort_evidence'],decisions=['item_I01_revise']);trace=ep.trace_item_decision(graph,'I01');assert ep.audit_evidence_dependencies(graph).summary.passed.iloc[0];assert ep.compare_decision_provenance(graph,graph).summary['count'].sum()==0;assert ep.plot_evidence_graph(graph)
    td=trial_data();miss=ep.fit_process_missingness_model(td,'observed',['dwell_ms','pupil']);assert miss.eyeprocess_class=='eye_process_observation_model';cm=ep.crossmodal_recurrence_model(np.arange(30),np.sin(np.arange(30)/5));assert 'recurrence_rate' in cm.features.columns;assert ep.plot_crossmodal_recurrence_model(cm,type='crossmodal')

def test_reconstructed_plot_surface_is_public():
    import csv
    from pathlib import Path
    with (Path(__file__).parents[1]/'parity'/'PLOT_PARITY.csv').open(newline='',encoding='utf-8') as fh:
        rows=list(csv.DictReader(fh))
    prefixes=tuple(f'R/{i:03d}-' for i in [33,34,35,37,38,39,40,41,42,46,47])
    expected=[r['python_function'] for r in rows if r['source_file'].startswith(prefixes)]
    missing=[name for name in expected if not callable(getattr(ep,name,None))]
    assert missing==[]
