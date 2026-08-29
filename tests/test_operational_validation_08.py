from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError, EyeProcessValidationError

EXPORTS = [
    'score_partial_response_pattern','score_response_stream','update_person_score','streaming_score_history',
    'collect_validation_evidence','validation_bundle_manifest','validation_report','write_validation_report','export_validation_bundle',
    'preaction_process_features','addm_glam_proxy_features','process_feature_family_registry','assign_process_feature_family',
    'process_feature_stability','plot_process_feature_stability',
]


def test_all_15_exports_are_public_callables():
    assert len(EXPORTS)==15
    assert [n for n in EXPORTS if not callable(getattr(ep,n,None))]==[]


def test_streaming_score_contract_is_gated_without_mirt():
    with pytest.raises(EyeProcessBackendError, match='mirt'):
        ep.score_partial_response_pattern(object(), [1,0,1,np.nan], method='MAP')
    s=ep.score_response_stream(object(), [1,0,1,1], observed_order=[1,2,3,4])
    assert s.eyeprocess_class=='eye_streaming_score'
    assert len(ep.streaming_score_history(s))==4
    assert s.history.theta.isna().all()
    with pytest.raises(EyeProcessValidationError):
        ep.score_response_stream(object(), [1,0], observed_order=[1,1])


def test_validation_evidence_bundle_produces_report_and_manifest(tmp_path: Path):
    b=ep.collect_validation_evidence(
        recovery=pd.DataFrame({'parameter':['a'],'bias':[.01]}),
        convergence=pd.DataFrame({'rate':[.99]}), model_name='demo_model')
    assert b.eyeprocess_class=='eye_validation_bundle'
    m=ep.validation_bundle_manifest(b)
    assert ((m.slot=='recovery') & (m.status=='available')).any()
    r=ep.validation_report(b,include_session=False)
    assert any('Convergence is not validation' in line for line in r)
    p=ep.write_validation_report(b,tmp_path/'report.txt',include_session=False); assert Path(p).exists()
    out=ep.export_validation_bundle(b,tmp_path/'bundle')
    assert out.eyeprocess_class=='eye_validation_export'
    assert Path(out.files['manifest']).exists() and Path(out.files['json_manifest']).exists()
    assert not any(Path(out.directory).glob('*.rds'))


def synthetic_process_data():
    rng=np.random.default_rng(12);rows=[]
    for p in ['P1','P2']:
        for tr in ['T1','T2']:
            for sample in range(1,51): rows.append((p,tr,sample))
    d=pd.DataFrame(rows,columns=['person_id','trial_id','sample'])
    d['time_ms']=d['sample']*20;d['response_time_ms']=1000
    d['aoi']=np.resize(['target','distractor','button','text'],len(d));d['pupil_bc']=rng.normal(size=len(d));d['blink']=False
    return d


def test_preaction_and_decision_proxy_features_generated():
    d=synthetic_process_data();p=ep.preaction_process_features(d,windows_ms=[500,1000]);q=ep.addm_glam_proxy_features(d)
    assert p.eyeprocess_class=='eye_preaction_process_features' and len(p.data)>0
    assert q.eyeprocess_class=='eye_decision_process_proxy' and len(q.features)>0
    assert 'target_minus_distractor_prop' in q.features


def test_feature_stability_assigns_conservative_families():
    d=pd.MultiIndex.from_product([['pupil_mean','valid_gaze','aoi_entropy'],range(1,5)],names=['feature','split']).to_frame(index=False)
    d['importance']=np.linspace(.1,.9,len(d))
    s=ep.process_feature_stability(d,top_n=2)
    assert {'feature_family','top_n_selection_rate'}.issubset(s.columns)
    fam=ep.assign_process_feature_family(['pupil_mean','valid_gaze','aoi_entropy'])
    assert list(fam)==['Pupil dynamics','Gaze dynamics','Scanpath organization']


def test_operational_plot_counterparts_have_data_layers():
    d=synthetic_process_data();p=ep.preaction_process_features(d,windows_ms=[500,1000]);q=ep.addm_glam_proxy_features(d)
    b=ep.collect_validation_evidence(recovery=pd.DataFrame({'bias':[.1]}),model_name='demo')
    s=ep.score_response_stream(object(),[1,0,1],observed_order=[1,2,3])
    stability=ep.process_feature_stability(pd.DataFrame({'feature':['pupil_mean','valid_gaze']*2,'split':[1,1,2,2],'importance':[.8,.4,.7,.5]}),top_n=1)
    for fun,obj in [(ep.plot_eye_preaction_process_features,p),(ep.plot_eye_decision_process_proxy,q),(ep.plot_eye_validation_bundle,b),(ep.plot_eye_streaming_score,s)]:
        ax=fun(obj);assert hasattr(ax,'eyeprocess_plot_data');plt.close(ax.figure)
    ax=ep.plot_process_feature_stability(stability,stability='top_n_selection_rate');assert len(ax.eyeprocess_plot_data)>0;plt.close(ax.figure)


def test_key_signature_names():
    expected={
        'score_response_stream':['model','response_pattern','observed_order','method','kwargs'],
        'update_person_score':['model','current_pattern','item_position','response','method','kwargs'],
        'preaction_process_features':['data','by','time','response_time','windows_ms','aoi','pupil','blink'],
        'addm_glam_proxy_features':['data','by','time','aoi','target_aoi','distractor_aoi','action_aoi'],
        'process_feature_stability':['data','feature','split','importance','top_n'],
        'plot_process_feature_stability':['data','feature','stability','top_n','kwargs'],
    }
    for name,args in expected.items(): assert list(inspect.signature(getattr(ep,name)).parameters)==args
