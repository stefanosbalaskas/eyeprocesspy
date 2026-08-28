import numpy as np
import pandas as pd
import pytest
import eyeprocesspy as ep

def test_schema_exact_table_order_and_count():
    assert ep.eye_schema()['version']=='0.1.0'
    assert ep.canonical_table_names()==['recordings','streams','gaze_samples','eye_samples','episodes','events','intervals','responses','coordinate_spaces','aoi_definitions','aoi_geometry','biometrics','calibrations','features','quality','provenance']
    assert ep.schema_table('recordings')[:4]==['recording_id','participant_id','session_id','vendor']

def test_standardize_eye_table_r_semantics():
    d=pd.DataFrame({'vendor':['Gazepoint'],'extra':[1]})
    x=ep.standardize_eye_table(d,'recordings')
    assert list(x.columns[:len(ep.schema_table('recordings'))])==ep.schema_table('recordings')
    assert x.columns[-1]=='extra'
    assert pd.isna(x.loc[0,'recording_id'])

def test_validate_eye_table_strict_extra_and_missing():
    d=pd.DataFrame({'recording_id':['r1'],'extra':[1]})
    issues=ep.validate_eye_table(d,'recordings',strict=True)
    assert 'missing_schema_field' in set(issues.code)
    assert ((issues.code=='extra_field') & (issues.field=='extra')).any()

def test_new_dataset_and_primary_key_validation():
    rec=pd.DataFrame({'recording_id':['r1','r1'],'participant_id':['p1','p1']})
    x=ep.new_eye_dataset(recordings=rec)
    assert ep.is_eye_dataset(x)
    assert 'duplicate_primary_key' in set(ep.validate_eye_dataset(x).code)

def test_orphan_recording_and_stop_contract():
    rec=pd.DataFrame({'recording_id':['r1']})
    gaze=pd.DataFrame({'recording_id':['r2'],'sample_id':['s1']})
    x=ep.new_eye_dataset(recordings=rec,gaze_samples=gaze)
    issues=ep.validate_eye_dataset(x)
    assert 'orphan_recording_id' in set(issues.code)
    with pytest.raises(ep.EyeProcessValidationError): ep.validate_eye_dataset(x,stop_on_error=True)

def test_coordinate_space_defaults_exact():
    s=ep.new_coordinate_space('display',space_type='display_pixels_top_left',width=1920,height=1080)
    assert s.loc[0,'origin']=='top_left' and s.loc[0,'x_unit']=='pixels'

def test_mapping_drops_none_and_merges_extra():
    m=ep.eye_mapping(participant='PARTICIPANT',x='X',extra={'custom':'C'})
    assert m=={'participant':'PARTICIPANT','x':'X','custom':'C'}

def test_sampling_rate_r_algorithm():
    assert ep.estimate_sampling_rate([0,.01,.02,.03])==pytest.approx(100)
    assert np.isnan(ep.estimate_sampling_rate([1]))

def test_clock_transform_offset_and_linear():
    o=ep.estimate_clock_transform([0,1],[2,3],method='offset')
    assert o.offset==pytest.approx(2) and o.slope==1
    l=ep.estimate_clock_transform([0,1,2],[1,3,5],method='linear')
    assert l.offset==pytest.approx(1) and l.slope==pytest.approx(2)

def test_normalize_timebase_per_recording():
    g=pd.DataFrame({'recording_id':['r1','r1','r2','r2'],'sample_id':['a','b','c','d'],'timestamp_native':[1000,1010,50,60]})
    r=pd.DataFrame({'recording_id':['r1','r2']})
    x=ep.new_eye_dataset(recordings=r,gaze_samples=g,validate=False)
    y=ep.normalize_timebase(x,component='gaze_samples',native_unit='ms')
    assert y['gaze_samples']['timestamp_seconds'].tolist()==pytest.approx([0,.01,0,.01])
    assert len(y['provenance'])==1

def test_coordinate_conversion_copy_and_overwrite():
    rec=pd.DataFrame({'recording_id':['r1']})
    spaces=pd.concat([ep.new_coordinate_space('norm'),ep.new_coordinate_space('px','display_pixels_top_left',width=1000,height=500)],ignore_index=True)
    gaze=pd.DataFrame({'recording_id':['r1'],'sample_id':['s1'],'gaze_x':[.5],'gaze_y':[.25],'coordinate_space_id':['norm']})
    x=ep.new_eye_dataset(recordings=rec,coordinate_spaces=spaces,gaze_samples=gaze,validate=False)
    y=ep.convert_coordinates(x,'norm','px')
    row=y['gaze_samples'][y['gaze_samples']['coordinate_space_id']=='px'].iloc[0]
    assert row.gaze_x==pytest.approx(500) and row.gaze_y==pytest.approx(125) and row.sample_id=='s1_px'

def test_provenance_and_compaction():
    x=ep.new_eye_dataset(validate=False)
    y=ep.add_provenance(x,'test','dataset','hello')
    assert len(y['provenance'])==1 and y['provenance'].iloc[0].action=='test'
    z=ep.compact_eye_dataset(y,drop_raw=True,drop_empty=True)
    assert 'recordings' in z.empty_components
