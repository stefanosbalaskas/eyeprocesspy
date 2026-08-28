from pathlib import Path
import eyeprocesspy as ep

FIX=Path(__file__).parent/'fixtures'/'gazepoint'


def test_gazepoint_sample_and_biometric_fields_are_imported():
    p=FIX/'demo-user.csv'
    assert ep.is_gazepoint_export(p)>.8
    x=ep.read_gazepoint(p,recording_id='R1',quiet=True)
    assert len(x['gaze_samples'])==12
    assert len(x['eye_samples'])==24
    assert {'heart_rate','gsr_raw'} <= set(x['biometrics'].channel)
    assert 'eda' not in set(x['biometrics'].channel)
    assert (x['events'].event_name=='TRIAL_START item01').any()
    assert x['recordings'].device_model.iloc[0]=='Gazepoint'


def test_gazepoint_fixation_export_retains_vendor_derivation():
    x=ep.read_gazepoint_fixations(FIX/'demo-user-fix.csv',recording_id='R1',quiet=True)
    assert len(x['episodes'])==4
    assert set(x['episodes'].derived_by)=={'vendor'}


def test_gazepoint_folder_combines_identity_tables_safely():
    x=ep.read_gazepoint_folder(FIX,recording_id='R1',quiet=True)
    assert ep.is_eye_dataset(x)
    assert list(x['recordings'].recording_id.unique())==['R1']
    assert not x['recordings'].recording_id.duplicated().any()
    via=ep.read_gazepoint(FIX,recording_id='R2',quiet=True)
    assert list(via['recordings'].recording_id.unique())==['R2']


def test_gazepoint_biometric_matching_uses_explicit_columns():
    matched=ep.gp_match_biometrics(FIX)
    assert set(matched.export_type) <= {'combined_biometrics','gaze'}
    assert not matched.empty


def test_gazepoint_identification_and_profile():
    assert ep.gp_identify_export_type(FIX/'demo-user.csv')=='combined_biometrics'
    assert ep.gp_identify_export_type(FIX/'demo-user-fix.csv')=='fixations'
    prof=ep.gp_profile_export(FIX)
    assert set(prof.export_type)=={'combined_biometrics','fixations'}
    fields=ep.gp_list_export_fields(FIX/'demo-user.csv')
    assert {'BPOGX','BPOGY','HR','GSR'} <= set(fields)
    assert ep.gp_validate_export(FIX/'demo-user.csv').empty


def test_gazepoint_biometrics_view_removes_gaze_samples():
    x=ep.read_gazepoint_biometrics(FIX/'demo-user.csv',recording_id='R1',quiet=True)
    assert x['gaze_samples'].empty
    assert {'heart_rate','gsr_raw'} <= set(x['biometrics'].channel)
    assert 'gaze_combined' not in set(x['streams'].stream_type)
