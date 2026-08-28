import inspect
import eyeprocesspy as ep

IMPLEMENTED={
'eye_schema','schema_table','empty_eye_table','standardize_eye_table','validate_eye_table','canonical_table_names','new_coordinate_space','eye_mapping','new_eye_dataset','is_eye_dataset','validate_eye_dataset','get_eye_table','set_eye_table','append_eye_table','add_provenance','provenance_manifest','compact_eye_dataset','estimate_sampling_rate','normalize_timebase','audit_timebase','align_clock','estimate_clock_transform','apply_clock_transform','register_coordinate_space','coordinate_space','convert_coordinates','audit_coordinate_spaces','validate_eye_mapping','infer_eye_mapping','read_eye_generic','register_eye_adapter','unregister_eye_adapter','supported_eye_formats','detect_eye_format','read_eye_export','read_eye_folder','combine_eye_datasets','remap_recording_ids','is_gazepoint_export','gp_identify_export_type','gp_profile_export','gp_list_export_fields','gp_validate_export','read_gazepoint','read_gazepoint_gaze','read_gazepoint_fixations','read_gazepoint_events','read_gazepoint_folder','gp_pair_exports','gp_match_recordings','gp_match_biometrics','gp_audit_file_pairs','read_gazepoint_biometrics','read_gazepoint_combined','gp_parse_user_events','gp_parse_media_events'
}

def test_implemented_exports_exist_without_placeholders():
    for name in IMPLEMENTED:
        obj=getattr(ep,name)
        assert callable(obj)
        source=inspect.getsource(obj)
        assert 'NotImplementedError' not in source
        assert source.strip() not in {f'def {name}(*args, **kwargs):\n    pass', f'def {name}(*args, **kwargs):\n    return None'}
