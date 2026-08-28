import inspect
import eyeprocesspy as ep

IMPLEMENTED={
'eye_schema','schema_table','empty_eye_table','standardize_eye_table','validate_eye_table','canonical_table_names','new_coordinate_space','eye_mapping','new_eye_dataset','is_eye_dataset','validate_eye_dataset','get_eye_table','set_eye_table','append_eye_table','add_provenance','provenance_manifest','compact_eye_dataset','estimate_sampling_rate','normalize_timebase','audit_timebase','align_clock','estimate_clock_transform','apply_clock_transform','register_coordinate_space','coordinate_space','convert_coordinates','audit_coordinate_spaces'
}

def test_implemented_exports_exist_without_placeholders():
    for name in IMPLEMENTED:
        obj=getattr(ep,name)
        assert callable(obj)
        source=inspect.getsource(obj)
        assert 'NotImplementedError' not in source
        assert source.strip() not in {f'def {name}(*args, **kwargs):\n    pass', f'def {name}(*args, **kwargs):\n    return None'}
