from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

from .exceptions import EyeProcessSchemaError

SCHEMA_VERSION = "0.1.0"

_TABLES = OrderedDict([
    ("recordings", ["recording_id","participant_id","session_id","vendor","vendor_family","device_model","firmware_version","software_name","software_version","experiment_type","nominal_sampling_rate","screen_width_px","screen_height_px","recording_start","source_timezone","source_file_set"]),
    ("streams", ["stream_id","recording_id","stream_type","source_device","source_clock","sampling_type","nominal_rate_hz","observed_rate_hz","timestamp_unit","value_unit","coordinate_space_id","processing_level"]),
    ("gaze_samples", ["recording_id","stream_id","sample_id","timestamp_native","timestamp_seconds","gaze_x","gaze_y","gaze_z","azimuth_deg","elevation_deg","valid","confidence","fixation_id_source","blink_id_source","trial_id","stimulus_id","coordinate_space_id"]),
    ("eye_samples", ["recording_id","sample_id","timestamp_native","timestamp_seconds","eye","pupil_diameter","pupil_unit","pupil_valid","eye_openness","gaze_origin_x","gaze_origin_y","gaze_origin_z","gaze_origin_valid","corneal_reflection_x","corneal_reflection_y","detector_method","confidence","trial_id","stimulus_id"]),
    ("episodes", ["episode_id","recording_id","episode_type","eye","start_time","end_time","duration_ms","start_x","start_y","end_x","end_y","centroid_x","centroid_y","amplitude","peak_velocity","dispersion","coordinate_space_id","source_algorithm","source_parameters","derived_by","trial_id","stimulus_id","aoi_id"]),
    ("events", ["event_id","recording_id","timestamp_native","timestamp_seconds","event_type","event_name","event_value","duration","source","native_record","trial_id","stimulus_id"]),
    ("intervals", ["interval_id","recording_id","interval_type","start_time","end_time","trial_id","participant_id","item_id","stimulus_id","condition_id","parent_interval_id","valid_interval"]),
    ("responses", ["response_id","recording_id","participant_id","trial_id","item_id","response","score","response_time","response_timestamp","response_type","valid_response"]),
    ("coordinate_spaces", ["coordinate_space_id","space_type","origin","x_unit","y_unit","width","height","reference_object","parent_space_id","transform_to_parent","clipping_policy"]),
    ("aoi_definitions", ["aoi_id","aoi_name","stimulus_id","shape_type","coordinate_space_id","parent_aoi_id","source"]),
    ("aoi_geometry", ["aoi_id","valid_from","valid_to","frame_id","x","y","width","height","polygon","visible","coordinate_space_id"]),
    ("biometrics", ["recording_id","stream_id","timestamp_native","timestamp_seconds","channel","value","unit","valid","processing_level","source_device","trial_id","stimulus_id"]),
    ("calibrations", ["calibration_id","recording_id","timestamp_seconds","calibration_type","eye","point_count","average_error","maximum_error","error_unit","validation_status","drift_offset","source_record"]),
    ("features", ["feature_id","recording_id","participant_id","trial_id","item_id","stimulus_id","aoi_id","feature_name","value","unit","level","window_start","window_end","observed_fraction","method","parameters","derived_at"]),
    ("quality", ["quality_id","recording_id","trial_id","stream_id","metric","value","threshold","status","message","computed_at"]),
    ("provenance", ["provenance_id","timestamp","action","component","details","source_files","file_hashes","software","software_version","reversible","warnings"]),
])

def eye_schema(version: str = SCHEMA_VERSION) -> dict:
    """Return the canonical eyeprocess schema (R `eye_schema`)."""
    return {"version": version, "tables": OrderedDict((k, list(v)) for k,v in _TABLES.items())}

def schema_table(name: str, schema: Mapping | None = None) -> list[str]:
    """Return canonical fields for one schema table."""
    if not isinstance(name, str) or not name:
        raise EyeProcessSchemaError("`name` must be a non-empty string.")
    schema = eye_schema() if schema is None else schema
    if name not in schema["tables"]:
        raise EyeProcessSchemaError(f"Unknown schema table `{name}`.")
    return list(schema["tables"][name])

def empty_eye_table(name: str, schema: Mapping | None = None) -> pd.DataFrame:
    """Create an empty canonical table with the R column order."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in schema_table(name, schema)})

def standardize_eye_table(data: pd.DataFrame, name: str, keep_extra: bool = True, schema: Mapping | None = None) -> pd.DataFrame:
    """Add absent canonical fields and reorder columns like R `standardize_eye_table`."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    cols=schema_table(name, schema)
    out=data.copy()
    for c in cols:
        if c not in out.columns:
            out[c]=pd.NA
    ordered=cols + ([c for c in out.columns if c not in cols] if keep_extra else [])
    return out.loc[:, ordered].reset_index(drop=True)

def validate_eye_table(data: pd.DataFrame, name: str, strict: bool = False, schema: Mapping | None = None) -> pd.DataFrame:
    """Return schema-field issues using the R validation table contract."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    cols=schema_table(name,schema)
    missing=[c for c in cols if c not in data.columns]
    extra=[c for c in data.columns if c not in cols]
    rows=[]
    for c in missing:
        rows.append(dict(severity="error" if strict else "warning",code="missing_schema_field",table=name,field=c,message=f"Schema field `{c}` is absent."))
    if strict:
        for c in extra:
            rows.append(dict(severity="warning",code="extra_field",table=name,field=c,message=f"Non-canonical field `{c}` is retained."))
    return pd.DataFrame(rows,columns=["severity","code","table","field","message"])

def canonical_table_names() -> list[str]:
    """Return canonical table names in frozen R order."""
    return list(_TABLES)

_SPACE_DEFAULTS={
    "display_normalized_top_left":("top_left","normalized","normalized"),
    "display_pixels_top_left":("top_left","pixels","pixels"),
    "surface_normalized_bottom_left":("bottom_left","normalized","normalized"),
    "world_camera_pixels":("top_left","pixels","pixels"),
    "reference_image_pixels":("top_left","pixels","pixels"),
    "user_coordinates_3d":("vendor_defined","millimetres","millimetres"),
    "headset_coordinates_3d":("vendor_defined","metres","metres"),
    "gaze_direction_vector":("origin","unit_vector","unit_vector"),
    "custom":("unknown","unknown","unknown"),
}

def new_coordinate_space(coordinate_space_id, space_type="display_normalized_top_left", origin=None, x_unit=None, y_unit=None, width=np.nan, height=np.nan, reference_object=pd.NA, parent_space_id=pd.NA, transform_to_parent=pd.NA, clipping_policy="retain") -> pd.DataFrame:
    """Create one coordinate-space record using the frozen R defaults."""
    if space_type not in _SPACE_DEFAULTS:
        raise ValueError(f"Invalid `space_type`: {space_type}")
    d=_SPACE_DEFAULTS[space_type]
    return pd.DataFrame([dict(coordinate_space_id=str(coordinate_space_id),space_type=space_type,origin=origin or d[0],x_unit=x_unit or d[1],y_unit=y_unit or d[2],width=float(width) if not pd.isna(width) else np.nan,height=float(height) if not pd.isna(height) else np.nan,reference_object=reference_object,parent_space_id=parent_space_id,transform_to_parent=transform_to_parent,clipping_policy=str(clipping_policy))])
