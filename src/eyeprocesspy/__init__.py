"""eyeprocesspy: Python parity implementation of R eyeprocess 0.11.1."""
__version__ = "0.1.0.dev0"
__r_reference_version__ = "0.11.1"

from .exceptions import *
from .schema import eye_schema, schema_table, empty_eye_table, standardize_eye_table, validate_eye_table, canonical_table_names, new_coordinate_space
from .mapping import eye_mapping
from .dataset import EyeDataset, new_eye_dataset, is_eye_dataset, validate_eye_dataset, get_eye_table, set_eye_table, append_eye_table, add_provenance, provenance_manifest, compact_eye_dataset
from .timebase import EyeClockTransform, estimate_sampling_rate, normalize_timebase, audit_timebase, align_clock, estimate_clock_transform, apply_clock_transform
from .coordinates import register_coordinate_space, coordinate_space, convert_coordinates, audit_coordinate_spaces

__all__ = [n for n in globals() if not n.startswith('_')]
