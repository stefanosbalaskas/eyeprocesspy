"""eyeprocesspy: Python parity implementation of R eyeprocess 0.11.1."""
__version__ = "0.1.0.dev0"
__r_reference_version__ = "0.11.1"

from .exceptions import *
from .schema import eye_schema, schema_table, empty_eye_table, standardize_eye_table, validate_eye_table, canonical_table_names, new_coordinate_space
from .mapping import eye_mapping
from .dataset import EyeDataset, new_eye_dataset, is_eye_dataset, validate_eye_dataset, get_eye_table, set_eye_table, append_eye_table, add_provenance, provenance_manifest, compact_eye_dataset
from .timebase import EyeClockTransform, estimate_sampling_rate, normalize_timebase, audit_timebase, align_clock, estimate_clock_transform, apply_clock_transform
from .coordinates import register_coordinate_space, coordinate_space, convert_coordinates, audit_coordinate_spaces

from .importers import validate_eye_mapping, infer_eye_mapping, read_eye_generic
from .adapters import register_eye_adapter, unregister_eye_adapter, supported_eye_formats, detect_eye_format, read_eye_export, read_eye_folder, combine_eye_datasets, remap_recording_ids

from .gazepoint import (is_gazepoint_export, gp_identify_export_type, gp_profile_export, gp_list_export_fields, gp_validate_export, read_gazepoint, read_gazepoint_gaze, read_gazepoint_fixations, read_gazepoint_events, read_gazepoint_folder, gp_pair_exports, gp_match_recordings, gp_match_biometrics, gp_audit_file_pairs, read_gazepoint_biometrics, read_gazepoint_combined, gp_parse_user_events, gp_parse_media_events)
register_eye_adapter("gazepoint", is_gazepoint_export, read_gazepoint, gp_validate_export, priority=100, overwrite=True)

from .irt import *
from .plots_irt import *
from .measurement_intelligence import *

__all__ = [n for n in globals() if not n.startswith('_')]
from .dynamic_irt import *

__all__ = [n for n in globals() if not n.startswith('_')]
from .process_irt_07 import *

__all__ = [n for n in globals() if not n.startswith('_')]
from .advanced_process_irt_07 import *
from .irt_validation_07 import *
from .plots_process_irt_07 import *

__all__ = [n for n in globals() if not n.startswith('_')]

from .semantic_validation_07 import *
from .requested_api_07 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .context_structure_08 import *
from .frontier_08 import *
from .sensitivity_08 import *
from .bayesian_3pl_08 import *
from .plots_irt_08 import *

__all__ = [n for n in globals() if not n.startswith("_")]
