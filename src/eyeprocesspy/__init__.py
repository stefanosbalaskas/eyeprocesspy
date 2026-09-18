"""eyeprocesspy: Python parity implementation of R eyeprocess 0.11.1."""
__version__ = "0.1.0"
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
from .detector_multiverse import (
    DetectorInferenceResult as DetectorInferenceResult,
    DetectorMultiverse as DetectorMultiverse,
    DetectorMultiverseResult as DetectorMultiverseResult,
    EventDetectorSpec as EventDetectorSpec,
    assess_detector_inference_stability as assess_detector_inference_stability,
    compare_event_catalogues as compare_event_catalogues,
    create_detector_multiverse as create_detector_multiverse,
    define_event_detector_spec as define_event_detector_spec,
    detect_events_with_spec as detect_events_with_spec,
    estimate_detector_agreement as estimate_detector_agreement,
    import_external_detector_events as import_external_detector_events,
    match_detected_events as match_detected_events,
    plot_detector_agreement as plot_detector_agreement,
    plot_detector_coefficient_stability as plot_detector_coefficient_stability,
    plot_detector_event_timeline as plot_detector_event_timeline,
    plot_detector_feature_distributions as plot_detector_feature_distributions,
    plot_detector_multiverse as plot_detector_multiverse,
    propagate_detector_to_aoi as propagate_detector_to_aoi,
    propagate_detector_to_features as propagate_detector_to_features,
    report_detector_multiverse as report_detector_multiverse,
    run_detector_inference_multiverse as run_detector_inference_multiverse,
    run_detector_multiverse as run_detector_multiverse,
    simulate_detector_multiverse_data as simulate_detector_multiverse_data,
    summarise_detector_disagreement as summarise_detector_disagreement,
    summarise_detector_events as summarise_detector_events,
    summarise_detector_robustness as summarise_detector_robustness,
    validate_event_detector_spec as validate_event_detector_spec,
)

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

from .multimodal_staged import *
from .plots_multimodal_staged import *
from .legacy_models import *
from .plots_legacy_models import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .functional_pupil import *
from .plots_functional_pupil import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .engine_adapters import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .process_quality_09 import *
from .plots_process_quality_09 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .measurement_quality_legacy import *
from .pupil_missingness import *
from .process_dynamics import *
from .evidence_graph import *

__all__ = [n for n in globals() if not n.startswith('_')]

from .process_governance_08 import *

__all__ = [n for n in globals() if not n.startswith('_')]
from .plots_governance_08 import *
__all__ = [n for n in globals() if not n.startswith('_')]

from .operational_validation_08 import *
from .plots_operational_08 import *

__all__ = [n for n in globals() if not n.startswith('_')]

from .governance_09 import *
from .plots_governance_09 import *

__all__ = [n for n in globals() if not n.startswith('_')]

from .foundation_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .preprocess_features_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .io_validation_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .gazepoint_real_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .gazepoint_workflow_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .interoperability_storage_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_program_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_evidence_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .grouped_validation_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_completion_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_orchestration_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_orchestration_completion_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .partitioned_storage_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .benchmark_reproducibility_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .vendor_importers_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .vendor_corpus_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .core_plots_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .measurement_intelligence_utils_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .probabilistic_aoi_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .compositional_aoi_10 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .aoi_perturbation import (  # noqa: E402, F401, I001
    AMBIGUOUS,
    OUTSIDE,
    aoi_perturbation_spec,
    apply_aoi_perturbation_grid,
    assess_aoi_inference_stability,
    compare_aoi_assignments,
    convert_aoi_margin_to_degrees,
    convert_aoi_margin_to_pixels,
    create_aoi_perturbation_grid,
    dilate_aoi,
    erode_aoi,
    estimate_aoi_assignment_stability,
    estimate_fixation_assignment_probability,
    jitter_aoi,
    perturb_aoi_geometry,
    plot_aoi_assignment_stability,
    plot_aoi_coefficient_stability,
    plot_aoi_perturbations,
    plot_aoi_robustness_surface,
    recompute_aoi_features,
    report_aoi_sensitivity,
    run_aoi_sensitivity_analysis,
    summarise_aoi_sensitivity,
    translate_aoi,
    validate_aoi_geometry,
)
__all__ = [n for n in globals() if not n.startswith("_")]

from .plots_completion_08 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_extras_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .negative_controls_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .benchmark_stress_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .reproducibility_provenance_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .software_paper_evidence_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_evidence_programs_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .validation_stress_freeze_09 import *
__all__ = [n for n in globals() if not n.startswith("_")]

from .measurement_accountability_11 import event_marker_qc, pupil_latency_sensitivity, validation_ladder
from .validation_atlas_09 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .multilevel_mediation import *
__all__ = [n for n in globals() if not n.startswith("_")]


