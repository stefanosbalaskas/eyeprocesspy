"""eyeprocesspy: Python parity implementation of R eyeprocess 0.11.1."""

__version__ = "0.1.0"
__r_reference_version__ = "0.11.1"

from .adapters import (
    combine_eye_datasets,
    detect_eye_format,
    read_eye_export,
    read_eye_folder,
    register_eye_adapter,
    remap_recording_ids,
    supported_eye_formats,
    unregister_eye_adapter,
)
from .coordinates import (
    audit_coordinate_spaces,
    convert_coordinates,
    coordinate_space,
    register_coordinate_space,
)
from .dataset import (
    EyeDataset,
    add_provenance,
    append_eye_table,
    compact_eye_dataset,
    get_eye_table,
    is_eye_dataset,
    new_eye_dataset,
    provenance_manifest,
    set_eye_table,
    validate_eye_dataset,
)
from .detector_multiverse import (
    DetectorInferenceResult as DetectorInferenceResult,
)
from .detector_multiverse import (
    DetectorMultiverse as DetectorMultiverse,
)
from .detector_multiverse import (
    DetectorMultiverseResult as DetectorMultiverseResult,
)
from .detector_multiverse import (
    EventDetectorSpec as EventDetectorSpec,
)
from .detector_multiverse import (
    assess_detector_inference_stability as assess_detector_inference_stability,
)
from .detector_multiverse import (
    compare_event_catalogues as compare_event_catalogues,
)
from .detector_multiverse import (
    create_detector_multiverse as create_detector_multiverse,
)
from .detector_multiverse import (
    define_event_detector_spec as define_event_detector_spec,
)
from .detector_multiverse import (
    detect_events_with_spec as detect_events_with_spec,
)
from .detector_multiverse import (
    estimate_detector_agreement as estimate_detector_agreement,
)
from .detector_multiverse import (
    import_external_detector_events as import_external_detector_events,
)
from .detector_multiverse import (
    match_detected_events as match_detected_events,
)
from .detector_multiverse import (
    plot_detector_agreement as plot_detector_agreement,
)
from .detector_multiverse import (
    plot_detector_coefficient_stability as plot_detector_coefficient_stability,
)
from .detector_multiverse import (
    plot_detector_event_timeline as plot_detector_event_timeline,
)
from .detector_multiverse import (
    plot_detector_feature_distributions as plot_detector_feature_distributions,
)
from .detector_multiverse import (
    plot_detector_multiverse as plot_detector_multiverse,
)
from .detector_multiverse import (
    propagate_detector_to_aoi as propagate_detector_to_aoi,
)
from .detector_multiverse import (
    propagate_detector_to_features as propagate_detector_to_features,
)
from .detector_multiverse import (
    report_detector_multiverse as report_detector_multiverse,
)
from .detector_multiverse import (
    run_detector_inference_multiverse as run_detector_inference_multiverse,
)
from .detector_multiverse import (
    run_detector_multiverse as run_detector_multiverse,
)
from .detector_multiverse import (
    simulate_detector_multiverse_data as simulate_detector_multiverse_data,
)
from .detector_multiverse import (
    summarise_detector_disagreement as summarise_detector_disagreement,
)
from .detector_multiverse import (
    summarise_detector_events as summarise_detector_events,
)
from .detector_multiverse import (
    summarise_detector_robustness as summarise_detector_robustness,
)
from .detector_multiverse import (
    validate_event_detector_spec as validate_event_detector_spec,
)
from .exceptions import *
from .gazepoint import (
    gp_audit_file_pairs,
    gp_identify_export_type,
    gp_list_export_fields,
    gp_match_biometrics,
    gp_match_recordings,
    gp_pair_exports,
    gp_parse_media_events,
    gp_parse_user_events,
    gp_profile_export,
    gp_validate_export,
    is_gazepoint_export,
    read_gazepoint,
    read_gazepoint_biometrics,
    read_gazepoint_combined,
    read_gazepoint_events,
    read_gazepoint_fixations,
    read_gazepoint_folder,
    read_gazepoint_gaze,
)
from .importers import infer_eye_mapping, read_eye_generic, validate_eye_mapping
from .mapping import eye_mapping
from .schema import (
    canonical_table_names,
    empty_eye_table,
    eye_schema,
    new_coordinate_space,
    schema_table,
    standardize_eye_table,
    validate_eye_table,
)
from .timebase import (
    EyeClockTransform,
    align_clock,
    apply_clock_transform,
    audit_timebase,
    estimate_clock_transform,
    estimate_sampling_rate,
    normalize_timebase,
)

register_eye_adapter(
    "gazepoint",
    is_gazepoint_export,
    read_gazepoint,
    gp_validate_export,
    priority=100,
    overwrite=True,
)

from .irt import *
from .measurement_intelligence import *
from .plots_irt import *

__all__ = [n for n in globals() if not n.startswith("_")]
from .dynamic_irt import *

__all__ = [n for n in globals() if not n.startswith("_")]
from .process_irt_07 import *

__all__ = [n for n in globals() if not n.startswith("_")]
from .advanced_process_irt_07 import *
from .irt_validation_07 import *
from .plots_process_irt_07 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .requested_api_07 import *
from .semantic_validation_07 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .bayesian_3pl_08 import *
from .context_structure_08 import *
from .frontier_08 import *
from .plots_irt_08 import *
from .sensitivity_08 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .legacy_models import *
from .multimodal_staged import *
from .plots_legacy_models import *
from .plots_multimodal_staged import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .functional_pupil import *
from .plots_functional_pupil import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .engine_adapters import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .plots_process_quality_09 import *
from .process_quality_09 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .evidence_graph import *
from .measurement_quality_legacy import *
from .process_dynamics import *
from .pupil_missingness import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .process_governance_08 import *

__all__ = [n for n in globals() if not n.startswith("_")]
from .plots_governance_08 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .operational_validation_08 import *
from .plots_operational_08 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .governance_09 import *
from .plots_governance_09 import *

__all__ = [n for n in globals() if not n.startswith("_")]

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

from .measurement_accountability_11 import (
    event_marker_qc,
    pupil_latency_sensitivity,
    validation_ladder,
)
from .validation_atlas_09 import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .multilevel_mediation import *

__all__ = [n for n in globals() if not n.startswith("_")]

from .survival import (  # noqa: E402, I001
    CANONICAL_GAZE_SURVIVAL_COLUMNS as CANONICAL_GAZE_SURVIVAL_COLUMNS,
    GazeSurvivalFit as GazeSurvivalFit,
    check_gaze_proportional_hazards as check_gaze_proportional_hazards,
    compare_gaze_survival_models as compare_gaze_survival_models,
    estimate_gaze_latency_quantiles as estimate_gaze_latency_quantiles,
    estimate_gaze_survival as estimate_gaze_survival,
    fit_gaze_aft_model as fit_gaze_aft_model,
    fit_gaze_cox_model as fit_gaze_cox_model,
    fit_gaze_mixed_cox_model as fit_gaze_mixed_cox_model,
    plot_gaze_cox_diagnostics as plot_gaze_cox_diagnostics,
    plot_gaze_cumulative_incidence as plot_gaze_cumulative_incidence,
    plot_gaze_hazard as plot_gaze_hazard,
    plot_gaze_survival_curve as plot_gaze_survival_curve,
    predict_gaze_survival as predict_gaze_survival,
    prepare_gaze_survival_data as prepare_gaze_survival_data,
    report_gaze_survival_model as report_gaze_survival_model,
    simulate_gaze_survival_example as simulate_gaze_survival_example,
    simulate_gaze_survival_inputs as simulate_gaze_survival_inputs,
    summarise_gaze_censoring as summarise_gaze_censoring,
    tidy_gaze_survival_model as tidy_gaze_survival_model,
    validate_gaze_survival_data as validate_gaze_survival_data,
)

__all__ = [n for n in globals() if not n.startswith("_")]
from .spatial_quality import *

__all__ = [n for n in globals() if not n.startswith("_")]
