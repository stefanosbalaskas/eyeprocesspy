# Bayesian-network data contract

The BN layer accepts a **feature table**, not raw gaze samples. A typical static network uses one row per participant × trial.

## Recommended columns

| Group | Examples |
|---|---|
| Identity | `participant_id`, `trial_id`, `stimulus_id` |
| Experimental | `condition`, `target_aoi` |
| Gaze | `first_fixation_latency`, `target_dwell_fraction`, `fixation_count`, `revisit_count`, `transition_entropy` |
| Pupil | `pupil_baseline`, `pupil_mean_change`, `pupil_peak_change`, `pupil_auc` |
| Physiology | `scr_count`, `scr_mean_amplitude`, `scr_peak_amplitude`, `phasic_auc` |
| Questionnaire | `perceived_control`, `trust`, `confidence`, `workload` |
| Behavior | `choice`, `reaction_time`, `accuracy`, `reliance` |
| Quality | `gaze_valid_fraction`, `pupil_valid_fraction`, `gaze_accuracy`, `gaze_precision` |
| Provenance | detector/AOI/preprocessing/timebase specification IDs, software version |

## Node metadata is scientific metadata

~~~python
from eyeprocesspy.bayesian_networks import define_bn_nodes

nodes = define_bn_nodes({
    "target_dwell_fraction": {
        "type": "continuous",
        "modality": "gaze",
        "level": "trial",
        "unit": "proportion",
        "range": [0, 1],
    },
    "choice": {
        "type": "categorical",
        "modality": "behavior",
        "states": ["Accept", "Reject"],
    },
    "gaze_valid_fraction": {
        "type": "continuous",
        "role": "quality",
        "include_in_structure": False,
    },
})
~~~

Quality nodes are excluded from structure learning by default. Include one only when recording quality itself is a prespecified substantive variable.

## Missingness is not zero

The preparation layer preserves missing values. It never applies `fillna(0)`. A zero fixation count, sensor loss, artifact rejection, and not-applicable state have different scientific meanings and should stay distinguishable upstream.

## No automatic discretization

Discrete teaching examples are convenient, but real pupil, EDA, dwell, latency, and questionnaire measurements should not be split into “high/low” merely to obtain a CPT. Use an appropriate model family or make discretization an explicit, justified preprocessing decision and include it in sensitivity analysis.
