"""CI-small trial-level mediation preparation example."""

import numpy as np
import pandas as pd

from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data

rng = np.random.default_rng(2026)
participants = np.repeat([f"p{i:02d}" for i in range(1, 7)], 6)
trials = np.tile(np.arange(1, 7), 6)
x = np.tile([0, 1, 0, 1, 0, 1], 6)
participant_shift = np.repeat(rng.normal(0, 0.15, 6), 6)
dwell = 1.2 + 0.45 * (x - 0.5) + participant_shift + rng.normal(0, 0.2, len(x))
y = (
    rng.random(len(x))
    < 1
    / (
        1
        + np.exp(
            -(
                -0.3
                + 0.5 * (x - 0.5)
                + 0.6 * (dwell - pd.Series(dwell).groupby(participants).transform("mean"))
            )
        )
    )
).astype(int)
quality = np.full(len(x), 0.95)
quality[4] = 0.40
# An observed zero is different from an unobserved mediator.
dwell[8] = 0.0
dwell[11] = np.nan

data = pd.DataFrame(
    {
        "participant_id": participants,
        "trial_id": trials,
        "ai_correct": x,
        "source_dwell": dwell,
        "correct_override": y,
        "valid_fraction": quality,
    }
)

prepared = prepare_multilevel_mediation_data(
    data,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    quality_col="valid_fraction",
    minimum_quality=0.80,
    source_id="synthetic-ci-example",
    preprocessing_spec={"example": "no preprocessing"},
    warn=False,
)

print(
    prepared.data[
        [
            "participant_id",
            "trial_id",
            "X_within",
            "X_between",
            "M_within",
            "M_between",
            "mediation_mediator_state",
            "mediation_analysis_eligible",
        ]
    ]
    .head(12)
    .to_string(index=False)
)
print("\nMissingness audit")
print(prepared.missingness.to_string(index=False))
print("\nVariation classification")
print(prepared.levels.to_string(index=False))
