"""Participant-aware cross-validation and synthetic structure recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import pgmpy_backend
from .fitting import fit_bayesian_network
from .learning import learn_bayesian_network
from .schema import BayesianConstraintSpec, BayesianDataSpec, BayesianNetworkValidationResult


def _group_folds(
    groups: pd.Series,
    n_splits: int,
    random_state: int | None,
) -> list[set[object]]:
    unique = np.asarray(pd.Index(groups.dropna().unique()), dtype=object)
    if len(unique) < n_splits:
        raise ValueError("n_splits cannot exceed the number of observed groups.")
    rng = np.random.default_rng(random_state)
    unique = unique[rng.permutation(len(unique))]
    return [set(chunk.tolist()) for chunk in np.array_split(unique, n_splits)]


def _subset(data_spec: BayesianDataSpec, rows: pd.Series) -> BayesianDataSpec:
    return BayesianDataSpec(
        data=data_spec.data.loc[rows].reset_index(drop=True),
        node_specs=data_spec.node_specs,
        participant_id=data_spec.participant_id,
        trial_id=data_spec.trial_id,
        stimulus_id=data_spec.stimulus_id,
        observation_level=data_spec.observation_level,
        provenance=data_spec.provenance,
        source_columns=data_spec.source_columns,
    )


def validate_bayesian_network(
    data_spec: BayesianDataSpec,
    *,
    algorithm: str = "hill_climb",
    score: str | None = "bic",
    constraints: BayesianConstraintSpec | None = None,
    method: str = "group_kfold",
    groups: str | None = None,
    n_splits: int = 10,
    estimator: str = "bayesian",
    prior: str = "BDeu",
    equivalent_sample_size: float = 10.0,
    max_parents: int | None = None,
    random_state: int | None = 42,
    backend: str = "pgmpy",
) -> BayesianNetworkValidationResult:
    """Learn and fit inside each training fold; score only held-out participants."""
    if method != "group_kfold":
        raise ValueError("The first BN tranche supports method='group_kfold' only.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    group_col = groups or data_spec.participant_id
    if group_col is None or group_col not in data_spec.data:
        raise ValueError("Grouped validation requires an explicit participant/group column.")

    folds = _group_folds(data_spec.data[group_col], n_splits, random_state)
    rows: list[dict[str, object]] = []
    for fold_id, held_out in enumerate(folds):
        is_test = data_spec.data[group_col].isin(held_out)
        train = _subset(data_spec, ~is_test)
        test = _subset(data_spec, is_test)
        try:
            learned = learn_bayesian_network(
                train,
                algorithm=algorithm,
                score=score,
                constraints=constraints,
                max_parents=max_parents,
                random_state=random_state,
                backend=backend,
                allow_unconstrained=True,
            )
            fitted = fit_bayesian_network(
                learned,
                estimator=estimator,
                prior=prior,
                equivalent_sample_size=equivalent_sample_size,
                backend=backend,
            )
            ll = pgmpy_backend.log_likelihood(
                fitted.backend_model,
                family=fitted.model_family,
                data=test.data,
                nodes=list(fitted.nodes),
            )
            status = "ok"
            error = ""
        except (ValueError, RuntimeError) as exc:
            ll = float("nan")
            status = "failed"
            error = str(exc)
        rows.append(
            {
                "fold": fold_id,
                "n_train": len(train.data),
                "n_test": len(test.data),
                "n_groups_test": len(held_out),
                "log_likelihood": ll,
                "mean_log_likelihood": ll / len(test.data),
                "status": status,
                "error": error,
            }
        )
    table = pd.DataFrame(rows)
    return BayesianNetworkValidationResult(
        fold_table=table,
        method=method,
        group_column=group_col,
        summary={
            "mean_log_likelihood": float(table["mean_log_likelihood"].mean()),
            "sd_log_likelihood": float(table["mean_log_likelihood"].std(ddof=1)),
            "n_splits": float(n_splits),
            "failed_folds": float((table["status"] != "ok").sum()),
        },
        provenance={"random_state": random_state, "algorithm": algorithm, "score": score},
    )


def validate_bayesian_network_recovery(
    learned_edges: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    true_edges: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> dict[str, float | int]:
    """Evaluate edge and direction recovery against a known synthetic DAG."""
    learned = set(learned_edges)
    truth = set(true_edges)
    tp = len(learned & truth)
    fp = len(learned - truth)
    fn = len(truth - learned)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    skeleton_learned = {tuple(sorted(edge)) for edge in learned}
    skeleton_truth = {tuple(sorted(edge)) for edge in truth}
    skeleton_errors = len(skeleton_learned ^ skeleton_truth)
    reversed_edges = sum((b, a) in learned and (a, b) not in learned for a, b in truth)
    return {
        "true_positive_edges": tp,
        "false_positive_edges": fp,
        "false_negative_edges": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skeleton_symmetric_difference": skeleton_errors,
        "reversed_true_edges": reversed_edges,
    }


def simulate_multimodal_bayesian_network_example(
    *,
    n_participants: int = 400,
    trials_per_participant: int = 1,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Simulate the documented discrete multimodal teaching network.

    The generated states are intentionally discrete for transparent validation;
    they are not a recommendation to dichotomize continuous pupil or EDA data.
    """
    if n_participants < 1 or trials_per_participant < 1:
        raise ValueError("n_participants and trials_per_participant must be positive.")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for participant in range(n_participants):
        for trial in range(trials_per_participant):
            explanation = bool(rng.random() < 0.5)
            gaze_high = bool(rng.random() < (0.75 if explanation else 0.25))
            pupil_dilated = bool(rng.random() < (0.75 if gaze_high else 0.25))
            eda_high = bool(rng.random() < ((2 / 3) if gaze_high else (1 / 3)))
            if explanation and gaze_high:
                trust_probability = 8 / 9
            elif explanation or gaze_high:
                trust_probability = 2 / 3
            else:
                trust_probability = 2 / 9
            trust_high = bool(rng.random() < trust_probability)
            accept = bool(rng.random() < ((6 / 7) if trust_high else 0.20))
            rows.append(
                {
                    "participant_id": f"P{participant + 1:04d}",
                    "trial_id": f"T{trial + 1:03d}",
                    "condition": "Explanation" if explanation else "Control",
                    "gaze": "High" if gaze_high else "Low",
                    "pupil": "Dilated" if pupil_dilated else "Normal",
                    "eda": "High" if eda_high else "Low",
                    "trust": "High" if trust_high else "Low",
                    "choice": "Accept" if accept else "Reject",
                }
            )
    return pd.DataFrame(rows)
