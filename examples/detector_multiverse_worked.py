"""CI-sized worked example: detector choice -> AOI features -> inference."""
from pathlib import Path

import eyeprocesspy as ep


def build_specs():
    specs = [
        ep.define_event_detector_spec(
            "ivt25", "ivt", velocity_threshold=25, minimum_duration_ms=60,
            maximum_gap_ms=75, sampling_rate=60, coordinate_unit="degrees",
        ),
        ep.define_event_detector_spec(
            "ivt30", "ivt", velocity_threshold=30, minimum_duration_ms=60,
            maximum_gap_ms=75, sampling_rate=60, coordinate_unit="degrees",
        ),
        ep.define_event_detector_spec(
            "ivt35", "ivt", velocity_threshold=35, minimum_duration_ms=60,
            maximum_gap_ms=75, sampling_rate=60, coordinate_unit="degrees",
        ),
        ep.define_event_detector_spec(
            "idt_A", "idt", dispersion_threshold=1.2, minimum_duration_ms=80,
            sampling_rate=60, coordinate_unit="degrees",
        ),
        ep.define_event_detector_spec(
            "adaptive", "adaptive_velocity", minimum_duration_ms=60,
            maximum_gap_ms=75, sampling_rate=60, coordinate_unit="degrees",
            parameters={"noise_factor": 4, "minimum_velocity_threshold": 20},
        ),
        ep.define_event_detector_spec(
            "remodnav", "remodnav", minimum_duration_ms=60,
            sampling_rate=60, coordinate_unit="degrees",
            parameters={"noise_factor": 5},
        ),
    ]
    return ep.create_detector_multiverse(specs, label="disclosure_detector_multiverse")


def main(output_dir=None, n_participants=8):
    data = ep.simulate_detector_multiverse_data(
        n_participants=n_participants,
        sampling_rate=60,
        seed=20260918,
    )
    multiverse = build_specs()
    result = ep.run_detector_multiverse(data, multiverse, continue_on_error=True)

    # REMoDNaV is an optional bridge. If it is absent, the branch is recorded as
    # failed; no substitute detector is silently used.
    result = ep.propagate_detector_to_aoi(result, overlap="error")
    result = ep.propagate_detector_to_features(result)

    model_spec = {
        "engine": "statsmodels_ols",
        "formula": "dwell_time_ms ~ C(condition_id) + C(participant_id)",
        "outcome": "dwell_time_ms",
        "aoi_id": "disclosure",
    }
    inference = ep.run_detector_inference_multiverse(
        result,
        model_spec,
        minimum_valid_fraction=0.5,
    )
    condition_terms = inference.coefficients[
        inference.coefficients["term"].astype(str).str.contains("condition_id", na=False)
    ]
    term = condition_terms["term"].iloc[0] if not condition_terms.empty else None

    summaries = ep.summarise_detector_robustness(
        result,
        inference=inference if term is not None else None,
        term=term,
        substantive_threshold=100 if term is not None else None,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ep.plot_detector_agreement(result).figure.savefig(output_dir / "detector-agreement.svg", bbox_inches="tight")
        ep.plot_detector_feature_distributions(
            result, feature="dwell_time_ms", aoi_id="disclosure"
        ).figure.savefig(output_dir / "disclosure-dwell-by-detector.svg", bbox_inches="tight")
        if term is not None:
            ep.plot_detector_coefficient_stability(inference, term=term).figure.savefig(
                output_dir / "condition-coefficient-stability.svg", bbox_inches="tight"
            )
        ep.report_detector_multiverse(
            result,
            inference=inference if term is not None else None,
            term=term,
            substantive_threshold=100 if term is not None else None,
            path=str(output_dir / "detector-multiverse-report.md"),
        )

    return data, result, inference, summaries


if __name__ == "__main__":
    _, result, inference, summaries = main()
    print(result.status.to_string(index=False))
    print("\nEvent summary\n", summaries["event_summary"].to_string(index=False))
    print("\nFeature sensitivity\n", summaries["feature_sensitivity"].to_string(index=False))
    if not summaries["inference_stability"].empty:
        print("\nInference stability\n", summaries["inference_stability"].to_string(index=False))
