import eyeprocesspy as ep


def test_measurement_accountability_public_api():
    assert callable(ep.pupil_latency_sensitivity)
    assert callable(ep.event_marker_qc)
    assert callable(ep.validation_ladder)
