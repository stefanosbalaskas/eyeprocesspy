from __future__ import annotations

import eyeprocesspy as ep


def test_eyeprocess_irt_model_spec_validation_contract() -> None:
    spec = ep.eyeprocess_irt_model_spec(
        family="2pl",
        dimensions=1,
        identification="theta_standard",
        engine="native_math",
    )
    assert ep.validate_eyeprocess_irt_model_spec(spec) is True
