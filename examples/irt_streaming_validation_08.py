"""Operational streaming-score and validation-bundle example."""
from pathlib import Path
import tempfile
import pandas as pd
import eyeprocesspy as ep
from eyeprocesspy.exceptions import EyeProcessBackendError

# The exact partial score is intentionally gated without the frozen R mirt engine.
try:
    ep.score_partial_response_pattern(object(), [1, 0, 1, None], method="MAP")
except EyeProcessBackendError:
    pass

stream = ep.score_response_stream(object(), [1, 0, 1, 1], method="MAP")
assert len(ep.streaming_score_history(stream)) == 4

bundle = ep.collect_validation_evidence(
    recovery=pd.DataFrame({"parameter": ["a"], "bias": [0.01]}),
    convergence=pd.DataFrame({"rate": [0.99]}),
    model_name="demo_model",
)
assert "available" in ep.validation_bundle_manifest(bundle).status.values
assert any("Convergence is not validation" in x for x in ep.validation_report(bundle, include_session=False))

with tempfile.TemporaryDirectory() as td:
    exported = ep.export_validation_bundle(bundle, Path(td) / "validation")
    assert Path(exported.files["manifest"]).exists()
