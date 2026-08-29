"""Semantic-fidelity and validation-evidence example."""
import pandas as pd
import eyeprocesspy as ep

samples = pd.DataFrame({
    "id": [1, 2, 3],
    "timestamp": [0.0, 1.0, 2.0],
    "x": [0.1, 0.2, 0.3],
    "y": [0.2, 0.3, 0.4],
})

contract = ep.vendor_schema_contract(
    "Gazepoint",
    required_fields=["timestamp", "x", "y"],
    timestamp={"device_time": "timestamp"},
)
validation = ep.validate_vendor_semantics(samples, contract)
assert validation["pass"]

audit = ep.semantic_roundtrip_audit(
    samples,
    samples.copy(),
    key="id",
    fields=["timestamp", "x", "y"],
)
assert audit.overall == "LOSSLESS"
loss = ep.semantic_loss_map(audit)
assert set(loss["status"]) == {"LOSSLESS"}
