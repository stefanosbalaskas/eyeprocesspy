"""Frozen R API lifecycle registry and Python implementation inventory."""
import eyeprocesspy as ep

registry = ep.eye_api_lifecycle()
reference_inventory = ep.eye_api_inventory()
python_inventory = ep.eye_api_inventory(package="eyeprocesspy")
audit = ep.audit_eye_api(reference_inventory, registry)
assert len(registry) == 1182
assert len(reference_inventory) == 1182
assert audit.valid is True
assert len(python_inventory) > 0
