"""Explicit research-only gaze reference and calibration provenance.

No free-viewing data can be relabelled absolute accuracy by metadata alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math


def declare_gaze_validation_provenance(
    *,
    reference_type: str,
    coordinate_space: str,
    reference_identifier: str,
    angular_range_deg: float | None = None,
    accuracy_deg: float | None = None,
    precision_deg: float | None = None,
    offset_correction_applied: bool = False,
    target_measurements_available: bool = False,
) -> Mapping[str, object]:
    """Describe reference evidence separately from reported precision and accuracy."""
    references = {"human_validation_target", "artificial_eye", "trajectory_relative", "unknown"}
    spaces = {"absolute_gaze", "trajectory_relative", "uncalibrated"}
    if reference_type not in references or coordinate_space not in spaces:
        raise ValueError("reference_type and coordinate_space must be valid declared categories")
    if not isinstance(reference_identifier, str) or not reference_identifier.strip():
        raise ValueError("a nonempty reference_identifier is required")
    for name, number in (
        ("accuracy_deg", accuracy_deg),
        ("precision_deg", precision_deg),
        ("angular_range_deg", angular_range_deg),
    ):
        if number is not None and (not math.isfinite(float(number)) or float(number) < 0):
            raise ValueError(f"{name} must be finite and nonnegative when supplied")
    if coordinate_space == "absolute_gaze":
        if reference_type not in {"human_validation_target", "artificial_eye"}:
            raise ValueError("absolute gaze claims require a physical target/eye reference")
        if not target_measurements_available or accuracy_deg is None:
            raise ValueError("absolute gaze accuracy requires target measurement evidence")
    if reference_type == "trajectory_relative" and accuracy_deg is not None:
        raise ValueError("relative trajectory evidence is not absolute gaze accuracy")
    value = {
        "reference_type": reference_type,
        "coordinate_space": coordinate_space,
        "reference_identifier": reference_identifier.strip(),
        "angular_range_deg": angular_range_deg,
        "accuracy_deg": accuracy_deg,
        "precision_deg": precision_deg,
        "offset_correction_applied": bool(offset_correction_applied),
        "target_measurements_available": bool(target_measurements_available),
        "evidence_stage": "reference_declaration_not_external_validation",
    }
    value["sha256"] = sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    return value
