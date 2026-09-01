"""Frozen R/079 software-paper evidence contracts.

Source reference:
``R/079-software-paper-evidence-0-9.R`` from eyeprocess 0.11.1.

The R source freezes evidence with native RDS. eyeprocesspy does not disguise a
different serialization format as RDS: JSON freezing is provided as the
Python-native analogue and ``.rds`` output is explicitly gated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .reproducibility_provenance_09 import (
    file_hash_manifest,
    object_hash,
)

__all__ = [
    "freeze_software_paper_evidence",
    "paper_reproducibility_manifest",
    "software_paper_claim_matrix",
    "software_paper_coverage",
    "software_paper_evidence_bundle",
    "software_paper_gap_analysis",
    "software_paper_readiness",
    "software_paper_validation_table",
    "write_software_paper_evidence",
]

_BUNDLE_CLASS = "eye_software_paper_evidence"
_READINESS_CLASS = "eye_software_paper_readiness"
_ALLOWED_STATUSES = {
    "supported",
    "qualified",
    "pending",
    "unsupported",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _tag(value: dict[str, Any], class_name: str) -> dict[str, Any]:
    value["eyeprocess_class"] = class_name
    return value


def _class_is(value: Any, class_name: str) -> bool:
    if isinstance(value, Mapping):
        return value.get("eyeprocess_class") == class_name
    return getattr(value, "eyeprocess_class", None) == class_name


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Path)):
        return [value]
    if isinstance(value, pd.Series):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _recycle(value: Any, n: int) -> list[Any]:
    values = _as_list(value)
    if not values:
        values = [None]
    return [values[index % len(values)] for index in range(n)]


def _as_frame(value: Any, *, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError(f"{name} must be coercible to a data frame.") from exc


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required column(s): " + ", ".join(missing) + ".")


def _length_nonzero(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return len(value) > 0
    if isinstance(value, Mapping):
        return len(value) > 0
    if isinstance(value, (str, bytes)):
        return len(value) > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [{str(key): _json_safe(item) for key, item in row.items()} for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def software_paper_evidence_bundle(
    claims=None,
    validation=None,
    examples=None,
    articles=None,
    benchmarks=None,
    reproducibility=None,
    metadata=None,
):
    """Construct a software-paper evidence bundle."""
    core = {
        "schema_version": "eyeprocess-paper-evidence-0.9",
        "claims": claims,
        "validation": validation,
        "examples": examples,
        "articles": articles,
        "benchmarks": benchmarks,
        "reproducibility": reproducibility,
        "metadata": {} if metadata is None else metadata,
        "created_utc": _utc_now(),
    }
    core["bundle_hash"] = object_hash(core)
    return _tag(core, _BUNDLE_CLASS)


def software_paper_claim_matrix(
    claim,
    evidence_id=None,
    evidence_type=None,
    status="pending",
    scope=None,
    source=None,
):
    """Create or normalize a software-paper claim matrix."""
    claims = _as_list(claim)
    if not claims or any(value is None or not str(value) for value in claims):
        raise EyeProcessValidationError("claim must contain non-empty claim text.")

    candidates = [
        claims,
        _as_list(evidence_id) or [None],
        _as_list(evidence_type) or [None],
        _as_list(status) or [None],
        _as_list(scope) or [None],
        _as_list(source) or [None],
    ]
    n = max(len(values) for values in candidates)

    statuses = [None if value is None else str(value) for value in _recycle(status, n)]
    if any(value is None or value not in _ALLOWED_STATUSES for value in statuses):
        raise EyeProcessValidationError("status must be one of supported, qualified, pending, or unsupported.")

    return pd.DataFrame(
        {
            "claim_id": [f"CL{index:03d}" for index in range(1, n + 1)],
            "claim": [str(value) for value in _recycle(claims, n)],
            "evidence_id": [None if value is None else str(value) for value in _recycle(evidence_id, n)],
            "evidence_type": [None if value is None else str(value) for value in _recycle(evidence_type, n)],
            "status": statuses,
            "scope": [None if value is None else str(value) for value in _recycle(scope, n)],
            "source": [None if value is None else str(value) for value in _recycle(source, n)],
        }
    )


def software_paper_validation_table(x):
    """Summarise validation evidence for a software paper."""
    if _class_is(x, "eye_process_validation_result"):
        from .governance_09 import summarise_process_validation

        return summarise_process_validation(x)

    if isinstance(x, Mapping) and not isinstance(x, pd.DataFrame):
        if not x or any(not str(name) for name in x):
            raise EyeProcessValidationError("List validation evidence must be named.")
        from .governance_09 import validation_evidence_matrix

        return validation_evidence_matrix(**dict(x))

    return _as_frame(x, name="x")


def _claims_from(value: Any) -> pd.DataFrame:
    claims = value.get("claims") if _class_is(value, _BUNDLE_CLASS) else value
    frame = _as_frame(claims, name="claims")
    _require_columns(
        frame,
        ["status"],
        name="claims",
    )
    return frame


def software_paper_coverage(
    x,
    supported=("supported", "qualified"),
):
    """Compute descriptive evidence coverage."""
    claims = _claims_from(x)
    supported_values = {str(value) for value in _as_list(supported) if value is not None}

    status = claims["status"]
    covered = status.isin(supported_values)
    n_claims = len(claims)
    coverage = float(covered.mean()) if n_claims else np.nan

    return pd.DataFrame(
        {
            "n_claims": [n_claims],
            "n_covered": [int(covered.sum())],
            "coverage": [coverage],
            "n_pending": [int((status == "pending").sum())],
            "n_unsupported": [int((status == "unsupported").sum())],
        }
    )


def software_paper_readiness(
    x,
    required_statuses=("supported", "qualified"),
    require_validation=True,
    require_reproducibility=True,
    require_examples=True,
    require_articles=True,
):
    """Run the frozen descriptive readiness completeness audit."""
    if not _class_is(x, _BUNDLE_CLASS):
        raise EyeProcessValidationError("x must be an eye_software_paper_evidence object.")

    try:
        claims = _as_frame(
            x.get("claims"),
            name="claims",
        )
    except EyeProcessValidationError:
        claims = pd.DataFrame()

    required_values = [str(value) for value in _as_list(required_statuses) if value is not None and str(value)]
    if not required_values:
        raise EyeProcessValidationError("required_statuses cannot be empty.")

    claim_ok = (
        len(claims) > 0
        and "status" in claims.columns
        and not claims["status"].isna().any()
        and claims["status"].isin(required_values).all()
    )

    reproducibility = x.get("reproducibility")
    checks = pd.DataFrame(
        {
            "requirement": [
                "claims",
                "validation",
                "reproducibility",
                "examples",
                "articles",
            ],
            "required": [
                True,
                bool(require_validation),
                bool(require_reproducibility),
                bool(require_examples),
                bool(require_articles),
            ],
            "satisfied": [
                bool(claim_ok),
                _length_nonzero(x.get("validation")),
                _class_is(
                    reproducibility,
                    "eye_reproducibility_fingerprint",
                ),
                _length_nonzero(x.get("examples")),
                _length_nonzero(x.get("articles")),
            ],
        }
    )

    required_mask = checks["required"].astype(bool)
    ready = bool(checks.loc[required_mask, "satisfied"].all())

    return _tag(
        {
            "ready": ready,
            "checks": checks,
            "interpretation": (
                "Readiness is a completeness audit against explicitly "
                "declared evidence requirements, not a prediction of "
                "peer-review outcome."
            ),
        },
        _READINESS_CLASS,
    )


def software_paper_gap_analysis(x):
    """Identify gaps in a software-paper evidence bundle."""
    readiness = software_paper_readiness(x)
    checks = readiness["checks"]
    requirement_gaps = checks.loc[checks["required"].astype(bool) & ~checks["satisfied"].astype(bool)].copy()

    try:
        claims = _as_frame(
            x.get("claims"),
            name="claims",
        )
    except EyeProcessValidationError:
        claims = pd.DataFrame()

    if len(claims) and "status" in claims.columns:
        claim_gaps = claims.loc[~claims["status"].isin(["supported", "qualified"])].copy()
    else:
        claim_gaps = pd.DataFrame()

    return {
        "requirement_gaps": requirement_gaps,
        "claim_gaps": claim_gaps,
    }


def freeze_software_paper_evidence(x, path):
    """Freeze evidence as Python-native JSON and return its MD5 manifest."""
    if not _class_is(x, _BUNDLE_CLASS):
        raise EyeProcessValidationError("x must be an eye_software_paper_evidence object.")

    output = Path(path)
    if output.suffix.lower() == ".rds":
        raise EyeProcessValidationError(
            "Native RDS serialization is R-specific and is intentionally "
            "not emulated by eyeprocesspy. Use a .json output path."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            _json_safe(x),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = hashlib.md5(
        output.read_bytes(),
        usedforsecurity=False,
    ).hexdigest()

    return pd.DataFrame(
        {
            "path": [output.resolve().as_posix()],
            "hash": [digest],
        }
    )


def _format_coverage(value: Any) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.4f}"


def write_software_paper_evidence(x, path):
    """Write the frozen human-readable evidence audit report."""
    if not _class_is(x, _BUNDLE_CLASS):
        raise EyeProcessValidationError("x must be an eye_software_paper_evidence object.")

    readiness = software_paper_readiness(x)
    try:
        coverage = software_paper_coverage(x)
    except EyeProcessValidationError:
        coverage = None

    lines = [
        "# eyeprocess software-paper evidence bundle",
        "",
        f"Bundle hash: `{x['bundle_hash']}`",
        "",
        ("Descriptive readiness: **" + ("PASS" if readiness["ready"] else "INCOMPLETE") + "**"),
        "",
        "## Requirement audit",
        "",
    ]

    for _, row in readiness["checks"].iterrows():
        lines.append(f"- {row['requirement']}: required={bool(row['required'])}, satisfied={bool(row['satisfied'])}")

    if coverage is not None:
        row = coverage.iloc[0]
        lines.extend(
            [
                "",
                "## Claim coverage",
                "",
                f"- Claims: {int(row['n_claims'])}",
                f"- Covered: {int(row['n_covered'])}",
                ("- Coverage: " + _format_coverage(row["coverage"])),
            ]
        )

    lines.extend(
        [
            "",
            (
                "> This report audits supplied evidence; it does not "
                "establish external validity or predict journal acceptance."
            ),
        ]
    )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output.as_posix()


def paper_reproducibility_manifest(
    evidence,
    manuscript=None,
    figures=None,
    tables=None,
):
    """Create the compact frozen paper reproducibility manifest."""
    if not _class_is(evidence, _BUNDLE_CLASS):
        raise EyeProcessValidationError("evidence must be an eye_software_paper_evidence object.")

    paths = []
    for value in _as_list(manuscript) + _as_list(figures) + _as_list(tables):
        if value is None or not str(value):
            continue
        paths.append(value)

    missing = [str(value) for value in paths if not Path(value).exists()]
    if missing:
        raise EyeProcessValidationError("All supplied paper files must exist.")

    return {
        "evidence_hash": evidence["bundle_hash"],
        "manuscript": manuscript,
        "files": (pd.DataFrame() if not paths else file_hash_manifest(paths)),
        "reproducibility": evidence.get("reproducibility"),
        "generated_utc": _utc_now(),
    }
