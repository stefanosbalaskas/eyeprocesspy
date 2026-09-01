"""Frozen R/094 paper-ready validation tables and evidence atlas.

Source reference:
``R/094-validation-paper-tables-atlas-0-9.R`` from eyeprocess 0.11.1.

This module closes the final top-level API gap in the frozen 1,182-export
reference. It ports paper-ready evidence tables, evidence artifact indexing,
the validation atlas, gap analysis, freezing/integrity verification, and the
compact Markdown validation report.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError
from .negative_controls_09 import summarise_process_negative_controls
from .reproducibility_provenance_09 import object_hash

__all__ = [
    "eyeprocess_irt_engine_evidence_table",
    "eyeprocess_irt_precision_evidence_table",
    "eyeprocess_negative_control_evidence_table",
    "eyeprocess_recovery_evidence_table",
    "eyeprocess_reliability_evidence_table",
    "eyeprocess_sbc_evidence_table",
    "eyeprocess_stress_evidence_table",
    "eyeprocess_validation_atlas_gaps",
    "eyeprocess_validation_evidence_atlas",
    "eyeprocess_validation_evidence_index",
    "freeze_eyeprocess_validation_atlas",
    "verify_eyeprocess_validation_atlas",
    "write_eyeprocess_validation_report",
]

_ATLAS_CLASS = "eye_validation_evidence_atlas"
_FREEZE_CLASS = "eye_validation_atlas_freeze"
_R_REFERENCE_VERSION = "0.11.1"

_COMPONENTS = (
    "recovery",
    "sbc",
    "stress",
    "reliability",
    "negative_controls",
    "irt",
    "provenance",
    "artifacts",
)
_RESOLVED_CLAIM_STATUS = {
    "supported",
    "qualified",
    "demonstrated",
}
_RECOVERY_COLUMNS = (
    "scenario_id",
    "parameter",
    "n",
    "bias",
    "rmse",
    "mae",
    "estimate_sd",
    "coverage",
    "failure_rate",
)


def _class_name(value: Any) -> str | None:
    if isinstance(value, pd.DataFrame):
        tagged = value.attrs.get("eyeprocess_class")
        if tagged is not None:
            return str(tagged)
    tagged = getattr(value, "eyeprocess_class", None)
    if tagged is not None:
        return str(tagged)
    if isinstance(value, Mapping):
        tagged = value.get("eyeprocess_class")
        if tagged is not None:
            return str(tagged)
    return None


def _class_is(value: Any, expected: str) -> bool:
    return _class_name(value) == expected


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
    columns: tuple[str, ...] | list[str],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EyeProcessValidationError(f"{name} is missing required columns: " + ", ".join(missing))


def _round_numeric(
    frame: pd.DataFrame,
    digits: int = 4,
) -> pd.DataFrame:
    try:
        digits0 = int(digits)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("digits must be an integer.") from exc

    out = frame.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    if len(numeric):
        out.loc[:, numeric] = out.loc[:, numeric].round(digits0)
    return out


def eyeprocess_recovery_evidence_table(
    x,
    digits=4,
):
    """Build the frozen paper-ready parameter-recovery table."""
    if _class_is(x, "eye_irt_recovery_result"):
        from .irt import eyeprocess_irt_recovery_summary

        table = eyeprocess_irt_recovery_summary(x)
    else:
        table = _as_frame(x, name="x")

    selected = [column for column in _RECOVERY_COLUMNS if column in table.columns]
    if not selected:
        raise EyeProcessValidationError("No recognized recovery columns were found.")
    return _round_numeric(
        table.loc[:, selected],
        digits,
    )


def eyeprocess_sbc_evidence_table(
    x,
    digits=4,
):
    """Build the frozen paper-ready simulation-calibration table."""
    if _class_is(x, "eye_sbc_diagnostics"):
        from .validation_extras_09 import sbc_ecdf_deviation

        ranks = x.get("ranks", [])
        table = pd.DataFrame(
            {
                "n_ranks": [len(ranks)],
                "n_draws": [x.get("n_draws")],
                "ecdf_max_deviation": [sbc_ecdf_deviation(x)],
            }
        )
    elif _class_is(x, "eye_irt_sbc_evidence"):
        record = {
            "n_ranks": x.get("n"),
            "n_draws": x.get("n_draws"),
            "ecdf_max_deviation": x.get("ecdf_deviation"),
        }
        for name in (
            "coverage",
            "nominal_coverage",
            "coverage_error",
        ):
            if x.get(name) is not None:
                record[name] = x.get(name)
        table = pd.DataFrame([record])
    else:
        table = _as_frame(x, name="x")

    return _round_numeric(table, digits)


def eyeprocess_stress_evidence_table(
    x,
    digits=4,
):
    """Build the frozen paper-ready stress-test table."""
    if _class_is(x, "eye_stress_test_summary"):
        table = _as_frame(x.get("table"), name="x$table")
    else:
        table = _as_frame(x, name="x")
    return _round_numeric(table, digits)


def eyeprocess_reliability_evidence_table(
    x,
    digits=4,
):
    """Build the frozen paper-ready reliability table."""
    if _class_is(x, "eye_process_reliability_profile"):
        icc = x.get("icc")
        if isinstance(icc, Mapping):
            icc_a1 = icc.get("icc_a1", np.nan)
        else:
            icc_a1 = getattr(icc, "icc_a1", np.nan)

        record = {
            "measure": x.get("measure"),
            "icc_a1": icc_a1,
        }
        if x.get("temporal") is not None:
            record["temporal_pairs"] = len(
                _as_frame(
                    x.get("temporal"),
                    name="x$temporal",
                )
            )
        table = pd.DataFrame([record])
    else:
        table = _as_frame(x, name="x")

    return _round_numeric(table, digits)


def eyeprocess_negative_control_evidence_table(
    x,
    digits=4,
):
    """Build the frozen paper-ready negative-control table."""
    if _class_is(x, "eye_process_negative_controls"):
        table = summarise_process_negative_controls(x)
    else:
        table = _as_frame(x, name="x")
    return _round_numeric(table, digits)


def eyeprocess_irt_precision_evidence_table(
    items,
    theta=None,
    digits=4,
):
    """Build the frozen IRT information/precision evidence table."""
    from .irt import eyeprocess_irt_test_information

    if theta is None:
        theta = np.arange(-3.0, 3.0 + 0.25, 0.5)
    table = eyeprocess_irt_test_information(theta, items)
    return _round_numeric(
        _as_frame(table, name="IRT information"),
        digits,
    )


def eyeprocess_irt_engine_evidence_table():
    """Build the frozen external-engine capability table."""
    from .irt import eyeprocess_irt_engine_registry

    table = _as_frame(
        eyeprocess_irt_engine_registry(),
        name="IRT engine registry",
    )
    _require_columns(
        table,
        ["available"],
        name="IRT engine registry",
    )
    table["policy"] = np.where(
        table["available"].astype(bool),
        "exact engine available",
        "gated; no substitute estimator",
    )
    return table


def _artifact_role(relative_path: str) -> str:
    if re.search(
        r"figure|plot",
        relative_path,
        flags=re.IGNORECASE,
    ):
        return "figure"
    if re.search(
        r"table|summary|result",
        relative_path,
        flags=re.IGNORECASE,
    ):
        return "table"
    if re.search(
        r"manifest|hash|provenance",
        relative_path,
        flags=re.IGNORECASE,
    ):
        return "provenance"
    return "artifact"


def eyeprocess_validation_evidence_index(
    root,
    recursive=True,
):
    """Create the frozen index over validation evidence artifacts."""
    root_path = Path(root)
    if not root_path.exists():
        raise EyeProcessValidationError("root must identify an existing path.")
    root_path = root_path.resolve()

    iterator = root_path.rglob("*") if bool(recursive) else root_path.glob("*")
    paths = sorted(
        (path.resolve() for path in iterator if path.is_file()),
        key=lambda path: path.as_posix(),
    )

    rows = []
    for path in paths:
        relative = path.relative_to(root_path).as_posix()
        suffix = path.suffix.lower().lstrip(".")
        digest = hashlib.md5(
            path.read_bytes(),
            usedforsecurity=False,
        ).hexdigest()
        rows.append(
            {
                "path": relative,
                "extension": suffix,
                "role": _artifact_role(relative),
                "bytes": path.stat().st_size,
                "hash": digest,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "path",
            "extension",
            "role",
            "bytes",
            "hash",
        ],
    )


def eyeprocess_validation_evidence_atlas(
    claims,
    recovery=None,
    sbc=None,
    stress=None,
    reliability=None,
    negative_controls=None,
    irt=None,
    provenance=None,
    artifacts=None,
):
    """Assemble the frozen organizational validation evidence atlas."""
    claims_frame = _as_frame(claims, name="claims")
    _require_columns(
        claims_frame,
        ["claim_id", "claim", "status"],
        name="claims",
    )

    components = {
        "recovery": recovery,
        "sbc": sbc,
        "stress": stress,
        "reliability": reliability,
        "negative_controls": negative_controls,
        "irt": irt,
        "provenance": provenance,
        "artifacts": artifacts,
    }
    present = {name: value is not None for name, value in components.items()}
    status = pd.DataFrame(
        {
            "component": list(_COMPONENTS),
            "present": [present[name] for name in _COMPONENTS],
        }
    )
    payload = {
        "claims": claims_frame,
        "components": components,
    }

    return {
        "claims": claims_frame,
        "components": components,
        "component_status": status,
        "coverage": float(np.mean(status["present"].to_numpy(dtype=bool))),
        "hash": object_hash(payload),
        "guardrail": (
            "The atlas indexes software-validation evidence and does not establish substantive construct validity."
        ),
        "eyeprocess_class": _ATLAS_CLASS,
    }


def eyeprocess_validation_atlas_gaps(atlas):
    """Summarize missing components and unresolved claims."""
    if not _class_is(atlas, _ATLAS_CLASS):
        raise EyeProcessValidationError("atlas must be created by eyeprocess_validation_evidence_atlas().")

    component_status = _as_frame(
        atlas["component_status"],
        name="atlas$component_status",
    )
    claims = _as_frame(
        atlas["claims"],
        name="atlas$claims",
    )
    _require_columns(
        component_status,
        ["component", "present"],
        name="atlas$component_status",
    )
    _require_columns(
        claims,
        ["status"],
        name="atlas$claims",
    )

    missing = (
        component_status.loc[
            ~component_status["present"].astype(bool),
            "component",
        ]
        .astype(str)
        .tolist()
    )

    unresolved = claims.loc[~claims["status"].astype(str).isin(_RESOLVED_CLAIM_STATUS)].copy()

    return {
        "missing_components": missing,
        "unresolved_claims": unresolved,
        "complete": not missing and unresolved.empty,
    }


def freeze_eyeprocess_validation_atlas(
    atlas,
    metadata=None,
):
    """Freeze an evidence atlas with a deterministic integrity hash."""
    if not _class_is(atlas, _ATLAS_CLASS):
        raise EyeProcessValidationError("atlas must be an eye_validation_evidence_atlas.")

    payload = {
        "atlas": atlas,
        "metadata": {} if metadata is None else metadata,
        "version": _R_REFERENCE_VERSION,
    }
    return {
        "payload": payload,
        "hash": object_hash(payload),
        "frozen": True,
        "eyeprocess_class": _FREEZE_CLASS,
    }


def verify_eyeprocess_validation_atlas(x):
    """Verify a frozen validation atlas integrity hash."""
    if not _class_is(x, _FREEZE_CLASS):
        raise EyeProcessValidationError("x must be an eye_validation_atlas_freeze.")
    return x.get("hash") == object_hash(x.get("payload"))


def write_eyeprocess_validation_report(
    atlas,
    path,
    title="eyeprocess validation evidence report",
):
    """Write the frozen compact Markdown validation report."""
    if not _class_is(atlas, _ATLAS_CLASS):
        raise EyeProcessValidationError("atlas must be an eye_validation_evidence_atlas.")

    gaps = eyeprocess_validation_atlas_gaps(atlas)
    component_status = atlas["component_status"]
    claims = atlas["claims"]

    lines = [
        f"# {title}",
        "",
        f"Atlas hash: `{atlas['hash']}`",
        "",
        "## Evidence components",
        "",
    ]
    lines.extend(
        (f"- {row.component}: " + ("present" if bool(row.present) else "missing"))
        for row in component_status.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Claim status",
            "",
        ]
    )
    lines.extend(
        (f"- `{row.claim_id}` - {row.status}: {row.claim}")
        for row in claims[["claim_id", "status", "claim"]].itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            atlas["guardrail"],
            "",
            (
                "Missing components: "
                + (", ".join(gaps["missing_components"]) if gaps["missing_components"] else "none")
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
    return output.resolve().as_posix()
