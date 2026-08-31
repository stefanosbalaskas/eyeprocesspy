"""Independent multi-vendor validation corpus.

Source-port of the 13 public non-S3 contracts in frozen
``R/025-vendor-corpus.R`` from eyeprocess 0.11.1.

The implementation deliberately separates declared support, fixture-tested
support, and independently empirically validated support.  Production
compatibility claims require the requested number of independent, licence-
reviewed, validated real-export cases.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import is_eye_dataset
from .exceptions import EyeProcessValidationError
from .schema import canonical_table_names
from .validation_orchestration_10 import _hash_int, _now_utc, _safe_name

__all__ = [
    "audit_roundtrip_loss",
    "audit_vendor_field_coverage",
    "build_compatibility_matrix",
    "compare_vendor_semantics",
    "fingerprint_validation_case",
    "init_vendor_corpus",
    "promote_vendor_support",
    "read_vendor_registry",
    "redact_validation_case",
    "register_validation_case",
    "register_vendor_semantics",
    "write_vendor_case_report",
    "write_vendor_registry",
]

REGISTRY_COLUMNS = [
    "case_id",
    "vendor",
    "support_level",
    "device_model",
    "hardware_version",
    "software_name",
    "software_version",
    "export_profile",
    "sampling_rate_hz",
    "coordinate_system",
    "timebase",
    "event_semantics",
    "ocular_structure",
    "missingness_convention",
    "vendor_fixations",
    "package_transformations",
    "unsupported_fields",
    "independent_source",
    "licence_reviewed",
    "redistribution_allowed",
    "source_path",
    "registered_utc",
    "status",
    "notes",
]

SEMANTICS_COLUMNS = [
    "vendor",
    "native_field",
    "native_meaning",
    "canonical_table",
    "canonical_field",
    "unit",
    "transformation",
    "loss_risk",
    "evidence_case_id",
]

LOSS_RISK_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "unsupported": 4,
}

SUPPORT_ORDER = {
    "declared": 1,
    "fixture-tested": 2,
    "empirically-validated": 3,
}

VENDOR_ALIASES = {
    "gazepointanalysis": "gazepoint",
    "gazepointbiometrics": "gazepoint",
    "tobiiprolab": "tobii",
    "pupillabs": "pupillabs",
    "pupilneon": "pupillabs",
    "pupilcore": "pupillabs",
    "srresearch": "eyelink",
    "dataviewer": "eyelink",
    "smibegaze": "smi",
}


class _EyeFrame(pd.DataFrame):
    _metadata = ["eyeprocess_class"]

    @property
    def _constructor(self):
        return _EyeFrame


class _EyeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class EyeRedactionResult(_EyeDict):
    eyeprocess_class = "eye_redaction_result"


class EyeRoundtripLossAudit(_EyeDict):
    eyeprocess_class = "eye_roundtrip_loss_audit"


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _tag_frame(frame: pd.DataFrame, class_name: str) -> _EyeFrame:
    out = _EyeFrame(frame.copy())
    out.eyeprocess_class = class_name
    return out


def _canonical_path(path: Path) -> str:
    return path.resolve().as_posix()


def _vendor(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else ""
    text = "".join(character for character in text if character.isalnum())
    if not text:
        _stop("Vendor must be non-empty.")
    return VENDOR_ALIASES.get(text, text)


def _match_arg(value: Any, choices: Sequence[str], name: str) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    text = str(value) if value is not None else ""
    if text not in choices:
        _stop(f"`{name}` must be one of: {', '.join(choices)}.")
    return text


def _r_character(value: Any) -> Any:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if bool(pd.isna(value)):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"
    return str(value)


def _registry_path(corpus_path: Path) -> Path:
    return corpus_path / "vendor-cases.csv"


def _semantics_path(corpus_path: Path) -> Path:
    return corpus_path / "vendor-semantics.csv"


def _empty_registry() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in REGISTRY_COLUMNS})


def _empty_semantics() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in SEMANTICS_COLUMNS})


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".csv",
        dir=path.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, na_rep="")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return _canonical_path(path)


def _write_delimited(frame: pd.DataFrame, path: Path, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep=delimiter, na_rep="")


def _read_delimited(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = [handle.readline() for _ in range(3)]
    tab_count = sum(line.count("\t") for line in first)
    comma_count = sum(line.count(",") for line in first)
    delimiter = "\t" if tab_count > comma_count else ","
    return pd.read_csv(path, sep=delimiter, dtype_backend="numpy_nullable")


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
    }
    return series.map(lambda value: pd.NA if pd.isna(value) else mapping.get(str(value).strip().lower(), pd.NA)).astype(
        "boolean"
    )


def init_vendor_corpus(path, overwrite=False):
    """Create or validate a multi-vendor validation-corpus directory."""
    corpus = Path(path).expanduser().resolve()
    registry = _registry_path(corpus)
    if corpus.is_dir() and not overwrite and registry.exists():
        return _canonical_path(corpus)
    if corpus.exists() and overwrite:
        if corpus.is_dir():
            shutil.rmtree(corpus)
        else:
            corpus.unlink()
    corpus.mkdir(parents=True, exist_ok=True)
    for child in ["cases", "fingerprints", "reports", "redacted"]:
        (corpus / child).mkdir(exist_ok=True)
    _atomic_write_csv(_empty_registry(), registry)
    _atomic_write_csv(_empty_semantics(), _semantics_path(corpus))
    (corpus / "README.md").write_text(
        "# eyeprocess independent multi-vendor validation corpus\n\n"
        "This directory records declared, fixture-tested, and empirically "
        "validated support separately.\n"
        "No real-export case is classified as empirically validated without "
        "independent source evidence, version/device metadata, completed "
        "licensing review, and successful validation artifacts.\n",
        encoding="utf-8",
        newline="\n",
    )
    return _canonical_path(corpus)


def read_vendor_registry(corpus_path):
    """Read the version-specific multi-vendor case registry."""
    corpus = Path(corpus_path).expanduser().resolve()
    path = _registry_path(corpus)
    if not path.exists():
        _stop(f"Vendor registry is missing: {path}")
    try:
        frame = pd.read_csv(path, dtype_backend="numpy_nullable")
    except pd.errors.EmptyDataError:
        frame = _empty_registry()
    for column in REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame.loc[:, REGISTRY_COLUMNS]
    frame["sampling_rate_hz"] = pd.to_numeric(
        frame["sampling_rate_hz"],
        errors="coerce",
    )
    for column in [
        "independent_source",
        "licence_reviewed",
        "redistribution_allowed",
    ]:
        frame[column] = _coerce_bool_series(frame[column])
    text_columns = [
        column
        for column in REGISTRY_COLUMNS
        if column
        not in {
            "sampling_rate_hz",
            "independent_source",
            "licence_reviewed",
            "redistribution_allowed",
        }
    ]
    for column in text_columns:
        frame[column] = frame[column].astype("object")
    return frame


def write_vendor_registry(x, corpus_path):
    """Write a vendor registry using the frozen canonical column order."""
    if not isinstance(x, pd.DataFrame):
        _stop("Registry must be a data frame.")
    corpus = Path(corpus_path).expanduser().resolve()
    if not corpus.is_dir():
        _stop(f"Vendor corpus does not exist: {corpus}")
    frame = x.copy()
    for column in REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame.loc[:, REGISTRY_COLUMNS]
    return _atomic_write_csv(frame, _registry_path(corpus))


def _iter_case_files(path: Path, include_hidden: bool) -> list[Path]:
    if path.is_file():
        return [path]
    files = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path)
        if not include_hidden and any(part.startswith(".") for part in relative.parts):
            continue
        files.append(candidate)
    return sorted(files, key=lambda value: value.relative_to(path).as_posix())


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_validation_case(
    path,
    algorithms=("md5", "sha256"),
    include_hidden=False,
):
    """Fingerprint every file in a real-export validation case."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        _stop(f"Validation-case path does not exist: {source}")
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    algorithms = [str(value).lower() for value in algorithms]
    unsupported = [algorithm for algorithm in algorithms if algorithm not in hashlib.algorithms_available]
    if unsupported:
        _stop("Unsupported hash algorithm(s): " + ", ".join(unsupported))
    files = _iter_case_files(source, bool(include_hidden))
    if not files:
        _stop("No files were found in the validation case.")

    rows = []
    for file in files:
        stat = file.stat()
        relative = file.relative_to(source).as_posix() if source.is_dir() else file.name
        row = {
            "relative_path": relative,
            "extension": file.suffix.lower().lstrip("."),
            "bytes": int(stat.st_size),
            "modified_utc": datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "md5": _file_digest(file, "md5"),
        }
        if "sha256" in algorithms:
            row["sha256"] = _file_digest(file, "sha256")
        rows.append(row)

    frame = pd.DataFrame(rows)
    key = "|".join(f"{row.relative_path} {int(row.bytes)} {row.md5}" for row in frame.itertuples(index=False))
    frame["case_fingerprint"] = f"case-{_hash_int(key):010d}"
    return _tag_frame(frame, "eye_validation_case_fingerprint")


def _case_id(
    vendor: str,
    device_model: Any,
    software_version: Any,
    source_path: Path,
) -> str:
    key = "|".join(
        [
            vendor,
            str(device_model),
            str(software_version),
            _canonical_path(source_path),
        ]
    )
    return f"{vendor}-{_hash_int(key):010d}"


def _copy_case(source: Path, target: Path, mode: str) -> Path:
    if mode == "reference":
        return source.resolve()
    if target.exists():
        _stop(f"Case target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, copy_function=shutil.copy2)
    else:
        shutil.copy2(source, target)
    return target.resolve()


def register_validation_case(
    corpus_path,
    source_path,
    vendor,
    device_model,
    software_name,
    software_version,
    hardware_version=pd.NA,
    export_profile=pd.NA,
    sampling_rate_hz=np.nan,
    coordinate_system=pd.NA,
    timebase=pd.NA,
    event_semantics=pd.NA,
    ocular_structure=pd.NA,
    missingness_convention=pd.NA,
    vendor_fixations=pd.NA,
    package_transformations=pd.NA,
    unsupported_fields=pd.NA,
    independent_source=True,
    licence_reviewed=False,
    redistribution_allowed=False,
    support_level=("declared", "fixture-tested", "empirically-validated"),
    mode=("reference", "copy"),
    case_id=None,
    notes=pd.NA,
):
    """Register a version-specific independent vendor validation case."""
    corpus = Path(init_vendor_corpus(corpus_path))
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        _stop(f"Validation-case source does not exist: {source}")
    vendor_name = _vendor(vendor)
    support = _match_arg(
        support_level,
        ["declared", "fixture-tested", "empirically-validated"],
        "support_level",
    )
    storage_mode = _match_arg(mode, ["reference", "copy"], "mode")

    for name, value in {
        "device_model": device_model,
        "software_name": software_name,
        "software_version": software_version,
    }.items():
        if value is None or not str(value).strip():
            _stop("Device model, software name, and software version are required.")

    sampling = pd.to_numeric(
        pd.Series([sampling_rate_hz]),
        errors="coerce",
    ).iloc[0]
    if not pd.isna(sampling) and (not math.isfinite(float(sampling)) or float(sampling) <= 0):
        _stop("`sampling_rate_hz` must be missing or a positive finite value.")

    if support == "empirically-validated" and (independent_source is not True or licence_reviewed is not True):
        _stop("Empirically validated cases require independent-source and licence-review evidence.")

    if case_id is None:
        case_id = _case_id(
            vendor_name,
            device_model,
            software_version,
            source,
        )
    case_id = _safe_name(case_id, f"{vendor_name}-case")

    registry = read_vendor_registry(corpus)
    if case_id in set(registry["case_id"].dropna().astype(str)):
        _stop(f"Case identifier already exists: {case_id}")

    stored = _copy_case(
        source,
        corpus / "cases" / case_id,
        storage_mode,
    )
    fingerprint = fingerprint_validation_case(stored)
    _atomic_write_csv(
        pd.DataFrame(fingerprint),
        corpus / "fingerprints" / f"{case_id}.csv",
    )

    row = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "vendor": vendor_name,
                "support_level": support,
                "device_model": _r_character(device_model),
                "hardware_version": _r_character(hardware_version),
                "software_name": _r_character(software_name),
                "software_version": _r_character(software_version),
                "export_profile": _r_character(export_profile),
                "sampling_rate_hz": (np.nan if pd.isna(sampling) else float(sampling)),
                "coordinate_system": _r_character(coordinate_system),
                "timebase": _r_character(timebase),
                "event_semantics": _r_character(event_semantics),
                "ocular_structure": _r_character(ocular_structure),
                "missingness_convention": _r_character(missingness_convention),
                "vendor_fixations": _r_character(vendor_fixations),
                "package_transformations": _r_character(package_transformations),
                "unsupported_fields": _r_character(unsupported_fields),
                "independent_source": bool(independent_source is True),
                "licence_reviewed": bool(licence_reviewed is True),
                "redistribution_allowed": bool(redistribution_allowed is True),
                "source_path": _canonical_path(stored),
                "registered_utc": _now_utc(),
                "status": "registered",
                "notes": _r_character(notes),
            }
        ],
        columns=REGISTRY_COLUMNS,
    )
    write_vendor_registry(
        pd.concat([registry, row], ignore_index=True, sort=False),
        corpus,
    )
    return _tag_frame(row, "eye_validation_case")


def _hash_ids(values: Iterable[Any], salt: str) -> list[str]:
    output = []
    for value in values:
        token = "NA" if pd.isna(value) else str(value)
        output.append(f"ID{_hash_int(f'{salt}|{token}'):010d}")
    return output


def redact_validation_case(
    source_path,
    output_path,
    id_columns=(
        "participant_id",
        "subject",
        "participant",
        "recording_id",
        "session_id",
    ),
    remove_columns=(
        "name",
        "email",
        "address",
        "birthdate",
        "date_of_birth",
    ),
    text_redactor: Callable[[pd.Series, str], Any] | None = None,
    salt=None,
    copy_non_tabular=False,
    overwrite=False,
):
    """Redact a validation case without inventing replacement data."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.exists():
        _stop(f"Validation-case source does not exist: {source}")
    if source.is_dir():
        try:
            output.relative_to(source)
        except ValueError:
            pass
        else:
            _stop("`output_path` must not be inside the source case directory.")
    if salt is None or not str(salt):
        _stop("A non-empty project-specific `salt` is required.")
    if output.exists() and not overwrite:
        _stop(f"Output already exists: {output}")
    if output.exists() and overwrite:
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True, exist_ok=True)

    files = _iter_case_files(source, include_hidden=True)
    rows = []
    lowered_remove = {str(value).lower() for value in remove_columns}
    lowered_ids = {str(value).lower() for value in id_columns}

    for file in files:
        relative = file.relative_to(source) if source.is_dir() else Path(file.name)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        extension = file.suffix.lower().lstrip(".")
        status = "excluded"

        if extension in {"csv", "tsv", "txt"}:
            try:
                data = _read_delimited(file)
            except Exception as exc:  # mirrors R's retained read-error status
                status = f"read-error: {exc}"
            else:
                keep = [column for column in data.columns if str(column).lower() not in lowered_remove]
                data = data.loc[:, keep]
                for column in list(data.columns):
                    if str(column).lower() in lowered_ids:
                        data[column] = _hash_ids(data[column].tolist(), str(salt))
                if callable(text_redactor):
                    for column in list(data.columns):
                        if pd.api.types.is_object_dtype(data[column].dtype) or pd.api.types.is_string_dtype(
                            data[column].dtype
                        ):
                            data[column] = text_redactor(data[column], str(column))
                delimiter = "\t" if extension == "tsv" else ","
                _write_delimited(data, target, delimiter)
                status = "redacted"
        elif copy_non_tabular:
            shutil.copy2(file, target)
            status = "copied-unchanged"

        rows.append(
            {
                "relative_path": relative.as_posix(),
                "extension": extension,
                "status": status,
            }
        )

    manifest = pd.DataFrame(
        rows,
        columns=["relative_path", "extension", "status"],
    )
    _atomic_write_csv(manifest, output / "redaction-manifest.csv")
    fingerprint = fingerprint_validation_case(output)
    return EyeRedactionResult(
        source_path=_canonical_path(source),
        output_path=_canonical_path(output),
        manifest=manifest,
        fingerprint=fingerprint,
        warning=(
            "Redaction is field-based and must be reviewed before "
            "redistribution; copied binary/media files may retain identifying "
            "information."
        ),
    )


def _read_semantics(corpus: Path) -> pd.DataFrame:
    path = _semantics_path(corpus)
    try:
        frame = pd.read_csv(path, dtype_backend="numpy_nullable")
    except pd.errors.EmptyDataError:
        frame = _empty_semantics()
    for column in SEMANTICS_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame.loc[:, SEMANTICS_COLUMNS]
    for column in SEMANTICS_COLUMNS:
        frame[column] = frame[column].astype("object")
    return frame


def register_vendor_semantics(
    corpus_path,
    vendor,
    native_field,
    native_meaning,
    canonical_table,
    canonical_field,
    unit=pd.NA,
    transformation="identity",
    loss_risk=("none", "low", "moderate", "high", "unsupported"),
    evidence_case_id=pd.NA,
):
    """Register or update a native-to-canonical vendor semantic mapping."""
    corpus = Path(init_vendor_corpus(corpus_path))
    risk = _match_arg(
        loss_risk,
        ["none", "low", "moderate", "high", "unsupported"],
        "loss_risk",
    )
    registry = _read_semantics(corpus)
    row = pd.DataFrame(
        [
            {
                "vendor": _vendor(vendor),
                "native_field": _r_character(native_field),
                "native_meaning": _r_character(native_meaning),
                "canonical_table": _r_character(canonical_table),
                "canonical_field": _r_character(canonical_field),
                "unit": _r_character(unit),
                "transformation": _r_character(transformation),
                "loss_risk": risk,
                "evidence_case_id": _r_character(evidence_case_id),
            }
        ],
        columns=SEMANTICS_COLUMNS,
    )
    keys = ["vendor", "native_field", "canonical_table", "canonical_field"]
    target_key = tuple(str(row.iloc[0][key]) for key in keys)
    if registry.empty:
        registry = row
    else:
        key_series = registry[keys].astype(str).apply(tuple, axis=1)
        matches = key_series.map(lambda value: value == target_key)
        if bool(matches.any()):
            registry.loc[matches, SEMANTICS_COLUMNS] = row.iloc[0].to_numpy()
        else:
            registry = pd.concat([registry, row], ignore_index=True)
    _atomic_write_csv(registry, _semantics_path(corpus))
    return registry.reset_index(drop=True)


def compare_vendor_semantics(x, vendors=None):
    """Compare semantic mappings and report maximum canonical loss risk."""
    if isinstance(x, (str, Path)):
        corpus = Path(x).expanduser().resolve()
        data = _read_semantics(corpus)
    elif isinstance(x, pd.DataFrame):
        data = x.copy()
    else:
        _stop("Expected a corpus path or semantics data frame.")

    required = {
        "vendor",
        "native_field",
        "canonical_table",
        "canonical_field",
        "loss_risk",
    }
    if not required.issubset(data.columns):
        _stop("Semantics registry is incomplete.")
    if vendors is not None:
        if isinstance(vendors, str):
            vendors = [vendors]
        allowed = {_vendor(value) for value in vendors}
        data = data.loc[data["vendor"].astype(str).isin(allowed)].copy()

    columns = [
        "canonical_table",
        "canonical_field",
        "vendors",
        "vendor_names",
        "native_fields",
        "transformations",
        "maximum_loss_risk",
    ]
    if data.empty:
        return _tag_frame(
            pd.DataFrame(columns=columns),
            "eye_vendor_semantic_comparison",
        )

    rows = []
    keys = data.loc[:, ["canonical_table", "canonical_field"]].drop_duplicates()
    for key in keys.itertuples(index=False):
        subset = data.loc[
            data["canonical_table"].eq(key.canonical_table) & data["canonical_field"].eq(key.canonical_field)
        ]
        names = sorted(set(subset["vendor"].dropna().astype(str)))
        native_fields = "; ".join(
            f"{vendor}:{field}"
            for vendor, field in zip(
                subset["vendor"].astype(str),
                subset["native_field"].astype(str),
                strict=True,
            )
        )
        transformations = "; ".join(
            dict.fromkeys(subset.get("transformation", pd.Series(dtype=str)).dropna().astype(str))
        )
        observed = [
            LOSS_RISK_ORDER[value] for value in subset["loss_risk"].dropna().astype(str) if value in LOSS_RISK_ORDER
        ]
        maximum = (
            max(LOSS_RISK_ORDER, key=LOSS_RISK_ORDER.get)
            if False
            else (next(name for name, rank in LOSS_RISK_ORDER.items() if rank == max(observed)) if observed else pd.NA)
        )
        rows.append(
            {
                "canonical_table": key.canonical_table,
                "canonical_field": key.canonical_field,
                "vendors": len(names),
                "vendor_names": ", ".join(names),
                "native_fields": native_fields,
                "transformations": transformations,
                "maximum_loss_risk": maximum,
            }
        )
    return _tag_frame(
        pd.DataFrame(rows, columns=columns),
        "eye_vendor_semantic_comparison",
    )


def _table_key(table: pd.DataFrame, name: str) -> list[str]:
    specified = {
        "recordings": ["recording_id"],
        "gaze_samples": ["recording_id", "sample_id"],
        "eye_samples": ["recording_id", "eye_sample_id"],
        "events": ["recording_id", "event_id"],
        "intervals": ["recording_id", "interval_id"],
        "responses": ["participant_id", "item_id", "trial_id"],
        "features": [
            "participant_id",
            "item_id",
            "trial_id",
            "feature_name",
        ],
    }
    candidates = specified.get(
        name,
        [
            column
            for column in [
                "recording_id",
                "participant_id",
                "trial_id",
                "item_id",
                "sample_id",
                "event_id",
            ]
            if column in table.columns
        ],
    )
    return [column for column in candidates if column in table.columns]


def _with_occurrence(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    if not keys:
        return out
    token = out[list(keys)].astype("string").fillna("<NA>").agg("\r".join, axis=1)
    out[".occurrence"] = token.groupby(token, sort=False).cumcount() + 1
    return out


def _compare_columns(
    a: pd.DataFrame,
    b: pd.DataFrame,
    keys: Sequence[str],
    tolerance: float,
) -> pd.DataFrame:
    shared = [column for column in a.columns if column not in keys and column in b.columns]
    columns = [
        "column",
        "source_nonmissing",
        "roundtrip_nonmissing",
        "missingness_change",
        "comparable",
        "mismatch_rate",
        "max_absolute_difference",
    ]
    if not shared:
        return pd.DataFrame(columns=columns)

    if keys:
        aa = _with_occurrence(a[list(keys) + shared], keys)
        bb = _with_occurrence(b[list(keys) + shared], keys)
        merged = aa.merge(
            bb,
            on=list(keys) + [".occurrence"],
            how="outer",
            suffixes=(".source", ".roundtrip"),
            sort=False,
        )
    else:
        length = max(len(a), len(b))
        merged = pd.DataFrame({".row": np.arange(1, length + 1)})
        for column in shared:
            merged[f"{column}.source"] = a[column].reset_index(drop=True).reindex(range(length))
            merged[f"{column}.roundtrip"] = b[column].reset_index(drop=True).reindex(range(length))

    rows = []
    for column in shared:
        source = merged[f"{column}.source"]
        roundtrip = merged[f"{column}.roundtrip"]
        numeric = (
            pd.api.types.is_numeric_dtype(source.dtype)
            and pd.api.types.is_numeric_dtype(roundtrip.dtype)
            and not pd.api.types.is_bool_dtype(source.dtype)
            and not pd.api.types.is_bool_dtype(roundtrip.dtype)
        )
        comparable = source.notna() & roundtrip.notna()
        if numeric:
            difference = (pd.to_numeric(source, errors="coerce") - pd.to_numeric(roundtrip, errors="coerce")).abs()
        else:
            difference = source.astype("string").ne(roundtrip.astype("string")).astype(float)
        comparison = difference.loc[comparable]
        rows.append(
            {
                "column": column,
                "source_nonmissing": int(source.notna().sum()),
                "roundtrip_nonmissing": int(roundtrip.notna().sum()),
                "missingness_change": (
                    float(roundtrip.isna().mean() - source.isna().mean()) if len(merged) else math.nan
                ),
                "comparable": int(comparable.sum()),
                "mismatch_rate": (float((comparison > tolerance).mean()) if len(comparison) else math.nan),
                "max_absolute_difference": (float(comparison.max()) if numeric and len(comparison) else math.nan),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def audit_roundtrip_loss(
    source,
    roundtrip,
    tables=None,
    tolerance=1e-8,
):
    """Audit canonical table/field loss across an import-export round trip."""
    if not is_eye_dataset(source) or not is_eye_dataset(roundtrip):
        _stop("Both inputs must be `eye_dataset` objects.")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        _stop("`tolerance` must be a non-negative finite number.")
    if not math.isfinite(tolerance) or tolerance < 0:
        _stop("`tolerance` must be a non-negative finite number.")
    canonical = list(canonical_table_names())
    if tables is None:
        tables = canonical
    if isinstance(tables, str):
        tables = [tables]
    tables = [table for table in tables if table in canonical]

    summaries = []
    details = []
    for table in tables:
        a = source.get(table)
        b = roundtrip.get(table)
        if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
            continue
        keys = _table_key(a, table)
        detail = _compare_columns(a, b, keys, tolerance)
        if not detail.empty:
            detail["table"] = table
            details.append(detail)
        maximum = (
            float(
                pd.to_numeric(
                    detail["mismatch_rate"],
                    errors="coerce",
                ).max()
            )
            if not detail.empty
            and pd.to_numeric(
                detail["mismatch_rate"],
                errors="coerce",
            )
            .notna()
            .any()
            else math.nan
        )
        mismatch_ok = (
            detail.empty
            or pd.to_numeric(
                detail["mismatch_rate"],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
            .all()
        )
        summaries.append(
            {
                "table": table,
                "source_rows": len(a),
                "roundtrip_rows": len(b),
                "row_difference": len(b) - len(a),
                "source_columns": len(a.columns),
                "roundtrip_columns": len(b.columns),
                "missing_source_columns": ",".join(column for column in a.columns if column not in b.columns),
                "extra_roundtrip_columns": ",".join(column for column in b.columns if column not in a.columns),
                "maximum_mismatch_rate": maximum,
                "status": (
                    "lossless"
                    if len(a) == len(b) and not any(column not in b.columns for column in a.columns) and mismatch_ok
                    else "review"
                ),
            }
        )

    summary_columns = [
        "table",
        "source_rows",
        "roundtrip_rows",
        "row_difference",
        "source_columns",
        "roundtrip_columns",
        "missing_source_columns",
        "extra_roundtrip_columns",
        "maximum_mismatch_rate",
        "status",
    ]
    detail_frame = pd.concat(details, ignore_index=True, sort=False) if details else pd.DataFrame()
    return EyeRoundtripLossAudit(
        summary=pd.DataFrame(summaries, columns=summary_columns),
        details=detail_frame,
        tolerance=tolerance,
    )


def audit_vendor_field_coverage(semantics, required_fields):
    """Audit vendor coverage of required canonical table/field pairs."""
    if isinstance(semantics, (str, Path)):
        data = _read_semantics(Path(semantics).expanduser().resolve())
    else:
        data = semantics
    if not isinstance(data, pd.DataFrame) or not isinstance(
        required_fields,
        pd.DataFrame,
    ):
        _stop("Semantics and required fields must be data frames.")
    if not {"canonical_table", "canonical_field"}.issubset(required_fields.columns):
        _stop("Required fields need canonical table and field columns.")
    vendors = sorted(set(data.get("vendor", pd.Series(dtype=str)).dropna().astype(str)))
    rows = []
    required_keys = [
        f"{table}::{field}"
        for table, field in zip(
            required_fields["canonical_table"].astype(str),
            required_fields["canonical_field"].astype(str),
            strict=True,
        )
    ]
    for vendor in vendors:
        subset = data.loc[data["vendor"].astype(str).eq(vendor)]
        keys = {
            f"{table}::{field}"
            for table, field in zip(
                subset["canonical_table"].astype(str),
                subset["canonical_field"].astype(str),
                strict=True,
            )
        }
        for (table, field), key in zip(
            required_fields[["canonical_table", "canonical_field"]].itertuples(index=False, name=None),
            required_keys,
            strict=True,
        ):
            rows.append(
                {
                    "vendor": vendor,
                    "canonical_table": table,
                    "canonical_field": field,
                    "supported": key in keys,
                }
            )
    return _tag_frame(
        pd.DataFrame(
            rows,
            columns=[
                "vendor",
                "canonical_table",
                "canonical_field",
                "supported",
            ],
        ),
        "eye_vendor_field_coverage",
    )


def _validation_pass(validation: Any) -> bool:
    if isinstance(validation, (bool, np.bool_)):
        return bool(validation)
    status = None
    if isinstance(validation, Mapping):
        status = validation.get("status")
    else:
        status = getattr(validation, "status", None)
    if status is None:
        return False
    return str(status).lower() in {"pass", "passed", "success"}


def promote_vendor_support(
    corpus_path,
    case_id,
    level=("fixture-tested", "empirically-validated"),
    validation=None,
    reviewer=None,
    notes=pd.NA,
):
    """Promote a case only when the required validation evidence passes."""
    support = _match_arg(
        level,
        ["fixture-tested", "empirically-validated"],
        "level",
    )
    reviewer_text = "" if reviewer is None else str(reviewer).strip()
    if not reviewer_text:
        _stop("A non-empty reviewer identifier is required.")
    registry = read_vendor_registry(corpus_path)
    matches = registry["case_id"].astype(str).eq(str(case_id))
    if not bool(matches.any()):
        _stop(f"Unknown case: {case_id}")
    index = registry.index[matches][0]
    if not _validation_pass(validation):
        _stop("Support cannot be promoted without passing validation evidence.")
    if support == "empirically-validated" and (
        registry.at[index, "independent_source"] is not True
        and not bool(registry.at[index, "independent_source"])
        or registry.at[index, "licence_reviewed"] is not True
        and not bool(registry.at[index, "licence_reviewed"])
    ):
        _stop("Empirical promotion requires independent-source and licensing evidence.")

    registry.at[index, "support_level"] = support
    registry.at[index, "status"] = "validated"
    existing = registry.at[index, "notes"]
    parts = []
    if not pd.isna(existing) and str(existing):
        parts.append(str(existing))
    suffix = f"Promoted by {reviewer_text} at {_now_utc()}: {'' if pd.isna(notes) else notes}"
    parts.append(suffix)
    registry.at[index, "notes"] = " | ".join(parts)
    write_vendor_registry(registry, corpus_path)
    return _tag_frame(
        registry.loc[[index]].reset_index(drop=True),
        "eye_validation_case",
    )


def build_compatibility_matrix(
    x,
    required_vendors=("gazepoint", "tobii", "pupillabs", "eyelink", "smi"),
    min_empirical_cases=2,
):
    """Build evidence-tiered vendor compatibility claims."""
    data = read_vendor_registry(x) if isinstance(x, (str, Path)) else x
    if not isinstance(data, pd.DataFrame):
        _stop("Expected a corpus path or registry data frame.")
    try:
        minimum = int(min_empirical_cases)
    except (TypeError, ValueError):
        _stop("`min_empirical_cases` must be a positive integer.")
    if minimum < 1:
        _stop("`min_empirical_cases` must be a positive integer.")
    if isinstance(required_vendors, str):
        required_vendors = [required_vendors]
    required = list(dict.fromkeys(_vendor(value) for value in required_vendors))
    observed = list(dict.fromkeys(data.get("vendor", pd.Series(dtype=str)).dropna().astype(str)))
    vendors = required + [vendor for vendor in observed if vendor not in required]

    rows = []
    for vendor in vendors:
        subset = data.loc[data.get("vendor", pd.Series(dtype=str)).astype(str).eq(vendor)].copy()
        support = subset.get("support_level", pd.Series(dtype=str)).astype(str)
        independent = (
            _coerce_bool_series(subset["independent_source"]).fillna(False)
            if "independent_source" in subset
            else pd.Series(False, index=subset.index, dtype=bool)
        )
        licensed = (
            _coerce_bool_series(subset["licence_reviewed"]).fillna(False)
            if "licence_reviewed" in subset
            else pd.Series(False, index=subset.index, dtype=bool)
        )
        status = subset.get("status", pd.Series(dtype=str)).astype(str)
        empirical = (
            support.eq("empirically-validated")
            & independent.astype(bool)
            & licensed.astype(bool)
            & status.eq("validated")
        )
        support_ranks = [SUPPORT_ORDER[value] for value in support if value in SUPPORT_ORDER]
        highest = (
            next(name for name, rank in SUPPORT_ORDER.items() if rank == max(support_ranks))
            if support_ranks
            else "none"
        )
        devices = sorted(set(subset.get("device_model", pd.Series(dtype=str)).dropna().astype(str)))
        versions = sorted(set(subset.get("software_version", pd.Series(dtype=str)).dropna().astype(str)))
        empirical_count = int(empirical.sum())
        rows.append(
            {
                "vendor": vendor,
                "declared_cases": int(support.eq("declared").sum()),
                "fixture_tested_cases": int(support.eq("fixture-tested").sum()),
                "empirical_cases": empirical_count,
                "devices": "; ".join(devices),
                "software_versions": "; ".join(versions),
                "highest_support": highest,
                "production_claim_allowed": empirical_count >= minimum,
            }
        )
    return _tag_frame(
        pd.DataFrame(rows),
        "eye_vendor_compatibility_matrix",
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "_No rows available._"
    data = frame.copy()
    data = data.map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(map(str, data.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in data.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_vendor_case_report(
    corpus_path,
    case_id,
    path,
    validation=None,
    roundtrip=None,
):
    """Write a Markdown evidence report for one registered vendor case."""
    corpus = Path(corpus_path).expanduser().resolve()
    registry = read_vendor_registry(corpus)
    row = registry.loc[registry["case_id"].astype(str).eq(str(case_id))]
    if row.empty:
        _stop(f"Unknown case: {case_id}")
    fingerprint_path = corpus / "fingerprints" / f"{case_id}.csv"
    fingerprint = (
        pd.read_csv(fingerprint_path, dtype_backend="numpy_nullable") if fingerprint_path.exists() else pd.DataFrame()
    )
    lines = [
        f"# Vendor validation case: {case_id}",
        "",
        f"Generated: {_now_utc()}",
        "",
        "## Registration",
        "",
        _markdown_table(row),
        "",
        "## Source fingerprint",
        "",
        _markdown_table(fingerprint),
        "",
    ]
    if validation is not None:
        if isinstance(validation, Mapping):
            status = validation.get("status", "not exposed")
        else:
            status = getattr(validation, "status", "not exposed")
        lines.extend(
            [
                "## Validation",
                "",
                (
                    "Validation evidence was supplied as a Python object. "
                    "The object class and status are recorded below."
                ),
                "",
                f"- Class: {validation.__class__.__name__}",
                f"- Status: {status}",
                "",
            ]
        )
    if isinstance(roundtrip, EyeRoundtripLossAudit):
        lines.extend(
            [
                "## Round-trip loss",
                "",
                _markdown_table(roundtrip["summary"]),
                "",
            ]
        )
    support = row["support_level"].iloc[0]
    lines.extend(
        [
            "## Claim boundary",
            "",
            f"Registered support level: **{support}**.",
            (
                "Fixture tests are not independent empirical validation. "
                "Production compatibility claims require the declared "
                "minimum number of independent real-export cases."
            ),
        ]
    )
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return _canonical_path(destination)
