from __future__ import annotations

import hashlib
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .exceptions import EyeProcessValidationError

__all__ = [
    "eyeprocess_benchmark_study",
    "read_benchmark_table",
    "benchmark_expected_outputs",
    "import_benchmark_study",
    "validate_benchmark_study",
    "run_benchmark_reproduction",
    "write_benchmark_data_dictionary",
    "package_reproducibility_manifest",
    "verify_reproducibility_manifest",
    "write_software_paper_reproduction",
    "audit_benchmark_release",
]

_BENCHMARK_VERSION = "1.0.0"
_REQUIRED_TABLES = (
    "participants",
    "items",
    "responses",
    "gaze_samples",
    "events",
    "aoi_definitions",
    "pupil_samples",
    "quality",
    "provenance",
)


class _EyeProcessMapping(dict):
    _eyeprocess_class = "eyeprocess_mapping"

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def eyeprocess_class(self) -> str:
        return self._eyeprocess_class


class EyeBenchmarkStudy(_EyeProcessMapping):
    _eyeprocess_class = "eye_benchmark_study"


class EyeBenchmarkValidation(_EyeProcessMapping):
    _eyeprocess_class = "eye_benchmark_validation"


class EyeBenchmarkReproduction(_EyeProcessMapping):
    _eyeprocess_class = "eye_benchmark_reproduction"


class EyeReproducibilityManifest(_EyeProcessMapping):
    _eyeprocess_class = "eye_reproducibility_manifest"


class EyeBenchmarkTables(_EyeProcessMapping):
    _eyeprocess_class = "eye_benchmark_tables"


def _benchmark_root(path: str | Path | None = None) -> Path:
    if path is not None:
        root = Path(path).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent / "resources" / "extdata" / "benchmark-study"
    if not root.is_dir():
        raise EyeProcessValidationError(
            "The eyeprocess benchmark-study directory is unavailable. "
            "Install a wheel containing resources/extdata/benchmark-study or provide `path`."
        )
    return root


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_csv_types(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
            continue
        observed = series.dropna().astype(str).str.strip().str.lower()
        if observed.empty or not bool(observed.isin(["true", "false"]).all()):
            continue
        mapped = series.map(
            lambda value: pd.NA if pd.isna(value) or not str(value).strip() else str(value).strip().lower() == "true"
        )
        if not bool(mapped.isna().any()):
            out[column] = mapped.astype(bool)
        else:
            out[column] = mapped.astype("boolean")
    return out


def eyeprocess_benchmark_study(path: str | Path | None = None) -> EyeBenchmarkStudy:
    """Locate the frozen synthetic multimodal benchmark and read its manifest."""
    root = _benchmark_root(path)
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise EyeProcessValidationError(f"Benchmark manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=True)
    licence_path = root / "LICENSE.txt"
    licence = licence_path.read_text(encoding="utf-8") if licence_path.is_file() else ""
    return EyeBenchmarkStudy(path=root, manifest=manifest, licence=licence)


def _coerce_study(study: EyeBenchmarkStudy | str | Path | None) -> EyeBenchmarkStudy:
    if study is None:
        return eyeprocess_benchmark_study()
    if isinstance(study, EyeBenchmarkStudy):
        return study
    if isinstance(study, (str, Path)):
        return eyeprocess_benchmark_study(study)
    if isinstance(study, Mapping) and "path" in study and "manifest" in study:
        return EyeBenchmarkStudy(study)
    raise TypeError("`study` must be an eye benchmark study, a benchmark directory, or None.")


def read_benchmark_table(
    study: EyeBenchmarkStudy | str | Path | None = None,
    table: str | None = None,
) -> pd.DataFrame:
    """Read one benchmark table with stable logical-column restoration."""
    benchmark = _coerce_study(study)
    if not isinstance(table, str) or not table.strip():
        raise EyeProcessValidationError("`table` must be one non-empty benchmark table name.")
    manifest = benchmark["manifest"]
    rows = manifest.loc[manifest["table"].astype(str) == table]
    if rows.empty:
        raise EyeProcessValidationError(f"Benchmark table `{table}` is absent.")
    path = Path(benchmark["path"]) / str(rows.iloc[0]["file"])
    if not path.is_file():
        raise EyeProcessValidationError(f"Benchmark table file is missing: {path}")
    return _normalise_csv_types(pd.read_csv(path, keep_default_na=True))


def benchmark_expected_outputs(
    study: EyeBenchmarkStudy | str | Path | None = None,
) -> pd.DataFrame:
    """Read frozen benchmark reproduction targets and tolerances."""
    benchmark = _coerce_study(study)
    path = Path(benchmark["path"]) / "expected_outputs.csv"
    if not path.is_file():
        raise EyeProcessValidationError(f"Benchmark expected-output file is missing: {path}")
    return pd.read_csv(path, keep_default_na=True)


def import_benchmark_study(
    study: EyeBenchmarkStudy | str | Path | None = None,
):
    """Import benchmark tables, preferring the canonical EyeDataset constructor when compatible."""
    benchmark = _coerce_study(study)
    tables = {
        str(name): read_benchmark_table(benchmark, str(name)) for name in benchmark["manifest"]["table"].astype(str)
    }
    try:
        from .dataset import new_eye_dataset
        from .schema import canonical_table_names

        canonical = set(canonical_table_names())
        accepted = {name: value for name, value in tables.items() if name in canonical}
        return new_eye_dataset(**accepted)
    except Exception:
        return EyeBenchmarkTables(tables)


def _relation_row(check: str, passed: bool, detail: str = "") -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def validate_benchmark_study(
    study: EyeBenchmarkStudy | str | Path | None = None,
    verify_hashes: bool = True,
) -> EyeBenchmarkValidation:
    """Validate benchmark file integrity, fingerprints, and cross-table relations."""
    benchmark = _coerce_study(study)
    root = Path(benchmark["path"])
    file_rows: list[dict[str, object]] = []
    for row in benchmark["manifest"].itertuples(index=False):
        path = root / str(row.file)
        exists = path.is_file()
        current_bytes = path.stat().st_size if exists else pd.NA
        bytes_match = bool(exists and int(current_bytes) == int(row.bytes))
        current_md5 = _md5(path) if exists and verify_hashes else pd.NA
        hash_match = bool(exists and str(current_md5) == str(row.md5)) if verify_hashes else pd.NA
        file_rows.append(
            {
                "table": str(row.table),
                "file": str(row.file),
                "exists": exists,
                "bytes_match": bytes_match,
                "hash_match": hash_match,
            }
        )
    files = pd.DataFrame(file_rows)
    relations: list[dict[str, object]] = []
    if not files.empty and bool(files["exists"].all()):
        data = {
            str(name): read_benchmark_table(benchmark, str(name)) for name in benchmark["manifest"]["table"].astype(str)
        }
        participants = set(data["participants"]["participant_id"].astype(str))
        items = set(data["items"]["item_id"].astype(str))
        trial_ids = set(data["responses"]["trial_id"].astype(str))
        relations.extend(
            [
                _relation_row(
                    "responses_participants",
                    set(data["responses"]["participant_id"].astype(str)).issubset(participants),
                    "Every response participant occurs in participants.",
                ),
                _relation_row(
                    "responses_items",
                    set(data["responses"]["item_id"].astype(str)).issubset(items),
                    "Every response item occurs in items.",
                ),
                _relation_row(
                    "gaze_trials",
                    set(data["gaze_samples"]["trial_id"].astype(str)).issubset(trial_ids),
                    "Every gaze trial occurs in responses.",
                ),
                _relation_row(
                    "pupil_trials",
                    set(data["pupil_samples"]["trial_id"].astype(str)).issubset(trial_ids),
                    "Every pupil trial occurs in responses.",
                ),
                _relation_row(
                    "event_trials",
                    set(data["events"]["trial_id"].astype(str)).issubset(trial_ids),
                    "Every event trial occurs in responses.",
                ),
            ]
        )
        response_key = data["responses"][["participant_id", "item_id", "trial_id"]]
        relations.append(
            _relation_row(
                "unique_response_key",
                not bool(response_key.duplicated().any()),
                "Participant-item-trial response keys are unique.",
            )
        )
        response_time = pd.to_numeric(data["responses"]["response_time"], errors="coerce")
        relations.append(
            _relation_row(
                "positive_response_time",
                bool((np.isfinite(response_time) & (response_time > 0)).all()),
                "Response times are finite and positive.",
            )
        )
        x = pd.to_numeric(data["gaze_samples"]["x"], errors="coerce")
        y = pd.to_numeric(data["gaze_samples"]["y"], errors="coerce")
        bounded = np.isfinite(x) & np.isfinite(y) & x.between(0, 1) & y.between(0, 1)
        relations.append(
            _relation_row(
                "bounded_gaze",
                bool(bounded.all()),
                "Normalised gaze coordinates are finite and within [0, 1].",
            )
        )
    relations_frame = pd.DataFrame(relations, columns=["check", "passed", "detail"])
    if verify_hashes:
        valid_files = bool(
            not files.empty
            and files["exists"].all()
            and files["bytes_match"].all()
            and files["hash_match"].fillna(False).all()
        )
    else:
        valid_files = bool(not files.empty and files["exists"].all() and files["bytes_match"].all())
    valid_relations = bool(not relations_frame.empty and relations_frame["passed"].all())
    return EyeBenchmarkValidation(
        valid=bool(valid_files and valid_relations),
        files=files,
        relations=relations_frame,
        benchmark_version=_BENCHMARK_VERSION,
    )


def run_benchmark_reproduction(
    study: EyeBenchmarkStudy | str | Path | None = None,
) -> EyeBenchmarkReproduction:
    """Recompute the frozen deterministic benchmark summary and compare it with reference outputs."""
    benchmark = _coerce_study(study)
    responses = read_benchmark_table(benchmark, "responses")
    gaze = read_benchmark_table(benchmark, "gaze_samples")
    pupil = read_benchmark_table(benchmark, "pupil_samples")
    quality = read_benchmark_table(benchmark, "quality")
    aoi = read_benchmark_table(benchmark, "aoi_definitions")
    if not pd.api.types.is_bool_dtype(gaze["valid"].dtype) or bool(gaze["valid"].isna().any()):
        raise EyeProcessValidationError("Benchmark `gaze_samples.valid` must be complete logical data.")
    outputs = {
        "participants": float(responses["participant_id"].nunique()),
        "items": float(responses["item_id"].nunique()),
        "trials": float(len(responses)),
        "accuracy": float(pd.to_numeric(responses["score"], errors="coerce").mean()),
        "mean_response_time": float(pd.to_numeric(responses["response_time"], errors="coerce").mean()),
        "gaze_samples": float(len(gaze)),
        "pupil_samples": float(len(pupil)),
        "valid_gaze_fraction": float(gaze["valid"].mean()),
        "mean_pupil": float(pd.to_numeric(pupil["pupil"], errors="coerce").mean()),
        "quality_rows": float(len(quality)),
        "aoi_count": float(len(aoi)),
    }
    observed = pd.DataFrame({"metric": list(outputs), "observed": list(outputs.values())})
    expected = benchmark_expected_outputs(benchmark).copy()
    comparison = observed.merge(expected, on="metric", how="left", validate="one_to_one")
    if bool(comparison[["expected", "tolerance"]].isna().any().any()):
        raise EyeProcessValidationError("Benchmark expected-output metrics and observations must match completely.")
    comparison["absolute_error"] = (comparison["observed"] - comparison["expected"]).abs()
    comparison["passed"] = comparison["absolute_error"] <= comparison["tolerance"]
    return EyeBenchmarkReproduction(
        outputs=outputs,
        comparison=comparison,
        passed=bool(comparison["passed"].all()),
        study=benchmark,
    )


def write_benchmark_data_dictionary(
    study: EyeBenchmarkStudy | str | Path | None = None,
    path: str | Path = "benchmark-data-dictionary.md",
) -> str:
    """Write the benchmark data dictionary as Markdown."""
    benchmark = _coerce_study(study)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# eyeprocess benchmark data dictionary",
        "",
        (
            "This benchmark is synthetic and openly redistributable. It is not evidence of "
            "compatibility with any commercial eye-tracker export."
        ),
        "",
    ]
    for table in benchmark["manifest"]["table"].astype(str):
        frame = read_benchmark_table(benchmark, table)
        lines.extend(
            [
                f"## `{table}`",
                "",
                f"Rows: {len(frame)}; columns: {len(frame.columns)}.",
                "",
                "| Column | Python dtype | Missing |",
                "|---|---|---:|",
            ]
        )
        for column in frame.columns:
            lines.append(f"| `{column}` | {frame[column].dtype} | {int(frame[column].isna().sum())} |")
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return str(output)


def _coerce_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        values = [paths]
    else:
        values = list(paths)
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = Path(value).expanduser().resolve()
        candidates = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result


def package_reproducibility_manifest(
    paths: str | Path | Iterable[str | Path],
    include_session: bool = True,
) -> EyeReproducibilityManifest:
    """Fingerprint files and capture a Python runtime/session record for reproducibility."""
    files = _coerce_paths(paths)
    frame = pd.DataFrame(
        [{"path": str(path), "bytes": path.stat().st_size, "md5": _md5(path)} for path in files],
        columns=["path", "bytes", "md5"],
    )
    session: list[str] = []
    if include_session:
        session.extend(
            [
                f"Python {platform.python_version()}",
                f"numpy {np.__version__}",
                f"pandas {pd.__version__}",
            ]
        )
        try:
            import scipy

            session.append(f"scipy {scipy.__version__}")
        except Exception:
            pass
        try:
            from . import __version__

            session.append(f"eyeprocesspy {__version__}")
        except Exception:
            pass
    return EyeReproducibilityManifest(
        files=frame,
        created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        session=session,
    )


def verify_reproducibility_manifest(manifest: EyeReproducibilityManifest | Mapping[str, object]) -> pd.DataFrame:
    """Verify that every file recorded in a reproducibility manifest is unchanged."""
    if not isinstance(manifest, Mapping) or "files" not in manifest:
        raise TypeError("`manifest` must be an eye reproducibility manifest.")
    files = manifest["files"]
    if not isinstance(files, pd.DataFrame):
        raise TypeError("`manifest.files` must be a pandas DataFrame.")
    rows: list[dict[str, object]] = []
    for row in files.itertuples(index=False):
        path = Path(str(row.path))
        exists = path.is_file()
        current_md5 = _md5(path) if exists else pd.NA
        rows.append(
            {
                "path": str(path),
                "exists": exists,
                "expected_md5": str(row.md5),
                "current_md5": current_md5,
                "unchanged": bool(exists and current_md5 == str(row.md5)),
            }
        )
    return pd.DataFrame(rows, columns=["path", "exists", "expected_md5", "current_md5", "unchanged"])


def write_software_paper_reproduction(
    directory: str | Path,
    study: EyeBenchmarkStudy | str | Path | None = None,
    overwrite: bool = False,
) -> EyeReproducibilityManifest:
    """Create a self-contained Python reproduction scaffold around the frozen benchmark."""
    benchmark = _coerce_study(study)
    output = Path(directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise EyeProcessValidationError("`directory` is non-empty; use `overwrite=True` to replace generated files.")
    data_dir = output / "data"
    results_dir = output / "results"
    scripts_dir = output / "scripts"
    for folder in (data_dir, results_dir, scripts_dir):
        folder.mkdir(parents=True, exist_ok=True)
    for source in Path(benchmark["path"]).iterdir():
        target = data_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    script = """from pathlib import Path\n\nimport pandas as pd\nimport eyeprocesspy as ep\n\nroot = Path(__file__).resolve().parents[1]\nstudy = ep.eyeprocess_benchmark_study(root / \"data\")\nvalidation = ep.validate_benchmark_study(study)\nif not validation[\"valid\"]:\n    raise RuntimeError(\"Benchmark validation failed.\")\nreproduction = ep.run_benchmark_reproduction(study)\nif not reproduction[\"passed\"]:\n    raise RuntimeError(\"Benchmark reproduction failed.\")\nreproduction[\"comparison\"].to_csv(root / \"results\" / \"benchmark-comparison.csv\", index=False)\nmanifest = ep.package_reproducibility_manifest([root / \"data\", root / \"results\", root / \"scripts\"])\nmanifest[\"files\"].to_csv(root / \"results\" / \"reproducibility-manifest.csv\", index=False)\nprint(\"eyeprocesspy benchmark reproduction: PASS\")\n"""
    (scripts_dir / "run_reproduction.py").write_text(script, encoding="utf-8", newline="\n")
    readme = """# eyeprocesspy software-paper reproduction scaffold\n\nThis directory contains the frozen synthetic benchmark and a Python reproduction script.\nThe benchmark is for software reproducibility testing; it is not empirical vendor validation.\n\nRun from an environment with `eyeprocesspy` installed:\n\n```powershell\npython scripts/run_reproduction.py\n```\n\nThe script validates benchmark hashes and relations, reproduces the frozen expected summaries,\nand writes results under `results/`.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    return package_reproducibility_manifest(output)


def audit_benchmark_release(
    study: EyeBenchmarkStudy | str | Path | None = None,
) -> _EyeProcessMapping:
    """Audit whether benchmark assets are ready for public release."""
    benchmark = _coerce_study(study)
    validation = validate_benchmark_study(benchmark)
    reproduction = run_benchmark_reproduction(benchmark) if validation["valid"] else None
    observed = set(benchmark["manifest"]["table"].astype(str))
    root = Path(benchmark["path"])
    findings = pd.DataFrame(
        {
            "check": [
                "integrity",
                "reproduction",
                "required_tables",
                "open_licence",
                "synthetic_label",
            ],
            "passed": [
                bool(validation["valid"]),
                bool(reproduction is not None and reproduction["passed"]),
                set(_REQUIRED_TABLES).issubset(observed),
                (root / "LICENSE.txt").is_file(),
                (root / "SYNTHETIC_DATA.txt").is_file(),
            ],
        }
    )
    return _EyeProcessMapping(
        ready=bool(findings["passed"].all()),
        findings=findings,
        validation=validation,
        reproduction=reproduction,
    )
