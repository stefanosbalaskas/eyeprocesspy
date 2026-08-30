"""Completion of the frozen R/021 validation-release programme."""

from __future__ import annotations

import gc
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import _assert_eye_dataset
from .engine_adapters import compare_model_engines
from .exceptions import EyeProcessValidationError
from .grouped_validation_10 import crossed_grouped_cv, grouped_cv, quantify_process_leakage
from .io_validation_10 import anonymize_eye_dataset, export_canonical, validate_eye_corpus
from .schema import canonical_table_names, empty_eye_table
from .validation_evidence_10 import (
    advanced_model_evidence_spec,
    audit_advanced_model_evidence,
    run_raven_reproduction,
    sbc_summary,
    simulation_based_calibration,
    write_advanced_model_evidence_report,
)
from .validation_program_10 import (
    audit_vendor_validation,
    model_validation_summary,
    run_model_validation,
    write_vendor_validation_report,
)

__all__ = [
    "benchmark_eyeprocess",
    "create_public_benchmark",
    "preprocessing_multiverse",
    "reporting_guideline_audit",
    "run_eyeprocess_validation_program",
    "write_reporting_guideline_report",
    "write_software_paper_scaffold",
]


class EyeMultiverse(dict):
    """Python counterpart of the frozen R ``eye_multiverse`` object."""

    eyeprocess_class = "eye_multiverse"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class EyeValidationProgram(dict):
    """Python counterpart of the frozen R ``eye_validation_program`` object."""

    eyeprocess_class = "eye_validation_program"

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _stop(message: str) -> None:
    raise EyeProcessValidationError(message)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    try:
        return pd.DataFrame(value)
    except Exception as exc:
        raise EyeProcessValidationError("`extract` must return an object coercible to a DataFrame.") from exc


def _bind_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _set_frame_class(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.attrs["eyeprocess_class"] = name
    return frame


def _is_multiverse(value: Any) -> bool:
    return isinstance(value, EyeMultiverse) or getattr(value, "eyeprocess_class", None) == "eye_multiverse"


def preprocessing_multiverse(
    x,
    specifications,
    transform,
    analyse,
    extract=lambda z: _frame(z),
):
    """Run the frozen-R preprocessing/AOI multiverse contract."""
    if isinstance(specifications, Mapping):
        specs = dict(specifications)
    elif isinstance(specifications, Sequence) and not isinstance(specifications, (str, bytes)):
        if not len(specifications):
            _stop("`specifications` must be a non-empty list.")
        specs = {f"spec_{index}": value for index, value in enumerate(specifications, start=1)}
    else:
        _stop("`specifications` must be a non-empty list.")

    if not specs:
        _stop("`specifications` must be a non-empty list.")
    if not callable(transform) or not callable(analyse) or not callable(extract):
        _stop("`transform`, `analyse`, and `extract` must be callable.")

    rows = []
    for name, specification in specs.items():
        label = str(name)
        try:
            transformed = transform(x, specification)
            fit = analyse(transformed)
            result = _frame(extract(fit))
            result["specification"] = label
            result["error"] = pd.NA
        except Exception as exc:
            result = pd.DataFrame([{"specification": label, "error": str(exc)}])
        rows.append(result)

    return EyeMultiverse(
        results=_bind_rows(rows),
        specifications=specs,
    )


def _deep_size(value: Any) -> int:
    """Portable approximation of R ``object.size`` for benchmark reporting."""
    seen: set[int] = set()

    def size(obj: Any) -> int:
        identity = id(obj)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(obj)
        if isinstance(obj, Mapping):
            return total + sum(size(k) + size(v) for k, v in obj.items())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return total + sum(size(v) for v in obj)
        if isinstance(obj, pd.DataFrame):
            return total + int(obj.memory_usage(index=True, deep=True).sum())
        if isinstance(obj, pd.Series):
            return total + int(obj.memory_usage(index=True, deep=True))
        if isinstance(obj, np.ndarray):
            return total + int(obj.nbytes)
        return total

    return int(size(value))


def benchmark_eyeprocess(expr, iterations=5, label="operation"):
    """Benchmark a zero-argument operation using elapsed time and result size."""
    if not callable(expr):
        _stop("`expr` must be a function with no required arguments.")
    try:
        iterations = int(iterations)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`iterations` must be a positive integer.") from exc
    if iterations < 1:
        _stop("`iterations` must be a positive integer.")

    rows = []
    for iteration in range(1, iterations + 1):
        gc.collect()
        start = time.perf_counter()
        result = expr()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "label": str(label),
                "iteration": iteration,
                "elapsed_seconds": float(elapsed),
                "result_size_bytes": _deep_size(result),
            }
        )
    return _set_frame_class(pd.DataFrame(rows), "eye_benchmark")


def _table(x, name: str) -> pd.DataFrame:
    value = x.get(name)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def reporting_guideline_audit(x, model=None, sensitivity=None):
    """Audit the twelve frozen eye-tracking reporting-guideline sections."""
    _assert_eye_dataset(x)
    recordings = _table(x, "recordings")
    streams = _table(x, "streams")
    calibrations = _table(x, "calibrations")
    coordinate_spaces = _table(x, "coordinate_spaces")
    provenance = _table(x, "provenance")
    quality = _table(x, "quality")
    aois = _table(x, "aoi_definitions")

    hardware = not recordings.empty and "device_model" in recordings and recordings["device_model"].notna().any()
    sampling = False
    if not streams.empty:
        observed = pd.to_numeric(
            streams.get("observed_rate_hz", pd.Series(index=streams.index, dtype=float)),
            errors="coerce",
        )
        nominal = pd.to_numeric(
            streams.get("nominal_rate_hz", pd.Series(index=streams.index, dtype=float)),
            errors="coerce",
        )
        sampling = bool(
            np.isfinite(observed.to_numpy(dtype=float)).any() or np.isfinite(nominal.to_numpy(dtype=float)).any()
        )

    exclusion = False
    if not quality.empty and "metric" in quality:
        exclusion = bool(
            quality["metric"]
            .astype("string")
            .str.contains(r"excl|invalid|missing", case=False, regex=True, na=False)
            .any()
        )

    sections = [
        "hardware",
        "sampling",
        "calibration",
        "coordinates",
        "preprocessing",
        "quality",
        "aois",
        "exclusions",
        "provenance",
        "model",
        "sensitivity",
        "interpretation",
    ]
    items = [
        "device and software metadata",
        "sampling frequency",
        "calibration evidence",
        "coordinate space",
        "processing parameters",
        "quality metrics",
        "AOI definitions",
        "exclusion evidence",
        "transformation provenance",
        "model specification",
        "preprocessing and AOI sensitivity",
        "construct caution",
    ]
    covered = [
        hardware,
        sampling,
        not calibrations.empty,
        not coordinate_spaces.empty,
        not provenance.empty,
        not quality.empty,
        not aois.empty,
        exclusion,
        not provenance.empty,
        model is not None,
        _is_multiverse(sensitivity),
        True,
    ]
    frame = pd.DataFrame(
        {
            "section": sections,
            "item": items,
            "covered": [bool(value) for value in covered],
        }
    )
    frame["status"] = np.where(frame["covered"], "pass", "missing")
    return _set_frame_class(frame, "eye_reporting_audit")


def write_reporting_guideline_report(x, path):
    """Write the frozen reporting-guideline Markdown audit."""
    if not isinstance(x, pd.DataFrame) or x.attrs.get("eyeprocess_class") != "eye_reporting_audit":
        _stop("Expected an `eye_reporting_audit` object.")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eye-tracking reporting-guideline audit",
        "",
        f"Generated: {_now_utc()}",
        "",
        "| Section | Item | Covered | Status |",
        "|---|---|---|---|",
    ]
    for row in x.itertuples(index=False):
        lines.append(f"| {row.section} | {row.item} | {bool(row.covered)} | {row.status} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output.resolve())


def create_public_benchmark(
    x,
    path,
    max_participants=50,
    include_samples=False,
    overwrite=False,
):
    """Create the frozen de-identified canonical public benchmark bundle."""
    _assert_eye_dataset(x)
    try:
        max_participants = int(max_participants)
    except (TypeError, ValueError) as exc:
        raise EyeProcessValidationError("`max_participants` must be a positive integer.") from exc
    if max_participants < 1:
        _stop("`max_participants` must be a positive integer.")

    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        _stop(f"Benchmark directory already exists: {output}")
    if output.exists():
        shutil.rmtree(output)

    y = anonymize_eye_dataset(x, drop_raw=True, retain_map=False)
    recordings = _table(y, "recordings")
    if "participant_id" in recordings:
        participants = recordings["participant_id"].dropna().astype(str).drop_duplicates().tolist()
        if len(participants) > max_participants:
            keep = set(participants[:max_participants])
            retained_recordings = set(
                recordings.loc[
                    recordings["participant_id"].astype(str).isin(keep),
                    "recording_id",
                ]
                .dropna()
                .astype(str)
            )
            for name in canonical_table_names():
                table = _table(y, name)
                if "recording_id" in table:
                    y[name] = table.loc[table["recording_id"].astype(str).isin(retained_recordings)].reset_index(
                        drop=True
                    )

    if not include_samples:
        y["gaze_samples"] = empty_eye_table("gaze_samples")
        y["eye_samples"] = empty_eye_table("eye_samples")
        y["biometrics"] = empty_eye_table("biometrics")

    export_canonical(
        y,
        output,
        overwrite=True,
        include_raw=False,
    )
    checks = reporting_guideline_audit(y)
    checks.to_csv(output / "reporting-guideline-audit.csv", index=False, lineterminator="\n")
    return str(output.resolve())


def write_software_paper_scaffold(
    path,
    title="eyeprocess: Reproducible Psychometric Process Modelling in R",
    author="Stefanos Balaskas",
):
    """Write the frozen methodological software-paper R Markdown scaffold."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f'title: "{title}"',
        f'author: "{author}"',
        "output: html_document",
        "---",
        "",
        "# Summary",
        "",
        "Describe the bounded contribution and non-goals.",
        "",
        "# Statement of need",
        "",
        "Document the fragmented eye-tracking, pupillometry, response-time, sequence, and IRT ecosystem.",
        "",
        "# Canonical data and provenance model",
        "",
        "Describe persons, recordings, trials, responses, samples, events, AOIs, features, quality, and provenance.",
        "",
        "# Interoperability",
        "",
        "Report vendor corpora, package adapters, Eye-Tracking-BIDS, and storage benchmarks.",
        "",
        "# Statistical models",
        "",
        (
            "Separate descriptive process evidence, explanatory IRT, joint models, "
            "strategy models, dynamic models, and diffusion models."
        ),
        "",
        "# Validation programme",
        "",
        (
            "Report parameter recovery, interval coverage, calibration, misspecification, "
            "grouped validation, leakage, multiverse, and empirical reproduction."
        ),
        "",
        "# Empirical demonstrations",
        "",
        (
            "Include the real Gazepoint workflow and independent multi-vendor cases. "
            "Add the Raven reproduction only after data/licensing review."
        ),
        "",
        "# Limitations and responsible interpretation",
        "",
        "Do not equate gaze or pupil variables with psychological constructs without validation.",
        "",
        "# Availability and reproducibility",
        "",
        "Provide package version, source tag, session information, manifests, and public benchmark bundles.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output.resolve())


def _jsonable(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"class": type(value).__name__, "repr": repr(value)}


def _write_snapshot(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _jobs(value: Any, prefix: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        output = {}
        used: set[str] = set()
        for index, (name, job) in enumerate(value.items(), start=1):
            base = str(name).strip() or f"{prefix}{index}"
            candidate = base
            suffix = 1
            while candidate in used:
                suffix += 1
                candidate = f"{base}.{suffix}"
            used.add(candidate)
            output[candidate] = job
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {f"{prefix}{index}": job for index, job in enumerate(value, start=1)}
    _stop("Validation jobs must be a mapping or sequence.")


def _sanitize(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return text or "parameter"


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise EyeProcessValidationError(
            "Validation-program plots require the optional plotting dependencies. Install `eyeprocesspy[plots]`."
        ) from exc
    return plt


def _save_bar(path: Path, labels, values, ylabel: str, title: str, ylim=None) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots()
    ax.bar([str(v) for v in labels], np.asarray(values, dtype=float))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_line(path: Path, x, y, ylabel: str, title: str) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_benchmark_plot(path: Path, frame: pd.DataFrame) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots()
    groups = [
        pd.to_numeric(part["elapsed_seconds"], errors="coerce").dropna().to_numpy()
        for _, part in frame.groupby("label", sort=False)
    ]
    labels = [str(name) for name, _ in frame.groupby("label", sort=False)]
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("Seconds")
    ax.set_title("eyeprocess benchmark")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _merge_evidence(evidence: dict[str, Any], name: str, component: str, value: Any) -> None:
    if value is None or not name:
        return
    record = evidence.setdefault(name, {})
    if component not in record:
        record[component] = value


def _corpus_result(corpus):
    if isinstance(corpus, Mapping) and isinstance(corpus.get("summary"), pd.DataFrame):
        return dict(corpus)
    if isinstance(getattr(corpus, "summary", None), pd.DataFrame):
        return {
            "summary": corpus.summary,
            "manifest": getattr(corpus, "manifest", None),
            "status": getattr(corpus, "status", "unknown"),
        }
    return validate_eye_corpus(corpus)


def _comparison_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, Mapping):
        frame = value.get("comparison")
    else:
        frame = getattr(value, "comparison", None)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _result_frame(value: Any, key: str) -> pd.DataFrame:
    if isinstance(value, Mapping):
        frame = value.get(key)
    else:
        frame = getattr(value, key, None)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def run_eyeprocess_validation_program(
    corpus,
    output_dir,
    model_jobs=None,
    sbc_jobs=None,
    engine_jobs=None,
    reproduction_jobs=None,
    grouped_jobs=None,
    leakage_jobs=None,
    multiverse_jobs=None,
    benchmark_jobs=None,
    reporting_dataset=None,
    public_benchmark_dataset=None,
    public_benchmark_include_samples=False,
    advanced_evidence=None,
    evidence_spec=None,
    overwrite=False,
):
    """Run the complete frozen R/021 validation-release programme.

    R's internal ``saveRDS`` checkpoints are emitted as transparent JSON
    snapshots in Python. No non-R data is written under an ``.rds`` extension.
    """
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not overwrite:
        _stop(f"Validation output already exists: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    plot_dir = output / "plots"
    plot_dir.mkdir()

    model_jobs = _jobs(model_jobs, "model-")
    sbc_jobs = _jobs(sbc_jobs, "sbc-")
    engine_jobs = _jobs(engine_jobs, "engine-")
    reproduction_jobs = _jobs(reproduction_jobs, "reproduction-")
    grouped_jobs = _jobs(grouped_jobs, "grouped-")
    leakage_jobs = _jobs(leakage_jobs, "leakage-")
    multiverse_jobs = _jobs(multiverse_jobs, "multiverse-")
    benchmark_jobs = _jobs(benchmark_jobs, "benchmark-")
    evidence = dict(advanced_evidence or {})
    if evidence_spec is None:
        evidence_spec = advanced_model_evidence_spec()

    corpus_result = _corpus_result(corpus)
    vendor_audit = audit_vendor_validation(corpus_result)
    corpus_result["summary"].to_csv(output / "corpus-summary.csv", index=False, lineterminator="\n")
    vendor_audit.to_csv(output / "vendor-audit.csv", index=False, lineterminator="\n")
    write_vendor_validation_report(vendor_audit, output / "vendor-validation.md")
    _save_bar(
        plot_dir / "vendor-pass-rate.png",
        vendor_audit["vendor"],
        vendor_audit["pass_rate"],
        "Pass rate",
        "Multi-vendor validation",
        (0, 1),
    )
    _save_bar(
        plot_dir / "vendor-case-counts.png",
        vendor_audit["vendor"],
        vendor_audit["cases"],
        "Cases",
        "Multi-vendor validation",
    )

    models = {}
    for name, job in model_jobs.items():
        result = run_model_validation(**dict(job))
        models[name] = result
        _write_snapshot(output / f"model-validation-{name}.json", result)
        summary = model_validation_summary(result)
        summary.to_csv(
            output / f"model-validation-{name}.csv",
            index=False,
            lineterminator="\n",
        )
        _merge_evidence(evidence, name, "recovery", result)

    sbcs = {}
    for name, job in sbc_jobs.items():
        result = simulation_based_calibration(**dict(job))
        sbcs[name] = result
        _write_snapshot(output / f"sbc-{name}.json", result)
        summary = sbc_summary(result)
        summary.to_csv(output / f"sbc-{name}.csv", index=False, lineterminator="\n")
        ranks = _result_frame(result, "ranks")
        if not ranks.empty and "parameter" in ranks:
            for parameter, part in ranks.dropna(subset=["parameter"]).groupby("parameter", sort=False):
                values = pd.to_numeric(part["normalized_rank"], errors="coerce").dropna()
                if len(values):
                    _save_bar(
                        plot_dir / f"sbc-{name}-{_sanitize(parameter)}.png",
                        np.arange(len(values)),
                        values,
                        "Normalized rank",
                        f"SBC: {parameter}",
                        (0, 1),
                    )
        _merge_evidence(evidence, name, "calibration", result)

    engines = {}
    for name, job in engine_jobs.items():
        result = compare_model_engines(**dict(job))
        engines[name] = result
        _write_snapshot(output / f"engine-comparison-{name}.json", result)
        estimates = _result_frame(result, "estimates")
        estimates.to_csv(
            output / f"engine-comparison-{name}.csv",
            index=False,
            lineterminator="\n",
        )
        _merge_evidence(evidence, name, "engine_equivalence", result)

    reproductions = {}
    for name, job in reproduction_jobs.items():
        result = run_raven_reproduction(**dict(job))
        reproductions[name] = result
        _write_snapshot(output / f"empirical-reproduction-{name}.json", result)
        comparison = _comparison_frame(result)
        comparison.to_csv(
            output / f"empirical-reproduction-{name}.csv",
            index=False,
            lineterminator="\n",
        )
        _merge_evidence(evidence, name, "empirical_reproduction", result)

    grouped = {}
    for name, original in grouped_jobs.items():
        job = dict(original)
        crossed = bool(job.pop("crossed", False))
        result = crossed_grouped_cv(**job) if crossed else grouped_cv(**job)
        grouped[name] = result
        _write_snapshot(output / f"grouped-validation-{name}.json", result)
        result["results"].to_csv(
            output / f"grouped-validation-{name}.csv",
            index=False,
            lineterminator="\n",
        )
        scores = result["results"]
        _save_line(
            plot_dir / f"grouped-validation-{name}.png",
            scores["fold"],
            scores["score"],
            result["metric"],
            "Cross-classified grouped cross-validation" if crossed else "Grouped cross-validation",
        )
        _merge_evidence(evidence, name, "grouped_validation", result)

    leakage = {}
    for name, job in leakage_jobs.items():
        result = quantify_process_leakage(**dict(job))
        leakage[name] = result
        result.to_csv(output / f"leakage-{name}.csv", index=False, lineterminator="\n")
        _save_bar(
            plot_dir / f"leakage-{name}.png",
            result["scheme"],
            result["mean_log_loss"],
            "Mean log loss",
            "Grouped versus row-wise validation",
        )

    multiverses = {}
    for name, job in multiverse_jobs.items():
        result = preprocessing_multiverse(**dict(job))
        multiverses[name] = result
        _write_snapshot(output / f"multiverse-{name}.json", result)
        result["results"].to_csv(
            output / f"multiverse-{name}.csv",
            index=False,
            lineterminator="\n",
        )
        numeric = [
            column
            for column in result["results"].select_dtypes(include=[np.number]).columns
            if column != "specification"
        ]
        if numeric:
            _save_bar(
                plot_dir / f"multiverse-{name}.png",
                result["results"]["specification"],
                result["results"][numeric[0]],
                numeric[0],
                "Multiverse estimates",
            )
        _merge_evidence(evidence, name, "sensitivity", result)

    benchmarks = {}
    for name, job in benchmark_jobs.items():
        if callable(job):
            result = benchmark_eyeprocess(job, label=name)
        else:
            arguments = dict(job)
            arguments["label"] = name
            result = benchmark_eyeprocess(**arguments)
        benchmarks[name] = result

    if benchmarks:
        benchmark_table = _bind_rows(list(benchmarks.values()))
        _set_frame_class(benchmark_table, "eye_benchmark")
        benchmark_table.to_csv(output / "benchmarks.csv", index=False, lineterminator="\n")
        _save_benchmark_plot(plot_dir / "benchmarks.png", benchmark_table)

    reporting = None
    if reporting_dataset is not None:
        sensitivity = next(iter(multiverses.values()), None)
        reporting = reporting_guideline_audit(
            reporting_dataset,
            sensitivity=sensitivity,
        )
        reporting.to_csv(
            output / "reporting-guideline-audit.csv",
            index=False,
            lineterminator="\n",
        )
        write_reporting_guideline_report(
            reporting,
            output / "reporting-guideline-audit.md",
        )
        _save_bar(
            plot_dir / "reporting-guideline-coverage.png",
            reporting["section"],
            reporting["covered"].astype(int),
            "Covered",
            "Reporting-guideline coverage",
            (0, 1),
        )

    public_benchmark = None
    if public_benchmark_dataset is not None:
        public_benchmark = create_public_benchmark(
            public_benchmark_dataset,
            output / "public-benchmark",
            include_samples=public_benchmark_include_samples,
            overwrite=True,
        )

    paper_scaffold = write_software_paper_scaffold(output / "eyeprocess-software-paper.Rmd")

    evidence_audit = audit_advanced_model_evidence(evidence, evidence_spec)
    evidence_audit.to_csv(
        output / "advanced-model-evidence.csv",
        index=False,
        lineterminator="\n",
    )
    write_advanced_model_evidence_report(
        evidence_audit,
        output / "advanced-model-evidence.md",
    )
    completed = pd.to_numeric(
        evidence_audit.get("completed", 0),
        errors="coerce",
    ).fillna(0)
    required = pd.to_numeric(
        evidence_audit.get("required", 0),
        errors="coerce",
    ).fillna(0)
    proportion = np.where(required.to_numpy() > 0, completed.to_numpy() / required.to_numpy(), 1.0)
    _save_bar(
        plot_dir / "advanced-model-evidence.png",
        evidence_audit["model"],
        proportion,
        "Required evidence completed",
        "Advanced-model evidence",
        (0, 1),
    )

    result = EyeValidationProgram(
        corpus=corpus_result,
        vendor_audit=vendor_audit,
        models=models,
        sbc=sbcs,
        engine_comparisons=engines,
        reproductions=reproductions,
        grouped_validation=grouped,
        leakage=leakage,
        multiverses=multiverses,
        benchmarks=benchmarks,
        reporting=reporting,
        public_benchmark=public_benchmark,
        paper_scaffold=paper_scaffold,
        advanced_evidence=evidence,
        evidence_audit=evidence_audit,
        output_dir=str(output),
    )
    _write_snapshot(output / "validation-program.json", result)
    (output / "serialization-boundary.md").write_text(
        "# Serialization boundary\n\n"
        "The frozen R workflow writes internal `.rds` checkpoints. "
        "eyeprocesspy does not disguise Python serialization as RDS; "
        "equivalent Python-native checkpoints use transparent JSON.\n",
        encoding="utf-8",
    )
    return result
