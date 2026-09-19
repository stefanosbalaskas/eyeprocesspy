from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_jsonable(v) for v in value.tolist()]
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def write_reporting_bundle(
    result: Any,
    output_dir: Path,
    *,
    inference_stability: pd.DataFrame,
    report_text: str,
    analysis_plan_text: str,
    figure_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Write a deterministic, manuscript-oriented AOI robustness bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Path, str]] = {}

    def add_text(name: str, text: str, role: str) -> None:
        path = output_dir / name
        path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        files[name] = (path, role)

    def add_csv(name: str, frame: pd.DataFrame, role: str) -> None:
        path = output_dir / name
        _write_csv(frame, path)
        files[name] = (path, role)

    add_text("analysis-plan.yml", analysis_plan_text, "prespecified analysis plan")
    add_csv("perturbation-audit.csv", result["grid_result"]["audit"], "geometry branch audit")
    add_csv("assignment-stability.csv", result["stability"]["overall"], "assignment robustness summary")
    add_csv("assignment-table.csv", result["assignment_table"], "observation-level assignments")
    add_csv("assignment-frequency.csv", result["assignment_probability"], "empirical assignment frequencies")
    add_csv("model-results.csv", result["models"], "branch-level model outputs")
    add_csv("inference-stability.csv", inference_stability, "model robustness summary")
    add_csv("failures.csv", result["failures"], "retained geometry/model failures")
    add_text("report.md", report_text, "manuscript-oriented robustness report")

    provenance_path = output_dir / "provenance.json"
    provenance_payload = {
        "provenance": _jsonable(result["provenance"]),
        "caveat": str(result.get("caveat", "")),
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    files["provenance.json"] = (provenance_path, "scientific provenance and interpretation caveat")

    for role, source in sorted((figure_paths or {}).items()):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Figure for {role!r} does not exist: {source}")
        target = output_dir / source.name
        if source.resolve() != target.resolve():
            target.write_bytes(source.read_bytes())
        files[target.name] = (target, f"visual diagnostic: {role}")

    manifest_entries = []
    for name in sorted(files):
        path, role = files[name]
        manifest_entries.append(
            {
                "path": name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "bundle_type": "eyeprocesspy_aoi_robustness_evidence",
        "files": manifest_entries,
        "scientific_contract": {
            "observation_level": _jsonable(result["provenance"].get("observation_level")),
            "overlap_policy": _jsonable(result["provenance"].get("overlap_policy")),
            "robustness_interpretation": "descriptive sensitivity, not probability of truth",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {name: path for name, (path, _) in files.items()} | {"manifest.json": manifest_path}


def run(output_dir: Path) -> dict[str, Path]:
    """Build the existing synthetic AOI result and write a reporting bundle."""
    import matplotlib.pyplot as plt
    import eyeprocesspy as ep
    from aoi_perturbation_sensitivity import (
        explicit_ols_callback,
        make_aois,
        make_fixations,
    )

    data = make_fixations()
    aois = make_aois()
    viewing = dict(
        screen_width_px=1024,
        screen_height_px=768,
        viewing_distance=60,
        physical_screen_size=(53.1, 29.9),
        boundary_policy="allow",
    )
    grid = ep.create_aoi_perturbation_grid(
        dilations=[0.25, 0.50, 1.00],
        erosions=[0.25],
        translations_x=[0.50],
        translations_y=[0.50],
        unit="deg",
        include_baseline=True,
        **viewing,
    )
    result = ep.run_aoi_sensitivity_analysis(
        data,
        aois,
        grid,
        x_col="x",
        y_col="y",
        observation_id_col="obs",
        participant_col="participant",
        trial_col="trial",
        duration_col="duration",
        time_col="time",
        observation_level="fixation",
        overlap_policy="ambiguous",
        model_callback=explicit_ols_callback,
        preprocessing_specification={"duration_unit": "seconds"},
        event_detector="synthetic_fixations",
        quality_rules={"missing": "preserve", "overlap": "ambiguous"},
        model_specification={
            "family": "OLS",
            "outcome": "disclosure_dwell",
            "predictor": "condition",
        },
    )
    inference = ep.assess_aoi_inference_stability(result, term="condition")
    plan = """analysis_id: synthetic-disclosure-aoi-robustness
observation_level: fixation
perturbation_unit: deg
overlap_policy: ambiguous
failure_handling: retain
"""

    figure_dir = output_dir / "_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    ax = ep.plot_aoi_perturbations(
        result,
        perturbation_id="dilate_0.5_deg",
        data=data.iloc[::12].copy(),
        x_col="x",
        y_col="y",
    )
    figures["geometry and reassignment"] = figure_dir / "aoi-geometry.svg"
    ax.figure.savefig(figures["geometry and reassignment"], bbox_inches="tight")
    plt.close(ax.figure)

    ax = ep.plot_aoi_assignment_stability(result)
    figures["assignment stability"] = figure_dir / "aoi-assignment-stability.svg"
    ax.figure.savefig(figures["assignment stability"], bbox_inches="tight")
    plt.close(ax.figure)

    ax = ep.plot_aoi_coefficient_stability(result, term="condition")
    figures["coefficient stability"] = figure_dir / "aoi-coefficient-stability.svg"
    ax.figure.savefig(figures["coefficient stability"], bbox_inches="tight")
    plt.close(ax.figure)

    paths = write_reporting_bundle(
        result,
        output_dir,
        inference_stability=inference,
        report_text=ep.report_aoi_sensitivity(result),
        analysis_plan_text=plan,
        figure_paths=figures,
    )
    for path in figure_dir.glob("*"):
        path.unlink()
    figure_dir.rmdir()
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workflow-output/aoi-reporting-bundle"),
    )
    args = parser.parse_args()
    for name, path in run(args.output_dir).items():
        print(f"{name}: {path}")
