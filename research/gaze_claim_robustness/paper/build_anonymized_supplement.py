#!/usr/bin/env python3
"""Build the single anonymized CHI review supplement ZIP.

The bundle is deliberately curated. It contains only files needed to audit the
frozen decision space, reconstruct the paper-facing summaries, and understand
the core analysis mechanics. Raw public datasets are not redistributed and the
bundle contains no live repository link or author-identifying provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research" / "gaze_claim_robustness"

SUPPLEMENT_FILES = (
    # Frozen decision contracts and source-grounded plans.
    "srl_decision_registry.csv",
    "srl_detector_plan.csv",
    "srl_aoi_evidence.csv",
    "srl_cohort_plan.csv",
    "srl_quality_plan.csv",
    "srl_model_plan.csv",
    "srl_outcome_plan.csv",
    "mcfw_decision_registry.csv",
    # Core executable mechanics (not the full package/repository).
    "srl_coordinate_branches.py",
    "srl_aoi_geometry.py",
    "srl_transition_outcome.py",
    "fit_srl_primary_glmm.R",
    "mcfw_coordinates.py",
    # Compact frozen paper-facing outputs.
    "results/srl_claim_summary.csv",
    "results/srl_decision_sensitivity.csv",
    "results/srl_detector_distance_sensitivity.csv",
    "results/srl_interval_stability_summary.csv",
    "results/mcfw_validation_summary.csv",
    "results/srl_paper_figure_snapshot.csv",
    "results/mcfw_context_jaccard_snapshot.csv",
)

FORBIDDEN_TEXT = (
    "stefanosbalaskas",
    "github.com",
    "raw.githubusercontent.com",
    "36683906120",
    "36708103076",
    "36776432517",
    "5fad0f192067ca74277c4b5f2bec462514395124",
    "878e351a82a48665990955853c55b2315b10fd27",
)

README = """ANONYMIZED REVIEW SUPPLEMENT
================================

Paper: From Gaze Signals to HCI Claims: Propagating Eye-Tracking Measurement
Uncertainty Through Analysis Pipelines

Purpose
-------
This single ZIP is an anonymized review supplement. The main paper is designed
to stand alone; these files provide additional auditability for reviewers who
wish to inspect the frozen decision contracts, core analysis mechanics, and
compact result summaries.

Contents
--------
- contracts/: frozen SRL and MCFW decision registries and prespecified plans.
- code/: core geometry, AOI, transition-outcome, model, and MCFW coordinate
  mechanics used by the study. This is a curated review subset rather than a
  distribution package.
- results/: compact frozen summaries and exact paper-facing figure snapshots.
- MANIFEST.tsv: SHA-256 digest and byte size for every bundled file.

Data availability
-----------------
The two source datasets are public third-party research datasets and are cited
in the paper. They are not redistributed in this supplement. No participant
records, credentials, private data, or unpublished source archives are included.

Anonymization
-------------
This review bundle intentionally contains no author names, institutional
identifiers, live repository links, workflow URLs, or identifying version-
control provenance. A build-time scan rejects known identifying strings before
this ZIP can be emitted.

Interpretation boundary
-----------------------
These materials do not define a new analysis path and do not alter either
frozen empirical universe. The SRL files support the 144-specification
claim-stability analysis. The MCFW files support independent measurement-level
generalization. Specification frequencies are descriptive properties of the
declared grid, not probabilities that an effect is true.
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arcname(relative: str) -> str:
    path = Path(relative)
    if relative.startswith("results/"):
        return relative
    if path.suffix in {".py", ".R"}:
        return f"code/{path.name}"
    return f"contracts/{path.name}"


def _scan_text(name: str, data: bytes) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Supplement source is not UTF-8 text: {name}") from exc
    lowered = text.casefold()
    hits = [token for token in FORBIDDEN_TEXT if token.casefold() in lowered]
    if hits:
        raise ValueError(f"Anonymization scan failed for {name}: {hits}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build(output: Path) -> None:
    entries: list[tuple[str, bytes]] = []
    for relative in SUPPLEMENT_FILES:
        source = RESEARCH / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        data = source.read_bytes()
        _scan_text(relative, data)
        entries.append((_arcname(relative), data))

    readme_bytes = README.encode("utf-8")
    _scan_text("README.txt", readme_bytes)
    entries.append(("README.txt", readme_bytes))

    manifest_lines = ["path\tbytes\tsha256"]
    for name, data in sorted(entries):
        manifest_lines.append(f"{name}\t{len(data)}\t{_sha256(data)}")
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    entries.append(("MANIFEST.tsv", manifest))

    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries):
            archive.writestr(_zip_info(name), data)
    output.write_bytes(buffer.getvalue())

    # Verify the emitted archive, including the generated metadata files.
    with zipfile.ZipFile(output) as archive:
        archive.testzip()
        names = archive.namelist()
        if names != sorted(names):
            raise ValueError("Supplement ZIP member order is not deterministic.")
        for name in names:
            _scan_text(name, archive.read(name))

    print(f"wrote {output} ({output.stat().st_size} bytes)")
    print(f"sha256={_sha256(output.read_bytes())}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("anonymous-review-supplement.zip"),
    )
    args = parser.parse_args()
    build(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
