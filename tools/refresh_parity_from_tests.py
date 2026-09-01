from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

MATRIX = Path("parity/PARITY_MATRIX.csv")

RECONSTRUCTED_P4_TEST = "tests/test_reconstructed_measurement_p4.py"
PROCESS_IRT_P4_TEST = "tests/test_process_irt_p4_completion.py"

SOURCE_TEST_FALLBACKS: dict[str, tuple[str, ...]] = {
    "R/022-advanced-models-v2.R": (
        "tests/test_dynamic_strategy_diffusion.py",
    ),
    "R/024-dynamic-irtree-engine.R": (
        "tests/test_dynamic_strategy_diffusion.py",
    ),
    "R/027-strategy-diffusion-engines.R": (
        "tests/test_dynamic_strategy_diffusion.py",
    ),
    "R/033-process-uncertainty.R": (RECONSTRUCTED_P4_TEST,),
    "R/034-offline-recalibration.R": (RECONSTRUCTED_P4_TEST,),
    "R/034-calibration-recalibration.R": (RECONSTRUCTED_P4_TEST,),
    "R/035-process-reliability.R": (RECONSTRUCTED_P4_TEST,),
    "R/037-pupil-registration.R": (RECONSTRUCTED_P4_TEST,),
    "R/038-process-missingness.R": (RECONSTRUCTED_P4_TEST,),
    "R/038-informative-missingness.R": (RECONSTRUCTED_P4_TEST,),
    "R/039-recurrence-process.R": (RECONSTRUCTED_P4_TEST,),
    "R/039-recurrence-analysis.R": (RECONSTRUCTED_P4_TEST,),
    "R/040-fixation-point-process.R": (RECONSTRUCTED_P4_TEST,),
    "R/041-scanpath-population.R": (RECONSTRUCTED_P4_TEST,),
    "R/041-representative-scanpaths.R": (RECONSTRUCTED_P4_TEST,),
    "R/042-process-episodes.R": (RECONSTRUCTED_P4_TEST,),
    "R/046-evidence-graph.R": (RECONSTRUCTED_P4_TEST,),
    "R/046-evidence-provenance-graph.R": (RECONSTRUCTED_P4_TEST,),
    "R/049-multimodal-irt-registry.R": (PROCESS_IRT_P4_TEST,),
    "R/050-process-irt-models-0-7.R": (PROCESS_IRT_P4_TEST,),
    "R/051-advanced-process-irt-0-7.R": (PROCESS_IRT_P4_TEST,),
    "R/052-irt-validation-0-7.R": (PROCESS_IRT_P4_TEST,),
    "R/054-additional-process-measurement-0-7.R": (PROCESS_IRT_P4_TEST,),
    "R/057-emerging-process-irt-0-7.R": (PROCESS_IRT_P4_TEST,),
    "R/083-irt-foundations-information-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
        "tests/test_irt_p4_model_spec.py",
    ),
    "R/084-irt-diagnostics-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/085-irt-scoring-adaptive-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/086-irt-linking-invariance-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/087-irt-process-joint-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/088-irt-engine-adapters-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/089-irt-validation-evidence-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/090-irt-multidimensional-cdm-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
    "R/093-irt-advanced-diagnostics-governance-0-9.R": (
        "tests/test_irt_0_9_parity.py",
        "tests/test_irt_p4_numerical_parity.py",
    ),
}


def _test_paths(value: str, source_file: str) -> list[Path]:
    values = [part.strip() for part in value.split("|") if part.strip()]
    values.extend(SOURCE_TEST_FALLBACKS.get(source_file, ()))
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        paths.append(Path(value))
    return paths


def _mentions(name: str, test_file: str, source_file: str) -> tuple[bool, list[str]]:
    checked: list[str] = []
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for path in _test_paths(test_file, source_file):
        if not path.exists():
            continue
        checked.append(path.as_posix())
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            return True, checked
    return False, checked


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh initial p4/p6 evidence only when an exact public API name is present in a declared "
            "or source-family fallback scientific test file."
        )
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    raw = MATRIX.read_text(encoding="utf-8-sig")
    newline = "\r\n" if "\r\n" in raw else "\n"
    with MATRIX.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    p4_upgraded: list[str] = []
    p6_upgraded: list[str] = []
    still_p4: list[str] = []
    still_p6: list[str] = []

    for row in rows:
        name = row["python_name"] or row["r_name"]
        referenced, checked = _mentions(
            name,
            row.get("python_test_file", ""),
            row["source_file"],
        )

        if row["p4_numerical"] == "not_started":
            if referenced:
                row["p4_numerical"] = "tested_initial"
                p4_upgraded.append(f"{row['source_file']}|{name}|{','.join(checked)}")
            else:
                still_p4.append(f"{row['source_file']}|{name}|{','.join(checked)}")

        if row.get("plot_candidate", "").lower() == "true" and row["p6_plot"] == "not_started":
            if referenced:
                row["p6_plot"] = "tested_initial"
                p6_upgraded.append(f"{row['source_file']}|{name}|{','.join(checked)}")
            else:
                still_p6.append(f"{row['source_file']}|{name}|{','.join(checked)}")

    print(f"P4_TEST_REFERENCED_UPGRADES={len(p4_upgraded)}")
    for value in p4_upgraded:
        print(f"P4_TEST_REFERENCED_ROW={value}")
    print(f"P4_STILL_NOT_STARTED={len(still_p4)}")
    for value in still_p4:
        print(f"P4_STILL_NOT_STARTED_ROW={value}")

    print(f"P6_TEST_REFERENCED_UPGRADES={len(p6_upgraded)}")
    for value in p6_upgraded:
        print(f"P6_TEST_REFERENCED_ROW={value}")
    print(f"P6_STILL_NOT_STARTED={len(still_p6)}")
    for value in still_p6:
        print(f"P6_STILL_NOT_STARTED_ROW={value}")

    if args.apply:
        with MATRIX.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator=newline)
            writer.writeheader()
            writer.writerows(rows)
        print("PARITY_MATRIX_UPDATED=1")
    else:
        print("PARITY_MATRIX_UPDATED=0")


if __name__ == "__main__":
    main()
