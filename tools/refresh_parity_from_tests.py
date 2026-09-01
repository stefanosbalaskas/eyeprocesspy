from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

MATRIX = Path("parity/PARITY_MATRIX.csv")


def _test_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for part in value.split("|"):
        part = part.strip()
        if part:
            paths.append(Path(part))
    return paths


def _mentions(name: str, test_file: str) -> tuple[bool, list[str]]:
    checked: list[str] = []
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for path in _test_paths(test_file):
        if not path.exists():
            continue
        checked.append(path.as_posix())
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            return True, checked
    return False, checked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh initial p4/p6 evidence only when a declared Python test file names the exact API."
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
        referenced, checked = _mentions(name, row.get("python_test_file", ""))

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
