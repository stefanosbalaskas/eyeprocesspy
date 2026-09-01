from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

MATRIX = Path("parity/PARITY_MATRIX.csv")


def main() -> None:
    with MATRIX.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    print(f"TOTAL_ROWS={len(rows)}")
    for column in (
        "p1_api",
        "p2_structural",
        "p3_semantic",
        "p4_numerical",
        "p5_algorithmic",
        "p6_plot",
        "p7_docs_examples",
    ):
        counts = Counter(row[column] for row in rows)
        print(f"[{column}]")
        for status, count in sorted(counts.items()):
            print(f"{status or '<blank>'}={count}")

    for column in ("p3_semantic", "p4_numerical", "p7_docs_examples"):
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            grouped[row["source_file"]][row[column]] += 1
        print(f"[{column}_BY_SOURCE]")
        for source in sorted(grouped):
            summary = ",".join(
                f"{status or '<blank>'}:{count}"
                for status, count in sorted(grouped[source].items())
            )
            print(f"{source}|{summary}")

    p4_not_started = [row for row in rows if row["p4_numerical"] == "not_started"]
    print(f"P4_NOT_STARTED={len(p4_not_started)}")
    for row in p4_not_started:
        print(f"P4_NOT_STARTED_ROW={row['source_file']}|{row['r_name']}|{row['python_module']}|{row['python_test_file']}")

    p4_diff = [row for row in rows if row["p4_numerical"] == "python_reference_differs"]
    p4_diff_without_blocker = [row for row in p4_diff if not row["blocker"].strip()]
    print(f"P4_PYTHON_REFERENCE_DIFFERS={len(p4_diff)}")
    print(f"P4_DIFF_WITHOUT_BLOCKER={len(p4_diff_without_blocker)}")
    for row in p4_diff_without_blocker:
        print(f"P4_DIFF_WITHOUT_BLOCKER_ROW={row['source_file']}|{row['r_name']}")

    p7_initial = [row for row in rows if row["p7_docs_examples"] == "docstring_initial"]
    print(f"P7_DOCSTRING_INITIAL={len(p7_initial)}")

    plot_rows = [row for row in rows if row["plot_candidate"].lower() == "true"]
    print(f"PLOT_CANDIDATES={len(plot_rows)}")
    print("PLOT_STATUS=" + ",".join(f"{k}:{v}" for k, v in sorted(Counter(row["p6_plot"] for row in plot_rows).items())))


if __name__ == "__main__":
    main()
