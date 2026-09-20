from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "coverage.json"
REPORT = ROOT / "coverage_gap_report.md"


def main() -> int:
    if not COVERAGE.exists():
        raise SystemExit(
            "coverage.json does not exist. Run the strict pytest coverage command first."
        )

    payload = json.loads(COVERAGE.read_text(encoding="utf-8"))
    files = payload.get("files", {})

    rows = []

    for path, info in sorted(files.items()):
        missing_lines = list(info.get("missing_lines", []))
        missing_branches = list(info.get("missing_branches", []))

        if not missing_lines and not missing_branches:
            continue

        source_path = ROOT / path
        source_lines = (
            source_path.read_text(encoding="utf-8").splitlines() if source_path.exists() else []
        )

        contexts = []
        interesting = sorted(
            set(
                missing_lines
                + [
                    branch[0]
                    for branch in missing_branches
                    if isinstance(branch, list) and len(branch) == 2 and isinstance(branch[0], int)
                ]
            )
        )

        for line_no in interesting:
            if not source_lines or line_no < 1 or line_no > len(source_lines):
                continue

            start = max(1, line_no - 2)
            end = min(len(source_lines), line_no + 2)

            snippet = []
            for current in range(start, end + 1):
                marker = ">>" if current == line_no else "  "
                snippet.append(f"{marker} {current:5d}: {source_lines[current - 1]}")

            contexts.append("\n".join(snippet))

        summary = info.get("summary", {})
        rows.append(
            {
                "path": path,
                "missing_lines": missing_lines,
                "missing_branches": missing_branches,
                "percent": summary.get("percent_covered"),
                "contexts": contexts,
            }
        )

    totals = payload.get("totals", {})
    total_missing_lines = int(totals.get("missing_lines", 0))
    total_missing_branches = int(totals.get("missing_branches", 0))
    percent = float(totals.get("percent_covered", 0.0))

    output = [
        "# eyeprocesspy strict coverage gap report",
        "",
        f"- Coverage: **{percent:.4f}%**",
        f"- Missing statements: **{total_missing_lines}**",
        f"- Missing branches: **{total_missing_branches}**",
        f"- Files with residual gaps: **{len(rows)}**",
        "",
    ]

    if not rows:
        output.extend(
            [
                "## RESULT",
                "",
                "**100% statement and branch coverage achieved.**",
                "",
            ]
        )
    else:
        for row in rows:
            output.extend(
                [
                    f"## `{row['path']}`",
                    "",
                    f"- Coverage: {row['percent']}",
                    f"- Missing lines: `{row['missing_lines']}`",
                    f"- Missing branches: `{row['missing_branches']}`",
                    "",
                ]
            )

            for snippet in row["contexts"]:
                output.extend(
                    [
                        "```text",
                        snippet,
                        "```",
                        "",
                    ]
                )

    REPORT.write_text("\n".join(output), encoding="utf-8")

    print("\n".join(output[:10]))
    print(f"\nFull report written to: {REPORT}")

    return 0 if not rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
