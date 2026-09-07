#!/usr/bin/env python3
"""Fail CI only when a change adds Ruff diagnostics beyond the repository baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ZERO_SHA = "0" * 40
CHECK_PATHS = ("src", "tests", "scripts")
Fingerprint = tuple[str, str, str]


def _run(command: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def _resolve_base(repo: Path, requested: str | None) -> str | None:
    if requested and requested != ZERO_SHA:
        probe = _run(
            ["git", "cat-file", "-e", f"{requested}^{{commit}}"],
            repo,
            check=False,
        )
        if probe.returncode == 0:
            return requested

    parent = _run(["git", "rev-parse", "HEAD^"], repo, check=False)
    if parent.returncode == 0:
        return parent.stdout.strip()
    return None


def _ruff(root: Path, config: Path) -> list[dict[str, Any]]:
    paths = [name for name in CHECK_PATHS if (root / name).exists()]
    if not paths:
        return []

    result = _run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *paths,
            "--config",
            str(config),
            "--output-format=json",
            "--exit-zero",
        ],
        root,
    )
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Ruff JSON payload.")
    return payload


def _relative_filename(diagnostic: dict[str, Any], root: Path) -> str:
    filename = Path(str(diagnostic["filename"]))
    if not filename.is_absolute():
        return filename.as_posix()

    try:
        return filename.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return filename.as_posix()


def _fingerprint(diagnostic: dict[str, Any], root: Path) -> Fingerprint:
    return (
        _relative_filename(diagnostic, root),
        str(diagnostic["code"]),
        str(diagnostic["message"]),
    )


def _print_new(
    diagnostics: list[dict[str, Any]],
    root: Path,
    new_counts: Counter[Fingerprint],
) -> None:
    remaining = new_counts.copy()
    for diagnostic in diagnostics:
        fingerprint = _fingerprint(diagnostic, root)
        if remaining[fingerprint] <= 0:
            continue

        location = diagnostic.get("location") or {}
        row = location.get("row", "?")
        column = location.get("column", "?")
        filename, code, message = fingerprint
        print(f"{filename}:{row}:{column}: {code} {message}")
        remaining[fingerprint] -= 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail only when the current revision adds Ruff diagnostics.",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Git base SHA. Falls back to HEAD^ when unavailable.",
    )
    args = parser.parse_args()

    repo = Path(
        _run(["git", "rev-parse", "--show-toplevel"], Path.cwd()).stdout.strip()
    ).resolve()
    config = repo / "pyproject.toml"
    base_sha = _resolve_base(repo, args.base)
    head_diagnostics = _ruff(repo, config)

    if base_sha is None:
        if head_diagnostics:
            print("No baseline commit is available; current Ruff diagnostics:")
            counts = Counter(_fingerprint(item, repo) for item in head_diagnostics)
            _print_new(head_diagnostics, repo, counts)
            return 1
        print("No baseline commit is available and Ruff is clean.")
        return 0

    with tempfile.TemporaryDirectory(prefix="eyeprocesspy-ruff-base-") as temp_dir:
        base_root = Path(temp_dir) / "base"
        _run(["git", "worktree", "add", "--detach", str(base_root), base_sha], repo)
        try:
            base_diagnostics = _ruff(base_root, config)
        finally:
            _run(
                ["git", "worktree", "remove", "--force", str(base_root)],
                repo,
                check=False,
            )

    head_counts = Counter(_fingerprint(item, repo) for item in head_diagnostics)
    base_counts = Counter(_fingerprint(item, base_root) for item in base_diagnostics)
    new_counts = head_counts - base_counts

    if new_counts:
        print(
            f"Ruff regression: {sum(new_counts.values())} new diagnostic(s) "
            f"relative to {base_sha}."
        )
        _print_new(head_diagnostics, repo, new_counts)
        return 1

    print(
        "Ruff delta clean: no new diagnostics "
        f"(baseline={len(base_diagnostics)}, head={len(head_diagnostics)})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
