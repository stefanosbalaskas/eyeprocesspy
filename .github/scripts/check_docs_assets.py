from __future__ import annotations

import os
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class StaticReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "img" and values.get("src"):
            self.references.append(("img", values["src"]))
        elif tag == "script" and values.get("src"):
            self.references.append(("script", values["src"]))
        elif tag == "link" and values.get("href"):
            self.references.append(("link", values["href"]))


def resolve_target(site_root: Path, html_path: Path, url: str) -> Path | None:
    if url.startswith(("http://", "https://", "//", "data:", "mailto:", "tel:", "#")):
        return None

    path = unquote(urlsplit(url).path)
    if not path:
        return None

    if path.startswith("/eyeprocesspy/"):
        return site_root / path.removeprefix("/eyeprocesspy/")
    if path.startswith("/"):
        return site_root / path.lstrip("/")
    return Path(os.path.normpath(html_path.parent / path))


def exists_as_site_target(target: Path) -> bool:
    return target.exists() or (target / "index.html").exists()


def main() -> int:
    site_root = Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve()
    if not site_root.is_dir():
        print(f"Documentation site directory not found: {site_root}", file=sys.stderr)
        return 2

    broken: list[tuple[Path, str, str, Path]] = []
    checked = 0

    for html_path in site_root.rglob("*.html"):
        parser = StaticReferenceParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        for kind, url in parser.references:
            target = resolve_target(site_root, html_path, url)
            if target is None:
                continue
            checked += 1
            if not exists_as_site_target(target):
                broken.append((html_path.relative_to(site_root), kind, url, target))

    if broken:
        print(f"Broken documentation static references: {len(broken)}", file=sys.stderr)
        for page, kind, url, target in broken:
            print(f"- {page}: <{kind}> {url!r} -> {target}", file=sys.stderr)
        return 1

    print(f"Documentation static-reference audit passed: {checked} local references checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
