"""Print one version's CHANGELOG section as its release's notes.

The release workflow runs this for a tag vX.Y.Z and publishes what it prints. It
refuses unless CHANGELOG.md has a dated section with notes for the version, and
`pyproject.toml` and `open_dneo/__init__.py` carry it, so a tag on a commit at
another version publishes nothing. GitHub shows a release's line breaks as they
are, so each paragraph and list item goes out on one line.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERSION = r"\d+\.\d+\.\d+"
# A line that begins a block of its own: a list item, a heading, a table row, a quote.
OPENS = re.compile(r"\s*(- |\d+\. |#|\||>)")


def section(changelog: str, version: str) -> str:
    """The notes under `## VERSION (YYYY-MM-DD)`. ValueError if there are none."""
    if not re.fullmatch(VERSION, version):
        raise ValueError(f"not a release version: {version!r}")
    found = re.search(
        rf"(?ms)^## {re.escape(version)} \(\d{{4}}-\d{{2}}-\d{{2}}\)\n(.*?)(?=^## |\Z)",
        changelog,
    )
    if found is None or not found[1].strip():
        raise ValueError(f"CHANGELOG.md has no dated section with notes for {version}")
    return found[1].strip() + "\n"


def unwrapped(text: str) -> str:
    """`text` with each paragraph and list item on one line; fenced blocks as they are."""
    lines: list[str] = []
    fenced = False
    open_line = False  # the last line is text that a wrapped line continues
    for line in text.splitlines():
        fence = line.startswith("```")
        if fenced or fence:
            lines.append(line)
            fenced, open_line = fenced != fence, False
        elif open_line and line.strip() and not OPENS.match(line):
            lines[-1] += " " + line.strip()
        else:
            lines.append(line)
            open_line = bool(line.strip())
    return "\n".join(lines) + "\n"


def notes(repo: Path, version: str) -> str:
    """The notes of `version`. ValueError unless the project at `repo` is at it."""
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    body = unwrapped(section(changelog, version))
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    package = re.search(
        r'(?m)^__version__ = "([^"]*)"$',
        (repo / "open_dneo" / "__init__.py").read_text(encoding="utf-8"),
    )
    written = {
        "pyproject.toml": project["project"]["version"],
        "open_dneo/__init__.py": package[1] if package else None,
    }
    for where, found in written.items():
        if found != version:
            raise ValueError(f"{where} is at {found}, not {version}")
    return body


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="as X.Y.Z, without the tag's v")
    args = parser.parse_args(argv)
    try:
        text = notes(REPO, args.version)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    # Bytes, so that the notes are UTF-8 with LF on every platform.
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
