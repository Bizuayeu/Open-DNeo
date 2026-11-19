"""The command line: `open-dneo shapes REPO` and `open-dneo entrance REPO`."""

from __future__ import annotations

import argparse
import sys

from . import __version__, entrance, shapes


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="open-dneo", description=__doc__)
    top.add_argument("--version", action="version", version=__version__)
    commands = top.add_subparsers(dest="command", required=True)

    listing = commands.add_parser(
        "shapes", help="a Hub checkpoint's files, config and tensor shapes"
    )
    listing.add_argument("repo", help="a Hub model repository, as OWNER/NAME")
    listing.add_argument("--revision", default="main")

    encoder = commands.add_parser(
        "entrance", help="the diagonal share of a dense feature encoder"
    )
    encoder.add_argument("repo", help="a Hub model repository, as OWNER/NAME")
    encoder.add_argument("--revision", default="main")
    encoder.add_argument("--tensor", default="fc.weight")
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "shapes":
        lines = shapes.report(args.repo, args.revision)
    else:
        lines = entrance.report(args.repo, args.revision, args.tensor)
    # Bytes, so that a saved report is UTF-8 with LF on every platform.
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
