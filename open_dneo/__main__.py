"""The command line: `open-dneo` and one of shapes, entrance, recording, greedy, score, rows, probe."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import __version__, entrance, recording, shapes


def _with_torch(name: str):
    """A module of this package that computes with torch, loaded when it is used.

    The other commands run without torch installed.
    """
    try:
        return importlib.import_module(f"{__package__}.{name}")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        sys.exit("this command computes with torch: pip install 'open-dneo[torch]'")


def parser() -> argparse.ArgumentParser:
    """Each command sets `run`: its arguments to its report's lines and soundness."""
    top = argparse.ArgumentParser(prog="open-dneo", description=__doc__)
    top.add_argument("--version", action="version", version=__version__)
    commands = top.add_subparsers(dest="command", required=True)

    listing = commands.add_parser(
        "shapes", help="a Hub checkpoint's files, config and tensor shapes"
    )
    listing.add_argument("repo", help="a Hub model repository, as OWNER/NAME")
    listing.add_argument("--revision", default="main")
    listing.set_defaults(run=lambda a: (shapes.report(a.repo, a.revision), True))

    encoder = commands.add_parser(
        "entrance", help="the diagonal share of a dense feature encoder"
    )
    encoder.add_argument("repo", help="a Hub model repository, as OWNER/NAME")
    encoder.add_argument("--revision", default="main")
    encoder.add_argument("--tensor", default="fc.weight")
    encoder.set_defaults(
        run=lambda a: (entrance.report(a.repo, a.revision, a.tensor), True)
    )

    recorded = commands.add_parser(
        "recording", help="check the rows a TensorFold server recorded with --record"
    )
    recorded.add_argument(
        "folder", help="the --record folder, or any folder of streams"
    )
    recorded.add_argument("--streams", action="store_true", help="list every stream")
    recorded.set_defaults(run=lambda a: recording.report(Path(a.folder), a.streams))

    own = commands.add_parser(
        "greedy", help="how much of a recording its target would have written itself"
    )
    scored = commands.add_parser(
        "score", help="a DFlash2 drafter's tokens a round over recorded replies"
    )
    scored.add_argument(
        "--drafter",
        required=True,
        help="the drafter's folder: config.json and model.safetensors",
    )
    scored.add_argument(
        "--served",
        action="store_true",
        help="the drafter's linears and the head at 4 bits and the head's scores in"
        " bfloat16, as serving holds them",
    )
    scored.add_argument(
        "--max-tokens",
        type=int,
        help="the requests' limit of new tokens, where the replies were cut by it",
    )
    scored.add_argument(
        "--run",
        dest="run_file",  # `run` is each command's own function
        help="a run file: each request paired with its stream, serving's tokens a"
        " round beside the instrument's, class by class",
    )
    scored.add_argument(
        "--drafts-log",
        help="with --run, a server's log with its DFlash2 rounds: the drafts serving"
        " logged beside the instrument's at the same anchors, class by class",
    )
    for command in (own, scored):
        command.add_argument(
            "folder", help="the --record folder, or any folder of streams"
        )
        command.add_argument(
            "--target", required=True, help="the target checkpoint's folder"
        )
        command.add_argument("--device", default="cpu", help="where torch computes")
        command.add_argument("--streams", action="store_true", help="list every stream")
    own.add_argument(
        "--save", help="a folder to keep each stream's choices in, <stream>.next"
    )
    own.set_defaults(
        run=lambda a: _with_torch("greedy").report(
            Path(a.folder),
            Path(a.target),
            a.device,
            a.streams,
            Path(a.save) if a.save else None,
        )
    )
    scored.set_defaults(
        run=lambda a: _with_torch("score").report(
            Path(a.folder),
            Path(a.drafter),
            Path(a.target),
            a.device,
            a.streams,
            a.served,
            a.max_tokens,
            None if a.run_file is None else Path(a.run_file),
            None if a.drafts_log is None else Path(a.drafts_log),
        )
    )
    paired = commands.add_parser(
        "rows",
        help="a DFlash2 drafter's proposals from reply rows against prompt rows of the"
        " same tokens",
    )
    paired.add_argument("folder", help="the --record folder with both kinds of stream")
    paired.add_argument(
        "--drafter",
        required=True,
        help="the drafter's folder: config.json and model.safetensors",
    )
    paired.add_argument(
        "--target", required=True, help="the target checkpoint's folder"
    )
    paired.add_argument(
        "--run",
        dest="run_file",
        required=True,
        help="the run file of the replies, each sent again as a prompt",
    )
    paired.add_argument(
        "--served",
        action="store_true",
        help="the drafter's linears and the head at 4 bits and the head's scores in"
        " bfloat16, as serving holds them",
    )
    paired.add_argument("--device", default="cpu", help="where torch computes")
    paired.set_defaults(
        run=lambda a: _with_torch("rows").report(
            Path(a.folder),
            Path(a.drafter),
            Path(a.target),
            Path(a.run_file),
            a.device,
            a.served,
        )
    )

    probed = commands.add_parser(
        "probe", help="where a recording's layers hold the target's next tokens"
    )
    probed.add_argument("folder", help="the --record folder, its streams' taps alike")
    probed.add_argument(
        "--target", required=True, help="the target checkpoint's folder"
    )
    probed.add_argument(
        "--next", required=True, help="greedy --save's folder: each stream's choices"
    )
    probed.add_argument("--parts", help="a file of lines: stream name, a tab, its part")
    probed.add_argument(
        "--k", type=int, action="append", help="positions ahead (default 1, 2 and 3)"
    )
    probed.add_argument(
        "--set",
        action="append",
        default=[],
        help="taps fitted side by side, comma-separated, e.g. 6,15,25,34,43",
    )
    probed.add_argument("--ridge", type=float, default=1e-3)
    probed.add_argument("--held-every", type=int, default=10)
    probed.add_argument("--train-every", type=int, default=4)
    probed.add_argument("--eval-every", type=int, default=8)
    probed.add_argument("--device", default="cpu", help="where torch computes")
    probed.set_defaults(
        run=lambda a: _with_torch("probe").report(
            Path(a.folder),
            Path(a.target),
            Path(a.next),
            tuple(a.k or (1, 2, 3)),
            tuple(tuple(int(t) for t in s.split(",")) for s in a.set),
            Path(a.parts) if a.parts else None,
            a.ridge,
            a.held_every,
            a.train_every,
            a.eval_every,
            a.device,
        )
    )
    return top


def main(argv: list[str] | None = None) -> int:
    """Print a command's report; 1 if the report found something wrong."""
    args = parser().parse_args(argv)
    lines, sound = args.run(args)
    # Bytes, so that a saved report is UTF-8 with LF on every platform.
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0 if sound else 1


if __name__ == "__main__":
    sys.exit(main())
