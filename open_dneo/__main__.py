"""The command line: `open-dneo` and one of shapes, entrance, recording, greedy, score, rows, probe,
bridge, train."""

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
    scored.add_argument(
        "--bridge",
        help="a file `open-dneo bridge` wrote: the entrance is DNeo's weighted sum,"
        " bridged into the drafter's layers",
    )
    scored.add_argument(
        "--bridge-served",
        action="store_true",
        help="with --served, the bridge at 4 bits too (it stays float32 without)",
    )
    scored.add_argument(
        "--dneo-mask",
        action="store_true",
        help="DNeo's mask: a block row sees the context and its block's first row only",
    )
    scored.add_argument(
        "--dneo-head",
        type=int,
        metavar="K",
        help="DNeo's head carried from DFlash2's: 0 over the whole vocabulary, K over"
        " the target head's best K",
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
    own.add_argument(
        "--top",
        type=int,
        default=0,
        help="with --save, also each row's best TOP tokens and their log"
        " probabilities (<stream>.best) and the next token's (<stream>.nextlp)",
    )
    own.set_defaults(
        run=lambda a: _with_torch("greedy").report(
            Path(a.folder),
            Path(a.target),
            a.device,
            a.streams,
            Path(a.save) if a.save else None,
            a.top,
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
            None if a.bridge is None else Path(a.bridge),
            a.bridge_served,
            a.dneo_mask,
            a.dneo_head,
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

    bridged = commands.add_parser(
        "bridge",
        help="DNeo's entrance fitted onto a DFlash2 drafter's layers, from recorded rows",
    )
    bridged.add_argument(
        "folders", nargs="+", help="--record folders, or any folders of streams"
    )
    bridged.add_argument(
        "--drafter",
        required=True,
        help="the DFlash2 drafter's folder: config.json and model.safetensors",
    )
    bridged.add_argument(
        "--out", required=True, help="where bridge-LABEL.safetensors are written"
    )
    bridged.add_argument(
        "--weights",
        action="append",
        help="the sum's weights, a tap each in the drafter's order, as NAME=1,2,... or"
        " 1,2,...: fitted beside the uniform and even ones",
    )
    bridged.add_argument(
        "--parts",
        action="append",
        help="a file of lines: stream name, a tab, its part (again for more)",
    )
    bridged.add_argument("--ridge", type=float, default=1e-3)
    bridged.add_argument("--held-every", type=int, default=10)
    bridged.add_argument("--train-every", type=int, default=4)
    bridged.add_argument(
        "--inputs",
        help="the taps the sum reads, comma-separated, as many as the drafter's"
        " (default: the drafter's own)",
    )
    bridged.set_defaults(
        run=lambda a: _with_torch("bridge").report(
            [Path(folder) for folder in a.folders],
            Path(a.drafter),
            Path(a.out),
            list(a.weights or []),
            [Path(listing) for listing in a.parts or []],
            a.ridge,
            a.held_every,
            a.train_every,
            tuple(int(t) for t in a.inputs.split(",")) if a.inputs else (),
        )
    )

    taught = commands.add_parser(
        "train", help="a drafter trained on recorded rows and the target's choices"
    )
    taught.add_argument(
        "folders", nargs="+", help="--record folders, or any folders of streams"
    )
    taught.add_argument(
        "--next", required=True, help="greedy --save's folder: each stream's choices"
    )
    taught.add_argument(
        "--drafter",
        required=True,
        help="the drafter to start from: a DFlash2 folder, or a trained one",
    )
    taught.add_argument(
        "--target", required=True, help="the target checkpoint's folder"
    )
    taught.add_argument("--out", required=True, help="where step-N folders are kept")
    taught.add_argument("--steps", type=int, required=True)
    taught.add_argument("--bridge", help="as score's: DNeo's entrance, bridged")
    taught.add_argument("--dneo-mask", action="store_true", help="as score's")
    taught.add_argument("--dneo-head", type=int, metavar="K", help="as score's")
    taught.add_argument(
        "--parts",
        action="append",
        help="a file of lines: stream name, a tab, its part (again for more)",
    )
    taught.add_argument(
        "--shares", help="each part's share of a step, as part=1,part=2 (default: rows)"
    )
    taught.add_argument(
        "--rates",
        help="a factor of the rate for parts of the drafter, as layers=0.1,head=1"
        " (default: 1 for every part)",
    )
    taught.add_argument("--held-every", type=int, default=10)
    taught.add_argument(
        "--resume",
        action="store_true",
        help="--drafter is a step-N folder of a run with the same settings: go on from it",
    )
    taught.add_argument("--device", default="cpu", help="where torch computes")
    for name, kind in (
        ("peak", float), ("end", float), ("warmup", int), ("decay", float),
        ("off-path", float), ("anchors", int), ("span", int), ("clip", float),
        ("held-anchors", int), ("held-streams", int), ("log-every", int),
        ("measure-every", int), ("keep-every", int), ("seed", int),
    ):  # fmt: skip
        taught.add_argument(f"--{name}", type=kind, help="see trainer.Settings")
    taught.set_defaults(run=_train)
    return top


def _train(a: argparse.Namespace) -> tuple[list[str], bool]:
    """`train`'s arguments to a run: the numbers not given keep Settings' defaults."""
    trainer = _with_torch("trainer")
    names = (
        "peak", "end", "warmup", "decay", "off_path", "anchors", "span", "clip",
        "held_anchors", "held_streams", "log_every", "measure_every", "keep_every",
        "seed",
    )  # fmt: skip
    given = {name: getattr(a, name) for name in names if getattr(a, name) is not None}
    settings = trainer.Settings(
        steps=a.steps,
        shares=trainer.shares(a.shares),
        rates=trainer.rates(a.rates),
        **given,
    )
    return trainer.command(
        [Path(folder) for folder in a.folders],
        Path(a.next),
        Path(a.drafter),
        Path(a.target),
        Path(a.out),
        settings,
        parts=[Path(listing) for listing in a.parts or []],
        bridge=None if a.bridge is None else Path(a.bridge),
        dneo_mask=a.dneo_mask,
        dneo_head=a.dneo_head,
        held_every=a.held_every,
        device=a.device,
        resume=a.resume,
    )


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
