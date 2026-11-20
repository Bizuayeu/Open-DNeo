"""A drafter's proposals from a reply's rows against a prompt's rows of the same tokens.

A server writes a reply's rows one round at a time, and a prompt's in one pass.
For the same tokens they differ a little (the receipt of 2026-11-19 measured 9 to
11% at the deeper taps). Training rows are cheaper to record from prompts, so
Stage 1 asks what the difference does to the drafts.

The run file names the replies (`runs`). Each reply was also sent again as a
prompt: its prompt and its reply as one prompt's tokens. `report` pairs each
request with its reply's stream, as `score` does, and with a prompt's stream
whose prompt rows are those tokens. Both are read from the later of their first
rows (a prompt can resume from a stored prefix), and the drafter proposes at the
reply's anchors from each. A reply is left out where the prompt's stream has no
row for one of its anchors. Each side is walked as `score.steps` walks it, every
round the drafter's: copies would be the same on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import logged, recording, runs, score


def agreement(mine: score.Proposals, other: score.Proposals) -> logged.Agreement:
    """`other`'s drafts against `mine` at `other`'s anchors (`logged.compare`)."""
    rounds = [
        logged.Logged(0, int(p), int(t), tuple(int(d) for d in drafts))
        for p, t, drafts in zip(
            other.positions, other.tokens, other.drafts, strict=True
        )
    ]
    return logged.compare(rounds, mine.positions, mine.tokens, mine.drafts)


def _prompted(
    request: runs.Request,
    streams: Sequence[tuple[str, recording.Stream]],
    taken: set[str],
) -> str | None:
    """The first stream not `taken` whose prompt is `request`'s prompt and reply."""
    whole = (*request.prompt, *request.reply)
    for name, stream in streams:
        summary = stream.summary
        if name in taken:
            continue
        kept = summary.prompt_len - summary.first_row
        if tuple(stream.tok[:kept].tolist()) == whole[summary.first_row :]:
            return name
    return None


def report(
    root: Path,
    drafter_folder: Path,
    checkpoint: Path,
    run: Path,
    device: str = "cpu",
    as_served: bool = False,
) -> tuple[list[str], bool]:
    """Reply rows beside prompt rows of the same tokens, class by class.

    Not sound if no reply could be compared, a stream could not be read, or the
    drafter's file holds tensors that are not read here.
    """
    model, target, reading, lines, unread = score.loaded(
        drafter_folder, checkpoint, device, as_served
    )
    named = [
        (base.relative_to(root).as_posix(), base)
        for base in recording.find(root)
        if recording.beside(base, ".json").exists()
    ]
    opened = [(name, recording.open_stream(base)) for name, base in named]
    streams = dict(opened)
    requests = runs.read(run)
    paired = runs.pair(requests, opened)
    taken = {name for _, name in paired if name is not None}
    pairs, missing = [], []
    for request, name in paired:
        if name is None:
            missing.append(f"{request.id}: no stream holds its prompt and reply")
            continue
        again = _prompted(request, opened, taken)
        if again is None:
            missing.append(
                f"{request.id}: no prompt's stream holds its prompt and reply"
            )
            continue
        taken.add(again)
        pairs.append((request, streams[name], streams[again]))
    lines.append(
        f"run: {len(requests)} requests,"
        f" {sum(name is not None for _, name in paired)} with their reply's stream,"
        f" {len(pairs)} with a prompt's stream of the same tokens"
    )
    lines += missing
    classes: dict[str, list] = {}
    for request, replied, prompted in pairs:
        held = classes.setdefault(request.klass, [0, 0, [], [], logged.Agreement()])
        held[0] += 1
        anchors = replied.pos[replied.kind == 1].astype(np.int64)
        start = max(replied.summary.first_row, prompted.summary.first_row)
        rows_held = prompted.summary.first_row + prompted.summary.rows
        if not len(anchors) or start >= anchors[0] or anchors[-1] >= rows_held:
            held[1] += len(anchors)  # no prompt row, or no context, for an anchor
            continue
        mine = score.proposals(model, target, replied, reading, context_from=start)
        other = score.proposals(
            model, target, prompted, reading, anchors=anchors, context_from=start
        )
        reply = list(request.reply)
        held[2] += score.steps(reply, mine)
        held[3] += score.steps(reply, other)
        held[4] += agreement(mine, other)
    for klass, (replies, without, alone, prompt, agreed) in classes.items():
        if not alone:
            lines.append(
                f"{klass}: {replies} replies, no anchor with both rows, {without}"
                " without"
            )
            continue
        lines.append(
            f"{klass}: {replies} replies, {agreed.rounds} anchors with both rows,"
            f" {without} without; the drafter alone over reply rows {sum(alone)}"
            f" tokens in {len(alone)} rounds, {sum(alone) / len(alone):.3f} a round;"
            f" over prompt rows {sum(prompt)} in {len(prompt)},"
            f" {sum(prompt) / len(prompt):.3f} a round; the first draft the same at"
            f" {agreed.first} anchors ({agreed.first / agreed.rounds:.3f}), every"
            f" draft at {agreed.whole}"
        )
    compared = any(agreed.rounds for *_, agreed in classes.values())
    return lines, compared and not unread
