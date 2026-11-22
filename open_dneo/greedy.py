"""How much of a recording is what its target would have written itself.

The tap numbered as the target's layers is the row after the final norm, the row
the target's head reads: the head's best score on it is the token the target
writes next. A stream's token that is that token is the target's own greedy
choice. A reply decoded without sampling is that all the way, which checks a head
tensor, its orientation and the tap before a drafter is scored against them. A
prompt's borrowed text is that here and there, and a drafter's labels along it
hold up to the first token that is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from . import recording, weights

# Rows a product: at a vocabulary of 154,880 their scores are 159 MB of float32.
AT_ONCE = 256


def final_tap(folder: Path) -> int:
    """The tap a checkpoint's head reads: its text model's number of layers."""
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    return config.get("text_config", config)["num_hidden_layers"]


class Choices(NamedTuple):
    """What the target's head makes of each row of a stream (`choices`)."""

    written: np.ndarray  # [rows] i64: the token it writes next, its best score
    best: np.ndarray  # [rows, k] i64: its best k tokens, best first
    best_logprobs: np.ndarray  # [rows, k] f32: their log probabilities
    next_logprob: np.ndarray  # [rows] f32: the log probability of the stream's next
    # token, nan after the last row


@torch.no_grad()
def choices(
    stream: recording.Stream,
    head: torch.Tensor,
    tap: int,
    top: int,
    *,
    at_once: int = AT_ONCE,
) -> Choices:
    """What the target writes after each row of `stream`, its best `top` tokens and
    their log probabilities, and its log probability of the token the stream holds
    next, from one product of the head with the rows of `tap`.

    `head` [vocab, hidden] scores the rows, `at_once` rows a product. The log
    probabilities are over the whole vocabulary, as float32. ValueError if the
    stream does not hold the tap at the head's width.
    """
    summary = stream.summary
    if at_once < 1:
        raise ValueError(f"at_once is {at_once}: a product takes 1 row or more")
    if tap not in summary.taps:
        raise ValueError(f"the stream holds taps {list(summary.taps)}, not {tap}")
    if summary.rows and summary.width() != head.shape[1]:
        raise ValueError(
            f"the stream's rows are {summary.width()} wide, the head's {head.shape[1]}"
        )
    column = summary.taps.index(tap)
    following = np.append(np.asarray(stream.tok, np.int64)[1:], 0)
    written, best, logprobs, nexts_lp = [], [], [], []
    for start in range(0, summary.rows, at_once):
        rows = torch.from_numpy(stream.tapped(start, start + at_once)[:, column])
        scores = (rows.to(head.device, head.dtype) @ head.T).float()
        written.append(scores.argmax(-1).cpu().numpy())
        logged = scores - scores.logsumexp(-1, keepdim=True)
        values, ids = logged.topk(top, -1)
        best.append(ids.cpu().numpy())
        logprobs.append(values.cpu().numpy())
        after = torch.from_numpy(following[start : start + at_once]).to(scores.device)
        nexts_lp.append(logged.gather(-1, after[:, None])[:, 0].cpu().numpy())
    if not written:
        empty = np.empty((0, top))
        return Choices(
            np.empty(0, np.int64), empty.astype(np.int64), empty.astype(np.float32),
            np.empty(0, np.float32),
        )  # fmt: skip
    next_logprob = np.concatenate(nexts_lp).astype(np.float32)
    next_logprob[-1] = np.nan
    return Choices(
        np.concatenate(written).astype(np.int64),
        np.concatenate(best).astype(np.int64),
        np.concatenate(logprobs).astype(np.float32),
        next_logprob,
    )


def nexts(
    stream: recording.Stream, head: torch.Tensor, tap: int, *, at_once: int = AT_ONCE
) -> np.ndarray:
    """The token the target writes after each row of `stream` [rows] (`choices`).

    `head` [vocab, hidden] scores the rows of `tap`, `at_once` rows a product.
    ValueError if the stream does not hold the tap at the head's width.
    """
    return choices(stream, head, tap, top=0, at_once=at_once).written


def read_best(path: Path, rows: int, top: int) -> tuple[np.ndarray, np.ndarray]:
    """A `.best` file `report` saved: each row's `top` ids [rows, top] and their log
    probabilities. ValueError if it is not that many values."""
    raw = np.fromfile(path, dtype="<u4")
    if raw.size != rows * 2 * top:
        raise ValueError(
            f"{path} holds {raw.size} values, not {rows} rows of {top} ids and their"
            " log probabilities"
        )
    table = raw.reshape(rows, 2 * top)
    return table[:, :top].astype(np.int64), table[:, top:].view("<f4").copy()


class Own(NamedTuple):
    """How many tokens are the target's own choice, of those counted."""

    reply: int
    reply_counted: int
    prompt: int
    prompt_counted: int

    def line(self) -> str:
        return (
            f"reply {self.reply} of {self.reply_counted} tokens the target's own,"
            f" prompt {self.prompt} of {self.prompt_counted}"
        )


def own(stream: recording.Stream, written: np.ndarray) -> Own:
    """How many of a stream's tokens are the target's own choice.

    `written` is `nexts` of the stream. A token is counted where the row before
    it was recorded, as the reply's or the prompt's by its own row's kind.
    ValueError if the rows are not at consecutive positions.
    """
    if not stream.in_order():
        raise ValueError("the stream's rows are not at consecutive positions")
    same = written[:-1] == np.asarray(stream.tok, np.int64)[1:]
    reply = np.asarray(stream.kind)[1:] == 1
    return Own(
        reply=int((same & reply).sum()),
        reply_counted=int(reply.sum()),
        prompt=int((same & ~reply).sum()),
        prompt_counted=int((~reply).sum()),
    )


def own_path(
    tokens: np.ndarray, written: np.ndarray, anchors: np.ndarray, drafts: int
) -> np.ndarray:
    """How many of each anchor's first `drafts` drafts have labels on the target's
    own path [anchors].

    `tokens` [rows] are a stream's tokens at consecutive positions, `written` [rows]
    the target's choice after each row (`nexts`) and `anchors` rows. Draft j of an
    anchor at row t (j from 1) is labelled with the choice after row t + j - 1. That
    is the token serving checks it against while the tokens at rows t + 1 to
    t + j - 1 are the target's own choices: the first that is not leaves the path,
    and every later label follows it. A label needs the row before it, so none lies
    past the last row: the run of choices stops there. ValueError if the rows do not
    line up.
    """
    tokens, written = np.asarray(tokens, np.int64), np.asarray(written, np.int64)
    anchors, rows = np.asarray(anchors, np.int64), len(tokens)
    if len(written) != rows:
        raise ValueError(f"{len(written)} choices for {rows} tokens")
    if drafts < 1:
        raise ValueError(f"drafts is {drafts}: a block drafts 1 token or more")
    outside = anchors[(anchors < 0) | (anchors >= rows)]
    if len(outside):
        raise ValueError(f"no row at {outside.tolist()}")
    same = written[:-1] == tokens[1:]  # [rows - 1]: the target chose row i + 1's token
    # from row i on, how many of those choices in a row: up to the first that is not
    stops = np.where(same, len(same), np.arange(len(same)))
    first_not = np.minimum.accumulate(stops[::-1])[::-1]
    run = np.append(first_not - np.arange(len(same)), 0)
    return np.minimum(run[anchors] + 1, drafts)


def report(
    root: Path,
    checkpoint: Path,
    device: str = "cpu",
    each: bool = False,
    save: Path | None = None,
    top: int = 0,
) -> tuple[list[str], bool]:
    """How much of the recordings under `root` is the checkpoint's own writing.

    `each` also lists every stream. `save` keeps each counted stream's `nexts`
    as little-endian int32 in `save/<stream>.next`, a token a row, for counts by
    part of a stream and for probes and training that need the target's choice at
    a row. With `top` it also keeps `<stream>.best`, each row's best `top` ids
    (u32) then their log probabilities (f32), as the recorder's `.top` lays them
    out, and `<stream>.nextlp`, the log probability of the stream's next token
    after each row (f32, nan after the last): what training's other losses and
    weights read (`choices`). Not sound if no finished stream could be counted, or
    one could not be read.
    """
    tap = final_tap(checkpoint)
    head = weights.head(checkpoint).to(device)
    lines, counted, sound = [], [], True
    for base in recording.find(root):
        name = base.relative_to(root).as_posix()
        if not recording.beside(base, ".json").exists():
            continue  # a stream that has not closed
        try:
            stream = recording.open_stream(base)
            read = choices(stream, head, tap, top)
            written = read.written
            counts = own(stream, written)
        except ValueError as error:
            lines.append(f"{name}: {error}")
            sound = False
            continue
        counted.append(counts)
        if save is not None:
            target = save / f"{name}.next"
            target.parent.mkdir(parents=True, exist_ok=True)
            written.astype("<i4").tofile(target)
            if top:
                table = np.concatenate(
                    [
                        read.best.astype("<u4"),
                        read.best_logprobs.astype("<f4").view("<u4"),
                    ],
                    axis=1,
                )
                table.tofile(save / f"{name}.best")
                read.next_logprob.astype("<f4").tofile(save / f"{name}.nextlp")
        lines += [f"{name}: {counts.line()}"] if each else []
    total = Own(*np.sum(counted, axis=0).tolist()) if counted else Own(0, 0, 0, 0)
    lines.append(f"streams {len(counted)}, tap {tap}: {total.line()}")
    return lines, sound and bool(counted)
