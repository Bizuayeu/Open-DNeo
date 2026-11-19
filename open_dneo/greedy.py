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


@torch.no_grad()
def nexts(
    stream: recording.Stream, head: torch.Tensor, tap: int, *, at_once: int = AT_ONCE
) -> np.ndarray:
    """The token the target writes after each row of `stream` [rows].

    `head` [vocab, hidden] scores the rows of `tap`, `at_once` rows a product.
    ValueError if the stream does not hold the tap at the head's width.
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
    column, best = summary.taps.index(tap), [np.empty(0, np.int64)]
    for start in range(0, summary.rows, at_once):
        rows = torch.from_numpy(stream.tapped(start, start + at_once)[:, column])
        scores = rows.to(head.device, head.dtype) @ head.T
        best.append(scores.argmax(-1).cpu().numpy())
    return np.concatenate(best)


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


def report(
    root: Path, checkpoint: Path, device: str = "cpu", each: bool = False
) -> tuple[list[str], bool]:
    """How much of the recordings under `root` is the checkpoint's own writing.

    `each` also lists every stream. Not sound if no finished stream could be
    counted, or one could not be read.
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
            counts = own(stream, nexts(stream, head, tap))
        except ValueError as error:
            lines.append(f"{name}: {error}")
            sound = False
            continue
        counted.append(counts)
        lines += [f"{name}: {counts.line()}"] if each else []
    total = Own(*np.sum(counted, axis=0).tolist()) if counted else Own(0, 0, 0, 0)
    lines.append(f"streams {len(counted)}, tap {tap}: {total.line()}")
    return lines, sound and bool(counted)
