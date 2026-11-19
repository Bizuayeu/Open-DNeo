"""A drafter's proposals along a recorded stream, and the rounds they make.

A recording holds what a drafter reads while its target writes a reply: each
row's tapped states, its token and its position. `proposals` takes every reply
row as an anchor: the block is the anchor's token and mask tokens, embedded as
the target embeds them, and its context is the stream's rows before the anchor.
That is what serving's drafter holds at a round that starts there, for a stream
decoded alone. `steps` walks a reply over them as `rounds.walk` counts, every
round the drafter's, and `lanes` as `rounds.served` does, some rounds copies.

A stream's rows begin at `first_row`: a prompt that resumed from kept state has
no rows before it, and serving's drafter had none either, in that launch. The
proposals carry `first_row`, so that two launches are compared only where their
contexts begin at the same row.

`report` can score with the weights as serving holds them (`served`): its 4-bit
linears and its 4-bit copy of the target's head, whose scores it keeps in bfloat16.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import drafter, recording, rounds, served, weights

# Anchors a pass unless told otherwise: at a vocabulary of 154,880 and 7 drafts an
# anchor, the scores of 32 anchors are 139 MB of float32, and a head holds a few
# tensors of that size at once.
AT_ONCE = 32


@dataclass(frozen=True)
class Target:
    """What a drafter borrows from its target, on the drafter's device and type."""

    embed: torch.Tensor  # [vocab, hidden]: a token's row as the target takes it in
    head: torch.Tensor  # [vocab, hidden]: a state's scores are its products with these
    bf16_scores: bool = False  # the scores rounded to bfloat16, as serving keeps them

    @classmethod
    def read(cls, folder: Path) -> Target:
        """A GLM-5.3-Flash checkpoint's embedding and head, as float32.

        The head is the values the target's own scores come from. Serving's
        drafts are scored on a 4-bit copy of them.
        """
        return cls(weights.embedding(folder), weights.head(folder))

    def scores(self, states: torch.Tensor) -> torch.Tensor:
        """The head's scores [..., vocab] on `states` [..., hidden].

        One product of two matrices. A product of more dimensions can take a path
        that copies the head once a leading row: for 32 anchors and a head of
        2.5 GB, an aarch64 build of torch asked for 81 GB.
        """
        scores = states.reshape(-1, states.shape[-1]) @ self.head.T
        if self.bf16_scores:
            scores = scores.to(torch.bfloat16).to(scores.dtype)
        return scores.unflatten(0, states.shape[:-1])


@dataclass(frozen=True)
class Reading:
    """How a drafter reads a stream."""

    taps: tuple[int, ...]  # the recording's taps it takes, in its entrance's order
    mask_token: int
    whole_block: bool  # a block's rows see all of it (DFlash2), or its first row (DNeo)
    window: int | None  # DFlash2's sliding window

    @classmethod
    def dflash2(cls, config: dict) -> Reading:
        """From a DFlash2 checkpoint's config.json.

        The checkpoint numbers the target layers whose outputs it reads from 0,
        and a recording's tap k is the row after k layers: layer i is tap i + 1.
        """
        own = config["dflash_config"]
        kinds = set(config["layer_types"])
        if config["is_causal"] or len(kinds) != 1:
            raise ValueError(
                "only layers that all see the whole block the same way are implemented"
            )
        scale = own.get("input_embedding_scale", config.get("input_embedding_scale", 1))
        if scale != 1:
            raise ValueError("only block embeddings without a scale are implemented")
        return cls(
            taps=tuple(layer + 1 for layer in own["target_layer_ids"]),
            mask_token=own["mask_token_id"],
            whole_block=True,
            window=config["sliding_window"] if "sliding_attention" in kinds else None,
        )


@dataclass(frozen=True, eq=False)
class Proposals:
    """What a drafter proposes at each reply row of a stream, taken as an anchor."""

    first_row: int  # where the stream's rows, and so every context, begin
    positions: np.ndarray  # [anchors] i64: each anchor's position
    tokens: np.ndarray  # [anchors] i64: each anchor's token
    drafts: np.ndarray  # [anchors, block - 1] i64: for the positions after it
    chances: np.ndarray  # [anchors, block - 1] f64


@torch.no_grad()
def proposals(
    model: drafter.Drafter,
    target: Target,
    stream: recording.Stream,
    reading: Reading,
    *,
    at_once: int = AT_ONCE,
    temperature: float = 1.0,
) -> Proposals:
    """The drafts and chances at every reply row of `stream`.

    `at_once` anchors go through the model in one pass; the result does not
    depend on it. ValueError if the stream is not what `reading` and the model
    take: a tap it does not hold, rows of another width, or rows that are not at
    consecutive positions (`recording.check` finds those).
    """
    summary, shape = stream.summary, model.shape
    if at_once < 1:
        raise ValueError(f"at_once is {at_once}: a pass takes 1 anchor or more")
    if len(reading.taps) != shape.taps:
        raise ValueError(
            f"the reading names {len(reading.taps)} taps, and the model's entrance"
            f" takes {shape.taps}"
        )
    missing = [tap for tap in reading.taps if tap not in summary.taps]
    if missing:
        raise ValueError(f"the stream holds taps {list(summary.taps)}, not {missing}")
    if summary.rows and summary.width() != shape.hidden:
        raise ValueError(
            f"the stream's rows are {summary.width()} wide, the model's {shape.hidden}"
        )
    columns = [summary.taps.index(tap) for tap in reading.taps]
    first = summary.first_row
    pos = np.asarray(stream.pos, np.int64)
    if not stream.in_order():
        raise ValueError(
            f"the stream's rows are not at consecutive positions from {first}"
        )
    replies = np.flatnonzero(np.asarray(stream.kind) == 1)
    positions, tokens = pos[replies], np.asarray(stream.tok, np.int64)[replies]
    like = model.norm.weight
    drafts = [np.empty((0, shape.block - 1), np.int64)]
    chances = [np.empty((0, shape.block - 1), np.float64)]
    for start in range(0, len(replies), at_once):
        at = positions[start : start + at_once]
        # the rows any of these anchors' blocks can see: before the last anchor,
        # and inside the first anchor's window
        low, high = first, int(at[-1])
        if reading.window is not None:
            low = max(low, int(at[0]) - reading.window + 1)
        rows = stream.tapped(low - first, high - first)[:, columns]
        features = torch.from_numpy(rows).to(like.device, like.dtype)[None]
        context_positions = torch.arange(low, high, device=like.device)[None]
        anchors = torch.from_numpy(at).to(like.device)[None]
        block = np.full((len(at), shape.block), reading.mask_token, np.int64)
        block[:, 0] = tokens[start : start + at_once]
        ids = torch.from_numpy(block).to(like.device)
        mask = drafter.visible(
            anchors,
            context_positions,
            shape.block,
            whole_block=reading.whole_block,
            window=reading.window,
        )
        hidden = model(
            target.embed[ids][None], features, anchors, context_positions, mask
        )
        # laid out anew: a head's own matrix is folded into one product only on
        # rows that lie in order (`Target.scores`)
        states = hidden[0, :, 1:].contiguous()
        path, _, chance = model.head(
            states, target.scores(states), ids[:, 0], temperature
        )
        drafts.append(path.cpu().numpy().astype(np.int64))
        chances.append(chance.double().cpu().numpy())
    return Proposals(
        first, positions, tokens, np.concatenate(drafts), np.concatenate(chances)
    )


def steps(
    tokens: Sequence[int], found: Proposals, cut: float = rounds.CUT
) -> list[int]:
    """The tokens each round gives over a reply (`rounds.walk`), from its first on.

    `tokens` are the reply's, in order. `found.tokens` are those the target read:
    a reply's last token has a row only when the reply ended inside a round, so
    the stream may hold one token fewer than the reply. Pass the reply's own
    tokens where they are known, or `found.tokens` to count without the last.
    ValueError if `tokens` are not the stream's with at most one more.
    """
    anchors = found.tokens.tolist()
    shared = min(len(tokens), len(anchors))
    if list(tokens[:shared]) != anchors[:shared]:
        at = next(i for i in range(shared) if tokens[i] != anchors[i])
        raise ValueError(
            f"reply token {at} is {tokens[at]}, and the stream's is {anchors[at]}"
        )
    if len(tokens) > len(anchors) + 1:
        raise ValueError(
            f"the reply has {len(tokens)} tokens, and the stream rows for"
            f" {len(anchors)}: more than one is without a row"
        )

    def propose(at: int) -> tuple[list[int], list[float]]:
        return found.drafts[at].tolist(), found.chances[at].tolist()

    return rounds.walk(tokens, propose, cut)


def lanes(
    stream: recording.Stream,
    found: Proposals,
    max_new: int | None = None,
    first_depth: int = rounds.Lanes.first_depth,
) -> list[rounds.Round]:
    """The rounds serving takes over a stream's reply (`rounds.served`), copies included.

    The prompt is the stream's own rows before the reply: where they do not begin
    at row 0, the copies of what came before are missed. The drafts a round asks
    for are the drafter's block less its anchor; `max_new` is the request's limit
    of new tokens, where the replies were cut by it. `first_depth` is the drafts
    the first round holds, which serving chooses from measured round times.
    """
    prompt = np.asarray(stream.tok, np.int64)[np.asarray(stream.kind) == 0]

    def propose(at: int) -> tuple[list[int], list[float]]:
        return found.drafts[at].tolist(), found.chances[at].tolist()

    facts = rounds.Lanes(drafts=found.drafts.shape[1], first_depth=first_depth)
    return rounds.served(
        prompt.tolist(), found.tokens.tolist(), propose, facts, max_new
    )


def report(
    root: Path,
    drafter_folder: Path,
    checkpoint: Path,
    device: str = "cpu",
    each: bool = False,
    as_served: bool = False,
    max_new: int | None = None,
) -> tuple[list[str], bool]:
    """A DFlash2 drafter's rounds over the replies recorded under `root`.

    The drafter is its folder's config.json and model.safetensors, the target the
    embedding and head of `checkpoint`. With `as_served` the drafter's linears and
    the head are rounded to 4 bits and the head's scores to bfloat16, as serving
    holds them by default. Each reply is walked over the tokens the stream holds:
    the count leaves out a reply's first token, which no round gives, and a last
    token that the target never read. It is walked twice: every round the
    drafter's (`steps`), and as serving takes the rounds, some of them copies
    (`lanes`, with `max_new`). `each` also lists every stream. Not sound if no
    finished stream could be walked, one could not be read, or the drafter's file
    holds tensors that are not read here.
    """
    config = json.loads((drafter_folder / "config.json").read_text(encoding="utf-8"))
    model = drafter.Drafter.dflash2(drafter.Shape.dflash2(config))
    unread = drafter.load(model, weights.read(drafter_folder / weights.WHOLE))
    read = Target.read(checkpoint)
    target = Target(read.embed.to(device), read.head.to(device))
    lines = ["weights as stored"]
    if as_served:
        served.as_served(model)
        target = Target(target.embed, served.four_bits(target.head), bf16_scores=True)
        lines = [
            "weights as served: the drafter's linears and the head at 4 bits,"
            " the head's scores in bfloat16"
        ]
    model, reading = model.to(device), Reading.dflash2(config)
    lines += [f"the drafter's {name} is not read" for name in unread]
    walked, late, alone, taken, sound = 0, 0, [], [], not unread
    least = most = 0  # rounds with copies if a reply's first round held 1 draft, or all
    for base in recording.find(root):
        name = base.relative_to(root).as_posix()
        if not recording.beside(base, ".json").exists():
            continue  # a stream that has not closed
        try:
            stream = recording.open_stream(base)
            found = proposals(model, target, stream, reading)
        except ValueError as error:
            lines.append(f"{name}: {error}")
            sound = False
            continue
        own, lane = steps(found.tokens, found), lanes(stream, found, max_new)
        walked += 1
        late += found.first_row > 0
        alone += own
        taken += lane
        least += len(lanes(stream, found, max_new, 1))
        most += len(lanes(stream, found, max_new, found.drafts.shape[1]))
        if each:
            counted = "0 rounds"
            if own:
                counted = f"the drafter alone {_alone(own)}; with copies {_taken(lane)}"
            lines.append(f"{name}: {counted}; context from row {found.first_row}")
    if alone:
        lines.append(f"streams {walked}, the drafter alone: {_alone(alone)}")
        lines.append(f"streams {walked}, with copies: {_taken(taken)}")
        lines.append(
            f"streams {walked}, with copies, a first round of 1 draft or of every"
            f" draft: {least} or {most} rounds"
        )
    else:
        lines.append(f"streams {walked}: 0 rounds")
    if late:
        lines.append(
            f"{late} of the streams begin after row 0: copies of the tokens before"
            " it are missed"
        )
    return lines, sound and walked > 0


def _alone(given: list[int]) -> str:
    """Rounds in words: how many, the tokens they give, and the tokens a round."""
    tokens = sum(given)
    return (
        f"{len(given)} rounds give {tokens} tokens, {tokens / len(given):.3f} a round"
    )


def _taken(given: list[rounds.Round]) -> str:
    """The same for rounds as serving takes them, and how many of them are copies."""
    copied = sum(each.kind == rounds.COPY for each in given)
    return f"{_alone([each.tokens for each in given])}, {copied} of them copies"
