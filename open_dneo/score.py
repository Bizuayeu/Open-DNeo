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
contexts begin at the same row. Serving's copies still look in the whole prompt,
which `lanes` takes where the request's tokens are known.

`report` can score with the weights as serving holds them (`served`): its 4-bit
linears and its 4-bit copy of the target's head, whose scores it keeps in bfloat16.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from . import bridge as bridging
from . import drafter, logged, recording, rounds, runs, served, weights

# A trained drafter's folder names its form here (`train.save`).
TRAINED = "train.json"
# Anchors a pass unless told otherwise: at a vocabulary of 154,880 and 7 drafts an
# anchor, the scores of 32 anchors are 139 MB of float32, and a head holds a few
# tensors of that size at once.
AT_ONCE = 32
# The cuts a run's report sweeps, the default among them as it is written.
CUTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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
    # the target head's best k (the drafter's top_k) on the drafter's state:
    # [anchors, block - 1] i8: at each draft, hold the token of the row it drafts,
    # 1 or 0, or -1 where that row is past the stream's last
    next_among: np.ndarray | None = None
    draft_among: np.ndarray | None = None  # [anchors] bool: hold the first draft


@torch.no_grad()
def proposals(
    model: drafter.Drafter,
    target: Target,
    stream: recording.Stream,
    reading: Reading,
    *,
    at_once: int = AT_ONCE,
    temperature: float = 1.0,
    anchors: np.ndarray | None = None,
    context_from: int | None = None,
) -> Proposals:
    """The drafts and chances at every reply row of `stream`, at each draft
    whether the target head's best k (the drafter's `top_k`, where it has one) on
    its state hold the token of the row it drafts, and at the first whether they
    hold the head's own draft.

    `at_once` anchors go through the model in one pass; the result does not
    depend on it. `anchors` takes the rows at those positions as the anchors in
    place of the reply rows, of any kind: a prompt's rows of the same tokens.
    `context_from` leaves out the rows before that position, so that two streams
    are read from the same row. ValueError if the stream is not what `reading` and
    the model take: a tap it does not hold, rows of another width, rows that are
    not at consecutive positions (`recording.check` finds those), or an anchor the
    stream has no row for.
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
    if anchors is not None:
        wanted = np.asarray(anchors, np.int64)
        outside = wanted[(wanted < first) | (wanted >= first + len(pos))]
        if len(outside):
            raise ValueError(f"no row at positions {outside.tolist()}")
        replies = wanted - first
    origin = first  # the stream's first row: where row 0 of its files lies
    if context_from is not None:
        first = max(first, context_from)
    every = np.asarray(stream.tok, np.int64)
    positions, tokens = pos[replies], every[replies]
    like = model.norm.weight
    drafts = [np.empty((0, shape.block - 1), np.int64)]
    chances = [np.empty((0, shape.block - 1), np.float64)]
    next_among = [np.empty((0, shape.block - 1), np.int8)]
    draft_among = [np.empty(0, bool)]
    for start in range(0, len(replies), at_once):
        at = positions[start : start + at_once]
        # the rows any of these anchors' blocks can see: before the last anchor,
        # and inside the first anchor's window
        low, high = first, int(at[-1])
        if reading.window is not None:
            low = max(low, int(at[0]) - reading.window + 1)
        rows = stream.tapped(low - origin, high - origin)[:, columns]
        features = torch.from_numpy(rows).to(like.device, like.dtype)[None]
        context_positions = torch.arange(low, high, device=like.device)[None]
        anchored = torch.from_numpy(at).to(like.device)[None]
        block = np.full((len(at), shape.block), reading.mask_token, np.int64)
        block[:, 0] = tokens[start : start + at_once]
        ids = torch.from_numpy(block).to(like.device)
        mask = drafter.visible(
            anchored,
            context_positions,
            shape.block,
            whole_block=reading.whole_block,
            window=reading.window,
        )
        hidden = model(
            target.embed[ids][None], features, anchored, context_positions, mask
        )
        # laid out anew: a head's own matrix is folded into one product only on
        # rows that lie in order (`Target.scores`)
        states = hidden[0, :, 1:].contiguous()
        scores = target.scores(states)
        path, _, chance = model.head(states, scores, ids[:, 0], temperature)
        drafts.append(path.cpu().numpy().astype(np.int64))
        chances.append(chance.double().cpu().numpy())
        if shape.top_k:
            best = scores.topk(shape.top_k, -1).indices.cpu().numpy()
            draft_among.append((best[:, 0] == drafts[-1][:, :1]).any(-1))
            after = replies[start : start + at_once, None] + np.arange(1, shape.block)
            nexts = np.where(
                after < len(every), every[np.minimum(after, len(every) - 1)], -1
            )
            held = (best == nexts[..., None]).any(-1)
            next_among.append(np.where(nexts >= 0, held, -1).astype(np.int8))
    among = (
        (np.concatenate(next_among), np.concatenate(draft_among))
        if shape.top_k
        else (None, None)
    )
    return Proposals(
        first,
        positions,
        tokens,
        np.concatenate(drafts),
        np.concatenate(chances),
        *among,
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

    def propose(at: int) -> tuple[list[int], list[float]]:
        return found.drafts[at].tolist(), found.chances[at].tolist()

    return rounds.walk(_reply(tokens, found), propose, cut)


def calibration(
    tokens: Sequence[int], found: Proposals, bins: int = 10
) -> list[tuple[int, int]]:
    """How often drafts land against their chances: (drafts, landed) a bin.

    The cut reads a draft's chance as the chance it lands once the drafts before
    it have (`rounds.held` multiplies them), so each anchor's drafts are counted
    while the ones before them landed, up to the first that does not. The bins
    are `bins` equal parts of [0, 1], a chance of 1 in the last. A draft past the
    reply's end is not counted. `tokens` are the reply's, as `steps` takes them.
    """
    reply, counts = _reply(tokens, found), [[0, 0] for _ in range(bins)]
    for at in range(len(found.tokens)):
        for i, (draft, chance) in enumerate(zip(found.drafts[at], found.chances[at])):
            if at + 1 + i >= len(reply):
                break
            landed, slot = (
                int(draft) == reply[at + 1 + i],
                min(int(chance * bins), bins - 1),
            )
            counts[slot][0] += 1
            counts[slot][1] += landed
            if not landed:
                break
    return [(drafts, landed) for drafts, landed in counts]


class Position(NamedTuple):
    """A draft position over a reply's anchors (`positions`)."""

    reached: int  # anchors whose drafts before it landed, it inside the reply
    landed: int  # of them, those whose draft there landed too: every draft to there
    anchors: int  # anchors with it inside the reply
    chance: float  # their chains' chances to there, summed: products of the chances


def positions(tokens: Sequence[int], found: Proposals) -> list[Position]:
    """How the drafts land at each position of a block, every draft held.

    A draft is reached where the drafts before it in its block landed: a round
    that held them all would verify it there. The cut is left out. Beside it is
    the chance the cut reads, the product of the chances up to the position, of
    every anchor's chain. A draft past the reply's end is not counted. `tokens`
    are the reply's, as `steps` takes them.
    """
    inside, entire, product = _chains(tokens, found)
    before = np.ones_like(entire)
    before[:, 1:] = entire[:, :-1]
    return [
        Position(int(reached), int(landed), int(anchors), float(chance))
        for reached, landed, anchors, chance in zip(
            (before & inside).sum(0),
            entire.sum(0),
            inside.sum(0),
            np.where(inside, product, 0.0).sum(0),
            strict=True,
        )
    ]


def chained(
    tokens: Sequence[int], found: Proposals, bins: int = 10
) -> list[tuple[int, int]]:
    """How chains land whole against the product of their chances: (chains,
    landed) a bin.

    The product of a block's chances up to a draft is what the cut reads as the
    chance that every draft to there lands. Each anchor's chain is counted at
    every position inside the reply, a position after a miss as not landed. The
    bins are `bins` equal parts of [0, 1], a product of 1 in the last. `tokens`
    are the reply's, as `steps` takes them.
    """
    inside, entire, product = _chains(tokens, found)
    slots = np.minimum((product[inside] * bins).astype(np.int64), bins - 1)
    chains = np.bincount(slots, minlength=bins)
    landed = np.bincount(slots, entire[inside].astype(np.float64), minlength=bins)
    return [(int(n), int(round(hit))) for n, hit in zip(chains, landed, strict=True)]


def whole(found: Proposals) -> Proposals:
    """`found` with every chance 1: a chain that every cut holds whole."""
    return dataclasses.replace(found, chances=np.ones_like(found.chances))


def _chains(
    tokens: Sequence[int], found: Proposals
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each anchor and draft [anchors, drafts]: inside the reply, every draft
    to there landed, and the product of the chances to there."""
    reply = np.asarray(_reply(tokens, found), np.int64)
    count, width = found.drafts.shape
    at = np.arange(count)[:, None] + 1 + np.arange(width)[None]
    inside = at < len(reply)
    truth = reply[np.minimum(at, len(reply) - 1)] if len(reply) else at
    hit = inside & (found.drafts == truth)
    return inside, np.cumprod(hit, axis=1).astype(bool), np.cumprod(found.chances, 1)


def _reply(tokens: Sequence[int], found: Proposals) -> list[int]:
    """`tokens` as a reply over `found`: the anchors' tokens, with at most one more."""
    tokens, anchors = list(tokens), found.tokens.tolist()
    shared = min(len(tokens), len(anchors))
    if tokens[:shared] != anchors[:shared]:
        at = next(i for i in range(shared) if tokens[i] != anchors[i])
        raise ValueError(
            f"reply token {at} is {tokens[at]}, and the stream's is {anchors[at]}"
        )
    if len(tokens) > len(anchors) + 1:
        raise ValueError(
            f"the reply has {len(tokens)} tokens, and the stream rows for"
            f" {len(anchors)}: more than one is without a row"
        )
    return tokens


def lanes(
    stream: recording.Stream,
    found: Proposals,
    max_new: int | None = None,
    first_depth: int = rounds.Lanes.first_depth,
    *,
    prompt: Sequence[int] | None = None,
    reply: Sequence[int] | None = None,
    cut: float = rounds.CUT,
    priced: bool = True,
) -> list[rounds.Round]:
    """The rounds serving takes over a stream's reply (`rounds.served`), copies included.

    Serving looks for a copy in the request's whole prompt. `prompt` gives it;
    without it the prompt is the stream's own rows before the reply, and where
    they do not begin at row 0 the copies of what came before are missed. The
    reply is the stream's reply rows, or `reply`, the reply's own tokens with the
    last one a row may lack (`steps`). The drafts a round asks for are the
    drafter's block less its anchor; `max_new` is the request's limit of new
    tokens, where the replies were cut by it. `first_depth` is the drafts the
    first round holds, which serving chooses from measured round times, and
    `cut` the chain's (`rounds.held`). `priced` takes serving's price of a row
    (`rounds.priced`) after the cut, as serving does; without it the cut alone.
    ValueError if `prompt` does not end in the stream's prompt rows or is not as
    long as its prompt, or `reply` is not the stream's (`steps`).
    """
    summary = stream.summary
    own = np.asarray(stream.tok, np.int64)[np.asarray(stream.kind) == 0].tolist()
    if prompt is None:
        prompt = own
    else:
        prompt = list(prompt)
        if len(prompt) != summary.prompt_len:
            raise ValueError(
                f"the prompt has {len(prompt)} tokens, and the stream's"
                f" {summary.prompt_len}"
            )
        tail = prompt[summary.first_row :]
        if tail != own:
            at = next(i for i in range(len(own)) if tail[i] != own[i])
            raise ValueError(
                f"prompt token {summary.first_row + at} is {tail[at]},"
                f" and the stream's is {own[at]}"
            )
    tokens = found.tokens.tolist() if reply is None else _reply(reply, found)

    def propose(at: int) -> tuple[list[int], list[float]]:
        return found.drafts[at].tolist(), found.chances[at].tolist()

    facts = rounds.Lanes(drafts=found.drafts.shape[1], first_depth=first_depth, cut=cut)
    if not priced:
        facts = dataclasses.replace(facts, price_one=None)
    return rounds.served(prompt, tokens, propose, facts, max_new)


def report(
    root: Path,
    drafter_folder: Path,
    checkpoint: Path,
    device: str = "cpu",
    each: bool = False,
    as_served: bool = False,
    max_new: int | None = None,
    run: Path | None = None,
    drafts_log: Path | None = None,
    bridge: Path | None = None,
    bridge_served: bool = False,
    dneo_mask: bool = False,
    dneo_head: int | None = None,
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

    With `run` (a run file, `runs`), each of its requests is paired with its stream
    and walked as serving takes it over the request's whole prompt and reply, to the
    request's own limit of new tokens. A class's line sets serving's tokens a round
    beside the instrument's, as G0 compares them. Not sound if a request has no
    stream or serving gave no rounds. Each class also says how its drafts land by
    position with every draft held (`positions`), how the product of their chances,
    which the cut reads, stands against their landing whole (`chained`), and its
    tokens and drafts verified a round with every draft held and at each of `CUTS`.
    Those are the cut's alone: serving also prices a row, which can drop a draft
    that a low cut keeps.

    With a run, `drafts_log` (a server's log with its DFlash2 rounds, `logged`)
    sets the drafts serving logged beside the instrument's at the same anchors, by
    class. Not sound if a logged anchor is not its stream's token.

    With `bridge` (a file `bridge.report` wrote) the drafter's entrance is DNeo's
    weighted sum bridged into its layers (`bridge.entrance`), its layers, mask and
    head as they are: the bridge's step-0 score, G1. The bridge stays float32 as
    fitted, or with `bridge_served` and `as_served` goes to 4 bits too.

    `dneo_mask` and `dneo_head` turn the drafter towards DNeo's form at step 0, one
    part at a time: a block row sees the context and its block's first row only,
    and the head is DNeo's carried from DFlash2's (`drafter.Folded.from_selector`),
    over the whole vocabulary (0) or the target head's best `dneo_head`.
    """
    model, target, reading, lines, unread = loaded(
        drafter_folder,
        checkpoint,
        device,
        as_served,
        bridge,
        bridge_served,
        dneo_mask=dneo_mask,
        dneo_head=dneo_head,
    )
    if drafts_log is not None and run is None:
        return lines + ["a drafts log is read beside a run file"], False
    if run is not None:
        if max_new is not None:
            return lines + [
                "a run gives each request's own limit of new tokens: not --max-tokens"
            ], False
        more, whole = _by_run(root, run, model, target, reading, drafts_log)
        return lines + more, whole and not unread
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


def loaded(
    drafter_folder: Path,
    checkpoint: Path,
    device: str,
    as_served: bool,
    bridge_file: Path | None = None,
    bridge_served: bool = False,
    *,
    dneo_mask: bool = False,
    dneo_head: int | None = None,
) -> tuple[drafter.Drafter, Target, Reading, list[str], list[str]]:
    """The drafter, the target and their reading, a report's first lines, and the
    names of the drafter's tensors that are not read.

    With `as_served` the drafter's linears and the head are rounded to 4 bits and
    the head's scores to bfloat16, as serving holds them by default. With
    `bridge_file` the entrance is bridged, and `dneo_mask` and `dneo_head` change
    the mask and the head (`report`). ValueError if `dneo_head` is below 0 or past
    the vocabulary.
    """
    config = json.loads((drafter_folder / "config.json").read_text(encoding="utf-8"))
    if (drafter_folder / TRAINED).exists():
        if bridge_file is not None or dneo_mask or dneo_head is not None:
            raise ValueError(
                "a trained drafter carries its own form: no bridge, DNeo mask or DNeo"
                " head beside it"
            )
        model, reading, said = _trained(drafter_folder, config)
        return _held(
            model, reading, checkpoint, device, as_served, bridge_served, said, []
        )
    model = drafter.Drafter.dflash2(drafter.Shape.dflash2(config))
    unread = drafter.load(model, weights.read(drafter_folder / weights.WHOLE))
    reading = Reading.dflash2(config)
    form = []
    if dneo_mask:
        reading = dataclasses.replace(reading, whole_block=False)
        form.append(
            "DNeo's mask (a block row sees the context and its block's first row)"
        )
    if dneo_head is not None:
        if not 0 <= dneo_head <= model.shape.vocab:
            raise ValueError(
                f"dneo_head is {dneo_head}: 0 for the whole vocabulary, or the best k"
                f" of {model.shape.vocab}"
            )
        carried = dataclasses.replace(model.shape, folded=model.shape.rank)
        model.head = drafter.Folded.from_selector(model.head, carried, dneo_head)
        over = (
            f"the target head's best {dneo_head}"
            if dneo_head
            else "the whole vocabulary"
        )
        form.append(f"DNeo's head carried from DFlash2's, over {over}")
    entered = []
    if bridge_file is not None:
        matrix, tap_weights, taps = bridging.read(bridge_file)
        model.entrance = bridging.entrance(model, matrix, tap_weights)
        reading = dataclasses.replace(reading, taps=taps)
        held = "at 4 bits" if as_served and bridge_served else "as fitted (float32)"
        shown = ",".join(f"{w:.3f}" for w in tap_weights)
        kept = (
            "DFlash2's layers"
            if form
            else "DFlash2's layers, mask and head as they are"
        )
        entered.append(
            f"the entrance: DNeo's weighted sum of taps {reading.taps}, weights {shown},"
            f" bridged by {bridge_file.name} {held}; {kept}"
        )
    if form:
        entered.append("the form: " + "; ".join(form))
    return _held(
        model, reading, checkpoint, device, as_served, bridge_served, entered, unread
    )


def _trained(folder: Path, config: dict) -> tuple[drafter.Drafter, Reading, list[str]]:
    """A drafter `train.save` kept, in its form, and the line that says so."""
    form = json.loads((folder / TRAINED).read_text(encoding="utf-8"))
    model = drafter.Drafter.dflash2(drafter.Shape.dflash2(config))
    reading = Reading.dflash2(config)
    said = [f"the drafter: trained to step {form['step']}"]
    if form["taps"] is not None:
        model.entrance = drafter.Bridged(model.shape)
        reading = dataclasses.replace(reading, taps=tuple(form["taps"]))
        said.append(
            f"the entrance: DNeo's weighted sum of taps {reading.taps}, bridged"
        )
    if form["dneo_mask"]:
        reading = dataclasses.replace(reading, whole_block=False)
        said.append("DNeo's mask")
    if form["dneo_head"] is not None:
        carried = dataclasses.replace(model.shape, folded=model.shape.rank)
        model.head = drafter.Folded(carried, form["dneo_head"])
        best = form["dneo_head"]
        said.append(
            "DNeo's head over "
            + (f"the target head's best {best}" if best else "the whole vocabulary")
        )
    model.load_state_dict(weights.read(folder / weights.WHOLE))
    return model, reading, ["; ".join(said)]


def _held(
    model: drafter.Drafter,
    reading: Reading,
    checkpoint: Path,
    device: str,
    as_served: bool,
    bridge_served: bool,
    entered: list[str],
    unread: list[str],
) -> tuple[drafter.Drafter, Target, Reading, list[str], list[str]]:
    """`loaded`'s end: the target read, the weights held as asked, the first lines."""
    read = Target.read(checkpoint)
    target = Target(read.embed.to(device), read.head.to(device))
    lines = ["weights as stored"]
    if as_served:
        served.as_served(model, bridge=bridge_served)
        target = Target(target.embed, served.four_bits(target.head), bf16_scores=True)
        lines = [
            "weights as served: the drafter's linears and the head at 4 bits,"
            " the head's scores in bfloat16"
        ]
    lines += entered
    lines += [f"the drafter's {name} is not read" for name in unread]
    return model.to(device), target, reading, lines, unread


def _by_run(
    root: Path,
    run: Path,
    model: drafter.Drafter,
    target: Target,
    reading: Reading,
    drafts_log: Path | None = None,
) -> tuple[list[str], bool]:
    """The lines of `report` with a run: serving beside the instrument, by class."""
    named = [
        (base.relative_to(root).as_posix(), base)
        for base in recording.find(root)
        if recording.beside(base, ".json").exists()
    ]
    opened = [(name, recording.open_stream(base)) for name, base in named]
    requests = runs.read(run)
    paired = runs.pair(requests, opened)
    streams = dict(opened)
    found = sum(name is not None for _, name in paired)
    lines = [f"run: {len(requests)} requests, {found} paired with a stream"]
    sound = found == len(requests) > 0
    classes: dict[str, list] = {}
    logs: list[list[logged.Logged]] = [[] for _ in requests]
    astray, agreed = 0, {}
    if drafts_log is not None:
        rounds_logged = logged.read(drafts_log)
        logs, astray = logged.assign(rounds_logged, requests)
    for (request, name), its_log in zip(paired, logs, strict=True):
        if name is None:
            lines.append(f"{request.id}: no stream holds its prompt and reply")
            continue
        held = streams[name]
        proposed = proposals(model, target, held, reading)
        if drafts_log is not None:
            try:
                agreed[request.klass] = agreed.get(
                    request.klass, logged.Agreement()
                ) + logged.compare(
                    its_log, proposed.positions, proposed.tokens, proposed.drafts
                )
            except ValueError as error:
                lines.append(f"{request.id}: {error}")
                sound = False

        def walk(depth: int = rounds.Lanes.first_depth) -> list[rounds.Round]:
            return lanes(
                held,
                proposed,
                request.max_tokens,
                depth,
                prompt=request.prompt,
                reply=request.reply,
            )

        def swept(variant: Proposals, cut: float, priced: bool) -> list[rounds.Round]:
            return lanes(
                held,
                variant,
                request.max_tokens,
                prompt=request.prompt,
                reply=request.reply,
                cut=cut,
                priced=priced,
            )

        def sweep(priced: bool) -> list[list[rounds.Round]]:
            return [swept(whole(proposed), rounds.CUT, priced)] + [
                swept(proposed, cut, priced) for cut in CUTS
            ]

        classes.setdefault(request.klass, []).append(
            _Walked(
                request,
                walk(),
                len(walk(1)),
                len(walk(proposed.drafts.shape[1])),
                steps(request.reply, proposed),
                calibration(request.reply, proposed),
                proposed,
                positions(request.reply, proposed),
                chained(request.reply, proposed),
                sweep(False),
                sweep(True),
            )
        )
    timed: list[list[list[rounds.Round]]] = []  # each class's rounds at each of `CUTS`
    replies = 0
    for klass, taken in classes.items():
        try:
            tokens, said = runs.served([each.request for each in taken])
        except ValueError as error:
            lines.append(f"{klass}: {error}")
            sound = False
            continue
        mine = sum(r.tokens for each in taken for r in each.lane)
        walked = sum(len(each.lane) for each in taken)
        serving, instrument = tokens / said, mine / walked
        lines.append(
            f"{klass}: {len(taken)} replies; serving {tokens} tokens over {said} rounds,"
            f" {serving:.3f} a round; the instrument {mine} over {walked},"
            f" {instrument:.3f} a round; apart {abs(serving - instrument):.3f};"
            f" a first round of 1 draft or of every draft:"
            f" {sum(each.least for each in taken)} or"
            f" {sum(each.most for each in taken)} rounds"
        )
        alone = [given for each in taken for given in each.own]
        lines.append(
            f"{klass}: the drafter alone, {_alone(alone) if alone else '0 rounds'}"
        )
        lines.append(f"{klass}: {_calibrated(_summed(e.calibration for e in taken))}")
        placed = [
            Position(*(sum(field) for field in zip(*column, strict=True)))
            for column in zip(*(each.positions for each in taken), strict=True)
        ]
        lines.append(f"{klass}: {_positioned(placed)}")
        lines.append(f"{klass}: {_chained(_summed(e.chained for e in taken), placed)}")
        for priced in (False, True):
            cuts = zip(
                *(each.priced if priced else each.swept for each in taken), strict=True
            )
            merged = [[r for lane in at for r in lane] for at in cuts]
            lines.append(f"{klass}: " + _cuts(merged, priced))
            if priced:
                timed.append(merged[1:])
                replies += len(taken)
        if model.shape.top_k:
            among = np.concatenate([each.proposed.next_among for each in taken])
            heard = among[:, 0][among[:, 0] >= 0]
            drew = np.concatenate([each.proposed.draft_among for each in taken])
            lines.append(
                f"{klass}: first drafts, the target head's best {model.shape.top_k}"
                f" on the drafter's states: the next token among them at"
                f" {_share(int(heard.sum()), len(heard), 'anchors')}, the head's draft at"
                f" {_share(int(drew.sum()), len(drew))}"
            )
            by = ", ".join(
                f"{j} {_share(int((column == 1).sum()), int((column >= 0).sum()))}"
                for j, column in enumerate(among.T, 1)
            )
            lines.append(
                f"{klass}: by position, the target head's best {model.shape.top_k} on"
                f" the drafter's states hold the next token: {by}"
            )
    at_cuts = [[r for at in each for r in at] for each in zip(*timed, strict=True)]
    if any(at_cuts):
        lines.append(_timed(at_cuts, replies))
    if drafts_log is not None:
        lines.append(
            f"drafts log: {len(rounds_logged)} rounds, {astray} held by no request,"
            f" {sum(not each for each in logs)} requests with none"
        )
        for klass, got in agreed.items():
            if not got.rounds:
                lines.append(
                    f"{klass}: no logged round has a row; without a row:"
                    f" {got.unrecorded}"
                )
                continue
            lines.append(
                f"{klass}: serving's logged drafts at {got.rounds} rounds with a row:"
                f" the instrument's first draft at {got.first}"
                f" ({got.first / got.rounds:.3f}), its every draft at {got.whole},"
                f" {got.leading / got.rounds:.3f} drafts a round the instrument's from"
                f" the first on; without a row: {got.unrecorded}"
            )
    return lines, sound


class _Walked(NamedTuple):
    """A run's request as `_by_run` walked it."""

    request: runs.Request
    lane: list[rounds.Round]  # as serving takes the rounds
    least: int  # their count with a first round of 1 draft
    most: int  # or of every draft
    own: list[int]  # the tokens of each round, every round the drafter's
    calibration: list[tuple[int, int]]
    proposed: Proposals
    positions: list[Position]
    chained: list[tuple[int, int]]
    swept: list[list[rounds.Round]]  # every draft held, then at each of `CUTS`
    priced: list[list[rounds.Round]]  # the same with serving's price of a row


def _summed(counts: object) -> list[tuple[int, int]]:
    """Counts by bin, (n, landed) a bin, summed over requests."""
    return [
        (sum(n for n, _ in column), sum(hit for _, hit in column))
        for column in zip(*counts, strict=True)
    ]


def _share(part: int, whole: int, noun: str = "") -> str:
    """A of B (and a noun), then A's share of B where B is not 0."""
    said = f"{part} of {whole}" + (f" {noun}" if noun else "")
    return said + (f" ({part / whole:.3f})" if whole else "")


def _calibrated(counts: list[tuple[int, int]]) -> str:
    """`calibration` in words: each bin's drafts, and the share of them that landed."""
    return "drafts by chance, each after the drafts before it landed: " + _binned(
        counts
    )


def _binned(counts: list[tuple[int, int]]) -> str:
    """Each bin of [0, 1]: its count, and the share of it that landed."""
    bins = len(counts)
    return ", ".join(
        f"{i / bins:.1f}-{(i + 1) / bins:.1f} {n}"
        + (f" landed {landed / n:.3f}" if n else "")
        for i, (n, landed) in enumerate(counts)
    )


def _positioned(placed: list[Position]) -> str:
    """`positions` in words: each position's drafts reached, and the share landed."""
    return "by position, each draft after the ones before it landed: " + ", ".join(
        f"{j} {p.reached}"
        + (f" landed {p.landed / p.reached:.3f}" if p.reached else "")
        for j, p in enumerate(placed, 1)
    )


def _chained(counts: list[tuple[int, int]], placed: list[Position]) -> str:
    """`chained` in words, and by position every anchor's mean product against the
    share of the anchors whose drafts to there all landed."""
    means = ", ".join(
        f"{j} {p.chance / p.anchors:.3f} {p.landed / p.anchors:.3f}"
        if p.anchors
        else f"{j} none"
        for j, p in enumerate(placed, 1)
    )
    return (
        "chains by the product of their chances, against every draft to there"
        f" landed: {_binned(counts)}; by position, every anchor's product and its"
        f" share landed whole: {means}"
    )


def _cuts(swept: list[list[rounds.Round]], priced: bool = False) -> str:
    """Tokens and drafts verified a round with every draft held, then at each cut."""
    names = ["every draft", *(f"{cut:.1f}" for cut in CUTS)]
    said = [
        f"{name} {sum(r.tokens for r in taken) / len(taken):.3f}"
        f" {sum(r.offered for r in taken) / len(taken):.2f}"
        if taken
        else f"{name} 0 rounds"
        for name, taken in zip(names, swept, strict=True)
    ]
    way = (
        "by the cut and serving's price of a row"
        if priced
        else "by the cut alone, not serving's price of a row"
    )
    return (
        f"{way}, as serving takes the rounds: tokens and drafts verified a round, "
        + ", ".join(said)
    )


def _timed(at_cuts: list[list[rounds.Round]], replies: int) -> str:
    """Seconds of rounds at each of `CUTS` at G2's time of a round, and the least.

    Serving holds one cut a launch, so the rounds of every class are summed. A tie
    takes the lower cut.
    """
    seconds = [rounds.timed(taken) / 1000 for taken in at_cuts]
    least = seconds.index(min(seconds))
    tokens = sum(r.tokens for r in at_cuts[least])
    return (
        f"at G2's time of a round ({rounds.ROUND_MS} ms and {rounds.DRAFT_MS} ms a draft"
        " verified), by the cut and serving's price of a row: seconds of rounds over"
        f" {replies} replies, "
        + ", ".join(f"{cut:.1f} {s:.1f}" for cut, s in zip(CUTS, seconds, strict=True))
        + f"; the least at {CUTS[least]:.1f}, {tokens / seconds[least]:.1f} tokens a"
        " second"
    )


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
