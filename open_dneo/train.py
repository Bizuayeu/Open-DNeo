"""Training a drafter on recorded rows: what each block is taught, and the loss.

Serving verifies drafts greedily: a draft lands where it is the target's own
choice and every draft before it landed, and a draft's chance is read as the
chance that it lands once those before it have. A block is therefore taught the
target's choices after its anchor (`blocks`), as a cross-entropy on them
(`loss`), which is meant to make the chances that (a reading until measured:
`score.calibration` measures it). A recording of borrowed text holds the
target's choice after every row, but only while the text keeps choosing what the
target would have do those choices make the target's own continuation: past the
first token it did not choose a label follows text it would not have written
(`greedy.own_path`), and `weights` gives those labels a weight of their own, none
by default. A draft's weight falls with its position, as the chance that a round
reaches it does.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from . import drafter, greedy, recording, score
from .weights import WHOLE as WEIGHTS_FILE
from .weights import write as write_weights

# What a kept step holds beyond the drafter, for a run to resume from it (`save`).
RESUME, DRAWS = "optimiser.pt", "draws.json"


class Blocks(NamedTuple):
    """What the blocks at some anchors of a stream are taught."""

    labels: np.ndarray  # [m, n] i64: the target's choice after the row before each
    # draft, -1 past the last row
    previous: np.ndarray  # [m, n] i64: the token the head is told comes before each
    # draft: the text's own, the anchor's first; 0 past the last row
    on_path: np.ndarray  # [m]: how many drafts have labels on the target's own path
    available: np.ndarray  # [m]: how many drafts have labels at all


def blocks(
    tokens: np.ndarray,
    written: np.ndarray,
    anchors: np.ndarray,
    drafts: int,
    cap: np.ndarray | None = None,
) -> Blocks:
    """The labels of each anchor's first `drafts` drafts, and what comes before them.

    `tokens` [rows] are a stream's tokens at consecutive positions, `written` [rows]
    the target's choice after each row (`greedy.nexts`), `anchors` rows. Draft j of
    an anchor at row t (j from 1) is labelled with the choice after row t + j - 1,
    and its head is told row t + j - 1's token: on the target's own path that is the
    choice before, and past it the token the label's row followed. `cap` [m] is
    the most drafts each anchor counts, as where the target's turn ends
    (`feed.blocks`). ValueError if the rows do not line up (`greedy.own_path`).
    """
    on_path = greedy.own_path(tokens, written, anchors, drafts)
    tokens, written = np.asarray(tokens, np.int64), np.asarray(written, np.int64)
    rows = len(tokens)
    before = np.asarray(anchors, np.int64)[:, None] + np.arange(drafts)
    inside = before < rows
    held = np.minimum(before, rows - 1)
    available = inside.sum(1)
    if cap is not None:
        on_path, available = np.minimum(on_path, cap), np.minimum(available, cap)
    return Blocks(
        labels=np.where(inside, written[held], -1),
        previous=np.where(inside, tokens[held], 0),
        on_path=on_path,
        available=available,
    )


def weights(taught: Blocks, decay: float, off_path: float = 0.0) -> np.ndarray:
    """Each draft's weight [m, n]: exp(-(j - 1) / decay) for draft j on the target's
    own path, `off_path` times that past it, and 0 where there is no label.

    The decay is DFlash's form for its loss (its paper sets 4 at a block of 8).
    ValueError if `decay` is not above 0.
    """
    if not decay > 0:
        raise ValueError(f"decay is {decay}: a draft's weight falls as exp(-j / decay)")
    position = np.arange(taught.labels.shape[1])
    on = position < taught.on_path[:, None]
    there = position < taught.available[:, None]
    share = np.where(on, 1.0, np.where(there, off_path, 0.0))
    return np.exp(-position / decay)[None, :] * share


def loss(
    taught: drafter.Taught, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """The labels' cross-entropy under the head's values, weighted, over the weights.

    `taught` is a head's `teach` [m, n, support], `labels` [m, n] (-1 for none) and
    `weights` [m, n]. A head that scores candidates finds a label by its slot among
    them. A draft without a label or with a weight of 0 counts for nothing, and
    nothing counted is a loss of 0. ValueError if a label with a weight is not
    among its candidates.
    """
    values = taught.values.float()
    counted = (weights > 0) & (labels >= 0)
    if taught.candidates is None:
        slots = labels.clamp_min(0)
    else:
        hits = taught.candidates == labels[..., None]
        missing = counted & ~hits.any(-1)
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} labels with a weight are not among the"
                " candidates: teach the head with them"
            )
        slots = hits.float().argmax(-1)
    each = torch.logsumexp(values, -1) - values.gather(-1, slots[..., None])[..., 0]
    weight = torch.where(counted, weights.to(values), torch.zeros_like(values[..., 0]))
    return (each * weight).sum() / weight.sum().clamp_min(1e-12)


@dataclass(frozen=True, eq=False)
class Batch:
    """The blocks at some anchors of one stream, and the rows they see."""

    features: torch.Tensor  # [1, s, taps, hidden] f32: the context rows' taps
    context_positions: torch.Tensor  # [1, s]
    anchors: torch.Tensor  # [1, m] the anchors' positions
    ids: torch.Tensor  # [m, block] each anchor's token, then mask tokens
    previous: torch.Tensor  # [m, n] (`Blocks`)
    labels: torch.Tensor  # [m, n]
    weights: torch.Tensor  # [m, n] f64 (`weights`)
    on_path: torch.Tensor  # [m] drafts with labels on the target's path (`Blocks`)


def batch(
    stream: recording.Stream,
    written: np.ndarray,
    rows: np.ndarray,
    reading: score.Reading,
    block: int,
    decay: float,
    off_path: float = 0.0,
    cap: np.ndarray | None = None,
) -> Batch:
    """The blocks at `rows` of `stream` as training takes them, with the rows they see.

    The context is what serving's drafter holds at those anchors (`score.proposals`):
    the stream's rows before the last anchor, from the first anchor's window on,
    each anchor seeing its own (`drafter.visible`). `written` is the target's choice
    after each row and `cap` the most drafts an anchor counts (`blocks`).
    ValueError if the stream does not hold the reading's taps or its rows are not
    at consecutive positions.
    """
    summary = stream.summary
    missing = [tap for tap in reading.taps if tap not in summary.taps]
    if missing:
        raise ValueError(f"the stream holds taps {list(summary.taps)}, not {missing}")
    if not stream.in_order():
        raise ValueError("the stream's rows are not at consecutive positions")
    rows = np.asarray(rows, np.int64)
    taught = blocks(stream.tok, written, rows, block - 1, cap)
    first = summary.first_row
    at = first + rows
    low, high = first, int(at.max())
    if reading.window is not None:
        low = max(low, int(at.min()) - reading.window + 1)
    columns = [summary.taps.index(tap) for tap in reading.taps]
    ids = np.full((len(rows), block), reading.mask_token, np.int64)
    ids[:, 0] = np.asarray(stream.tok, np.int64)[rows]
    return Batch(
        features=torch.from_numpy(stream.tapped(low - first, high - first)[:, columns])[
            None
        ],
        context_positions=torch.arange(low, high)[None],
        anchors=torch.from_numpy(at)[None],
        ids=torch.from_numpy(ids),
        previous=torch.from_numpy(taught.previous),
        labels=torch.from_numpy(taught.labels),
        weights=torch.from_numpy(weights(taught, decay, off_path)),
        on_path=torch.from_numpy(np.asarray(taught.on_path, np.int64)),
    )


def counted(batch: Batch) -> torch.Tensor:
    """Where a batch's drafts count [m, n]: a label, and a weight above 0. The loss
    and `measured` read nothing else."""
    return (batch.weights > 0) & (batch.labels >= 0)


def chosen(values: torch.Tensor, only: torch.Tensor) -> torch.Tensor:
    """`values` [m, n, ...] where `only` [m, n] holds, position by position: [k, ...],
    in the order `taught` scores them."""
    return values.transpose(0, 1)[only.T.to(values.device)]


def taught(
    model: drafter.Drafter,
    target: score.Target,
    batch: Batch,
    reading: score.Reading,
    carry: bool = True,
    only: torch.Tensor | None = None,
) -> drafter.Taught:
    """The model's head on a batch's blocks, told the tokens before each draft and,
    with `carry`, given their labels, which a head of candidates takes among them
    (`drafter.carrying`): for training. Without it the head scores what it would
    serve, for measuring.

    With `only` [m, n] (`counted`) the target's head and the drafter's are run at
    those drafts alone, a row each in the order of `chosen`: the loss over them and
    `measured` are as over every draft, at a fraction of the vocabulary's products
    where the labels leave the path early."""
    like = model.norm.weight
    context_positions = batch.context_positions.to(like.device)
    anchors = batch.anchors.to(like.device)
    mask = drafter.visible(
        anchors,
        context_positions,
        model.shape.block,
        whole_block=reading.whole_block,
        window=reading.window,
    )
    ids = batch.ids.to(like.device)
    hidden = model(
        target.embed[ids][None],
        batch.features.to(like.device, like.dtype),
        anchors,
        context_positions,
        mask,
    )
    states = hidden[0, :, 1:].contiguous()
    if only is None:
        return model.head.teach(
            states,
            target.scores(states),
            batch.previous.to(like.device),
            batch.labels.to(like.device) if carry else None,
        )
    only = only.to(like.device)
    rows = chosen(states, only)
    position = torch.arange(states.shape[1], device=like.device)
    return model.head.teach_rows(
        rows,
        target.scores(rows),
        chosen(batch.previous.to(like.device), only),
        chosen(batch.labels.to(like.device), only) if carry else None,
        chosen(position.expand(states.shape[:2]), only),
    )


def rate(step: int, steps: int, peak: float, end: float, warmup: int) -> float:
    """The learning rate at `step` of `steps` (from 0): up in a line to `peak` over
    `warmup` steps, then down a cosine to `end` at the last step. Kolibri's drafter
    went from 2e-4 to 2e-5 after 100 warm-up steps (TensorFold's drafter recipe).
    ValueError if the warm-up is not shorter than the run.
    """
    if not 0 < warmup < steps:
        raise ValueError(f"warmup {warmup} is not below steps {steps}, or not above 0")
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup + 1) / (steps - warmup)
    return end + (peak - end) * (1 + math.cos(math.pi * progress)) / 2


def save(
    model: drafter.Drafter,
    folder: Path,
    config: Path,
    reading: score.Reading,
    *,
    step: int,
    optimiser: dict | None = None,
    draws: dict | None = None,
    **arguments: object,
) -> None:
    """A trained drafter as a folder `score` reads: the base drafter's config.json,
    model.safetensors with every parameter as float32, and score.TRAINED, which
    names its form (the taps of a bridged entrance, DNeo's mask, DNeo's head and
    its best k), the step and the run's `arguments`. With `optimiser` (its state)
    and `draws` (the run's random generator's state) a run can resume from it
    (RESUME, DRAWS). FileExistsError where the folder exists.
    """
    folder = Path(folder)
    folder.mkdir(parents=True)
    shutil.copyfile(config, folder / "config.json")
    tensors = {
        name: value.detach().float().cpu().numpy()
        for name, value in model.state_dict().items()
    }
    write_weights(folder / WEIGHTS_FILE, tensors)
    form = {
        "step": step,
        "taps": list(reading.taps)
        if isinstance(model.entrance, drafter.Bridged)
        else None,
        "dneo_mask": not reading.whole_block,
        "dneo_head": model.head.best
        if isinstance(model.head, drafter.Folded)
        else None,
        "arguments": arguments,
    }
    (folder / score.TRAINED).write_bytes((json.dumps(form, indent=1) + "\n").encode())
    if optimiser is not None:
        torch.save(optimiser, folder / RESUME)
    if draws is not None:
        (folder / DRAWS).write_bytes((json.dumps(draws) + "\n").encode())


def measured(
    taught: drafter.Taught,
    labels: torch.Tensor,
    on_path: torch.Tensor,
    only: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """How the head's best tokens meet the labels on the target's own path.

    `on_path` [m] is how many of each anchor's drafts have labels on it (with a cap
    where the target's turn ends, `blocks`). Gives, by position [n], the labels on
    the path the best token hits and how many there are, and for each anchor [m]
    how many drafts in a row from the first hit: the drafts that would land in a
    round, but that none is counted past the path, where the target's own labels
    are not known. A head taught `only` where it holds (`taught`) has its best
    tokens put back in their places, and none elsewhere.
    """
    best = taught.values.argmax(-1)
    if taught.candidates is not None:
        best = taught.candidates.gather(-1, best[..., None])[..., 0]
    if only is not None:
        placed = torch.full(labels.shape, -2, dtype=best.dtype, device=best.device)
        placed.transpose(0, 1)[only.T.to(best.device)] = best
        best = placed
    position = torch.arange(labels.shape[1], device=labels.device)
    counted = (position < on_path[:, None]) & (labels >= 0)
    hit = (best == labels) & counted
    landed = hit.int().cumprod(-1).sum(-1)
    return hit.sum(0), counted.sum(0), landed
