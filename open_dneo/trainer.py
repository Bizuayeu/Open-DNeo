"""A training run: steps over the recorded streams, and its measure on held ones.

Each step takes a stream of every part (`feed.Sampler`), a window of anchors in
each (`feed.window`), and the parts' losses in their shares; AdamW then steps at
the learning rate of `train.rate`, times a factor for each part of the drafter
(`groups`), its gradient clipped. Every so often the run
measures the drafter on windows of the held-out streams drawn once at its start
(`train.measured`), over every part and part by part, and keeps it as a folder
`score` reads (`train.save`). Before its first step it says how the loss's weight
falls by position in each part's held windows (`profile`), which the decay, the
path and the parts' text set together.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import drafter, feed, score, train


@dataclass(frozen=True)
class Settings:
    """A run's numbers. The defaults are placeholders, each from a source."""

    steps: int
    peak: float = 2e-4  # Kolibri's (TensorFold's drafter recipe)
    end: float = 2e-5  # Kolibri's
    warmup: int = 100  # Kolibri's
    decay: float = 4.0  # DFlash's at a block of 8
    off_path: float = 0.0  # the first run's (11-20)
    anchors: int = 1024  # Kolibri's, a window
    span: int = 2048  # serving's ring, and Kolibri's window
    clip: float = 1.0  # DFlash's and DSpark's
    held_anchors: int = 256  # a held-out stream's, for the measure
    held_streams: int | None = None  # the most held-out streams a part measures
    # cc-defer: every held stream by default; cap it once a measure is timed at scale
    log_every: int = 10
    measure_every: int = 500
    keep_every: int = 2500  # Kolibri's checkpoints
    seed: int = 0
    shares: Mapping[str, float] | None = field(default=None)
    rates: Mapping[str, float] | None = field(default=None)  # a part's factor
    # cc-defer: 1 for every part; lower the pretrained parts' once a test or the user
    # sets it (Kolibri warm-starts at a tenth of its peak, TensorFold #2667 at 3e-5)


def groups(model: drafter.Drafter, rates: Mapping[str, float] | None) -> list[dict]:
    """The optimiser's parameter groups: one a part of the drafter (its entrance,
    layers, norm and head), each with its `factor` of the rate, 1 where `rates` do
    not name it; one group for all where there are no rates. ValueError if a name
    is not a part's or a factor is below 0."""
    if rates is None:
        return [{"params": list(model.parameters()), "factor": 1.0, "part": "all"}]
    parts = dict(model.named_children())
    unknown = sorted(set(rates) - set(parts))
    if unknown:
        raise ValueError(f"the drafter's parts are {sorted(parts)}, not {unknown}")
    for part, factor in rates.items():
        if not factor >= 0:
            raise ValueError(f"the factor of {part} is {factor}: 0 or more")
    return [
        {
            "params": list(module.parameters()),
            "factor": float(rates.get(part, 1.0)),
            "part": part,
        }
        for part, module in sorted(parts.items())
    ]


def held_windows(
    held: Sequence[feed.Fed], settings: Settings
) -> list[tuple[feed.Fed, np.ndarray]]:
    """A window of each held-out stream, drawn once with the run's seed: of every
    one, or of `held_streams` a part drawn with the same seed, in the streams' order.
    ValueError if `held_streams` is below 1."""
    rng = np.random.default_rng(settings.seed + 1)
    chosen = [fed for fed in held if fed.anchorable]
    if settings.held_streams is not None:
        if settings.held_streams < 1:
            raise ValueError(
                f"held_streams is {settings.held_streams}: a part measures 1 or more"
            )
        parts: dict[str, list[int]] = {}
        for i, fed in enumerate(chosen):
            parts.setdefault(fed.part, []).append(i)
        kept = set()
        for indices in parts.values():
            order = rng.permutation(len(indices))[: settings.held_streams]
            kept.update(indices[j] for j in order)
        chosen = [fed for i, fed in enumerate(chosen) if i in kept]
    windows = []
    for fed in chosen:
        opened = fed.open()
        rows = feed.window(opened.eligible, settings.held_anchors, settings.span, rng)
        windows.append((fed, rows))
    return windows


def profile(
    windows: Sequence[tuple[feed.Fed, np.ndarray]], settings: Settings, block: int
) -> list[str]:
    """How the loss's weight falls by position in each part's held windows, a line
    a part: its anchors, the drafts an anchor on the target's path, and each
    position's share of the part's weight (`train.weights`)."""
    parts: dict[str, list] = {}
    for fed, rows in windows:
        opened = fed.open()
        taught = train.blocks(
            opened.stream.tok, opened.written, rows, block - 1, opened.room[rows]
        )
        summed = train.weights(taught, settings.decay, settings.off_path).sum(0)
        got = parts.setdefault(fed.part, [np.zeros(block - 1), 0, 0])
        got[0] += summed
        got[1] += len(rows)
        got[2] += int(taught.on_path.sum())
    lines = []
    for part, (summed, anchors, on_path) in parts.items():
        total = summed.sum()
        shares = ", ".join(
            f"{100 * w / total:.1f}%" if total else "0.0%" for w in summed
        )
        lines.append(
            f"weights, held part {part!r}: {anchors} anchors,"
            f" {on_path / anchors if anchors else 0.0:.2f} drafts an anchor on the"
            f" path; by position {shares}"
        )
    return lines


class _Measured:
    """What `measure` sums over some held windows."""

    def __init__(self, drafts: int) -> None:
        self.streams, self.weighed, self.summed = 0, 0.0, 0.0
        self.hits = torch.zeros(drafts, dtype=torch.int64)
        self.counted = torch.zeros(drafts, dtype=torch.int64)
        self.landed = self.anchors = 0

    def said(self, name: str) -> str:
        by_position = ", ".join(
            f"{int(h)}/{int(c)}"
            for h, c in zip(self.hits.tolist(), self.counted.tolist(), strict=True)
        )
        loss = self.summed / self.weighed if self.weighed else 0.0
        mean = self.landed / self.anchors if self.anchors else 0.0
        return (
            f"{name}: {self.streams} streams, {self.anchors} anchors; loss {loss:.4f};"
            f" best token on the path by position {by_position};"
            f" drafts landed on the path {mean:.3f} an anchor"
        )


@torch.no_grad()
def measure(
    model: drafter.Drafter,
    target: score.Target,
    reading: score.Reading,
    windows: Sequence[tuple[feed.Fed, np.ndarray]],
    settings: Settings,
) -> list[str]:
    """The loss and `train.measured` over the held-out windows, a line, and where
    they hold more than one part a line a part after it."""
    block = model.shape.block
    every, parts = _Measured(block - 1), {}
    for fed, rows in windows:
        opened = fed.open()
        got = train.batch(
            opened.stream, opened.written, rows, reading, block, settings.decay,
            settings.off_path, cap=opened.room[rows],
        )  # fmt: skip
        only = train.counted(got)
        taught = train.taught(model, target, got, reading, only=only)
        weight = float(got.weights[(got.labels >= 0)].sum())
        summed = float(_loss(taught, got, only)) * weight
        if taught.candidates is not None:  # hits only among what it would serve
            taught = train.taught(model, target, got, reading, carry=False, only=only)
        h, c, a = train.measured(
            taught,
            got.labels.to(taught.values.device),
            got.on_path.to(taught.values.device),
            only,
        )
        for sums in (every, parts.setdefault(fed.part, _Measured(block - 1))):
            sums.streams += 1
            sums.summed, sums.weighed = sums.summed + summed, sums.weighed + weight
            sums.hits, sums.counted = sums.hits + h.cpu(), sums.counted + c.cpu()
            sums.landed, sums.anchors = (
                sums.landed + int(a.sum()),
                sums.anchors + len(a),
            )
    lines = [every.said("held")]
    if len(parts) > 1:
        lines += [sums.said(f"held part {part!r}") for part, sums in parts.items()]
    return lines


def run(
    model: drafter.Drafter,
    target: score.Target,
    reading: score.Reading,
    config: Path,
    trained: Sequence[feed.Fed],
    held: Sequence[feed.Fed],
    out: Path,
    settings: Settings,
    log: Callable[[str], None] = print,
    resume: Path | None = None,
) -> None:
    """Trains `model` on `trained` as `settings` say, measuring it on `held` and
    keeping it under `out` as step-N folders (`train.save`).

    `resume` is a step-N folder of a run with the same settings: `model` is the
    drafter kept there, and the run goes on from step N + 1 with the optimiser's
    state and the draws kept beside it, to end where the whole run would have.
    ValueError if the settings differ or the folder holds no state to resume.
    """
    rng = np.random.default_rng(settings.seed)
    sampler = feed.Sampler(trained, settings.shares, rng)
    windows = held_windows(held, settings)
    optimiser = torch.optim.AdamW(
        groups(model, settings.rates), lr=settings.peak, weight_decay=0.0
    )
    block, kept = model.shape.block, asdict(settings)
    first = 1
    log(f"run: {len(trained)} streams, {len(held)} held, parts {sampler.shares}")
    for line in profile(windows, settings, block):
        log(line)
    if resume is not None:
        first = _resumed(Path(resume), kept, optimiser, rng) + 1
        log(f"resumed from {Path(resume).name}: step {first} on")
    else:
        for line in measure(model, target, reading, windows, settings):
            log(f"step 0, {line}")
    for step in range(first, settings.steps + 1):
        rate = train.rate(
            step - 1, settings.steps, settings.peak, settings.end, settings.warmup
        )
        for group in optimiser.param_groups:
            group["lr"] = rate * group["factor"]
        optimiser.zero_grad(set_to_none=True)
        total = 0.0
        for fed, share in sampler.step():
            opened = fed.open()
            rows = feed.window(opened.eligible, settings.anchors, settings.span, rng)
            got = train.batch(
            opened.stream, opened.written, rows, reading, block, settings.decay,
            settings.off_path, cap=opened.room[rows],
        )  # fmt: skip
            only = train.counted(got)
            value = share * _loss(
                train.taught(model, target, got, reading, only=only), got, only
            )
            value.backward()
            total += float(value.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.clip)
        optimiser.step()
        if step % settings.log_every == 0:
            log(f"step {step}: loss {total:.4f}, rate {rate:.3g}")
        last = step == settings.steps
        if step % settings.measure_every == 0 or last:
            for line in measure(model, target, reading, windows, settings):
                log(f"step {step}, {line}")
        if step % settings.keep_every == 0 or last:
            train.save(
                model, Path(out) / f"step-{step}", config, reading, step=step,
                optimiser=optimiser.state_dict(), draws=rng.bit_generator.state, **kept,
            )  # fmt: skip


def _loss(taught: drafter.Taught, got: train.Batch, only: torch.Tensor) -> torch.Tensor:
    """`train.loss` over the drafts a head was taught `only` at (`train.taught`)."""
    device = taught.values.device
    return train.loss(
        taught,
        train.chosen(got.labels.to(device), only),
        train.chosen(got.weights.to(device), only),
    )


def _resumed(
    folder: Path, kept: dict, optimiser: torch.optim.Optimizer, rng: np.random.Generator
) -> int:
    """The step a kept folder ends, its optimiser's state and draws put back."""
    form = json.loads((folder / score.TRAINED).read_text(encoding="utf-8"))
    if form["arguments"] != json.loads(json.dumps(kept)):
        raise ValueError(f"{folder.name} was kept by a run of other settings")
    if not (folder / train.RESUME).exists() or not (folder / train.DRAWS).exists():
        raise ValueError(f"{folder.name} holds no optimiser state or draws to resume")
    optimiser.load_state_dict(torch.load(folder / train.RESUME, weights_only=True))
    rng.bit_generator.state = json.loads(
        (folder / train.DRAWS).read_text(encoding="utf-8")
    )
    return int(form["step"])


def shares(text: str | None) -> dict[str, float] | None:
    """`part=share,part=share` as a mapping; None for none. ValueError otherwise."""
    return _pairs(text, "a share")


def rates(text: str | None) -> dict[str, float] | None:
    """`part=factor,part=factor` as a mapping (`groups`); None for none. ValueError
    otherwise."""
    return _pairs(text, "a rate's factor")


def _pairs(text: str | None, noun: str) -> dict[str, float] | None:
    """`name=number,name=number` as a mapping; None for none."""
    if not text:
        return None
    parsed = {}
    for item in text.split(","):
        part, _, value = item.partition("=")
        try:
            parsed[part.strip()] = float(value)
        except ValueError:
            raise ValueError(f"{noun} is part=number, not {item!r}") from None
    return parsed


def command(
    folders: Sequence[Path],
    nexts: Path,
    drafter_folder: Path,
    checkpoint: Path,
    out: Path,
    settings: Settings,
    *,
    parts: Sequence[Path] = (),
    bridge: Path | None = None,
    dneo_mask: bool = False,
    dneo_head: int | None = None,
    held_every: int = 10,
    device: str = "cpu",
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[str], bool]:
    """A run from the command line: the drafter as `score` builds it (a bridge and
    DNeo's mask and head, or a trained drafter's own form), the streams under
    `folders` with their choices under `nexts`, every `held_every`-th held out.
    With `resume` the drafter's folder is a kept step to go on from (`run`). Not sound
    where the drafter's file holds tensors that are not read.
    """
    model, target, reading, lines, unread = score.loaded(
        Path(drafter_folder), Path(checkpoint), device, False, bridge,
        dneo_mask=dneo_mask, dneo_head=dneo_head,
    )  # fmt: skip
    if unread:
        return lines + [
            "a drafter whose file holds tensors not read is no start"
        ], False
    for line in lines:
        log(line)
    fed = feed.streams(
        [Path(f) for f in folders], Path(nexts), [Path(p) for p in parts]
    )
    trained, held = feed.held(fed, held_every)
    run(
        model,
        target,
        reading,
        Path(drafter_folder) / "config.json",
        trained,
        held,
        Path(out),
        settings,
        log,
        resume=Path(drafter_folder) if resume else None,
    )
    return [f"== train finished: {settings.steps} steps, kept under {out}"], True
