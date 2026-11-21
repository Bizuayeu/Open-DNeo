"""The bridge: DNeo's entrance in front of DFlash2's layers, fitted by least squares.

DFlash2's entrance joins a context row's tapped states and shrinks them with `fc`
[hidden, taps * hidden] before its norm (`drafter.Fused`). DNeo's takes a weighted
sum with a scalar a tap (`drafter.Router`). A bridge B [hidden, hidden] maps the
sum to what `fc` gave. Fitted on recorded rows by least squares, it lets DFlash2's
layers read DNeo's entrance at step 0 (`drafter.Bridged`): the plan's move 2,
judged at G1. The sum may read other taps than `fc` does, as many.

The norm after the entrance keeps a row's direction only. A row's states and its
target are both divided by the target's size (its root mean square), so that every
row weighs alike and the fit stays linear in the states. The sums keep each pair of
taps apart, so the sum's weights are chosen after the pass: uniform, even (each
tap's states brought to the same size) or given. Products are float32 and sums
float64, as the probe's; the ridge is a share of the mean of the Gram's diagonal.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from . import drafter, recording, score, weights

# Rows gathered before they are added to the sums: a product of a few rows costs a
# pass over the Gram all the same.
CHUNK = 8192
# A bridge file's tensors: B, which gives fc's output from the sum, the weights, and the
# taps the sum reads.
MATRIX, TAP_WEIGHTS, TAPS = "bridge.weight", "bridge.tap_weights", "bridge.taps"
# The weights made here, which given weights cannot be named.
MADE = ("uniform", "even")


def scaled(
    rows: np.ndarray, fc: np.ndarray, target_rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rows [n, taps, hidden] as states [m, taps * hidden] and targets [m, hidden].

    A row's target is `fc` [hidden, taps * hidden] on it, or on the same rows'
    `target_rows` where the sum reads other taps than `fc`. Both are divided by
    the target's root mean square. Rows whose target is zero have no direction to
    fit and are left out.
    """
    states = rows.reshape(len(rows), -1).astype(np.float32, copy=False)
    read = states if target_rows is None else target_rows.reshape(len(rows), -1)
    targets = read.astype(np.float32, copy=False) @ fc.T
    size = np.sqrt((targets.astype(np.float64) ** 2).mean(-1))
    kept = size > 0
    scale = (1.0 / size[kept]).astype(np.float32)[:, None]
    return states[kept] * scale, targets[kept] * scale


class Sums:
    """What a bridge is fitted from, each pair of taps' products kept apart."""

    def __init__(self, taps: int, width: int) -> None:
        self.taps, self.width = taps, width
        self.gram = np.zeros((taps * width, taps * width))
        self.cross = np.zeros((taps * width, width))
        self.rows, self.energy = 0, 0.0

    def add(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Rows as `scaled` gives them."""
        self.gram += states.T @ states
        self.cross += states.T @ targets
        self.rows += len(states)
        self.energy += float((targets.astype(np.float64) ** 2).sum())

    def _block(self, t: int, u: int) -> np.ndarray:
        h = self.width
        return self.gram[t * h : (t + 1) * h, u * h : (u + 1) * h]

    def matrices(self, tap_weights: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """The Gram and the cross products [hidden, hidden] of the weighted sum."""
        w, h = np.asarray(tap_weights, np.float64), self.width
        gram, cross = np.zeros((h, h)), np.zeros((h, h))
        for t in range(self.taps):
            cross += w[t] * self.cross[t * h : (t + 1) * h]
            for u in range(self.taps):
                gram += w[t] * w[u] * self._block(t, u)
        return gram, cross

    def tap_sizes(self) -> np.ndarray:
        """Each tap's root mean square over the rows added, as they were divided."""
        cells = max(self.rows * self.width, 1)
        return np.array(
            [np.sqrt(np.trace(self._block(t, t)) / cells) for t in range(self.taps)]
        )


def even(sizes: Sequence[float]) -> list[float]:
    """Weights that bring each tap to the same size: inversely as its size."""
    sizes = np.asarray(sizes, np.float64)
    if (sizes <= 0).any():
        raise ValueError(f"tap sizes {sizes.tolist()}: a tap holds only zeros")
    inverse = 1.0 / sizes
    return (inverse / inverse.sum()).tolist()


def parse(text: str) -> tuple[str | None, list[float]]:
    """Weights given as `1,2,1` or `NAME=1,2,1`, brought to a sum of 1."""
    name, _, listed = text.rpartition("=")
    if name in MADE:
        raise ValueError(f"{name} names weights made here")
    try:
        values = [float(each) for each in listed.split(",")]
    except ValueError:
        raise ValueError(f"weights {text!r} are not numbers") from None
    if not np.isfinite(values).all() or min(values) < 0 or sum(values) <= 0:
        raise ValueError(f"weights {text!r}: none below 0, and some above it")
    total = sum(values)
    return name or None, [value / total for value in values]


def solve(gram: np.ndarray, cross: np.ndarray, ridge: float) -> np.ndarray:
    """M [hidden, hidden] with sum @ M nearest the targets, damped by the ridge."""
    scale = float(np.mean(np.diag(gram))) or 1.0
    return np.linalg.solve(gram + ridge * scale * np.eye(len(gram)), cross)


def write(
    path: Path, matrix: np.ndarray, tap_weights: Sequence[float], taps: Sequence[int]
) -> None:
    """A bridge file (safetensors): B [hidden, hidden], the weights, and the taps.

    FileExistsError where the file exists.
    """
    tensors = {
        MATRIX: np.ascontiguousarray(matrix, "<f4"),
        TAP_WEIGHTS: np.asarray(tap_weights, "<f4"),
        TAPS: np.asarray(taps, "<i8"),
    }
    header: dict = {"__metadata__": {"taps": ",".join(map(str, taps))}}
    data = []
    start = 0
    for name, array in tensors.items():
        raw = array.tobytes()
        header[name] = {
            "dtype": {"f": "F32", "i": "I64"}[array.dtype.kind],
            "shape": list(array.shape),
            "data_offsets": [start, start + len(raw)],
        }
        data.append(raw)
        start += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(data))


def read(path: Path) -> tuple[torch.Tensor, tuple[float, ...], tuple[int, ...]]:
    """A bridge file's B as float32, its weights, and the taps its sum reads."""
    held = weights.read(path, [MATRIX, TAP_WEIGHTS, TAPS])
    return (
        held[MATRIX].float(),
        tuple(held[TAP_WEIGHTS].double().tolist()),
        tuple(int(tap) for tap in held[TAPS].tolist()),
    )


def entrance(
    model: drafter.Drafter, matrix: torch.Tensor, tap_weights: Sequence[float]
) -> drafter.Bridged:
    """`model`'s DFlash2 entrance bridged: DNeo's sum, `matrix`, and DFlash2's norm."""
    if not isinstance(model.entrance, drafter.Fused):
        raise ValueError("only DFlash2's entrance is bridged")
    if len(tap_weights) != model.shape.taps:
        raise ValueError(
            f"{len(tap_weights)} weights for an entrance of {model.shape.taps} taps"
        )
    like = model.entrance.norm.weight
    new = drafter.Bridged(model.shape)
    with torch.no_grad():
        new.router.logits.copy_(torch.tensor(tap_weights, dtype=torch.float64).log())
        new.fc.weight.copy_(matrix)
        new.norm.weight.copy_(like)
    new.norm.eps = model.entrance.norm.eps
    return new.to(like.device, like.dtype)


def _said(values: Sequence[float]) -> str:
    return ",".join(f"{value:.3f}" for value in values)


def report(
    roots: Sequence[Path],
    drafter_folder: Path,
    out: Path,
    given: Sequence[str] = (),
    parts: Sequence[Path] = (),
    ridge: float = 1e-3,
    held_every: int = 10,
    train_every: int = 4,
    inputs: Sequence[int] = (),
) -> tuple[list[str], bool]:
    """Bridges fitted on the streams under `roots`, written as out/bridge-LABEL.safetensors.

    The target is `fc` of the DFlash2 drafter in `drafter_folder` on the taps it
    reads; the sum reads `inputs`, as many taps, or the same ones. Streams are taken in name order across the folders: every
    `held_every`-th is held out, and every `train_every`-th row of a stream is
    taken. A line a set of weights (uniform, even, and each of `given`) tells the
    share of the targets' energy the bridge explains on the fitted rows and on
    the held ones, and the mean cosine between the bridged sum and the target on
    the held rows, by part where `parts` name them. ValueError for weights of
    another number of taps, a stream without the taps, or two streams of one
    name; FileExistsError where a bridge would be written over.
    """
    config = json.loads((drafter_folder / "config.json").read_text(encoding="utf-8"))
    reads = score.Reading.dflash2(config).taps
    taps = tuple(inputs) or reads
    if len(taps) != len(reads):
        raise ValueError(
            f"a sum of {len(taps)} taps, and DFlash2 reads {len(reads)} taps"
        )
    chosen: dict[str, list[float] | None] = {
        "uniform": [1 / len(taps)] * len(taps),
        "even": None,  # from the sums
    }
    for n, text in enumerate(given, 1):
        name, values = parse(text)
        if len(values) != len(taps):
            raise ValueError(
                f"weights {text!r} are {len(values)}, and DFlash2 reads {len(taps)} taps"
            )
        label = name or f"given-{n}"
        if label in chosen:
            raise ValueError(f"two sets of weights are named {label}")
        chosen[label] = values
    paths = {label: out / f"bridge-{label}.safetensors" for label in chosen}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"{path} exists")

    streams: list[tuple[str, recording.Stream]] = []
    for root in roots:
        for base in recording.find(root):
            if recording.beside(base, ".json").exists():
                streams.append(
                    (base.relative_to(root).as_posix(), recording.open_stream(base))
                )
    names = [name for name, _ in streams]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"two folders hold a stream {twice[0]}")
    streams.sort(key=lambda named: named[0])
    fc = weights.read(drafter_folder / weights.WHOLE, ["fc.weight"])["fc.weight"]
    fc = fc.float().numpy()
    width = fc.shape[0]
    held = {
        name for i, (name, _) in enumerate(streams) if i % held_every == held_every - 1
    }
    part_of = {}
    for listing in parts:
        for line in listing.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                name, part = line.split("\t")[:2]
                part_of[name] = part

    def rows_of(name: str, stream: recording.Stream) -> tuple[np.ndarray, np.ndarray]:
        """The rows of the taps the sum reads, and of those fc reads."""
        summary = stream.summary
        missing = [
            tap for tap in dict.fromkeys(taps + reads) if tap not in summary.taps
        ]
        if missing:
            raise ValueError(f"{name} holds taps {list(summary.taps)}, not {missing}")
        if summary.rows and summary.width() != width:
            raise ValueError(f"{name}'s rows are {summary.width()} wide, fc's {width}")
        cells = stream.tapped(0, summary.rows)[::train_every]
        return (
            cells[:, [summary.taps.index(tap) for tap in taps]],
            cells[:, [summary.taps.index(tap) for tap in reads]],
        )

    def added(into: Sums, waiting: list[tuple[np.ndarray, np.ndarray]]) -> None:
        into.add(
            *scaled(
                np.concatenate([w[0] for w in waiting]),
                fc,
                np.concatenate([w[1] for w in waiting]),
            )
        )

    sums = Sums(len(taps), width)
    waiting, count = [], 0
    for name, stream in streams:
        if name in held:
            continue
        waiting.append(rows_of(name, stream))
        count += len(waiting[-1][0])
        if count >= CHUNK:
            added(sums, waiting)
            waiting, count = [], 0
    if waiting:
        added(sums, waiting)
    if not sums.rows:
        return ["no row to fit"], False
    sizes = sums.tap_sizes()
    chosen["even"] = even(sizes)
    fitted = {}
    for label, tap_weights in chosen.items():
        gram, cross = sums.matrices(tap_weights)
        mapped = solve(gram, cross, ridge)
        residual = (
            sums.energy
            - 2 * float((mapped * cross).sum())
            + float((mapped * (gram @ mapped)).sum())
        )
        fitted[label] = (mapped.astype(np.float32), 1 - residual / sums.energy)
    fitted_rows = sums.rows
    del sums

    tallies = {label: [0.0, 0.0, defaultdict(lambda: [0.0, 0])] for label in chosen}
    held_rows = 0
    for name, stream in streams:
        if name not in held:
            continue
        summed_rows, target_rows = rows_of(name, stream)
        states, targets = scaled(summed_rows, fc, target_rows)
        split = states.reshape(len(states), len(taps), width)
        held_rows += len(states)
        for label, tap_weights in chosen.items():
            summed = np.einsum("t,ntd->nd", np.asarray(tap_weights, np.float32), split)
            got = summed @ fitted[label][0]
            tally = tallies[label]
            tally[0] += float(((got - targets).astype(np.float64) ** 2).sum())
            tally[1] += float((targets.astype(np.float64) ** 2).sum())
            lengths = np.linalg.norm(got, axis=-1) * np.linalg.norm(targets, axis=-1)
            cosines = (got * targets).sum(-1) / np.where(lengths > 0, lengths, 1.0)
            part = tally[2][part_of.get(name, "?")]
            part[0], part[1] = part[0] + float(cosines.sum()), part[1] + len(cosines)

    out.mkdir(parents=True, exist_ok=True)
    for label, path in paths.items():
        write(path, fitted[label][0].T, chosen[label], taps)
    lines = [
        f"{len(streams)} streams in {len(roots)} folders, {len(held)} held out (one in"
        f" every {held_every}, by name); the sum of taps {taps}, the target fc on taps"
        f" {reads} as DFlash2 reads them; one row in"
        f" every {train_every} of a stream: {fitted_rows} fitted, {held_rows} held;"
        f" ridge {ridge} (times the mean of the Gram's diagonal); each row and its"
        " target divided by the target's size",
        "the taps' sizes so divided: "
        + ", ".join(f"{tap} {size:.4g}" for tap, size in zip(taps, sizes, strict=True)),
    ]
    for label, tap_weights in chosen.items():
        residual, energy, by_part = tallies[label]
        cosine = sum(c for c, _ in by_part.values()) / max(held_rows, 1)
        line = (
            f"{label} weights {_said(tap_weights)}: explained {fitted[label][1]:.4f}"
            f" fitted, {1 - residual / max(energy, 1e-300):.4f} held;"
            f" cos {cosine:.4f} held"
        )
        if part_of:
            line += "; by part: " + ", ".join(
                f"{p} {c / max(n, 1):.4f} ({n})"
                for p, (c, n) in sorted(by_part.items())
            )
        lines.append(line)
    lines.append("written: " + ", ".join(path.name for path in paths.values()))
    return lines, held_rows > 0
