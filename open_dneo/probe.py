"""Where a recording's layers hold the target's next tokens: the layer probe.

Stage 1 picks the layers a DNeo drafter reads from a pilot recording. For each
tap, and for sets of taps side by side, a linear map with an intercept is fitted
by ridge least squares from the rows at position p to the target's final row at
p + k, on some streams; on the others, the mapped row goes through the target's
head, and its best token is checked against the target's own choice at p + k
(greedy's saved choices, `greedy.report(save=...)`). The floor is how often the
commonest choice in the fitted streams is right. The final tap at k 0 maps a row
to itself: it scores 1 where the saved choices belong to the rows they sit beside.

Fitting reads every tapped row once a configuration: the single taps share a
pass, each set has its own. Products are float32 and sums float64.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from . import greedy, recording, weights


def _streams(root: Path) -> list[tuple[str, recording.Stream]]:
    named = []
    for base in recording.find(root):
        if recording.beside(base, ".json").exists():
            named.append(
                (base.relative_to(root).as_posix(), recording.open_stream(base))
            )
    return sorted(named)


def _choices(nexts: Path, name: str, rows: int) -> np.ndarray:
    path = nexts / f"{name}.next"
    if not path.exists():
        raise ValueError(f"no saved choices for {name} ({path})")
    got = np.fromfile(path, dtype="<i4")
    if len(got) != rows:
        raise ValueError(f"{name}: {len(got)} saved choices for {rows} rows")
    return got


class _Fit:
    """Sums for one configuration: features (some taps side by side) to final rows."""

    def __init__(self, columns: Sequence[int], width: int, ks: Sequence[int]):
        self.columns, self.ks = list(columns), list(ks)
        size = width * len(self.columns) + 1
        self.gram = np.zeros((size, size))
        self.cross = {k: np.zeros((size, width)) for k in self.ks}

    def features(self, cells: np.ndarray, at: np.ndarray) -> np.ndarray:
        picked = cells[at][:, self.columns].reshape(len(at), -1)
        return np.concatenate([picked, np.ones((len(at), 1), np.float32)], axis=1)

    def add(self, cells: np.ndarray, final: int, stride: int) -> None:
        for k in self.ks:
            at = np.arange(0, len(cells) - k, stride)
            if not len(at):
                continue
            x = self.features(cells, at)
            self.cross[k] += (x.T @ cells[at + k, final]).astype(np.float64)
        at = np.arange(0, len(cells) - min(self.ks), stride)
        if len(at):
            x = self.features(cells, at)
            self.gram += (x.T @ x).astype(np.float64)

    def solve(self, ridge: float) -> dict[int, np.ndarray]:
        size = len(self.gram)
        scale = float(np.mean(np.diag(self.gram)[: size - 1])) or 1.0
        damped = self.gram + ridge * scale * np.diag([1.0] * (size - 1) + [0.0])
        return {
            k: np.linalg.solve(damped, c).astype(np.float32)
            for k, c in self.cross.items()
        }


def report(
    root: Path,
    checkpoint: Path,
    nexts: Path,
    ks: Sequence[int] = (1, 2, 3),
    sets: Sequence[Sequence[int]] = (),
    parts: Path | None = None,
    ridge: float = 1e-3,
    held_every: int = 10,
    train_every: int = 4,
    eval_every: int = 8,
    device: str = "cpu",
) -> tuple[list[str], bool]:
    """The probe's scores, a line a tap (or set) and k; per part where `parts` names them.

    Streams are taken in name order; every `held_every`-th is held out. Rows are
    fitted every `train_every`-th and scored every `eval_every`-th. ValueError where
    a stream lacks its saved choices or the streams' taps differ.
    """
    streams = _streams(root)
    if not streams:
        return ["no finished stream"], False
    taps = list(streams[0][1].summary.taps)
    if any(list(s.summary.taps) != taps for _, s in streams):
        raise ValueError("the streams hold different taps")
    final_tap = greedy.final_tap(checkpoint)
    if final_tap not in taps:
        raise ValueError(f"the streams hold taps {taps}, not the final tap {final_tap}")
    final, width = taps.index(final_tap), streams[0][1].summary.width()
    choices = {name: _choices(nexts, name, s.summary.rows) for name, s in streams}
    part_of = {}
    if parts is not None:
        for line in parts.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                name, part = line.split("\t")[:2]
                part_of[name] = part
    held = {
        name for i, (name, _) in enumerate(streams) if i % held_every == held_every - 1
    }
    configs = {f"tap {t}": [taps.index(t)] for t in taps}
    for chosen in sets:
        missing = [t for t in chosen if t not in taps]
        if missing:
            raise ValueError(
                f"set {list(chosen)} names taps {missing} the streams do not hold"
            )
        configs["taps " + "+".join(map(str, chosen))] = [taps.index(t) for t in chosen]

    groups = [[label for label in configs if label.startswith("tap ")]]
    groups += [[label] for label in configs if label.startswith("taps ")]
    head = weights.head(checkpoint).to(device)
    seen: dict[int, Counter] = {k: Counter() for k in ks}
    scored: dict[str, dict[int, tuple[int, int, dict[str, list[int]]]]] = {}
    for g, group in enumerate(groups):
        # One configuration's sums at a time: a set's Gram alone can be gigabytes.
        fits = {label: _Fit(configs[label], width, ks) for label in group}
        for name, stream in streams:
            if name in held:
                continue
            cells = stream.tapped(0, stream.summary.rows)
            for fit in fits.values():
                fit.add(cells, final, train_every)
            if g == 0:
                for k in ks:
                    at = np.arange(0, len(cells) - k, train_every)
                    seen[k].update(choices[name][at + k].tolist())
        solved = {label: fit.solve(ridge) for label, fit in fits.items()}
        columns = {label: fit for label, fit in fits.items()}
        for fit in fits.values():
            fit.gram = fit.cross = None
        tallies = {
            (label, k): [0, 0, defaultdict(lambda: [0, 0])]
            for label in group
            for k in ks
        }
        for name, stream in streams:
            if name not in held:
                continue
            cells = stream.tapped(0, stream.summary.rows)
            for label in group:
                for k in ks:
                    at = np.arange(0, len(cells) - k, eval_every)
                    if not len(at):
                        continue
                    mapped = columns[label].features(cells, at) @ solved[label][k]
                    mapped = torch.from_numpy(mapped).to(head.device, head.dtype)
                    best = (mapped @ head.T).argmax(-1).cpu().numpy()
                    hit = int((best == choices[name][at + k]).sum())
                    tally = tallies[label, k]
                    tally[0], tally[1] = tally[0] + hit, tally[1] + len(at)
                    part = tally[2][part_of.get(name, "?")]
                    part[0], part[1] = part[0] + hit, part[1] + len(at)
        for (label, k), (right, total, by_part) in tallies.items():
            scored.setdefault(label, {})[k] = (right, total, dict(by_part))
        del fits, solved, columns

    lines = [
        f"{len(streams)} streams, {len(held)} held out (every {held_every}th by name),"
        f" taps {taps}, final tap {final_tap}; rows fitted every {train_every}th,"
        f" scored every {eval_every}th; ridge {ridge} (times the mean of the Gram's"
        " diagonal)"
    ]
    for k in ks:
        common = seen[k].most_common(1)[0][0] if seen[k] else None
        right = total = 0
        for name, stream in streams:
            if name in held:
                at = np.arange(0, stream.summary.rows - k, eval_every)
                right += int((choices[name][at + k] == common).sum())
                total += len(at)
        lines.append(
            f"the floor k {k}: {right / max(total, 1):.3f} ({total} positions, token {common})"
        )
        for label in configs:
            right, total, by_part = scored[label][k]
            line = f"{label} k {k}: {right / max(total, 1):.3f} ({total})"
            if part_of:
                line += ";" + ",".join(
                    f" {p} {r / max(n, 1):.3f} ({n})"
                    for p, (r, n) in sorted(by_part.items())
                )
            lines.append(line)
    return lines, True
