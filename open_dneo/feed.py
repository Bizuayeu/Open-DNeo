"""What a trainer is fed: the rows where the target writes, and anchors before them.

Serving's drafter drafts while the target writes: an assistant's turn under GLM's
chat template, whose markers name each turn, or a plain text no marker opened,
which stands in for one. In an assistant's turn every token but the role markers is
the target's, its thinking and tool-call tags too, and so is the marker or the end
of text that closes the turn: the target writes it to end its reply. A token the
template puts there is left to the labels' path (`greedy.own_path`): the target did
not choose it, so the labels after it are off the path. The other turns, and a
plain text's special tokens, are not the target's. An anchor drafts the rows after
it while they are the target's to write (`ahead`), and a block counts no draft past
them (`blocks`). A stream that resumed from kept state begins at a row of a turn it
does not name, so its rows before its first marker are not taken as the target's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import recording, train

MARKERS = (154826, 154827, 154828, 154829)  # system, user, assistant, observation
ASSISTANT = 154828
END = 154820  # <|endoftext|>, the first of GLM-5.3-Flash's special tokens
TEMPLATE = END  # tokens from here on are special or added: tags, markers, the end
_TEXT, _UNKNOWN = -1, -2  # the turn of rows before any marker: plain text, or not known


def writable(tokens: np.ndarray, resumed: bool = False) -> np.ndarray:
    """Whether the target writes each row's token [rows] (the module's rule).

    `resumed` says the stream began after its first row, mid-turn: its rows before
    its first marker are then in a turn not known, and none is the target's.
    """
    tokens = np.asarray(tokens, np.int64)
    rows = len(tokens)
    marker = np.isin(tokens, MARKERS)
    marked = np.where(marker, np.arange(rows), -1)
    last = np.maximum.accumulate(marked) if rows else marked
    head = _UNKNOWN if resumed else _TEXT
    turn = np.where(last >= 0, tokens[np.maximum(last, 0)], head)
    inside = ((turn == ASSISTANT) & ~marker) | ((turn == _TEXT) & (tokens < TEMPLATE))
    before = np.concatenate([[head], turn[:-1]]) if rows else turn
    closes = (marker | (tokens == END)) & (before == ASSISTANT)
    return inside | closes


def ahead(mask: np.ndarray) -> np.ndarray:
    """For each row, how many rows in a row after it are true in `mask` [rows]."""
    mask = np.asarray(mask, bool)
    stops = np.where(mask, len(mask), np.arange(len(mask)))
    first_false = np.minimum.accumulate(stops[::-1])[::-1] if len(mask) else stops
    return np.append((first_false - np.arange(len(mask)))[1:], 0)[: len(mask)]


def anchors(tokens: np.ndarray, resumed: bool = False) -> np.ndarray:
    """The rows whose next row the target writes: where a block can start."""
    return np.flatnonzero(ahead(writable(tokens, resumed)) > 0)


def blocks(
    tokens: np.ndarray,
    written: np.ndarray,
    rows: np.ndarray,
    drafts: int,
    resumed: bool = False,
) -> train.Blocks:
    """`train.blocks`, counting no draft past the rows the target writes."""
    rows = np.asarray(rows, np.int64)
    cap = ahead(writable(tokens, resumed))[rows]
    return train.blocks(tokens, written, rows, drafts, cap)


def window(
    eligible: np.ndarray, anchors: int, span: int, rng: np.random.Generator
) -> np.ndarray:
    """Up to `anchors` of the `eligible` rows, drawn at random from one span of
    `span` rows that begins at one of them, in order: the rows a batch's context
    must reach stay within a span and serving's window before it."""
    eligible = np.asarray(eligible, np.int64)
    starts = eligible[eligible <= eligible[-1] - span + 1]
    start = rng.choice(starts) if len(starts) else eligible[0]
    inside = eligible[(eligible >= start) & (eligible < start + span)]
    picked = rng.choice(inside, min(anchors, len(inside)), replace=False)
    return np.sort(picked)


class Opened(NamedTuple):
    """A stream opened for a batch (`Fed.open`)."""

    stream: recording.Stream
    written: np.ndarray  # [rows] the target's choice after each row
    room: np.ndarray  # [rows] how many rows after each the target writes (`ahead`)
    eligible: np.ndarray  # the rows a block can start at: room above 0


def _choices(path: Path, name: str, rows: int) -> np.ndarray:
    if not path.exists():
        raise ValueError(f"no saved choices for {name} ({path})")
    written = np.fromfile(path, dtype="<i4").astype(np.int64)
    if len(written) != rows:
        raise ValueError(f"{name}: {len(written)} saved choices for {rows} rows")
    return written


class Fed(NamedTuple):
    """A stream as a trainer takes it: named, and opened only while a batch needs it,
    so that a run over many streams holds one stream's files at a time."""

    name: str
    base: Path  # the stream's files, base.*
    choices: Path  # its saved choices, <name>.next
    part: str  # "" where no listing names it
    rows: int
    anchorable: int  # rows a block can start at

    def open(self) -> Opened:
        """The stream, its choices, and where its blocks can start."""
        stream = recording.open_stream(self.base)
        written = _choices(self.choices, self.name, stream.summary.rows)
        room = ahead(writable(stream.tok, stream.summary.first_row > 0))
        return Opened(stream, written, room, np.flatnonzero(room > 0))


def streams(
    roots: Sequence[Path], nexts: Path, parts: Sequence[Path] = ()
) -> list[Fed]:
    """Every finished stream under `roots`, by name, with its choices and part.

    A stream's choices are `nexts/<name>.next`, as `greedy.report(save=...)` keeps
    them; a listing names a stream's part, a line `name<TAB>part`. Each stream is
    read once here, to check it and count where its blocks can start, and closed.
    ValueError if two folders hold a stream of one name, a stream has no choices or
    not one a row, or its rows are not at consecutive positions.
    """
    part_of: dict[str, str] = {}
    for listing in parts:
        for line in Path(listing).read_text(encoding="utf-8").split("\n"):
            if line.strip():
                name, part = line.split("\t")[:2]
                part_of[name] = part
    found: dict[str, Path] = {}
    for root in roots:
        for base in recording.find(Path(root)):
            if not recording.beside(base, ".json").exists():
                continue  # a stream that has not closed
            name = base.relative_to(root).as_posix()
            if name in found:
                raise ValueError(f"two folders hold a stream {name}")
            found[name] = base
    fed = []
    for name in sorted(found):
        stream = recording.open_stream(found[name])
        rows = stream.summary.rows
        choices = Path(nexts) / f"{name}.next"
        _choices(choices, name, rows)
        if not stream.in_order():
            raise ValueError(f"{name}: the rows are not at consecutive positions")
        anchorable = int(
            (ahead(writable(stream.tok, stream.summary.first_row > 0)) > 0).sum()
        )
        del stream  # its tapped rows are mapped: close them before the next
        fed.append(
            Fed(name, found[name], choices, part_of.get(name, ""), rows, anchorable)
        )
    return fed


def held(fed: Sequence[Fed], every: int) -> tuple[list[Fed], list[Fed]]:
    """The streams trained on, and those held out: every `every`-th by name, as
    the bridge holds them out (`bridge.report`)."""
    out = {i for i in range(len(fed)) if i % every == every - 1}
    return (
        [f for i, f in enumerate(fed) if i not in out],
        [f for i, f in enumerate(fed) if i in out],
    )


class Sampler:
    """A stream of every part a step, each drawn as its rows to anchor weigh.

    Every step holding every part follows TensorFold's drafter recipe (one batch
    without a language cost that language's acceptance). A part's share of a step's
    loss is `shares` (normalised), or by default its rows to anchor.
    """

    def __init__(
        self,
        fed: Sequence[Fed],
        shares: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.parts: dict[str, list[Fed]] = {}
        for f in fed:
            if f.anchorable:
                self.parts.setdefault(f.part, []).append(f)
        if shares is None:
            shares = {p: sum(f.anchorable for f in s) for p, s in self.parts.items()}
        for part in shares:
            if part not in self.parts:
                raise ValueError(f"no stream of the part {part} has a row to anchor")
        for part in self.parts:
            if part not in shares:
                raise ValueError(f"the part {part} has streams and no share")
        total = sum(shares.values())
        self.shares = {p: shares[p] / total for p in self.parts}

    def step(self) -> list[tuple[Fed, float]]:
        """A stream of each part, drawn by its rows to anchor, and the part's share."""
        taken = []
        for part, held_streams in self.parts.items():
            sizes = np.array([f.anchorable for f in held_streams], float)
            pick = self.rng.choice(len(held_streams), p=sizes / sizes.sum())
            taken.append((held_streams[pick], self.shares[part]))
        return taken
