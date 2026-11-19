"""Read and check the rows a TensorFold server records with `--record DIR`.

Each server makes its own folder `DIR/<pid>-<k>`, and each stream writes `<n>.*` in it:

    .tok   u32: the token at each committed row
    .pos   u32: the row's position in the cache
    .kind  u8: 1 for a reply's row, 0 for a prompt's
    .st    the tapped rows, `row_bytes` each: every tap in `taps` order, equally wide
    .top   only with logprobs: entry j holds the best k ids (u32), then their log
           probabilities (f32), from the row at position `prompt_len - 1 + j`
    .json  rows, first_row, prompt_len, taps, row_bytes, dtype, top_k and gap,
           written when the stream closes

Row i of the first four files is one row, at position `.pos[i]`. The format is
TensorFold's: `zig/src/core/lanes/recorder.zig` and
`docs/recipes/training-a-drafter.md` at commit
e3fb56cb09061a979cb02080601fc6f7870d3362.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A tapped value's bytes, by the recorder's name for its type.
VALUE_BYTES = {"bf16": 2, "f16": 2, "f32": 4}
# Every file a stream may have.
EXTENSIONS = (".tok", ".pos", ".kind", ".st", ".top", ".json")


@dataclass(frozen=True)
class Summary:
    """A stream's .json, written when the stream closed."""

    rows: int
    first_row: int  # the first position the prompt pass computed: the rows start here
    prompt_len: int
    taps: tuple[int, ...]
    row_bytes: int
    dtype: str
    top_k: int  # 0: no .top
    gap: str | None  # why the rows stop before the stream's end

    def width(self) -> int:
        """The values one tap holds in a row; ValueError if the row does not divide."""
        if self.dtype not in VALUE_BYTES:
            known = ", ".join(VALUE_BYTES)
            raise ValueError(f'dtype "{self.dtype}" is not one of {known}')
        taps_bytes = len(self.taps) * VALUE_BYTES[self.dtype]
        if not taps_bytes or not self.row_bytes or self.row_bytes % taps_bytes:
            raise ValueError(
                f"row_bytes {self.row_bytes} does not divide into {len(self.taps)}"
                f" taps of {self.dtype} values"
            )
        return self.row_bytes // taps_bytes

    def file_rows(self) -> dict[str, int]:
        """The bytes one row adds to each of the four files a stream appends to."""
        return {".tok": 4, ".pos": 4, ".kind": 1, ".st": self.row_bytes}


def parse_summary(text: str | bytes) -> Summary:
    """A .json's text; ValueError names the field that is missing or wrong."""
    try:
        fields = json.loads(text)
    except ValueError as error:
        raise ValueError(f"not JSON ({error})") from error
    if not isinstance(fields, dict):
        raise ValueError("not a JSON object")

    def whole(name: str) -> int:
        value = fields.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f'"{name}" is not a whole number of 0 or more')
        return value

    rows, first_row, prompt_len = whole("rows"), whole("first_row"), whole("prompt_len")
    taps = fields.get("taps")
    if not isinstance(taps, list) or any(type(t) is not int or t < 0 for t in taps):
        raise ValueError('"taps" is not a list of whole numbers of 0 or more')
    row_bytes, dtype = whole("row_bytes"), fields.get("dtype")
    if not isinstance(dtype, str):
        raise ValueError('"dtype" is not text')
    top_k, gap = whole("top_k"), fields.get("gap")
    if gap is not None and not isinstance(gap, str):
        raise ValueError('"gap" is neither null nor text')
    return Summary(
        rows, first_row, prompt_len, tuple(taps), row_bytes, dtype, top_k, gap
    )


@dataclass(frozen=True, eq=False)
class Stream:
    """A finished stream's rows, as read-only views of its files."""

    summary: Summary
    tok: np.ndarray  # [rows] u32
    pos: np.ndarray  # [rows] u32
    kind: np.ndarray  # [rows] u8
    st: np.ndarray  # [rows, row_bytes] u8, as written
    top_ids: np.ndarray  # [entries, top_k] u32
    top_logprobs: np.ndarray  # [entries, top_k] f32

    def tapped(self, start: int, stop: int) -> np.ndarray:
        """Rows [start, stop) as float32, shaped [rows, taps, width]."""
        summary = self.summary
        width = summary.width()
        raw = np.ascontiguousarray(self.st[start:stop])
        if summary.dtype == "bf16":  # a bfloat16 is the high half of a float32
            values = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            plain = {"f16": "<f2", "f32": "<f4"}[summary.dtype]
            values = raw.view(plain).astype(np.float32)
        return values.reshape(len(raw), len(summary.taps), width)

    def in_order(self) -> bool:
        """Whether row i is at position first_row + i, as the recorder writes them."""
        positions = np.asarray(self.pos, np.int64)
        return bool(
            (positions == self.summary.first_row + np.arange(len(positions))).all()
        )

    def top_entry(self) -> np.ndarray:
        """Each row's entry in .top, or -1 for a row that has none.

        Entry j is what the target computed at position prompt_len - 1 + j: the
        distribution that drew reply token j.
        """
        entry = self.pos.astype(np.int64) - (self.summary.prompt_len - 1)
        return np.where((entry >= 0) & (entry < len(self.top_ids)), entry, -1)


def beside(base: Path, ext: str) -> Path:
    """The file of the stream `base` with extension `ext`."""
    return base.parent / (base.name + ext)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        raise ValueError(f"no {path.suffix} file") from None


def _view(path: Path, dtype: np.dtype | str, shape: tuple[int, ...]) -> np.ndarray:
    """The first `shape` items of a file, read-only; ValueError if it holds fewer."""
    need, have = math.prod(shape) * np.dtype(dtype).itemsize, _size(path)
    if have < need:
        raise ValueError(
            f"{path.suffix} holds {have} bytes; {shape[0]} rows need {need}"
        )
    if need == 0:  # an empty file cannot be mapped
        return np.empty(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def read_summary(base: Path) -> Summary:
    """The summary of the stream `base`; ValueError if it has none or a wrong one."""
    try:
        text = beside(base, ".json").read_bytes()
    except FileNotFoundError:
        raise ValueError("no .json: the stream has not closed") from None
    try:
        return parse_summary(text)
    except ValueError as error:
        raise ValueError(f".json: {error}") from error


def open_stream(base: Path) -> Stream:
    """The finished stream whose files are `base`.*; ValueError if it cannot be read.

    Only the first `rows` rows of each file are read. A stream whose write failed
    stops with a gap, and its files may hold a part of the rows that failed.
    """
    summary = read_summary(base)
    rows, k = summary.rows, summary.top_k
    shapes = {".tok": (rows,), ".pos": (rows,), ".kind": (rows,)}
    shapes[".st"] = (rows, summary.row_bytes)
    types = {".tok": "<u4", ".pos": "<u4", ".kind": "u1", ".st": "u1"}
    views = {ext: _view(beside(base, ext), types[ext], shapes[ext]) for ext in shapes}
    entry = np.dtype([("ids", "<u4", (k,)), ("logprobs", "<f4", (k,))])
    entries = 0
    if k:
        held = _size(beside(base, ".top"))
        if held % entry.itemsize:
            raise ValueError(
                f".top holds {held} bytes, not whole entries of {entry.itemsize}"
                f" (the best {k})"
            )
        entries = held // entry.itemsize
    top = _view(beside(base, ".top"), entry, (entries,)) if k else np.empty(0, entry)
    return Stream(
        summary,
        views[".tok"],
        views[".pos"],
        views[".kind"],
        views[".st"],
        top["ids"],
        top["logprobs"],
    )


@dataclass(frozen=True)
class Checked:
    """What a look at one stream found."""

    summary: Summary | None = None  # None: no .json, or one that cannot be read
    unfinished: bool = False  # no .json: the stream had not closed
    first: int = 0  # the position of its first row
    last: int = 0
    prompt_rows: int = 0
    reply_rows: int = 0
    paired: int = 0  # rows with an entry in .top
    entries: int = 0  # entries in .top
    findings: tuple[str, ...] = ()


def check(base: Path) -> Checked:
    """Whether the stream `base` is what the recorder writes, and what it holds."""
    if not beside(base, ".json").exists():
        return Checked(unfinished=True)
    try:
        stream = open_stream(base)
    except ValueError as error:
        return Checked(findings=(str(error),))
    summary, findings = stream.summary, []
    if summary.rows:
        try:
            summary.width()
        except ValueError as error:
            findings.append(str(error))
    if summary.gap is None:
        for ext, row in summary.file_rows().items():
            held, need = _size(beside(base, ext)), summary.rows * row
            if held > need:
                findings.append(
                    f"{ext} holds {held} bytes; {summary.rows} rows are {need},"
                    " and the stream names no gap"
                )
    if not summary.top_k and beside(base, ".top").exists():
        findings.append("top_k is 0, but there is a .top file")
    pos, kind = np.asarray(stream.pos, np.int64), np.asarray(stream.kind)
    astray = np.flatnonzero(pos != summary.first_row + np.arange(summary.rows))
    if astray.size:
        i = int(astray[0])
        findings.append(
            f"row {i} is at position {pos[i]}, not {summary.first_row + i}"
            f" (first_row {summary.first_row} + {i})"
        )
    unkind = np.flatnonzero(kind != (pos >= summary.prompt_len))
    if unkind.size:
        i = int(unkind[0])
        findings.append(
            f"row {i} at position {pos[i]} has kind {kind[i]};"
            f" the prompt is {summary.prompt_len} tokens"
        )
    replies = int(np.count_nonzero(kind))
    return Checked(
        summary=summary,
        first=int(pos[0]) if summary.rows else 0,
        last=int(pos[-1]) if summary.rows else 0,
        prompt_rows=summary.rows - replies,
        reply_rows=replies,
        paired=int(np.count_nonzero(stream.top_entry() >= 0)),
        entries=len(stream.top_ids),
        findings=tuple(findings),
    )


def _natural(text: str) -> list:
    """A sort key that reads runs of digits as numbers: 7-9 comes before 7-10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def find(root: Path) -> list[Path]:
    """Every stream under `root`, as its files' path without the extension.

    A stream is the files `<n>.*` of one folder, `n` a number. Links are not
    followed. Streams come folder by folder, in number order.
    """
    bases = set()
    for folder, _, names in os.walk(root):
        for name in names:
            stem, ext = os.path.splitext(name)
            if stem.isascii() and stem.isdigit() and ext in EXTENSIONS:
                bases.add(Path(folder) / stem)
    return sorted(bases, key=lambda b: (_natural(b.parent.as_posix()), int(b.name)))


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _line(checked: Checked) -> str:
    """One stream in a few words."""
    summary = checked.summary
    if not summary.rows:
        text = "0 rows"
    else:
        text = (
            f"{summary.rows} rows at {checked.first}..{checked.last},"
            f" {checked.prompt_rows} prompt and {checked.reply_rows} reply"
        )
    if summary.top_k:
        text += f", top {summary.top_k} x {checked.entries}"
    if summary.gap is not None:
        text += f", gap: {summary.gap}"
    return text


def report(root: Path, each: bool = False) -> tuple[list[str], bool]:
    """What the recordings under `root` hold, and whether nothing is wrong with them.

    `each` also lists every stream. A stream that has not closed is counted and
    left unread: it is not a finding.
    """
    bases = find(root)
    if not bases:
        return ["no streams found"], False
    lines, unfinished, found = [], 0, 0
    rows = Counter()
    layouts: dict[tuple, list] = {}  # by what a row holds: a summary, streams, rows
    gaps = Counter()
    for base in bases:
        name = base.relative_to(root).as_posix()
        checked = check(base)
        found += len(checked.findings)
        if checked.unfinished:
            unfinished += 1
            lines += [f"{name}: unfinished, no .json"] if each else []
            continue
        if checked.summary is not None:
            summary = checked.summary
            lines += [f"{name}: {_line(checked)}"] if each else []
            rows.update(
                prompt=checked.prompt_rows,
                reply=checked.reply_rows,
                paired=checked.paired,
            )
            if summary.gap is not None:
                gaps[summary.gap] += 1
            if summary.rows:
                key = (summary.taps, summary.row_bytes, summary.dtype)
                layout = layouts.setdefault(key, [summary, 0, 0])
                layout[1] += 1
                layout[2] += summary.rows
        lines += [f"{name}: {finding}" for finding in checked.findings]
    lines.append(
        f"streams {len(bases)}: {len(bases) - unfinished} finished,"
        f" {unfinished} unfinished"
    )
    lines.append(
        f"rows {rows['prompt'] + rows['reply']}: {rows['prompt']} prompt,"
        f" {rows['reply']} reply, {rows['paired']} with a .top entry"
    )
    for summary, streams, held in layouts.values():
        try:
            wide = f" x {summary.width()}"
        except ValueError:  # already a finding of each of its streams
            wide = ""
        lines.append(
            f"layout taps {list(summary.taps)}{wide} {summary.dtype},"
            f" {summary.row_bytes} bytes a row:"
            f" {_count(streams, 'stream')}, {_count(held, 'row')}"
        )
    lines += [f'gap "{gap}": {_count(n, "stream")}' for gap, n in gaps.items()]
    lines.append(f"findings {found}")
    return lines, found == 0
