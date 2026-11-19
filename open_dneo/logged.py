"""Serving's drafts as its log gives them, beside the instrument's at the same position.

TensorFold's GLM engine logs its first rounds' DFlash2 drafts on rank 0 when
`TF_GLM_DFLASH2_LOG` is set (`dflash2 round` in `zig/src/families/glm/dflash.zig`):
a round's slot, its anchor's position and token, and the drafts it proposed for
the positions after the anchor, all of the block's, in order. `read` takes those
lines and leaves the rest of the log.

The log names no request. Requests sent one at a time have their rounds in their
order, so `assign` walks the rounds and the requests together: a round stays with
the current request where its anchor is that request's token at that position,
after the request's rounds so far, and otherwise goes to the first later request
whose token it is. A round no request holds is counted, not placed.

`compare` sets a request's logged rounds beside the instrument's proposals at the
same anchors: how many rounds have the instrument's first draft, how many drafts
agree from the first on, and how many rounds agree in every draft.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from . import runs

ROUND = re.compile(
    r"dflash2 round \d+: slot (\d+) pos (\d+) kept \d+ anchor (\d+) drafts \{([\d,\s]*)\}"
)


@dataclass(frozen=True)
class Logged:
    """One round serving logged: its anchor and the drafts after it."""

    slot: int
    pos: int  # the anchor's position
    anchor: int  # the anchor's token
    drafts: tuple[int, ...]  # for positions pos + 1, pos + 2, ...


@dataclass(frozen=True)
class Agreement:
    """Logged rounds against the instrument's proposals at their anchors."""

    rounds: int = 0  # rounds whose anchor has a row in the recording
    first: int = 0  # of them, rounds whose first draft is the instrument's
    leading: int = 0  # drafts that are the instrument's from each round's first on
    whole: int = 0  # rounds whose every draft is the instrument's
    unrecorded: int = 0  # rounds whose anchor has no row in the recording

    def __add__(self, other: Agreement) -> Agreement:
        return Agreement(
            *(getattr(self, f.name) + getattr(other, f.name) for f in fields(self))
        )


def read(path: Path) -> list[Logged]:
    """The rounds a log holds, in its order; ValueError for a round's line it cannot take."""
    found = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for number, each in enumerate(text.split("\n"), 1):
        if "dflash2 round " not in each:
            continue
        match = ROUND.search(each)
        if match is None:
            raise ValueError(f"line {number} is a round's, and not as it is read")
        slot, pos, anchor, drafts = match.groups()
        listed = tuple(int(d) for d in drafts.replace(",", " ").split())
        found.append(Logged(int(slot), int(pos), int(anchor), listed))
    return found


def _holds(request: runs.Request, pos: int, anchor: int) -> bool:
    """Whether `anchor` is `request`'s reply token at `pos`."""
    at = pos - len(request.prompt)
    return 0 <= at < len(request.reply) and request.reply[at] == anchor


def assign(
    rounds: Sequence[Logged], requests: Sequence[runs.Request]
) -> tuple[list[list[Logged]], int]:
    """Each request's logged rounds, in the requests' order, and the rounds none holds."""
    taken: list[list[Logged]] = [[] for _ in requests]
    current, last, astray = 0, -1, 0
    for each in rounds:
        if (
            current < len(requests)
            and each.pos > last
            and _holds(requests[current], each.pos, each.anchor)
        ):
            taken[current].append(each)
            last = each.pos
            continue
        later = next(
            (
                j
                for j in range(current + 1, len(requests))
                if _holds(requests[j], each.pos, each.anchor)
            ),
            None,
        )
        if later is None:
            astray += 1
            continue
        current, last = later, each.pos
        taken[current].append(each)
    return taken, astray


def compare(
    rounds: Sequence[Logged],
    positions: np.ndarray,
    tokens: np.ndarray,
    drafts: np.ndarray,
) -> Agreement:
    """Logged rounds beside the instrument's anchors: their `positions`, `tokens`, `drafts`.

    ValueError where a logged anchor is not the stream's token at its position.
    """
    at = {int(p): i for i, p in enumerate(positions.tolist())}
    total = Agreement()
    for each in rounds:
        i = at.get(each.pos)
        if i is None:
            total += Agreement(unrecorded=1)
            continue
        if int(tokens[i]) != each.anchor:
            raise ValueError(
                f"position {each.pos}: the log's anchor is {each.anchor}, the"
                f" stream's {int(tokens[i])}"
            )
        mine = drafts[i].tolist()
        width = min(len(mine), len(each.drafts))
        lead = next((k for k in range(width) if mine[k] != each.drafts[k]), width)
        total += Agreement(
            rounds=1, first=int(lead > 0), leading=lead, whole=int(lead == width)
        )
    return total
