"""Tokens a round: how far a chain of drafts carries one verify pass.

A round starts at an anchor, the last token known. A drafter proposes drafts for
the positions after it, each with a chance. TensorFold's GLM lanes always verify
the first draft; from the second on a chain ends where the product of its
chances falls under a cut (TF_GLM_DRAFT_CONFIDENCE, 0.7 by default). The target
verifies the kept drafts in one pass. Those that match its own tokens, up to the
first that does not, are accepted, and the pass gives one more token, the
target's own.

`walk` counts a reply as if every round were the drafter's. Serving's are not all:
where a stream's last tokens repeat enough of its earlier ones, the lanes verify
a copy of what followed in place of the drafter's chain. `served` counts the
rounds a stream decoded alone takes, copies included, as the lane core plans them
at TensorFold commit e3fb56cb09061a979cb02080601fc6f7870d3362 (`plan` and
`copyProposal` in `zig/src/core/lanes/windows.zig`, the round in `engine.zig`).

The lanes also size a drafter's window by what its rows cost (`priced`): of the
drafts the cut keeps, they verify the prefix that gives the most expected tokens
a ms, a draft's expected share being the product of the chances to it, at the
price a lone window pays (`allocateDrafts` in `zig/src/core/lanes/allocate.zig`,
called from `allocate` in `windows.zig`). For DFlash2 that drops no draft: a
draft the cut keeps has a chance of 0.7 or more, a first draft one of 1/16 at
the least, being the best share among 16 candidates, and a row's price asks for
less than either (0.285 at the most, for a last row behind drafts sure to land).
A head that takes its chances over the whole vocabulary, as DNeo's does, can
fall under that price with its first draft, and a cut below about 0.29 lets the
price end a chain.

One thing of serving's is left out of `served`: the drafts held for a reply's
first round are chosen from measured round times, and `Lanes.first_depth` is what
an engine that has measured nothing chooses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from . import copies

CUT = 0.7
COPY, HEAD, PLAIN = "copy", "head", "plain"


def held(chances: Sequence[float], cut: float = CUT) -> int:
    """How many of a chain's drafts are verified."""
    chain, count = 1.0, 0
    for i, chance in enumerate(chances):
        chain *= chance
        if chance == 0 or (i > 0 and chain < cut):
            break
        count += 1
    return count


def landed(drafts: Sequence[int], truth: Sequence[int]) -> int:
    """How many drafts match the target's tokens before the first that does not."""
    count = 0
    for draft, token in zip(drafts, truth):
        if draft != token:
            break
        count += 1
    return count


def walk(
    tokens: Sequence[int],
    propose: Callable[[int], tuple[Sequence[int], Sequence[float]]],
    cut: float = CUT,
) -> list[int]:
    """The tokens each round gives over a reply, from its first token on.

    `tokens` are the target's own; the first is known when the rounds begin.
    `propose(at)` gives the drafts for the positions after `at` and their chances.
    A round gives its accepted drafts and one token of the target's own, and the
    last round no more than the reply holds.
    """
    at, steps = 0, []
    while at + 1 < len(tokens):
        drafts, chances = propose(at)
        kept = held(chances, cut)
        step = landed(drafts[:kept], tokens[at + 1 :]) + 1
        steps.append(min(step, len(tokens) - 1 - at))
        at += steps[-1]
    return steps


@dataclass(frozen=True)
class Lanes:
    """What the lane core goes by for a stream decoded alone.

    The defaults are GLM-5.3-Flash's with a DFlash2 drafter and TF_GLM_COPIES unset.
    """

    drafts: int = 7  # the most a round asks the drafter for: a block's
    first_depth: int = 4  # those held for a reply's first round, on a cold engine
    cut: float = CUT
    first_copy: int = 3  # a copy's tokens until one lands whole
    max_copy: int = 31  # the most it grows to: the verify window less the pending row
    enter_match: int = 8  # the tokens that must match for a copy to take a round
    least_copy: int = 2  # a copy of fewer tokens is none
    min_match: int = 4  # the lookup's own least match
    # A lone window's ms at w rows: price_one + price_row * (w - 1), GLM's table of
    # shared rows for DFlash2 (`shared_costs` in `zig/src/families/glm/
    # cuda_copy_policy.zig`), which the lane config lets override the window table
    # at every width (`init` in `zig/src/core/lanes/config.zig`). None: the cut alone.
    price_one: float | None = 76.0
    price_row: float = 4.1
    # What the lanes add to a window's price: 8 ms before any round is timed, then
    # what the rounds take past the table, never under 0 (`overheadMs` and
    # `observeOverhead` in `zig/src/core/lanes/depth.zig`): 0 where rounds take
    # less than the table.
    overhead: float = 0.0


def priced(chances: Sequence[float], kept: int, lanes: Lanes = Lanes()) -> int:
    """How many of the first `kept` drafts the lanes verify for their price.

    The prefix with the most expected tokens a ms, the pending token counting one
    and each draft the product of the chances to it; a tie keeps the shorter.
    """
    if lanes.price_one is None:
        return kept
    reach = expected = 1.0
    best, count = expected / (lanes.price_one + lanes.overhead), 0
    for rows, chance in enumerate(chances[:kept], 1):
        reach *= chance
        expected += reach
        rate = expected / (lanes.price_one + lanes.price_row * rows + lanes.overhead)
        if rate > best:
            best, count = rate, rows
    return count


class Round(NamedTuple):
    """One verify pass over a reply."""

    kind: str  # COPY, HEAD (the drafter's chain) or PLAIN (the pending token alone)
    offered: int  # drafts verified beside the pending token
    kept: int  # of them accepted
    tokens: int  # what the round gave: the accepted and one of the target's own


# G2's time of a round (docs/plan.md): the ms one round of DFlash2 took as served and
# the ms each draft verified added, fitted to the seconds of the same replies served
# at cuts 0.9, 0.7 (twice), 0.5 and 0.2 on 2026-11-23, on two GB10s at tensor
# parallel 2 (R² 0.979 within each request). A drafter whose own round costs more is
# not measured and borrows it. Serving sizes a window by the lanes' price instead
# (`Lanes.price_one`, `priced`).
ROUND_MS = 41.87
DRAFT_MS = 5.87


def timed(taken: Sequence[Round]) -> float:
    """The ms `taken` takes at G2's time of a round."""
    return sum(ROUND_MS + DRAFT_MS * r.offered for r in taken)


def served(
    prompt: Sequence[int],
    reply: Sequence[int],
    propose: Callable[[int], tuple[Sequence[int], Sequence[float]]],
    lanes: Lanes = Lanes(),
    max_new: int | None = None,
) -> list[Round]:
    """The rounds serving takes over `reply` after `prompt`, decoded alone and greedily.

    `propose(at)` gives the drafter's drafts for the positions after reply token
    `at` and their chances. `max_new` is the request's limit of new tokens, which
    bounds a copy and the drafts asked near it; without it the reply is taken to
    have ended of itself. A copy is looked up in all of `prompt` and the reply so
    far, so a prompt given without its beginning misses the copies of that part.
    """

    def room(committed: int) -> int | None:
        return None if max_new is None else max_new - committed

    def most(committed: int) -> int:
        """The drafts a round asks for with `committed` reply tokens out."""
        left = room(committed)
        return lanes.drafts if left is None else min(lanes.drafts, max(1, left - 1))

    def copy(committed: int) -> list[int]:
        """The copy that takes the next round, if one does: an ask of the lookup."""
        left = room(committed)
        wide = width if left is None else min(width, left - 1)
        copied = look.propose(context, wide)
        if len(copied) >= lanes.least_copy and look.last_match >= lanes.enter_match:
            return copied
        return []

    look = copies.SuffixLookup(min_match=lanes.min_match)
    context = [*prompt, *reply[:1]]
    width, depth = lanes.first_copy, min(lanes.first_depth, most(0))
    at, rounds = 0, []
    while at + 1 < len(reply):
        truth = reply[at + 1 :]
        copied = copy(at + 1)
        if copied:
            kind, offered, kept = COPY, len(copied), landed(copied, truth)
            look.observe(offered, kept)
            whole = kept == offered
            width = min(lanes.max_copy, 2 * width + 1) if whole else lanes.first_copy
        else:
            drafts, chances = propose(at)
            offered = priced(chances, held(chances[:depth], lanes.cut), lanes)
            kind, kept = HEAD if offered else PLAIN, landed(drafts[:offered], truth)
        step = min(kept + 1, len(reply) - 1 - at)
        rounds.append(Round(kind, offered, kept, step))
        context += reply[at + 1 : at + 1 + step]
        at += step
        # The lanes ask once more, for the next round's drafts: with a copy ahead they
        # ask the drafter for one. That copy then takes the round, so of this ask only
        # its place in the lookup's silence shows.
        copy(at + 1)
        depth = most(at + 1)
    return rounds
