"""Copies of a stream's own earlier tokens, as TensorFold's lanes propose them.

Where a stream's last tokens repeat a run from earlier in the stream, what
followed that run then is a guess at what follows now. Serving's lane core asks a
`SuffixLookup` for such a continuation before each round, and verifies it in
place of a drafter's chain where enough tokens match. A drafter's tokens a round
are therefore not a reply's tokens a round: the rounds that are copies have to be
told apart.

`SuffixLookup` follows `zig/src/core/lanes/proposer.zig` of TensorFold at commit
e3fb56cb09061a979cb02080601fc6f7870d3362, state for state: its answers depend on
what it was asked before, and on how its earlier proposals fared.
"""

from __future__ import annotations

from collections.abc import Sequence

# The longest n-gram the lookup indexes by.
MAX_NGRAM = 8


class SuffixLookup:
    """One stream's copy proposer.

    The history's n-grams are indexed as it grows. A proposal is what followed
    the earlier run that matches the most tokens back from the end, the latest of
    equals, if that match is `min_match` long. After `window` proposals in a row
    of which nothing landed, the next `silence_rounds` asks are left unanswered.
    The defaults are the lane host's, which asks with a `min_match` of 4.
    """

    def __init__(
        self,
        ngram: int = 3,
        min_match: int = 4,
        max_extension: int = 64,
        silence_rounds: int = 16,
        window: int = 4,
    ) -> None:
        if not 1 <= ngram <= MAX_NGRAM:
            raise ValueError(f"an n-gram of 1 to {MAX_NGRAM} tokens, not {ngram}")
        if min_match < ngram:
            raise ValueError(
                f"a match of {min_match} is shorter than the n-gram's {ngram}"
            )
        self.ngram, self.min_match, self.max_extension = ngram, min_match, max_extension
        self.silence_rounds, self.window = silence_rounds, max(1, window)
        self.index: dict[tuple[int, ...], list[int]] = {}
        self.indexed = 0  # the history's tokens that the index covers
        self.last_key: tuple[int, ...] = ()  # the run that ended the indexed history
        self.recent: list[int] = []  # what landed of each of the last proposals
        self.silent_for = 0  # asks still to be left unanswered
        self.last_match = 0  # the last ask's longest match, proposed from or not
        self.last_end = 0  # where in the history that match's continuation starts
        self.proposals = self.proposed_tokens = 0
        self.judged_tokens = self.accepted_tokens = 0
        self.silenced = 0  # asks that the silence left unanswered

    def propose(self, history: Sequence[int], max_draft: int) -> list[int]:
        """At most `max_draft` tokens that followed the history's last run before."""
        return self.propose_after(history, (), max_draft)

    def propose_after(
        self, history: Sequence[int], tail: Sequence[int], max_draft: int
    ) -> list[int]:
        """The same for `history` followed by `tail`, which is matched and never copied from.

        An ask without room (`max_draft` under 1), or of a history too short to
        hold a run and a token after it, is no ask: it uses up no silence.
        """
        n = self.ngram
        self.last_match = 0
        # Another history than the one indexed: the run that ended the indexed part
        # is not there now (a shorter history has no run there at all).
        if self.indexed and self._run(history, self.indexed) != self.last_key:
            self.index, self.indexed = {}, 0
        if max_draft <= 0 or len(history) < n + 1:
            return []
        if self.silent_for > 0:
            self.silent_for -= 1
            self.silenced += 1
            return []
        for position in range(max(self.indexed, n - 1), len(history)):
            self.index.setdefault(self._run(history, position + 1), []).append(position)
        self.indexed = len(history)
        self.last_key = self._run(history, self.indexed)
        joined = [*history, *tail] if tail else history
        positions = self.index.get(tuple(joined[-n:]))
        if positions is None:
            return []  # as the lookup does: where the last match's continuation was stays
        best_end = best = 0
        for position in reversed(positions):
            end = position + 1
            if end >= len(history):
                continue  # nothing follows it in the history
            length = 0
            while (
                length < self.max_extension
                and length < end
                and history[end - 1 - length] == joined[len(joined) - 1 - length]
            ):
                length += 1
            if length > best:
                best, best_end = length, end
                if length >= self.max_extension:
                    break
        self.last_match, self.last_end = best, best_end
        if best < self.min_match:
            return []
        copied = list(history[best_end : best_end + max_draft])
        if copied:
            self.proposals += 1
            self.proposed_tokens += len(copied)
        return copied

    def observe(self, proposed: int, accepted: int) -> None:
        """A verified proposal of `proposed` tokens, of which `accepted` landed."""
        self.judged_tokens += proposed
        self.accepted_tokens += accepted
        if proposed <= 0:
            return
        self.recent = [*self.recent, accepted][-self.window :]
        if len(self.recent) >= self.window and max(self.recent) == 0:
            self.silent_for = self.silence_rounds
            self.recent = []

    def _run(self, tokens: Sequence[int], end: int) -> tuple[int, ...]:
        """The n-gram of `tokens` that ends before `end`."""
        return tuple(tokens[end - self.ngram : end])
