"""copies.py FOLDER: where the lane core's copy gate could pass in each recorded reply.

Seen, not judged. For every finished stream under FOLDER (a --record folder, or a launch's folder), the stream's
tokens up to each reply row are asked for a first copy as the lanes ask: a lookup of 3 tokens after a match of 8
or more, of which 2 at least can be copied. This is the gate alone, asked afresh at every row: not the silence
after rejected copies, not the growth of a copy's width, not which rows a round starts at. So the count is of rows
where a copy round could begin, and a reply with none cannot have had a copy round.

Per stream: its reply rows, the rows where the gate passes, how many of those would have landed their first
token, and the longest match. A stream whose rows do not begin at 0 was resumed from kept state: the tokens before
its first row are not in the recording, the lanes had them, and its count is a lower bound (marked).
Last line: "copies: N streams, R reply rows, G at the gate (P%), L of them land their first token".
"""

import sys
from pathlib import Path

import numpy as np

from open_dneo import copies, recording

FIRST_COPY = 3  # the lanes' first copy window for GLM: first_copy_rows 4, one of them the pending row
ENTER_MATCH = 8  # the match a copy needs to take a round (copy_match)
LEAST = 2  # tokens a copy must hold


def gate(tokens: list[int], reply_from: int) -> tuple[int, int, int]:
    """Rows of `tokens` from `reply_from` on where a first copy passes, those whose first token is right, the longest match."""
    passed = landed = longest = 0
    for row in range(reply_from, len(tokens)):
        look = copies.SuffixLookup()
        copied = look.propose(tokens[: row + 1], FIRST_COPY)
        longest = max(longest, look.last_match)
        if len(copied) >= LEAST and look.last_match >= ENTER_MATCH:
            passed += 1
            landed += row + 1 < len(tokens) and copied[0] == tokens[row + 1]
    return passed, landed, longest


def main(root: Path) -> None:
    streams = rows = at_gate = land = 0
    for base in recording.find(root):
        if not recording.beside(base, ".json").exists():
            continue
        stream = recording.open_stream(base)
        tokens = np.asarray(stream.tok, np.int64).tolist()
        reply = int((np.asarray(stream.kind) == 1).sum())
        if reply == 0:
            continue
        reply_from = len(tokens) - reply
        passed, landed, longest = gate(tokens, reply_from)
        first = stream.summary.first_row
        mark = "" if first == 0 else f" (rows begin at {first}: a lower bound)"
        print(
            f"{base.relative_to(root).as_posix()}: {reply} reply rows, {passed} at the gate, {landed} land their first"
            f" token, longest match {longest}{mark}"
        )
        streams, rows, at_gate, land = (
            streams + 1,
            rows + reply,
            at_gate + passed,
            land + landed,
        )
    share = f"{100 * at_gate / rows:.1f}%" if rows else "none"
    print(
        f"copies: {streams} streams, {rows} reply rows, {at_gate} at the gate ({share}), {land} of them land their first token"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
