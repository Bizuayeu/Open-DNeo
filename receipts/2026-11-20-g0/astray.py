"""Whose are the logged DFlash2 rounds that no request of a run holds?

    python astray.py LOG RUN AGAIN

`open_dneo.logged` gives each round of a server's log to a request of RUN. This
takes the rounds left over and asks whether each is an again prompt's: AGAIN
holds each reply's prompt and reply sent again as one prompt for 1 token, a JSON
line each with its `prompt_ids` and `reply_ids`. A round is that prompt's where
its anchor sits at the prompt's length and is the token it gave.
"""

import json
import pathlib
import sys

from open_dneo import logged, runs


def main(log: str, run: str, again: str) -> int:
    rounds = logged.read(pathlib.Path(log))
    taken, _ = logged.assign(rounds, runs.read(pathlib.Path(run)))
    held = {id(r) for each in taken for r in each}
    left = [r for r in rounds if id(r) not in held]
    text = pathlib.Path(again).read_text(encoding="utf-8")
    agains = [json.loads(x) for x in text.split("\n") if x.strip()]
    ends = {(len(a["prompt_ids"]), a["reply_ids"][0]) for a in agains if a["reply_ids"]}
    fits = sum((r.pos, r.anchor) in ends for r in left)
    after = all(rounds.index(r) >= len(rounds) - len(left) for r in left)
    print(f"rounds {len(rounds)}, held by a request {len(rounds) - len(left)}")
    print(f"left {len(left)}, at an again prompt's end with the token it gave: {fits}")
    print(f"the left rounds come after every held one in the log: {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
