"""Serving's tokens a round, class by class, from run files.

    python serving.py RUN [RUN ...]

A run file is a JSON line a request (`open_dneo.runs`). Per class: the answered
replies, their tokens less one summed over serving's rounds summed, the count
G0 uses (docs/plan.md, "How G0 is measured"). From the second file on, how many
replies are the same tokens as the first file's reply to the same request.
"""

import collections
import pathlib
import sys

from open_dneo import runs


def main(paths: list[str]) -> int:
    first = None
    for path in paths:
        requests = runs.read(pathlib.Path(path))
        name = pathlib.Path(path).parent.name
        unrounded = sum(r.rounds is None for r in requests)
        print(f"{name}: {len(requests)} answered, {unrounded} without rounds")
        by_class = collections.defaultdict(list)
        for request in requests:
            if request.rounds is not None:
                by_class[request.klass].append(request)
        for klass in sorted(by_class):
            tokens, rounds = runs.served(by_class[klass])
            print(
                f"  {klass}: {len(by_class[klass])} replies, {tokens} tokens over"
                f" {rounds} rounds, {tokens / rounds:.3f} a round"
            )
        if first is None:
            first = (name, {r.id: r.reply for r in requests})
        else:
            same = sum(first[1].get(r.id) == r.reply for r in requests)
            print(
                f"  replies the same tokens as {first[0]}'s: {same} of {len(requests)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
