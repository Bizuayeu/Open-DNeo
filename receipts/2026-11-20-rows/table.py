"""The receipt's table, from the report `open-dneo rows` printed (rows.txt).

python table.py rows.txt > table.txt
"""

import re
import sys

LINE = re.compile(
    r"^(\w+): (\d+) replies, (\d+) anchors with both rows, (\d+) without; "
    r"the drafter alone over reply rows (\d+) tokens in (\d+) rounds, [\d.]+ a round; "
    r"over prompt rows (\d+) in (\d+), [\d.]+ a round; "
    r"the first draft the same at (\d+) anchors \([\d.]+\), every draft at (\d+)$"
)


def main(path):
    print("class replies positions reply_rows prompt_rows apart first_same all_same")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            klass = m.group(1)
            replies, positions, without, t1, r1, t2, r2, first, every = map(
                int, m.groups()[1:]
            )
            if without or t1 != t2:
                sys.exit(
                    f"{klass}: {without} positions without both rows, tokens {t1} and {t2}"
                )
            a, b = t1 / r1, t2 / r2
            print(
                f"{klass} {replies} {positions} {a:.3f} {b:.3f} {a - b:.3f}"
                f" {100 * first / positions:.1f}% {100 * every / positions:.1f}%"
            )


if __name__ == "__main__":
    main(sys.argv[1])
