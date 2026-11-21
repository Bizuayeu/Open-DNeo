"""table.py ARMS_DIR [ARM ...]: G1's numbers from `open-dneo score --served --run` reports, beside DFlash2's.

Reads ARMS_DIR/dflash2.txt (DFlash2 as it is) and ARMS_DIR/ARM.txt for each ARM given (default uniform, fc, null,
uniform4, those that exist). Per class and arm: tokens a round with copies (as serving takes the rounds) and with the drafter alone, each
as a share of DFlash2's, and the drafts a round the drafter alone lands (tokens a round less the one the target
always gives), as a share of DFlash2's. Prints markdown tables. Checks that every report paired 320 requests.
"""

import pathlib
import re
import sys

CLASSES = ("code", "agent", "chat", "ja", "prose")
COPIES = re.compile(
    r"^(\w+): \d+ replies; serving .*?; the instrument (\d+) over (\d+), "
)
ALONE = re.compile(r"^(\w+): the drafter alone, (\d+) rounds give (\d+) tokens, ")


def read(path):
    text = path.read_text(encoding="utf-8")
    if "run: 320 requests, 320 paired with a stream" not in text:
        raise SystemExit(f"{path}: not 320 requests paired")
    got = {}
    for line in text.split("\n"):
        if found := COPIES.match(line):
            got.setdefault(found[1], {})["copies"] = (int(found[2]), int(found[3]))
        elif found := ALONE.match(line):
            got.setdefault(found[1], {})["alone"] = (int(found[3]), int(found[2]))
    entrance = next(
        (line for line in text.split("\n") if line.startswith("the entrance: ")),
        "DFlash2's own",
    )
    return got, entrance


def main(folder, *arms):
    folder = pathlib.Path(folder)
    base, _ = read(folder / "dflash2.txt")
    arms = arms or [
        a
        for a in ("uniform", "fc", "null", "uniform4")
        if (folder / f"{a}.txt").exists()
    ]
    print(
        "| class | arm | with copies | share | the drafter alone | share | drafts landed a round | share |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for klass in CLASSES:
        t, r = base[klass]["copies"]
        a, ar = base[klass]["alone"]
        print(
            f"| {klass} | DFlash2 | {t / r:.3f} | 1 | {a / ar:.3f} | 1 | {a / ar - 1:.3f} | 1 |"
        )
        for arm in arms:
            got, _ = read(folder / f"{arm}.txt")
            t2, r2 = got[klass]["copies"]
            a2, ar2 = got[klass]["alone"]
            if (t2, a2) != (t, a):
                raise SystemExit(f"{arm} {klass}: other tokens than DFlash2's")
            print(
                f"| {klass} | {arm} | {t2 / r2:.3f} | {(t2 / r2) / (t / r):.3f} | {a2 / ar2:.3f}"
                f" | {(a2 / ar2) / (a / ar):.3f} | {a2 / ar2 - 1:.3f} | {(a2 / ar2 - 1) / (a / ar - 1):.3f} |"
            )
    for arm in arms:
        print(f"- {arm}: {read(folder / f'{arm}.txt')[1]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
