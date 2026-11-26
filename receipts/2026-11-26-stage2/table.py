"""table.py REPORTS [LABEL ...]: this receipt's two numbers from `open-dneo score` reports.

    python table.py reports > table.txt

Reads REPORTS/DFlash2.txt and REPORTS/LABEL.txt for each LABEL given (by default every
report there). For each drafter it prints:

- the cut-free sum: per class, the shares landed whole at positions 1 to 7 summed (the
  drafts an anchor lands with every draft held, from the line "by position, every
  anchor's product and its share landed whole"), as a share of DFlash2's, and the mean
  of code, chat and ja;
- G2's price: per class and cut from 0.3 to 0.9, the line "by the cut alone" gives
  tokens and drafts verified a round, and a class's seconds of rounds are its reply
  tokens over tokens a round times (41.87 + 5.87 x drafts verified a round) ms. The
  drafter takes the one cut with the least seconds over the 320 requests, and its speed,
  the reply tokens over those seconds, is written as a share of DFlash2's at DFlash2's
  own cut, over all, as the mean of code and agent, and per class.

Refuses a report without 320 requests paired, without the five classes, or with a
class's reply tokens unlike DFlash2's.
"""

import pathlib
import re
import sys

CLASSES = ("code", "agent", "chat", "ja", "prose")
JUDGED = ("code", "chat", "ja")
G2 = ("code", "agent")
CUTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ROUND_MS, DRAFT_MS = 41.87, 5.87  # rounds.ROUND_MS and rounds.DRAFT_MS
HEAD = re.compile(r"^(\w+): \d+ replies; serving (\d+) tokens over ")
WHOLE = re.compile(r"^(\w+): chains by .* share landed whole: (.*)$")
CUT = re.compile(r"^(\w+): by the cut alone, not serving's price of a row, .*?, (.*)$")


def read(path):
    text = path.read_text(encoding="utf-8")
    if "run: 320 requests, 320 paired with a stream" not in text:
        raise SystemExit(f"{path}: not 320 requests paired")
    tokens, whole, at = {}, {}, {}
    for line in text.split("\n"):
        if m := HEAD.match(line):
            tokens[m[1]] = int(m[2])
        elif m := WHOLE.match(line):
            shares = [float(item.split(" ")[2]) for item in m[2].split(", ")]
            if len(shares) != 7:
                raise SystemExit(f"{path} {m[1]}: {len(shares)} positions, not 7")
            whole[m[1]] = sum(shares)
        elif m := CUT.match(line):
            pairs = {}
            for item in m[2].split(", "):
                name, tok, drafted = item.rsplit(" ", 2)
                pairs[name] = (float(tok), float(drafted))
            at[m[1]] = {x: pairs[f"{x:.1f}"] for x in CUTS}
    if not set(CLASSES) == set(tokens) == set(whole) == set(at):
        raise SystemExit(f"{path}: not the five classes")
    return tokens, whole, at


def price(tokens, at):
    """The cut with the least ms of rounds, and the tokens a ms by class there."""
    ms = {
        x: {
            c: tokens[c] / at[c][x][0] * (ROUND_MS + DRAFT_MS * at[c][x][1])
            for c in CLASSES
        }
        for x in CUTS
    }
    best = min(CUTS, key=lambda x: sum(ms[x].values()))
    speed = sum(tokens.values()) / sum(ms[best].values())
    return best, speed, {c: tokens[c] / ms[best][c] for c in CLASSES}


def main(folder, *labels):
    folder = pathlib.Path(folder)
    ref_tokens, ref_whole, ref_at = read(folder / "DFlash2.txt")
    _, ref_speed, ref_by = price(ref_tokens, ref_at)
    labels = labels or sorted(
        p.stem for p in folder.glob("*.txt") if p.stem != "DFlash2"
    )
    for label in ("DFlash2", *labels):
        tokens, whole, at = read(folder / f"{label}.txt")
        if tokens != ref_tokens:
            raise SystemExit(f"{label}: other reply tokens than DFlash2's")
        share = {c: whole[c] / ref_whole[c] for c in CLASSES}
        mean = sum(share[c] for c in JUDGED) / len(JUDGED)
        cut, speed, by = price(tokens, at)
        by = {c: by[c] / ref_by[c] for c in CLASSES}
        print(
            f"{label}: cut-free sum mean(code,chat,ja) {mean:.4f}; "
            + " ".join(f"{c} {share[c]:.3f}" for c in CLASSES)
            + f"; G2's price at cut {cut}: {speed / ref_speed:.4f} of DFlash2's;"
            + f" code and agent {sum(by[c] for c in G2) / len(G2):.4f}; "
            + " ".join(f"{c} {by[c]:.3f}" for c in CLASSES)
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
