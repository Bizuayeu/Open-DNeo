"""judge.py REC OUT [--width 4096]: what the launches of README.md show about a row recorder for GLM.

REC holds, for each launch LABEL, what it was launched with (LABEL.want: record, reduce, pair and taps), a line about
its startup (LABEL-hook.txt) and its replies (LABEL/rows.jsonl, again.jsonl and speed.jsonl for the R, F and S lists:
a line a request, with its name, its status and the server's response). OUT holds each recording launch's --record
folder as OUT/LABEL. Streams are read with open_dneo.recording.

Gates (a failure is printed as FAIL and counted):
  replies    every request answered; a reply's tokens are the same drafted and with "draft": false, and the same in
             every launch as in recb (no recorder, the default prompt path)
  files      each recording passes the reader's checks, holds one closed stream a request, and its rows are the launch's
             taps, WIDTH values a tap, bf16
  startup    each launch's hook line ends ok and counts no disabled, fallback or compile failed line in a rank's log
  again      rect and recu (the same launch twice) wrote the same bytes
  shared     the taps rect and recv share hold the same bytes
  feedback   a reply's rows computed again as a prompt's (rect's F list) are closer at the same position than one off;
             the two streams are laid side by side by position, since a pass may start at a kept state

Seen, not judged: each launch's prompt-pass speed on the S list; each R reply's recorded reply rows beside its
completion_tokens and its rounds (a round keeps a row a token it gives, so a reply's last token has a row only when the
reply ended inside a round); whether recf's final rows (the split path) equal rect's (the gather path); whether a
reply's rows are the same drafted and plain; whether a prompt's rows computed again are the same bytes. Last line
"judge: N failures"."""

import argparse
import hashlib
import json
import pathlib
import statistics
import sys

import numpy as np

from open_dneo import recording

p = argparse.ArgumentParser()
p.add_argument("rec")
p.add_argument("out")
p.add_argument("--width", type=int, default=4096)
args = p.parse_args()

REC, OUT = pathlib.Path(args.rec), pathlib.Path(args.out)
fails = 0


def gate(ok: bool, text: str) -> None:
    global fails
    fails += not ok
    print(("  pass " if ok else "  FAIL ") + text)


def replies(label: str, name: str) -> dict[str, dict]:
    """A list's lines of a launch by request name; none if the launch did not send it."""
    path = REC / label / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as lines:
        return {got["name"]: got for got in map(json.loads, lines)}


def sha(got: dict) -> str:
    return got["response"]["tensorfold"]["token_sha256"]


def want(label: str) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in (REC / f"{label}.want").read_text().split()
    )


def taps_of(label: str, layers: int = 45) -> list[int]:
    mask = want(label)["taps"]
    if mask == "unset":
        return [layers]
    return [k + 1 for k in range(64) if int(mask) >> k & 1]


labels = [path.stem for path in sorted(REC.glob("rec*.want"))]
print("launches:", " ".join(labels))

print("replies")
base = replies("recb", "rows.jsonl")
gate(
    len(base) == 8 and all(g["status"] == 200 for g in base.values()),
    f"recb answered its {len(base)} requests",
)
for label in labels:
    rows = replies(label, "rows.jsonl")
    for name, got in rows.items():
        if got["status"] != 200:
            gate(False, f"{label} {name}: status {got['status']}")
    if not rows or any(g["status"] != 200 for g in rows.values()):
        continue
    plain = [n for n in rows if n.endswith("-plain")]
    same = [n for n in plain if sha(rows[n]) == sha(rows[n.removesuffix("-plain")])]
    gate(
        len(same) == len(plain) == 4,
        f"{label}: {len(same)} of {len(plain)} replies are the same drafted and plain",
    )
    if label != "recb":
        equal = [n for n in rows if sha(rows[n]) == sha(base[n])]
        gate(
            len(equal) == len(rows),
            f"{label}: {len(equal)} of {len(rows)} replies are recb's",
        )
speed_base = replies("recb", "speed.jsonl")
for label in labels:
    speed = replies(label, "speed.jsonl")
    if speed and label != "recb":
        equal = [
            n
            for n in speed
            if speed[n]["status"] == 200 and sha(speed[n]) == sha(speed_base[n])
        ]
        gate(
            len(equal) == len(speed),
            f"{label}: {len(equal)} of {len(speed)} first tokens after a long prompt are recb's",
        )

print("startup")
for label in labels:
    path = REC / f"{label}-hook.txt"
    line = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    words = (
        line.split(", startup ")[1].split(",")[0]
        if ", startup " in line
        else "no hook line"
    )
    gate(
        words == "0+0",
        f"{label}: lines with disabled, fallback or compile failed in the ranks' logs: {words}",
    )
    gate(line.endswith(" -> ok"), f"{label}: the hook's look: {line[-60:] or 'none'}")

print(
    "prompt-pass speed on the S list (tokens computed a second of prefill; seen, not judged)"
)
for label in labels:
    speed = replies(label, "speed.jsonl")
    if not speed:
        continue
    rates, cached = [], 0
    for got in speed.values():
        usage, runtime = got["response"]["usage"], got["response"]["tensorfold"]
        kept = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        cached += kept
        rates.append((usage["prompt_tokens"] - kept) / runtime["prefill_seconds"])
    w = want(label)
    print(
        f"  {label} (record {w['record']}, reduce {w['reduce']}, pair {w['pair']}, taps {w['taps']}): median"
        f" {statistics.median(rates):.0f}, from {min(rates):.0f} to {max(rates):.0f} tok/s, {cached} cached tokens"
    )


def computed(got: dict) -> float:
    """The prompt tokens a reply's prompt pass computed a second."""
    usage, runtime = got["response"]["usage"], got["response"]["tensorfold"]
    kept = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return (usage["prompt_tokens"] - kept) / runtime["prefill_seconds"]


ends = [
    replies(label, "speed.jsonl") for label in ("recb", "recb2")
]  # the default path, first and last launch
if all(ends):
    for label in labels:
        speed = replies(label, "speed.jsonl")
        if not speed or label in ("recb", "recb2"):
            continue
        over = [
            computed(got) / statistics.mean(computed(end[name]) for end in ends)
            for name, got in speed.items()
        ]
        print(
            f"  {label} over recb and recb2, prompt by prompt: median {statistics.median(over):.4f},"
            f" from {min(over):.4f} to {max(over):.4f}"
        )


def streams(label: str) -> list[pathlib.Path]:
    return recording.find(OUT / label)


def digest(base: pathlib.Path) -> str:
    """One hash over a stream's four row files."""
    h = hashlib.sha256()
    for ext in (".tok", ".pos", ".kind", ".st"):
        h.update(recording.beside(base, ext).read_bytes())
    return h.hexdigest()


def columns(stream: recording.Stream, tap: int) -> bytes:
    """One tap's values of every row of a stream, as written."""
    i = stream.summary.taps.index(tap)
    w = stream.summary.row_bytes // len(stream.summary.taps)
    return np.ascontiguousarray(stream.st[:, i * w : (i + 1) * w]).tobytes()


print("files")
recorded = [label for label in labels if want(label)["record"] == "yes"]
for label in recorded:
    sent = sum(
        len(replies(label, name))
        for name in ("rows.jsonl", "again.jsonl", "speed.jsonl")
    )
    lines, sound = recording.report(OUT / label)
    bases = streams(label)
    count = next(line for line in lines if line.startswith("streams "))
    gate(
        sound and count.endswith(", 0 unfinished"),
        f"{label}: the reader's checks: {lines[-1]}; {count}",
    )
    gate(len(bases) == sent, f"{label}: {len(bases)} streams for {sent} requests")
    layouts = set()
    for base in bases:
        if (
            recording.beside(base, ".json").exists()
            and recording.beside(base, ".st").exists()
        ):
            s = recording.read_summary(base)
            layouts.add((s.taps, s.row_bytes // max(1, len(s.taps)) // 2, s.dtype))
    gate(
        layouts == {(tuple(taps_of(label)), args.width, "bf16")},
        f"{label}: rows of {sorted(layouts)}",
    )


print("reply rows against the reply's tokens on the R list (seen, not judged)")
for label in recorded:
    for (name, got), base in zip(
        replies(label, "rows.jsonl").items(), streams(label)[:8], strict=False
    ):
        if got["status"] != 200 or not recording.beside(base, ".json").exists():
            continue
        held, usage = recording.open_stream(base), got["response"]["usage"]
        spec = got["response"].get("speculative") or {}
        by_rounds = (spec.get("rounds") or 0) * (spec.get("tokens_per_round") or 0)
        other = (
            ""
            if held.summary.prompt_len == usage["prompt_tokens"]
            else " (NOT this request's stream: prompts differ)"
        )
        print(
            f"  {label} {name}: {int(np.count_nonzero(held.kind == 1))} reply rows, {usage['completion_tokens']} tokens,"
            f" {spec.get('rounds')} rounds, {spec.get('drafted')} drafted, {spec.get('accepted')} accepted,"
            f" rounds x tokens_per_round {by_rounds:.1f}{other}"
        )


def opened(label: str, count: int) -> list[recording.Stream]:
    return [recording.open_stream(base) for base in streams(label)[:count]]


if {"rect", "recu"} <= set(recorded):
    print("again")
    a, b = streams("rect")[:8], streams("recu")[:8]
    same = sum(digest(x) == digest(y) for x, y in zip(a, b, strict=True))
    gate(
        same == 8,
        f"{same} of 8 streams of the R list are the same bytes in rect and recu",
    )

if {"rect", "recv"} <= set(recorded):
    print("shared")
    shared = sorted(set(taps_of("rect")) & set(taps_of("recv")))
    a, b = opened("rect", 8), opened("recv", 8)
    for tap in shared:
        same = sum(
            columns(x, tap) == columns(y, tap) and x.pos.tolist() == y.pos.tolist()
            for x, y in zip(a, b, strict=True)
        )
        gate(
            same == 8,
            f"tap {tap}: {same} of 8 streams hold the same bytes in rect and recv",
        )

if {"rect", "recf"} <= set(recorded):
    last = taps_of("recf")[-1]
    same = sum(
        columns(x, last) == columns(y, last)
        for x, y in zip(opened("rect", 8), opened("recf", 8), strict=True)
    )
    print(
        f"seen: the final norm's rows (tap {last}) are the same bytes on the gather and the split path in {same} of 8 streams"
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """The relative L2 distance of two rows, over all their taps."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


if "rect" in recorded:
    rows = opened("rect", 8)
    pairs = [
        (rows[i], rows[i + 1]) for i in range(0, 8, 2)
    ]  # each prompt drafted, then plain
    whole = part = 0
    for drafted, plain in pairs:
        at = {int(p): i for i, p in enumerate(plain.pos)}
        both = [
            (i, at[int(p)])
            for i, p in enumerate(drafted.pos)
            if int(p) in at and drafted.kind[i] == 1
        ]
        part += len(both)
        whole += sum(bytes(drafted.st[i]) == bytes(plain.st[j]) for i, j in both)
    print(f"seen: {whole} of {part} reply rows are the same bytes drafted and plain")

    print("feedback")
    again = [recording.open_stream(base) for base in streams("rect")[8:12]]
    gate(len(again) == 4, f"{len(again)} streams of the F list")
    near, off, exact, prompts = [], [], 0, 0
    for (first, _), back in zip(pairs, again, strict=False):
        f0, b0 = (
            first.summary.first_row,
            back.summary.first_row,
        )  # a pass may start at a kept state: rows by position
        lo, hi = max(f0, b0), min(f0 + len(first.tok), b0 + len(back.tok))
        differ = (
            np.flatnonzero(first.tok[lo - f0 : hi - f0] != back.tok[lo - b0 : hi - b0])
            if hi > lo
            else np.zeros(0)
        )
        end = lo + int(differ[0]) if differ.size else max(lo, hi)
        replies_held = int(np.count_nonzero(first.kind == 1))
        print(
            f"  seen: positions {lo} to {end - 1} hold the same tokens in a first stream (from {f0}, prompt"
            f" {first.summary.prompt_len}, {replies_held} reply rows) and its F stream (from {b0}, {len(back.tok)} rows)"
        )
        a, b = (
            first.tapped(lo - f0, end - f0),
            back.tapped(lo - b0, end - b0),
        )  # row i of both is position lo + i
        for i in range(end - lo):
            if first.kind[lo - f0 + i] == 1:
                if i + 1 < end - lo:
                    near.append(distance(a[i], b[i]))
                    off.append(distance(a[i], b[i + 1]))
            else:
                prompts += 1
                exact += bytes(first.st[lo - f0 + i]) == bytes(back.st[lo - b0 + i])
    if near:
        gate(
            statistics.median(near) < 0.5 * statistics.median(off),
            f"{len(near)} reply rows against their prompt-pass twins: relative L2 median {statistics.median(near):.4f}"
            f" (most {max(near):.4f}) at the same position, {statistics.median(off):.4f} one position off",
        )
    else:
        gate(
            False,
            "no reply row has a twin in the F list (the tokens differ from the first)",
        )
    print(
        f"seen: {exact} of {prompts} prompt rows are the same bytes when the prompt is computed again with its reply behind it"
    )

print(f"judge: {fails} failures")
sys.exit(1 if fails else 0)
