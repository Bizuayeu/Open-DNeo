"""twins.py FOLDER [--worst N]: how far a reply's rows are from the same positions computed as a prompt's.

Seen, not judged. FOLDER is a launch's recording whose first 8 streams are the R list (each prompt drafted, then
plain) and whose next 4 are the F list: each drafted stream's tokens sent again as one prompt (launch rect of
README.md). A reply row was computed by a verify pass over the cache, its twin by a prompt pass. judge.py gates their
distance over all taps at once; this lays it out tap by tap, since a drafter trained on prompt-pass rows reads reply
rows when it is served.

Per tap, over every reply row that has a twin: the relative L2 distance |reply - twin| / |reply| (median, the 90th
percentile, the most), the same against the twin one position on, and the cosine. Then the N rows furthest from their
twins, with their place in the reply. Last line: "twins: R reply rows with twins, T taps".
"""

import argparse
from pathlib import Path

import numpy as np

from open_dneo import recording

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--worst", type=int, default=8)
args = p.parse_args()

bases = recording.find(Path(args.folder))
first = [recording.open_stream(base) for base in bases[0:8:2]]
back = [recording.open_stream(base) for base in bases[8:12]]
taps = first[0].summary.taps
near, off, cosine, where = [], [], [], []
for n, (reply, twin) in enumerate(zip(first, back, strict=True)):
    f0, b0 = reply.summary.first_row, twin.summary.first_row
    lo, hi = max(f0, b0), min(f0 + len(reply.tok), b0 + len(twin.tok))
    differ = np.flatnonzero(reply.tok[lo - f0 : hi - f0] != twin.tok[lo - b0 : hi - b0])
    end = lo + int(differ[0]) if differ.size else hi
    a, b = (
        reply.tapped(lo - f0, end - f0),
        twin.tapped(lo - b0, end - b0),
    )  # row i of both is position lo + i
    for i in range(end - lo - 1):
        if reply.kind[lo - f0 + i] != 1:
            continue
        size = np.linalg.norm(a[i], axis=1)
        near.append(np.linalg.norm(a[i] - b[i], axis=1) / size)
        off.append(np.linalg.norm(a[i] - b[i + 1], axis=1) / size)
        cosine.append((a[i] * b[i]).sum(axis=1) / (size * np.linalg.norm(b[i], axis=1)))
        where.append((n, lo + i, lo + i - reply.summary.prompt_len))
near, off, cosine = np.array(near), np.array(off), np.array(cosine)

print(
    f"{len(near)} reply rows with twins, from {len(first)} replies; taps {list(taps)}"
)
print("tap   median    90th     most   one on (median)   cosine (median, least)")
for t, tap in enumerate(taps):
    print(
        f"{tap:3d}  {np.median(near[:, t]):.4f}  {np.percentile(near[:, t], 90):.4f}  {near[:, t].max():.4f}"
        f"   {np.median(off[:, t]):.4f}            {np.median(cosine[:, t]):.4f}  {cosine[:, t].min():.4f}"
    )
print(f"the {args.worst} rows furthest from their twins, by the most over their taps:")
for row in np.argsort(-near.max(axis=1))[: args.worst]:
    n, position, j = where[row]
    print(
        f"  reply {n} position {position} (reply row {j}): "
        + "  ".join(f"{tap}: {near[row, t]:.4f}" for t, tap in enumerate(taps))
    )
print(
    "by a row's place in the reply (median over its replies, the last tap; then every tap's median):"
)
for j in sorted({w[2] for w in where}):
    rows = [i for i, w in enumerate(where) if w[2] == j]
    if j < 4 or j % 8 == 0:
        print(
            f"  reply row {j:2d}: {np.median(near[rows, -1]):.4f}   "
            + "  ".join(f"{np.median(near[rows, t]):.4f}" for t in range(len(taps)))
        )
print(f"twins: {len(near)} reply rows with twins, {len(taps)} taps")
