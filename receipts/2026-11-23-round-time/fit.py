"""G2's time of a round: DFlash2's round as served, fitted within each request.

    python fit.py runs.tsv > fit.txt

runs.tsv holds a row for each request in each launch: the launch, the request's id and class, the cut, its status,
its cached tokens, the sha256 of its reply's token ids, its seconds, and the rounds and drafts verified that serving
counted for it. The launch named evd2 is the reference: the open half served at the default cut. A launch counts when
every request of evd2 is there with status 200, no cached tokens and the same reply tokens; the lines say why one
does not. Over the launches that count, for request i and launch c:

    seconds_ic = s_i + a * rounds_ic + b * drafted_ic

s_i is the request's own time (its prompt pass and the rest that the cut does not move). The fit is least squares on
the values less each request's mean over the launches, per class and over all, printed in ms with the within R^2.
Then each counted launch's drafts a round and the round's ms by the fit over all, and how the two launches at cut
0.7 differ. numpy only.
"""

import csv
import statistics
import sys

import numpy as np

CLASSES = ("code", "agent", "chat", "ja", "prose")
REFERENCE = "evd2"
DRIFT = ("c70a", "c70b")  # the two launches at cut 0.7, in the order they ran


def read(path):
    """{launch: {id: row}}, launches in the order they first appear."""
    launches = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            launches.setdefault(row["launch"], {})[row["id"]] = row
    return launches


def counts(ref, run):
    """Why a launch does not count (an empty list when it does)."""
    why = []
    missing = [i for i in ref if i not in run]
    if missing:
        why.append(f"{len(missing)} requests missing (first {missing[0]})")
    bad = [i for i in ref if i in run and run[i]["status"] != "200"]
    if bad:
        why.append(f"{len(bad)} not status 200 (first {bad[0]})")
    cached = [i for i in ref if i in run and run[i]["cached_tokens"] != "0"]
    if cached:
        why.append(f"{len(cached)} with cached tokens (first {cached[0]})")
    other = [
        i for i in ref if i in run and run[i]["reply_sha256"] != ref[i]["reply_sha256"]
    ]
    if other:
        why.append(
            f"{len(other)} replies not the reference's tokens (first {other[0]})"
        )
    return why


def fit(ids, runs):
    """a and b in ms, the within R^2 and the requests, over ids and the counted runs."""
    ys, xs = [], []
    for i in ids:
        sec = np.array([float(r[i]["seconds"]) for r in runs])
        rnd = np.array([float(r[i]["rounds"]) for r in runs])
        dft = np.array([float(r[i]["drafted"]) for r in runs])
        ys.append(sec - sec.mean())
        xs.append(np.stack([rnd - rnd.mean(), dft - dft.mean()], axis=1))
    y = np.concatenate(ys)
    x = np.concatenate(xs)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    rss = float(((y - x @ coef) ** 2).sum())
    tss = float((y**2).sum())
    return (
        coef[0] * 1000,
        coef[1] * 1000,
        (1 - rss / tss) if tss > 0 else float("nan"),
        len(ids),
    )


def main(argv):
    launches = read(argv[0])
    ref = launches.pop(REFERENCE)
    good = {}
    for label, run in launches.items():
        why = counts(ref, run)
        print(
            f"{label}: {len(run)} requests; "
            + ("counts" if not why else "does not count: " + "; ".join(why))
        )
        if not why:
            good[label] = run
    if len(good) < 2:
        print("fit: fewer than 2 launches count: no fit")
        return 1
    runs = list(good.values())
    print(
        f"fit over {len(runs)} launches ({', '.join(good)}): seconds = s_i + a rounds + b drafted, within each request"
    )
    fits = {}
    for klass in CLASSES + ("all",):
        ids = [i for i in ref if klass == "all" or ref[i]["class"] == klass]
        a, b, r2, n = fit(ids, runs)
        fits[klass] = (a, b)
        print(
            f"  {klass}: a {a:.2f} ms, b {b:.2f} ms a draft, within R^2 {r2:.4f}, {n} requests"
        )
    for label, run in good.items():
        rounds = sum(int(run[i]["rounds"]) for i in ref)
        drafted = sum(int(run[i]["drafted"]) for i in ref)
        k = drafted / rounds
        a, b = fits["all"]
        print(
            f"  {label}: {k:.3f} drafts a round, {a + b * k:.1f} ms a round by the fit over all"
        )
    x, y = launches[DRIFT[0]], launches[DRIFT[1]]
    d = [float(y[i]["seconds"]) - float(x[i]["seconds"]) for i in ref]
    same = sum(
        1
        for i in ref
        if (x[i]["rounds"], x[i]["drafted"]) == (y[i]["rounds"], y[i]["drafted"])
    )
    print(
        f"drift {DRIFT[1]} - {DRIFT[0]}: median {statistics.median(d) * 1000:.1f} ms a request over {len(d)}; "
        f"rounds and drafted the same in {same}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
