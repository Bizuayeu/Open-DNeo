"""role_greedy.py TOK_ROOT NEXT_ROOT STREAMS_TSV RUN_JSONL [OUT_PARTS_TSV]: greedy's agreement by part and by role.

TOK_ROOT holds a recording's .tok files, NEXT_ROOT the .next files `open-dneo greedy --save` wrote for it,
STREAMS_TSV a line a stream (name, rows, first row, prompt length, sha256 of its .tok bytes), and RUN_JSONL the
client's line a prompt (its part, its id and its prompt's token ids). Needs numpy.

Streams are paired with the run's items by the sha256 of their .tok bytes against that of the item's prompt ids as
little-endian uint32 (the streams start at row 0: no prompt resumed). A position p counts whether the target's choice
at p (.next) is the stream's token at p + 1. That token's role is the role GLM's template last named before it:
<|system|> 154826, <|user|> 154827, <|assistant|> 154828, <|observation|> 154829; "template" for those markers and the
other special tokens (154820 and up), "text" where no marker came before (a plain prompt).

Prints per part: positions and agreement overall and by role; for chat-human and chat-glm, the assistant role over the
prompts both parts hold (the id after the part's name); for agent-swe, the assistant positions an item. Writes
OUT_PARTS_TSV (stream name, a tab, its part) for `open-dneo probe --parts`.
"""

import hashlib
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROLES = {154826: "system", 154827: "user", 154828: "assistant", 154829: "observation"}
SPECIAL = 154820


def roles_of(tok):
    """The role of each token: the last marker before it, or template/text."""
    out, now = [], "text"
    for t in tok.tolist():
        if t in ROLES:
            out.append("template")
            now = ROLES[t]
        elif t >= SPECIAL:
            out.append("template")
        else:
            out.append(now)
    return out


def main(tok_root, next_root, streams_tsv, run_jsonl, out_parts=None):
    tok_root, next_root = pathlib.Path(tok_root), pathlib.Path(next_root)
    by_sha = {}
    for line in pathlib.Path(run_jsonl).read_text(encoding="utf-8").split("\n"):
        if line:
            item = json.loads(line)
            if item.get("status") == 200:
                ids = np.asarray(item["prompt_ids"], "<u4").tobytes()
                by_sha[hashlib.sha256(ids).hexdigest()] = item
    tally = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    chat = defaultdict(
        dict
    )  # prompt id -> part -> [hits, counted] over assistant positions
    agent_rows = []
    named, unpaired = [], 0
    for line in pathlib.Path(streams_tsv).read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        name, rows, first, _plen, sha = line.split("\t")
        item = by_sha.get(sha)
        if item is None or int(first) != 0:
            unpaired += 1
            continue
        part = item["part"]
        named.append((name, part))
        tok = np.fromfile(tok_root / f"{name}.tok", dtype="<u4")
        nxt = np.fromfile(next_root / f"{name}.next", dtype="<i4")
        if len(tok) != int(rows) or len(nxt) != int(rows):
            raise SystemExit(
                f"{name}: {len(tok)} tokens and {len(nxt)} choices for {rows} rows"
            )
        hit = nxt[:-1] == tok[1:].astype(np.int64)
        role = roles_of(tok)[1:]
        for h, r in zip(hit.tolist(), role):
            tally[part][r][0] += h
            tally[part][r][1] += 1
            tally[part]["all"][0] += h
            tally[part]["all"][1] += 1
        if part in ("chat-human", "chat-glm"):
            key = item["id"].split("-", 2)[2]
            a = [h for h, r in zip(hit.tolist(), role) if r == "assistant"]
            chat[key][part] = [sum(a), len(a)]
        if part == "agent-swe":
            agent_rows.append(sum(1 for r in role if r == "assistant"))
    print(f"streams paired {len(named)}, unpaired {unpaired}")
    for part in sorted(tally):
        t = tally[part]
        cells = ", ".join(
            f"{r} {h / n:.3f} ({n})" for r, (h, n) in sorted(t.items()) if r != "all"
        )
        h, n = t["all"]
        print(f"{part}: {h / n:.3f} of {n} positions; {cells}")
    both = [k for k, v in chat.items() if len(v) == 2]
    for part in ("chat-human", "chat-glm"):
        h = sum(chat[k][part][0] for k in both)
        n = sum(chat[k][part][1] for k in both)
        if n:
            print(
                f"{part} over the {len(both)} prompts both hold, assistant positions: {h / n:.3f} ({n})"
            )
    if agent_rows:
        print(
            f"agent-swe assistant positions an item: {agent_rows} (of {len(agent_rows)} items)"
        )
    if out_parts:
        pathlib.Path(out_parts).write_text(
            "".join(f"{n}\t{p}\n" for n, p in named), encoding="utf-8"
        )


if __name__ == "__main__":
    main(*sys.argv[1:6])
