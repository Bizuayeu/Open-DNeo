import json

import numpy as np
import pytest

from open_dneo import recording, runs


def line(id_, prompt, reply, *, klass="code", rounds=3, status=200, max_tokens=8):
    return {
        "id": id_,
        "class": klass,
        "max_tokens": max_tokens,
        "prompt_ids": prompt,
        "reply_ids": reply,
        "completion_tokens": len(reply),
        "speculative": {"rounds": rounds, "drafted": 9, "accepted": 4},
        "status": status,
    }


def written(tmp_path, lines):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return path


def stream(prompt, reply_rows, first=0):
    """A recorded stream's tokens: the prompt's rows from `first`, then reply rows."""
    tokens = [*prompt[first:], *reply_rows]
    rows = len(tokens)
    positions = first + np.arange(rows, dtype=np.uint32)
    summary = recording.Summary(rows, first, len(prompt), (45,), 4, "f32", 0, None)
    return recording.Stream(
        summary,
        tok=np.asarray(tokens, np.uint32),
        pos=positions,
        kind=(positions >= len(prompt)).astype(np.uint8),
        st=np.zeros((rows, 4), np.uint8),
        top_ids=np.empty((0, 0), np.uint32),
        top_logprobs=np.empty((0, 0), np.float32),
    )


def test_read_keeps_the_answered_requests(tmp_path):
    path = written(
        tmp_path,
        [line("a", [1, 2], [5, 6, 7]), line("b", [1, 3], [], status=400)],
    )
    (got,) = runs.read(path)
    assert got == runs.Request("a", "code", 8, (1, 2), (5, 6, 7), 3)


def test_read_refuses_a_reply_whose_ids_are_not_its_tokens(tmp_path):
    path = written(tmp_path, [line("a", [1], [2, 3]) | {"completion_tokens": 3}])
    with pytest.raises(ValueError, match="a: 2 reply ids, and 3 completion tokens"):
        runs.read(path)


def test_read_splits_lines_at_newlines_alone(tmp_path):
    # a request's text may hold U+2028, at which str.splitlines() would also split
    entry = line("a", [1], [2, 3]) | {"text": "one two"}
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    assert [r.id for r in runs.read(path)] == ["a"]


def test_pair_finds_each_requests_stream_in_the_order_they_opened():
    first = runs.Request("a", "code", 8, (1, 2, 3), (5, 6, 7), 2)
    again = runs.Request("b", "code", 8, (1, 2, 3), (5, 6, 7), 2)
    other = runs.Request("c", "chat", 8, (4, 2, 3), (9, 9), 1)
    streams = [
        ("s0", stream([4, 2, 3], [9])),
        ("s1", stream([1, 2, 3], [5, 6])),
        ("s2", stream([1, 2, 3], [5, 6], first=2)),  # the same prompt again, resumed
    ]
    assert runs.pair([first, again, other], streams) == [
        (first, "s1"),
        (again, "s2"),
        (other, "s0"),
    ]


def test_pair_leaves_a_request_without_its_stream_unpaired():
    request = runs.Request("a", "code", 8, (1, 2, 3), (5, 6, 7), 2)
    streams = [
        ("longer prompt", stream([1, 2, 3, 4], [5])),
        ("other prompt", stream([9, 2, 3], [5])),
        ("other reply", stream([1, 2, 3], [5, 8])),
        ("reply rows past the reply", stream([1, 2, 3], [5, 6, 7, 8])),
    ]
    assert runs.pair([request], streams) == [(request, None)]


def test_served_counts_a_replys_tokens_less_its_first_over_its_rounds():
    requests = [
        runs.Request("a", "code", 8, (1,), (5, 6, 7, 8), 2),
        runs.Request("b", "code", 8, (1,), (5, 6), 1),
    ]
    assert runs.served(requests) == (4, 3)
    with pytest.raises(ValueError, match="b: serving gave no rounds"):
        runs.served([*requests[:1], runs.Request("b", "code", 8, (1,), (5,), None)])
