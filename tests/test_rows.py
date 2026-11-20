import json

import numpy as np
import pytest

from open_dneo import logged, rows, score

from .test_score import drawn, folders, on_disk, stream


def test_proposals_at_given_anchors_read_prompt_rows_as_they_read_reply_rows(tmp_path):
    _, _, (model, target, reading) = folders(tmp_path)
    values, tokens = drawn(12)
    replied = stream(values, tokens, first=0, prompt=4)
    prompted = stream(values, tokens, first=0, prompt=12)  # the same rows, all prompt
    by_reply = score.proposals(model, target, replied, reading)
    by_prompt = score.proposals(
        model, target, prompted, reading, anchors=by_reply.positions
    )
    assert np.array_equal(by_prompt.positions, by_reply.positions)
    assert np.array_equal(by_prompt.tokens, by_reply.tokens)
    assert np.array_equal(by_prompt.drafts, by_reply.drafts)
    assert np.allclose(by_prompt.chances, by_reply.chances)


def test_proposals_can_leave_out_the_rows_before_a_given_row(tmp_path):
    _, _, (model, target, reading) = folders(tmp_path)
    values, tokens = drawn(12)
    whole = stream(values, tokens, first=0, prompt=4)
    later = stream(values[3:], tokens[3:], first=3, prompt=4)
    clipped = score.proposals(model, target, whole, reading, context_from=3)
    assert np.array_equal(
        clipped.drafts, score.proposals(model, target, later, reading).drafts
    )
    assert clipped.first_row == 3


def test_proposals_refuse_an_anchor_the_stream_has_no_row_for(tmp_path):
    _, _, (model, target, reading) = folders(tmp_path)
    values, tokens = drawn(12)
    held = stream(values[2:], tokens[2:], first=2, prompt=12)
    with pytest.raises(ValueError, match=r"no row at positions \[1, 12\]"):
        score.proposals(model, target, held, reading, anchors=np.array([1, 5, 12]))


def written(path, lines):
    path.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    return path


def asked(id_, klass, prompt, reply):
    return {
        "id": id_,
        "class": klass,
        "max_tokens": 64,
        "prompt_ids": [int(t) for t in prompt],
        "reply_ids": [int(t) for t in reply],
        "completion_tokens": len(reply),
        "speculative": {"rounds": 3},
        "status": 200,
    }


def test_report_sets_reply_rows_beside_prompt_rows_of_the_same_tokens(tmp_path):
    drafted, checkpoint_folder, (model, target, reading) = folders(tmp_path)
    values, tokens = drawn(13)
    tok = tokens.astype(int)
    root = tmp_path / "rec"
    # a reply's stream (its last token has no row), and the prompt's stream that
    # sent its prompt and reply again: the same rows here, so every draft agrees
    on_disk(root, values[:12], tokens[:12], first=0, prompt=4, n=0)
    on_disk(root, values, tokens, first=0, prompt=13, n=1)
    # a second reply whose tokens were never sent again
    on_disk(root, values[:12], tokens[:12], first=0, prompt=5, n=2)
    run = written(
        tmp_path / "run.jsonl",
        [
            asked("code-00", "code", tok[:4], tok[4:13]),
            asked("chat-00", "chat", tok[:5], [*tok[5:12], 9]),
        ],
    )
    lines, sound = rows.report(root, drafted, checkpoint_folder, run)
    replied = stream(values[:12], tokens[:12], first=0, prompt=4)
    found = score.proposals(model, target, replied, reading)
    alone = score.steps(tok[4:13].tolist(), found)
    taken, anchors = sum(alone), len(found.positions)
    assert sound
    assert lines[1:] == [
        "run: 2 requests, 2 with their reply's stream, 1 with a prompt's stream of"
        " the same tokens",
        "chat-00: no prompt's stream holds its prompt and reply",
        f"code: 1 replies, {anchors} anchors with both rows, 0 without; the drafter"
        f" alone over reply rows {taken} tokens in {len(alone)} rounds,"
        f" {taken / len(alone):.3f} a round; over prompt rows {taken} in"
        f" {len(alone)}, {taken / len(alone):.3f} a round; the first draft the same"
        f" at {anchors} anchors (1.000), every draft at {anchors}",
    ]


def test_report_reads_both_contexts_from_the_later_first_row(tmp_path, monkeypatch):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    values, tokens = drawn(13)
    tok = tokens.astype(int)
    root = tmp_path / "rec"
    on_disk(root, values[:12], tokens[:12], first=0, prompt=4, n=0)
    # the prompt's stream resumed from kept state at row 2
    on_disk(root, values[2:], tokens[2:], first=2, prompt=13, n=1)
    run = written(tmp_path / "run.jsonl", [asked("code-00", "code", tok[:4], tok[4:])])
    seen, made = [], score.proposals
    monkeypatch.setattr(
        score,
        "proposals",
        lambda *given, **named: (
            seen.append(named.get("context_from")) or made(*given, **named)
        ),
    )
    _, sound = rows.report(root, drafted, checkpoint_folder, run)
    assert sound and seen == [2, 2]


def test_agreement_counts_two_proposals_at_the_same_anchors():
    def made(drafts):
        at, tokens = np.array([4, 5]), np.array([1, 2])
        return score.Proposals(0, at, tokens, np.array(drafts), np.zeros((2, 2)))

    got = rows.agreement(made([[3, 4], [5, 6]]), made([[3, 4], [5, 0]]))
    assert got == logged.Agreement(rounds=2, first=2, leading=3, whole=1)


def test_report_walks_each_side_over_its_own_rows(tmp_path, monkeypatch):
    drafted, checkpoint_folder, (model, target, reading) = folders(tmp_path)
    values, tokens = drawn(13)
    tok = tokens.astype(int)
    shifted = values + np.float32(0.75) * values[::-1]  # the prompt pass's rows differ
    root = tmp_path / "rec"
    on_disk(root, values[:12], tokens[:12], first=0, prompt=4, n=0)
    on_disk(root, shifted, tokens, first=0, prompt=13, n=1)
    run = written(tmp_path / "run.jsonl", [asked("code-00", "code", tok[:4], tok[4:])])
    walked, steps = [], score.steps
    monkeypatch.setattr(
        score, "steps", lambda *given: walked.append(given[1]) or steps(*given)
    )
    lines, sound = rows.report(root, drafted, checkpoint_folder, run)
    monkeypatch.setattr(score, "steps", steps)
    mine = score.proposals(
        model, target, stream(values[:12], tokens[:12], first=0, prompt=4), reading
    )
    other = score.proposals(
        model,
        target,
        stream(shifted, tokens, first=0, prompt=13),
        reading,
        anchors=mine.positions,
    )
    assert not np.array_equal(mine.drafts, other.drafts)
    # each side walked over its own proposals
    assert [
        np.array_equal(w.drafts, p.drafts) for w, p in zip(walked, (mine, other))
    ] == [True, True]
    alone, prompt = (score.steps(tok[4:].tolist(), p) for p in (mine, other))
    agreed = rows.agreement(mine, other)
    assert sound
    assert lines[-1] == (
        f"code: 1 replies, {agreed.rounds} anchors with both rows, 0 without; the"
        f" drafter alone over reply rows {sum(alone)} tokens in {len(alone)} rounds,"
        f" {sum(alone) / len(alone):.3f} a round; over prompt rows {sum(prompt)} in"
        f" {len(prompt)}, {sum(prompt) / len(prompt):.3f} a round; the first draft"
        f" the same at {agreed.first} anchors ({agreed.first / agreed.rounds:.3f}),"
        f" every draft at {agreed.whole}"
    )


def test_report_leaves_out_a_reply_whose_prompt_stream_resumed_past_it(tmp_path):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    values, tokens = drawn(13)
    tok = tokens.astype(int)
    root = tmp_path / "rec"
    on_disk(root, values[:12], tokens[:12], first=0, prompt=4, n=0)
    # resumed at row 4: the reply's first anchor, at 4, has no context to share
    on_disk(root, values[4:], tokens[4:], first=4, prompt=13, n=1)
    run = written(tmp_path / "run.jsonl", [asked("code-00", "code", tok[:4], tok[4:])])
    lines, sound = rows.report(root, drafted, checkpoint_folder, run)
    assert not sound
    assert lines[-1] == "code: 1 replies, no anchor with both rows, 8 without"


def test_report_gives_each_prompt_stream_to_one_reply(tmp_path):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    values, tokens = drawn(13)
    tok = tokens.astype(int)
    root = tmp_path / "rec"
    # the same request twice, each with its reply's stream, and one prompt's stream
    on_disk(root, values[:12], tokens[:12], first=0, prompt=4, n=0)
    on_disk(root, values[:12], tokens[:12], first=0, prompt=4, n=1)
    on_disk(root, values, tokens, first=0, prompt=13, n=2)
    one = asked("code-00", "code", tok[:4], tok[4:])
    run = written(tmp_path / "run.jsonl", [one, one | {"id": "code-01"}])
    lines, _ = rows.report(root, drafted, checkpoint_folder, run)
    assert lines[1:3] == [
        "run: 2 requests, 2 with their reply's stream, 1 with a prompt's stream of"
        " the same tokens",
        "code-01: no prompt's stream holds its prompt and reply",
    ]
