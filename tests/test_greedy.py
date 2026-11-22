import json
import struct

import numpy as np
import pytest
import torch

from open_dneo import greedy, recording

from .test_hub import safetensors_blob
from .test_recording import summary, write

# A head over five tokens: each of the first four scores its own coordinate, and
# the fifth a little of every one.
HEAD = torch.tensor(
    [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0], [0.4] * 4]
)
# Tap 9's rows, which the head reads: tokens 1, 2, 4, 1 and 3 score best on them.
FINAL = [[0, 1.0, 0, 0], [0, 0, 1.0, 0], [1.0] * 4, [0, 1.0, 0, 0], [0, 0, 0, 1.0]]
TOKENS = [3, 1, 2, 0, 1]  # a prompt of two, then the reply


def recorded(folder, n=0, taps=(2, 9), positions=(0, 1, 2, 3, 4)):
    """A stream of five rows on disk: tap 2 would score token 0 best at every row."""
    other = [1.0, 0, 0, 0]
    rows = b"".join(
        struct.pack("<4f", *(other if tap == 2 else final))
        for final in FINAL
        for tap in taps
    )
    fields = summary(
        rows=5, prompt_len=2, taps=list(taps), row_bytes=16 * len(taps), dtype="f32"
    )
    return write(
        folder,
        n,
        fields,
        tok=struct.pack("<5I", *TOKENS),
        pos=struct.pack("<5I", *positions),
        kind=bytes([0, 0, 1, 1, 1]),
        st=rows,
    )


@pytest.mark.parametrize("at_once", [1, 2, 256])
def test_the_targets_next_token_is_its_heads_best_score_on_the_final_row(
    tmp_path, at_once
):
    stream = recording.open_stream(recorded(tmp_path))
    written = greedy.nexts(stream, HEAD, 9, at_once=at_once)
    assert written.tolist() == [1, 2, 4, 1, 3]


def test_rows_the_head_cannot_read_are_refused(tmp_path):
    stream = recording.open_stream(recorded(tmp_path))
    with pytest.raises(ValueError, match=r"holds taps \[2, 9\], not 45"):
        greedy.nexts(stream, HEAD, 45)
    with pytest.raises(ValueError, match="rows are 4 wide, the head's 3"):
        greedy.nexts(stream, HEAD[:, :3], 9)
    with pytest.raises(ValueError, match="at_once is 0"):
        greedy.nexts(stream, HEAD, 9, at_once=0)


def test_a_token_is_the_targets_own_when_the_row_before_it_chose_it(tmp_path):
    stream = recording.open_stream(recorded(tmp_path))
    counts = greedy.own(stream, np.array([1, 2, 4, 1, 3]))
    # after the rows: 1 (the prompt's, chosen), 2 (chosen), 0 (4 was), 1 (chosen)
    assert counts == greedy.Own(reply=2, reply_counted=3, prompt=1, prompt_counted=1)
    assert counts.line() == "reply 2 of 3 tokens the target's own, prompt 1 of 1"
    apart = recording.open_stream(recorded(tmp_path / "x", positions=(0, 1, 2, 4, 5)))
    with pytest.raises(ValueError, match="not at consecutive positions"):
        greedy.own(apart, np.array([1, 2, 4, 1, 3]))


def test_a_drafts_label_is_on_the_targets_path_to_the_first_token_it_did_not_choose():
    tokens, written = np.array(TOKENS), np.array([1, 2, 4, 1, 3])
    # the target chose the tokens at rows 1 and 2, not row 3's (0, where it chose 4):
    # an anchor at row 0 drafts rows 1 to 3 with labels on its path, and the label of
    # row 4 follows a token it did not choose
    got = greedy.own_path(tokens, written, np.arange(5), drafts=7)
    assert got.tolist() == [3, 2, 1, 2, 1]
    # a draft's label is the choice after the row before it: none past the last row
    last = greedy.own_path(tokens, written, np.array([4, 3]), drafts=7)
    assert last.tolist() == [1, 2]
    assert greedy.own_path(tokens, written, np.arange(5), drafts=2).tolist() == [
        2, 2, 1, 2, 1
    ]  # fmt: skip
    assert greedy.own_path(tokens, written, np.arange(5), drafts=1).tolist() == [1] * 5


def test_the_second_drafts_on_the_targets_path_are_the_tokens_own_counts(tmp_path):
    stream = recording.open_stream(recorded(tmp_path))
    written = greedy.nexts(stream, HEAD, 9)
    counts = greedy.own(stream, written)
    tokens = np.asarray(stream.tok, np.int64)
    second = greedy.own_path(tokens, written, np.arange(5), drafts=2) - 1
    assert second.sum() == counts.reply + counts.prompt == 3


def test_a_path_is_not_read_from_what_does_not_line_up():
    tokens, written = np.array(TOKENS), np.array([1, 2, 4, 1, 3])
    with pytest.raises(ValueError, match="4 choices for 5 tokens"):
        greedy.own_path(tokens, written[:4], np.arange(5), drafts=7)
    with pytest.raises(ValueError, match="drafts is 0"):
        greedy.own_path(tokens, written, np.arange(5), drafts=0)
    with pytest.raises(ValueError, match=r"no row at \[5, -1\]"):
        greedy.own_path(tokens, written, np.array([0, 5, -1]), drafts=7)


def test_the_final_tap_is_the_text_models_layers(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 45}, "num_hidden_layers": 2})
    )
    assert greedy.final_tap(tmp_path) == 45
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 12}))
    assert greedy.final_tap(tmp_path) == 12


def checkpoint(folder):
    """A checkpoint of nine layers whose head is HEAD."""
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"num_hidden_layers": 9}))
    head = {"lm_head.weight": ("F32", [5, 4], HEAD.numpy().tobytes())}
    (folder / "model.safetensors").write_bytes(safetensors_blob(head))
    return folder


def test_report_counts_every_finished_stream(tmp_path):
    root = tmp_path / "rec"
    recorded(root / "7-0", 0)
    recorded(root / "7-0", 1)
    write(root / "7-0", 2)  # not closed: left alone
    lines, sound = greedy.report(root, checkpoint(tmp_path / "ck"), each=True)
    assert sound
    assert lines == [
        "7-0/0: reply 2 of 3 tokens the target's own, prompt 1 of 1",
        "7-0/1: reply 2 of 3 tokens the target's own, prompt 1 of 1",
        "streams 2, tap 9: reply 4 of 6 tokens the target's own, prompt 2 of 2",
    ]
    assert greedy.report(root, tmp_path / "ck")[0] == lines[2:]


def test_report_names_a_stream_it_cannot_count(tmp_path):
    root = tmp_path / "rec"
    recorded(root, 0)
    recorded(root, 1, taps=(2,))
    lines, sound = greedy.report(root, checkpoint(tmp_path / "ck"))
    assert not sound
    assert lines == [
        "1: the stream holds taps [2], not 9",
        "streams 1, tap 9: reply 2 of 3 tokens the target's own, prompt 1 of 1",
    ]
    empty = tmp_path / "none"
    empty.mkdir()
    lines, sound = greedy.report(empty, tmp_path / "ck")
    assert not sound
    assert lines == [
        "streams 0, tap 9: reply 0 of 0 tokens the target's own, prompt 0 of 0"
    ]


def test_report_saves_each_streams_choices_where_asked(tmp_path):
    root = tmp_path / "rec"
    recorded(root / "7-0", 0)
    recorded(root / "7-0", 1, taps=(2,))  # not counted: nothing saved
    save = tmp_path / "next"
    lines, _ = greedy.report(root, checkpoint(tmp_path / "ck"), save=save)
    assert lines[-1].startswith("streams 1, tap 9")
    saved = np.fromfile(save / "7-0" / "0.next", dtype="<i4")
    assert saved.tolist() == [1, 2, 4, 1, 3]
    assert sorted(p.relative_to(save).as_posix() for p in save.rglob("*")) == [
        "7-0",
        "7-0/0.next",
    ]


@pytest.mark.parametrize("at_once", [1, 2, 256])
def test_the_targets_best_k_and_its_chance_of_the_next_token_come_with_its_choice(
    tmp_path, at_once
):
    stream = recording.open_stream(recorded(tmp_path))
    read = greedy.choices(stream, HEAD, 9, top=2, at_once=at_once)
    assert read.written.tolist() == greedy.nexts(stream, HEAD, 9).tolist()
    finals = torch.tensor(FINAL)
    logprobs = torch.log_softmax(finals @ HEAD.T, -1)
    want, ids = logprobs.topk(2, -1)
    assert read.best.tolist() == ids.tolist()
    np.testing.assert_allclose(read.best_logprobs, want.numpy(), rtol=1e-6)
    # after each row, the target's log probability of the stream's next token;
    # none after the last row
    after = torch.tensor(TOKENS[1:])
    np.testing.assert_allclose(
        read.next_logprob[:-1],
        logprobs[:-1].gather(-1, after[:, None])[:, 0].numpy(),
        rtol=1e-6,
    )
    assert np.isnan(read.next_logprob[-1])
    assert read.best.shape == (5, 2) and greedy.choices(
        stream, HEAD, 9, top=0
    ).best.shape == (5, 0)


def test_report_saves_the_best_k_and_the_next_tokens_chance_where_asked(tmp_path):
    root = tmp_path / "rec"
    recorded(root / "7-0", 0)
    save = tmp_path / "next"
    greedy.report(root, checkpoint(tmp_path / "ck"), save=save, top=2)
    stream = recording.open_stream(root / "7-0" / "0")
    read = greedy.choices(stream, HEAD, 9, top=2)
    ids, logprobs = greedy.read_best(save / "7-0" / "0.best", rows=5, top=2)
    assert ids.tolist() == read.best.tolist()
    np.testing.assert_array_equal(logprobs, read.best_logprobs)
    nextlp = np.fromfile(save / "7-0" / "0.nextlp", dtype="<f4")
    np.testing.assert_array_equal(nextlp, read.next_logprob)
    # each row's k ids, then their k log probabilities: the recorder's .top layout
    raw = np.fromfile(save / "7-0" / "0.best", dtype="<u4").reshape(5, 4)
    assert raw[:, :2].tolist() == read.best.tolist()
    with pytest.raises(ValueError, match="holds 20 values, not 5 rows of 3"):
        greedy.read_best(save / "7-0" / "0.best", rows=5, top=3)
