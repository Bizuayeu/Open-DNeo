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
