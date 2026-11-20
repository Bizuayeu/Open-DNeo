import json
import struct

import numpy as np
import pytest

from open_dneo import probe

from .test_hub import safetensors_blob
from .test_recording import summary, write

TAPS = (2, 5, 9)  # 9 is the final tap of a checkpoint of nine layers
WIDTH = 4
VOCAB = 6


def checkpoint(folder, head):
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"num_hidden_layers": 9}))
    blob = {"lm_head.weight": ("F32", list(head.shape), head.astype("<f4").tobytes())}
    (folder / "model.safetensors").write_bytes(safetensors_blob(blob))
    return folder


def stream(folder, n, rng, rows=120):
    """Final rows that are one of the head's rows plus noise; tap 5 holds the final
    row one position ahead, tap 2 noise. Returns the target's choice at each row."""
    head = HEAD
    choice = rng.integers(0, VOCAB, rows + 1)
    final = head[choice] + rng.normal(0, 0.05, (rows + 1, WIDTH))
    noise = rng.normal(0, 1, (rows, WIDTH))
    cells = np.stack([noise, final[1:], final[:rows]], axis=1).astype("<f4")
    fields = summary(
        rows=rows,
        prompt_len=rows,
        taps=list(TAPS),
        row_bytes=4 * WIDTH * len(TAPS),
        dtype="f32",
    )
    write(
        folder,
        n,
        fields,
        tok=struct.pack(f"<{rows}I", *range(rows)),
        pos=struct.pack(f"<{rows}I", *range(rows)),
        kind=bytes(rows),
        st=cells.tobytes(),
    )
    return choice[:rows]


HEAD = np.array(
    [
        [3.0, 0, 0, 0],
        [0, 3.0, 0, 0],
        [0, 0, 3.0, 0],
        [0, 0, 0, 3.0],
        [-3.0, 0, 0, 0],
        [0, -3.0, 0, 0],
    ]
)


@pytest.fixture
def recorded(tmp_path):
    rng = np.random.default_rng(7)
    root, nexts = tmp_path / "rec" / "9-0", tmp_path / "next" / "9-0"
    nexts.mkdir(parents=True)
    for n in range(4):
        stream(root, n, rng).astype("<i4").tofile(nexts / f"{n}.next")
    return tmp_path / "rec", tmp_path / "next", checkpoint(tmp_path / "ck", HEAD)


def score(lines, name):
    """The fraction a report line gives for `name` (e.g. "tap 5 k 1")."""
    line = next(x for x in lines if x.startswith(name + ":"))
    return float(line.split(":")[1].split()[0])


def test_the_tap_that_holds_the_row_ahead_scores_and_noise_falls_to_the_floor(recorded):
    root, nexts, ck = recorded
    lines, sound = probe.report(root, ck, nexts, ks=(1,), held_every=2, train_every=1)
    assert sound
    assert score(lines, "tap 5 k 1") > 0.95
    assert score(lines, "tap 2 k 1") < 0.4
    assert score(lines, "the floor k 1") < 0.4


def test_the_final_tap_at_k_0_is_the_saved_choice(recorded):
    root, nexts, ck = recorded
    lines, _ = probe.report(root, ck, nexts, ks=(0,), held_every=2, train_every=1)
    assert score(lines, "tap 9 k 0") == 1.0


def test_a_set_of_taps_is_fitted_on_their_rows_side_by_side(recorded):
    root, nexts, ck = recorded
    lines, _ = probe.report(
        root, ck, nexts, ks=(1,), held_every=2, train_every=1, sets=((2, 5),)
    )
    assert score(lines, "taps 2+5 k 1") > 0.95


def test_parts_are_scored_apart(recorded, tmp_path):
    root, nexts, ck = recorded
    parts = tmp_path / "parts.tsv"
    parts.write_text("9-0/0\ta\n9-0/1\ta\n9-0/2\tb\n9-0/3\tb\n")
    lines, _ = probe.report(
        root, ck, nexts, ks=(1,), held_every=2, train_every=1, parts=parts
    )
    line = next(x for x in lines if x.startswith("tap 5 k 1:"))
    assert " a " in line and " b " in line


def test_a_stream_without_its_choices_is_refused(recorded):
    root, nexts, ck = recorded
    (nexts / "9-0" / "3.next").unlink()
    with pytest.raises(ValueError, match="no saved choices for 9-0/3"):
        probe.report(root, ck, nexts, ks=(1,), held_every=2, train_every=1)


def test_the_report_names_the_ridge_and_the_split(recorded):
    root, nexts, ck = recorded
    lines, _ = probe.report(
        root, ck, nexts, ks=(1,), held_every=2, train_every=1, ridge=0.01
    )
    head = lines[0]
    assert "4 streams, 2 held out" in head and "ridge 0.01" in head
