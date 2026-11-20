import json
import struct

import numpy as np
import pytest

from open_dneo import recording

# One row of two taps, two bf16 values each: 1.0 and -2.0, then 0.5 and 2.0.
ROW = struct.pack("<4H", 0x3F80, 0xC000, 0x3F00, 0x4000)


def write(folder, n, summary=None, tok=b"", pos=b"", kind=b"", st=b"", top=None):
    """Stream `n`'s files in `folder`, byte for byte; no .json without a summary."""
    folder.mkdir(parents=True, exist_ok=True)
    for ext, data in ((".tok", tok), (".pos", pos), (".kind", kind), (".st", st)):
        (folder / f"{n}{ext}").write_bytes(data)
    if top is not None:
        (folder / f"{n}.top").write_bytes(top)
    if summary is not None:
        (folder / f"{n}.json").write_text(json.dumps(summary))
    return folder / str(n)


def summary(**fields):
    """A .json's fields: a stream with no rows, unless `fields` say otherwise."""
    empty = {
        "rows": 0,
        "first_row": 0,
        "prompt_len": 0,
        "taps": [5, 45],
        "row_bytes": 0,
        "dtype": "",
        "top_k": 0,
        "gap": None,
    }
    return empty | fields


def three(**fields):
    """The summary of `three_rows`."""
    return (
        summary(rows=3, first_row=2, prompt_len=4, row_bytes=8, dtype="bf16") | fields
    )


def three_rows(folder, n=0, **changes):
    """Rows at positions 2, 3 and 4 of a 4-token prompt: two prompt rows, one reply."""
    files = {
        "summary": three(),
        "tok": struct.pack("<3I", 11, 12, 13),
        "pos": struct.pack("<3I", 2, 3, 4),
        "kind": bytes([0, 0, 1]),
        "st": 3 * ROW,
    }
    return write(folder, n, **(files | changes))


def test_a_summary_as_the_recorder_writes_it():
    text = (
        '{"rows":3,"first_row":2,"prompt_len":4,"taps":[5,45],"row_bytes":8,'
        '"dtype":"bf16","top_k":0,"gap":null}'
    )
    assert recording.parse_summary(text) == recording.Summary(
        rows=3,
        first_row=2,
        prompt_len=4,
        taps=(5, 45),
        row_bytes=8,
        dtype="bf16",
        top_k=0,
        gap=None,
    )


def test_a_summary_names_the_field_it_cannot_read():
    fields = three()
    wrong = (
        ("rows", -1),
        ("first_row", 1.5),
        ("prompt_len", True),
        ("taps", [5, "x"]),
        ("row_bytes", None),
        ("dtype", 2),
        ("top_k", "2"),
        ("gap", 7),
    )
    for name, value in wrong:
        with pytest.raises(ValueError, match=f'"{name}"'):
            recording.parse_summary(json.dumps(fields | {name: value}))
    without_taps = {name: value for name, value in fields.items() if name != "taps"}
    with pytest.raises(ValueError, match='"taps"'):
        recording.parse_summary(json.dumps(without_taps))
    with pytest.raises(ValueError, match="not JSON"):
        recording.parse_summary("{")
    with pytest.raises(ValueError, match="not a JSON object"):
        recording.parse_summary("[1]")


def test_a_summary_may_leave_gap_out_and_carry_fields_unknown_here():
    fields = three() | {"later": 1}
    del fields["gap"]
    assert recording.parse_summary(json.dumps(fields)).gap is None


def test_a_stream_is_read_row_by_row(tmp_path):
    stream = recording.open_stream(three_rows(tmp_path / "7-0"))
    assert stream.tok.tolist() == [11, 12, 13]
    assert stream.pos.tolist() == [2, 3, 4]
    assert stream.kind.tolist() == [0, 0, 1]
    tapped = stream.tapped(1, 3)
    assert tapped.dtype == np.float32
    assert tapped.tolist() == [[[1.0, -2.0], [0.5, 2.0]]] * 2


def test_a_stream_keeps_one_file_open_for_its_rows_alone(tmp_path):
    # a run opens hundreds of streams at once: tokens, positions and kinds are
    # read whole, and only the tapped rows stay mapped (one file a stream)
    stream = recording.open_stream(three_rows(tmp_path / "7-0"))
    assert not any(
        isinstance(a, np.memmap) for a in (stream.tok, stream.pos, stream.kind)
    )
    assert isinstance(stream.st, np.memmap)


def test_rows_are_in_order_when_each_is_one_position_past_the_last(tmp_path):
    assert recording.open_stream(three_rows(tmp_path / "a")).in_order()
    skipped = three_rows(tmp_path / "b", pos=struct.pack("<3I", 2, 4, 5))
    assert not recording.open_stream(skipped).in_order()
    late = three_rows(tmp_path / "c", pos=struct.pack("<3I", 3, 4, 5))
    assert not recording.open_stream(late).in_order()  # first_row says 2


@pytest.mark.parametrize(("dtype", "code"), [("f16", "<2e"), ("f32", "<2f")])
def test_f16_and_f32_rows_are_read_too(tmp_path, dtype, code):
    raw = struct.pack(code, 1.5, -3.0)
    base = write(
        tmp_path,
        0,
        summary(rows=1, taps=[45], row_bytes=len(raw), dtype=dtype),
        tok=struct.pack("<I", 1),
        pos=struct.pack("<I", 0),
        kind=b"\x00",
        st=raw,
    )
    assert recording.open_stream(base).tapped(0, 1).tolist() == [[[1.5, -3.0]]]


def test_top_entries_pair_with_rows_by_position(tmp_path):
    # TensorFold's example: with a 36-token prompt, entry 0 belongs to position 35, a
    # prompt row, and the first reply row, position 36, pairs with entry 1.
    base = write(
        tmp_path,
        0,
        summary(
            rows=4, first_row=34, prompt_len=36, row_bytes=8, dtype="bf16", top_k=2
        ),
        tok=struct.pack("<4I", 1, 2, 3, 4),
        pos=struct.pack("<4I", 34, 35, 36, 37),
        kind=bytes([0, 0, 1, 1]),
        st=4 * ROW,
        # The best 2 ids, then their log probabilities; the second entry is padded.
        top=struct.pack("<2I2f", 7, 9, -0.25, -1.5)
        + struct.pack("<2I2f", 4, 0, -0.5, 0.0),
    )
    stream = recording.open_stream(base)
    assert stream.top_entry().tolist() == [-1, 0, 1, -1]
    assert stream.top_ids.tolist() == [[7, 9], [4, 0]]
    assert stream.top_logprobs.tolist() == [[-0.25, -1.5], [-0.5, 0.0]]
    checked = recording.check(base)
    assert checked.findings == ()
    assert (checked.prompt_rows, checked.reply_rows, checked.paired) == (2, 2, 2)


def test_an_entry_past_the_rows_pairs_with_none(tmp_path):
    # Three entries, from positions 3, 4 and 5; the stream's rows end at position 4.
    base = three_rows(
        tmp_path, summary=three(top_k=1), top=3 * struct.pack("<If", 7, -0.25)
    )
    assert recording.open_stream(base).top_entry().tolist() == [-1, 0, 1]
    checked = recording.check(base)
    assert checked.findings == ()
    assert (checked.paired, checked.entries) == (2, 3)


def test_a_sound_stream_has_no_findings(tmp_path):
    checked = recording.check(three_rows(tmp_path))
    assert checked.findings == ()
    assert not checked.unfinished
    assert (checked.prompt_rows, checked.reply_rows, checked.paired) == (2, 1, 0)


def test_a_stream_without_a_summary_is_unfinished(tmp_path):
    base = three_rows(tmp_path, summary=None)
    checked = recording.check(base)
    assert checked.unfinished
    assert checked.findings == ()
    with pytest.raises(ValueError, match="no .json"):
        recording.open_stream(base)


def test_a_stream_with_no_rows_is_legal(tmp_path):
    base = write(tmp_path, 0, summary())
    assert recording.check(base).findings == ()
    stream = recording.open_stream(base)
    assert len(stream.tok) == 0
    assert stream.st.shape == (0, 0)


def test_a_gap_keeps_the_rows_before_it(tmp_path):
    # A write failed on the next two rows: .tok took them, the other files did not.
    base = three_rows(
        tmp_path,
        summary=three(gap="a write failed"),
        tok=struct.pack("<5I", 11, 12, 13, 14, 15),
    )
    assert recording.check(base).findings == ()
    assert recording.open_stream(base).tok.tolist() == [11, 12, 13]


def test_bytes_past_the_rows_need_a_gap(tmp_path):
    base = three_rows(tmp_path, tok=struct.pack("<5I", 11, 12, 13, 14, 15))
    assert recording.check(base).findings == (
        ".tok holds 20 bytes; 3 rows are 12, and the stream names no gap",
    )


def test_a_short_or_missing_file_is_a_finding(tmp_path):
    short = three_rows(tmp_path / "a", st=2 * ROW)
    assert recording.check(short).findings == (".st holds 16 bytes; 3 rows need 24",)
    missing = three_rows(tmp_path / "b")
    (tmp_path / "b" / "0.kind").unlink()
    assert recording.check(missing).findings == ("no .kind file",)
    unparsed = three_rows(tmp_path / "c")
    (tmp_path / "c" / "0.json").write_text('{"rows": 3}')
    assert recording.check(unparsed).findings == (
        '.json: "first_row" is not a whole number of 0 or more',
    )


def test_positions_must_run_on_from_first_row(tmp_path):
    late = three_rows(
        tmp_path / "a", pos=struct.pack("<3I", 3, 4, 5), kind=bytes([0, 1, 1])
    )
    assert recording.check(late).findings == (
        "row 0 is at position 3, not 2 (first_row 2 + 0)",
    )
    broken = three_rows(tmp_path / "b", pos=struct.pack("<3I", 2, 3, 9))
    assert recording.check(broken).findings == (
        "row 2 is at position 9, not 4 (first_row 2 + 2)",
    )


def test_a_kind_must_agree_with_its_position(tmp_path):
    base = three_rows(tmp_path, kind=bytes([0, 1, 1]))
    assert recording.check(base).findings == (
        "row 1 at position 3 has kind 1; the prompt is 4 tokens",
    )


def test_top_must_match_top_k_and_hold_whole_entries(tmp_path):
    entry = struct.pack("<2I2f", 7, 9, -0.25, -1.5)
    stray = three_rows(tmp_path / "a", top=entry)
    assert recording.check(stray).findings == ("top_k is 0, but there is a .top file",)
    absent = three_rows(tmp_path / "b", summary=three(top_k=2))
    assert recording.check(absent).findings == ("no .top file",)
    torn = three_rows(tmp_path / "c", summary=three(top_k=2), top=entry + b"\x00" * 4)
    assert recording.check(torn).findings == (
        ".top holds 20 bytes, not whole entries of 16 (the best 2)",
    )


def test_a_row_must_divide_into_its_taps(tmp_path):
    odd = three_rows(tmp_path / "a", summary=three(row_bytes=6), st=3 * ROW[:6])
    assert recording.check(odd).findings == (
        "row_bytes 6 does not divide into 2 taps of bf16 values",
    )
    # TensorFold's test target records its tokens as u32 rows; a model taps floats.
    other = three_rows(tmp_path / "b", summary=three(dtype="u32"))
    assert recording.check(other).findings == (
        'dtype "u32" is not one of bf16, f16, f32',
    )
    with pytest.raises(ValueError, match="u32"):
        recording.open_stream(other).tapped(0, 3)


def test_streams_are_found_in_number_order_folder_by_folder(tmp_path):
    for folder, n in (("7-10", 0), ("7-9", 10), ("7-9", 9), ("7-9", 2)):
        write(tmp_path / "run" / folder, n, summary())
    (tmp_path / "run" / "7-9" / "notes.json").write_text("{}")
    found = [base.relative_to(tmp_path).as_posix() for base in recording.find(tmp_path)]
    assert found == ["run/7-9/2", "run/7-9/9", "run/7-9/10", "run/7-10/0"]


def test_report_totals_what_it_read(tmp_path):
    three_rows(tmp_path / "7-0", 0)
    three_rows(tmp_path / "7-0", 1, summary=three(gap="a one-token round"))
    three_rows(tmp_path / "7-0", 2, summary=None)
    write(tmp_path / "7-0", 3, summary())
    lines, sound = recording.report(tmp_path)
    assert sound
    assert lines == [
        "streams 4: 3 finished, 1 unfinished",
        "rows 6: 4 prompt, 2 reply, 0 with a .top entry",
        "layout taps [5, 45] x 2 bf16, 8 bytes a row: 2 streams, 6 rows",
        'gap "a one-token round": 1 stream',
        "findings 0",
    ]


def test_report_lists_each_stream_when_asked_and_names_what_is_wrong(tmp_path):
    three_rows(tmp_path / "7-0", 0, summary=three(gap="a one-token round"))
    three_rows(tmp_path / "7-0", 1, st=2 * ROW)
    three_rows(tmp_path / "7-0", 2, summary=None)
    write(tmp_path / "7-0", 3, summary())
    three_rows(tmp_path / "7-0", 4, kind=bytes([0, 1, 1]))
    lines, sound = recording.report(tmp_path, each=True)
    assert not sound
    assert lines == [
        "7-0/0: 3 rows at 2..4, 2 prompt and 1 reply, gap: a one-token round",
        "7-0/1: .st holds 16 bytes; 3 rows need 24",
        "7-0/2: unfinished, no .json",
        "7-0/3: 0 rows",
        "7-0/4: 3 rows at 2..4, 1 prompt and 2 reply",
        "7-0/4: row 1 at position 3 has kind 1; the prompt is 4 tokens",
        "streams 5: 4 finished, 1 unfinished",
        "rows 6: 3 prompt, 3 reply, 0 with a .top entry",
        "layout taps [5, 45] x 2 bf16, 8 bytes a row: 2 streams, 6 rows",
        'gap "a one-token round": 1 stream',
        "findings 2",
    ]
    assert recording.report(tmp_path)[0][:2] == [
        "7-0/1: .st holds 16 bytes; 3 rows need 24",
        "7-0/4: row 1 at position 3 has kind 1; the prompt is 4 tokens",
    ]


def test_a_folder_without_streams_is_not_sound(tmp_path):
    assert recording.report(tmp_path) == (["no streams found"], False)
    assert recording.report(tmp_path / "absent") == (["no streams found"], False)
