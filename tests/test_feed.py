"""What a trainer is fed: the rows where the target writes, and anchors before them.

GLM's chat template names each turn with a marker; the target writes the
assistant's turns, tags and all, and the marker that closes them, and a plain text
with no marker stands in for one. The ids are the public tokenizer's.
"""

import numpy as np
import pytest

from open_dneo import feed, train

from .test_score import on_disk

END = 154820  # <|endoftext|>
SYSTEM, USER, ASSISTANT, OBSERVATION = 154826, 154827, 154828, 154829
THINK, UNTHINK, CALL, UNCALL = 154841, 154842, 154843, 154844  # tags the target writes


def test_the_target_writes_its_own_turns_and_the_marker_that_closes_them():
    tokens = np.array([
        SYSTEM, 5, 6, USER, 7, ASSISTANT, THINK, UNTHINK, CALL, 8, UNCALL, OBSERVATION,
        10, ASSISTANT, 11, END,
    ])  # fmt: skip
    assert feed.writable(tokens).tolist() == [
        False, False, False,  # the system's turn
        False, False,  # the user's
        False, True, True, True, True, True,  # the assistant's, its tags among them
        True,  # the marker that closes it: the target writes its turn's end
        False,  # a tool's
        False, True, True,  # the end of the text closes the last turn
    ]  # fmt: skip
    assert feed.writable(np.array([5, 6, 7])).tolist() == [True] * 3  # plain text
    assert feed.writable(np.array([5, USER, 7])).tolist() == [True, False, False]


def test_a_stream_that_resumed_mid_turn_is_not_read_as_plain_text():
    tokens = np.array([5, 6, USER, 7, ASSISTANT, 8])
    # it began after row 0: what turn its first rows are in is not known
    assert feed.writable(tokens, resumed=True).tolist() == [
        False, False, False, False, False, True
    ]  # fmt: skip
    assert feed.writable(tokens).tolist() == [True, True, False, False, False, True]


def test_an_anchor_drafts_while_the_rows_after_it_are_the_targets_to_write():
    tokens = np.array([USER, 7, ASSISTANT, 8, 9, 10, USER, 11])
    # after row 2 (the marker) rows 3 to 6 are written, the closing marker last
    assert feed.ahead(feed.writable(tokens)).tolist() == [0, 0, 4, 3, 2, 1, 0, 0]
    assert feed.anchors(tokens).tolist() == [2, 3, 4, 5]


def test_a_block_counts_no_draft_past_the_targets_turn():
    tokens = np.array([ASSISTANT, 8, 9, 10, USER, 11, 12])
    written = np.array([8, 9, 10, USER, 11, 12, 13])  # the target chose each one
    taught = feed.blocks(tokens, written, np.array([0, 2]), drafts=3)
    plain = train.blocks(tokens, written, np.array([0, 2]), drafts=3)
    assert plain.on_path.tolist() == [3, 3]
    # from row 2, rows 3 and 4 are the target's to write: its turn ends with row 4
    assert taught.on_path.tolist() == [3, 2] and taught.available.tolist() == [3, 2]
    assert taught.labels.tolist() == plain.labels.tolist()


@pytest.mark.parametrize("seed", [0, 1])
def test_a_window_takes_anchors_at_random_from_one_span_of_a_stream(seed):
    rng = np.random.default_rng(seed)
    eligible = np.arange(10, 400)
    rows = feed.window(eligible, anchors=8, span=50, rng=rng)
    assert len(rows) == 8 and len(set(rows.tolist())) == 8
    assert (np.diff(rows) > 0).all() and rows[-1] - rows[0] < 50
    assert set(rows.tolist()) <= set(eligible.tolist())
    few = feed.window(np.array([3, 9]), anchors=8, span=50, rng=rng)
    assert few.tolist() == [3, 9]  # fewer than asked: every one


def recorded(tmp_path, n, tokens, folder="5-0"):
    """A stream of `tokens` on disk in plain text (no prompt), with its saved choices."""
    values = np.zeros((len(tokens), 3, 8), np.float32)
    on_disk(tmp_path / "rec" / folder, values, tokens, first=0, prompt=len(tokens), n=n)
    choices = tmp_path / "next" / folder / f"{n}.next"
    choices.parent.mkdir(parents=True, exist_ok=True)
    np.roll(np.asarray(tokens), -1).astype("<i4").tofile(choices)


def test_streams_come_by_name_with_their_choices_and_where_a_block_can_start(tmp_path):
    recorded(tmp_path, 1, [5, 6, 7, 8])
    recorded(tmp_path, 0, [USER, 7, ASSISTANT, 8, 9, USER])
    listing = tmp_path / "parts.tsv"
    listing.write_text("5-0/0\tchat\n", encoding="utf-8")
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next", [listing])
    assert [f.name for f in fed] == ["5-0/0", "5-0/1"]
    assert [f.part for f in fed] == ["chat", ""]
    assert [f.rows for f in fed] == [6, 4] and [f.anchorable for f in fed] == [3, 3]
    first, second = fed[0].open(), fed[1].open()
    assert first.written.tolist() == [7, ASSISTANT, 8, 9, USER, USER]
    assert first.eligible.tolist() == [2, 3, 4] and second.eligible.tolist() == [
        0,
        1,
        2,
    ]
    assert first.room.tolist() == [0, 0, 3, 2, 1, 0]  # the closing USER is written
    assert first.stream.tok.tolist() == [USER, 7, ASSISTANT, 8, 9, USER]


def test_streams_keep_no_file_mapped_until_a_stream_is_opened(tmp_path, monkeypatch):
    import gc

    for n in range(3):
        recorded(tmp_path, n, [5, 6, 7, 8])

    def mapped():
        gc.collect()
        return sum(
            isinstance(o, np.memmap)
            and str(tmp_path) in str(getattr(o, "filename", ""))
            for o in gc.get_objects()
        )

    seen, real = [], feed.recording.open_stream
    monkeypatch.setattr(
        feed.recording, "open_stream", lambda base: seen.append(mapped()) or real(base)
    )
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next")
    monkeypatch.setattr(feed.recording, "open_stream", real)
    # while reading them in, each stream is closed before the next is opened
    assert seen == [0, 0, 0]
    assert len(fed) == 3 and mapped() == 0
    opened = fed[0].open()
    assert mapped() == 1  # the tapped rows, while the stream is open
    del opened
    assert mapped() == 0


def test_a_stream_without_its_choices_is_refused(tmp_path):
    recorded(tmp_path, 0, [5, 6, 7])
    (tmp_path / "next" / "5-0" / "0.next").write_bytes(b"\0" * 8)
    with pytest.raises(ValueError, match="5-0/0: 2 saved choices for 3 rows"):
        feed.streams([tmp_path / "rec"], tmp_path / "next")
    (tmp_path / "next" / "5-0" / "0.next").unlink()
    with pytest.raises(ValueError, match="no saved choices for 5-0/0"):
        feed.streams([tmp_path / "rec"], tmp_path / "next")


def test_every_nth_stream_by_name_is_held_out_as_the_bridge_holds_them(tmp_path):
    for n in range(5):
        recorded(tmp_path, n, [5, 6, 7])
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next")
    trained, held = feed.held(fed, every=2)
    assert [f.name for f in held] == ["5-0/1", "5-0/3"]
    assert [f.name for f in trained] == ["5-0/0", "5-0/2", "5-0/4"]


def test_a_step_takes_a_stream_of_every_part_weighted_by_its_share(tmp_path):
    recorded(tmp_path, 0, [5, 6, 7, 8, 9])  # 4 rows to anchor
    recorded(tmp_path, 1, [5, 6])  # 1
    recorded(tmp_path, 2, [5, 6, 7])  # 2
    listing = tmp_path / "parts.tsv"
    listing.write_text("5-0/0\tcode\n5-0/1\tchat\n5-0/2\tchat\n", encoding="utf-8")
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next", [listing])
    sampler = feed.Sampler(fed, rng=np.random.default_rng(0))
    taken = sampler.step()
    assert sorted(f.part for f, _ in taken) == ["chat", "code"]
    # by default a part weighs its anchors: code 4, chat 3
    shares = {f.part: share for f, share in taken}
    assert shares == pytest.approx({"code": 4 / 7, "chat": 3 / 7})
    given = feed.Sampler(fed, {"code": 1.0, "chat": 3.0}, np.random.default_rng(0))
    assert {f.part: s for f, s in given.step()} == pytest.approx(
        {"code": 0.25, "chat": 0.75}
    )
    with pytest.raises(ValueError, match="no stream of the part prose"):
        feed.Sampler(
            fed, {"code": 1.0, "chat": 1.0, "prose": 1.0}, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="the part chat has streams and no share"):
        feed.Sampler(fed, {"code": 1.0}, np.random.default_rng(0))


def test_streams_sort_by_name_as_the_bridge_sorts_them(tmp_path):
    recorded(tmp_path, 2, [5, 6, 7])
    recorded(tmp_path, 10, [5, 6, 7])
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next")
    # by the name as written, not by its number: the bridge holds out the same ones
    assert [f.name for f in fed] == ["5-0/10", "5-0/2"]
    assert [f.name for f in feed.held(fed, every=2)[1]] == ["5-0/2"]


def test_a_recorded_stream_that_resumed_is_read_from_its_first_marker(tmp_path):
    tokens = [5, 6, USER, 7, ASSISTANT, 8, 9]
    values = np.zeros((len(tokens), 3, 8), np.float32)
    on_disk(tmp_path / "rec", values, tokens, first=3, prompt=10)
    choices = tmp_path / "next" / "0.next"
    choices.parent.mkdir(parents=True)
    np.roll(np.asarray(tokens), -1).astype("<i4").tofile(choices)
    (fed,) = feed.streams([tmp_path / "rec"], tmp_path / "next")
    # its first rows are in a turn it does not name: only the assistant's rows count
    assert fed.anchorable == 2 and fed.open().eligible.tolist() == [4, 5]
