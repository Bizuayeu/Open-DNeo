import numpy as np
import pytest

from open_dneo import logged, runs


def request(id_, prompt, reply, klass="code"):
    return runs.Request(id_, klass, 16, tuple(prompt), tuple(reply), 3)


def line(round_, slot, pos, anchor, drafts, kept=0):
    """A round as TensorFold's GLM engine logs it (`dflash2 round` in dflash.zig)."""
    shown = ", ".join(str(d) for d in drafts)
    return (
        f"info: dflash2 round {round_}: slot {slot} pos {pos} kept {kept}"
        f" anchor {anchor} drafts {{ {shown} }}"
    )


def test_read_takes_the_engines_round_lines_and_nothing_else(tmp_path):
    path = tmp_path / "serve.log"
    path.write_text(
        "\n".join(
            [
                "info: dflash2: drafter from /x (Apache-2.0), linears at 4 bits",
                line(0, 1, 5, 7, range(100, 115), kept=2),
                "info: dflash2: 256 rounds, 0 contexts restarted",
                line(1, 1, 8, 9, [3] * 15),
            ]
        ),
        encoding="utf-8",
    )
    first, second = logged.read(path)
    assert first == logged.Logged(1, 5, 7, tuple(range(100, 115)))
    assert second.pos == 8 and second.drafts == (3,) * 15


def test_read_refuses_a_round_line_it_cannot_take(tmp_path):
    path = tmp_path / "serve.log"
    path.write_text("info: dflash2 round 3: slot 0 pos 5 anchor 7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is a round's, and not as it is read"):
        logged.read(path)


def test_assign_gives_each_request_the_rounds_at_its_own_tokens():
    a = request("a", [1, 2, 3], [10, 11, 12, 13, 14])  # positions 3..7
    b = request("b", [1, 2], [20, 21, 22])  # positions 2..4
    c = request("c", [1], [30, 31])  # a reply no round of the log took
    d = request("d", [1, 2, 3, 4, 5], [40, 41, 42, 43])  # positions 5..8
    rounds = [
        logged.Logged(0, 9, 99, ()),  # a round before the first request's: no request's
        logged.Logged(0, 3, 10, ()),
        logged.Logged(0, 5, 12, ()),
        logged.Logged(1, 2, 20, ()),
        logged.Logged(1, 3, 21, ()),
        logged.Logged(0, 5, 40, ()),
        logged.Logged(0, 7, 42, ()),
    ]
    taken, astray = logged.assign(rounds, [a, b, c, d])
    assert [[r.pos for r in each] for each in taken] == [[3, 5], [2, 3], [], [5, 7]]
    assert astray == 1


def test_assign_takes_a_second_request_like_the_first_in_its_own_turn():
    same = [request("a", [1, 2], [5, 6, 7]), request("b", [1, 2], [5, 6, 7])]
    rounds = [logged.Logged(0, 2, 5, ()), logged.Logged(0, 3, 6, ())] * 2
    taken, astray = logged.assign(rounds, same)
    assert [len(each) for each in taken] == [2, 2] and astray == 0


def test_compare_counts_drafts_that_are_the_instruments_from_the_first_on():
    positions = np.array([4, 5, 6])
    tokens = np.array([10, 11, 12])
    drafts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rounds = [
        logged.Logged(0, 4, 10, (1, 2, 3)),  # all three the instrument's
        logged.Logged(0, 5, 11, (4, 0, 6)),  # the first only: the second differs
        logged.Logged(0, 6, 12, (0, 8, 9)),  # none from the first on
        logged.Logged(0, 7, 13, (1, 1, 1)),  # a position the recording has no row for
    ]
    got = logged.compare(rounds, positions, tokens, drafts)
    assert got == logged.Agreement(rounds=3, first=2, leading=4, whole=1, unrecorded=1)


def test_compare_refuses_a_round_whose_anchor_is_not_the_streams_token():
    with pytest.raises(
        ValueError, match="position 4: the log's anchor is 9, the stream's 10"
    ):
        logged.compare(
            [logged.Logged(0, 4, 9, (1,))],
            np.array([4]),
            np.array([10]),
            np.array([[1]]),
        )


def test_agreements_add_up():
    one = logged.Agreement(rounds=3, first=2, leading=4, whole=1, unrecorded=1)
    assert one + one == logged.Agreement(6, 4, 8, 2, 2)
    assert sum([one, one], logged.Agreement()) == one + one
