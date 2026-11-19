"""Counting tokens a round as the engine's lanes cut a chain of drafts."""

import pytest

from open_dneo import rounds


@pytest.mark.parametrize(
    ("chances", "kept"),
    [
        ([], 0),
        ([0.9, 0.9, 0.5], 2),  # 0.81 holds, 0.405 is under the cut
        ([0.1, 0.99], 1),  # the first draft is verified whatever its chance
        ([0.7, 1.0, 1.0], 3),  # a product at the cut holds
        ([0.9, 0.7, 1.0], 1),  # 0.63 ends the chain for the drafts behind it too
        ([0.0, 1.0], 0),  # no candidate at all
        ([1.0, 0.0, 1.0], 1),
    ],
)
def test_a_chain_ends_where_its_chances_product_falls_under_the_cut(chances, kept):
    assert rounds.held(chances) == kept


def test_the_cut_is_a_setting():
    assert rounds.held([0.9, 0.9, 0.5], cut=0.4) == 3
    assert rounds.held([0.9, 0.9, 0.5], cut=0.9) == 1


def test_drafts_land_up_to_the_first_that_differs():
    assert rounds.landed([1, 2, 3], [1, 2, 4]) == 2
    assert rounds.landed([1, 2, 3], [9, 2, 3]) == 0
    assert rounds.landed([1, 2, 3], [1, 2]) == 2  # the reply ends first
    assert rounds.landed([], [1]) == 0


def test_a_round_gives_its_accepted_drafts_and_one_token_of_the_targets():
    reply = [10, 11, 12, 13, 14, 15, 16]
    asked = []

    def propose(at):
        asked.append(at)
        drafts = {0: [11, 12, 99], 3: [99, 15, 16], 4: [15, 16, 17]}[at]
        return drafts, [1.0, 1.0, 1.0]

    # two land and the target adds 13; none lands and it adds 14; two land, and
    # the reply holds no token for the target to add
    assert rounds.walk(reply, propose) == [3, 1, 2]
    assert asked == [0, 3, 4]


def test_a_draft_behind_the_cut_is_not_verified_even_when_it_is_right():
    reply = [10, 11, 12, 13]
    steps = rounds.walk(reply, lambda at: (reply[at + 1 :], [0.9, 0.5, 1.0]))
    assert steps == [2, 1]  # the first draft alone each round: 0.45 is under the cut


# The rounds serving takes: a copy where the stream's own tokens back one, else the
# drafter's chain.

Round, Lanes = rounds.Round, rounds.Lanes
RUN = [1, 2, 3, 4, 5, 6, 7, 8]
# a prompt that counts up, and one whose twelve tokens end otherwise than a reply will
LONG = [*range(1, 81), 500]
SHORT = [*range(1, 13), 50]


def wrong(at):
    """A drafter sure of seven drafts that never land."""
    return [-1] * 7, [1.0] * 7


def right(reply):
    """A drafter sure of seven drafts that all land, while the reply lasts."""
    return lambda at: ((reply[at + 1 :] + [-1] * 7)[:7], [1.0] * 7)


def copies(given):
    return [each for each in given if each.kind == "copy"]


def test_the_first_round_holds_fewer_drafts_than_the_ones_after_it():
    reply = list(range(100, 120))  # nothing repeats: no copy
    assert rounds.served([1, 2, 3], reply, right(reply)) == [
        Round("head", 4, 4, 5),
        Round("head", 7, 7, 8),
        Round("head", 7, 6, 6),  # the reply ends inside the round
    ]
    whole = rounds.served([1, 2, 3], reply, right(reply), Lanes(first_depth=7))
    assert [given.tokens for given in whole] == rounds.walk(reply, right(reply))


def test_a_copy_takes_the_round_once_eight_tokens_match():
    reply = [*range(1, 13), 60, 61]
    assert rounds.served(SHORT, reply, wrong) == [
        Round("head", 4, 0, 1),
        *[Round("head", 7, 0, 1)] * 6,  # seven tokens match: not enough
        Round("copy", 3, 3, 4),  # 9 10 11 land, and the target adds 12
        Round("copy", 7, 0, 1),  # twice as wide and one: 50 is not what follows now
        Round("head", 7, 0, 1),
    ]


def test_a_copy_that_lands_whole_grows_to_the_widest():
    reply = list(range(1, 81))
    assert copies(rounds.served(LONG, reply, wrong)) == [
        Round("copy", 3, 3, 4),
        Round("copy", 7, 7, 8),
        Round("copy", 15, 15, 16),
        Round("copy", 31, 31, 32),
        Round("copy", 31, 12, 12),  # the reply ends inside the round
    ]
    narrow = copies(rounds.served(LONG, reply, wrong, Lanes(max_copy=7)))
    assert [given.offered for given in narrow[:4]] == [3, 7, 7, 7]


def test_a_copy_that_does_not_land_whole_starts_the_next_one_small():
    missed = [*range(1, 13), 60, *range(20, 32)]
    assert copies(rounds.served(LONG, missed, wrong)) == [
        Round("copy", 3, 3, 4),
        Round("copy", 7, 0, 1),  # 13 14 ... was proposed, and 60 came
        Round("copy", 3, 3, 4),  # eight of 20 ... are in again: 28 29 30, then 31
    ]
    partly = [*range(1, 16), 900, *range(20, 41)]
    assert copies(rounds.served(LONG, partly, wrong)) == [
        Round("copy", 3, 3, 4),
        Round("copy", 7, 3, 4),  # 13 14 15 land, then 900 comes
        Round("copy", 3, 3, 4),
        Round("copy", 7, 7, 8),
        Round("copy", 15, 1, 1),
    ]


def test_four_copies_in_a_row_that_land_nothing_silence_sixteen_asks_two_a_round():
    prompt = [*RUN, 100]
    reply = [*RUN, 101, *RUN, 102, *RUN, 103, *RUN, 101, *RUN, 102]
    got = rounds.served(prompt, reply, wrong)
    at = [i for i, given in enumerate(got) if given.kind == "copy"]
    assert at == [7, 16, 25, 34, 43]
    assert [got[i] for i in at[:4]] == [Round("copy", 3, 0, 1)] * 4
    # after the fourth, the run and 101 match the reply's own start nine tokens deep, and
    # the lookup is silent: eight rounds of two asks go by before a copy is taken again
    assert got[35:43] == [Round("head", 7, 0, 1)] * 8
    assert got[43] == Round("copy", 3, 1, 1)
    assert sum(given.tokens for given in got) == len(reply) - 1


def test_the_room_left_bounds_a_copy_and_the_drafts_asked():
    reply = list(range(1, 11))
    assert rounds.served(SHORT, reply, wrong)[-1] == Round("copy", 3, 2, 2)
    bounded = rounds.served(SHORT, reply, wrong, max_new=10)
    # room for one token after the pending one is no copy of two; the drafts asked
    # shrink with the room
    assert [given.kind for given in bounded] == ["head"] * 9
    assert [given.offered for given in bounded] == [4, 7, 6, 5, 4, 3, 2, 1, 1]
    assert rounds.served(SHORT, reply, wrong, max_new=3)[0] == Round("head", 2, 0, 1)


def test_a_round_without_a_draft_gives_the_targets_token_alone():
    got = rounds.served([1, 2], [10, 11, 12], lambda at: ([11, 12], [0.0, 1.0]))
    assert got == [Round("plain", 0, 0, 1), Round("plain", 0, 0, 1)]


def test_a_reply_of_one_token_takes_no_round():
    assert rounds.served([1, 2, 3], [9], wrong) == []
    assert rounds.served([1, 2, 3], [], wrong) == []


def test_the_copies_follow_the_lanes_settings():
    reply = [*range(1, 13), 60, 61]

    def kinds(lanes, count):
        return [
            given.kind for given in rounds.served(SHORT, reply, wrong, lanes)[:count]
        ]

    assert kinds(Lanes(enter_match=4), 4) == ["head", "head", "head", "copy"]
    # three tokens match: the lookup proposes from that only if its own least is three
    assert kinds(Lanes(enter_match=3, min_match=3), 3) == ["head", "head", "copy"]
    assert kinds(Lanes(enter_match=3), 4) == ["head", "head", "head", "copy"]
    two = rounds.served(SHORT, reply, wrong, Lanes(first_copy=2))
    assert copies(two)[0] == Round("copy", 2, 2, 3)
    assert copies(rounds.served(SHORT, reply, wrong, Lanes(least_copy=4))) == []
    assert copies(rounds.served(SHORT, reply, wrong, Lanes(first_copy=1))) == []


def test_the_drafts_follow_the_lanes_settings():
    def unsure(at):
        return [-1] * 7, [0.9, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]

    def offered(lanes, max_new=None):
        got = rounds.served([1], [5, 6, 7], unsure, lanes, max_new)
        return [given.offered for given in got]

    assert offered(Lanes()) == [1, 1]  # 0.45 is under the cut
    assert offered(Lanes(cut=0.4)) == [4, 7]
    assert offered(Lanes(cut=0.4, drafts=3)) == [3, 3]
    assert offered(Lanes(cut=0.4, drafts=3), max_new=20) == [3, 3]
    assert offered(Lanes(cut=0.4, first_depth=2)) == [2, 7]
