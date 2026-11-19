import pytest

from open_dneo import copies


def test_a_repeated_run_proposes_what_followed_it():
    # TensorFold's own case: "suffix lookup copies the continuation after a repeated n-gram"
    look = copies.SuffixLookup(ngram=2, min_match=2)
    assert look.propose([1, 2, 3, 4, 5, 9, 1, 2, 3], 3) == [4, 5, 9]
    assert look.last_match == 3
    assert look.last_end == 3


def test_a_tail_continues_the_match_and_only_the_history_is_indexed():
    # TensorFold's own case: "a drafted tail continues the history's copy, and history alone is indexed"
    look = copies.SuffixLookup(ngram=3, min_match=4)
    history = [7, 1, 2, 3, 4, 5, 6, 8, 9, 7, 1]
    assert look.propose_after(history, [2, 3, 4], 3) == [5, 6, 8]
    assert (look.last_match, look.last_end) == (5, 5)
    assert look.indexed == len(history)
    assert look.propose_after(history, [2, 3, 9], 3) == []
    # a run the history never held: no match, and the last match's place is left as it was
    assert (look.last_match, look.last_end) == (0, 5)


def test_silence_lasts_its_asks_and_an_ask_without_room_is_none():
    # TensorFold's own case: "silence lasts silence_rounds asks however many a round makes"
    look = copies.SuffixLookup(ngram=2, min_match=2, silence_rounds=3, window=2)
    context = [1, 2, 3, 4, 5, 9, 1, 2, 3]
    look.observe(3, 0)
    look.observe(3, 0)
    assert look.silent_for == 3
    assert look.propose_after(context, [], 0) == []  # no room: not an ask
    assert look.silent_for == 3
    assert look.propose_after(context, [3], 3) == []
    assert look.propose(context, 3) == []
    assert look.propose(context, 3) == []
    assert look.silenced == 3
    assert look.propose(context, 3) == [4, 5, 9]


def test_the_run_at_the_very_end_is_not_its_own_match():
    look = copies.SuffixLookup(ngram=2, min_match=2)
    assert look.propose([1, 2, 9, 1, 2], 3) == [9, 1, 2]
    assert look.propose([5, 1, 2], 3) == []
    assert (look.last_match, look.last_end) == (0, 0)


def test_a_match_ends_at_the_first_token_of_the_history():
    look = copies.SuffixLookup(ngram=2, min_match=2)
    # the history's last token is also the one before the later run: not a fourth match
    assert look.propose([1, 2, 3, 4, 5, 3, 1, 2, 3], 1) == [4]
    assert look.last_match == 3


def test_the_longest_match_wins_and_the_latest_of_equals():
    look = copies.SuffixLookup(ngram=3, min_match=3)
    longer_first = [9, 1, 2, 3, 10, 8, 1, 2, 3, 20, 9, 1, 2, 3]
    assert look.propose(longer_first, 2) == [10, 8]
    assert (look.last_match, look.last_end) == (4, 4)
    look = copies.SuffixLookup(ngram=3, min_match=3)
    equal = [7, 1, 2, 3, 10, 7, 1, 2, 3, 20, 7, 1, 2, 3]
    assert look.propose(equal, 2) == [20, 7]
    assert (look.last_match, look.last_end) == (4, 9)


def test_a_match_is_counted_no_further_than_the_longest_extension():
    look = copies.SuffixLookup(ngram=2, min_match=2, max_extension=3)
    context = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 7, 1, 2, 3, 4, 5]
    # the later run matches 5 tokens back and the earlier one 5 too: both count as 3, and the later is taken
    assert look.propose(context, 1) == [7]
    assert look.last_match == 3


def test_a_match_shorter_than_the_least_proposes_nothing_but_is_told():
    look = copies.SuffixLookup(ngram=2, min_match=4)
    assert look.propose([1, 2, 3, 4, 5, 9, 1, 2, 3], 3) == []
    assert look.last_match == 3
    assert (look.proposals, look.proposed_tokens) == (0, 0)


def test_a_proposal_is_as_long_as_the_room_and_the_history_allow():
    context = [1, 2, 3, 4, 5, 9, 1, 2, 3]
    look = copies.SuffixLookup(ngram=2, min_match=2)
    assert look.propose(context, 1) == [4]
    assert look.propose(context, 50) == [4, 5, 9, 1, 2, 3]
    assert (look.proposals, look.proposed_tokens) == (2, 7)


def test_a_history_too_short_for_a_run_and_a_token_is_no_ask():
    look = copies.SuffixLookup(ngram=3, min_match=3, silence_rounds=2, window=1)
    look.observe(2, 0)
    assert look.silent_for == 2
    assert look.propose([1, 2, 3], 3) == []
    assert (look.silent_for, look.silenced) == (2, 0)


def test_a_longer_history_adds_to_the_index_and_another_one_starts_it_again():
    look = copies.SuffixLookup(ngram=2, min_match=2)
    first = [1, 2, 3, 4, 1, 2]
    assert look.propose(first, 2) == [3, 4]
    grown = first + [7, 8, 3, 7, 8]
    assert look.propose(grown, 2) == [3, 7]  # a run that only the added tokens hold
    assert look.indexed == len(grown)
    back = grown + [1, 2]
    assert look.propose(back, 2) == [7, 8]  # a run indexed at the first ask
    shorter = [9, 1, 2, 7, 7, 1, 2]
    assert look.propose(shorter, 2) == [7, 7]  # nothing of the history before is left
    assert look.indexed == len(shorter)
    # as long, but not the same where the index ended: started again as well
    another = [3, 3, 3, 3, 3, 4, 5, 6, 4, 5]
    assert look.propose(another, 2) == [6, 4]


def test_rejections_in_a_row_silence_the_lookup_and_one_landing_starts_the_count_again():
    look = copies.SuffixLookup(ngram=2, min_match=2, silence_rounds=5, window=3)
    for accepted in (0, 0, 2, 0, 0):
        look.observe(3, accepted)
    assert look.silent_for == 0  # the last three hold a landing
    look.observe(3, 0)
    assert look.silent_for == 5
    context = [1, 2, 3, 4, 5, 9, 1, 2, 3]
    assert look.propose(context, 3) == look.propose(context, 3) == []
    look.observe(3, 0)
    look.observe(3, 0)
    assert look.silent_for == 3  # the count began again when the silence was set
    assert (look.judged_tokens, look.accepted_tokens) == (24, 2)


def test_an_observation_without_a_proposal_is_not_a_rejection():
    look = copies.SuffixLookup(ngram=2, min_match=2, silence_rounds=5, window=1)
    look.observe(0, 0)
    assert look.silent_for == 0


@pytest.mark.parametrize(
    ("given", "why"),
    [
        ({"ngram": 0}, "an n-gram of 1 to 8"),
        ({"ngram": 9, "min_match": 9}, "an n-gram of 1 to 8"),
        ({"ngram": 3, "min_match": 2}, "shorter than the n-gram"),
    ],
)
def test_sizes_that_cannot_match_are_refused(given, why):
    with pytest.raises(ValueError, match=why):
        copies.SuffixLookup(**given)


def test_a_window_is_one_proposal_at_least():
    look = copies.SuffixLookup(ngram=2, min_match=2, silence_rounds=4, window=0)
    look.observe(1, 2)
    look.observe(1, 0)
    assert look.silent_for == 4  # the landing before it is not remembered
