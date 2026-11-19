"""The walk against TensorFold's lane core itself.

`fixtures/lane_core_rounds.jsonl` holds the rounds the lane core took over one
stream at a time, each reply a fake target's, under GLM-5.3-Flash's facts with a
DFlash2 drafter (tools/oracle/). Each line gives the prompt, the reply, the
drafts and chances the core asked for by reply index, and the rounds: their kind,
the verify window's rows, the drafts accepted and the tokens given.
"""

import json
from pathlib import Path

import pytest

from open_dneo import rounds

FIXTURE = Path(__file__).parent / "fixtures" / "lane_core_rounds.jsonl"
CASES = [json.loads(line) for line in FIXTURE.read_text("utf-8").splitlines()]
KINDS = {"copy": rounds.COPY, "head": rounds.HEAD, "none": rounds.PLAIN}


def theirs(case) -> list[rounds.Round]:
    return [
        rounds.Round(KINDS[r["kind"]], r["rows"] - 1, r["kept"], r["tokens"])
        for r in case["rounds"]
    ]


@pytest.mark.parametrize("case", CASES, ids=[case["case"] for case in CASES])
def test_the_walk_takes_the_rounds_the_lane_core_took(case):
    asked = {int(at): given for at, given in case["asked"].items()}
    walked = rounds.served(
        case["prompt"],
        case["reply"],
        lambda at: asked[at],  # a KeyError is a draft the core never asked for
        rounds.Lanes(**case["lanes"]),
        case["max_new"],
    )
    assert walked == theirs(case)
    # the core went by the walk's defaults; the first depth is the engine's choice
    setup = {k: v for k, v in case["lanes"].items() if k != "first_depth"}
    assert setup == {k: getattr(rounds.Lanes(), k) for k in setup}


def test_the_cases_reach_every_rule_of_the_walk():
    taken = [round for case in CASES for round in theirs(case)]
    copies = [r for r in taken if r.kind == rounds.COPY]
    # the least a copy takes, then each width it grows to
    assert {r.offered for r in copies} >= {2, 3, 7, 15, 31}
    assert any(0 < r.kept < r.offered for r in copies)  # a copy that lands in part
    assert any(r.kept == 0 for r in copies)  # and one that misses
    heads = [r for r in taken if r.kind == rounds.HEAD]
    assert {r.offered for r in heads} == set(range(1, 8))
    assert {case["lanes"]["first_depth"] for case in CASES} == {2, 4}
    assert any(sum(r.tokens for r in theirs(c)) + 1 < c["max_new"] for c in CASES)
