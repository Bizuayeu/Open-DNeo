"""Training's blocks, weights and loss, on small cases worked by hand.

A block is taught what greedy verification checks: after its anchor, the target's
own choices, while the text keeps making them (`greedy.own_path`). Its head is told
the text's tokens before each position, which on that path are the same choices.
"""

import dataclasses

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from open_dneo import bridge, drafter, score, train, weights

from .test_score import TAKEN, drawn, folders, made, stream

TOKENS = np.array([3, 1, 2, 0, 1])
WRITTEN = np.array([1, 2, 4, 1, 3])  # the target's choice after each row (test_greedy)


def test_a_block_is_taught_the_choices_after_its_anchor_and_told_the_texts_tokens():
    taught = train.blocks(TOKENS, WRITTEN, np.array([0, 2, 4]), drafts=3)
    # after row 0, draft j is taught the choice after row j - 1 and told row j - 1's
    # token; a draft past the last row has no label (-1) and is told token 0
    assert taught.labels.tolist() == [[1, 2, 4], [4, 1, 3], [3, -1, -1]]
    assert taught.previous.tolist() == [[3, 1, 2], [2, 0, 1], [1, 0, 0]]
    # row 3's token is not the target's choice (0 for 4): from row 2 one label is on
    # the path; past the last row there is none to take
    assert taught.on_path.tolist() == [3, 1, 1]
    assert taught.available.tolist() == [3, 3, 1]


def test_a_drafts_weight_falls_with_its_position_and_off_the_path():
    taught = train.blocks(TOKENS, WRITTEN, np.array([0, 2, 4]), drafts=3)
    falling = np.exp(-np.arange(3) / 4.0)
    weights = train.weights(taught, decay=4.0)
    np.testing.assert_allclose(
        weights, [falling, [falling[0], 0, 0], [falling[0], 0, 0]]
    )
    # off the path a lower weight, and past the last row none
    weights = train.weights(taught, decay=4.0, off_path=0.5)
    np.testing.assert_allclose(
        weights,
        [falling, [falling[0], 0.5 * falling[1], 0.5 * falling[2]], [falling[0], 0, 0]],
    )
    with pytest.raises(ValueError, match="decay is 0"):
        train.weights(taught, decay=0.0)


def test_the_loss_is_the_labels_cross_entropy_weighted_and_over_the_weights():
    values = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 3.0, 0.5]]])  # [m 1, n 2, 3]
    weights = torch.tensor([[1.0, 0.5]])
    want = (
        F.cross_entropy(values[0, :1], torch.tensor([2]))
        + 0.5 * F.cross_entropy(values[0, 1:], torch.tensor([1]))
    ) / 1.5
    whole = drafter.Taught(values, None)
    torch.testing.assert_close(train.loss(whole, torch.tensor([[2, 1]]), weights), want)
    # among candidates a label is found by its slot
    among = drafter.Taught(values, torch.tensor([[[5, 7, 9], [4, 8, 6]]]))
    torch.testing.assert_close(train.loss(among, torch.tensor([[9, 8]]), weights), want)
    # a weight of 0 or no label counts for nothing, and nothing to count is 0
    none = torch.tensor([[2, -1]])
    torch.testing.assert_close(
        train.loss(whole, none, weights),
        F.cross_entropy(values[0, :1], torch.tensor([2])),
    )
    zero = train.loss(whole, torch.tensor([[2, 1]]), torch.zeros(1, 2))
    assert zero.item() == 0.0
    with pytest.raises(ValueError, match="1 labels with a weight are not among"):
        train.loss(among, torch.tensor([[9, 3]]), weights)


SMALL = drafter.Shape(
    hidden=8, heads=2, kv_heads=1, head_dim=4, ffn=8, layers=1, block=4, group=2,
    eps=1e-5, theta=10000.0, vocab=11, rank=3, taps=2, top_k=4, folded=3,
)  # fmt: skip


@pytest.mark.parametrize("kind", ["Selector", "Folded"])
def test_a_head_taught_a_label_it_would_not_score_swaps_its_weakest_for_it(kind):
    torch.manual_seed(4)
    head = drafter.Selector(SMALL) if kind == "Selector" else drafter.Folded(SMALL, 4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_()
    n = SMALL.block - 1
    hidden = torch.randn(2, n, SMALL.hidden)
    scores = torch.arange(SMALL.vocab, dtype=torch.float32).expand(2, n, -1).clone()
    previous = torch.randint(SMALL.vocab, (2, n))
    # the best 4 are 10, 9, 8, 7: label 3 is not among them, 9 is, -1 is none
    labels = torch.tensor([[3, 9, -1], [10, 3, 3]])
    with torch.no_grad():
        plain = head.teach(hidden, scores, previous)
        taught = head.teach(hidden, scores, previous, labels)
    swapped = (labels == 3)[..., None] & (torch.arange(4) == 3)
    want = torch.where(swapped, torch.tensor(3), plain.candidates)
    assert torch.equal(taught.candidates, want)
    kept = ~swapped
    torch.testing.assert_close(taught.values[kept], plain.values[kept])
    # the swapped slot holds what the head scores 3 at among candidates that hold
    # it: the target head's score of 3 the same, the others' lowered
    lowered = scores.clone()
    lowered[..., 4:] = -1.0  # the best 4 are now 3, 2, 1, 0
    with torch.no_grad():
        among = head.teach(hidden, lowered, previous)
    slot = (among.candidates == 3).float().argmax(-1, keepdim=True)
    own = among.values.gather(-1, slot)[..., 0]
    torch.testing.assert_close(taught.values[..., 3][labels == 3], own[labels == 3])


def test_a_batch_holds_the_rows_its_anchors_see_and_what_their_blocks_are_taught():
    model, target, reading = made("dneo")
    values, tokens = drawn(12)
    held = stream(values, tokens, first=2, prompt=4)
    written = np.roll(np.asarray(tokens, np.int64), -1)  # the text, as if chosen
    written[[6, 8]] = 0  # but for two rows
    rows = np.array([3, 5, 9])
    got = train.batch(held, written, rows, reading, model.shape.block, decay=4.0)
    # the rows before the last anchor (positions 2 to 10), the taps the reading takes
    taken = torch.from_numpy(values[:9][:, [2, 0]])
    torch.testing.assert_close(got.features[0], taken.to(got.features))
    assert got.context_positions.tolist() == [list(range(2, 11))]
    assert got.anchors.tolist() == [[5, 7, 11]]
    assert got.ids[:, 0].tolist() == [int(tokens[r]) for r in rows]
    assert (got.ids[:, 1:] == reading.mask_token).all()
    taught = train.blocks(np.asarray(tokens, np.int64), written, rows, 3)
    assert got.labels.tolist() == taught.labels.tolist()
    assert got.previous.tolist() == taught.previous.tolist()
    np.testing.assert_allclose(got.weights.numpy(), train.weights(taught, 4.0))
    assert got.on_path.tolist() == taught.on_path.tolist()


@pytest.mark.parametrize("kind", ["dflash2", "dneo"])
def test_at_step_0_a_blocks_first_draft_is_the_instruments(kind):
    model, target, reading = made(kind)
    values, tokens = drawn(12)
    held = stream(values, tokens, first=0, prompt=5)
    found = score.proposals(model, target, held, reading)
    rows = np.arange(5, 12)  # the reply rows proposals took as anchors
    unlabelled = np.full(12, -1)  # no label: a head scores as it drafts
    got = train.batch(held, unlabelled, rows, reading, model.shape.block, decay=4.0)
    with torch.no_grad():
        taught = train.taught(model, target, got, reading)
    first = taught.values[:, 0].argmax(-1)
    if taught.candidates is not None:
        first = taught.candidates[:, 0].gather(-1, first[:, None])[:, 0]
    assert first.tolist() == found.drafts[:, 0].tolist()


@pytest.mark.parametrize("kind, best", [("dflash2", None), ("dneo", 0), ("dneo", 5)])
@pytest.mark.parametrize("off_path", [0.0, 0.5])
def test_a_batch_scored_only_where_it_weighs_gives_the_dense_loss_and_measure(
    kind, best, off_path
):
    model, target, reading = made(kind)
    if best:
        model.head.best = best
    values, tokens = drawn(20)
    held = stream(values, tokens, first=0, prompt=3)
    written = np.roll(np.asarray(tokens, np.int64), -1)  # the text, as if chosen
    written[[6, 9, 13]] = 0  # but for three rows: paths that leave it early
    got = train.batch(
        held, written, np.arange(3, 20), reading, 4, decay=4.0, off_path=off_path
    )
    only = train.counted(got)
    assert only.any() and (~only).any()

    def gradients(value):
        model.zero_grad()
        value.backward()
        return {
            n: torch.zeros_like(p) if p.grad is None else p.grad.clone()
            for n, p in model.named_parameters()
        }

    dense = train.taught(model, target, got, reading)
    value = train.loss(dense, got.labels, got.weights)
    want = gradients(value)
    sparse = train.taught(model, target, got, reading, only=only)
    assert sparse.values.shape[0] == int(only.sum())
    value_sparse = train.loss(
        sparse, train.chosen(got.labels, only), train.chosen(got.weights, only)
    )
    torch.testing.assert_close(value_sparse, value)
    # the loss is taken in float32 (`train.loss`): rows in another order part in
    # its last bits
    for name, grad in gradients(value_sparse).items():
        torch.testing.assert_close(grad, want[name], rtol=1e-5, atol=1e-5, msg=name)
    # what the head would serve, measured the same
    for carry in (True, False):
        with torch.no_grad():
            whole = train.taught(model, target, got, reading, carry)
            picked = train.taught(model, target, got, reading, carry, only=only)
        assert [
            x.tolist() for x in train.measured(picked, got.labels, got.on_path, only)
        ] == [x.tolist() for x in train.measured(whole, got.labels, got.on_path)]
    # nothing to score: a loss of 0 that still steps back
    none = torch.zeros_like(only)
    empty = train.taught(model, target, got, reading, only=none)
    nothing = train.loss(
        empty, train.chosen(got.labels, none), train.chosen(got.weights, none)
    )
    assert nothing.item() == 0.0
    gradients(nothing)


def test_a_few_steps_lower_the_loss_of_a_batch():
    model, target, reading = made("dneo")
    values, tokens = drawn(16)
    held = stream(values, tokens, first=0, prompt=3)
    written = np.asarray(tokens, np.int64)[::-1].copy()
    got = train.batch(held, written, np.arange(3, 15), reading, 4, decay=4.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        optimiser.zero_grad()
        value = train.loss(
            train.taught(model, target, got, reading), got.labels, got.weights
        )
        value.backward()
        optimiser.step()
        losses.append(value.item())
    assert losses[-1] < 0.5 * losses[0]
    # the target's embedding and head are borrowed: training does not move them
    assert not target.embed.requires_grad and not target.head.requires_grad


def test_the_learning_rate_warms_up_then_falls_to_its_end_on_a_cosine():
    def at(step):
        return train.rate(step, steps=110, peak=2e-4, end=2e-5, warmup=10)

    assert at(0) == pytest.approx(2e-5) and at(9) == pytest.approx(2e-4)
    # halfway down the cosine, the middle of peak and end
    assert at(59) == pytest.approx((2e-4 + 2e-5) / 2, rel=1e-2)
    assert at(109) == pytest.approx(2e-5)
    assert all(at(s) >= at(s + 1) for s in range(9, 109))
    with pytest.raises(ValueError, match="warmup 120 is not below steps 110"):
        train.rate(0, steps=110, peak=2e-4, end=2e-5, warmup=120)


@pytest.mark.parametrize("best", [0, 5])
def test_a_trained_drafter_is_kept_and_scored_as_it_was_trained(tmp_path, best):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    path = tmp_path / "b.safetensors"
    rng = np.random.default_rng(15)
    bridge.write(
        path, rng.standard_normal((8, 8)).astype(np.float32), [0.25, 0.75], TAKEN
    )
    model, target, reading, _, _ = score.loaded(
        drafted, checkpoint_folder, "cpu", False, path, dneo_mask=True, dneo_head=best
    )
    with torch.no_grad():  # as if trained: every part moved
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    kept = tmp_path / "step-1"
    train.save(model, kept, drafted / "config.json", reading, step=1, lr=2e-4)
    back, _, back_reading, lines, unread = score.loaded(
        kept, checkpoint_folder, "cpu", False
    )
    assert unread == [] and back_reading == reading
    assert lines[1] == (
        "the drafter: trained to step 1; the entrance: DNeo's weighted sum of taps"
        " (5, 9), bridged; DNeo's mask; DNeo's head over "
        + ("the target head's best 5" if best else "the whole vocabulary")
    )
    assert back.head.best == best
    for name, value in model.state_dict().items():
        assert torch.equal(back.state_dict()[name], value), name
    values, tokens = drawn(16)
    held = stream(values, tokens, first=0, prompt=3)
    mine = score.proposals(model, target, held, reading)
    again = score.proposals(back, target, held, back_reading)
    assert np.array_equal(mine.drafts, again.drafts)
    with pytest.raises(ValueError, match="a trained drafter carries its own form"):
        score.loaded(kept, checkpoint_folder, "cpu", False, path)
    with pytest.raises(FileExistsError):
        train.save(model, kept, drafted / "config.json", reading, step=1)


def test_a_measure_counts_hits_by_position_and_the_drafts_landed_on_the_path():
    # three anchors, three drafts; the head's best token at each position
    values = torch.full((3, 3, 4), -1.0)
    best = torch.tensor([[1, 2, 3], [1, 0, 3], [2, 2, 2]])
    values.scatter_(-1, best[..., None], 1.0)
    labels = torch.tensor([[1, 2, 0], [1, 2, 3], [2, 2, -1]])
    on_path = torch.tensor([3, 3, 1])  # the third anchor's labels leave the path at 1
    hits, counted, landed = train.measured(
        drafter.Taught(values, None), labels, on_path
    )
    # by position, of the labels on the path: 3 of 3, 1 of 2, 1 of 2
    assert hits.tolist() == [3, 1, 1] and counted.tolist() == [3, 2, 2]
    # in a row from the first: 2, 1, 1 (the third stops at the path's end)
    assert landed.tolist() == [2, 1, 1]
    among = drafter.Taught(
        values[..., [3, 2, 1, 0]], torch.tensor([3, 2, 1, 0]).expand(3, 3, 4)
    )
    assert [x.tolist() for x in train.measured(among, labels, on_path)] == [
        [3, 1, 1], [3, 2, 2], [2, 1, 1]
    ]  # fmt: skip


def test_a_run_trains_measures_and_keeps_a_drafter_score_reads(tmp_path):
    from open_dneo import feed, trainer

    from .test_score import on_disk

    drafted, checkpoint_folder, _ = folders(tmp_path)
    model, target, reading, _, _ = score.loaded(
        drafted, checkpoint_folder, "cpu", False
    )
    rng = np.random.default_rng(16)
    for n in range(4):  # plain text the target chose all of
        tokens = rng.integers(0, 20, 30)
        values = rng.standard_normal((30, 3, 8)).astype(np.float32)
        on_disk(tmp_path / "rec" / "5-0", values, tokens, first=0, prompt=30, n=n)
        choices = tmp_path / "next" / "5-0" / f"{n}.next"
        choices.parent.mkdir(parents=True, exist_ok=True)
        np.roll(tokens, -1).astype("<i4").tofile(choices)
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next")
    kept_streams, held = feed.held(fed, every=4)
    settings = trainer.Settings(
        steps=6, warmup=2, anchors=8, span=16, held_anchors=8, log_every=2,
        measure_every=3, keep_every=4, peak=1e-2,
    )  # fmt: skip
    said: list[str] = []
    trainer.run(
        model, target, reading, drafted / "config.json", kept_streams, held,
        tmp_path / "out", settings, said.append,
    )  # fmt: skip
    assert said[0] == "run: 3 streams, 1 held, parts {'': 1.0}"
    assert said[1].startswith("weights, held part '': 8 anchors, ")
    assert said[2].startswith("step 0, held: 1 streams, 8 anchors; loss ")
    assert [line.split(":")[0] for line in said[3:]] == [
        "step 2", "step 3, held", "step 4", "step 6", "step 6, held",
    ]  # fmt: skip
    first, last = (
        float(line.split("loss ")[1].split(";")[0]) for line in (said[2], said[-1])
    )
    assert last < first  # trained on the same text it is measured on, of its kind
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["step-4", "step-6"]
    back, _, back_reading, _, _ = score.loaded(
        tmp_path / "out" / "step-6", checkpoint_folder, "cpu", False
    )
    for name, value in model.state_dict().items():
        assert torch.equal(back.state_dict()[name], value), name


def two_parts(tmp_path, streams=4):
    """Streams of 30 rows in parts a (even, the target chose the text's tokens) and b
    (odd, its choices at random), with their choices and a listing of their parts."""
    from open_dneo import feed

    from .test_score import on_disk

    rng = np.random.default_rng(20)
    listing = []
    for n in range(streams):
        tokens = rng.integers(0, 20, 30)
        values = rng.standard_normal((30, 3, 8)).astype(np.float32)
        on_disk(tmp_path / "rec" / "5-0", values, tokens, first=0, prompt=30, n=n)
        choices = tmp_path / "next" / "5-0" / f"{n}.next"
        choices.parent.mkdir(parents=True, exist_ok=True)
        chose = np.roll(tokens, -1) if n % 2 == 0 else rng.integers(0, 20, 30)
        chose.astype("<i4").tofile(choices)
        listing.append(f"5-0/{n}\t{'ab'[n % 2]}\n")
    (tmp_path / "parts.tsv").write_text("".join(listing), encoding="utf-8")
    return feed.streams([tmp_path / "rec"], tmp_path / "next", [tmp_path / "parts.tsv"])


def test_the_weights_of_each_parts_held_windows_are_said_by_position(tmp_path):
    from open_dneo import trainer

    fed = two_parts(tmp_path)
    settings = trainer.Settings(steps=2, warmup=1, held_anchors=8, span=16)
    windows = trainer.held_windows(fed, settings)
    lines = trainer.profile(windows, settings, block=4)
    want = []
    for part in "ab":
        summed, anchors, on_path = np.zeros(3), 0, 0
        for fed_one, rows in windows:
            if fed_one.part != part:
                continue
            opened = fed_one.open()
            taught = train.blocks(
                opened.stream.tok, opened.written, rows, 3, opened.room[rows]
            )
            summed += train.weights(taught, 4.0).sum(0)
            anchors, on_path = anchors + len(rows), on_path + int(taught.on_path.sum())
        want.append(
            f"weights, held part {part!r}: {anchors} anchors, {on_path / anchors:.2f}"
            " drafts an anchor on the path; by position "
            + ", ".join(f"{100 * w / summed.sum():.1f}%" for w in summed)
        )
    assert lines == want
    # the target's own text keeps its labels deep; choices at random leave it at once
    first = [float(line.split("by position ")[1].split("%")[0]) for line in lines]
    assert first[0] < first[1]


def test_held_streams_are_capped_a_part_where_the_settings_say(tmp_path):
    from open_dneo import trainer

    fed = two_parts(tmp_path, streams=6)
    every = trainer.Settings(steps=2, warmup=1, held_anchors=8, span=16)
    assert [f.name for f, _ in trainer.held_windows(fed, every)] == [
        f.name for f in fed
    ]
    capped = dataclasses.replace(every, held_streams=2)
    windows = trainer.held_windows(fed, capped)
    assert sorted(f.part for f, _ in windows) == ["a", "a", "b", "b"]
    assert [f.name for f, _ in windows] == sorted(f.name for f, _ in windows)
    assert [f.name for f, _ in trainer.held_windows(fed, capped)] == [
        f.name for f, _ in windows
    ]  # drawn with the run's seed
    with pytest.raises(ValueError, match="held_streams is 0"):
        trainer.held_windows(fed, dataclasses.replace(every, held_streams=0))


def test_a_part_of_the_drafter_learns_at_its_own_factor_of_the_rate(tmp_path):
    from open_dneo import feed, trainer

    fed = two_parts(tmp_path)
    kept_streams, held = feed.held(fed, every=4)
    drafted, checkpoint_folder, _ = folders(tmp_path)
    model, target, reading, _, _ = score.loaded(
        drafted, checkpoint_folder, "cpu", False
    )
    (one,) = trainer.groups(model, None)  # by default one rate for every part
    assert one["factor"] == 1.0 and len(one["params"]) == len(list(model.parameters()))
    groups = trainer.groups(model, {"layers": 0.0, "head": 2.0})
    assert {g["part"]: g["factor"] for g in groups} == {
        "entrance": 1.0, "head": 2.0, "layers": 0.0, "norm": 1.0
    }  # fmt: skip
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    settings = trainer.Settings(
        steps=2, warmup=1, anchors=8, span=16, held_anchors=8, measure_every=10,
        keep_every=10, peak=1e-2, rates={"layers": 0.0},
    )  # fmt: skip
    trainer.run(
        model, target, reading, drafted / "config.json", kept_streams, held,
        tmp_path / "out", settings, lambda line: None,
    )  # fmt: skip
    moved = {
        n.split(".")[0]
        for n, p in model.named_parameters()
        if not torch.equal(p.detach(), before[n])
    }
    assert "layers" not in moved and {"entrance", "head"} <= moved
    for wrong, said in (
        ({"layer": 0.1}, r"the drafter's parts are \['entrance', 'head', 'layers',"),
        ({"head": -1.0}, "the factor of head is -1.0: 0 or more"),
    ):
        with pytest.raises(ValueError, match=said):
            trainer.groups(model, wrong)


def test_the_measure_is_said_over_every_part_and_part_by_part(tmp_path):
    from open_dneo import trainer

    fed = two_parts(tmp_path)
    drafted, checkpoint_folder, _ = folders(tmp_path)
    model, target, reading, _, _ = score.loaded(
        drafted, checkpoint_folder, "cpu", False
    )
    settings = trainer.Settings(steps=2, warmup=1, held_anchors=8, span=16)
    windows = trainer.held_windows(fed, settings)
    lines = trainer.measure(model, target, reading, windows, settings)
    assert len(lines) == 3
    assert lines[0].startswith("held: 4 streams, 32 anchors; loss ")
    a, b = ([w for w in windows if w[0].part == p] for p in "ab")
    assert lines[1:] == [
        "held part 'a': " + trainer.measure(model, target, reading, a, settings)[0][6:],
        "held part 'b': " + trainer.measure(model, target, reading, b, settings)[0][6:],
    ]
    # one part: the line over every part alone
    assert len(trainer.measure(model, target, reading, a, settings)) == 1


def test_the_train_command_runs_from_folders_to_a_kept_drafter(tmp_path, capfdbinary):
    from open_dneo import __main__ as cli

    from .test_score import on_disk

    drafted, checkpoint_folder, _ = folders(tmp_path)
    rng = np.random.default_rng(17)
    for n in range(4):
        tokens = rng.integers(0, 20, 20)
        values = rng.standard_normal((20, 3, 8)).astype(np.float32)
        on_disk(tmp_path / "rec" / "5-0", values, tokens, first=0, prompt=20, n=n)
        choices = tmp_path / "next" / "5-0" / f"{n}.next"
        choices.parent.mkdir(parents=True, exist_ok=True)
        np.roll(tokens, -1).astype("<i4").tofile(choices)
    asked = [
        "train", str(tmp_path / "rec"), "--next", str(tmp_path / "next"),
        "--drafter", str(drafted), "--target", str(checkpoint_folder),
        "--out", str(tmp_path / "out"), "--steps", "2", "--warmup", "1",
        "--anchors", "4", "--span", "8", "--held-anchors", "4", "--held-every", "2",
    ]  # fmt: skip
    assert cli.main(asked) == 0
    said = capfdbinary.readouterr().out.decode("utf-8").splitlines()
    assert said[0] == "weights as stored" and said[1].startswith(
        "run: 2 streams, 2 held"
    )
    assert said[-1] == f"== train finished: 2 steps, kept under {tmp_path / 'out'}"
    assert (tmp_path / "out" / "step-2" / "train.json").exists()


def test_the_measure_counts_hits_among_the_candidates_the_head_would_serve(
    tmp_path, monkeypatch
):
    from open_dneo import feed, trainer

    from .test_score import on_disk

    drafted, checkpoint_folder, _ = folders(tmp_path)
    model, target, reading, _, _ = score.loaded(
        drafted, checkpoint_folder, "cpu", False
    )
    rng = np.random.default_rng(18)
    tokens = rng.integers(0, 20, 24)
    values = rng.standard_normal((24, 3, 8)).astype(np.float32)
    on_disk(tmp_path / "rec" / "5-0", values, tokens, first=0, prompt=24)
    choices = tmp_path / "next" / "5-0" / "0.next"
    choices.parent.mkdir(parents=True)
    rng.integers(0, 20, 24).astype("<i4").tofile(choices)  # labels at random
    (fed,) = feed.streams([tmp_path / "rec"], tmp_path / "next")
    settings = trainer.Settings(steps=2, warmup=1, held_anchors=12, span=24)
    windows = trainer.held_windows([fed], settings)
    seen, real = [], train.measured
    monkeypatch.setattr(
        train, "measured", lambda taught, *a: seen.append(taught) or real(taught, *a)
    )
    trainer.measure(model, target, reading, windows, settings)
    ((_, rows),) = windows
    opened = fed.open()
    got = train.batch(
        opened.stream, opened.written, rows, reading, 4, 4.0, cap=opened.room[rows]
    )
    plain = train.taught(model, target, got, reading, carry=False)
    # the head's own best k, not a label put among them, where the drafts count
    only = train.counted(got)
    assert torch.equal(seen[0].candidates, train.chosen(plain.candidates, only))
    assert not torch.equal(
        train.taught(model, target, got, reading).candidates, plain.candidates
    )
    # a label outside them is a miss
    outside = ~(plain.candidates == got.labels[..., None]).any(-1) & (got.labels >= 0)
    assert outside.any()
    hits, _, _ = real(plain, got.labels, got.on_path)
    best = plain.candidates.gather(-1, plain.values.argmax(-1, keepdim=True))[..., 0]
    assert not ((best == got.labels) & outside).any()


def test_a_run_resumed_from_a_kept_step_ends_where_the_whole_run_ends(tmp_path):
    from open_dneo import feed, trainer

    from .test_score import on_disk

    drafted, checkpoint_folder, _ = folders(tmp_path)
    rng = np.random.default_rng(19)
    for n in range(4):
        tokens = rng.integers(0, 20, 30)
        values = rng.standard_normal((30, 3, 8)).astype(np.float32)
        on_disk(tmp_path / "rec" / "5-0", values, tokens, first=0, prompt=30, n=n)
        choices = tmp_path / "next" / "5-0" / f"{n}.next"
        choices.parent.mkdir(parents=True, exist_ok=True)
        np.roll(tokens, -1).astype("<i4").tofile(choices)
    fed = feed.streams([tmp_path / "rec"], tmp_path / "next")
    kept_streams, held = feed.held(fed, every=4)
    settings = trainer.Settings(
        steps=4, warmup=1, anchors=8, span=16, held_anchors=8, log_every=1,
        measure_every=10, keep_every=2, peak=1e-2,
    )  # fmt: skip

    def run(start, out, resume=None):
        model, target, reading, _, _ = score.loaded(
            start, checkpoint_folder, "cpu", False
        )
        trainer.run(
            model, target, reading, drafted / "config.json", kept_streams, held, out,
            settings, lambda line: None, resume=resume,
        )  # fmt: skip

    run(drafted, tmp_path / "whole")
    half = tmp_path / "whole" / "step-2"
    assert (half / "optimiser.pt").exists() and (half / "draws.json").exists()
    assert b"\r" not in (half / "train.json").read_bytes()
    run(half, tmp_path / "again", resume=half)
    assert sorted(p.name for p in (tmp_path / "again").iterdir()) == ["step-4"]
    whole = weights.read(tmp_path / "whole" / "step-4" / "model.safetensors")
    again = weights.read(tmp_path / "again" / "step-4" / "model.safetensors")
    for name, value in whole.items():
        assert torch.equal(again[name], value), name
    other = dataclasses.replace(settings, peak=2e-2)
    model, target, reading, _, _ = score.loaded(half, checkpoint_folder, "cpu", False)
    with pytest.raises(ValueError, match="step-2 was kept by a run of other settings"):
        trainer.run(
            model, target, reading, drafted / "config.json", kept_streams, held,
            tmp_path / "x", other, lambda line: None, resume=half,
        )  # fmt: skip
