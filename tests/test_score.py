"""A drafter's proposals along a stream, against blocks passed one at a time by hand.

The drafters are small and every parameter is random. What the passes compute is
tests/test_drafter.py's subject; here it is which rows, tokens and positions a
stream gives them. They compute in float64, where passing blocks together or one
at a time differs by 2e-15 at most (in float32 by 7e-6, random weights being what
they are), so the comparisons are tight.
"""

import json
import struct

import numpy as np
import pytest
import torch

from open_dneo import drafter, recording, rounds, score, served, weights

from .test_drafter import receipt
from .test_hub import safetensors_blob

HIDDEN, VOCAB, MASK = 8, 23, 22
STORED = (9, 2, 5)  # the taps the streams here hold, in their order
TAKEN = (5, 9)  # the taps the readings here take: the stored rows' columns 2 and 0


def made(
    kind: str, hidden: int = HIDDEN
) -> tuple[drafter.Drafter, score.Target, score.Reading]:
    """A small drafter of `kind` with random parameters, a target and its reading."""
    sizes = drafter.Shape(
        hidden=hidden, heads=4, kv_heads=2, head_dim=max(4, hidden // 4),
        ffn=3 * hidden // 2, layers=2, block=4, group=2, eps=1e-5, theta=10000.0,
        vocab=VOCAB, rank=3, taps=len(TAKEN), top_k=5, folded=6,
    )  # fmt: skip
    if kind == "dflash2":
        model = drafter.Drafter.dflash2(sizes)
        reading = score.Reading(TAKEN, MASK, whole_block=True, window=5)
    else:
        model = drafter.Drafter.dneo(sizes)
        reading = score.Reading(TAKEN, MASK, whole_block=False, window=None)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    target = score.Target(
        embed=torch.randn(VOCAB, hidden, generator=generator).double(),
        head=torch.randn(VOCAB, hidden, generator=generator).double(),
    )
    return model.double(), target, reading


def drawn(rows: int, hidden: int = HIDDEN) -> tuple[np.ndarray, np.ndarray]:
    """Random rows [rows, stored taps, hidden] and a token for each."""
    generator = np.random.default_rng(11)
    values = generator.standard_normal((rows, len(STORED), hidden)).astype(np.float32)
    return values, generator.integers(0, MASK, rows)


def stream(values, tokens, *, first: int, prompt: int, taps=STORED) -> recording.Stream:
    """A stream in memory: `values` at positions `first`.. of a `prompt`-token prompt."""
    rows = len(values)
    positions = first + np.arange(rows, dtype=np.uint32)
    summary = recording.Summary(
        rows=rows,
        first_row=first,
        prompt_len=prompt,
        taps=tuple(taps),
        row_bytes=values[0].size * 4 if rows else 0,
        dtype="f32",
        top_k=0,
        gap=None,
    )
    return recording.Stream(
        summary,
        tok=np.asarray(tokens, np.uint32),
        pos=positions,
        kind=(positions >= prompt).astype(np.uint8),
        st=np.ascontiguousarray(values, "<f4").reshape(rows, -1).view(np.uint8),
        top_ids=np.empty((0, 0), np.uint32),
        top_logprobs=np.empty((0, 0), np.float32),
    )


def alone(model, target, reading, values, positions, anchor, token, temperature=1.0):
    """One block passed by hand: the drafts and chances after `token` at `anchor`.

    `values` [s, taken taps, HIDDEN] are every row of the stream at `positions`,
    those at and after the anchor too: `drafter.visible` hides them.
    """
    block = model.shape.block
    ids = torch.full((1, block), reading.mask_token)
    ids[0, 0] = token
    anchors, context_positions = torch.tensor([[anchor]]), positions[None]
    mask = drafter.visible(
        anchors,
        context_positions,
        block,
        whole_block=reading.whole_block,
        window=reading.window,
    )
    with torch.no_grad():
        hidden = model(
            target.embed[ids][None],
            values[None].double(),
            anchors,
            context_positions,
            mask,
        )
        states = hidden[0, :, 1:]
        scores = states @ target.head.T
        if target.bf16_scores:
            scores = scores.to(torch.bfloat16).double()
        path, _, chances = model.head(states, scores, ids[:, 0], temperature)
    return path[0].tolist(), chances[0].numpy()


@pytest.mark.parametrize("kind", ["dflash2", "dneo"])
@pytest.mark.parametrize("at_once", [1, 4, 100])
@pytest.mark.parametrize(("first", "prompt"), [(0, 2), (4, 7), (7, 7)])
def test_proposals_are_each_reply_rows_block_passed_alone(kind, at_once, first, prompt):
    model, target, reading = made(kind)
    values, tokens = drawn(12)
    recorded = stream(values, tokens, first=first, prompt=prompt)
    found = score.proposals(model, target, recorded, reading, at_once=at_once)

    replies = range(prompt - first, 12)  # (7, 7): no row before the first anchor
    assert found.first_row == first
    assert found.positions.tolist() == [first + i for i in replies]
    assert found.tokens.tolist() == [int(tokens[i]) for i in replies]
    assert found.drafts.shape == found.chances.shape == (len(replies), 3)
    taken = torch.from_numpy(values[:, [2, 0]])
    positions = first + torch.arange(12)
    for n, i in enumerate(replies):
        path, chances = alone(
            model, target, reading, taken, positions, first + i, int(tokens[i])
        )
        assert found.drafts[n].tolist() == path
        np.testing.assert_allclose(found.chances[n], chances, rtol=1e-9, atol=1e-12)


def test_the_chances_are_at_the_temperature_asked_for():
    model, target, reading = made("dflash2")
    values, tokens = drawn(6)
    recorded = stream(values, tokens, first=0, prompt=5)
    found = score.proposals(model, target, recorded, reading, temperature=3.0)
    taken, positions = torch.from_numpy(values[:, [2, 0]]), torch.arange(6)
    cool = alone(model, target, reading, taken, positions, 5, int(tokens[5]))[1]
    warm = alone(model, target, reading, taken, positions, 5, int(tokens[5]), 3.0)[1]
    assert np.abs(warm - cool).max() > 1e-3
    np.testing.assert_allclose(found.chances[0], warm, rtol=1e-9, atol=1e-12)


def test_scores_kept_in_bfloat16_are_rounded_before_the_head_reads_them():
    model, target, reading = made("dflash2")
    values, tokens = drawn(8)
    recorded = stream(values, tokens, first=0, prompt=3)
    plain = score.proposals(model, target, recorded, reading)
    kept = score.Target(target.embed, target.head, bf16_scores=True)
    found = score.proposals(model, kept, recorded, reading)
    assert np.abs(found.chances - plain.chances).max() > 1e-4
    taken, positions = torch.from_numpy(values[:, [2, 0]]), torch.arange(8)
    for n, i in enumerate(range(3, 8)):
        path, chances = alone(model, kept, reading, taken, positions, i, int(tokens[i]))
        assert found.drafts[n].tolist() == path
        np.testing.assert_allclose(found.chances[n], chances, rtol=1e-9, atol=1e-12)


class Watched(torch.Tensor):
    """A tensor that notes the shapes of every product it is taken into."""

    products: list[list[tuple[int, ...]]] = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", "") in ("matmul", "__matmul__", "__rmatmul__"):
            given = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
            cls.products.append(given)
        return super().__torch_function__(func, types, args, kwargs)


@pytest.mark.parametrize("kind", ["dflash2", "dneo"])
def test_the_heads_scores_are_one_product_of_two_matrices(kind):
    # A product of three dimensions can take a path that copies the head once an
    # anchor: for 32 anchors and a head of 2.5 GB that asked an aarch64 torch for 81 GB.
    model, target, reading = made(kind)
    values, tokens = drawn(12)
    recorded = stream(values, tokens, first=0, prompt=2)
    plain = score.proposals(model, target, recorded, reading, at_once=4)
    Watched.products = []
    watched = score.Target(target.embed, target.head.as_subclass(Watched))
    laid_out, forward = [], model.head.forward

    def noted(hidden, *rest):
        laid_out.append(hidden.is_contiguous())
        return forward(hidden, *rest)

    model.head.forward = noted
    found = score.proposals(model, watched, recorded, reading, at_once=4)

    assert found.drafts.tolist() == plain.drafts.tolist()
    np.testing.assert_allclose(found.chances, plain.chances, rtol=1e-12, atol=0)
    assert Watched.products == [[(4 * 3, HIDDEN), (HIDDEN, VOCAB)]] * 2 + [
        [(2 * 3, HIDDEN), (HIDDEN, VOCAB)]
    ]  # 10 anchors, 4 a pass, 3 drafts each
    assert laid_out == [True, True, True]  # the head's own matrix folds the same way


@pytest.mark.parametrize("at_once", [1, 100])
def test_a_block_sees_the_rows_before_its_anchor_inside_the_window(at_once):
    model, target, reading = made("dflash2")
    values, tokens = drawn(14)
    assert reading.window == 5

    def chances(anchor: int, changed: int | None = None) -> np.ndarray:
        rows = values.copy()
        if changed is not None:
            rows[changed] += (
                3.0  # the row at position `changed`: the stream starts at 0
            )
        recorded = stream(rows, tokens, first=0, prompt=2)
        found = score.proposals(model, target, recorded, reading, at_once=at_once)
        return found.chances[anchor - 2]

    plain = chances(10)
    # position 6 is the oldest the anchor's own row sees: 10 - 6 is inside the window
    for seen in (9, 6):
        assert np.abs(chances(10, seen) - plain).max() > 1e-3
    for unseen in (5, 10, 11):
        np.testing.assert_array_equal(chances(10, unseen), plain)


def test_a_tap_the_reading_does_not_take_changes_nothing():
    model, target, reading = made("dneo")
    values, tokens = drawn(8)
    other = values.copy()
    other[:, 1] += 3.0  # tap 2, which the reading leaves
    plain = score.proposals(
        model, target, stream(values, tokens, first=0, prompt=3), reading
    )
    same = score.proposals(
        model, target, stream(other, tokens, first=0, prompt=3), reading
    )
    np.testing.assert_array_equal(same.drafts, plain.drafts)
    np.testing.assert_array_equal(same.chances, plain.chances)


def test_a_stream_without_reply_rows_has_no_anchor():
    model, target, reading = made("dflash2")
    values, tokens = drawn(3)
    found = score.proposals(
        model, target, stream(values, tokens, first=0, prompt=3), reading
    )
    assert found.positions.shape == found.tokens.shape == (0,)
    assert found.drafts.shape == found.chances.shape == (0, 3)


def test_a_stream_the_reading_cannot_take_is_refused():
    model, target, reading = made("dflash2")
    values, tokens = drawn(6)
    good = stream(values, tokens, first=0, prompt=3)

    with pytest.raises(ValueError, match=r"taps \[9, 2\], not \[5\]"):
        short = stream(values[:, :2], tokens, first=0, prompt=3, taps=(9, 2))
        score.proposals(model, target, short, reading)
    with pytest.raises(ValueError, match="names 3 taps, and the model's entrance"):
        three = score.Reading((5, 9, 2), MASK, whole_block=True, window=5)
        score.proposals(model, target, good, three)
    with pytest.raises(ValueError, match="rows are 4 wide, the model's 8"):
        narrow = stream(values[:, :, :4], tokens, first=0, prompt=3)
        score.proposals(model, target, narrow, reading)
    with pytest.raises(ValueError, match="not at consecutive positions from 0"):
        good.pos[4] = 9
        score.proposals(model, target, good, reading)
    good.pos[4] = 4
    with pytest.raises(ValueError, match="at_once is 0"):
        score.proposals(model, target, good, reading, at_once=0)
    assert len(score.proposals(model, target, good, reading).positions) == 3


def found(drafts, chances) -> score.Proposals:
    """Proposals at the anchors 10, 11, .. of a stream that starts at position 0."""
    tokens = 10 + np.arange(len(drafts))
    return score.Proposals(
        0, 5 + np.arange(len(drafts)), tokens, np.array(drafts), np.array(chances)
    )


def test_steps_walk_a_reply_over_the_proposals():
    sure = [[1.0, 1.0]] * 5
    held = found([[11, 12], [12, 0], [13, 14], [0, 0], [15, 0]], sure)
    # two land and the target adds 13; nothing lands after 13 and it adds 14
    assert score.steps(held.tokens, held) == [3, 1]
    # the reply's last token, which the target never read: one more round gives it
    assert score.steps([10, 11, 12, 13, 14, 15], held) == [3, 1, 1]
    assert score.steps([10, 11, 12], held) == [2]  # a reply cut short of the stream

    doubted = found(held.drafts, [[1.0, 0.5]] + sure[1:])
    assert score.steps(held.tokens, doubted) == [2, 2]  # 12 was not verified
    assert score.steps(held.tokens, doubted, cut=0.5) == [3, 1]

    # a doubt at the third anchor is that round's alone: 13 lands, 14 is not verified
    later = found(
        [[11, 0], [0, 0], [13, 14], [0, 0], [15, 0]],
        sure[:2] + [[1.0, 0.5]] + sure[3:],
    )
    assert score.steps([10, 11, 12, 13, 14, 15], later) == [2, 2, 1]
    assert score.steps([10, 11, 12, 13, 14, 15], later, cut=0.5) == [2, 3]


def test_steps_refuse_a_reply_that_is_not_the_streams():
    held = found([[0, 0]] * 3, [[1.0, 1.0]] * 3)
    with pytest.raises(ValueError, match="reply token 1 is 99, and the stream's is 11"):
        score.steps([10, 99, 12], held)
    with pytest.raises(ValueError, match="5 tokens, and the stream rows for 3"):
        score.steps([10, 11, 12, 13, 14], held)


def never(reply, drafts=7) -> score.Proposals:
    """Proposals after a 13-token prompt that are sure of `drafts` drafts and never land."""
    rows = len(reply)
    return score.Proposals(
        0,
        13 + np.arange(rows),
        np.array(reply),
        np.full((rows, drafts), -1),
        np.ones((rows, drafts)),
    )


def test_lanes_walk_a_streams_reply_after_its_prompt_as_serving_takes_it():
    prompt, reply = [*range(1, 13), 50], [*range(1, 13), 60, 61]
    values, _ = drawn(len(prompt) + len(reply))
    held = stream(values, prompt + reply, first=0, prompt=len(prompt))

    def wrong(at):
        return [-1] * 7, [1.0] * 7

    got = score.lanes(held, never(reply))
    assert got == rounds.served(prompt, reply, wrong)
    assert [given.kind for given in got].count("copy") == 2
    # the limit of new tokens reaches the walk: no room for the second copy
    bounded = score.lanes(held, never(reply), max_new=14)
    assert bounded == rounds.served(prompt, reply, wrong, max_new=14) != got


def test_lanes_ask_for_as_many_drafts_as_the_drafters_block_holds():
    prompt, reply = [*range(1, 13), 50], list(range(100, 130))
    values, _ = drawn(len(prompt) + len(reply))
    held = stream(values, prompt + reply, first=0, prompt=len(prompt))
    assert [given.offered for given in score.lanes(held, never(reply, 9))[:2]] == [4, 9]
    assert [given.offered for given in score.lanes(held, never(reply, 3))[:2]] == [3, 3]
    # the first round's depth is a setting: serving chooses it from measured times
    shallow = score.lanes(held, never(reply, 9), first_depth=1)
    assert [given.offered for given in shallow[:2]] == [1, 9]


def test_lanes_give_each_round_its_own_anchors_drafts():
    sure = [[1.0, 1.0]] * 6
    held = found([[11, 12], [12, 0], [13, 14], [0, 0], [15, 0], [0, 0]], sure)
    values, _ = drawn(3 + 6)
    recorded = stream(values, [1, 2, 3, *held.tokens.tolist()], first=0, prompt=3)
    # two land and the target adds 13; nothing lands after 13; 15 lands and ends the reply
    assert score.lanes(recorded, held) == [
        rounds.Round("head", 2, 2, 3),
        rounds.Round("head", 2, 0, 1),
        rounds.Round("head", 2, 1, 1),
    ]


def alone_said(own) -> str:
    """Rounds of the drafter alone as a report words them."""
    return (
        f"{len(own)} rounds give {sum(own)} tokens, {sum(own) / len(own):.3f} a round"
    )


def taken_said(lane) -> str:
    """Rounds as serving takes them as a report words them."""
    copied = [given.kind for given in lane].count("copy")
    return f"{alone_said([given.tokens for given in lane])}, {copied} of them copies"


def test_dflash2s_reading_is_the_published_checkpoints():
    config, _ = receipt("glm-dflash2.shapes.txt")
    # the checkpoint's layers 5, 14, 24, 33 and 42, counted from 0, as recorded taps
    assert score.Reading.dflash2(config) == score.Reading(
        taps=(6, 15, 25, 34, 43), mask_token=154856, whole_block=True, window=2048
    )
    sizes = drafter.Shape.dflash2(config)
    assert (sizes.block, sizes.taps, sizes.top_k) == (8, 5, 16)


def test_a_dflash2_reading_refuses_what_is_not_implemented():
    config, _ = receipt("glm-dflash2.shapes.txt")
    whole = ["full_attention"] * 5
    assert score.Reading.dflash2(config | {"layer_types": whole}).window is None
    mixed = ["full_attention"] + ["sliding_attention"] * 4
    scaled = config["dflash_config"] | {"input_embedding_scale": 2.0}
    for wrong, said in (
        ({"is_causal": True}, "see the whole block the same way"),
        ({"layer_types": mixed}, "see the whole block the same way"),
        ({"dflash_config": scaled}, "without a scale"),
        ({"input_embedding_scale": 2.0}, "without a scale"),
    ):
        with pytest.raises(ValueError, match=f"only .* {said} are implemented"):
            score.Reading.dflash2(config | wrong)


def checkpoint(folder, head):
    """A two-token checkpoint, a file a tensor: an embedding, and `head`'s tensors."""
    row = struct.pack("<2H", 0x3F80, 0xC000) * 16  # 1.0 and -2.0, 16 times
    tensors = {weights.EMBED: ("BF16", [2, 32], row + row)} | head
    files = {name: f"{n}.safetensors" for n, name in enumerate(tensors)}
    for name, file in files.items():
        (folder / file).write_bytes(safetensors_blob({name: tensors[name]}))
    index = {"metadata": {}, "weight_map": files}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    return folder


def test_a_target_is_read_from_its_checkpoints_files(tmp_path):
    plain = struct.pack("<64f", *[0.5] * 32, *[-3.0] * 32)
    target = score.Target.read(
        checkpoint(tmp_path, {weights.HEAD: ("F32", [2, 32], plain)})
    )
    assert target.embed.dtype == target.head.dtype == torch.float32
    assert target.embed.tolist() == [[1.0, -2.0] * 16] * 2
    assert target.head.tolist() == [[0.5] * 32, [-3.0] * 32]


def test_an_mxfp8_head_is_read_with_the_scales_beside_it(tmp_path):
    head = {
        # 1.5 in a group scaled by 4, and -1 in a group scaled by a half
        weights.HEAD: ("F8_E4M3", [2, 32], bytes([0x3C] * 32 + [0xB8] * 32)),
        weights.HEAD_SCALES: ("U8", [2, 1], bytes([129, 126])),
    }
    target = score.Target.read(checkpoint(tmp_path, head))
    assert target.head.dtype == torch.float32
    assert target.head.tolist() == [[6.0] * 32, [-0.5] * 32]


def on_disk(folder, values, tokens, *, first, prompt, n=0, taps=STORED):
    """The stream `stream` makes, as the files a recorder writes."""
    held = stream(values, tokens, first=first, prompt=prompt, taps=taps)
    fields = {
        "rows": len(values),
        "first_row": first,
        "prompt_len": prompt,
        "taps": list(taps),
        "row_bytes": held.summary.row_bytes,
        "dtype": "f32",
        "top_k": 0,
        "gap": None,
    }
    folder.mkdir(parents=True, exist_ok=True)
    for ext, data in ((".tok", held.tok), (".pos", held.pos), (".kind", held.kind)):
        (folder / f"{n}{ext}").write_bytes(data.tobytes())
    (folder / f"{n}.st").write_bytes(held.st.tobytes())
    (folder / f"{n}.json").write_text(json.dumps(fields))


def folders(tmp_path, extra=None, hidden=HIDDEN):
    """A small DFlash2 drafter and its target as folders, and both in memory."""
    model, target, reading = made("dflash2", hidden)
    model, sizes = model.float(), model.shape
    target = score.Target(target.embed.float(), target.head.float())
    config = {
        "hidden_size": sizes.hidden,
        "num_attention_heads": sizes.heads,
        "num_key_value_heads": sizes.kv_heads,
        "head_dim": sizes.head_dim,
        "intermediate_size": sizes.ffn,
        "num_hidden_layers": sizes.layers,
        "rms_norm_eps": sizes.eps,
        "rope_parameters": {"rope_theta": sizes.theta},
        "vocab_size": sizes.vocab,
        "layer_types": ["sliding_attention"] * sizes.layers,
        "is_causal": False,
        "sliding_window": reading.window,
        "dflash_config": {
            "block_size": sizes.block,
            "conv_group_size": sizes.group,
            "conv_kernel_size": 2,
            "mask_token_id": reading.mask_token,
            "selector_rank": sizes.rank,
            "selector_top_k": sizes.top_k,
            "target_layer_ids": [tap - 1 for tap in reading.taps],
        },
    }
    assert score.Reading.dflash2(config) == reading

    def plain(tensor):
        return ("F32", list(tensor.shape), tensor.detach().numpy().tobytes())

    held = {
        theirs: plain(model.get_parameter(own))
        for own, theirs in drafter.layout(model).items()
    }
    for folder, tensors, said in (
        (tmp_path / "drafter", held | (extra or {}), config),
        (
            tmp_path / "target",
            {weights.EMBED: plain(target.embed), weights.HEAD: plain(target.head)},
            {},
        ),
    ):
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(safetensors_blob(tensors))
        (folder / "config.json").write_text(json.dumps(said))
    return tmp_path / "drafter", tmp_path / "target", (model, target, reading)


def test_report_walks_every_finished_reply(tmp_path):
    drafted, checkpoint_folder, (model, target, reading) = folders(tmp_path)
    values, tokens = drawn(12)
    root = tmp_path / "rec"
    on_disk(root / "5-0", values, tokens, first=0, prompt=2, n=0)
    on_disk(root / "5-0", values[::-1].copy(), tokens[::-1], first=3, prompt=6, n=1)
    (root / "5-0" / "2.tok").write_bytes(b"")  # a stream that has not closed

    want, alone, taken = [], [], []
    for held, first in (
        (stream(values, tokens, first=0, prompt=2), 0),
        (stream(values[::-1].copy(), tokens[::-1], first=3, prompt=6), 3),
    ):
        found = score.proposals(model, target, held, reading)
        own, lane = score.steps(found.tokens, found), score.lanes(held, found)
        assert len(own) > 1 and sum(own) == len(found.tokens) - 1
        alone += own
        taken += lane
        want.append(
            f"the drafter alone {alone_said(own)}; with copies {taken_said(lane)};"
            f" context from row {first}"
        )
    lines, sound = score.report(root, drafted, checkpoint_folder, each=True)
    assert sound
    assert lines == [
        "weights as stored",
        f"5-0/0: {want[0]}",
        f"5-0/1: {want[1]}",
        f"streams 2, the drafter alone: {alone_said(alone)}",
        f"streams 2, with copies: {taken_said(taken)}",
        lines[5],  # the first round's depth: its own test below
        "1 of the streams begin after row 0: copies of the tokens before it are missed",
    ]
    assert lines[5].startswith("streams 2, with copies, a first round of 1 draft or")
    assert score.report(root, drafted, checkpoint_folder)[0] == lines[:1] + lines[3:]


def test_report_tells_what_the_first_rounds_depth_is_worth(tmp_path, monkeypatch):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    values, tokens = drawn(14)
    on_disk(tmp_path / "rec", values, tokens, first=0, prompt=2)
    reply = np.asarray(tokens[2:], np.int64)
    # a drafter whose three drafts always land, whatever the recording holds
    after = np.append(reply, [-1, -1, -1])
    drafts = np.stack([after[row + 1 : row + 4] for row in range(len(reply))])
    sure = score.Proposals(
        0, 2 + np.arange(len(reply)), reply, drafts, np.ones((12, 3))
    )
    monkeypatch.setattr(score, "proposals", lambda *given: sure)
    lines, sound = score.report(tmp_path / "rec", drafted, checkpoint_folder)
    assert sound
    # eleven tokens to give: 4 4 3 with every draft held from the first round, 2 4 4 1 with one
    assert lines[1:] == [
        "streams 1, the drafter alone: 3 rounds give 11 tokens, 3.667 a round",
        "streams 1, with copies: 3 rounds give 11 tokens, 3.667 a round, 0 of them copies",
        "streams 1, with copies, a first round of 1 draft or of every draft: 4 or 3 rounds",
    ]


def test_report_counts_the_rounds_that_are_copies(tmp_path):
    drafted, checkpoint_folder, (model, target, reading) = folders(tmp_path)
    run = list(range(3, 15))  # the reply is its prompt's twelve tokens again, less one
    tokens = run + run[:-1]
    values, _ = drawn(len(tokens))
    root = tmp_path / "rec"
    on_disk(root, values, tokens, first=0, prompt=len(run))
    held = stream(values, tokens, first=0, prompt=len(run))
    found = score.proposals(model, target, held, reading)
    own, lane = score.steps(found.tokens, found), score.lanes(held, found)
    assert [given.kind for given in lane].count("copy") >= 1 and len(lane) < len(own)
    # the reply ends inside the copy: its round gives fewer tokens than it accepted and one
    assert lane[-1].kind == "copy" and lane[-1].tokens == lane[-1].kept

    lines, sound = score.report(root, drafted, checkpoint_folder, each=True)
    assert sound
    assert lines[1:4] == [
        f"0: the drafter alone {alone_said(own)}; with copies {taken_said(lane)};"
        " context from row 0",
        f"streams 1, the drafter alone: {alone_said(own)}",
        f"streams 1, with copies: {taken_said(lane)}",
    ]
    # a limit of new tokens that leaves no room for a copy of two: none is taken
    bounded = score.lanes(held, found, 10)
    assert [given.kind for given in bounded].count("copy") == 0
    lines, _ = score.report(root, drafted, checkpoint_folder, max_new=10)
    assert lines[2] == f"streams 1, with copies: {taken_said(bounded)}"
    # and it reaches the two counts of another first round
    one, every = (len(score.lanes(held, found, 10, depth)) for depth in (1, 3))
    assert lines[3].endswith(f"of every draft: {one} or {every} rounds")


def test_a_reply_of_one_token_is_listed_without_rounds(tmp_path):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    values, tokens = drawn(4)
    on_disk(tmp_path / "rec", values, tokens, first=0, prompt=3)
    lines, sound = score.report(tmp_path / "rec", drafted, checkpoint_folder, each=True)
    assert sound
    assert lines[1:] == ["0: 0 rounds; context from row 0", "streams 1: 0 rounds"]


def test_report_names_what_it_could_not_walk(tmp_path):
    drafted, checkpoint_folder, _ = folders(tmp_path)
    values, tokens = drawn(6)
    root = tmp_path / "rec"
    on_disk(root, values[:, :2], tokens, first=0, prompt=5, taps=(9, 2))
    lines, sound = score.report(root, drafted, checkpoint_folder)
    assert not sound
    assert lines == [
        "weights as stored",
        "0: the stream holds taps [9, 2], not [5]",
        "streams 0: 0 rounds",
    ]
    on_disk(root, values, tokens, first=0, prompt=3, n=1)  # one that can be walked
    lines, sound = score.report(root, drafted, checkpoint_folder)
    assert not sound
    assert lines[1] == "0: the stream holds taps [9, 2], not [5]"
    assert lines[2].startswith("streams 1, the drafter alone: ")
    assert " rounds give 2 tokens, " in lines[2]
    (tmp_path / "none").mkdir()
    lines, sound = score.report(tmp_path / "none", drafted, checkpoint_folder)
    assert (lines, sound) == (["weights as stored", "streams 0: 0 rounds"], False)


def test_a_drafter_with_a_tensor_that_is_not_read_is_not_sound(tmp_path):
    more = {"later.weight": ("F32", [1], struct.pack("<f", 1.0))}
    drafted, checkpoint_folder, _ = folders(tmp_path, more)
    values, tokens = drawn(8)
    on_disk(tmp_path / "rec", values, tokens, first=0, prompt=3)
    lines, sound = score.report(tmp_path / "rec", drafted, checkpoint_folder)
    assert not sound
    assert lines[1] == "the drafter's later.weight is not read"
    assert lines[2].startswith("streams 1, the drafter alone: ")
    assert " rounds give 4 tokens, " in lines[2]


def test_report_can_score_with_the_weights_as_serving_holds_them(tmp_path, monkeypatch):
    drafted, checkpoint_folder, (model, target, reading) = folders(tmp_path, hidden=64)
    values, tokens = drawn(40, hidden=64)
    root = tmp_path / "rec"
    on_disk(root, values, tokens, first=0, prompt=4)
    held = stream(values, tokens, first=0, prompt=4)

    stored = score.proposals(model, target, held, reading)
    served.as_served(model)
    rounded = score.Target(target.embed, served.four_bits(target.head), True)
    found = score.proposals(model, rounded, held, reading)
    assert (found.drafts != stored.drafts).any()  # 4 bits change what is proposed
    own = score.steps(found.tokens, found)

    scored, proposals = [], score.proposals
    monkeypatch.setattr(
        score,
        "proposals",
        lambda *given: scored.append(given[:2]) or proposals(*given),
    )
    lines, sound = score.report(root, drafted, checkpoint_folder, as_served=True)
    assert sound
    assert lines[:3] == [
        "weights as served: the drafter's linears and the head at 4 bits,"
        " the head's scores in bfloat16",
        f"streams 1, the drafter alone: {alone_said(own)}",
        f"streams 1, with copies: {taken_said(score.lanes(held, found))}",
    ]
    assert len(lines) == 4
    # what it scored with, since so short a reply's rounds may hide one part's rounding
    ((drafter_used, target_used),) = scored
    assert target_used.bf16_scores and torch.equal(target_used.embed, target.embed)
    assert torch.equal(target_used.head, rounded.head)
    assert not torch.equal(target_used.head, target.head)
    for name, parameter in drafter_used.named_parameters():
        assert torch.equal(parameter, model.get_parameter(name)), name
