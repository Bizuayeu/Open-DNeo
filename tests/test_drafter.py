"""The drafters against the public definitions and the published checkpoints.

tests/fixtures holds what the public definitions computed at a small size with
every parameter random (tools/reference made them); receipts holds the published
checkpoints' configurations and tensor shapes.
"""

import ast
import dataclasses
import json
import pathlib

import numpy as np
import pytest
import torch

from open_dneo import drafter

HERE = pathlib.Path(__file__).parent
RECEIPTS = HERE.parent / "receipts" / "2026-11-19-survey"


def fixture(name: str):
    """A fixture's configuration, its parameters by name, and a reader of the rest."""
    arrays = np.load(HERE / "fixtures" / name)
    state = {
        key.removeprefix("state."): torch.from_numpy(arrays[key])
        for key in arrays.files
        if key.startswith("state.")
    }
    config = json.loads(str(arrays["config"]))
    return config, state, lambda key: torch.from_numpy(arrays[key])


def close(got: torch.Tensor, want: torch.Tensor) -> None:
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def receipt(name: str) -> tuple[dict, dict[str, tuple[int, ...]]]:
    """A receipt's config.json and its checkpoint's tensor shapes by name."""
    lines = (RECEIPTS / name).read_text(encoding="utf-8").splitlines()
    start = lines.index("== config.json") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("== "))
    tensors = {}
    for line in lines[end + 1 :]:
        if not line.startswith("  total parameters"):
            tensor, _, shape = line.split(None, 2)
            tensors[tensor] = tuple(ast.literal_eval(shape))
    return json.loads("\n".join(lines[start:end])), tensors


def test_dflash2_matches_the_public_definition():
    config, state, at = fixture("dflash2_small.npz")
    shape = drafter.Shape.dflash2(config)
    model = drafter.Drafter.dflash2(shape).eval()
    head = state.pop("head.weight")
    for side in ("predecessor", "successor"):
        # a module's state names a codebook's weight; a checkpoint names the codebook
        name = f"candidate_selector.{side}_codebook"
        state[name] = state.pop(name + ".weight")
    assert drafter.load(model, state) == []

    rows = at("in.target_hidden").shape[1]
    positions = at("in.position_ids")
    anchors, context_positions = positions[:, rows : rows + 1], positions[:, :rows]
    features = at("in.target_hidden").unflatten(-1, (shape.taps, shape.hidden))
    mask = drafter.visible(
        anchors,
        context_positions,
        shape.block,
        whole_block=True,
        window=config["sliding_window"],
    )
    assert rows > config["sliding_window"] and not mask[..., :rows].all()
    states: list[torch.Tensor] = []
    with torch.no_grad():
        close(model.entrance(features), at("out.entrance"))
        hidden = model(
            at("in.noise")[:, None], features, anchors, context_positions, mask, states
        )[:, 0]
        for i, got in enumerate(states):
            close(got[:, 0], at(f"out.layer{i}"))
        close(hidden, at("out.hidden"))
        scores = hidden[:, 1:] @ head.T
        close(scores, at("out.logits"))
        path, candidates, _ = model.head(hidden[:, 1:], scores, at("in.anchor_ids"))
    assert torch.equal(path, at("out.path"))
    assert torch.equal(candidates.sort(-1).values, at("out.candidates"))


def test_dneo_matches_the_public_definition():
    config, state, at = fixture("dneo_small.npz")
    shape = drafter.Shape.dneo(config)
    model = drafter.Drafter.dneo(shape).eval()
    head = state.pop("lm_head.weight")
    assert drafter.load(model, state) == sorted(drafter.DNEO_UNUSED)

    features = at("in.prefix_features")[:, :, config["target_layer_ids"]]
    anchors = at("in.anchor_positions")
    context_positions = torch.arange(features.shape[1]).expand(len(anchors), -1)
    mask = drafter.visible(anchors, context_positions, shape.block, whole_block=False)
    context = at("out.context")
    assert torch.equal(context[:, :, 0], context[:, :, 1])  # one router for the heads
    states: list[torch.Tensor] = []
    with torch.no_grad():
        close(model.entrance(features), context[:, :, 0])
        hidden = model(
            at("in.block_embeddings"),
            features,
            anchors,
            context_positions,
            mask,
            states,
        )
        for i, got in enumerate(states):
            close(got.flatten(1, 2), at(f"out.layer{i}"))
        close(hidden, at("out.hidden"))
        drafts = hidden[:, :, 1:].flatten(0, 1)
        anchor_ids = at("in.anchor_ids").flatten()
        path, rows, chances = model.head(drafts, drafts @ head.T, anchor_ids)
    close(rows, at("out.logits"))
    assert torch.equal(path, at("out.path"))
    # a draft's chance is its share of the definition's own scores over the vocabulary
    shares = at("out.logits").softmax(-1).gather(-1, at("out.path")[..., None])
    close(chances, shares[..., 0])


def test_a_convolution_with_its_fixed_kernel_alone_mixes_a_row_with_the_one_before():
    conv = drafter.Convolution(hidden=4, group=2)
    with torch.no_grad():
        conv.base_kernel[0] = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 0.0, 0.0]]
        )
        conv.kernel_projection.weight.zero_()  # no coefficient drawn from the rows
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
    got, drawn = conv.prepare(x[None])
    assert got[0].tolist() == [
        [1.0, 2.0, 3.0, 4.0],  # a block's first row has no row before it
        [12.0, 4.0, 6.0, 8.0],
        [23.0, 6.0, 9.0, 12.0],
    ]
    assert drawn.shape == (1, 3, 2, 2) and not drawn.any()


def test_the_selector_scores_a_position_from_the_token_it_chose_before():
    shape = drafter.Shape(
        hidden=1, heads=1, kv_heads=1, head_dim=2, ffn=1, layers=1, block=3, group=1,
        eps=1e-5, theta=10000.0, vocab=4, rank=1, taps=1, top_k=2,
    )  # fmt: skip
    head = drafter.Selector(shape)
    with torch.no_grad():
        head.projection.weight.fill_(1.0)
        head.predecessor.weight.copy_(torch.tensor([[0.0], [-1.0], [1.0], [0.0]]))
        head.successor.weight.copy_(torch.tensor([[0.0], [3.0], [2.0], [0.0]]))
    scores = torch.tensor([[[5.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 4.0]]])
    hidden, anchors = torch.ones(1, 2, 1), torch.tensor([2])
    path, candidates, chances = head(hidden, scores, anchors)
    # after anchor 2, token 1 wins (4 + 1 * 3 over 5); after token 1, token 3 wins
    # (4 over 5 - 1 * 2), where after the anchor again token 2 would (5 + 1 * 2)
    assert path.tolist() == [[1, 3]]
    assert candidates.sort(-1).values.tolist() == [[[0, 1], [2, 3]]]
    # a draft's chance among its position's candidates: 7 against 5, then 4 against 3
    want = torch.tensor([[2.0, 1.0]]).sigmoid()
    torch.testing.assert_close(chances, want)
    halved = head(hidden, scores, anchors, temperature=2.0)[2]
    torch.testing.assert_close(halved, (want.logit() / 2).sigmoid())


TAUGHT = drafter.Shape(
    hidden=8, heads=2, kv_heads=1, head_dim=4, ffn=8, layers=1, block=4, group=2,
    eps=1e-5, theta=10000.0, vocab=11, rank=3, taps=2, top_k=4, folded=5,
)  # fmt: skip


@pytest.mark.parametrize("kind", ["Selector", "Folded"])
def test_a_head_taught_its_own_drafts_scores_what_its_chain_scored(kind):
    torch.manual_seed(0)
    head = getattr(drafter, kind)(TAUGHT)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_()
    n = TAUGHT.block - 1
    hidden, scores = torch.randn(6, n, TAUGHT.hidden), torch.randn(6, n, TAUGHT.vocab)
    anchors = torch.randint(TAUGHT.vocab, (6,))
    with torch.no_grad():
        path, chained, chances = head(hidden, scores, anchors)
        before = torch.cat((anchors[:, None], path[:, :-1]), 1)
        taught = head.teach(hidden, scores, before)
    if kind == "Folded":  # every token of the vocabulary, as the chain's rows
        assert taught.candidates is None
        assert torch.equal(taught.values, chained)
        best = taught.values.argmax(-1)
    else:  # the best k of the target head's scores, as the chain's candidates
        assert torch.equal(taught.candidates, chained)
        picked = taught.values.argmax(-1, keepdim=True)
        best = taught.candidates.gather(-1, picked)[..., 0]
    assert torch.equal(best, path)
    torch.testing.assert_close(taught.values.softmax(-1).amax(-1), chances)


def test_a_taught_head_scores_each_position_from_the_token_given_before_it():
    torch.manual_seed(1)
    head = drafter.Folded(TAUGHT)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_()
    n = TAUGHT.block - 1
    hidden, scores = torch.randn(2, n, TAUGHT.hidden), torch.randn(2, n, TAUGHT.vocab)
    given = torch.randint(TAUGHT.vocab, (2, n))
    with torch.no_grad():
        taught = head.teach(hidden, scores, given).values
        state = head.projection(hidden)
        for i in range(n):  # not the chain's own choice: the token given
            torch.testing.assert_close(
                taught[:, i], head.row(scores[:, i], state[:, i], given[:, i], i)
            )


def test_dneos_head_from_dflash2s_scores_dflash2s_candidates_as_dflash2_does():
    shape = dataclasses.replace(TAUGHT, folded=TAUGHT.rank)
    torch.manual_seed(2)
    selector = drafter.Selector(shape)
    with torch.no_grad():
        for parameter in selector.parameters():
            parameter.normal_()
    folded = drafter.Folded.from_selector(selector, shape)
    n = shape.block - 1
    hidden, scores = torch.randn(4, n, shape.hidden), torch.randn(4, n, shape.vocab)
    given = torch.randint(shape.vocab, (4, n))
    with torch.no_grad():
        theirs = selector.teach(hidden, scores, given)
        ours = folded.teach(hidden, scores, given).values
    # the same score on each of DFlash2's candidates; past them it scores the rest
    torch.testing.assert_close(ours.gather(-1, theirs.candidates), theirs.values)
    with pytest.raises(ValueError, match="latent width 5 is not the rank 3"):
        drafter.Folded.from_selector(selector, TAUGHT)


def test_dneos_head_on_the_best_k_drafts_as_dflash2s_head_it_came_from():
    shape = dataclasses.replace(TAUGHT, folded=TAUGHT.rank)
    torch.manual_seed(3)
    selector = drafter.Selector(shape)
    with torch.no_grad():
        for parameter in selector.parameters():
            parameter.normal_()
    folded = drafter.Folded.from_selector(selector, shape, best=shape.top_k)
    n = shape.block - 1
    hidden, scores = torch.randn(5, n, shape.hidden), torch.randn(5, n, shape.vocab)
    anchors = torch.randint(shape.vocab, (5,))
    with torch.no_grad():
        theirs = selector(hidden, scores, anchors)
        ours = folded(hidden, scores, anchors)
        given = torch.randint(shape.vocab, (5, n))
        taught = folded.teach(hidden, scores, given)
        wanted = selector.teach(hidden, scores, given)
    # the drafts, the candidates and the chances among them
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
    torch.testing.assert_close(ours[2], theirs[2])
    assert torch.equal(taught.candidates, wanted.candidates)
    torch.testing.assert_close(taught.values, wanted.values)
    # on the whole vocabulary it may draft past them
    whole = drafter.Folded.from_selector(selector, shape)
    assert whole.best == 0 and folded.best == shape.top_k


def test_a_block_row_sees_the_context_before_its_anchor_and_its_own_block():
    anchors = torch.tensor([[2, 5]])
    context = torch.arange(6)[None]
    dneo = drafter.visible(anchors, context, 2, whole_block=False)[0, 0]
    assert dneo.int().tolist() == [
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    ]
    dflash2 = drafter.visible(anchors, context, 2, whole_block=True, window=3)[0, 0]
    assert dflash2.int().tolist() == [
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    ]


@pytest.mark.parametrize(
    ("name", "kind", "tensors", "beside"),
    [
        ("glm-dflash2.shapes.txt", "dflash2", 81, ()),
        (
            "qwen3-4b-dneo.shapes.txt",
            "dneo",
            89,
            (drafter.DNEO_ROUTER, drafter.DNEO_CANDIDATES, *drafter.DNEO_UNUSED),
        ),
    ],
)
def test_the_layout_is_the_published_checkpoints(name, kind, tensors, beside):
    config, published = receipt(name)
    assert len(published) == tensors
    shape = getattr(drafter.Shape, kind)(config)
    with torch.device("meta"):  # shapes without the memory: 1.2 billion parameters
        model = getattr(drafter.Drafter, kind)(shape)
    ours = drafter.shapes(model)
    assert ours == {k: v for k, v in published.items() if k not in beside}
    assert set(published) - set(ours) == set(beside)
