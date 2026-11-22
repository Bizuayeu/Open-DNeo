"""DFlash2's and DNeo's draft passes in PyTorch, on one layer.

A drafter reads rows of the target (the context) and a block: the last known token,
the anchor, then mask tokens. Both drafters stack the same layer and differ in
three places:

- the entrance, which turns each context row's tapped states into one row: DFlash2
  joins them and shrinks them with a dense matrix (`Fused`), DNeo takes a weighted
  sum with one scalar a tap (`Router`);
- what a block's rows see (`visible`): in DFlash2 the whole block and a window of
  the context, in DNeo the context and the block's first row only;
- the head, which turns the target head's scores into drafts: DFlash2 re-scores
  the best k candidates (`Selector`), DNeo corrects every score (`Folded`).

The arithmetic follows the public definitions named in NOTICE. DNeo's rejection
memory and per-layer routers are not here.

A head drafts in a chain, each position after the token it chose before. Training
gives it those tokens instead (`teach`), and it scores every position at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Shape:
    """A drafter's sizes."""

    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    layers: int
    block: int
    group: int  # channels in a convolution group
    eps: float
    theta: float
    vocab: int
    rank: int  # the head's width
    taps: int  # tapped states a context row
    top_k: int = 0  # DFlash2: candidates the head re-scores
    folded: int = 0  # DNeo: the head's latent width
    bf16_frequencies: bool = False  # DNeo: rotary frequencies rounded to bf16

    @classmethod
    def dflash2(cls, config: dict) -> Shape:
        """From a DFlash2 checkpoint's config.json."""
        own = config["dflash_config"]
        if own["conv_kernel_size"] != 2:
            raise ValueError("only two-tap convolutions are implemented")
        return cls(
            hidden=config["hidden_size"],
            heads=config["num_attention_heads"],
            kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            ffn=config["intermediate_size"],
            layers=config["num_hidden_layers"],
            block=own["block_size"],
            group=own["conv_group_size"],
            eps=config["rms_norm_eps"],
            theta=config["rope_parameters"]["rope_theta"],
            vocab=config["vocab_size"],
            rank=own["selector_rank"],
            taps=len(own["target_layer_ids"]),
            top_k=own["selector_top_k"],
        )

    @classmethod
    def dneo(cls, config: dict) -> Shape:
        """From a DNeo checkpoint's config.json."""
        if not config["shared_layer_mofa"]:
            raise ValueError("only a router shared by the layers is implemented")
        return cls(
            hidden=config["hidden_size"],
            heads=config["num_attention_heads"],
            kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            ffn=config["intermediate_size"],
            layers=config["num_hidden_layers"],
            block=config["block_size"],
            group=config["conv_group_size"],
            eps=config["rms_norm_eps"],
            theta=float(config["rope_theta"]),
            vocab=config["vocab_size"],
            rank=config["head_rank"],
            taps=len(config["target_layer_ids"]),
            folded=config["folded_rank"],
            bf16_frequencies=config["rotary_frequency_dtype"] == "bfloat16",
        )


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


def rotary(positions: torch.Tensor, shape: Shape) -> tuple[torch.Tensor, torch.Tensor]:
    """The cosines and sines that turn a head's pairs at `positions` [b, n]."""
    steps = torch.arange(0, shape.head_dim, 2, device=positions.device).float()
    frequencies = 1.0 / (shape.theta ** (steps / shape.head_dim))
    if shape.bf16_frequencies:
        frequencies = frequencies.to(torch.bfloat16).float()
    angles = positions[..., None].float() * frequencies
    angles = torch.cat((angles, angles), -1)
    return angles.cos(), angles.sin()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x` [b, heads, n, head_dim] turned by its rows' positions."""
    first, second = x.chunk(2, -1)
    turned = torch.cat((-second, first), -1)
    return x * cos[:, None].to(x.dtype) + turned * sin[:, None].to(x.dtype)


class Convolution(nn.Module):
    """A two-tap convolution along a block's rows.

    A row's output is itself and the row before it (nothing for a block's first
    row), each scaled a channel by a fixed kernel plus, a group of channels, a
    coefficient drawn from the row `prepare` saw. `prepare` and `finish` are the
    two halves around attention or the MLP: the first draws both halves'
    coefficients.
    """

    def __init__(self, hidden: int, group: int) -> None:
        super().__init__()
        self.group = group
        self.base_kernel = nn.Parameter(torch.zeros(2, 2, hidden))
        self.kernel_projection = nn.Linear(hidden, 4 * (hidden // group), bias=False)

    def _convolve(
        self, x: torch.Tensor, drawn: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        own = drawn.repeat_interleave(self.group, -1)  # a group's on its channels
        before = F.pad(x[..., :-1, :], (0, 0, 1, 0))
        return x * (base[0] + own[..., 0, :]) + before * (base[1] + own[..., 1, :])

    def prepare(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x` [..., block, hidden] convolved, and the coefficients `finish` takes."""
        drawn = self.kernel_projection(x).unflatten(-1, (2, 2, -1))
        first = self._convolve(x, drawn[..., 0, :, :], self.base_kernel[0])
        return first, drawn[..., 1, :, :]

    def finish(self, x: torch.Tensor, drawn: torch.Tensor) -> torch.Tensor:
        return self._convolve(x, drawn, self.base_kernel[1])


class Layer(nn.Module):
    def __init__(self, shape: Shape) -> None:
        super().__init__()
        self.heads, self.kv_heads = shape.heads, shape.kv_heads
        wide, narrow = shape.heads * shape.head_dim, shape.kv_heads * shape.head_dim
        self.q_proj = nn.Linear(shape.hidden, wide, bias=False)
        self.k_proj = nn.Linear(shape.hidden, narrow, bias=False)
        self.v_proj = nn.Linear(shape.hidden, narrow, bias=False)
        self.o_proj = nn.Linear(wide, shape.hidden, bias=False)
        self.q_norm = RMSNorm(shape.head_dim, shape.eps)
        self.k_norm = RMSNorm(shape.head_dim, shape.eps)
        self.input_norm = RMSNorm(shape.hidden, shape.eps)
        self.post_norm = RMSNorm(shape.hidden, shape.eps)
        self.gate_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
        self.up_proj = nn.Linear(shape.hidden, shape.ffn, bias=False)
        self.down_proj = nn.Linear(shape.ffn, shape.hidden, bias=False)
        self.attention_conv = Convolution(shape.hidden, shape.group)
        self.mlp_conv = Convolution(shape.hidden, shape.group)

    def _heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        """[b, n, heads * head_dim] as [b, heads, n, head_dim]."""
        return x.unflatten(-1, (heads, -1)).transpose(1, 2)

    def context(
        self, rows: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The keys and values of context rows [b, s, hidden] in this layer."""
        keys = self.k_norm(self._heads(self.k_proj(rows), self.kv_heads))
        return rotate(keys, cos, sin), self._heads(self.v_proj(rows), self.kv_heads)

    def forward(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Blocks [b, m, block, hidden] through attention and the MLP.

        `keys` and `values` are the context's (`context`), `cos` and `sin` the
        block rows' turns, `mask` [b, 1, m * block, s + m * block] what each block
        row sees of the context rows and then of every block row.
        """
        size = hidden.shape
        normed, drawn = self.attention_conv.prepare(self.input_norm(hidden))
        flat = normed.flatten(1, 2)
        q = rotate(self.q_norm(self._heads(self.q_proj(flat), self.heads)), cos, sin)
        k = rotate(self.k_norm(self._heads(self.k_proj(flat), self.kv_heads)), cos, sin)
        v = self._heads(self.v_proj(flat), self.kv_heads)
        each = self.heads // self.kv_heads
        k = torch.cat((keys, k), 2).repeat_interleave(each, 1)
        v = torch.cat((values, v), 2).repeat_interleave(each, 1)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = self.o_proj(attended.transpose(1, 2).flatten(2)).view(size)
        hidden = hidden + self.attention_conv.finish(attended, drawn)
        normed, drawn = self.mlp_conv.prepare(self.post_norm(hidden))
        mlp = self.down_proj(F.silu(self.gate_proj(normed)) * self.up_proj(normed))
        return hidden + self.mlp_conv.finish(mlp, drawn)


class Fused(nn.Module):
    """DFlash2's entrance: a row's tapped states joined, shrunk and normed."""

    def __init__(self, shape: Shape) -> None:
        super().__init__()
        self.fc = nn.Linear(shape.taps * shape.hidden, shape.hidden, bias=False)
        self.norm = RMSNorm(shape.hidden, shape.eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[b, s, taps, hidden] to [b, s, hidden]."""
        return self.norm(self.fc(features.flatten(-2)))


class Router(nn.Module):
    """DNeo's entrance: a weighted sum of a row's tapped states, a scalar a tap."""

    def __init__(self, shape: Shape) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(shape.taps))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[b, s, taps, hidden] to [b, s, hidden]."""
        weights = self.logits.float().softmax(-1).to(features.dtype)
        return torch.einsum("t,bstd->bsd", weights, features)


class Bridged(nn.Module):
    """DNeo's entrance in front of DFlash2's layers: the weighted sum, bridged and normed.

    `fc` [hidden, hidden] maps the router's sum to what DFlash2's entrance gave
    before its norm, and the norm is DFlash2's (`bridge`).
    """

    def __init__(self, shape: Shape) -> None:
        super().__init__()
        self.router = Router(shape)
        self.fc = nn.Linear(shape.hidden, shape.hidden, bias=False)
        self.norm = RMSNorm(shape.hidden, shape.eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[b, s, taps, hidden] to [b, s, hidden]."""
        return self.norm(self.fc(self.router(features)))


class Taught(NamedTuple):
    """A head's scores with the token before each position given (`teach`), or at
    some positions of some blocks, one a row (`teach_rows`): the leading [m, n]
    is then [k]."""

    values: torch.Tensor  # [m, n, support]
    candidates: torch.Tensor | None  # [m, n, support] the tokens scored; None: all


def carrying(candidates: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    """`candidates` [m, n, k] with each label [m, n] among them.

    Where a label (not -1) is not, it takes the last slot, the weakest candidate's,
    so that training can teach a head the token it should have scored: a token
    outside its candidates it can never draft.
    """
    if labels is None:
        return candidates
    absent = (labels >= 0) & ~(candidates == labels[..., None]).any(-1)
    last = torch.where(absent, labels, candidates[..., -1])
    return torch.cat((candidates[..., :-1], last[..., None]), -1)


class Selector(nn.Module):
    """DFlash2's head: the target head's best k a position, re-scored in a chain."""

    def __init__(self, shape: Shape) -> None:
        super().__init__()
        self.top_k = shape.top_k
        self.predecessor = nn.Embedding(shape.vocab, shape.rank)
        self.successor = nn.Embedding(shape.vocab, shape.rank)
        self.projection = nn.Linear(shape.hidden, shape.rank, bias=False)

    def _rescored(
        self,
        unary: torch.Tensor,
        candidates: torch.Tensor,
        state: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        """One position's candidates [m, k] re-scored after the tokens `previous` [m]."""
        pair = torch.einsum(
            "mr,mkr->mk",
            self.predecessor(previous) * state,
            self.successor(candidates),
        )
        return unary + pair

    def teach(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        previous: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Taught:
        """Each position's candidates re-scored after the token `previous` [m, n]
        gives before it, the block's own token first. With `labels` [m, n] each is
        among them (`carrying`)."""
        unary, candidates = scores.topk(self.top_k, -1)
        if labels is not None:
            candidates = carrying(candidates, labels)
            unary = scores.gather(-1, candidates)
        state = self.projection(hidden)
        values = [
            self._rescored(unary[:, i], candidates[:, i], state[:, i], previous[:, i])
            for i in range(hidden.shape[1])
        ]
        return Taught(torch.stack(values, 1), candidates)

    def teach_rows(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        previous: torch.Tensor,
        labels: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> Taught:
        """`teach` at k positions of some blocks, one a row: `hidden` [k, hidden],
        `scores` [k, vocab], `previous`, `labels` and `positions` [k]. A position's
        re-scoring does not depend on its place in the block."""
        unary, candidates = scores.topk(self.top_k, -1)
        if labels is not None:
            candidates = carrying(candidates, labels)
            unary = scores.gather(-1, candidates)
        state = self.projection(hidden)
        return Taught(self._rescored(unary, candidates, state, previous), candidates)

    def forward(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        anchors: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The greedy drafts [m, n], each position's candidates [m, n, k], chances.

        `hidden` [m, n, hidden] are the mask rows' states, `scores` [m, n, vocab]
        the target head's on them, `anchors` [m] each block's known token. A
        draft's chance [m, n] is its share among the candidates' re-scored values
        at `temperature`.
        """
        unary, candidates = scores.topk(self.top_k, -1)
        state = self.projection(hidden)
        previous, path, chances = anchors, [], []
        for i in range(hidden.shape[1]):
            rescored = self._rescored(
                unary[:, i], candidates[:, i], state[:, i], previous
            )
            best = rescored.argmax(-1, keepdim=True)
            chances.append((rescored / temperature).softmax(-1).gather(-1, best)[:, 0])
            previous = candidates[:, i].gather(-1, best)[:, 0]
            path.append(previous)
        return torch.stack(path, 1), candidates, torch.stack(chances, 1)


class Folded(nn.Module):
    """DNeo's head: every score corrected from the previous token, in a chain.

    With `best` it drafts among the target head's best k a position only, as
    DNeo's runtime can serve it, and a draft's chance is its share among them.
    """

    def __init__(self, shape: Shape, best: int = 0) -> None:
        super().__init__()
        self.best = best  # 0: the whole vocabulary
        self.token = nn.Embedding(shape.vocab, shape.folded)
        self.predecessor = nn.Embedding(shape.vocab, shape.rank)
        self.projection = nn.Linear(shape.hidden, shape.rank, bias=False)
        self.latent = nn.Linear(shape.rank, shape.folded, bias=False)
        self.out = nn.Linear(shape.folded, shape.vocab, bias=False)
        self.dspark_scale = nn.Parameter(torch.ones(shape.block - 1))
        self.df2_scale = nn.Parameter(torch.zeros(shape.block - 1))

    @classmethod
    def from_selector(cls, selector: Selector, shape: Shape, best: int = 0) -> Folded:
        """DFlash2's head as DNeo's, its re-scoring carried over whole.

        With the latent projection as the identity, b = 1 and a = 0, DNeo's
        correction is DFlash2's re-scoring read over the whole vocabulary: on
        DFlash2's best k it gives DFlash2's scores, so with `best` as k it drafts
        as DFlash2 does. The token table starts at zero. ValueError if the latent
        width is not the rank.
        """
        if shape.folded != shape.rank:
            raise ValueError(
                f"the latent width {shape.folded} is not the rank {shape.rank}"
            )
        head = cls(shape, best).to(selector.projection.weight)
        with torch.no_grad():
            head.predecessor.weight.copy_(selector.predecessor.weight)
            head.projection.weight.copy_(selector.projection.weight)
            head.latent.weight.copy_(torch.eye(shape.rank))
            head.out.weight.copy_(selector.successor.weight)
            head.token.weight.zero_()
            head.dspark_scale.zero_()
            head.df2_scale.fill_(1.0)
        return head

    def row(
        self,
        scores: torch.Tensor,
        state: torch.Tensor,
        previous: torch.Tensor,
        position: int | torch.Tensor,
    ) -> torch.Tensor:
        """One position's corrected scores [m, vocab], or with `position` [m] each
        row's at its own position."""
        a, b = self.dspark_scale[position], self.df2_scale[position]
        a, b = a[..., None], b[..., None]
        pair = self.latent(self.predecessor(previous) * state)
        return (a + b) * scores + self.out(a * self.token(previous) + b * pair)

    def teach(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        previous: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Taught:
        """Each position's corrected scores after the token `previous` [m, n] gives
        before it, the block's own token first: over the vocabulary, or with `best`
        over the target head's best k, where `labels` [m, n] are each among them
        (`carrying`)."""
        state = self.projection(hidden)
        rows = torch.stack(
            [
                self.row(scores[:, i], state[:, i], previous[:, i], i)
                for i in range(hidden.shape[1])
            ],
            1,
        )
        if not self.best:
            return Taught(rows, None)
        candidates = carrying(scores.topk(self.best, -1).indices, labels)
        return Taught(rows.gather(-1, candidates), candidates)

    def teach_rows(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        previous: torch.Tensor,
        labels: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> Taught:
        """`teach` at k positions of some blocks, one a row: `hidden` [k, hidden],
        `scores` [k, vocab], `previous`, `labels` and `positions` [k], each row
        corrected at its own position's scales."""
        rows = self.row(scores, self.projection(hidden), previous, positions)
        if not self.best:
            return Taught(rows, None)
        candidates = carrying(scores.topk(self.best, -1).indices, labels)
        return Taught(rows.gather(-1, candidates), candidates)

    def forward(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        anchors: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The greedy drafts [m, n], the corrected scores [m, n, vocab], chances.

        A draft's chance [m, n] is its share of the corrected scores over the
        vocabulary at `temperature`. With `best` the second is each position's
        candidates [m, n, k], and the shares are among them.
        """
        state = self.projection(hidden)
        candidates = scores.topk(self.best, -1).indices if self.best else None
        previous, path, rows, chances = anchors, [], [], []
        for i in range(hidden.shape[1]):
            rows.append(self.row(scores[:, i], state[:, i], previous, i))
            values = (
                rows[-1]
                if candidates is None
                else rows[-1].gather(-1, candidates[:, i])
            )
            share = (values / temperature).softmax(-1).max(-1)
            previous = share.indices
            if candidates is not None:
                previous = candidates[:, i].gather(-1, previous[:, None])[:, 0]
            path.append(previous)
            chances.append(share.values)
        second = torch.stack(rows, 1) if candidates is None else candidates
        return torch.stack(path, 1), second, torch.stack(chances, 1)


def visible(
    anchors: torch.Tensor,
    context_positions: torch.Tensor,
    block: int,
    *,
    whole_block: bool,
    window: int | None = None,
) -> torch.Tensor:
    """What each block row sees: [b, 1, m * block, s + m * block], true where seen.

    `anchors` [b, m] are the positions of the blocks' first rows and
    `context_positions` [b, s] the context rows'. A block row sees the context
    rows before its anchor, and of the blocks its own: every row of it
    (`whole_block`, DFlash2) or its first row only (DNeo). With `window`
    (DFlash2's sliding window) it sees nothing `window` positions or more away.
    """
    b, m = anchors.shape
    offsets = torch.arange(block, device=anchors.device)
    rows = (anchors[..., None] + offsets).flatten(1)  # [b, m * block]
    starts = anchors.repeat_interleave(block, 1)
    owner = torch.arange(m, device=anchors.device).repeat_interleave(block)
    context = context_positions[:, None, :] < starts[..., None]
    local = (owner[:, None] == owner[None, :]).expand(b, -1, -1)
    if not whole_block:
        local = local & (offsets.repeat(m) == 0)[None, None, :]
    if window is not None:
        context = context & (rows[..., None] - context_positions[:, None, :] < window)
        local = local & ((rows[..., None] - rows[:, None, :]).abs() < window)
    return torch.cat((context, local), -1)[:, None]


class Drafter(nn.Module):
    """A drafter's layers behind an entrance (`Fused` or `Router`)."""

    def __init__(self, shape: Shape, entrance: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.shape = shape
        self.entrance, self.head = entrance, head
        self.layers = nn.ModuleList(Layer(shape) for _ in range(shape.layers))
        self.norm = RMSNorm(shape.hidden, shape.eps)

    @classmethod
    def dflash2(cls, shape: Shape) -> Drafter:
        return cls(shape, Fused(shape), Selector(shape))

    @classmethod
    def dneo(cls, shape: Shape) -> Drafter:
        return cls(shape, Router(shape), Folded(shape))

    def forward(
        self,
        blocks: torch.Tensor,
        features: torch.Tensor,
        anchors: torch.Tensor,
        context_positions: torch.Tensor,
        mask: torch.Tensor,
        states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The blocks' final states [b, m, block, hidden].

        `blocks` [b, m, block, hidden] are the embeddings of each block's anchor
        token and its mask tokens, `features` [b, s, taps, hidden] the context
        rows' tapped states at `context_positions` [b, s], `anchors` [b, m] the
        positions of the blocks' first rows, `mask` what `visible` gives. Each
        layer's output is appended to `states` when one is given.
        """
        context = self.entrance(features)
        context_turns = rotary(context_positions, self.shape)
        offsets = torch.arange(self.shape.block, device=anchors.device)
        turns = rotary((anchors[..., None] + offsets).flatten(1), self.shape)
        hidden = blocks
        for layer in self.layers:
            keys, values = layer.context(context, *context_turns)
            hidden = layer(hidden, keys, values, *turns, mask)
            if states is not None:
                states.append(hidden)
        return self.norm(hidden)


_LAYER = {
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "o_proj.weight": "self_attn.o_proj.weight",
    "q_norm.weight": "self_attn.q_norm.weight",
    "k_norm.weight": "self_attn.k_norm.weight",
    "input_norm.weight": "input_layernorm.weight",
    "post_norm.weight": "post_attention_layernorm.weight",
    "gate_proj.weight": "mlp.gate_proj.weight",
    "up_proj.weight": "mlp.up_proj.weight",
    "down_proj.weight": "mlp.down_proj.weight",
    "attention_conv.base_kernel": "attention_conv.base_kernel",
    "attention_conv.kernel_projection.weight": "attention_conv.kernel_projection.weight",
    "mlp_conv.base_kernel": "mlp_conv.base_kernel",
    "mlp_conv.kernel_projection.weight": "mlp_conv.kernel_projection.weight",
}
_DFLASH2 = {
    "norm.weight": "norm.weight",
    "entrance.fc.weight": "fc.weight",
    "entrance.norm.weight": "hidden_norm.weight",
    "head.projection.weight": "candidate_selector.hidden_projection.weight",
    "head.predecessor.weight": "candidate_selector.predecessor_codebook",
    "head.successor.weight": "candidate_selector.successor_codebook",
}
_DNEO = {
    "norm.weight": "draft.norm.weight",
    "head.token.weight": "ar_head.folded_token_embed.weight",
    "head.predecessor.weight": "ar_head.predecessor_codebook.weight",
    "head.projection.weight": "ar_head.hidden_projection.weight",
    "head.latent.weight": "ar_head.df2_latent_projection.weight",
    "head.out.weight": "ar_head.folded_out_proj.weight",
    "head.dspark_scale": "ar_head.dspark_logit_scale",
    "head.df2_scale": "ar_head.df2_logit_scale",
}
# What a DNeo checkpoint holds beyond the layout: the router (read by `load_dneo`
# through its candidate mask), and tensors of parts that are not implemented here.
DNEO_ROUTER = "conditioner.router_logits"
DNEO_CANDIDATES = "conditioner.shared_mofa_candidate_mask"
DNEO_UNUSED = (
    "conditioner.fixed_layer_indices",
    "conditioner.shared_mofa_router_dual",
    "conditioner.shared_mofa_router_dual_fixed",
    "draft.rejection_source_embeddings",
)


def layout(model: Drafter) -> dict[str, str]:
    """The checkpoint's name for each of the model's parameters.

    DNeo's router is not in it: a checkpoint stores a scalar a target layer and a
    mask of the candidates (`load_dneo`).
    """
    dneo = isinstance(model.entrance, Router)
    names = dict(_DNEO if dneo else _DFLASH2)
    for i in range(model.shape.layers):
        for own, theirs in _LAYER.items():
            theirs = theirs.replace("self_attn.", "") if dneo else theirs
            prefix = "draft.layers" if dneo else "layers"
            names[f"layers.{i}.{own}"] = f"{prefix}.{i}.{theirs}"
    return names


def shapes(model: Drafter) -> dict[str, tuple[int, ...]]:
    """The shape a checkpoint gives each tensor of `layout`, by checkpoint name."""
    return {
        theirs: tuple(model.get_parameter(own).shape)
        for own, theirs in layout(model).items()
    }


def load(model: Drafter, tensors: dict[str, torch.Tensor]) -> list[str]:
    """Copies a checkpoint's tensors into the model; the names it did not use.

    Every parameter of `layout` must be there with its shape. A DNeo model's
    router is read from the candidates' entries of the checkpoint's.
    """
    names = layout(model)
    with torch.no_grad():
        for own, theirs in names.items():
            parameter = model.get_parameter(own)
            if tuple(tensors[theirs].shape) != tuple(parameter.shape):
                raise ValueError(f"{theirs} is {tuple(tensors[theirs].shape)}")
            parameter.copy_(tensors[theirs])
        used = set(names.values())
        if isinstance(model.entrance, Router):
            candidates = tensors[DNEO_CANDIDATES].bool()
            model.entrance.logits.copy_(tensors[DNEO_ROUTER][0, 0][candidates])
            used |= {DNEO_ROUTER, DNEO_CANDIDATES}
    return sorted(set(tensors) - used)
