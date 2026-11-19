"""Serving's 4-bit weights, against the kernel's steps worked one value at a time."""

import struct

import numpy as np
import pytest
import torch

from open_dneo import drafter, served

F = np.float32


def bf16(x) -> np.float32:
    """A float32 rounded to the nearest bfloat16, ties to the even one, by its bits."""
    (bits,) = struct.unpack("<I", struct.pack("<f", float(x)))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return F(struct.unpack("<f", struct.pack("<I", bits))[0])


def group(values) -> list[float]:
    """64 values through `tf_glm_quantize4`'s steps, each in float32."""
    x = [F(v) for v in values]
    low, high = min(x), max(x)
    scale = bf16(max((high - low) * F(1.0 / 15.0), F(1e-8)))
    bias = bf16(low)
    steps = [min(max(np.rint((v - bias) / scale), F(0)), F(15)) for v in x]
    return [float(step * scale + bias) for step in steps]


def worked(rows: np.ndarray) -> list[list[float]]:
    """`group` on every 64 inputs of every row."""
    return [
        [value for at in range(0, len(row), 64) for value in group(row[at : at + 64])]
        for row in rows
    ]


def test_a_weight_is_rounded_group_by_group_as_the_kernel_rounds_it():
    generator = np.random.default_rng(3)
    rows = (generator.standard_normal((5, 192)) * 0.05).astype(np.float32)
    rows[1, :64] += 100.0  # far from zero: the bias's own rounding outweighs the spread
    rows[2, 64:128] = 0.25  # no spread at all
    rows[3, 128:] *= 1e-12  # a spread under the scale's floor
    # a bias rounded to 100, well under the values: most are over the top step
    rows[4, :64] = 100.2 + generator.random(64, dtype=np.float32) * 0.3
    got = served.four_bits(torch.from_numpy(rows))
    assert got.dtype == torch.float32 and got.shape == (5, 192)
    assert got.tolist() == worked(rows)
    for row in got.unflatten(1, (-1, 64)).flatten(0, 1):
        assert len(set(row.tolist())) <= 16
    # the far group's bias rounds to 100 on the dot, above many of its values: those
    # would be steps below 0, and are held at the bias
    under = torch.from_numpy(rows[1, :64] < 100.0)
    assert under.sum() > 16 and got[1, :64][under].eq(100.0).all()
    # and the group that starts over its bias tops out at the fifteenth step
    top = got[4, :64].max()
    assert (got[4, :64] == top).sum() > 16 and top < torch.from_numpy(
        rows[4, :64]
    ).max()
    assert set(got[2, 64:128].tolist()) == {0.25}


def test_sixteen_evenly_spaced_values_are_kept_as_they_are():
    steps = torch.arange(16.0).repeat_interleave(4) * 0.5 - 3.0
    assert served.four_bits(steps[None]).tolist() == [steps.tolist()]


def test_a_bias_is_rounded_to_bfloat16_with_ties_to_the_even_one():
    below, above = 1 + 2.0**-8, 1 + 2.0**-7 + 2.0**-8  # halfway between two bfloat16s
    flat = torch.tensor([[below] * 64 + [above] * 64])
    got = served.four_bits(flat)
    # a flat group's scale is its floor, 1e-8 as a bfloat16. The first group's bias
    # is 1, under its values, which are the top step; the second's is over them
    top = float(F(15) * bf16(F(1e-8)) + F(1.0))
    assert top > 1.0 and set(got[0, :64].tolist()) == {top}
    assert set(got[0, 64:].tolist()) == {1 + 2.0**-6}


def test_rows_that_do_not_go_into_groups_are_refused():
    with pytest.raises(ValueError, match="a row of 96 inputs does not go into groups"):
        served.four_bits(torch.zeros(2, 96))


def small(kind: str) -> drafter.Drafter:
    """A drafter whose linears all take inputs in 64s, every parameter random."""
    sizes = drafter.Shape(
        hidden=64, heads=4, kv_heads=2, head_dim=16, ffn=96, layers=2, block=4,
        group=16, eps=1e-5, theta=10000.0, vocab=23, rank=3, taps=2, top_k=5,
        folded=6,
    )  # fmt: skip
    model = getattr(drafter.Drafter, kind)(sizes)
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


@pytest.mark.parametrize("kind", ["dflash2", "dneo"])
def test_a_served_drafter_holds_its_linears_at_four_bits_and_the_rest_as_it_was(kind):
    model = small(kind)
    before = {name: value.clone() for name, value in model.named_parameters()}
    served.as_served(model)
    # what TensorFold's loader reads through its 4-bit path: q, k and v, o, gate and
    # up, the two convolutions' projections, and DFlash2's entrance
    linears = (
        "q_proj k_proj v_proj o_proj gate_proj up_proj"
        " attention_conv.kernel_projection mlp_conv.kernel_projection"
    ).split()
    rounded = {f"layers.{i}.{name}.weight" for i in range(2) for name in linears}
    rounded |= {"entrance.fc.weight"} if kind == "dflash2" else set()
    for name, value in model.named_parameters():
        if name in rounded:
            assert torch.equal(value, served.four_bits(before[name])), name
            assert not torch.equal(value, before[name]), name
        else:
            assert torch.equal(value, before[name]), name
    kept = set(before) - rounded
    assert {"layers.0.down_proj.weight", "head.projection.weight"} <= kept
