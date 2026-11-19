"""A drafter's weights as TensorFold serves them by default: at 4 bits.

Serving loads a DFlash2 drafter's linears at 4 bits unless it is told otherwise,
and scores drafts on a 4-bit copy of the target's head. A row's inputs go 64 at a
time. A group keeps its lowest value as a bias and a fifteenth of its spread as a
scale, both rounded to bfloat16, and each value becomes the nearest of the 16
steps from the bias up. `four_bits` gives the values such a weight then holds, so
that a model here computes with the weights serving computes with.

Only the weights are serving's. The arithmetic is still float32 products, where
serving keeps its rows in bfloat16.
"""

from __future__ import annotations

import torch

from . import drafter

# The inputs of a row that share a scale and a bias.
GROUP = 64
# A layer's linears that serving holds at 4 bits. down_proj and the selector's
# projection stay bfloat16, as the norms, the fixed kernels and the codebooks do.
LINEARS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "attention_conv.kernel_projection",
    "mlp_conv.kernel_projection",
)


def _bf16(values: torch.Tensor) -> torch.Tensor:
    """float32 values rounded to the nearest bfloat16, as float32."""
    return values.to(torch.bfloat16).float()


def four_bits(weight: torch.Tensor) -> torch.Tensor:
    """A weight [rows, inputs] as a 4-bit linear of serving's holds it, as float32.

    The rounding is TensorFold's `tf_glm_quantize4`, step for step in float32.
    ValueError if the inputs do not go into groups of 64.
    """
    if weight.shape[1] % GROUP:
        raise ValueError(
            f"a row of {weight.shape[1]} inputs does not go into groups of {GROUP}"
        )
    groups = weight.float().unflatten(1, (-1, GROUP))
    low = groups.amin(-1, keepdim=True)
    spread = groups.amax(-1, keepdim=True) - low
    fifteenth = torch.tensor(1 / 15, dtype=torch.float32, device=weight.device)
    scale, bias = _bf16((spread * fifteenth).clamp_min(1e-8)), _bf16(low)
    steps = ((groups - bias) / scale).round().clamp(0, 15)
    return (steps * scale + bias).flatten(1)


@torch.no_grad()
def as_served(model: drafter.Drafter) -> None:
    """Rounds a drafter's linears to 4 bits in place, as serving loads them."""
    names = [
        f"layers.{i}.{name}" for i in range(model.shape.layers) for name in LINEARS
    ]
    if isinstance(model.entrance, drafter.Fused):
        names.append("entrance.fc")
    for name in names:
        weight = model.get_parameter(f"{name}.weight")
        weight.copy_(four_bits(weight))
