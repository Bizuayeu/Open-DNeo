"""How much of a dense feature encoder's weight sits on its per-layer diagonals.

A DFlash-style drafter joins c hidden states of the target (c * D values) and projects
them to D with one dense matrix. A structure-preserving encoder, as in DNeo, keeps
coordinate j of every layer in coordinate j. The diagonal share of each D x D block says
which of the two a trained encoder is close to: 1/D is chance, 1 is fully diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hub

ENERGY_FRACTIONS = (0.5, 0.9, 0.99)


@dataclass(frozen=True)
class Block:
    """One layer's D x D part of the encoder."""

    index: int
    energy_share: float  # its part of the whole matrix's squared weight
    diagonal_share: float  # the part of its own squared weight on its diagonal
    rms_diagonal: float
    rms_off_diagonal: float
    positive_diagonal_share: float


def decode(raw: bytes, dtype: str) -> np.ndarray:
    """A safetensors tensor's bytes as float64."""
    if dtype == "BF16":  # a bfloat16 is the high half of a float32
        halves = np.frombuffer(raw, dtype="<u2")
        return (halves.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    plain = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
    if dtype not in plain:
        raise ValueError(f"dtype {dtype} is not read here")
    return np.frombuffer(raw, dtype=plain[dtype]).astype(np.float64)


def block_shares(weight: np.ndarray) -> list[Block]:
    """Split a [D, c * D] encoder into c square blocks and measure each."""
    rows, cols = weight.shape
    if cols % rows != 0:
        raise ValueError(f"shape [{rows}, {cols}] is not c square blocks")
    total = float((weight * weight).sum())
    blocks = []
    for index in range(cols // rows):
        block = weight[:, index * rows : (index + 1) * rows]
        energy = float((block * block).sum())
        diagonal = np.diag(block)
        on_diagonal = float((diagonal * diagonal).sum())
        off_count = rows * rows - rows
        blocks.append(
            Block(
                index=index,
                energy_share=energy / total,
                diagonal_share=on_diagonal / energy,
                rms_diagonal=float(np.sqrt(on_diagonal / rows)),
                rms_off_diagonal=float(np.sqrt((energy - on_diagonal) / off_count)),
                positive_diagonal_share=float((diagonal > 0).mean()),
            )
        )
    return blocks


def energy_ranks(
    weight: np.ndarray, fractions: tuple[float, ...] = ENERGY_FRACTIONS
) -> list[int]:
    """How many singular directions hold each fraction of the squared weight."""
    squared = np.clip(np.linalg.eigvalsh(weight @ weight.T), 0, None)[::-1]
    cumulative = np.cumsum(squared) / float(squared.sum())
    return [int(np.searchsorted(cumulative, fraction)) + 1 for fraction in fractions]


def report(
    repo: str, revision: str, tensor: str, fetch: hub.Fetch = hub.http_fetch
) -> list[str]:
    """The diagonal shares and energy ranks of `tensor` in `repo`'s model.safetensors."""
    commit = hub.model_info(repo, revision, fetch)["sha"]
    url = hub.file_url(repo, commit, "model.safetensors")
    header, data_start = hub.safetensors_header(url, fetch)
    raw = hub.tensor_bytes(url, header, data_start, tensor, fetch)
    entry = header[tensor]
    weight = decode(raw, entry["dtype"]).reshape(entry["shape"])
    rows = weight.shape[0]
    blocks = block_shares(weight)
    lines = [
        f"repo {repo} commit {commit}",
        f"tensor {tensor} {entry['dtype']} {entry['shape']}: {len(blocks)} blocks",
        f"chance level of a diagonal share: {100 / rows:.5f}%",
    ]
    for b in blocks:
        lines.append(
            f"block {b.index}: energy share {100 * b.energy_share:.2f}%"
            f"  diagonal share {100 * b.diagonal_share:.3f}%"
            f"  rms diagonal {b.rms_diagonal:.5f}"
            f"  rms off-diagonal {b.rms_off_diagonal:.5f}"
            f"  diagonal > 0 {100 * b.positive_diagonal_share:.1f}%"
        )
    on_diagonal = sum(b.energy_share * b.diagonal_share for b in blocks)
    lines.append(f"all blocks: diagonal share {100 * on_diagonal:.3f}%")
    ranks = energy_ranks(weight)
    shown = ", ".join(
        f"{int(100 * fraction)}%: {rank}"
        for fraction, rank in zip(ENERGY_FRACTIONS, ranks, strict=True)
    )
    lines.append(f"singular directions holding the squared weight: {shown} of {rows}")
    return lines
