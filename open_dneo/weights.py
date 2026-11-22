"""Tensors of a checkpoint's safetensors files on disk as torch tensors."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from . import hub

# The torch type of each safetensors dtype the checkpoints read here hold. Values
# are little-endian in the file and read as the machine's own. torch has no 8-bit
# float: F8_E4M3 comes as its codes, for `mxfp8`.
TYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "F8_E4M3": torch.uint8,
}
# Where a checkpoint of several files names each tensor's file, and the one file
# of a checkpoint without it.
INDEX, WHOLE = "model.safetensors.index.json", "model.safetensors"
# GLM-5.3-Flash's names for its embedding and its head. An MXFP8 head has its
# scales beside it.
EMBED = "model.language_model.embed_tokens.weight"
HEAD, HEAD_SCALES = "lm_head.weight", "lm_head.weight_scale"
# The inputs of a row that one MXFP8 scale byte covers.
MX_GROUP = 32
# Rows decoded at a time: their codes as indices are 4096 * 4096 * 8 bytes, 134 MB,
# at the 4096 inputs of the head this is for.
MX_ROWS = 4096


def _e4m3(code: int) -> float:
    """An E4M3 code's value; nan for the two codes that are no number."""
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    sign = -1.0 if code & 0x80 else 1.0
    if exponent == 0:
        return sign * math.ldexp(mantissa / 8, -6)
    return sign * math.ldexp(1 + mantissa / 8, exponent - 7)


# Every code's value, and every scale byte's power of two (0xff is no number).
_E4M3 = torch.tensor([_e4m3(code) for code in range(256)], dtype=torch.float32)
_POW2 = torch.tensor(
    [math.ldexp(1.0, byte - 127) for byte in range(255)] + [math.nan],
    dtype=torch.float32,
)


def _file(path: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """A local file, or its inclusive byte range: `hub.Fetch` for a path."""
    with open(path, "rb") as file:
        if byte_range is None:
            return file.read()
        file.seek(byte_range[0])
        return file.read(byte_range[1] - byte_range[0] + 1)


def names(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Each tensor's dtype and shape, by name."""
    header, _ = hub.safetensors_header(str(path), _file)
    return {
        name: (entry["dtype"], tuple(entry["shape"]))
        for name, entry in header.items()
        if name != "__metadata__"
    }


def read(path: Path, only: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """The file's tensors by name, in the types it holds; `only` these names.

    KeyError for a name the file does not hold, ValueError for a dtype that is not
    read here or data that is not its shape's size.
    """
    header, start = hub.safetensors_header(str(path), _file)
    wanted = [name for name in header if name != "__metadata__"]
    tensors = {}
    for name in wanted if only is None else only:
        raw = hub.tensor_bytes(str(path), header, start, name, _file)
        dtype, shape = header[name]["dtype"], header[name]["shape"]
        if dtype not in TYPES:
            raise ValueError(f"{path}: {name} is {dtype}, which is not read here")
        size = torch.empty(0, dtype=TYPES[dtype]).element_size()
        if len(raw) != size * torch.Size(shape).numel():
            raise ValueError(
                f"{path}: {name} holds {len(raw)} bytes, not {dtype} {shape}"
            )
        if raw:
            flat = torch.frombuffer(bytearray(raw), dtype=TYPES[dtype])
        else:  # an empty buffer cannot be read as a tensor
            flat = torch.empty(0, dtype=TYPES[dtype])
        tensors[name] = flat.reshape(shape)
    return tensors


def write(
    path: Path,
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """A safetensors file of float32 and int64 arrays, in the order given.

    The header is made from the arrays' shapes, then each array goes to the file
    in turn: no copy of all the bytes at once beside the arrays (a trained
    drafter's float32 copy is gigabytes, on a host whose GPU shares its memory).
    FileExistsError where the file exists; ValueError for another type.
    """
    header: dict = {} if metadata is None else {"__metadata__": dict(metadata)}
    kinds = {np.dtype("<f4"): "F32", np.dtype("<i8"): "I64"}
    start = 0
    for name, array in tensors.items():
        if array.dtype not in kinds:
            raise ValueError(
                f"{name} is {array.dtype}: only float32 and int64 are written"
            )
        header[name] = {
            "dtype": kinds[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [start, start + array.nbytes],
        }
        start += array.nbytes
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with Path(path).open("xb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        for array in tensors.values():
            file.write(np.ascontiguousarray(array).reshape(-1).view(np.uint8).data)


def holder(folder: Path, name: str) -> Path:
    """The file of the checkpoint in `folder` that holds tensor `name`.

    A checkpoint of several files says which in its model.safetensors.index.json;
    one without it is model.safetensors alone. KeyError for a name the index does
    not hold, ValueError for an index that points outside the folder.
    """
    index = folder / INDEX
    if not index.exists():
        return folder / WHOLE
    files = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    if name not in files:
        raise KeyError(f"{index}: no tensor named {name}")
    if Path(files[name]).name != files[name]:
        raise ValueError(f"{index}: {name} is in {files[name]!r}, not a file beside it")
    return folder / files[name]


def mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """MXFP8 rows as float32: each E4M3 code times its group's power of two.

    `codes` [n, k] and `scales` [n, k / 32] are the bytes a checkpoint holds as a
    module's `weight` and `weight_scale`. A code with exponent e and mantissa m is
    (1 + m / 8) * 2 ** (e - 7), or m / 8 * 2 ** -6 when e is 0, and a scale byte s
    stands for 2 ** (s - 127): the same reading as TensorFold's. ValueError for a
    code or a scale that is no number, or a value float32 does not hold.
    """
    rows, inputs = codes.shape
    if inputs % MX_GROUP or scales.shape != (rows, inputs // MX_GROUP):
        raise ValueError(
            f"codes {tuple(codes.shape)} do not go with scales {tuple(scales.shape)},"
            f" one a {MX_GROUP} inputs"
        )
    values = torch.empty(codes.shape, dtype=torch.float32)
    for start in range(0, rows, MX_ROWS):
        part = slice(start, start + MX_ROWS)
        grouped = _E4M3[codes[part].long()].unflatten(1, (-1, MX_GROUP))
        values[part] = (grouped * _POW2[scales[part].long()][..., None]).flatten(1)
    if not values.isfinite().all():
        raise ValueError("an MXFP8 code or scale is no number, or too large a value")
    return values


def embedding(folder: Path) -> torch.Tensor:
    """A GLM-5.3-Flash checkpoint's embedding [vocab, hidden], as float32."""
    return read(holder(folder, EMBED), [EMBED])[EMBED].float()


def head(folder: Path) -> torch.Tensor:
    """A GLM-5.3-Flash checkpoint's head [vocab, hidden], as float32.

    Read as the checkpoint holds it, plain or MXFP8.
    """
    path = holder(folder, HEAD)
    held = read(path, [HEAD])[HEAD]
    if names(path)[HEAD][0] != "F8_E4M3":
        return held.float()
    return mxfp8(held, read(holder(folder, HEAD_SCALES), [HEAD_SCALES])[HEAD_SCALES])
