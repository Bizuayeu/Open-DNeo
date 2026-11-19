import json
import struct

import numpy as np
import pytest

from open_dneo import entrance, hub
from tests.test_hub import safetensors_blob, serving


def bf16(values) -> bytes:
    """float32 values cut to their high halves, as bfloat16 stores them."""
    words = np.asarray(values, dtype="<f4").view("<u4") >> 16
    return words.astype("<u2").tobytes()


def test_bf16_is_the_high_half_of_a_float32():
    raw = struct.pack("<3H", 0x3F80, 0xC000, 0x3F00)
    assert entrance.decode(raw, "BF16").tolist() == [1.0, -2.0, 0.5]


def test_plain_floats_and_an_unknown_dtype():
    raw = np.asarray([1.5, -3.0], dtype="<f4").tobytes()
    assert entrance.decode(raw, "F32").tolist() == [1.5, -3.0]
    with pytest.raises(ValueError, match="I64"):
        entrance.decode(b"\x00" * 8, "I64")


def test_a_diagonal_block_and_a_mixing_block_are_told_apart():
    rng = np.random.default_rng(0)
    size = 64
    weight = np.hstack([3.0 * np.eye(size), rng.standard_normal((size, size))])
    diagonal, mixing = entrance.block_shares(weight)
    assert diagonal.diagonal_share == pytest.approx(1.0)
    assert diagonal.rms_off_diagonal == 0.0
    assert diagonal.positive_diagonal_share == 1.0
    assert mixing.diagonal_share == pytest.approx(1 / size, rel=0.5)
    assert diagonal.energy_share + mixing.energy_share == pytest.approx(1.0)


def test_a_shape_that_is_not_square_blocks_is_refused():
    with pytest.raises(ValueError, match=r"\[4, 6\]"):
        entrance.block_shares(np.zeros((4, 6)))


def test_energy_ranks_count_the_directions():
    weight = np.diag([10.0, 1.0, 1.0, 1.0])
    # squared weights 100, 1, 1, 1: one direction holds 50% and 90%, three hold 99%
    assert entrance.energy_ranks(weight, (0.5, 0.9, 0.99)) == [1, 1, 3]


def test_report_reads_one_tensor_and_names_the_chance_level():
    size = 8
    weight = np.hstack([np.eye(size), np.eye(size)])
    info = {"sha": "abc", "siblings": [{"rfilename": "model.safetensors"}]}
    files = {
        f"{hub.HUB}/api/models/o/n/revision/main?blobs=true": json.dumps(info).encode(),
        hub.file_url("o/n", "abc", "model.safetensors"): safetensors_blob(
            {
                "other": ("BF16", [2], bf16([1, 2])),
                "fc.weight": ("BF16", [size, 2 * size], bf16(weight.ravel())),
            }
        ),
    }
    lines = entrance.report("o/n", "main", "fc.weight", serving(files))
    assert lines[1] == "tensor fc.weight BF16 [8, 16]: 2 blocks"
    assert lines[2] == "chance level of a diagonal share: 12.50000%"
    assert lines[3].startswith("block 0: energy share 50.00%  diagonal share 100.000%")
    assert lines[5] == "all blocks: diagonal share 100.000%"
