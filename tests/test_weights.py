import json
import os
import struct

import pytest
import torch

from open_dneo import drafter, weights

from .test_hub import safetensors_blob


def saved(tmp_path, tensors):
    """A safetensors file holding `tensors` (name: dtype, shape, raw bytes)."""
    path = tmp_path / "model.safetensors"
    path.write_bytes(safetensors_blob(tensors))
    return path


def test_tensors_come_in_the_types_the_file_holds(tmp_path):
    path = saved(
        tmp_path,
        {
            # bfloat16 is the high half of a float32: 1.0, -2.0, 0.5, 2.0
            "half": (
                "BF16",
                [2, 2],
                struct.pack("<4H", 0x3F80, 0xC000, 0x3F00, 0x4000),
            ),
            "whole": ("F32", [3], struct.pack("<3f", 1.5, -0.25, 3.0)),
            "count": ("I64", [1], struct.pack("<q", -7)),
            "flags": ("BOOL", [3], bytes([1, 0, 1])),
            "none": ("I64", [0], b""),
        },
    )
    assert weights.names(path) == {
        "half": ("BF16", (2, 2)),
        "whole": ("F32", (3,)),
        "count": ("I64", (1,)),
        "flags": ("BOOL", (3,)),
        "none": ("I64", (0,)),
    }
    read = weights.read(path)
    assert list(read) == ["half", "whole", "count", "flags", "none"]
    assert read["half"].dtype == torch.bfloat16
    assert read["half"].float().tolist() == [[1.0, -2.0], [0.5, 2.0]]
    assert read["whole"].tolist() == [1.5, -0.25, 3.0]
    assert read["count"].tolist() == [-7] and read["count"].dtype == torch.int64
    assert read["flags"].tolist() == [True, False, True]
    assert read["none"].shape == (0,) and read["none"].dtype == torch.int64


def test_arrays_written_are_read_back_as_they_were(tmp_path):
    import numpy as np

    path = tmp_path / "w.safetensors"
    written = {
        "a.weight": np.arange(6, dtype="<f4").reshape(2, 3) / 7,
        "b.taps": np.array([40, 44], "<i8"),
    }
    weights.write(path, written, {"note": "x"})
    back = weights.read(path)
    assert list(back) == ["a.weight", "b.taps"]
    assert back["b.taps"].dtype == torch.int64 and back["b.taps"].tolist() == [40, 44]
    assert torch.equal(back["a.weight"], torch.from_numpy(written["a.weight"]))
    with pytest.raises(FileExistsError):
        weights.write(path, written)
    with pytest.raises(ValueError, match="only float32 and int64"):
        weights.write(tmp_path / "x", {"c": np.zeros(2, np.float64)})


def test_a_written_file_is_the_header_then_each_array_in_order(tmp_path):
    """The bytes as the first version wrote them, every array joined after the
    header: the arrays now go to the file one at a time, as they come."""
    import json

    import numpy as np

    written = {
        "w": np.arange(12, dtype="<f4").reshape(3, 4).T,  # not contiguous: C order
        "n": np.array([3, -1], "<i8"),
        "e": np.zeros((0, 2), "<f4"),
    }
    path = tmp_path / "w.safetensors"
    weights.write(path, written, {"step": "2"})
    header, start, data = {"__metadata__": {"step": "2"}}, 0, b""
    for name, array in written.items():
        raw = np.ascontiguousarray(array).tobytes()
        kind = "F32" if array.dtype == np.dtype("<f4") else "I64"
        header[name] = {
            "dtype": kind,
            "shape": list(array.shape),
            "data_offsets": [start, start + len(raw)],
        }
        data += raw
        start += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    assert path.read_bytes() == struct.pack("<Q", len(encoded)) + encoded + data


def test_only_the_tensors_asked_for_are_read(tmp_path):
    path = saved(
        tmp_path,
        {"a": ("F32", [1], struct.pack("<f", 1.0)), "b": ("U16", [1], b"\x00\x00")},
    )
    assert list(weights.read(path, ["a"])) == ["a"]  # b's type is never looked at
    with pytest.raises(KeyError, match="no tensor named c"):
        weights.read(path, ["c"])
    with pytest.raises(ValueError, match="b is U16, which is not read here"):
        weights.read(path)


def test_data_that_is_not_its_shapes_size_is_refused(tmp_path):
    path = saved(tmp_path, {"a": ("F32", [3], struct.pack("<2f", 1.0, 2.0))})
    with pytest.raises(ValueError, match=r"a holds 8 bytes, not F32 \[3\]"):
        weights.read(path)


def test_a_checkpoint_on_disk_loads_into_its_model(tmp_path):
    sizes = drafter.Shape(
        hidden=8, heads=4, kv_heads=2, head_dim=4, ffn=12, layers=2, block=4,
        group=2, eps=1e-5, theta=10000.0, vocab=23, rank=3, taps=2, top_k=5,
    )  # fmt: skip
    source = drafter.Drafter.dflash2(sizes)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    path = saved(
        tmp_path,
        {
            theirs: (
                "F32",
                list(source.get_parameter(own).shape),
                source.get_parameter(own).detach().numpy().tobytes(),
            )
            for own, theirs in drafter.layout(source).items()
        },
    )
    model = drafter.Drafter.dflash2(sizes)
    assert drafter.load(model, weights.read(path)) == []
    for own, parameter in model.named_parameters():
        assert torch.equal(parameter, source.get_parameter(own)), own


def e4m3(code: int) -> float:
    """An E4M3 code's value, from its bits: a sign, 4 of exponent, 3 of mantissa."""
    sign = -1.0 if code >> 7 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (8 + mantissa) * 2.0 ** (exponent - 10)


def test_an_mxfp8_value_is_its_code_times_its_groups_power_of_two():
    # 1, 2, 1.5, -1, the smallest step, the largest value, and both zeros
    codes = [0x38, 0x40, 0x3C, 0xB8, 0x01, 0x7E, 0x00, 0x80]
    row = torch.tensor([codes + [0x38] * 24 + [0x38] * 32], dtype=torch.uint8)
    read = weights.mxfp8(row, torch.tensor([[127, 130]], dtype=torch.uint8))
    assert read.dtype == torch.float32
    assert read[0, :8].tolist() == [1.0, 2.0, 1.5, -1.0, 2.0**-9, 448.0, 0.0, -0.0]
    assert read[0, 7].signbit()
    assert read[0, 8:32].eq(1.0).all() and read[0, 32:].eq(8.0).all()


def test_every_mxfp8_code_is_read_as_its_bits_say():
    numbers = [code for code in range(256) if code & 0x7F != 0x7F]
    codes = torch.tensor(numbers + [0, 0], dtype=torch.uint8).reshape(4, 64)
    scales = torch.tensor([[127, 128], [126, 0], [254 - 9, 120], [1, 127]])
    read = weights.mxfp8(codes, scales.to(torch.uint8))
    for r in range(4):
        for c in range(64):
            power = 2.0 ** (int(scales[r, c // 32]) - 127)
            assert read[r, c].item() == e4m3(int(codes[r, c])) * power, (r, c)


def test_mxfp8_rows_are_read_the_same_part_by_part(monkeypatch):
    generator = torch.Generator().manual_seed(5)
    codes = torch.randint(0, 0x7F, (7, 64), generator=generator, dtype=torch.uint8)
    scales = torch.randint(100, 150, (7, 2), generator=generator, dtype=torch.uint8)
    whole = weights.mxfp8(codes, scales)
    assert len({tuple(row) for row in whole.tolist()}) == 7  # no two rows alike
    monkeypatch.setattr(weights, "MX_ROWS", 3)
    assert torch.equal(weights.mxfp8(codes, scales), whole)


def test_mxfp8_refuses_what_is_no_number():
    ones = torch.full((2, 32), 0x38, dtype=torch.uint8)
    middle = torch.full((2, 1), 127, dtype=torch.uint8)
    for code in (0x7F, 0xFF):
        wrong = ones.clone()
        wrong[1, 5] = code
        with pytest.raises(ValueError, match="no number"):
            weights.mxfp8(wrong, middle)
    with pytest.raises(ValueError, match="no number"):
        weights.mxfp8(ones, torch.tensor([[127], [255]], dtype=torch.uint8))
    with pytest.raises(ValueError, match="too large a value"):  # 448 * 2 ** 127
        large = torch.full((2, 32), 0x7E, dtype=torch.uint8)
        weights.mxfp8(large, torch.full((2, 1), 254, dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"codes \(2, 32\) do not go with scales"):
        weights.mxfp8(ones, torch.full((2, 2), 127, dtype=torch.uint8))
    with pytest.raises(ValueError, match=r"codes \(2, 48\) do not go with scales"):
        weights.mxfp8(torch.zeros(2, 48, dtype=torch.uint8), middle)


def test_bytes_come_as_bytes(tmp_path):
    path = saved(
        tmp_path,
        {
            "w": ("F8_E4M3", [1, 2], bytes([0x38, 0xB8])),
            "s": ("U8", [1, 1], bytes([127])),
        },
    )
    read = weights.read(path)
    assert read["w"].dtype == read["s"].dtype == torch.uint8
    assert read["w"].tolist() == [[0x38, 0xB8]] and read["s"].tolist() == [[127]]


def test_a_tensors_file_is_the_one_the_index_names(tmp_path):
    assert weights.holder(tmp_path, "a") == tmp_path / "model.safetensors"
    index = {"metadata": {}, "weight_map": {"a": "one.safetensors", "b": "../two"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert weights.holder(tmp_path, "a") == tmp_path / "one.safetensors"
    with pytest.raises(KeyError, match="no tensor named c"):
        weights.holder(tmp_path, "c")
    with pytest.raises(ValueError, match="b is in '../two', not a file beside it"):
        weights.holder(tmp_path, "b")


def test_a_snapshot_whose_files_are_links_is_read_through_them(tmp_path):
    # a Hugging Face cache snapshot: its files are links into ../../blobs
    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    shard = "model-00001-of-00001.safetensors"
    head = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    held = {weights.HEAD: ("F32", [3, 2], head.numpy().tobytes())}
    (blobs / "a").write_bytes(safetensors_blob(held))
    (blobs / "b").write_text(json.dumps({"weight_map": {weights.HEAD: shard}}))
    try:
        for name, blob in ((shard, "a"), (weights.INDEX, "b")):
            os.symlink(os.path.join("..", "..", "blobs", blob), snapshot / name)
    except OSError:
        pytest.skip("this machine does not let a test make links")
    assert torch.equal(weights.head(snapshot), head)
