import json
import struct

import pytest

from open_dneo import hub


def safetensors_blob(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    """A safetensors file holding `tensors` (name: dtype, shape, raw bytes)."""
    header, data = {"__metadata__": {"format": "pt"}}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + data


def serving(files: dict[str, bytes]):
    """A fetch that serves `files` by URL and honours inclusive byte ranges."""

    def fetch(url, byte_range):
        blob = files[url]
        if byte_range is None:
            return blob
        return blob[byte_range[0] : byte_range[1] + 1]

    return fetch


def test_header_and_one_tensor_are_read_by_range():
    blob = safetensors_blob(
        {"a": ("F32", [2], b"\x01" * 8), "b": ("F32", [1], b"\x02" * 4)}
    )
    fetch = serving({"u": blob})
    header, data_start = hub.safetensors_header("u", fetch)
    assert header["b"]["shape"] == [1]
    assert hub.tensor_bytes("u", header, data_start, "a", fetch) == b"\x01" * 8
    assert hub.tensor_bytes("u", header, data_start, "b", fetch) == b"\x02" * 4


def test_a_missing_tensor_is_named():
    fetch = serving({"u": safetensors_blob({"a": ("F32", [1], b"\x00" * 4)})})
    header, data_start = hub.safetensors_header("u", fetch)
    with pytest.raises(KeyError, match="no tensor named fc.weight"):
        hub.tensor_bytes("u", header, data_start, "fc.weight", fetch)


@pytest.mark.parametrize("length", [0, hub.MAX_HEADER_BYTES + 1])
def test_a_header_length_out_of_range_is_refused(length):
    fetch = serving({"u": struct.pack("<Q", length) + b"{}"})
    with pytest.raises(ValueError, match="refused"):
        hub.safetensors_header("u", fetch)


def test_urls_name_the_commit():
    assert hub.file_url("o/n", "abc", "config.json") == (
        "https://huggingface.co/o/n/resolve/abc/config.json"
    )
