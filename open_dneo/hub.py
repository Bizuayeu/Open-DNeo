"""Read a public Hugging Face repository by HTTP range requests, without credentials."""

from __future__ import annotations

import json
import struct
import urllib.request
from collections.abc import Callable

HUB = "https://huggingface.co"
USER_AGENT = "open-dneo"
# safetensors' own limit on a header (MAX_HEADER_SIZE in the reference implementation).
MAX_HEADER_BYTES = 100_000_000
# cc-defer: one timeout for a small header and a 160 MB tensor; split it when a slow link
# cuts a large tensor off.
TIMEOUT_S = 600

Fetch = Callable[[str, "tuple[int, int] | None"], bytes]


def http_fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    """GET `url`, or only its inclusive byte range."""
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return response.read()


def model_info(repo: str, revision: str, fetch: Fetch = http_fetch) -> dict:
    """The Hub's record of `repo` at `revision`: its commit, card data and files."""
    url = f"{HUB}/api/models/{repo}/revision/{revision}?blobs=true"
    return json.loads(fetch(url, None))


def file_url(repo: str, commit: str, name: str) -> str:
    """Where the Hub serves file `name` of `repo` at `commit`."""
    return f"{HUB}/{repo}/resolve/{commit}/{name}"


def safetensors_header(url: str, fetch: Fetch = http_fetch) -> tuple[dict, int]:
    """A safetensors file's header, and the offset at which its tensor data starts."""
    (length,) = struct.unpack("<Q", fetch(url, (0, 7)))
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError(f"{url}: a safetensors header of {length} bytes is refused")
    header = json.loads(fetch(url, (8, 8 + length - 1)))
    return header, 8 + length


def tensor_bytes(
    url: str, header: dict, data_start: int, name: str, fetch: Fetch = http_fetch
) -> bytes:
    """The raw bytes of tensor `name`, fetched alone."""
    if name not in header:
        raise KeyError(f"{url}: no tensor named {name}")
    begin, end = header[name]["data_offsets"]
    return fetch(url, (data_start + begin, data_start + end - 1))
