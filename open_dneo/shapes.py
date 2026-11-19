"""List a checkpoint's files, config and tensors, reading only safetensors headers."""

from __future__ import annotations

import math

from . import hub


def tensor_lines(header: dict) -> list[str]:
    """One line a tensor (name, dtype, shape) in name order, then the parameter total."""
    tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
    lines = [
        f"  {name:<72} {entry['dtype']:<5} {entry['shape']}"
        for name, entry in sorted(tensors.items())
    ]
    total = sum(math.prod(entry["shape"]) for entry in tensors.values())
    lines.append(f"  total parameters {total}")
    return lines


def report(repo: str, revision: str, fetch: hub.Fetch = hub.http_fetch) -> list[str]:
    """What `repo` holds at `revision`: files, config.json, and every tensor's shape."""
    info = hub.model_info(repo, revision, fetch)
    commit = info["sha"]
    card = info.get("cardData") or {}
    lines = [
        f"repo {repo} commit {commit}",
        f"license {card.get('license')} base_model {card.get('base_model')}",
    ]
    names = [sibling["rfilename"] for sibling in info.get("siblings", [])]
    sizes = {s["rfilename"]: s.get("size") for s in info.get("siblings", [])}
    lines += [f"file {name} {sizes[name]}" for name in names]
    if "config.json" in names:
        config = fetch(hub.file_url(repo, commit, "config.json"), None)
        lines += ["== config.json", config.decode("utf-8").rstrip()]
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        header, _ = hub.safetensors_header(hub.file_url(repo, commit, name), fetch)
        lines += [f"== {name}: {len(header) - ('__metadata__' in header)} tensors"]
        lines += tensor_lines(header)
    return lines
