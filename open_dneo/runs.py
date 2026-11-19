"""A run: the requests a client sent a server, and what came back.

A run file holds a JSON line a request: "id", "class", "max_tokens", "prompt_ids"
(the prompt the server ran, as token ids), "reply_ids" (the reply's, from the
server's `return_token_ids`), "completion_tokens", "speculative" (serving's
"rounds" among them) and "status". `read` keeps the requests the server answered.

A recording's streams carry no request's name. `pair` finds each request's: a
stream whose prompt is the request's (its prompt length the same, and its rows
from its first row on the request's prompt from there) and whose reply rows begin
the request's reply. Streams are taken in the order they opened, each once, so two
requests with the same prompt get their own streams in turn.

`served` is serving's tokens a round over requests, as G0 counts it: a reply's
first token comes from the prompt pass and no round gives it, so each reply counts
its tokens less one, summed, over its rounds summed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import recording


@dataclass(frozen=True)
class Request:
    """One answered request of a run."""

    id: str
    klass: str
    max_tokens: int
    prompt: tuple[int, ...]
    reply: tuple[int, ...]
    rounds: int | None  # serving's, None where the reply did not say


def read(path: Path) -> list[Request]:
    """The answered requests of a run file, in its order.

    ValueError where a reply's ids are not as many as its completion tokens.
    """
    found = []
    # split at newlines alone: a request's text may hold U+2028, a line end to
    # str.splitlines()
    for text in Path(path).read_text(encoding="utf-8").split("\n"):
        if not text.strip():
            continue
        line = json.loads(text)
        if line.get("status") != 200:
            continue
        reply = tuple(line["reply_ids"])
        if line["completion_tokens"] != len(reply):
            raise ValueError(
                f"{line['id']}: {len(reply)} reply ids, and"
                f" {line['completion_tokens']} completion tokens"
            )
        found.append(
            Request(
                line["id"],
                line["class"],
                line["max_tokens"],
                tuple(line["prompt_ids"]),
                reply,
                (line.get("speculative") or {}).get("rounds"),
            )
        )
    return found


def _holds(stream: recording.Stream, request: Request) -> bool:
    """Whether `stream` is `request`'s: the same prompt, and reply rows it begins with."""
    summary = stream.summary
    tokens = stream.tok.tolist()
    kept = summary.prompt_len - summary.first_row
    replied = tuple(tokens[kept:])
    return (
        tuple(tokens[:kept]) == request.prompt[summary.first_row :]
        and replied == request.reply[: len(replied)]
    )


def pair(
    requests: Sequence[Request], streams: Sequence[tuple[str, recording.Stream]]
) -> list[tuple[Request, str | None]]:
    """Each request with the name of its stream, None where none is.

    `streams` are named, in the order they opened.
    """
    taken: set[int] = set()
    paired = []
    for request in requests:
        at = next(
            (
                i
                for i, (_, stream) in enumerate(streams)
                if i not in taken and _holds(stream, request)
            ),
            None,
        )
        if at is not None:
            taken.add(at)
        paired.append((request, None if at is None else streams[at][0]))
    return paired


def served(requests: Sequence[Request]) -> tuple[int, int]:
    """Serving's tokens and rounds over `requests`: each reply's tokens less one.

    ValueError where a reply did not say its rounds.
    """
    tokens = rounds = 0
    for request in requests:
        if request.rounds is None:
            raise ValueError(f"{request.id}: serving gave no rounds")
        tokens += len(request.reply) - 1
        rounds += request.rounds
    return tokens, rounds
