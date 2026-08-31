"""Shared HTTP helpers for upstream fetches.

Guards against decompression bombs: httpx transparently gunzips responses, so a
~1 MB gzip payload can expand to gigabytes once decoded. A plain
``client.get(...)`` buffers the whole decoded body in memory before we ever see
it, so the cap has to be applied while the body streams in.

These helpers stream an httpx response and abort as soon as the accumulated
*decoded* body exceeds :data:`MAX_RESPONSE_BYTES`. For normal-sized responses
the returned text/JSON is identical to ``resp.text`` / ``resp.json()``.

Usage (sync)::

    with client.stream("GET", url, headers=...) as resp:
        if resp.status_code == 200:
            data = read_capped_json(resp)

Usage (async)::

    async with client.stream("GET", url, headers=...) as resp:
        if resp.status_code == 200:
            data = await aread_capped_json(resp)
"""

from __future__ import annotations

import json
from typing import Any

import httpx

#: Maximum decoded (post-gzip) response body we will buffer, in bytes (~20 MB).
MAX_RESPONSE_BYTES = 20 * 1024 * 1024


class ResponseTooLargeError(Exception):
    """Raised when an upstream response's decoded body exceeds the size cap."""


def _decode(raw: bytes, response: httpx.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except (LookupError, TypeError):
        return raw.decode("utf-8", errors="replace")


def read_capped_bytes(
    response: httpx.Response, max_bytes: int = MAX_RESPONSE_BYTES
) -> bytes:
    """Read a streamed response's decoded body, aborting past ``max_bytes``.

    ``response`` must come from ``client.stream(...)`` (body not yet consumed).
    Raises :class:`ResponseTooLargeError` once the accumulated decoded size
    exceeds ``max_bytes``.
    """
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError(
                f"Upstream response exceeded {max_bytes}-byte decoded cap"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def aread_capped_bytes(
    response: httpx.Response, max_bytes: int = MAX_RESPONSE_BYTES
) -> bytes:
    """Async counterpart of :func:`read_capped_bytes`."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError(
                f"Upstream response exceeded {max_bytes}-byte decoded cap"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def read_capped_text(
    response: httpx.Response, max_bytes: int = MAX_RESPONSE_BYTES
) -> str:
    """Return the streamed response body as text, capped at ``max_bytes``."""
    return _decode(read_capped_bytes(response, max_bytes), response)


def read_capped_json(
    response: httpx.Response, max_bytes: int = MAX_RESPONSE_BYTES
) -> Any:
    """Return the streamed response body parsed as JSON, capped at ``max_bytes``."""
    return json.loads(read_capped_text(response, max_bytes))


async def aread_capped_text(
    response: httpx.Response, max_bytes: int = MAX_RESPONSE_BYTES
) -> str:
    """Async counterpart of :func:`read_capped_text`."""
    return _decode(await aread_capped_bytes(response, max_bytes), response)


async def aread_capped_json(
    response: httpx.Response, max_bytes: int = MAX_RESPONSE_BYTES
) -> Any:
    """Async counterpart of :func:`read_capped_json`."""
    return json.loads(await aread_capped_text(response, max_bytes))
