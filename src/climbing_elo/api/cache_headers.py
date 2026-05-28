"""Edge cache-control middleware (Issue #97, Tier 1).

Tags cacheable GET responses with a short shared-cache ``Cache-Control`` so
Vercel's edge (and browsers) can serve repeat hits without re-invoking the
Python function. Rating data refreshes once daily (the 04:00 UTC scrape), so a
brief edge TTL with ``stale-while-revalidate`` is safe: the edge serves
instantly and refreshes in the background.

Implemented as a **pure-ASGI** middleware rather than ``BaseHTTPMiddleware`` so
it never buffers the response body — important for the SSE streaming endpoint
under ``/live`` (which we also exclude from caching anyway).

Excluded from caching:

* ``/live`` — real-time event pages + the SSE stream. Must never serve stale
  live scores; the SSE handler sets its own ``no-cache``.
* ``/static`` — ``StaticFiles`` already emits ETag / Last-Modified validators.

Only ``GET`` requests returning ``200`` are tagged, and the header is set with
``setdefault`` so any route that sets its own ``Cache-Control`` wins.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Shared-cache directive. ``s-maxage`` targets the CDN/edge specifically
#: (browsers use the implicit default); ``stale-while-revalidate`` lets the
#: edge serve a slightly-stale copy instantly while it refreshes in the
#: background. 5 min fresh / 10 min stale-grace is well within the daily
#: data-refresh cadence.
DEFAULT_CACHE_CONTROL = "public, s-maxage=300, stale-while-revalidate=600"

#: Path prefixes that must NOT be edge-cached.
NO_CACHE_PREFIXES = ("/live", "/static")


class CacheControlMiddleware:
    """Set a short shared-cache ``Cache-Control`` on cacheable GET 200s."""

    def __init__(self, app: ASGIApp, header_value: str = DEFAULT_CACHE_CONTROL) -> None:
        self.app = app
        self.header_value = header_value

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "GET":
            await self.app(scope, receive, send)
            return
        if scope["path"].startswith(NO_CACHE_PREFIXES):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start" and message["status"] == 200:
                headers = MutableHeaders(raw=message["headers"])
                headers.setdefault("cache-control", self.header_value)
            await send(message)

        await self.app(scope, receive, send_wrapper)
