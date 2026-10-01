"""Separate browser and CDN caching without caching private or live data."""

from __future__ import annotations

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from climbing_elo.api.api_auth import is_private_api_path

DEFAULT_CACHE_CONTROL = "public, max-age=60, must-revalidate"
DEFAULT_CDN_CACHE_CONTROL = "public, s-maxage=600, stale-while-revalidate=3600"
STATIC_CACHE_CONTROL = "public, max-age=3600, must-revalidate"
STATIC_CDN_CACHE_CONTROL = "public, s-maxage=86400"
VERCEL_CDN_CACHE_CONTROL_HEADER = "vercel-cdn-cache-control"
NO_CACHE_PREFIXES = ("/live",)


class CacheControlMiddleware:
    """Cache public GETs; honor route overrides and deny shared private caches.

    Daily data gets a ten-minute CDN freshness window and up to an hour of
    stale-while-revalidate grace. Browsers revalidate after one minute. Live
    routes, health checks, authenticated requests and cookie-setting responses
    never enter shared caches. Static files keep their ETag/Last-Modified
    validators and gain an explicit browser/CDN freshness window.
    """

    def __init__(self, app: ASGIApp, header_value: str = DEFAULT_CACHE_CONTROL) -> None:
        self.app = app
        self.header_value = header_value

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        private = (
            is_private_api_path(path)
            or path == "/health"
            or path == "/live"
            or path.startswith("/live/")
            or "authorization" in Headers(scope=scope)
        )

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                policy = headers.get("cache-control", "").lower()
                directives = {
                    part.strip().split("=", 1)[0] for part in policy.split(",")
                }
                if (
                    private
                    or "set-cookie" in headers
                    or directives & {"private", "no-store", "no-cache"}
                ):
                    headers["cache-control"] = "private, no-store"
                    headers[VERCEL_CDN_CACHE_CONTROL_HEADER] = "no-store"
                elif scope.get("method") == "GET" and message["status"] == 200:
                    if path.startswith("/static/") or path == "/favicon.ico":
                        headers.setdefault("cache-control", STATIC_CACHE_CONTROL)
                        headers.setdefault(
                            VERCEL_CDN_CACHE_CONTROL_HEADER, STATIC_CDN_CACHE_CONTROL
                        )
                    elif policy:
                        # A route's own TTL must also control the CDN, rather
                        # than silently getting the default public policy.
                        headers.setdefault(
                            VERCEL_CDN_CACHE_CONTROL_HEADER, headers["cache-control"]
                        )
                    else:
                        headers["cache-control"] = self.header_value
                        headers.setdefault(
                            VERCEL_CDN_CACHE_CONTROL_HEADER, DEFAULT_CDN_CACHE_CONTROL
                        )
            await send(message)

        await self.app(scope, receive, send_wrapper)
