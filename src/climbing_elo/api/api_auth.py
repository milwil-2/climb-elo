"""Keep the standalone API and its interactive documentation private."""

from __future__ import annotations

import secrets
import math
import time

from limits import parse

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from climbing_elo.api.limiter import client_ip, limiter

_FAILED_AUTH_LIMIT = parse("60/minute")


def is_private_api_path(path: str) -> bool:
    return (
        path == "/api/v1"
        or path.startswith("/api/v1/")
        or path in {"/docs", "/redoc", "/openapi.json"}
        or path.startswith("/docs/")
    )


class PrivateAPIMiddleware:
    """Authenticate before parsing request bodies or accessing the database.

    An unset key disables standalone API access. Secrets are accepted only in
    the Authorization header, never in URLs or browser-facing JavaScript.
    Authenticated responses and authentication errors cannot enter a CDN cache.
    """

    def __init__(self, app: ASGIApp, api_key: str | None = None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_private_api_path(scope["path"]):
            await self.app(scope, receive, send)
            return

        authorization = Headers(scope=scope).get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if (
            not self.api_key
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(token.encode(), self.api_key.encode())
        ):
            key = client_ip(Request(scope))
            if not limiter.limiter.hit(_FAILED_AUTH_LIMIT, key, "api-auth"):
                reset, _ = limiter.limiter.get_window_stats(
                    _FAILED_AUTH_LIMIT, key, "api-auth"
                )
                response = JSONResponse(
                    {"detail": "Too many API authentication attempts."},
                    status_code=429,
                    headers={
                        "Retry-After": str(max(1, math.ceil(reset - time.time()))),
                        "Cache-Control": "private, no-store",
                        "Vercel-CDN-Cache-Control": "no-store",
                    },
                )
                await response(scope, receive, send)
                return
            response = JSONResponse(
                {"detail": "Private API: a valid bearer token is required."},
                status_code=401,
                headers={
                    "WWW-Authenticate": "Bearer",
                    "Cache-Control": "private, no-store",
                    "Vercel-CDN-Cache-Control": "no-store",
                },
            )
            await response(scope, receive, send)
            return

        async def send_private(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                headers["cache-control"] = "private, no-store"
                headers["vercel-cdn-cache-control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_private)
