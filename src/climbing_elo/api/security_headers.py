"""Security response headers, including a Content-Security-Policy (Issue #208).

Defense-in-depth: the dashboard is a read-only, no-auth site, but a strict CSP
plus the standard hardening headers shrink the blast radius of any future XSS
(e.g. a reflected value that slips through Jinja autoescaping) and stop
clickjacking / MIME-sniffing.

Implemented as a **pure-ASGI** middleware (like ``CacheControlMiddleware``) so it
never buffers the response body - important for the SSE streaming endpoint under
``/live``.

Scope
-----
Headers are applied only to **HTML** responses (``Content-Type: text/html``) so
JSON API payloads and static assets are untouched. The interactive API docs are
excluded entirely: Swagger UI (``/docs``), ReDoc (``/redoc``) and the schema
(``/openapi.json``) pull their own CDN assets and rely on inline scripts/styles
and blob workers that a strict CSP would break.

Content-Security-Policy
-----------------------
Built to allow exactly what the templates load and nothing else:

* ``default-src 'self'`` - same-origin by default.
* ``script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net`` - Chart.js is
  loaded from jsDelivr (SRI-pinned). ``'unsafe-inline'`` is a **known
  relaxation**: the templates carry many inline ``<script>`` blocks (page-local
  glue: theme toggle, chart bootstrapping, filter handlers) plus a handful of
  inline event-handler attributes (``onload``/``onerror``/``onclick``/
  ``onchange``). Externalising all of these (or moving to per-request nonces) is
  a larger refactor tracked as a follow-up; until then inline scripts must run.
* ``style-src 'self' 'unsafe-inline' https://fonts.googleapis.com`` - every
  template ships an inline ``<style>`` block and inline ``style=`` attributes,
  and the Google Fonts stylesheet is loaded from ``fonts.googleapis.com``.
* ``img-src 'self' data: https://ifsc.results.info`` - athlete photos are
  hot-linked from ``ifsc.results.info``; ``data:`` covers inline SVG/data URIs.
* ``font-src 'self' https://fonts.gstatic.com`` - Google Fonts webfont files.
* ``connect-src 'self'`` - same-origin fetch/XHR + the ``/live`` SSE stream.
* ``frame-src https://www.youtube.com https://www.youtube-nocookie.com`` - the
  ``/live/{id}`` livestream embed.
* ``frame-ancestors 'self'``, ``base-uri 'self'``, ``object-src 'none'`` -
  anti-clickjacking / anti-base-tag-injection / no plugins.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: The Content-Security-Policy applied to HTML responses. See module docstring
#: for the rationale behind each directive.
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "base-uri 'self'",
        "object-src 'none'",
        "frame-ancestors 'self'",
        "img-src 'self' data: https://ifsc.results.info",
        "font-src 'self' https://fonts.gstatic.com",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "connect-src 'self'",
        "frame-src https://www.youtube.com https://www.youtube-nocookie.com",
    )
)

#: Static hardening headers set alongside the CSP on HTML responses.
SECURITY_HEADERS: dict[str, str] = {
    "content-security-policy": CONTENT_SECURITY_POLICY,
    "x-content-type-options": "nosniff",
    "x-frame-options": "SAMEORIGIN",
    "referrer-policy": "strict-origin-when-cross-origin",
}

#: Path prefixes served with their own CDN assets + inline scripts/styles that a
#: strict CSP would break - the interactive API docs. Excluded entirely.
DOCS_PREFIXES = ("/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware:
    """Attach CSP + hardening headers to HTML responses (Issue #208)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path", "").startswith(DOCS_PREFIXES):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                content_type = headers.get("content-type", "")
                if content_type.startswith("text/html"):
                    for name, value in SECURITY_HEADERS.items():
                        headers[name] = value
            await send(message)

        await self.app(scope, receive, send_wrapper)
