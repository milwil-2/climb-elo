"""Tests for HTML/static utility routes (Issue #73).

Covers:
- GET /favicon.ico returns 200 with an image content-type (no longer 500/404).
- base.html includes a <link rel="icon"> tag so browser tabs render the icon.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from climbing_elo.api.app import create_app


# ---------------------------------------------------------------------------
# /favicon.ico
# ---------------------------------------------------------------------------


def test_favicon_route_returns_200():
    """GET /favicon.ico must return 200 — no more 500s on browser tab opens."""
    app = create_app()
    with TestClient(app) as tc:
        r = tc.get("/favicon.ico")
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"


def test_favicon_route_has_image_content_type():
    """The favicon response must declare an image content-type."""
    app = create_app()
    with TestClient(app) as tc:
        r = tc.get("/favicon.ico")
    ctype = r.headers.get("content-type", "")
    accepted = (
        "image/x-icon",
        "image/png",
        "image/svg+xml",
        "image/vnd.microsoft.icon",
    )
    assert any(ctype.startswith(t) for t in accepted), (
        f"expected one of {accepted}, got {ctype!r}"
    )


def test_favicon_response_has_body():
    """The favicon response must include a non-empty body."""
    app = create_app()
    with TestClient(app) as tc:
        r = tc.get("/favicon.ico")
    assert len(r.content) > 0


# ---------------------------------------------------------------------------
# base.html <link rel="icon">
# ---------------------------------------------------------------------------


def test_base_html_includes_favicon_link_tag():
    """base.html must declare a <link rel="icon"> tag in its <head> block."""
    base_html_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "climbing_elo"
        / "templates"
        / "base.html"
    )
    text = base_html_path.read_text(encoding="utf-8")
    assert 'rel="icon"' in text, 'base.html missing <link rel="icon"> tag'
    # The link must reference the static favicon (not an external URL).
    assert "favicon" in text.lower()
