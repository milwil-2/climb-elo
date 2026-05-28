"""Tests for the edge cache-control middleware (Issue #97, Tier 1)."""

from fastapi.testclient import TestClient

from climbing_elo.api.app import create_app
from climbing_elo.api.cache_headers import DEFAULT_CACHE_CONTROL

# raise_server_exceptions=False so routes that touch the (unseeded) throwaway
# DB surface as 500 responses rather than re-raising — we only care about the
# response headers, not the route's data behaviour.
client = TestClient(create_app(), raise_server_exceptions=False)


def test_get_api_route_has_cache_control():
    """A read-only GET API route gets the shared-cache header."""
    r = client.get("/api/v1/disciplines")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == DEFAULT_CACHE_CONTROL
    assert "s-maxage" in r.headers["cache-control"]
    assert "stale-while-revalidate" in r.headers["cache-control"]


def test_get_api_route_has_vercel_cdn_cache_control():
    """The same cacheable GET 200 also carries the Vercel CDN directive
    (Issue #101) — this is the header Vercel's edge actually honors. Same
    policy value as plain Cache-Control."""
    r = client.get("/api/v1/disciplines")
    assert r.status_code == 200
    assert r.headers.get("vercel-cdn-cache-control") == DEFAULT_CACHE_CONTROL


def test_docs_route_has_cache_control():
    """A generic GET 200 (FastAPI docs) is tagged — broad coverage check."""
    r = client.get("/docs")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == DEFAULT_CACHE_CONTROL
    assert r.headers.get("vercel-cdn-cache-control") == DEFAULT_CACHE_CONTROL


def test_live_prefix_in_no_cache_list():
    """`/live` (real-time pages + SSE) is excluded by prefix."""
    from climbing_elo.api.cache_headers import NO_CACHE_PREFIXES

    assert "/live" in NO_CACHE_PREFIXES
    assert "/static" in NO_CACHE_PREFIXES


def test_live_path_excluded_from_edge_cache():
    """A `/live` path never carries the public shared-cache directive,
    regardless of response status — including the Vercel CDN header (#101)."""
    r = client.get("/live/99999999")
    assert "public" not in r.headers.get("cache-control", "")
    assert "public" not in r.headers.get("vercel-cdn-cache-control", "")


def test_static_path_excluded():
    """`/static` is served by StaticFiles with its own validators; the
    middleware must not stamp the shared-cache directive on top."""
    # styles.css is the canonical static asset; if absent the 404 still must
    # not carry our public directive (plain or Vercel CDN — #101).
    r = client.get("/static/styles.css")
    assert "public, s-maxage" not in r.headers.get("cache-control", "")
    assert "public, s-maxage" not in r.headers.get("vercel-cdn-cache-control", "")


def test_non_get_not_cached():
    """Non-GET requests are never tagged (POST to the projections endpoint)."""
    # Missing/invalid body → 422, but the point is the method, not the status.
    r = client.post("/api/v1/projections", json={})
    assert "public, s-maxage" not in r.headers.get("cache-control", "")
    assert "public, s-maxage" not in r.headers.get("vercel-cdn-cache-control", "")


def test_404_get_not_tagged():
    """A 404 GET (router-level, no DB) must not be edge-cached — only 200s
    are tagged (plain or Vercel CDN header — #101)."""
    r = client.get("/this-route-does-not-exist")
    assert r.status_code == 404
    assert "public, s-maxage" not in r.headers.get("cache-control", "")
    assert "public, s-maxage" not in r.headers.get("vercel-cdn-cache-control", "")
