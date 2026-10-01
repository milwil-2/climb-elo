"""Browser/CDN policy and regressions for shared-cache disclosure."""

import pytest
from fastapi.responses import Response
from fastapi.testclient import TestClient

from climbing_elo.api.app import create_app
from climbing_elo.api.cache_headers import (
    DEFAULT_CACHE_CONTROL,
    DEFAULT_CDN_CACHE_CONTROL,
    STATIC_CACHE_CONTROL,
    STATIC_CDN_CACHE_CONTROL,
)


@pytest.fixture
def client():
    app = create_app()

    @app.get("/cache-test")
    def public_page():
        return Response("public")

    @app.get("/cache-test/custom")
    def custom_page():
        return Response("custom", headers={"Cache-Control": "public, max-age=15"})

    @app.get("/cache-test/private")
    def private_page():
        return Response(
            "private",
            headers={
                "Cache-Control": "private, no-store",
                "Vercel-CDN-Cache-Control": "public, s-maxage=600",
            },
        )

    @app.get("/cache-test/cookie")
    def cookie_page():
        response = Response("cookie", headers={"Cache-Control": "public, max-age=15"})
        response.set_cookie("session", "secret")
        return response

    return TestClient(app)


def test_public_get_has_separate_browser_and_cdn_policy(client):
    response = client.get("/cache-test")
    assert response.status_code == 200
    assert response.headers["cache-control"] == DEFAULT_CACHE_CONTROL
    assert response.headers["vercel-cdn-cache-control"] == DEFAULT_CDN_CACHE_CONTROL


def test_route_ttl_also_controls_the_cdn(client):
    response = client.get("/cache-test/custom")
    assert response.headers["cache-control"] == "public, max-age=15"
    assert response.headers["vercel-cdn-cache-control"] == "public, max-age=15"


@pytest.mark.parametrize(
    "path",
    ["/cache-test/private", "/cache-test/cookie", "/health", "/live/not-an-event-id"],
)
def test_private_cookie_and_live_responses_cannot_enter_public_cache(client, path):
    response = client.get(path)
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["vercel-cdn-cache-control"] == "no-store"


def test_authorization_prevents_shared_caching_even_on_public_routes(client):
    response = client.get("/cache-test", headers={"Authorization": "Bearer owner"})
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["vercel-cdn-cache-control"] == "no-store"


def test_static_files_cache_and_revalidate_with_etag(client):
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert response.headers["cache-control"] == STATIC_CACHE_CONTROL
    assert response.headers["vercel-cdn-cache-control"] == STATIC_CDN_CACHE_CONTROL
    conditional = client.get(
        "/static/styles.css", headers={"If-None-Match": response.headers["etag"]}
    )
    assert conditional.status_code == 304
    assert conditional.content == b""


def test_non_get_and_errors_are_not_publicly_cached(client):
    for response in (client.post("/cache-test"), client.get("/not-found")):
        assert "public" not in response.headers.get("cache-control", "")
        assert "public" not in response.headers.get("vercel-cdn-cache-control", "")
