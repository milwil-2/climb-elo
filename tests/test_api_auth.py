"""Private API authentication must run before request parsing and DB work."""

import pytest
from fastapi.testclient import TestClient

from climbing_elo.api.app import create_app


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/disciplines",
        "/api/v1/",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
    ],
)
def test_private_routes_require_a_token(path):
    response = TestClient(create_app()).get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["vercel-cdn-cache-control"] == "no-store"


@pytest.mark.parametrize(
    "authorization",
    ["", "Bearer incorrect", "Basic test-api-key", "Bearer", "Bearer tést"],
)
def test_invalid_tokens_cannot_access_the_api(authorization):
    response = TestClient(create_app()).get(
        "/api/v1/disciplines",
        headers=[(b"Authorization", authorization.encode("latin-1"))],
    )
    assert response.status_code == 401
    assert "test-api-key" not in response.text


def test_authenticated_api_response_is_never_publicly_cached():
    response = TestClient(create_app()).get(
        "/api/v1/disciplines", headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    assert len(response.json()) == 4
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["vercel-cdn-cache-control"] == "no-store"


def test_missing_key_disables_api_access(monkeypatch):
    monkeypatch.delenv("CLIMBING_ELO_API_KEY")
    response = TestClient(create_app()).get(
        "/api/v1/disciplines", headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 401


def test_api_auth_rejects_before_parsing_invalid_json(monkeypatch):
    def database_must_not_run():
        pytest.fail("Unauthenticated request opened a database session")

    monkeypatch.setattr("climbing_elo.api.v1_routes._session", database_must_not_run)
    response = TestClient(create_app()).post(
        "/api/v1/projections",
        content="not JSON",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401


def test_query_string_is_not_accepted_as_a_secret():
    response = TestClient(create_app()).get("/api/v1/disciplines?api_key=test-api-key")
    assert response.status_code == 401


def test_failed_auth_is_rate_limited():
    client = TestClient(create_app())
    for _ in range(60):
        assert client.get("/api/v1/disciplines").status_code == 401
    response = client.get("/api/v1/disciplines")
    assert response.status_code == 429
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert response.headers["vercel-cdn-cache-control"] == "no-store"


def test_health_is_public_and_uncached():
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["vercel-cdn-cache-control"] == "no-store"
