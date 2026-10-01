"""Tests for application-level per-IP rate limiting (Issue #34).

Coverage:
- POST /api/v1/projections is limited to 10 req/min per IP; 11th request → 429
- 429 response includes a Retry-After header and does not leak sensitive info
- GET /api/v1/predictions/upcoming is limited to 10 req/min; limit enforced
- GET /api/v1/leaderboard (default limit) uses 120/min; not hit by short bursts
- HTML route GET / shares the 120/min default; not hit by short bursts
- limiter.reset() clears state between tests (in-memory backend)

Design notes
------------
slowapi's default in-memory backend stores counters keyed on
``<key_func_result>:<endpoint_name>``.  The FastAPI TestClient uses a
loopback ``request.client.host`` of ``"testclient"`` for all requests.
We call ``limiter.reset()`` in an ``autouse`` fixture so each test starts
with a clean slate.

The tests drive requests to structured endpoints — they do NOT start the real
server.  create_app() is patched so routes use the test DB (same monkey-patch
pattern as test_api.py).
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import climbing_elo.api.v1_routes as _v1
import climbing_elo.database as _db
from climbing_elo.api.app import create_app
from climbing_elo.api.limiter import limiter
from climbing_elo.api.limiter import client_ip
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
)


# ---------------------------------------------------------------------------
# Module-scoped test DB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("rl_db") / "test.db"


@pytest.fixture(scope="module")
def test_factory(test_db_path):
    """Create a minimal seeded DB for rate-limit tests."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    adam = Athlete(name="Adam Ondra", gender=Gender.M, nationality="CZE")
    janja = Athlete(name="Janja Garnbret", gender=Gender.F, nationality="SVN")
    session.add_all([adam, janja])
    session.flush()

    session.add(
        Rating(
            athlete_id=adam.id,
            discipline=Discipline.LEAD,
            mu=1750.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=date(2024, 6, 1),
        )
    )
    session.add(
        Rating(
            athlete_id=janja.id,
            discipline=Discipline.LEAD,
            mu=1800.0,
            sigma=100.0,
            n_events=12,
            provisional=False,
            last_event_at=date(2024, 6, 1),
        )
    )

    event = Event(
        name="Innsbruck World Cup",
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=2024,
        start_date=date(2024, 6, 1),
        discipline=Discipline.LEAD,
    )
    session.add(event)
    session.commit()
    session.close()

    # Store IDs for use in tests
    s2 = factory()
    adam_id = s2.query(Athlete).filter_by(name="Adam Ondra").one().id
    janja_id = s2.query(Athlete).filter_by(name="Janja Garnbret").one().id
    s2.close()

    factory._adam_id = adam_id
    factory._janja_id = janja_id

    return factory


@pytest.fixture(scope="module")
def client(test_db_path, test_factory):
    """TestClient wired to the test DB."""
    original_session = _v1._session
    original_get_engine = _db.get_engine

    def patched_session():
        return test_factory()

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{test_db_path}")

    _v1._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(
        app,
        raise_server_exceptions=False,
        headers={"Authorization": "Bearer test-api-key"},
    )

    yield tc

    _db.get_engine = original_get_engine
    _v1._session = original_session


# ---------------------------------------------------------------------------
# Reset limiter state between every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset in-memory rate-limit counters before each test."""
    limiter.reset()
    yield
    limiter.reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _athlete_ids(test_factory) -> list[int]:
    return [test_factory._adam_id, test_factory._janja_id]


def _post_projection(client, test_factory) -> int:
    resp = client.post(
        "/api/v1/projections",
        json={"discipline": "lead", "athlete_ids": _athlete_ids(test_factory)},
    )
    return resp.status_code


# ---------------------------------------------------------------------------
# Tests — POST /api/v1/projections  (10/minute limit)
# ---------------------------------------------------------------------------


class TestProjectionsRateLimit:
    def test_first_ten_requests_succeed(self, client, test_factory):
        """First 10 POST /api/v1/projections requests within a minute → 200."""
        for i in range(10):
            status = _post_projection(client, test_factory)
            assert status == 200, f"Request {i + 1} expected 200, got {status}"

    def test_eleventh_request_is_429(self, client, test_factory):
        """11th request within the same minute → 429."""
        for _ in range(10):
            _post_projection(client, test_factory)

        resp = client.post(
            "/api/v1/projections",
            json={"discipline": "lead", "athlete_ids": _athlete_ids(test_factory)},
        )
        assert resp.status_code == 429

    def test_429_includes_retry_after_header(self, client, test_factory):
        """429 response must carry a Retry-After header."""
        for _ in range(10):
            _post_projection(client, test_factory)

        resp = client.post(
            "/api/v1/projections",
            json={"discipline": "lead", "athlete_ids": _athlete_ids(test_factory)},
        )
        assert resp.status_code == 429
        assert "retry-after" in {h.lower() for h in resp.headers}

    def test_429_body_is_json_and_not_sensitive(self, client, test_factory):
        """429 body is JSON with an error message; no sensitive data."""
        for _ in range(10):
            _post_projection(client, test_factory)

        resp = client.post(
            "/api/v1/projections",
            json={"discipline": "lead", "athlete_ids": _athlete_ids(test_factory)},
        )
        assert resp.status_code == 429
        # Must be parseable JSON
        body = resp.json()
        assert isinstance(body, dict)
        # The default slowapi handler returns {"error": "Rate limit exceeded: ..."}.
        # Verify it contains a user-facing message only — no stack traces, file paths, etc.
        assert "error" in body
        error_text = body["error"]
        assert "rate limit" in error_text.lower() or "10" in error_text
        # No Python internals leaked
        assert "traceback" not in error_text.lower()
        assert "/Users" not in error_text
        assert "sqlite" not in error_text.lower()


def test_expensive_html_routes_share_one_limit(client):
    for _ in range(6):
        assert client.get("/predictions").status_code == 200
        assert client.get("/projections").status_code == 200
    response = client.get("/predictions/999999")
    assert response.status_code == 429
    assert "retry-after" in response.headers


def test_public_search_has_a_limit_and_needs_no_api_key(client):
    public = TestClient(client.app)
    for _ in range(60):
        assert public.get("/search/athletes?q=Adam").status_code == 200
    response = public.get("/search/athletes?q=Janja")
    assert response.status_code == 429


def test_forwarded_ip_is_trusted_only_on_vercel(monkeypatch):
    from starlette.requests import Request

    request = Request(
        {
            "type": "http",
            "client": ("127.0.0.1", 1234),
            "headers": [(b"x-vercel-forwarded-for", b"203.0.113.7")],
        }
    )
    monkeypatch.delenv("VERCEL", raising=False)
    assert client_ip(request) == "127.0.0.1"
    monkeypatch.setenv("VERCEL", "1")
    assert client_ip(request) == "203.0.113.7"


def test_invalid_forwarded_ip_uses_socket_peer(monkeypatch):
    from starlette.requests import Request

    monkeypatch.setenv("VERCEL", "1")
    request = Request(
        {
            "type": "http",
            "client": ("127.0.0.1", 1234),
            "headers": [(b"x-vercel-forwarded-for", b"spoofed, 203.0.113.7")],
        }
    )
    assert client_ip(request) == "127.0.0.1"


# ---------------------------------------------------------------------------
# Tests — GET /api/v1/predictions/upcoming  (10/minute limit)
# ---------------------------------------------------------------------------


class TestPredictionsUpcomingRateLimit:
    def test_first_ten_requests_succeed(self, client):
        """First 10 GET /api/v1/predictions/upcoming → 200 (or 200 with no data)."""
        for i in range(10):
            resp = client.get("/api/v1/predictions/upcoming")
            assert resp.status_code == 200, (
                f"Request {i + 1} expected 200, got {resp.status_code}"
            )

    def test_11th_request_is_429(self, client):
        """11th request within the same minute → 429."""
        for _ in range(10):
            client.get("/api/v1/predictions/upcoming")

        resp = client.get("/api/v1/predictions/upcoming")
        assert resp.status_code == 429

    def test_429_has_retry_after(self, client):
        for _ in range(10):
            client.get("/api/v1/predictions/upcoming")

        resp = client.get("/api/v1/predictions/upcoming")
        assert resp.status_code == 429
        assert "retry-after" in {h.lower() for h in resp.headers}


# ---------------------------------------------------------------------------
# Tests — default limit (120/min) on other GET /api/v1/* endpoints
# ---------------------------------------------------------------------------


class TestDefaultLimitNotHitBySmallBursts:
    def test_leaderboard_small_burst_passes(self, client):
        """A burst of 30 requests to GET /api/v1/leaderboard all return 200."""
        for i in range(30):
            resp = client.get("/api/v1/leaderboard?discipline=lead&gender=M")
            assert resp.status_code == 200, (
                f"Request {i + 1} returned {resp.status_code}"
            )

    def test_disciplines_small_burst_passes(self, client):
        """A burst of 20 requests to GET /api/v1/disciplines all return 200."""
        for i in range(20):
            resp = client.get("/api/v1/disciplines")
            assert resp.status_code == 200, (
                f"Request {i + 1} returned {resp.status_code}"
            )


# ---------------------------------------------------------------------------
# Tests — default limit (120/min) on HTML routes
# ---------------------------------------------------------------------------


class TestHtmlRoutesDefaultLimit:
    def test_root_small_burst_passes(self, client):
        """A burst of 20 GET / requests all return 200."""
        for i in range(20):
            resp = client.get("/")
            assert resp.status_code == 200, (
                f"Request {i + 1} returned {resp.status_code}"
            )
