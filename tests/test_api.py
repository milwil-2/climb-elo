"""Tests for the public v1 REST API endpoints.

Uses FastAPI's TestClient with a temporary SQLite file containing seeded test
data. The module-level _session() in v1_routes is monkey-patched to use the
test sessionmaker so no production DB is touched.
"""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import climbing_elo.api.v1_routes as _v1
from climbing_elo.api.app import create_app
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Module-scoped test DB + seeding
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("db") / "test.db"


@pytest.fixture(scope="module")
def test_factory(test_db_path):
    """Create and seed a persistent (file-based) SQLite DB for the module."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    # -- Athletes
    adam = Athlete(name="Adam Ondra", gender=Gender.M, nationality="CZE", year_of_birth=1993)
    janja = Athlete(name="Janja Garnbret", gender=Gender.F, nationality="SVN", year_of_birth=1999)
    session.add_all([adam, janja])
    session.flush()

    # -- Event
    event = Event(
        name="Innsbruck World Cup",
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=2024,
        start_date=date(2024, 6, 1),
        discipline=Discipline.LEAD,
    )
    session.add(event)
    session.flush()

    # -- Rounds
    rnd_m = Round(event_id=event.id, round_type=RoundType.FINAL, gender=Gender.M, athlete_count=1)
    rnd_f = Round(event_id=event.id, round_type=RoundType.FINAL, gender=Gender.F, athlete_count=1)
    session.add_all([rnd_m, rnd_f])
    session.flush()

    # -- Results
    session.add(Result(round_id=rnd_m.id, athlete_id=adam.id, rank=1, raw_score="TOP"))
    session.add(Result(round_id=rnd_f.id, athlete_id=janja.id, rank=1, raw_score="TOP"))
    session.flush()

    # -- Ratings
    session.add(Rating(
        athlete_id=adam.id, discipline=Discipline.LEAD,
        mu=1750.0, sigma=120.0, n_events=10, provisional=False,
        last_event_at=date(2024, 6, 1),
    ))
    session.add(Rating(
        athlete_id=janja.id, discipline=Discipline.LEAD,
        mu=1800.0, sigma=100.0, n_events=12, provisional=False,
        last_event_at=date(2024, 6, 1),
    ))
    session.flush()

    # -- Rating history
    session.add(RatingHistory(
        athlete_id=adam.id, event_id=event.id, round_id=rnd_m.id,
        mu_before=1740.0, mu_after=1750.0,
        sigma_before=125.0, sigma_after=120.0,
        contributing_pairs=[],
    ))
    session.add(RatingHistory(
        athlete_id=janja.id, event_id=event.id, round_id=rnd_f.id,
        mu_before=1790.0, mu_after=1800.0,
        sigma_before=105.0, sigma_after=100.0,
        contributing_pairs=[],
    ))
    session.commit()
    session.close()

    return factory


@pytest.fixture(scope="module")
def client(test_db_path, test_factory):
    """TestClient that uses the seeded test DB via monkey-patching _session()."""
    import climbing_elo.database as _db

    original_session = _v1._session
    original_get_engine = _db.get_engine

    def patched_session():
        return test_factory()

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{test_db_path}")

    _v1._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)

    yield tc

    # restore originals after all module-scoped tests complete
    _db.get_engine = original_get_engine
    _v1._session = original_session


# ---------------------------------------------------------------------------
# /api/v1/disciplines
# ---------------------------------------------------------------------------

def test_disciplines_returns_list(client):
    r = client.get("/api/v1/disciplines")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    codes = [d["code"] for d in data]
    assert "lead" in codes
    assert "boulder" in codes
    assert "speed" in codes
    assert "boulder_lead" in codes


def test_disciplines_schema(client):
    r = client.get("/api/v1/disciplines")
    assert r.status_code == 200
    item = r.json()[0]
    assert "code" in item
    assert "name" in item
    assert "description" in item


# ---------------------------------------------------------------------------
# /api/v1/leaderboard
# ---------------------------------------------------------------------------

def test_leaderboard_default(client):
    r = client.get("/api/v1/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert body["discipline"] == "lead"
    assert body["gender"] == "M"
    assert "items" in body
    assert "total" in body
    assert "limit" in body
    assert "offset" in body


def test_leaderboard_has_adam(client):
    r = client.get("/api/v1/leaderboard?gender=M&discipline=lead")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    first = body["items"][0]
    assert first["rank"] == 1
    assert first["name"] == "Adam Ondra"
    assert first["mu"] == 1750.0
    assert first["provisional"] is False


def test_leaderboard_female(client):
    r = client.get("/api/v1/leaderboard?gender=F")
    assert r.status_code == 200
    body = r.json()
    assert body["gender"] == "F"
    assert body["total"] >= 1
    assert body["items"][0]["name"] == "Janja Garnbret"


def test_leaderboard_pagination(client):
    r = client.get("/api/v1/leaderboard?gender=M&limit=1&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 1
    assert body["offset"] == 0
    assert body["limit"] == 1


def test_leaderboard_invalid_discipline(client):
    r = client.get("/api/v1/leaderboard?discipline=nonexistent")
    assert r.status_code == 422


def test_leaderboard_invalid_gender(client):
    r = client.get("/api/v1/leaderboard?gender=X")
    assert r.status_code == 422


def test_leaderboard_limit_too_large(client):
    r = client.get("/api/v1/leaderboard?limit=999")
    assert r.status_code == 422


def test_leaderboard_discipline_aliases(client):
    for alias in ("lead", "LEAD", "boulder", "speed", "combined", "boulder_lead"):
        r = client.get(f"/api/v1/leaderboard?discipline={alias}")
        assert r.status_code == 200, f"alias '{alias}' returned {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# /api/v1/athletes/{athlete_id}
# ---------------------------------------------------------------------------

def test_athlete_detail_adam(client):
    lb = client.get("/api/v1/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]

    r = client.get(f"/api/v1/athletes/{adam_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == adam_id
    assert body["name"] == "Adam Ondra"
    assert body["nationality"] == "CZE"
    assert body["gender"] == "M"
    assert body["year_of_birth"] == 1993
    assert isinstance(body["ratings"], list)
    assert len(body["ratings"]) >= 1
    rating = body["ratings"][0]
    assert rating["discipline"] == "lead"
    assert rating["mu"] == 1750.0
    assert rating["provisional"] is False
    assert isinstance(body["recent_events"], list)


def test_athlete_detail_not_found(client):
    r = client.get("/api/v1/athletes/999999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /api/v1/athletes/{athlete_id}/history
# ---------------------------------------------------------------------------

def test_athlete_history_found(client):
    lb = client.get("/api/v1/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]

    r = client.get(f"/api/v1/athletes/{adam_id}/history?discipline=lead")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == adam_id
    assert body["athlete_name"] == "Adam Ondra"
    assert body["discipline"] == "lead"
    assert isinstance(body["points"], list)
    assert len(body["points"]) >= 1
    pt = body["points"][0]
    assert "event_id" in pt
    assert "mu_after" in pt
    assert "mu_before" in pt
    assert "delta" in pt
    assert "event_date" in pt
    assert "season" in pt
    assert pt["mu_after"] == 1750.0


def test_athlete_history_not_found(client):
    r = client.get("/api/v1/athletes/999999/history?discipline=lead")
    assert r.status_code == 404


def test_athlete_history_invalid_discipline(client):
    r = client.get("/api/v1/athletes/1/history?discipline=xyz")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/events
# ---------------------------------------------------------------------------

def test_events_list(client):
    r = client.get("/api/v1/events")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "total" in body
    assert body["total"] >= 1
    item = body["items"][0]
    assert "id" in item
    assert "name" in item
    assert "discipline" in item
    assert "season" in item
    assert "start_date" in item
    assert "tier" in item


def test_events_filter_discipline(client):
    r = client.get("/api/v1/events?discipline=lead")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    for item in body["items"]:
        assert item["discipline"] == "lead"


def test_events_filter_season(client):
    r = client.get("/api/v1/events?season=2024")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    for item in body["items"]:
        assert item["season"] == 2024


def test_events_innsbruck_present(client):
    r = client.get("/api/v1/events?discipline=lead&season=2024")
    assert r.status_code == 200
    names = [e["name"] for e in r.json()["items"]]
    assert "Innsbruck World Cup" in names


def test_events_invalid_discipline(client):
    r = client.get("/api/v1/events?discipline=badval")
    assert r.status_code == 422


def test_events_invalid_season(client):
    r = client.get("/api/v1/events?season=1900")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/events/{event_id}
# ---------------------------------------------------------------------------

def test_event_detail(client):
    events = client.get("/api/v1/events?discipline=lead").json()["items"]
    assert len(events) >= 1
    event_id = events[0]["id"]

    r = client.get(f"/api/v1/events/{event_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == event_id
    assert body["name"] == "Innsbruck World Cup"
    assert body["discipline"] == "lead"
    assert body["season"] == 2024
    assert body["tier"] == "world_cup"
    assert "rounds" in body
    assert len(body["rounds"]) >= 1
    rnd = body["rounds"][0]
    assert "round_type" in rnd
    assert "gender" in rnd
    assert "results" in rnd
    assert len(rnd["results"]) >= 1
    res = rnd["results"][0]
    assert "athlete_id" in res
    assert "athlete_name" in res
    assert "mu_before" in res
    assert "mu_after" in res
    assert "delta" in res
    assert res["rank"] == 1


def test_event_detail_not_found(client):
    r = client.get("/api/v1/events/999999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

def test_cors_allows_all_origins(client):
    r = client.get("/api/v1/disciplines", headers={"Origin": "https://example.com"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# OpenAPI docs
# ---------------------------------------------------------------------------

def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "Climbing ELO"
    paths = schema["paths"]
    assert "/api/v1/leaderboard" in paths
    assert "/api/v1/disciplines" in paths
    assert "/api/v1/events" in paths
    assert "/api/v1/athletes/{athlete_id}" in paths
    assert "/api/v1/athletes/{athlete_id}/history" in paths


def test_docs_endpoint(client):
    r = client.get("/docs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
