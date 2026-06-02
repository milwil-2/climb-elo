"""Tests for the public v1 REST API endpoints.

Uses FastAPI's TestClient with a temporary SQLite file containing seeded test
data. The module-level _session() in v1_routes is monkey-patched to use the
test sessionmaker so no production DB is touched.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import climbing_elo.api.v1_routes as _v1
import climbing_elo.database as _db
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
    """Create and seed a persistent (file-based) SQLite DB for the module.

    Note (#91): ``last_event_at`` for the seeded ratings is anchored to
    ``date.today()`` minus a few weeks so the default ``view=active`` filter
    surfaces both Adam and Janja. Earlier this used a hard-coded 2024 date,
    which left the leaderboard empty under the post-#91 default.
    """
    recent = date.today() - timedelta(days=30)

    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    # -- Athletes
    adam = Athlete(
        name="Adam Ondra", gender=Gender.M, nationality="CZE", year_of_birth=1993
    )
    janja = Athlete(
        name="Janja Garnbret", gender=Gender.F, nationality="SVN", year_of_birth=1999
    )
    session.add_all([adam, janja])
    session.flush()

    # -- Event (kept on the original 2024-06-01 date so existing tests that
    # assert ``season == 2024`` keep working; only the rating ``last_event_at``
    # is anchored to recent to satisfy the post-#91 active-view filter).
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
    rnd_m = Round(
        event_id=event.id, round_type=RoundType.FINAL, gender=Gender.M, athlete_count=1
    )
    rnd_f = Round(
        event_id=event.id, round_type=RoundType.FINAL, gender=Gender.F, athlete_count=1
    )
    session.add_all([rnd_m, rnd_f])
    session.flush()

    # -- Results
    session.add(Result(round_id=rnd_m.id, athlete_id=adam.id, rank=1, raw_score="TOP"))
    session.add(Result(round_id=rnd_f.id, athlete_id=janja.id, rank=1, raw_score="TOP"))
    session.flush()

    # -- Ratings
    session.add(
        Rating(
            athlete_id=adam.id,
            discipline=Discipline.LEAD,
            mu=1750.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=recent,
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
            last_event_at=recent,
        )
    )
    session.flush()

    # -- Rating history
    session.add(
        RatingHistory(
            athlete_id=adam.id,
            event_id=event.id,
            round_id=rnd_m.id,
            mu_before=1740.0,
            mu_after=1750.0,
            sigma_before=125.0,
            sigma_after=120.0,
            contributing_pairs=[],
        )
    )
    session.add(
        RatingHistory(
            athlete_id=janja.id,
            event_id=event.id,
            round_id=rnd_f.id,
            mu_before=1790.0,
            mu_after=1800.0,
            sigma_before=105.0,
            sigma_after=100.0,
            contributing_pairs=[],
        )
    )
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
        assert r.status_code == 200, (
            f"alias '{alias}' returned {r.status_code}: {r.text}"
        )


# ---------------------------------------------------------------------------
# /api/v1/leaderboard — view parameter (Issue #91)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def view_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("view_db") / "test.db"


@pytest.fixture(scope="module")
def view_factory(view_db_path):
    """Seed four athletes covering every #91 view branch (mirrors
    ``leaderboard_factory`` in ``test_routes.py`` — kept separate so this test
    module stays self-contained).

    NOTE: ``view_client`` below is **function-scoped** (not module-scoped) so
    its monkey-patches of ``_v1._session`` and ``_db.get_engine`` are restored
    between tests. Otherwise the still-alive module-scoped patches would
    clobber the ``client`` fixture's patches for the rest of the module,
    breaking tests like ``test_athlete_detail_adam`` and the events suite.
    """
    engine = create_engine(f"sqlite:///{view_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    today = date.today()
    athletes = [
        # (name, retired_at, last_event_at, mu)
        ("Recent Racer", None, today - timedelta(days=30), 2100.0),
        ("Sabbatical Sam", None, today - timedelta(days=540), 2050.0),  # ~18 mo
        ("Ancient Albert", None, today - timedelta(days=int(5 * 365.25)), 2000.0),
        (
            "Flagged Frank",
            today - timedelta(days=10),
            today - timedelta(days=15),
            1950.0,
        ),
    ]
    name_to_id: dict[str, int] = {}
    for name, retired, _, _ in athletes:
        a = Athlete(
            name=name,
            gender=Gender.M,
            nationality="USA",
            retired_at=retired,
        )
        session.add(a)
        session.flush()
        name_to_id[name] = a.id

    for name, _, last_event, mu in athletes:
        session.add(
            Rating(
                athlete_id=name_to_id[name],
                discipline=Discipline.LEAD,
                mu=mu,
                sigma=110.0,
                n_events=10,
                provisional=False,
                last_event_at=last_event,
            )
        )
    session.commit()
    session.close()
    return factory


@pytest.fixture(scope="function")
def view_client(view_db_path, view_factory):
    original_session = _v1._session
    original_get_engine = _db.get_engine

    def patched_session():
        return view_factory()

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{view_db_path}")

    _v1._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)
    yield tc

    _db.get_engine = original_get_engine
    _v1._session = original_session


def _names(body: dict) -> list[str]:
    return [item["name"] for item in body["items"]]


def test_leaderboard_view_default_is_active(view_client):
    """Default ``view=active`` returns only the recently-active athlete."""
    r = view_client.get("/api/v1/leaderboard?discipline=lead&gender=M")
    assert r.status_code == 200
    body = r.json()
    names = _names(body)
    assert "Recent Racer" in names
    # Flagged Frank competed recently (15 days ago) → still surfaces in
    # ``active``; the retired_at flag only applies to the ``all`` view.
    assert "Flagged Frank" in names
    assert "Sabbatical Sam" not in names
    assert "Ancient Albert" not in names


def test_leaderboard_view_all_includes_on_break(view_client):
    """``view=all`` adds Sabbatical Sam back but still hides Albert + Frank."""
    r = view_client.get("/api/v1/leaderboard?discipline=lead&gender=M&view=all")
    assert r.status_code == 200
    names = _names(r.json())
    assert "Recent Racer" in names
    assert "Sabbatical Sam" in names
    assert "Ancient Albert" not in names  # >5y gap → heuristic filters
    assert "Flagged Frank" not in names  # retired_at set → manual filter


def test_leaderboard_view_legacy_returns_everyone(view_client):
    """``view=legacy`` is the pre-#91 unfiltered behaviour."""
    r = view_client.get("/api/v1/leaderboard?discipline=lead&gender=M&view=legacy")
    assert r.status_code == 200
    names = _names(r.json())
    assert set(names) == {
        "Recent Racer",
        "Sabbatical Sam",
        "Ancient Albert",
        "Flagged Frank",
    }


def test_leaderboard_view_counts_are_monotone(view_client):
    """active ⊆ all ⊆ legacy (in terms of total athletes returned)."""
    base = "/api/v1/leaderboard?discipline=lead&gender=M&limit=100"
    active = view_client.get(f"{base}&view=active").json()["total"]
    all_view = view_client.get(f"{base}&view=all").json()["total"]
    legacy = view_client.get(f"{base}&view=legacy").json()["total"]
    assert active <= all_view <= legacy
    assert legacy == 4  # all seeded athletes
    # In this fixture they happen to be: active=2, all=2, legacy=4
    assert active == 2
    assert all_view == 2


def test_leaderboard_view_invalid_returns_422(view_client):
    """An unknown ``view`` value yields a 422 with a useful error."""
    r = view_client.get("/api/v1/leaderboard?discipline=lead&gender=M&view=junk")
    assert r.status_code == 422
    # FastAPI's Literal validation includes the bad value in the detail.
    detail = r.json().get("detail")
    assert detail is not None


def test_leaderboard_view_active_explicit_matches_default(view_client):
    """Explicit ``view=active`` returns the same items as the default."""
    r_default = view_client.get("/api/v1/leaderboard?discipline=lead&gender=M")
    r_active = view_client.get(
        "/api/v1/leaderboard?discipline=lead&gender=M&view=active"
    )
    assert r_default.status_code == 200
    assert r_active.status_code == 200
    assert _names(r_default.json()) == _names(r_active.json())


def test_leaderboard_sigma_inflates_for_stale_athlete(view_client):
    """#151 PR A: the wired-through ``sigma_now`` call surfaces an inflated σ
    for Sabbatical Sam (~540 days inactive) while leaving Recent Racer
    (within grace) at the stored 110.0.

    This is the end-to-end assertion that the helper is plumbed correctly
    through the v1 leaderboard response — not just unit-tested in isolation.
    """
    r = view_client.get("/api/v1/leaderboard?discipline=lead&gender=M&view=all")
    assert r.status_code == 200
    by_name = {item["name"]: item for item in r.json()["items"]}

    # Recent Racer: 30d inactive == grace boundary → σ unchanged from stored 110.0
    assert by_name["Recent Racer"]["sigma"] == pytest.approx(110.0, abs=0.01)

    # Sabbatical Sam: 540d inactive → σ should be strictly inflated.
    sam_sigma = by_name["Sabbatical Sam"]["sigma"]
    assert sam_sigma > 110.0, (
        f"expected Sabbatical Sam's σ to inflate beyond stored 110.0, got {sam_sigma}"
    )
    # Loose upper bound — the formula gives ~150 here, but tolerate day-skew.
    assert sam_sigma < 200.0


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
# /api/v1/athletes  — name search / typeahead (Step 1)
# ---------------------------------------------------------------------------


def test_athlete_search_match(client):
    """A substring match (case-insensitive) returns the athlete with mu."""
    r = client.get("/api/v1/athletes?q=ondra")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)
    names = [a["name"] for a in body]
    assert "Adam Ondra" in names
    adam = next(a for a in body if a["name"] == "Adam Ondra")
    assert adam["nationality"] == "CZE"
    assert adam["gender"] == "M"
    # No discipline → highest rating across disciplines (Adam only has Lead).
    assert adam["mu"] == 1750.0


def test_athlete_search_no_match(client):
    """A query that matches nobody returns an empty list (still 200)."""
    r = client.get("/api/v1/athletes?q=zzzznobody")
    assert r.status_code == 200
    assert r.json() == []


def test_athlete_search_gender_filter(client):
    """The gender filter restricts the result set."""
    # 'a' matches both Adam and Janja; filtering to F drops Adam.
    r = client.get("/api/v1/athletes?q=a&gender=F")
    assert r.status_code == 200
    body = r.json()
    names = [a["name"] for a in body]
    assert "Janja Garnbret" in names
    assert "Adam Ondra" not in names
    assert all(a["gender"] == "F" for a in body)


def test_athlete_search_discipline_mu(client):
    """With a discipline, mu reflects that discipline's rating."""
    r = client.get("/api/v1/athletes?q=garnbret&discipline=lead")
    assert r.status_code == 200
    body = r.json()
    janja = next(a for a in body if a["name"] == "Janja Garnbret")
    assert janja["mu"] == 1800.0


def test_athlete_search_limit_cap(client):
    """limit must be capped at 50 (422 above)."""
    r_ok = client.get("/api/v1/athletes?q=a&limit=50")
    assert r_ok.status_code == 200
    r_too_big = client.get("/api/v1/athletes?q=a&limit=51")
    assert r_too_big.status_code == 422


def test_athlete_search_limit_applied(client):
    """A small limit truncates the result list."""
    r = client.get("/api/v1/athletes?q=a&limit=1")
    assert r.status_code == 200
    assert len(r.json()) <= 1


def test_athlete_search_missing_query_is_422(client):
    """q is required (min_length=1)."""
    r = client.get("/api/v1/athletes")
    assert r.status_code == 422


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


# ---------------------------------------------------------------------------
# Extended fixture — includes combined ratings, boulder ratings, upcoming event
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extended_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("db_ext") / "test_ext.db"


@pytest.fixture(scope="module")
def extended_factory(extended_db_path):
    """Seed a DB with combined ratings, boulder, and an upcoming future event."""
    engine = create_engine(f"sqlite:///{extended_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    today = date.today()

    # Athletes
    adam = Athlete(
        name="Adam Ondra", gender=Gender.M, nationality="CZE", year_of_birth=1993
    )
    janja = Athlete(
        name="Janja Garnbret", gender=Gender.F, nationality="SVN", year_of_birth=1999
    )
    session.add_all([adam, janja])
    session.flush()

    # Past lead event
    lead_event = Event(
        name="Innsbruck Lead WC",
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=today.year,
        start_date=date(today.year, 1, 15),
        discipline=Discipline.LEAD,
    )
    session.add(lead_event)
    session.flush()

    rnd_lead_m = Round(
        event_id=lead_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=2,
    )
    rnd_lead_f = Round(
        event_id=lead_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.F,
        athlete_count=1,
    )
    session.add_all([rnd_lead_m, rnd_lead_f])
    session.flush()

    session.add(
        Result(
            round_id=rnd_lead_m.id,
            athlete_id=adam.id,
            rank=1,
            raw_score="TOP",
            dns=False,
        )
    )
    session.add(
        Result(
            round_id=rnd_lead_f.id,
            athlete_id=janja.id,
            rank=1,
            raw_score="TOP",
            dns=False,
        )
    )
    session.flush()

    # Past boulder event
    boulder_event = Event(
        name="Innsbruck Boulder WC",
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=today.year,
        start_date=date(today.year, 2, 10),
        discipline=Discipline.BOULDER,
    )
    session.add(boulder_event)
    session.flush()

    rnd_boul_m = Round(
        event_id=boulder_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=1,
    )
    session.add(rnd_boul_m)
    session.flush()

    session.add(
        Result(
            round_id=rnd_boul_m.id,
            athlete_id=adam.id,
            rank=1,
            raw_score="4T",
            dns=False,
        )
    )
    session.flush()

    # Upcoming lead event (in the future)
    upcoming_lead = Event(
        name="Future Lead WC",
        tier=EventTier.WORLD_CUP,
        country="FRA",
        season=today.year,
        start_date=today + timedelta(days=30),
        discipline=Discipline.LEAD,
    )
    session.add(upcoming_lead)
    session.flush()

    # Ratings
    # Lead ratings
    session.add(
        Rating(
            athlete_id=adam.id,
            discipline=Discipline.LEAD,
            mu=1750.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=date(today.year, 1, 15),
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
            last_event_at=date(today.year, 1, 15),
        )
    )
    # Boulder ratings
    session.add(
        Rating(
            athlete_id=adam.id,
            discipline=Discipline.BOULDER,
            mu=1700.0,
            sigma=130.0,
            n_events=8,
            provisional=False,
            last_event_at=date(today.year, 2, 10),
        )
    )
    session.add(
        Rating(
            athlete_id=janja.id,
            discipline=Discipline.BOULDER,
            mu=1720.0,
            sigma=110.0,
            n_events=9,
            provisional=False,
            last_event_at=date(today.year, 2, 10),
        )
    )
    # Combined (BOULDER_LEAD) ratings — geometric mean
    import math

    adam_combined_mu = round(math.sqrt(1750.0 * 1700.0), 2)
    adam_combined_sigma = round(math.sqrt((120.0**2 + 130.0**2) / 2), 2)
    janja_combined_mu = round(math.sqrt(1800.0 * 1720.0), 2)
    janja_combined_sigma = round(math.sqrt((100.0**2 + 110.0**2) / 2), 2)

    session.add(
        Rating(
            athlete_id=adam.id,
            discipline=Discipline.BOULDER_LEAD,
            mu=adam_combined_mu,
            sigma=adam_combined_sigma,
            n_events=8,
            provisional=False,
            last_event_at=date(today.year, 2, 10),
        )
    )
    session.add(
        Rating(
            athlete_id=janja.id,
            discipline=Discipline.BOULDER_LEAD,
            mu=janja_combined_mu,
            sigma=janja_combined_sigma,
            n_events=9,
            provisional=False,
            last_event_at=date(today.year, 2, 10),
        )
    )
    session.flush()

    # Rating history for lead event
    session.add(
        RatingHistory(
            athlete_id=adam.id,
            event_id=lead_event.id,
            round_id=rnd_lead_m.id,
            mu_before=1740.0,
            mu_after=1750.0,
            sigma_before=125.0,
            sigma_after=120.0,
            contributing_pairs=[],
        )
    )
    session.add(
        RatingHistory(
            athlete_id=janja.id,
            event_id=lead_event.id,
            round_id=rnd_lead_f.id,
            mu_before=1790.0,
            mu_after=1800.0,
            sigma_before=105.0,
            sigma_after=100.0,
            contributing_pairs=[],
        )
    )

    session.commit()
    session.close()
    return factory


@pytest.fixture(scope="module")
def ext_client(extended_db_path, extended_factory):
    """TestClient backed by the extended DB with combined ratings + upcoming event."""
    original_session = _v1._session
    original_get_engine = _db.get_engine

    def patched_session():
        return extended_factory()

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{extended_db_path}")

    _v1._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)

    yield tc

    _db.get_engine = original_get_engine
    _v1._session = original_session


# ---------------------------------------------------------------------------
# /api/v1/combined/leaderboard
# ---------------------------------------------------------------------------


def test_combined_leaderboard_200(ext_client):
    r = ext_client.get("/api/v1/combined/leaderboard?gender=M&limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "total" in body
    assert "limit" in body
    assert "offset" in body
    assert body["gender"] == "M"
    assert body["total"] >= 1


def test_combined_leaderboard_shape(ext_client):
    r = ext_client.get("/api/v1/combined/leaderboard?gender=M")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) >= 1
    entry = body["items"][0]
    for field in (
        "rank",
        "athlete_id",
        "name",
        "mu",
        "sigma",
        "mu_boulder",
        "mu_lead",
        "sigma_boulder",
        "sigma_lead",
        "n_events",
        "provisional",
    ):
        assert field in entry, f"Missing field: {field}"
    # mu_boulder and mu_lead should be positive floats
    assert entry["mu_boulder"] > 0
    assert entry["mu_lead"] > 0


def test_combined_leaderboard_female(ext_client):
    r = ext_client.get("/api/v1/combined/leaderboard?gender=F")
    assert r.status_code == 200
    body = r.json()
    assert body["gender"] == "F"
    assert body["total"] >= 1


def test_combined_leaderboard_invalid_gender(ext_client):
    r = ext_client.get("/api/v1/combined/leaderboard?gender=X")
    assert r.status_code == 422


def test_combined_leaderboard_limit_too_large(ext_client):
    r = ext_client.get("/api/v1/combined/leaderboard?limit=999")
    assert r.status_code == 422


def test_combined_leaderboard_pagination(ext_client):
    r = ext_client.get("/api/v1/combined/leaderboard?gender=M&limit=1&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert body["items"][0]["rank"] == 1


# ---------------------------------------------------------------------------
# /api/v1/athletes/{id}/combined
# ---------------------------------------------------------------------------


def test_athlete_combined_200(ext_client):
    # Get adam's ID via the combined leaderboard
    lb = ext_client.get("/api/v1/combined/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]

    r = ext_client.get(f"/api/v1/athletes/{adam_id}/combined")
    assert r.status_code == 200
    body = r.json()
    assert body["athlete_id"] == adam_id
    assert body["name"] == "Adam Ondra"
    assert body["nationality"] == "CZE"
    assert body["gender"] == "M"
    # Combined fields
    assert body["mu_combined"] > 0
    assert body["sigma_combined"] > 0
    assert body["mu_boulder"] > 0
    assert body["mu_lead"] > 0
    assert body["sigma_boulder"] > 0
    assert body["sigma_lead"] > 0
    assert isinstance(body["provisional_combined"], bool)
    assert isinstance(body["n_events_combined"], int)


def test_athlete_combined_404_no_combined_rating(ext_client):
    """An athlete with no BOULDER_LEAD rating should return 404."""
    # athlete_id 999999 does not exist
    r = ext_client.get("/api/v1/athletes/999999/combined")
    assert r.status_code == 404


def test_athlete_not_found_combined(ext_client):
    r = ext_client.get("/api/v1/athletes/999999/combined")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /api/v1/projections
# ---------------------------------------------------------------------------


def test_projections_valid(ext_client):
    """Valid request with 2+ athletes returns 200 with proper shape."""
    lb = ext_client.get("/api/v1/combined/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]

    # Get Janja's ID from female lead leaderboard
    lb_f = ext_client.get("/api/v1/leaderboard?gender=F&discipline=lead").json()
    janja_id = lb_f["items"][0]["athlete_id"]

    r = ext_client.post(
        "/api/v1/projections",
        json={"discipline": "lead", "athlete_ids": [adam_id, janja_id]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["discipline"] == "lead"
    assert body["n_athletes"] == 2
    assert body["n_simulations"] == 10_000
    assert len(body["items"]) == 2
    entry = body["items"][0]
    for field in (
        "athlete_id",
        "name",
        "mu",
        "sigma",
        "win",
        "podium",
        "top_8",
        "expected_rank",
    ):
        assert field in entry, f"Missing field: {field}"
    # Probabilities should sum to ~1
    total_win = sum(e["win"] for e in body["items"])
    assert abs(total_win - 1.0) < 0.05


def test_projections_too_few_athletes(ext_client):
    """Fewer than 2 athletes → 422."""
    lb = ext_client.get("/api/v1/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]
    r = ext_client.post(
        "/api/v1/projections",
        json={"discipline": "lead", "athlete_ids": [adam_id]},
    )
    assert r.status_code == 422


def test_projections_too_many_athletes(ext_client):
    """More than 64 athletes → 422 (Pydantic max_length)."""
    r = ext_client.post(
        "/api/v1/projections",
        json={"discipline": "lead", "athlete_ids": list(range(1, 66))},  # 65 IDs
    )
    assert r.status_code == 422


def test_projections_invalid_discipline(ext_client):
    """Invalid discipline → 422."""
    lb = ext_client.get("/api/v1/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]
    lb_f = ext_client.get("/api/v1/leaderboard?gender=F&discipline=lead").json()
    janja_id = lb_f["items"][0]["athlete_id"]
    r = ext_client.post(
        "/api/v1/projections",
        json={"discipline": "baddisc", "athlete_ids": [adam_id, janja_id]},
    )
    assert r.status_code == 422


def test_projections_non_integer_athlete_ids(ext_client):
    """Non-integer athlete IDs → 422."""
    r = ext_client.post(
        "/api/v1/projections",
        json={"discipline": "lead", "athlete_ids": ["not_an_int", "also_not"]},
    )
    assert r.status_code == 422


def test_projections_duplicate_athlete_ids(ext_client):
    """Duplicate athlete IDs → 422."""
    lb = ext_client.get("/api/v1/leaderboard?gender=M").json()
    adam_id = lb["items"][0]["athlete_id"]
    r = ext_client.post(
        "/api/v1/projections",
        json={"discipline": "lead", "athlete_ids": [adam_id, adam_id]},
    )
    assert r.status_code == 422


def test_projections_missing_body_fields(ext_client):
    """Missing required fields → 422."""
    r = ext_client.post("/api/v1/projections", json={"discipline": "lead"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/predictions/upcoming
# ---------------------------------------------------------------------------


def test_predictions_upcoming_200(ext_client):
    """Endpoint returns 200 with valid shape."""
    r = ext_client.get("/api/v1/predictions/upcoming")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "total" in body
    assert isinstance(body["items"], list)


def test_predictions_upcoming_filter_discipline(ext_client):
    r = ext_client.get("/api/v1/predictions/upcoming?discipline=lead")
    assert r.status_code == 200
    body = r.json()
    assert body["discipline"] == "lead"
    for item in body["items"]:
        assert item["discipline"] == "lead"


def test_predictions_upcoming_filter_season(ext_client):
    today = date.today()
    r = ext_client.get(f"/api/v1/predictions/upcoming?season={today.year}")
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == today.year


def test_predictions_upcoming_shape(ext_client):
    """Each entry must have the expected fields."""
    today = date.today()
    r = ext_client.get(
        f"/api/v1/predictions/upcoming?discipline=lead&season={today.year}"
    )
    assert r.status_code == 200
    body = r.json()
    # The future lead event should appear
    assert body["total"] >= 1
    entry = body["items"][0]
    for field in (
        "event_id",
        "event_name",
        "discipline",
        "season",
        "start_date",
        "tier",
        "has_registered_athletes",
        "from_likely_roster",
        "genders",
    ):
        assert field in entry, f"Missing field: {field}"
    assert isinstance(entry["genders"], list)


def test_predictions_upcoming_empty_result(ext_client):
    """Past season with no upcoming events returns empty list gracefully."""
    r = ext_client.get("/api/v1/predictions/upcoming?season=2000")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_predictions_upcoming_invalid_discipline(ext_client):
    r = ext_client.get("/api/v1/predictions/upcoming?discipline=badval")
    assert r.status_code == 422


def test_predictions_upcoming_combined_rejected(ext_client):
    """boulder_lead / combined is not a valid discipline for upcoming predictions."""
    r = ext_client.get("/api/v1/predictions/upcoming?discipline=combined")
    assert r.status_code == 422


def test_predictions_upcoming_invalid_season_low(ext_client):
    r = ext_client.get("/api/v1/predictions/upcoming?season=1999")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI schema includes new endpoints
# ---------------------------------------------------------------------------


def test_openapi_includes_combined_endpoints(ext_client):
    r = ext_client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/combined/leaderboard" in paths
    assert "/api/v1/athletes/{athlete_id}/combined" in paths
    assert "/api/v1/projections" in paths
    assert "/api/v1/predictions/upcoming" in paths
