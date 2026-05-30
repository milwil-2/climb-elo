"""Tests for the persistent-forecast v1 endpoints.

Covers ``GET /api/v1/events/{event_id}/forecast`` and
``GET /api/v1/model-performance``. Both endpoints read frozen
:class:`EventForecast` / :class:`EventForecastScore` rows; the tests seed
those rows directly rather than running the snapshotting engine.

Each test seeds its own per-test SQLite file so the module-scoped client
fixture in :mod:`tests.test_api` doesn't interfere.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import climbing_elo.api.v1_routes as _v1
import climbing_elo.database as _db
from climbing_elo.api.app import create_app
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventForecast,
    EventForecastScore,
    EventTier,
    Gender,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "forecast_test.db"


@pytest.fixture
def factory(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def client(db_path: Path, factory) -> TestClient:
    """TestClient wired to a fresh SQLite DB.

    Patches ``v1_routes._session`` and ``database.get_engine`` so the route
    handlers see the test DB. Originals are restored on teardown.
    """
    original_session = _v1._session
    original_get_engine = _db.get_engine

    def patched_session():
        return factory()

    def patched_get_engine(_db_path=None):
        return create_engine(f"sqlite:///{db_path}")

    _v1._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)
    try:
        yield tc
    finally:
        _v1._session = original_session
        _db.get_engine = original_get_engine


@pytest.fixture
def seed(factory) -> Callable[..., dict]:
    """Helper that seeds an event + N athletes and returns their IDs.

    Each call uses a unique name suffix so successive seeds in the same test
    don't collide on the ``uq_athlete_name_gender`` constraint.
    """

    counter = {"n": 0}

    def _seed(
        *,
        event_name: str = "Test World Cup",
        season: int = 2026,
        start_date: date = date(2026, 6, 1),
        discipline: Discipline = Discipline.LEAD,
        tier: EventTier = EventTier.WORLD_CUP,
        athlete_names: list[str] | None = None,
        athlete_gender: Gender = Gender.M,
    ) -> dict:
        counter["n"] += 1
        suffix = counter["n"]
        athlete_names = athlete_names or [f"Athlete {suffix}-{i}" for i in range(8)]
        with factory() as session:  # type: Session
            event = Event(
                name=event_name,
                tier=tier,
                country="AUT",
                season=season,
                start_date=start_date,
                discipline=discipline,
            )
            session.add(event)
            session.flush()

            athletes: list[Athlete] = []
            for name in athlete_names:
                a = Athlete(name=name, gender=athlete_gender, nationality="USA")
                session.add(a)
                athletes.append(a)
            session.flush()

            athlete_ids = [a.id for a in athletes]
            event_id = event.id
            session.commit()

        return {"event_id": event_id, "athlete_ids": athlete_ids}

    return _seed


def _add_forecast_row(
    session: Session,
    *,
    event_id: int,
    athlete_id: int,
    gender: Gender = Gender.M,
    prob_win: float,
    expected_rank: float,
    is_backfill: bool = False,
    engine_version: str = "abcdef012345-deadbee",
) -> EventForecast:
    fc = EventForecast(
        event_id=event_id,
        gender=gender,
        athlete_id=athlete_id,
        prob_qualify=1.0,
        prob_reach_semi=0.9,
        prob_reach_final=0.6,
        prob_podium=max(prob_win, 0.1),
        prob_win=prob_win,
        expected_rank=expected_rank,
        mu_at_forecast=1700.0,
        sigma_at_forecast=120.0,
        n_simulations=10_000,
        roster_source="confirmed",
        is_backfill=is_backfill,
        engine_version=engine_version,
        generated_at=datetime.now(timezone.utc),
    )
    session.add(fc)
    return fc


def _add_score_row(
    session: Session,
    *,
    event_id: int,
    gender: Gender = Gender.M,
    is_backfill: bool = False,
    brier_podium: float = 0.15,
    brier_win: float = 0.05,
    logloss_podium: float = 0.35,
    logloss_win: float = 0.10,
    top3_intersection: int = 2,
    top8_intersection: int = 6,
    n_athletes: int = 8,
    engine_version: str = "abcdef012345-deadbee",
) -> EventForecastScore:
    sc = EventForecastScore(
        event_id=event_id,
        gender=gender,
        is_backfill=is_backfill,
        engine_version=engine_version,
        n_athletes=n_athletes,
        n_simulations=10_000,
        brier_semi=0.1,
        brier_final=0.12,
        brier_podium=brier_podium,
        brier_win=brier_win,
        logloss_semi=0.3,
        logloss_final=0.32,
        logloss_podium=logloss_podium,
        logloss_win=logloss_win,
        top3_intersection=top3_intersection,
        top8_intersection=top8_intersection,
        spearman_rank=0.75,
        computed_at=datetime.now(timezone.utc),
    )
    session.add(sc)
    return sc


# ---------------------------------------------------------------------------
# GET /api/v1/events/{event_id}/forecast
# ---------------------------------------------------------------------------


def test_event_forecast_404_when_event_missing(client: TestClient) -> None:
    r = client.get("/api/v1/events/999999/forecast?gender=M")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_event_forecast_404_when_no_forecast(
    client: TestClient, factory, seed: Callable[..., dict]
) -> None:
    info = seed()
    r = client.get(f"/api/v1/events/{info['event_id']}/forecast?gender=M")
    assert r.status_code == 404
    assert "no forecast" in r.json()["detail"].lower()


def test_event_forecast_200_with_rows(
    client: TestClient, factory, seed: Callable[..., dict]
) -> None:
    info = seed()
    ids = info["athlete_ids"]

    # Seed 3 forecast rows with distinct prob_win so the order assertion is
    # unambiguous, plus a score row.
    with factory() as session:
        _add_forecast_row(
            session,
            event_id=info["event_id"],
            athlete_id=ids[0],
            prob_win=0.10,
            expected_rank=5.0,
        )
        _add_forecast_row(
            session,
            event_id=info["event_id"],
            athlete_id=ids[1],
            prob_win=0.40,
            expected_rank=1.5,
        )
        _add_forecast_row(
            session,
            event_id=info["event_id"],
            athlete_id=ids[2],
            prob_win=0.25,
            expected_rank=3.0,
        )
        _add_score_row(session, event_id=info["event_id"])
        session.commit()

    r = client.get(f"/api/v1/events/{info['event_id']}/forecast?gender=M")
    assert r.status_code == 200, r.text
    body = r.json()

    assert set(body.keys()) == {"forecast", "score"}
    assert len(body["forecast"]) == 3

    # Sorted by prob_win DESC: 0.40, 0.25, 0.10
    probs = [row["prob_win"] for row in body["forecast"]]
    assert probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(0.40)
    assert body["forecast"][0]["athlete_id"] == ids[1]

    # Row shape — every documented field present.
    first = body["forecast"][0]
    for key in (
        "athlete_id",
        "name",
        "mu_at_forecast",
        "sigma_at_forecast",
        "prob_qualify",
        "prob_reach_semi",
        "prob_reach_final",
        "prob_podium",
        "prob_win",
        "expected_rank",
        "roster_source",
        "engine_version",
        "generated_at",
    ):
        assert key in first, key
    assert first["name"]  # joined Athlete.name

    # Score row shape
    score = body["score"]
    assert score is not None
    assert score["event_id"] == info["event_id"]
    assert score["gender"] == "M"
    assert score["is_backfill"] is False
    assert score["top3_intersection"] == 2
    assert score["n_athletes"] == 8


def test_event_forecast_200_without_score(
    client: TestClient, factory, seed: Callable[..., dict]
) -> None:
    info = seed()
    with factory() as session:
        _add_forecast_row(
            session,
            event_id=info["event_id"],
            athlete_id=info["athlete_ids"][0],
            prob_win=0.5,
            expected_rank=1.0,
        )
        session.commit()

    r = client.get(f"/api/v1/events/{info['event_id']}/forecast?gender=M")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["forecast"]) == 1
    assert body["score"] is None


# ---------------------------------------------------------------------------
# GET /api/v1/model-performance
# ---------------------------------------------------------------------------


def test_model_performance_default_returns_200(client: TestClient) -> None:
    r = client.get("/api/v1/model-performance")
    assert r.status_code == 200
    body = r.json()
    assert body["n_events_scored"] == 0
    assert body["aggregates"] == {}
    assert body["events"] == []
    # Default discipline label is "all" when nothing is supplied.
    assert body["discipline"] == "all"


def test_model_performance_filters_by_season(
    client: TestClient, factory, seed: Callable[..., dict]
) -> None:
    info_2025 = seed(event_name="WCS 2025", season=2025, start_date=date(2025, 6, 1))
    info_2026 = seed(event_name="WCS 2026", season=2026, start_date=date(2026, 6, 1))

    with factory() as session:
        _add_score_row(
            session,
            event_id=info_2025["event_id"],
            brier_podium=0.20,
            top3_intersection=1,
        )
        _add_score_row(
            session,
            event_id=info_2026["event_id"],
            brier_podium=0.10,
            top3_intersection=3,
        )
        session.commit()

    r = client.get("/api/v1/model-performance?season=2026")
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == 2026
    assert body["n_events_scored"] == 1
    assert len(body["events"]) == 1
    assert body["events"][0]["event_id"] == info_2026["event_id"]
    # Aggregates reflect only the 2026 row.
    assert body["aggregates"]["brier_podium_mean"] == pytest.approx(0.10)
    # top3_intersection_rate = 3 / (3 * 1) = 1.0
    assert body["aggregates"]["top3_intersection_rate"] == pytest.approx(1.0)


def test_model_performance_include_backfill_toggle(
    client: TestClient, factory, seed: Callable[..., dict]
) -> None:
    info_live = seed(event_name="Live event", season=2026, start_date=date(2026, 5, 1))
    info_backfill = seed(
        event_name="Backfill event", season=2026, start_date=date(2026, 4, 1)
    )

    with factory() as session:
        _add_score_row(
            session,
            event_id=info_live["event_id"],
            is_backfill=False,
            brier_podium=0.12,
            top3_intersection=2,
        )
        _add_score_row(
            session,
            event_id=info_backfill["event_id"],
            is_backfill=True,
            brier_podium=0.30,
            top3_intersection=0,
        )
        session.commit()

    # Default (include_backfill=False) sees only the live row.
    r = client.get("/api/v1/model-performance?season=2026")
    assert r.status_code == 200
    body = r.json()
    assert body["n_events_scored"] == 1
    assert body["events"][0]["event_id"] == info_live["event_id"]

    # With ?include_backfill=1, only the backfill row matches (the live row
    # has is_backfill=False so the filter excludes it).
    r = client.get("/api/v1/model-performance?season=2026&include_backfill=1")
    assert r.status_code == 200
    body = r.json()
    assert body["n_events_scored"] == 1
    assert body["events"][0]["event_id"] == info_backfill["event_id"]
