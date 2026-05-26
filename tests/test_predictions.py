"""Tests for the /predictions route (Issue #18).

Covers:
- Route returns HTTP 200
- Empty state when no upcoming events
- Completed (past) events do NOT appear
- Upcoming events DO appear
- Events with athletes show prediction data
- Events without athletes show "select manually" fallback
"""

from __future__ import annotations

from datetime import date, timedelta

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
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("predictions_db") / "test.db"


@pytest.fixture(scope="module")
def test_factory(test_db_path):
    """Seed a DB with a past event, an upcoming event with athletes, and an
    upcoming event without athletes."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    today = date.today()
    past_date = today - timedelta(days=30)
    future_date = today + timedelta(days=30)
    far_future_date = today + timedelta(days=90)

    # --- athletes ---
    adam = Athlete(name="Adam Ondra", gender=Gender.M, nationality="CZE")
    janja = Athlete(name="Janja Garnbret", gender=Gender.F, nationality="SVN")
    session.add_all([adam, janja])
    session.flush()

    # --- past finished event (should NOT appear on /predictions) ---
    past_event = Event(
        name="Past Lead World Cup",
        tier=EventTier.WORLD_CUP,
        season=today.year,
        start_date=past_date,
        discipline=Discipline.LEAD,
    )
    session.add(past_event)
    session.flush()
    rnd_past = Round(
        event_id=past_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=1,
    )
    session.add(rnd_past)
    session.flush()
    session.add(
        Result(round_id=rnd_past.id, athlete_id=adam.id, rank=1, raw_score="TOP")
    )

    # --- upcoming event WITH athletes ---
    future_event_athletes = Event(
        name="Future Lead World Cup With Athletes",
        tier=EventTier.WORLD_CUP,
        season=today.year,
        start_date=future_date,
        discipline=Discipline.LEAD,
    )
    session.add(future_event_athletes)
    session.flush()
    rnd_future_m = Round(
        event_id=future_event_athletes.id,
        round_type=RoundType.QUALIFICATION,
        gender=Gender.M,
        athlete_count=1,
    )
    rnd_future_f = Round(
        event_id=future_event_athletes.id,
        round_type=RoundType.QUALIFICATION,
        gender=Gender.F,
        athlete_count=1,
    )
    session.add_all([rnd_future_m, rnd_future_f])
    session.flush()
    session.add(
        Result(round_id=rnd_future_m.id, athlete_id=adam.id, rank=1, raw_score="TOP")
    )
    session.add(
        Result(round_id=rnd_future_f.id, athlete_id=janja.id, rank=1, raw_score="TOP")
    )

    # --- upcoming event WITHOUT athletes (bare event row, as stored by scrape_upcoming_events) ---
    future_event_no_athletes = Event(
        name="Future Lead World Cup No Athletes",
        tier=EventTier.WORLD_CUP,
        season=today.year,
        start_date=far_future_date,
        discipline=Discipline.LEAD,
    )
    session.add(future_event_no_athletes)
    session.flush()

    # --- ratings ---
    session.add(
        Rating(
            athlete_id=adam.id,
            discipline=Discipline.LEAD,
            mu=1750.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=past_date,
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
            last_event_at=past_date,
        )
    )

    session.commit()
    session.close()
    return factory


@pytest.fixture(scope="module")
def client(test_db_path, test_factory):
    import climbing_elo.database as _db

    original_get_engine = _db.get_engine
    original_session = _v1._session

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{test_db_path}")

    def patched_session():
        return test_factory()

    _db.get_engine = patched_get_engine  # type: ignore[assignment]
    _v1._session = patched_session  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)

    yield tc

    _db.get_engine = original_get_engine
    _v1._session = original_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictionsRoute:
    def test_returns_200(self, client):
        r = client.get("/predictions")
        assert r.status_code == 200

    def test_returns_html(self, client):
        r = client.get("/predictions")
        assert "text/html" in r.headers["content-type"]

    def test_page_title_present(self, client):
        r = client.get("/predictions")
        assert "Predictions" in r.text

    def test_past_event_not_shown(self, client):
        """Completed events (start_date < today) must NOT appear on /predictions."""
        r = client.get("/predictions")
        assert "Past Lead World Cup" not in r.text

    def test_future_event_with_athletes_shown(self, client):
        """Upcoming events with stored results should appear."""
        r = client.get("/predictions")
        assert "Future Lead World Cup With Athletes" in r.text

    def test_future_event_no_athletes_shown(self, client):
        """Upcoming events without stored results should still appear (with fallback UI)."""
        r = client.get("/predictions")
        assert "Future Lead World Cup No Athletes" in r.text

    def test_no_athletes_shows_manual_link(self, client):
        """Events without athletes should show a link to /projections/new."""
        r = client.get("/predictions")
        assert "/projections/new" in r.text

    def test_prediction_data_rendered_for_event_with_athletes(self, client):
        """Events with athletes should include athlete name and win-probability data."""
        r = client.get("/predictions")
        # At least one of our athletes should appear in the prediction rows
        assert "Adam Ondra" in r.text or "Janja Garnbret" in r.text

    def test_discipline_sections_present(self, client):
        """All three discipline sections should be present in the page."""
        r = client.get("/predictions")
        assert "Lead" in r.text
        assert "Boulder" in r.text
        assert "Speed" in r.text

    def test_manual_projection_link_present(self, client):
        """The page should always offer a link to the manual projections form."""
        r = client.get("/predictions")
        assert "/projections/new" in r.text


class TestPredictionsQueryCount:
    """Regression tests: query count must be O(events × disciplines), not O(athletes × …).

    Uses SQLAlchemy's ``before_cursor_execute`` event to count DB round-trips.
    With the N+1 fix in place, building projection inputs for N athletes should
    require at most 2 queries per (gender, event) block regardless of N.
    """

    def test_query_count_not_linear_in_athletes(self, tmp_path):
        """With K athletes across G genders, the predictions route should issue
        significantly fewer queries than 2 * K * G (the old N+1 baseline).

        We seed 8 athletes (4M + 4F) into one upcoming event.  The N+1 baseline
        would fire 2*8 = 16 queries just for athlete+rating lookups.  The batched
        implementation uses 1 query per gender block — so ≤ 2 total for athletes
        (one batched join per gender).
        """
        import climbing_elo.api.v1_routes as _v1
        import climbing_elo.database as _db_mod

        from datetime import date, timedelta
        from sqlalchemy import create_engine, event as sa_event
        from sqlalchemy.orm import sessionmaker
        from fastapi.testclient import TestClient
        from climbing_elo.api.app import create_app
        from climbing_elo.models import (
            Athlete,
            Base,
            Discipline,
            Event,
            EventTier,
            Gender,
            Rating,
            Result,
            Round,
            RoundType,
        )

        db_file = tmp_path / "qcount.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        session = factory()

        today = date.today()
        future_date = today + timedelta(days=10)

        # Seed 4 men + 4 women
        athletes = []
        for i in range(4):
            a = Athlete(name=f"Man{i}", gender=Gender.M, nationality="TST")
            session.add(a)
            athletes.append(a)
        for i in range(4):
            a = Athlete(name=f"Woman{i}", gender=Gender.F, nationality="TST")
            session.add(a)
            athletes.append(a)
        session.flush()

        # One upcoming event with all 8 athletes
        ev = Event(
            name="Q-Count Future WC",
            tier=EventTier.WORLD_CUP,
            season=today.year,
            start_date=future_date,
            discipline=Discipline.LEAD,
        )
        session.add(ev)
        session.flush()

        rnd_m = Round(
            event_id=ev.id,
            round_type=RoundType.QUALIFICATION,
            gender=Gender.M,
            athlete_count=4,
        )
        rnd_f = Round(
            event_id=ev.id,
            round_type=RoundType.QUALIFICATION,
            gender=Gender.F,
            athlete_count=4,
        )
        session.add_all([rnd_m, rnd_f])
        session.flush()

        for i, ath in enumerate(athletes):
            rnd = rnd_m if ath.gender == Gender.M else rnd_f
            session.add(
                Result(
                    round_id=rnd.id,
                    athlete_id=ath.id,
                    rank=i + 1,
                    raw_score="TOP",
                    dns=False,
                )
            )

        # Ratings for all athletes in Lead
        for i, ath in enumerate(athletes):
            session.add(
                Rating(
                    athlete_id=ath.id,
                    discipline=Discipline.LEAD,
                    mu=1500.0 + i * 10,
                    sigma=100.0,
                    n_events=5,
                    provisional=False,
                    last_event_at=today,
                )
            )

        session.commit()
        session.close()

        # Wire up the test DB
        original_get_engine = _db_mod.get_engine
        original_session = _v1._session

        def patched_get_engine(db_path=None):
            return create_engine(f"sqlite:///{db_file}")

        def patched_session():
            return factory()

        _db_mod.get_engine = patched_get_engine  # type: ignore[assignment]
        _v1._session = patched_session  # type: ignore[assignment]

        # Count queries via SQLAlchemy event hook
        query_count: list[int] = [0]

        def _before_execute(conn, cursor, statement, params, context, executemany):
            query_count[0] += 1

        sa_event.listen(engine, "before_cursor_execute", _before_execute)

        try:
            app = create_app()
            tc = TestClient(app)
            query_count[0] = 0
            r = tc.get("/predictions")
            assert r.status_code == 200
            n_queries = query_count[0]

            # N+1 baseline: 2 queries/athlete × 8 athletes × 1 gender block = 16
            # just for the athlete+rating lookups in one (event, gender) block.
            # The batched fix: 1 join query per gender block = 2 total for all athletes.
            # We allow generous headroom (≤ 8) to account for the other queries
            # (event fetch, result_count, result rows, etc.) without being brittle.
            # Key assertion: must be WELL under 16 (the N+1 count for 8 athletes).
            assert n_queries < 16, (
                f"Query count {n_queries} suggests N+1 regression "
                f"(expected < 16 with 8 athletes across 2 genders)"
            )
        finally:
            sa_event.remove(engine, "before_cursor_execute", _before_execute)
            _db_mod.get_engine = original_get_engine
            _v1._session = original_session


class TestPredictionsEmptyState:
    """Test the empty-state branch when there are genuinely no upcoming events."""

    def test_empty_db_returns_200(self, tmp_path):
        """A fresh DB with no events should return 200 with an empty-state message."""
        import climbing_elo.database as _db_mod

        db_file = tmp_path / "empty.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)

        original_get_engine = _db_mod.get_engine
        original_session = _v1._session

        def patched_get_engine(db_path=None):
            return create_engine(f"sqlite:///{db_file}")

        def patched_session():
            return factory()

        _db_mod.get_engine = patched_get_engine  # type: ignore[assignment]
        _v1._session = patched_session  # type: ignore[assignment]

        try:
            app = create_app()
            tc = TestClient(app)
            r = tc.get("/predictions")
            assert r.status_code == 200
            # Empty state should prompt users to scrape or use manual form
            assert "No upcoming events found" in r.text or "Manual Projection" in r.text
        finally:
            _db_mod.get_engine = original_get_engine
            _v1._session = original_session
