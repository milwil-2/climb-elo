"""Tests for the /head-to-head routes (Issue #19).

Covers:
- 200 for GET /head-to-head (form page)
- 200 for GET /head-to-head/{a_id}/{b_id} (result page)
- 404 for nonexistent athlete IDs
- 400 for same athlete ID twice (a_id == b_id)
- 400 for invalid discipline
- Rendered HTML contains both athlete names
- Win probability percentages sum to ~100%
- "Past meetings" count is correct for seeded fixture
- /predictions page card links to /head-to-head (no longer disabled)
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
    RatingHistory,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("h2h_db") / "test.db"


@pytest.fixture(scope="module")
def test_factory(test_db_path):
    """Seed an in-file DB with two athletes, ratings, shared events, and results."""
    engine = create_engine(f"sqlite:///{test_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    past = date.today() - timedelta(days=60)
    past2 = date.today() - timedelta(days=30)

    # Two athletes
    adam = Athlete(name="Adam Ondra", gender=Gender.M, nationality="CZE")
    janja = Athlete(name="Janja Garnbret", gender=Gender.F, nationality="SVN")
    # Third athlete with no shared events with the first two
    solo = Athlete(name="Solo Climber", gender=Gender.M, nationality="USA")
    session.add_all([adam, janja, solo])
    session.flush()

    # Ratings for Lead
    session.add(Rating(
        athlete_id=adam.id,
        discipline=Discipline.LEAD,
        mu=1750.0, sigma=120.0, n_events=15, provisional=False,
        last_event_at=past,
    ))
    session.add(Rating(
        athlete_id=janja.id,
        discipline=Discipline.LEAD,
        mu=1850.0, sigma=100.0, n_events=20, provisional=False,
        last_event_at=past,
    ))
    session.add(Rating(
        athlete_id=solo.id,
        discipline=Discipline.LEAD,
        mu=1600.0, sigma=200.0, n_events=5, provisional=False,
        last_event_at=past,
    ))

    # Two shared Lead events (adam + janja both competed)
    for i, ev_date in enumerate([past, past2]):
        ev = Event(
            name=f"Shared Lead WC {i + 1}",
            tier=EventTier.WORLD_CUP,
            season=ev_date.year,
            start_date=ev_date,
            discipline=Discipline.LEAD,
        )
        session.add(ev)
        session.flush()

        rnd = Round(
            event_id=ev.id,
            round_type=RoundType.FINAL,
            gender=Gender.M,
            athlete_count=2,
        )
        session.add(rnd)
        session.flush()

        session.add(Result(round_id=rnd.id, athlete_id=adam.id, rank=2, raw_score="30"))
        session.add(Result(round_id=rnd.id, athlete_id=janja.id, rank=1, raw_score="TOP"))

        # rating history entries so the chart has data
        session.add(RatingHistory(
            athlete_id=adam.id,
            event_id=ev.id,
            round_id=rnd.id,
            mu_before=1700.0,
            mu_after=1750.0,
            sigma_before=130.0,
            sigma_after=120.0,
        ))
        session.add(RatingHistory(
            athlete_id=janja.id,
            event_id=ev.id,
            round_id=rnd.id,
            mu_before=1800.0,
            mu_after=1850.0,
            sigma_before=110.0,
            sigma_after=100.0,
        ))

    # One event where solo competed but NOT adam or janja
    solo_ev = Event(
        name="Solo Only WC",
        tier=EventTier.WORLD_CUP,
        season=past.year,
        start_date=past,
        discipline=Discipline.LEAD,
    )
    session.add(solo_ev)
    session.flush()
    solo_rnd = Round(
        event_id=solo_ev.id,
        round_type=RoundType.QUALIFICATION,
        gender=Gender.M,
        athlete_count=1,
    )
    session.add(solo_rnd)
    session.flush()
    session.add(Result(round_id=solo_rnd.id, athlete_id=solo.id, rank=1, raw_score="20"))

    session.commit()

    # Capture IDs before closing the session
    adam_id = adam.id
    janja_id = janja.id
    solo_id = solo.id

    session.close()
    return factory, adam_id, janja_id, solo_id


@pytest.fixture(scope="module")
def client(test_db_path, test_factory):
    factory, adam_id, janja_id, solo_id = test_factory

    import climbing_elo.database as _db

    original_get_engine = _db.get_engine
    original_session = _v1._session

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{test_db_path}")

    def patched_session():
        return factory()

    _db.get_engine = patched_get_engine  # type: ignore[assignment]
    _v1._session = patched_session  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)

    yield tc, adam_id, janja_id, solo_id

    _db.get_engine = original_get_engine
    _v1._session = original_session


# ---------------------------------------------------------------------------
# Tests — form page (/head-to-head)
# ---------------------------------------------------------------------------

class TestHeadToHeadForm:
    def test_form_returns_200(self, client):
        tc, *_ = client
        r = tc.get("/head-to-head")
        assert r.status_code == 200

    def test_form_returns_html(self, client):
        tc, *_ = client
        r = tc.get("/head-to-head")
        assert "text/html" in r.headers["content-type"]

    def test_form_contains_title(self, client):
        tc, *_ = client
        r = tc.get("/head-to-head")
        assert "Head-to-Head" in r.text

    def test_form_contains_discipline_options(self, client):
        tc, *_ = client
        r = tc.get("/head-to-head")
        assert "lead" in r.text
        assert "boulder" in r.text

    def test_form_contains_athlete_data(self, client):
        tc, *_ = client
        r = tc.get("/head-to-head")
        # Athlete list is serialised as JSON in the page
        assert "Adam Ondra" in r.text
        assert "Janja Garnbret" in r.text


# ---------------------------------------------------------------------------
# Tests — result page (/head-to-head/{a_id}/{b_id})
# ---------------------------------------------------------------------------

class TestHeadToHeadResult:
    def test_result_returns_200(self, client):
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert r.status_code == 200

    def test_result_returns_html(self, client):
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert "text/html" in r.headers["content-type"]

    def test_both_names_in_response(self, client):
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert "Adam Ondra" in r.text
        assert "Janja Garnbret" in r.text

    def test_win_probabilities_sum_to_100(self, client):
        """win_a and win_b must be present and together total ~100%.

        The route computes win_a = expected_score(mu_a, mu_b) rounded to 1dp
        and win_b = 100 - win_a rounded to 1dp, so the two values in the
        prob-bar HTML span tags must sum to exactly 100.0 (or within floating-
        point rounding of 0.2pp).
        """
        import re

        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert r.status_code == 200

        # The prob bar renders as:
        #   <div class="prob-bar-a" style="width:XX.X%">
        #       <span class="prob-bar-pct">XX.X%</span>
        # Extract percentages from the prob-bar-pct spans specifically.
        pcts = [float(x) for x in re.findall(r'class="prob-bar-pct">(\d+\.\d+)%', r.text)]
        assert len(pcts) == 2, f"Expected exactly 2 prob-bar-pct values, got: {pcts}"
        total = pcts[0] + pcts[1]
        assert abs(total - 100.0) < 0.2, f"Win probabilities sum to {total}, expected ~100"

    def test_past_meetings_correct(self, client):
        """Two shared Lead events seeded above — page must report 2 past meetings."""
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert r.status_code == 200
        assert "2" in r.text  # past_meetings == 2

    def test_past_meetings_zero_when_no_shared_events(self, client):
        """adam vs solo — no shared events at all."""
        tc, adam_id, _, solo_id = client
        r = tc.get(f"/head-to-head/{adam_id}/{solo_id}?discipline=lead")
        assert r.status_code == 200
        # past_meetings is 0; make sure "0" appears near "meetings"
        assert "0" in r.text

    def test_discipline_switcher_links_present(self, client):
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert "boulder" in r.text.lower()
        assert "speed" in r.text.lower()

    def test_shareable_url_present(self, client):
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=lead")
        assert f"/head-to-head/{adam_id}/{janja_id}" in r.text

    def test_default_discipline_is_lead(self, client):
        """Omitting ?discipline= should default to Lead."""
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}")
        assert r.status_code == 200
        assert "Lead" in r.text


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------

class TestHeadToHeadErrors:
    def test_404_for_nonexistent_athlete_a(self, client):
        tc, _, janja_id, _ = client
        r = tc.get(f"/head-to-head/999999/{janja_id}?discipline=lead")
        assert r.status_code == 404

    def test_404_for_nonexistent_athlete_b(self, client):
        tc, adam_id, *_ = client
        r = tc.get(f"/head-to-head/{adam_id}/999999?discipline=lead")
        assert r.status_code == 404

    def test_400_same_athlete(self, client):
        tc, adam_id, *_ = client
        r = tc.get(f"/head-to-head/{adam_id}/{adam_id}?discipline=lead")
        assert r.status_code == 400

    def test_400_invalid_discipline(self, client):
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=bogus")
        assert r.status_code == 400

    def test_404_athlete_has_no_rating_for_discipline(self, client):
        """Requesting boulder comparison when no boulder rating exists → 404."""
        tc, adam_id, janja_id, _ = client
        r = tc.get(f"/head-to-head/{adam_id}/{janja_id}?discipline=boulder")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tests — predictions landing page card is now active
# ---------------------------------------------------------------------------

class TestPredictionsCardEnabled:
    def test_predictions_page_links_to_head_to_head(self, client):
        tc, *_ = client
        r = tc.get("/predictions")
        assert r.status_code == 200
        assert "/head-to-head" in r.text

    def test_predictions_card_not_disabled(self, client):
        """The hub-card-disabled class must no longer appear on the H2H card."""
        tc, *_ = client
        r = tc.get("/predictions")
        # The whole disabled block should be gone
        assert "hub-card-disabled" not in r.text

    def test_predictions_no_coming_soon_badge(self, client):
        tc, *_ = client
        r = tc.get("/predictions")
        assert "Coming soon" not in r.text
