"""Tests for scrape_upcoming_events — issue #25.

Exercises the two bugs fixed:
  Bug 1: UPCOMING_STATUSES now includes "registration_pending"
  Bug 2: _dcat_discipline_keyword falls back to name parsing when
         dc["discipline"] is null (upcoming events on the IFSC API).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from climbing_elo.models import Base, Discipline, Event
from climbing_elo.scraper.ifsc_api import (
    UPCOMING_STATUSES,
    _dcat_discipline_keyword,
    scrape_upcoming_events,
)


# ---------------------------------------------------------------------------
# Unit tests for _dcat_discipline_keyword
# ---------------------------------------------------------------------------


class TestDcatDisciplineKeyword:
    """_dcat_discipline_keyword returns the canonical keyword or None."""

    def test_finished_lead_dcat(self):
        """Old shape: finished event, discipline field set."""
        dc = {"discipline": "lead", "status": "finished", "name": "LEAD Men"}
        assert _dcat_discipline_keyword(dc) == "lead"

    def test_finished_boulder_dcat(self):
        dc = {"discipline": "boulder", "status": "finished", "name": "BOULDER Women"}
        assert _dcat_discipline_keyword(dc) == "boulder"

    def test_finished_speed_dcat(self):
        dc = {"discipline": "speed", "status": "finished", "name": "SPEED Men"}
        assert _dcat_discipline_keyword(dc) == "speed"

    def test_upcoming_null_discipline_lead(self):
        """New shape: upcoming event, discipline is null — name fallback."""
        dc = {"discipline": None, "status": "registration_pending", "name": "LEAD Men"}
        assert _dcat_discipline_keyword(dc) == "lead"

    def test_upcoming_null_discipline_boulder(self):
        dc = {"discipline": None, "status": "registration_pending", "name": "BOULDER Women"}
        assert _dcat_discipline_keyword(dc) == "boulder"

    def test_upcoming_null_discipline_speed(self):
        dc = {"discipline": None, "status": "registration_pending", "name": "SPEED Men"}
        assert _dcat_discipline_keyword(dc) == "speed"

    def test_upcoming_empty_string_discipline(self):
        """Empty-string discipline should also fall back to name."""
        dc = {"discipline": "", "status": "registration_pending", "name": "LEAD Women"}
        assert _dcat_discipline_keyword(dc) == "lead"

    def test_combined_boulder_lead_excludes_via_caller(self):
        """Boulder&Lead combined d_cats: keyword = 'boulder' (has both).
        Caller is responsible for excluding combined categories for
        discipline-specific scrapes."""
        dc = {"discipline": "boulder&lead", "status": "finished", "name": "BOULDER&LEAD Mixed"}
        # The helper returns the first match — "boulder" appears in "boulder&lead"
        result = _dcat_discipline_keyword(dc)
        assert result == "boulder"

    def test_unknown_discipline_and_name_returns_none(self):
        dc = {"discipline": None, "status": "registration_pending", "name": "PARACLIMBING Open"}
        assert _dcat_discipline_keyword(dc) is None

    def test_arbitrary_discipline_string_not_propagated(self):
        """Arbitrary discipline strings not in the allowlist return None."""
        dc = {"discipline": "something_weird", "name": "SOMETHING WEIRD Men"}
        assert _dcat_discipline_keyword(dc) is None


# ---------------------------------------------------------------------------
# Unit test: UPCOMING_STATUSES contains the required values
# ---------------------------------------------------------------------------


class TestUpcomingStatuses:
    def test_registration_pending_included(self):
        assert "registration_pending" in UPCOMING_STATUSES

    def test_legacy_statuses_included(self):
        for s in ("scheduled", "registration", "live"):
            assert s in UPCOMING_STATUSES

    def test_in_progress_included(self):
        assert "in_progress" in UPCOMING_STATUSES


# ---------------------------------------------------------------------------
# Integration test: scrape_upcoming_events with mocked API
# ---------------------------------------------------------------------------


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


MOCK_SEASONS = [
    {
        "name": "2026",
        "leagues": [{"league_id": 1, "url": "/api/v1/season_leagues/457", "name": "IFSC World Cups 2026"}],
    }
]

MOCK_LEAGUE = {
    "d_cats": [
        # Upcoming LEAD Men — discipline null, name-encoded
        {"id": 1001, "discipline": None, "name": "LEAD Men"},
        # Upcoming LEAD Women — discipline null, name-encoded
        {"id": 1002, "discipline": None, "name": "LEAD Women"},
        # Upcoming BOULDER Men — different discipline, should not be matched for "lead"
        {"id": 1003, "discipline": None, "name": "BOULDER Men"},
        # Finished LEAD Men from a past round — should NOT be stored as upcoming
        {"id": 1004, "discipline": "lead", "name": "LEAD Men"},
    ],
    "events": [
        {
            "event_id": 501,
            "event": "World Climbing Series Innsbruck 2026",
            "local_start_date": "2026-07-01",
            "d_cats": [
                # Both lead dcats are upcoming
                {"id": 1001, "status": "registration_pending"},
                {"id": 1002, "status": "registration_pending"},
                # Boulder dcat is also upcoming but separate discipline
                {"id": 1003, "status": "registration_pending"},
                # A finished dcat for the same event — should not count
                {"id": 1004, "status": "finished"},
            ],
        },
        {
            "event_id": 502,
            "event": "World Climbing Series Chamonix 2026",
            "local_start_date": "2026-08-01",
            "d_cats": [
                {"id": 1001, "status": "scheduled"},
                {"id": 1002, "status": "scheduled"},
                {"id": 1003, "status": "scheduled"},
            ],
        },
        {
            "event_id": 503,
            "event": "Past Lead World Cup 2026",
            "local_start_date": "2026-01-01",
            "d_cats": [
                # All finished — should NOT appear
                {"id": 1004, "status": "finished"},
            ],
        },
    ],
}


def _api_get_side_effect(client, path):
    if path == "/api/v1/":
        return {"seasons": MOCK_SEASONS}
    if "/season_leagues/" in path:
        return MOCK_LEAGUE
    return None


class TestScrapeUpcomingEventsIntegration:
    def test_upcoming_lead_events_stored(self):
        """Both upcoming lead events should be stored; the past event should not."""
        session = _make_session()

        with patch("climbing_elo.scraper.ifsc_api.get_seasons", return_value=MOCK_SEASONS), \
             patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect):

            import httpx
            client = MagicMock(spec=httpx.Client)

            # Patch _api_get inside the module
            with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect):
                from climbing_elo.scraper import ifsc_api as api_mod

                # Directly call with a mock client; _api_get is patched
                report = scrape_upcoming_events(client, session, discipline="lead", seasons_ahead=1)

        stored = session.execute(select(Event)).scalars().all()
        names = [e.name for e in stored]

        assert report.events_stored == 2, f"Expected 2 stored events, got {report.events_stored}. Names: {names}"
        assert "World Climbing Series Innsbruck 2026" in names
        assert "World Climbing Series Chamonix 2026" in names
        assert "Past Lead World Cup 2026" not in names

    def test_upcoming_boulder_events_stored(self):
        """Boulder upcoming events are stored when discipline='boulder'."""
        session = _make_session()

        with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect):
            import httpx
            client = MagicMock(spec=httpx.Client)

            report = scrape_upcoming_events(client, session, discipline="boulder", seasons_ahead=1)

        stored = session.execute(select(Event)).scalars().all()
        names = [e.name for e in stored]

        assert report.events_stored == 2
        for e in stored:
            assert e.discipline == Discipline.BOULDER

    def test_finished_events_not_stored(self):
        """Finished-only events (dc status='finished') must not be stored."""
        session = _make_session()

        with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect):
            import httpx
            client = MagicMock(spec=httpx.Client)

            report = scrape_upcoming_events(client, session, discipline="lead", seasons_ahead=1)

        stored = session.execute(select(Event)).scalars().all()
        names = [e.name for e in stored]
        assert "Past Lead World Cup 2026" not in names

    def test_idempotent_on_second_call(self):
        """Calling scrape_upcoming_events twice should not duplicate events."""
        session = _make_session()

        with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect):
            import httpx
            client = MagicMock(spec=httpx.Client)

            report1 = scrape_upcoming_events(client, session, discipline="lead", seasons_ahead=1)
            report2 = scrape_upcoming_events(client, session, discipline="lead", seasons_ahead=1)

        stored = session.execute(select(Event)).scalars().all()
        assert len(stored) == 2
        assert report2.events_stored == 0
        assert report2.events_skipped == 2
