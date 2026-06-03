"""Tests for scrape_upcoming_events — issue #25.

Exercises the two bugs fixed:
  Bug 1: UPCOMING_STATUSES now includes "registration_pending"
  Bug 2: _dcat_discipline_keyword falls back to name parsing when
         dc["discipline"] is null (upcoming events on the IFSC API).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from climbing_elo.models import (
    Base,
    Discipline,
    Event,
    EventTier,
    Result,
    Round,
    RoundType,
)
from climbing_elo.scraper.ifsc_api import (
    UPCOMING_STATUSES,
    _dcat_discipline_keyword,
    _parse_round_type,
    scrape_season,
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
        dc = {
            "discipline": None,
            "status": "registration_pending",
            "name": "BOULDER Women",
        }
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
        dc = {
            "discipline": "boulder&lead",
            "status": "finished",
            "name": "BOULDER&LEAD Mixed",
        }
        # The helper returns the first match — "boulder" appears in "boulder&lead"
        result = _dcat_discipline_keyword(dc)
        assert result == "boulder"

    def test_unknown_discipline_and_name_returns_none(self):
        dc = {
            "discipline": None,
            "status": "registration_pending",
            "name": "PARACLIMBING Open",
        }
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
        "leagues": [
            {
                "league_id": 1,
                "url": "/api/v1/season_leagues/457",
                "name": "IFSC World Cups 2026",
            }
        ],
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

        with (
            patch(
                "climbing_elo.scraper.ifsc_api.get_seasons", return_value=MOCK_SEASONS
            ),
            patch(
                "climbing_elo.scraper.ifsc_api._api_get",
                side_effect=_api_get_side_effect,
            ),
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)

            # Patch _api_get inside the module
            with patch(
                "climbing_elo.scraper.ifsc_api._api_get",
                side_effect=_api_get_side_effect,
            ):
                # Directly call with a mock client; _api_get is patched
                report = scrape_upcoming_events(
                    client, session, discipline="lead", seasons_ahead=1
                )

        stored = session.execute(select(Event)).scalars().all()
        names = [e.name for e in stored]

        assert report.events_stored == 2, (
            f"Expected 2 stored events, got {report.events_stored}. Names: {names}"
        )
        assert "World Climbing Series Innsbruck 2026" in names
        assert "World Climbing Series Chamonix 2026" in names
        assert "Past Lead World Cup 2026" not in names

    def test_upcoming_boulder_events_stored(self):
        """Boulder upcoming events are stored when discipline='boulder'."""
        session = _make_session()

        with patch(
            "climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)

            report = scrape_upcoming_events(
                client, session, discipline="boulder", seasons_ahead=1
            )

        stored = session.execute(select(Event)).scalars().all()

        assert report.events_stored == 2
        for e in stored:
            assert e.discipline == Discipline.BOULDER

    def test_finished_events_not_stored(self):
        """Finished-only events (dc status='finished') must not be stored."""
        session = _make_session()

        with patch(
            "climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)

            scrape_upcoming_events(client, session, discipline="lead", seasons_ahead=1)

        stored = session.execute(select(Event)).scalars().all()
        names = [e.name for e in stored]
        assert "Past Lead World Cup 2026" not in names

    def test_idempotent_on_second_call(self):
        """Calling scrape_upcoming_events twice should not duplicate events."""
        session = _make_session()

        with patch(
            "climbing_elo.scraper.ifsc_api._api_get", side_effect=_api_get_side_effect
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)

            scrape_upcoming_events(client, session, discipline="lead", seasons_ahead=1)
            report2 = scrape_upcoming_events(
                client, session, discipline="lead", seasons_ahead=1
            )

        stored = session.execute(select(Event)).scalars().all()
        assert len(stored) == 2
        assert report2.events_stored == 0
        assert report2.events_skipped == 2


# ---------------------------------------------------------------------------
# Regression test: scrape_season must ingest results for events pre-created
# as empty placeholders by scrape_upcoming_events (issue #155).
# ---------------------------------------------------------------------------


_FINISHED_LEAGUE = {
    "d_cats": [
        {"id": 1001, "discipline": "lead", "name": "LEAD Men"},
        {"id": 1002, "discipline": "lead", "name": "LEAD Women"},
    ],
    "events": [
        {
            "event_id": 9001,
            "event": "World Climbing Series Madrid 2026",
            "local_start_date": "2026-05-28",
            "d_cats": [
                {"id": 1001, "status": "finished"},
                {"id": 1002, "status": "finished"},
            ],
        },
    ],
}

_FINISHED_RESULTS_MEN = {
    "ranking": [
        {
            "athlete_id": 7001,
            "firstname": "Alice",
            "lastname": "Alpha",
            "country": "USA",
            "rounds": [
                {"round_name": "Qualification", "rank": 1, "score": "TOP"},
                {"round_name": "Final", "rank": 1, "score": "TOP"},
            ],
        },
        {
            "athlete_id": 7002,
            "firstname": "Bob",
            "lastname": "Beta",
            "country": "FRA",
            "rounds": [
                {"round_name": "Qualification", "rank": 2, "score": "40+"},
                {"round_name": "Final", "rank": 2, "score": "35+"},
            ],
        },
    ]
}

_FINISHED_RESULTS_WOMEN = {
    "ranking": [
        {
            "athlete_id": 7003,
            "firstname": "Carol",
            "lastname": "Gamma",
            "country": "JPN",
            "rounds": [
                {"round_name": "Qualification", "rank": 1, "score": "TOP"},
                {"round_name": "Final", "rank": 1, "score": "TOP"},
            ],
        },
    ]
}


def _finished_api_get(client, path):
    if "/season_leagues/" in path:
        return _FINISHED_LEAGUE
    if path.endswith("/result/1001"):
        return _FINISHED_RESULTS_MEN
    if path.endswith("/result/1002"):
        return _FINISHED_RESULTS_WOMEN
    return None


class TestScrapeSeasonIngestsExistingEvent:
    """scrape_season must NOT skip events that already exist as empty rows.

    Regression for issue #155: scrape_upcoming_events pre-creates an empty
    Event row when an upstream d_cat is in UPCOMING_STATUSES; once the d_cat
    flips to "finished", scrape_season previously short-circuited and never
    ingested rounds/results.
    """

    def test_empty_placeholder_gets_rounds_and_results(self):
        from datetime import date as _date

        session = _make_session()

        # Pre-create an empty Event row, mimicking what scrape_upcoming_events
        # would have stored when Madrid's d_cats were still "scheduled".
        placeholder = Event(
            name="World Climbing Series Madrid 2026",
            tier=EventTier.WORLD_CUP,
            country=None,
            season=2026,
            start_date=_date(2026, 5, 28),
            discipline=Discipline.LEAD,
        )
        session.add(placeholder)
        session.commit()
        placeholder_id = placeholder.id

        with patch(
            "climbing_elo.scraper.ifsc_api._api_get", side_effect=_finished_api_get
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)
            report = scrape_season(
                client,
                session,
                season_info={"name": "2026"},
                league_url="/api/v1/season_leagues/457",
                league_name="World Cups and World Championships",
                discipline="lead",
            )

        # No new Event row — the placeholder is reused.
        events = session.execute(select(Event)).scalars().all()
        assert len(events) == 1
        assert events[0].id == placeholder_id

        # Rounds and results materialized against the placeholder.
        rounds = (
            session.execute(select(Round).where(Round.event_id == placeholder_id))
            .scalars()
            .all()
        )
        assert len(rounds) == 4  # qual M/F + final M/F
        results = (
            session.execute(
                select(Result).join(Round).where(Round.event_id == placeholder_id)
            )
            .scalars()
            .all()
        )
        assert len(results) == 6  # 2M qual + 2M final + 1F qual + 1F final
        assert report.results_created == 6

    def test_rerun_is_idempotent(self):
        """A second scrape_season pass over an already-ingested event is a no-op."""
        session = _make_session()

        with patch(
            "climbing_elo.scraper.ifsc_api._api_get", side_effect=_finished_api_get
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)
            scrape_season(
                client,
                session,
                season_info={"name": "2026"},
                league_url="/api/v1/season_leagues/457",
                league_name="World Cups and World Championships",
                discipline="lead",
            )
            rounds_before = session.execute(select(Round)).scalars().all()
            results_before = session.execute(select(Result)).scalars().all()

            report2 = scrape_season(
                client,
                session,
                season_info={"name": "2026"},
                league_url="/api/v1/season_leagues/457",
                league_name="World Cups and World Championships",
                discipline="lead",
            )

        rounds_after = session.execute(select(Round)).scalars().all()
        results_after = session.execute(select(Result)).scalars().all()
        assert len(rounds_after) == len(rounds_before)
        assert len(results_after) == len(results_before)
        assert report2.results_created == 0


# ---------------------------------------------------------------------------
# Unit tests for _parse_round_type (issue #157)
# ---------------------------------------------------------------------------


class TestParseRoundType:
    """The IFSC API emits exactly four round_name strings 2012–2026:
    'Qualification', 'Semi-Final', 'Semi-final', 'Final'. The semi check
    must run before the final check because both contain 'final'."""

    def test_semi_final_capital_f(self):
        assert _parse_round_type("Semi-Final") == RoundType.SEMI

    def test_semi_final_lowercase_f(self):
        assert _parse_round_type("Semi-final") == RoundType.SEMI

    def test_semi_final_all_caps(self):
        assert _parse_round_type("SEMI-FINAL") == RoundType.SEMI

    def test_semi_final_with_space(self):
        assert _parse_round_type("semi final") == RoundType.SEMI

    def test_final(self):
        assert _parse_round_type("Final") == RoundType.FINAL

    def test_qualification(self):
        assert _parse_round_type("Qualification") == RoundType.QUALIFICATION

    def test_qualifications_plural(self):
        assert _parse_round_type("Qualifications") == RoundType.QUALIFICATION

    def test_unknown_name_warns_and_defaults_to_qualification(self):
        with patch("climbing_elo.scraper.ifsc_api.log.warning") as mock_warn:
            result = _parse_round_type("Bracket Round 1")
        assert result == RoundType.QUALIFICATION
        assert mock_warn.called
        assert "Unrecognized round_name" in mock_warn.call_args[0][0]

    def test_empty_string_does_not_crash(self):
        with patch("climbing_elo.scraper.ifsc_api.log.warning"):
            assert _parse_round_type("") == RoundType.QUALIFICATION

    def test_none_does_not_crash(self):
        with patch("climbing_elo.scraper.ifsc_api.log.warning"):
            assert _parse_round_type(None) == RoundType.QUALIFICATION  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration: scrape_season must split semi + final into distinct Round
# rows. Models the Inzai 2012 LEAD Women case where the real final (8
# athletes) was being eaten by the per-(round_id, athlete_id) existence
# check after both rounds collided on the same round_key (issue #157).
# ---------------------------------------------------------------------------


_INZAI_LEAGUE = {
    "d_cats": [{"id": 5, "discipline": "lead", "name": "LEAD Women"}],
    "events": [
        {
            "event_id": 745,
            "event": "IFSC Climbing Worldcup (L) - Inzai (JPN) 2012",
            "local_start_date": "2012-10-27",
            "d_cats": [{"id": 5, "status": "finished"}],
        },
    ],
}

# Top-3 finalists each have 3 rounds; #4 has qual+semi only (didn't make
# the final). The crucial property: rank IDs differ between semi and
# final (real final has Markovic 1, Kim 2, Vidmar 3; semi had Kim 1,
# Vidmar 1, with Markovic 7). Pre-fix, the semi row would survive labeled
# FINAL and the actual final's per-athlete rows would be dropped by the
# (round_id, athlete_id) existence check.
_INZAI_RESULTS = {
    "ranking": [
        {
            "athlete_id": 100,
            "firstname": "Mina",
            "lastname": "MARKOVIC",
            "country": "SLO",
            "rounds": [
                {"round_name": "Qualification", "rank": 5, "score": "TOP"},
                {"round_name": "Semi-Final", "rank": 7, "score": "30+"},
                {"round_name": "Final", "rank": 1, "score": "TOP"},
            ],
        },
        {
            "athlete_id": 101,
            "firstname": "Jain",
            "lastname": "KIM",
            "country": "KOR",
            "rounds": [
                {"round_name": "Qualification", "rank": 1, "score": "TOP"},
                {"round_name": "Semi-Final", "rank": 1, "score": "TOP"},
                {"round_name": "Final", "rank": 2, "score": "51+"},
            ],
        },
        {
            "athlete_id": 102,
            "firstname": "Maja",
            "lastname": "VIDMAR",
            "country": "SLO",
            "rounds": [
                {"round_name": "Qualification", "rank": 1, "score": "TOP"},
                {"round_name": "Semi-Final", "rank": 1, "score": "TOP"},
                {"round_name": "Final", "rank": 3, "score": "49+"},
            ],
        },
        {
            "athlete_id": 103,
            "firstname": "Did",
            "lastname": "NOTMAKEFINAL",
            "country": "FRA",
            "rounds": [
                {"round_name": "Qualification", "rank": 4, "score": "TOP"},
                {"round_name": "Semi-Final", "rank": 9, "score": "20"},
            ],
        },
    ]
}


def _inzai_api_get(client, path):
    if "/season_leagues/" in path:
        return _INZAI_LEAGUE
    if path.endswith("/result/5"):
        return _INZAI_RESULTS
    return None


class TestScrapeSeasonSplitsSemiFromFinal:
    def test_semi_and_final_become_distinct_rounds(self):
        """Issue #157: semi + final must materialize as two separate
        Round rows with the correct round_type. The Inzai 2012 final
        winner Markovic must be recorded as rank 1, not rank 7."""
        session = _make_session()

        with patch(
            "climbing_elo.scraper.ifsc_api._api_get", side_effect=_inzai_api_get
        ):
            import httpx

            client = MagicMock(spec=httpx.Client)
            scrape_season(
                client,
                session,
                season_info={"name": "2012"},
                league_url="/api/v1/season_leagues/2",
                league_name="IFSC World Cup",
                discipline="lead",
            )

        rounds = session.execute(select(Round)).scalars().all()
        round_types = {r.round_type for r in rounds}
        assert round_types == {
            RoundType.QUALIFICATION,
            RoundType.SEMI,
            RoundType.FINAL,
        }, f"Expected QUAL+SEMI+FINAL rounds, got {round_types}"

        final_round = next(r for r in rounds if r.round_type == RoundType.FINAL)
        semi_round = next(r for r in rounds if r.round_type == RoundType.SEMI)

        # Real final: 3 athletes (the finalists in our fixture).
        assert final_round.athlete_count == 3
        # Semi: 4 athletes (all who competed).
        assert semi_round.athlete_count == 4

        ranks_in_final = sorted(
            r.rank
            for r in session.execute(
                select(Result).where(Result.round_id == final_round.id)
            )
            .scalars()
            .all()
        )
        assert ranks_in_final == [1, 2, 3], (
            f"Final ranks should be [1, 2, 3], got {ranks_in_final}"
        )

        # The athlete who came 7th in the semi (Markovic) must be the
        # same athlete recorded 1st in the final. Pre-fix this would
        # fail because the final row was silently the semi data.
        semi_rank_7 = session.execute(
            select(Result).where(Result.round_id == semi_round.id, Result.rank == 7)
        ).scalar_one()
        final_rank_1 = session.execute(
            select(Result).where(Result.round_id == final_round.id, Result.rank == 1)
        ).scalar_one()
        assert semi_rank_7.athlete_id == final_rank_1.athlete_id, (
            "Markovic (semi rank 7) must be the same athlete as final rank 1"
        )


# ---------------------------------------------------------------------------
# Ingest validator (issue #157): scrape_season must warn when a final
# row exceeds the sane size threshold or has duplicate rank-1 finishers.
# ---------------------------------------------------------------------------


def _make_oversized_final_payload(n_finalists: int) -> dict:
    """Mock an IFSC API response with a single final round containing
    n_finalists athletes — emulates the bug shape we want to flag."""
    return {
        "ranking": [
            {
                "athlete_id": 200 + i,
                "firstname": f"Athlete{i}",
                "lastname": "TEST",
                "country": "USA",
                "rounds": [
                    {"round_name": "Final", "rank": i + 1, "score": "TOP"},
                ],
            }
            for i in range(n_finalists)
        ]
    }


def _oversized_league(n_finalists: int):
    return {
        "d_cats": [{"id": 9, "discipline": "lead", "name": "LEAD Women"}],
        "events": [
            {
                "event_id": 9999,
                "event": "Mock Oversized Final Event",
                "local_start_date": "2020-01-01",
                "d_cats": [{"id": 9, "status": "finished"}],
            },
        ],
    }


class TestIngestValidator:
    def test_oversized_final_is_flagged(self):
        """A final with > 12 athletes triggers a warning + report.errors
        entry. Threshold is the documented sane upper bound for real
        finals (Olympics=8, World Cup=6–8, slack for ties)."""
        session = _make_session()
        payload = _make_oversized_final_payload(26)

        def _api(client, path):
            if "/season_leagues/" in path:
                return _oversized_league(26)
            if path.endswith("/result/9"):
                return payload
            return None

        with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api):
            import httpx

            client = MagicMock(spec=httpx.Client)
            report = scrape_season(
                client,
                session,
                season_info={"name": "2020"},
                league_url="/api/v1/season_leagues/2",
                league_name="IFSC World Cup",
                discipline="lead",
            )

        assert any("Suspicious final size" in e for e in report.errors), (
            f"Expected a 'Suspicious final size' error, got: {report.errors}"
        )
        assert any("Mock Oversized Final Event" in e for e in report.errors)

    def test_normal_final_size_does_not_flag(self):
        """A normal 6-athlete final must not trigger the validator."""
        session = _make_session()
        payload = _make_oversized_final_payload(6)

        def _api(client, path):
            if "/season_leagues/" in path:
                return _oversized_league(6)
            if path.endswith("/result/9"):
                return payload
            return None

        with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api):
            import httpx

            client = MagicMock(spec=httpx.Client)
            report = scrape_season(
                client,
                session,
                season_info={"name": "2020"},
                league_url="/api/v1/season_leagues/2",
                league_name="IFSC World Cup",
                discipline="lead",
            )

        assert not any("Suspicious final size" in e for e in report.errors), (
            f"Validator misfired on a 6-athlete final: {report.errors}"
        )

    def test_multiple_rank_one_in_final_is_flagged(self):
        """A final with multiple rank-1 finishers is the semifinal-
        countback signature and must trigger a warning."""
        session = _make_session()
        # 4 finalists, two tied at rank 1 (the bug shape).
        payload = {
            "ranking": [
                {
                    "athlete_id": 300,
                    "firstname": "A",
                    "lastname": "ONE",
                    "country": "USA",
                    "rounds": [{"round_name": "Final", "rank": 1, "score": "TOP"}],
                },
                {
                    "athlete_id": 301,
                    "firstname": "B",
                    "lastname": "ONE",
                    "country": "USA",
                    "rounds": [{"round_name": "Final", "rank": 1, "score": "TOP"}],
                },
                {
                    "athlete_id": 302,
                    "firstname": "C",
                    "lastname": "THREE",
                    "country": "USA",
                    "rounds": [{"round_name": "Final", "rank": 3, "score": "40+"}],
                },
                {
                    "athlete_id": 303,
                    "firstname": "D",
                    "lastname": "FOUR",
                    "country": "USA",
                    "rounds": [{"round_name": "Final", "rank": 4, "score": "35"}],
                },
            ]
        }

        def _api(client, path):
            if "/season_leagues/" in path:
                return _oversized_league(4)
            if path.endswith("/result/9"):
                return payload
            return None

        with patch("climbing_elo.scraper.ifsc_api._api_get", side_effect=_api):
            import httpx

            client = MagicMock(spec=httpx.Client)
            report = scrape_season(
                client,
                session,
                season_info={"name": "2020"},
                league_url="/api/v1/season_leagues/2",
                league_name="IFSC World Cup",
                discipline="lead",
            )

        assert any("multiple rank-1 finishers" in e for e in report.errors), (
            f"Expected multi-rank-1 error, got: {report.errors}"
        )
