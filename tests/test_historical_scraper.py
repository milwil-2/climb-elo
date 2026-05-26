"""Tests for the historical IFSC backfill scraper and validate_db_counts.

Covers:
  1. Argument parsing — year-range flags, --delay-ms, --discipline.
  2. Idempotency — re-running scrape_all_seasons with mocked API produces 0
     new Event rows on the second call.
  3. validate_db_counts output format — get_counts returns the expected
     structure and _print_table / _print_csv produce parseable output.
"""

from __future__ import annotations

import csv
import io
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Ensure src/ is on the path even when tests run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climbing_elo.models import (
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)
from climbing_elo.scraper.ifsc_api import scrape_all_seasons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# Minimal mock API payloads
# ---------------------------------------------------------------------------

MOCK_SEASONS = [
    {
        "name": "2018",
        "leagues": [
            {
                "league_id": 1,
                "url": "/api/v1/season_leagues/300",
                "name": "IFSC World Cups 2018",
            }
        ],
    },
    {
        "name": "2019",
        "leagues": [
            {
                "league_id": 1,
                "url": "/api/v1/season_leagues/310",
                "name": "IFSC World Cups 2019",
            }
        ],
    },
    {
        # Season outside the target range — must be skipped.
        "name": "2015",
        "leagues": [
            {
                "league_id": 1,
                "url": "/api/v1/season_leagues/200",
                "name": "IFSC World Cups 2015",
            }
        ],
    },
]

MOCK_LEAGUE_2018 = {
    "d_cats": [
        {"id": 3001, "discipline": "lead", "name": "LEAD Men"},
        {"id": 3002, "discipline": "lead", "name": "LEAD Women"},
    ],
    "events": [
        {
            "event_id": 701,
            "event": "IFSC World Cup Briançon 2018",
            "local_start_date": "2018-07-05",
            "d_cats": [
                {"id": 3001, "status": "finished"},
                {"id": 3002, "status": "finished"},
            ],
        },
    ],
}

MOCK_LEAGUE_2019 = {
    "d_cats": [
        {"id": 3003, "discipline": "lead", "name": "LEAD Men"},
    ],
    "events": [
        {
            "event_id": 702,
            "event": "IFSC World Cup Chamonix 2019",
            "local_start_date": "2019-07-13",
            "d_cats": [
                {"id": 3003, "status": "finished"},
            ],
        },
    ],
}

# Minimal result payload for a single athlete in a single round.
_RESULT_PAYLOAD = {
    "ranking": [
        {
            "athlete_id": 99001,
            "firstname": "Mock",
            "lastname": "Athlete",
            "country": "FRA",
            "rounds": [
                {
                    "round_name": "Final",
                    "rank": 1,
                    "score": "42",
                    "ascents": [],
                }
            ],
        }
    ]
}


def _api_get_side_effect(client, path):
    """Fake _api_get that returns canned data for known paths."""
    if path == "/api/v1/":
        return {"seasons": MOCK_SEASONS}
    if "/season_leagues/300" in path:
        return MOCK_LEAGUE_2018
    if "/season_leagues/310" in path:
        return MOCK_LEAGUE_2019
    if "/season_leagues/200" in path:
        # Season 2015 — would be called only if year-filter fails
        return {"d_cats": [], "events": []}
    if "/events/" in path and "/result/" in path:
        return _RESULT_PAYLOAD
    return None


# ---------------------------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------------------------


def _load_historical_module():
    """Import the historical scraper script as a module (not installed)."""
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "scrape_ifsc_historical.py"
    )
    spec = importlib.util.spec_from_file_location("scrape_ifsc_historical", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_validate_module():
    """Import the validate_db_counts script as a module (not installed)."""
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "validate_db_counts.py"
    )
    spec = importlib.util.spec_from_file_location("validate_db_counts", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestArgumentParsing:
    """_parse_args() in scrape_ifsc_historical.py honours all flags."""

    def test_defaults(self):
        mod = _load_historical_module()
        with patch("sys.argv", ["scrape_ifsc_historical.py"]):
            args = mod._parse_args()
        assert args.min_year == 2012
        assert args.max_year == 2026
        assert args.discipline == "all"
        assert args.delay_ms == 200

    def test_custom_year_range(self):
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--min-year",
                "2018",
                "--max-year",
                "2022",
            ],
        ):
            args = mod._parse_args()
        assert args.min_year == 2018
        assert args.max_year == 2022

    def test_discipline_lead(self):
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--discipline",
                "lead",
            ],
        ):
            args = mod._parse_args()
        assert args.discipline == "lead"

    def test_discipline_boulder(self):
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--discipline",
                "boulder",
            ],
        ):
            args = mod._parse_args()
        assert args.discipline == "boulder"

    def test_discipline_speed(self):
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--discipline",
                "speed",
            ],
        ):
            args = mod._parse_args()
        assert args.discipline == "speed"

    def test_discipline_all(self):
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--discipline",
                "all",
            ],
        ):
            args = mod._parse_args()
        assert args.discipline == "all"

    def test_delay_ms(self):
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--delay-ms",
                "500",
            ],
        ):
            args = mod._parse_args()
        assert args.delay_ms == 500

    def test_invalid_discipline_rejected(self):
        """argparse should exit on an unknown --discipline value."""
        mod = _load_historical_module()
        with patch(
            "sys.argv",
            [
                "scrape_ifsc_historical.py",
                "--discipline",
                "paraclimbing",
            ],
        ):
            with pytest.raises(SystemExit):
                mod._parse_args()


# ---------------------------------------------------------------------------
# 2. Year-range filtering
# ---------------------------------------------------------------------------


class TestYearRangeFiltering:
    """Seasons outside [min_year, max_year] must not be scraped."""

    def test_only_seasons_in_range_are_scraped(self):
        """Seasons 2018 and 2019 are fetched; 2015 is outside range and skipped."""
        session = _make_session()

        with (
            patch(
                "climbing_elo.scraper.ifsc_api.get_seasons",
                return_value=MOCK_SEASONS,
            ),
            patch(
                "climbing_elo.scraper.ifsc_api._api_get",
                side_effect=_api_get_side_effect,
            ),
        ):
            scrape_all_seasons(
                session,
                min_year=2018,
                max_year=2019,
                discipline="lead",
            )

        events = session.execute(select(Event)).scalars().all()
        names = {e.name for e in events}

        assert "IFSC World Cup Briançon 2018" in names
        assert "IFSC World Cup Chamonix 2019" in names
        # 2015 must not appear
        assert all(e.season >= 2018 for e in events), (
            f"Found event from outside range: {[e.season for e in events]}"
        )

    def test_no_events_when_range_excludes_all_seasons(self):
        """An empty year range should produce 0 events."""
        session = _make_session()

        with (
            patch(
                "climbing_elo.scraper.ifsc_api.get_seasons",
                return_value=MOCK_SEASONS,
            ),
            patch(
                "climbing_elo.scraper.ifsc_api._api_get",
                side_effect=_api_get_side_effect,
            ),
        ):
            report = scrape_all_seasons(
                session,
                min_year=2025,
                max_year=2025,
                discipline="lead",
            )

        events = session.execute(select(Event)).scalars().all()
        assert events == []
        assert report.events_scraped == 0


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------


class TestHistoricalScraperIdempotency:
    """Re-running scrape_all_seasons must produce 0 new Event rows."""

    def test_second_run_creates_no_new_events(self):
        """Two consecutive calls to scrape_all_seasons store exactly the same rows."""
        session = _make_session()

        with (
            patch(
                "climbing_elo.scraper.ifsc_api.get_seasons",
                return_value=MOCK_SEASONS,
            ),
            patch(
                "climbing_elo.scraper.ifsc_api._api_get",
                side_effect=_api_get_side_effect,
            ),
        ):
            scrape_all_seasons(
                session,
                min_year=2018,
                max_year=2019,
                discipline="lead",
            )
            count_after_first = len(session.execute(select(Event)).scalars().all())

            scrape_all_seasons(
                session,
                min_year=2018,
                max_year=2019,
                discipline="lead",
            )
            count_after_second = len(session.execute(select(Event)).scalars().all())

        assert count_after_first == count_after_second, (
            f"Event count changed on re-run: {count_after_first} → {count_after_second}"
        )

    def test_second_run_creates_no_new_results(self):
        """Re-running the scraper must not insert duplicate Result rows."""
        session = _make_session()

        with (
            patch(
                "climbing_elo.scraper.ifsc_api.get_seasons",
                return_value=MOCK_SEASONS,
            ),
            patch(
                "climbing_elo.scraper.ifsc_api._api_get",
                side_effect=_api_get_side_effect,
            ),
        ):
            scrape_all_seasons(
                session,
                min_year=2018,
                max_year=2018,
                discipline="lead",
            )
            results_after_first = len(session.execute(select(Result)).scalars().all())

            scrape_all_seasons(
                session,
                min_year=2018,
                max_year=2018,
                discipline="lead",
            )
            results_after_second = len(session.execute(select(Result)).scalars().all())

        assert results_after_first == results_after_second, (
            f"Result count changed on re-run: {results_after_first} → {results_after_second}"
        )

    def test_event_unique_constraint_prevents_duplicates(self):
        """The UNIQUE(name, season, discipline) constraint is present on the events table."""
        from sqlalchemy import inspect as sa_inspect

        engine = _make_engine()
        inspector = sa_inspect(engine)
        constraint_names = [
            c["name"] for c in inspector.get_unique_constraints("events")
        ]
        assert "uq_event_name_season_discipline" in constraint_names, (
            f"uq_event_name_season_discipline missing; found: {constraint_names}"
        )


# ---------------------------------------------------------------------------
# 4. validate_db_counts — get_counts structure
# ---------------------------------------------------------------------------


def _seed_validate_data(engine) -> None:
    """Populate the in-memory DB with two events across two seasons/disciplines."""
    with engine.begin():
        from sqlalchemy.orm import Session as _Session

        session = _Session(bind=engine)

        ev1 = Event(
            name="WC Lead 2018",
            tier=EventTier.WORLD_CUP,
            season=2018,
            start_date=date(2018, 7, 1),
            discipline=Discipline.LEAD,
        )
        ev2 = Event(
            name="WC Boulder 2018",
            tier=EventTier.WORLD_CUP,
            season=2018,
            start_date=date(2018, 8, 1),
            discipline=Discipline.BOULDER,
        )
        ev3 = Event(
            name="WC Lead 2019",
            tier=EventTier.WORLD_CUP,
            season=2019,
            start_date=date(2019, 7, 1),
            discipline=Discipline.LEAD,
        )
        from climbing_elo.models import Athlete

        athlete = Athlete(name="Test Athlete", gender=Gender.M)
        session.add_all([ev1, ev2, ev3, athlete])
        session.flush()

        rnd1 = Round(
            event_id=ev1.id,
            round_type=RoundType.FINAL,
            gender=Gender.M,
            athlete_count=1,
        )
        session.add(rnd1)
        session.flush()

        result1 = Result(
            round_id=rnd1.id,
            athlete_id=athlete.id,
            rank=1,
        )
        session.add(result1)
        session.commit()


class TestValidateDbCounts:
    """get_counts() and the format helpers work correctly."""

    def test_get_counts_returns_rows(self):
        """get_counts returns a list of dicts with the expected keys."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)

        rows = validate_mod.get_counts(engine)

        assert isinstance(rows, list)
        assert len(rows) > 0
        for row in rows:
            assert "season" in row
            assert "discipline" in row
            assert "events" in row
            assert "results" in row

    def test_get_counts_correct_event_totals(self):
        """Event counts match what was seeded."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)

        rows = validate_mod.get_counts(engine)
        total_events = sum(r["events"] for r in rows)
        # 3 events were seeded
        assert total_events == 3

    def test_get_counts_result_count_correct(self):
        """Result counts reflect actual inserted rows."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)

        rows = validate_mod.get_counts(engine)
        total_results = sum(r["results"] for r in rows)
        # Only ev1 (2018 Lead) got a result row
        assert total_results == 1

    def test_get_counts_year_filter_min(self):
        """min_year filter excludes earlier seasons."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)

        rows = validate_mod.get_counts(engine, min_year=2019)
        assert all(r["season"] >= 2019 for r in rows)

    def test_get_counts_year_filter_max(self):
        """max_year filter excludes later seasons."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)

        rows = validate_mod.get_counts(engine, max_year=2018)
        assert all(r["season"] <= 2018 for r in rows)

    def test_get_counts_empty_db_returns_empty_list(self):
        """An empty database returns an empty list, not an error."""
        validate_mod = _load_validate_module()
        engine = _make_engine()  # no seeding

        rows = validate_mod.get_counts(engine)
        assert rows == []

    def test_print_table_no_crash_on_populated_data(self, capsys):
        """_print_table outputs at minimum a header and a total line."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)
        rows = validate_mod.get_counts(engine)

        validate_mod._print_table(rows)
        captured = capsys.readouterr()
        assert "Season" in captured.out
        assert "Events" in captured.out
        assert "Results" in captured.out
        assert "TOTAL" in captured.out

    def test_print_table_empty_prints_no_data_message(self, capsys):
        """_print_table on empty data prints a friendly message."""
        validate_mod = _load_validate_module()
        validate_mod._print_table([])
        captured = capsys.readouterr()
        assert "No data" in captured.out

    def test_print_csv_parseable(self, capsys):
        """_print_csv produces valid CSV with the expected header fields."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)
        rows = validate_mod.get_counts(engine)

        validate_mod._print_csv(rows)
        captured = capsys.readouterr()

        reader = csv.DictReader(io.StringIO(captured.out))
        header = reader.fieldnames
        assert header is not None
        assert set(header) == {"season", "discipline", "events", "results"}

        data_rows = list(reader)
        assert len(data_rows) == len(rows)

    def test_print_csv_values_match_get_counts(self, capsys):
        """CSV values are consistent with what get_counts returned."""
        validate_mod = _load_validate_module()
        engine = _make_engine()
        _seed_validate_data(engine)
        rows = validate_mod.get_counts(engine)

        validate_mod._print_csv(rows)
        captured = capsys.readouterr()

        reader = csv.DictReader(io.StringIO(captured.out))
        csv_rows = list(reader)

        total_csv_events = sum(int(r["events"]) for r in csv_rows)
        total_raw_events = sum(r["events"] for r in rows)
        assert total_csv_events == total_raw_events

    def test_validate_args_defaults(self):
        """_parse_args in validate_db_counts has correct defaults."""
        validate_mod = _load_validate_module()
        with patch("sys.argv", ["validate_db_counts.py"]):
            args = validate_mod._parse_args()
        assert args.db is None
        assert args.min_year is None
        assert args.max_year is None
        assert args.csv is False

    def test_validate_args_csv_flag(self):
        validate_mod = _load_validate_module()
        with patch("sys.argv", ["validate_db_counts.py", "--csv"]):
            args = validate_mod._parse_args()
        assert args.csv is True

    def test_validate_args_year_range(self):
        validate_mod = _load_validate_module()
        with patch(
            "sys.argv",
            [
                "validate_db_counts.py",
                "--min-year",
                "2016",
                "--max-year",
                "2023",
            ],
        ):
            args = validate_mod._parse_args()
        assert args.min_year == 2016
        assert args.max_year == 2023
