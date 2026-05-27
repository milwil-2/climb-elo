"""Tests for the athlete-profile scraper helper added for Issue #86.

Covers :func:`climbing_elo.scraper.ifsc_api.scrape_athlete_profile` —
the function that turns an IFSC ``/api/v1/athletes/{id}`` payload into the
dict our updater uses to set ``photo_url`` / ``height_cm`` / ``wingspan_cm`` /
``year_of_birth`` on local ``Athlete`` rows.

Issue #93 adds ``--only-missing`` / ``--force`` flag coverage for the
``scripts/scrape_athlete_profiles.py`` driver in :class:`TestScriptFlags`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climbing_elo.models import Athlete, Base, Gender
from climbing_elo.scraper.ifsc_api import scrape_athlete_profile


def _mock_response(data):
    """Build a MagicMock httpx response with the given JSON payload."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=data)
    return resp


class TestScrapeAthleteProfile:
    """``scrape_athlete_profile`` returns a sparse dict — only present fields."""

    def test_full_payload_returns_all_fields(self):
        """Sorato Anraku-style payload: photo + height + arm_span + birthday."""
        payload = {
            "id": 13040,
            "firstname": "Sorato",
            "lastname": "ANRAKU",
            "birthday": "2006-11-14",
            "height": 168,
            "arm_span": 181,
            "photo_url": "https://example.com/sorato.jpg",
            "country": "JPN",
        }
        client = MagicMock()
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(13040, client=client)
        assert out == {
            "photo_url": "https://example.com/sorato.jpg",
            "height_cm": 168,
            "wingspan_cm": 181,
            "year_of_birth": 2006,
        }

    def test_payload_with_no_metadata_returns_empty(self):
        """Athlete page with everything null — no fields should be returned."""
        payload = {
            "id": 1,
            "firstname": "Arno",
            "lastname": "DIMMLER",
            "birthday": None,
            "height": None,
            "arm_span": None,
            "photo_url": None,
        }
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert out == {}

    def test_partial_payload_only_returns_present_fields(self):
        """Only birthday is set — only year_of_birth is returned."""
        payload = {
            "id": 14924,
            "firstname": "Felipe",
            "lastname": "FERREIRA PINTO",
            "birthday": "2005-08-19",
            "height": None,
            "arm_span": None,
            "photo_url": None,
        }
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(14924, client=MagicMock())
        assert out == {"year_of_birth": 2005}

    def test_api_failure_returns_empty(self):
        """If the API returns None / non-dict, the function returns {} safely."""
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=None):
            out = scrape_athlete_profile(999999, client=MagicMock())
        assert out == {}

    def test_api_returns_list_instead_of_dict_returns_empty(self):
        """Defensive: if the API returns a list (e.g. bad endpoint), don't crash."""
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=[]):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert out == {}

    def test_empty_string_photo_url_skipped(self):
        """An empty-string photo_url shouldn't be persisted as a value."""
        payload = {"id": 1, "photo_url": "   ", "birthday": "1990-01-01"}
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert "photo_url" not in out
        assert out["year_of_birth"] == 1990

    def test_zero_height_skipped(self):
        """height=0 is not a real measurement — treat as missing."""
        payload = {"id": 1, "height": 0, "arm_span": 0}
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert "height_cm" not in out
        assert "wingspan_cm" not in out

    def test_float_height_coerced_to_int(self):
        """IFSC sometimes returns floats — make sure we coerce cleanly."""
        payload = {"id": 1, "height": 175.0, "arm_span": 188.5}
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert out["height_cm"] == 175
        assert out["wingspan_cm"] == 188

    def test_malformed_birthday_skipped(self):
        """Birthday strings shorter than 4 chars or non-numeric prefix are ignored."""
        payload = {"id": 1, "birthday": "ab"}
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert out == {}

        payload2 = {"id": 1, "birthday": "abcd-01-01"}
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload2):
            out2 = scrape_athlete_profile(1, client=MagicMock())
        assert out2 == {}

    def test_weight_field_never_returned(self):
        """IFSC has no weight field — even if upstream adds one, our shape is fixed."""
        # Our scraper never reads "weight", so adding it to the payload should
        # not surface anywhere.
        payload = {"id": 1, "weight": 65, "height": 175}
        with patch("climbing_elo.scraper.ifsc_api._api_get", return_value=payload):
            out = scrape_athlete_profile(1, client=MagicMock())
        assert "weight_kg" not in out
        assert out["height_cm"] == 175


# ---------------------------------------------------------------------------
# Issue #93: CLI flag coverage for scripts/scrape_athlete_profiles.py
# ---------------------------------------------------------------------------


def _load_script_module():
    """Import scripts/scrape_athlete_profiles.py as a module (not installed)."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "scrape_athlete_profiles.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scrape_athlete_profiles", script_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def seeded_session_factory():
    """In-memory DB with 5 athletes: 3 already have photo_url, 2 are missing.

    Returned: ``(session_factory, expected_missing_names, all_names)``.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as s:
        s.add_all(
            [
                Athlete(
                    name="Has Photo One",
                    gender=Gender.F,
                    photo_url="https://example.com/1.jpg",
                ),
                Athlete(
                    name="Has Photo Two",
                    gender=Gender.M,
                    photo_url="https://example.com/2.jpg",
                ),
                Athlete(
                    name="Has Photo Three",
                    gender=Gender.F,
                    photo_url="https://example.com/3.jpg",
                ),
                Athlete(name="Missing One", gender=Gender.M, photo_url=None),
                Athlete(name="Missing Two", gender=Gender.F, photo_url=None),
            ]
        )
        s.commit()
    missing = ["Missing One", "Missing Two"]
    all_names = [
        "Has Photo One",
        "Has Photo Two",
        "Has Photo Three",
        "Missing One",
        "Missing Two",
    ]
    return factory, missing, all_names


class TestScriptFlags:
    """``--only-missing`` / ``--force`` / no-flag semantics (Issue #93)."""

    def test_no_flag_errors_out(self, seeded_session_factory, capsys):
        """Without --only-missing or --force the script must refuse to run.

        argparse's ``parser.error()`` exits with code 2 and writes to stderr.
        """
        factory, _, _ = seeded_session_factory
        mod = _load_script_module()
        with (
            patch.object(mod, "init_db", return_value=factory),
            patch("sys.argv", ["scrape_athlete_profiles.py"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "--only-missing" in captured.err
        assert "--force" in captured.err

    def test_only_missing_filters_to_null_photo_rows(self, seeded_session_factory):
        """``--only-missing`` should queue exactly the 2 athletes whose
        photo_url is NULL — the 3 with photos must be untouched.
        """
        factory, expected_missing, _ = seeded_session_factory
        mod = _load_script_module()

        # Map every queued athlete to a fake IFSC id so the discovery walk is
        # a no-op (mocked) and every athlete looks scrapeable.
        scraped: list[str] = []

        def _fake_id_index(session, client, since_year, delay):
            return {
                ("Missing One", Gender.M): 100,
                ("Missing Two", Gender.F): 200,
                ("Has Photo One", Gender.F): 300,
                ("Has Photo Two", Gender.M): 400,
                ("Has Photo Three", Gender.F): 500,
            }

        def _fake_scrape(ifsc_id, client):
            # Reverse-lookup the athlete by ifsc_id for assertion convenience.
            name_by_id = {
                100: "Missing One",
                200: "Missing Two",
                300: "Has Photo One",
                400: "Has Photo Two",
                500: "Has Photo Three",
            }
            scraped.append(name_by_id[ifsc_id])
            return {"photo_url": f"https://example.com/new-{ifsc_id}.jpg"}

        with (
            patch.object(mod, "init_db", return_value=factory),
            patch.object(mod, "_build_ifsc_id_index", _fake_id_index),
            patch.object(mod, "scrape_athlete_profile", _fake_scrape),
            patch.object(mod.time, "sleep", lambda *a, **kw: None),
            patch(
                "sys.argv",
                ["scrape_athlete_profiles.py", "--only-missing", "--delay", "0"],
            ),
        ):
            mod.main()

        # Only the 2 NULL-photo athletes hit the API.
        assert sorted(scraped) == sorted(expected_missing)

    def test_force_scrapes_every_athlete(self, seeded_session_factory):
        """``--force`` should hit the IFSC API for every athlete (5 of 5),
        even those that already have a photo_url.
        """
        factory, _, all_names = seeded_session_factory
        mod = _load_script_module()

        scraped: list[str] = []

        def _fake_id_index(session, client, since_year, delay):
            return {
                ("Missing One", Gender.M): 100,
                ("Missing Two", Gender.F): 200,
                ("Has Photo One", Gender.F): 300,
                ("Has Photo Two", Gender.M): 400,
                ("Has Photo Three", Gender.F): 500,
            }

        def _fake_scrape(ifsc_id, client):
            name_by_id = {
                100: "Missing One",
                200: "Missing Two",
                300: "Has Photo One",
                400: "Has Photo Two",
                500: "Has Photo Three",
            }
            scraped.append(name_by_id[ifsc_id])
            return {"photo_url": f"https://example.com/new-{ifsc_id}.jpg"}

        with (
            patch.object(mod, "init_db", return_value=factory),
            patch.object(mod, "_build_ifsc_id_index", _fake_id_index),
            patch.object(mod, "scrape_athlete_profile", _fake_scrape),
            patch.object(mod.time, "sleep", lambda *a, **kw: None),
            patch(
                "sys.argv",
                ["scrape_athlete_profiles.py", "--force", "--delay", "0"],
            ),
        ):
            mod.main()

        assert sorted(scraped) == sorted(all_names)

    def test_only_missing_and_force_are_mutually_exclusive(
        self, seeded_session_factory, capsys
    ):
        """argparse should reject ``--only-missing --force`` together."""
        mod = _load_script_module()
        with (
            patch(
                "sys.argv",
                ["scrape_athlete_profiles.py", "--only-missing", "--force"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            mod.main()
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert (
            "not allowed with" in captured.err or "mutually exclusive" in captured.err
        )
