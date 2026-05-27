"""Tests for the athlete-profile scraper helper added for Issue #86.

Covers :func:`climbing_elo.scraper.ifsc_api.scrape_athlete_profile` —
the function that turns an IFSC ``/api/v1/athletes/{id}`` payload into the
dict our updater uses to set ``photo_url`` / ``height_cm`` / ``wingspan_cm`` /
``year_of_birth`` on local ``Athlete`` rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
