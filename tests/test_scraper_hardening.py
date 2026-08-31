"""Scraper / engine input-hardening tests (issues #199-#203).

Covers the five security-review findings:
  #199  NaN/Inf lead & boulder scores are rejected; margin multipliers stay finite.
  #200  _api_get refuses absolute/off-host (SSRF) targets.
  #201  athlete_id=None no longer collapses a whole season onto one athlete.
  #202  photo_url is only stored when it is https on the IFSC host, within a cap.
  #203  Upstream fetches abort past a decoded-body size cap (decompression bomb).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from climbing_elo.engine.elo import (
    compute_boulder_margin_multiplier,
    compute_margin_multiplier,
    compute_speed_margin_multiplier,
)
from climbing_elo.models import Athlete, Gender
from climbing_elo.scraper.http_utils import (
    ResponseTooLargeError,
    read_capped_bytes,
    read_capped_text,
)
from climbing_elo.scraper.ifsc_api import (
    _api_get,
    _get_or_create_athlete,
    _is_valid_photo_url,
    _parse_lead_score,
    scrape_athlete_profile,
)


class _FakeStream:
    """Minimal stand-in for the object returned by ``client.stream(...)``."""

    def __init__(self, status_code=200, chunks=(b"{}",), encoding="utf-8"):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.encoding = encoding

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        yield from self._chunks


# ---------------------------------------------------------------------------
# #199 — non-finite score rejection
# ---------------------------------------------------------------------------


class TestNonFiniteScores:
    @pytest.mark.parametrize(
        "bad", ["nan", "NaN", "inf", "-inf", "Infinity", "-Infinity"]
    )
    def test_lead_score_nonfinite_rejected(self, bad):
        raw, value = _parse_lead_score(bad)
        assert raw == bad
        assert value is None

    def test_lead_score_nonfinite_plus_branch(self):
        # The "34+" plus-branch must also reject a non-finite mantissa.
        assert _parse_lead_score("nan+")[1] is None
        assert _parse_lead_score("inf+")[1] is None

    def test_lead_score_valid_still_parses(self):
        assert _parse_lead_score("34+")[1] == 34.5
        assert _parse_lead_score("TOP")[1] == 999.0
        assert _parse_lead_score("28")[1] == 28.0
        assert _parse_lead_score("bad")[1] is None

    def test_margin_multiplier_neutral_on_nonfinite(self):
        assert compute_margin_multiplier(float("nan"), 10.0) == 1.0
        assert compute_margin_multiplier(10.0, float("inf")) == 1.0
        assert compute_margin_multiplier(30.0, 10.0, rating_gap=float("nan")) == 1.0

    def test_margin_multiplier_stays_finite_normal(self):
        assert math.isfinite(compute_margin_multiplier(30.0, 10.0))

    def test_boulder_margin_multiplier_neutral_on_nonfinite(self):
        # Delegates to compute_margin_multiplier, so it inherits the guard.
        assert compute_boulder_margin_multiplier(float("nan"), 100.0) == 1.0

    def test_speed_margin_multiplier_neutral_on_nonfinite(self):
        assert compute_speed_margin_multiplier(float("inf"), 5.0) == 1.0
        assert math.isfinite(compute_speed_margin_multiplier(5.0, 6.0))


# ---------------------------------------------------------------------------
# #200 — SSRF guard in _api_get
# ---------------------------------------------------------------------------


class TestApiGetSSRF:
    def test_absolute_offhost_refused(self):
        client = MagicMock()
        assert _api_get(client, "http://evil.example.com/x") is None
        client.stream.assert_not_called()
        client.get.assert_not_called()

    def test_https_offhost_refused(self):
        client = MagicMock()
        assert _api_get(client, "https://evil.example.com/api/v1/") is None
        client.stream.assert_not_called()

    def test_lookalike_host_refused(self):
        client = MagicMock()
        assert _api_get(client, "https://ifsc.results.info.evil.com/x") is None
        client.stream.assert_not_called()

    def test_non_http_scheme_refused(self):
        client = MagicMock()
        assert _api_get(client, "file:///etc/passwd") is None
        client.stream.assert_not_called()

    def test_relative_path_allowed(self):
        client = MagicMock()
        client.stream = MagicMock(return_value=_FakeStream(chunks=[b'{"ok": true}']))
        assert _api_get(client, "/api/v1/") == {"ok": True}
        args, _ = client.stream.call_args
        assert args[1] == "https://ifsc.results.info/api/v1/"

    def test_same_host_absolute_allowed(self):
        client = MagicMock()
        client.stream = MagicMock(return_value=_FakeStream(chunks=[b'{"ok": true}']))
        url = "https://ifsc.results.info/api/v1/season_leagues/1"
        assert _api_get(client, url) == {"ok": True}
        args, _ = client.stream.call_args
        assert args[1] == url


# ---------------------------------------------------------------------------
# #201 — null athlete_id must not collapse the season
# ---------------------------------------------------------------------------


class TestNullAthleteId:
    def test_two_null_ids_stay_distinct(self, db_session):
        cache: dict = {}
        a = _get_or_create_athlete(
            db_session, None, "John", "Doe", "USA", Gender.M, cache
        )
        b = _get_or_create_athlete(
            db_session, None, "Jane", "Smith", "CAN", Gender.F, cache
        )
        assert a.id != b.id
        rows = db_session.execute(select(Athlete)).scalars().all()
        assert len(rows) == 2
        # Nothing is cached under an unusable key.
        assert cache == {}

    def test_non_int_id_bypasses_cache(self, db_session):
        cache: dict = {}
        a = _get_or_create_athlete(
            db_session, "not-an-int", "A", "B", "GER", Gender.M, cache
        )
        assert a.id is not None
        assert cache == {}

    def test_valid_id_is_cached_and_reused(self, db_session):
        cache: dict = {}
        a = _get_or_create_athlete(db_session, 42, "Foo", "Bar", "GER", Gender.M, cache)
        assert cache[42] is a
        a2 = _get_or_create_athlete(
            db_session, 42, "Foo", "Bar", "GER", Gender.M, cache
        )
        assert a2 is a


# ---------------------------------------------------------------------------
# #202 — photo_url allowlist
# ---------------------------------------------------------------------------


class TestPhotoUrlValidation:
    def test_valid_https_ifsc(self):
        assert _is_valid_photo_url("https://ifsc.results.info/media/a.jpg")

    def test_http_rejected(self):
        assert not _is_valid_photo_url("http://ifsc.results.info/media/a.jpg")

    def test_offhost_rejected(self):
        assert not _is_valid_photo_url("https://evil.example.com/a.jpg")

    def test_javascript_uri_rejected(self):
        assert not _is_valid_photo_url("javascript:alert(1)")

    def test_too_long_rejected(self):
        assert not _is_valid_photo_url("https://ifsc.results.info/" + "a" * 600)

    def test_scrape_profile_drops_invalid_keeps_valid(self, monkeypatch):
        import climbing_elo.scraper.ifsc_api as mod

        monkeypatch.setattr(
            mod,
            "_api_get",
            lambda c, p: {"photo_url": "https://evil.example.com/x.jpg"},
        )
        out = scrape_athlete_profile(5, client=MagicMock())
        assert "photo_url" not in out

        monkeypatch.setattr(
            mod,
            "_api_get",
            lambda c, p: {"photo_url": "https://ifsc.results.info/x.jpg"},
        )
        out = scrape_athlete_profile(5, client=MagicMock())
        assert out["photo_url"] == "https://ifsc.results.info/x.jpg"


# ---------------------------------------------------------------------------
# #203 — response-size cap
# ---------------------------------------------------------------------------


class TestResponseCap:
    def test_over_cap_aborts(self):
        resp = _FakeStream(chunks=[b"x" * 100, b"y" * 100])
        with pytest.raises(ResponseTooLargeError):
            read_capped_bytes(resp, max_bytes=150)

    def test_under_cap_returns_body(self):
        resp = _FakeStream(chunks=[b"hel", b"lo"])
        assert read_capped_bytes(resp, max_bytes=1000) == b"hello"

    def test_text_decode_utf8(self):
        resp = _FakeStream(chunks=["café".encode()])
        assert read_capped_text(resp) == "café"
