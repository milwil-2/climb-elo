"""Tests for the external-ranking scraper + cache (Issue #44 + AscentStats).

Network-free by construction: live ``fetch_*`` functions are exercised
against synthetic HTML strings, not real HTTP. The cache + fixture-fallback
helpers are exercised against ``tmp_path`` cache dirs.

If you want to validate the live fetch path against the real upstream,
``uv run pytest tests/test_external_rankings.py -m network`` — the network
marker is intentionally off by default so CI doesn't depend on Wikipedia
or ascentstats.com uptime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from climbing_elo.scraper.external_rankings import (
    FIXTURE_DIR,
    RankedAthlete,
    _parse_ascentstats_table,
    _parse_wikipedia_standings,
    fetch_ascentstats_ranking,
    fetch_ifsc_official_ranking,
    load_snapshot,
    normalize_name,
    refresh_ascentstats,
    refresh_ifsc_official,
)


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


def test_normalize_name_strips_diacritics_and_lowercases():
    assert normalize_name("Janja Garnbret") == "janja garnbret"
    assert normalize_name("Anže Peharc") == "anze peharc"
    assert normalize_name("  EXTRA   spaces  ") == "extra spaces"


def test_normalize_name_applies_overrides():
    # The override table maps the Wikipedia spelling to the IFSC spelling.
    assert normalize_name("Oceana Mackenzie") == "oceania mackenzie"
    assert normalize_name("Anastasia Sanders") == "annie sanders"
    # The canonical form is idempotent.
    assert normalize_name("oceania mackenzie") == "oceania mackenzie"


def test_normalize_name_is_idempotent():
    for name in ("Janja Garnbret", "Anže Peharc", "Sorato Anraku"):
        once = normalize_name(name)
        twice = normalize_name(once)
        assert once == twice


# ---------------------------------------------------------------------------
# load_snapshot — reads fixtures, falls back cleanly on miss
# ---------------------------------------------------------------------------


def test_load_snapshot_returns_recorded_fixture():
    ranking = load_snapshot("ifsc_official", 2024, "boulder", "M")
    assert ranking, "expected fixture data for ifsc_official 2024 boulder M"
    assert ranking[0].rank == 1
    assert ranking[0].name == "Sorato Anraku"
    assert ranking[0].country == "JPN"


def test_load_snapshot_missing_returns_empty(tmp_path):
    # Point at empty dirs so we exercise the miss-case.
    empty_cache = tmp_path / "cache"
    empty_fixture = tmp_path / "fixture"
    result = load_snapshot(
        "ifsc_official",
        1999,  # year that won't be in fixtures
        "boulder",
        "M",
        cache_dir=empty_cache,
        fixture_dir=empty_fixture,
    )
    assert result == []


def test_load_snapshot_prefers_cache_over_fixture(tmp_path):
    cache_dir = tmp_path / "cache"
    fixture_dir = tmp_path / "fixture"
    # Cache has rank-1 = "Cache Winner".
    cache_path = cache_dir / "ifsc_official" / "2030-boulder-M.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "source": "ifsc_official",
                "season": 2030,
                "discipline": "boulder",
                "gender": "M",
                "fetched_at": "2030-01-01",
                "ranking": [
                    {
                        "rank": 1,
                        "name": "Cache Winner",
                        "country": "USA",
                        "rating": 9999.0,
                    }
                ],
            }
        )
    )
    # Fixture has rank-1 = "Fixture Winner".
    fixture_path = fixture_dir / "ifsc_official" / "2030-boulder-M.json"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text(
        json.dumps(
            {
                "source": "ifsc_official",
                "season": 2030,
                "discipline": "boulder",
                "gender": "M",
                "fetched_at": "2030-01-01",
                "ranking": [
                    {
                        "rank": 1,
                        "name": "Fixture Winner",
                        "country": "USA",
                        "rating": 0.0,
                    }
                ],
            }
        )
    )

    result = load_snapshot(
        "ifsc_official",
        2030,
        "boulder",
        "M",
        cache_dir=cache_dir,
        fixture_dir=fixture_dir,
    )
    assert result[0].name == "Cache Winner"


# ---------------------------------------------------------------------------
# AscentStats HTML parser
# ---------------------------------------------------------------------------


_ASCENTSTATS_SAMPLE_HTML = """
<html><body>
<h1>2026 Men's Boulder Rankings</h1>
<table>
<thead><tr><th>Rank</th><th>Athlete</th><th>Country</th><th>Rating</th></tr></thead>
<tbody>
<tr><td>1</td><td>Sorato Anraku</td><td>JPN</td><td>9.3306</td></tr>
<tr><td>2</td><td>Mejdi Schalck</td><td>FRA</td><td>7.6401</td></tr>
<tr><td>3</td><td>Dohyun Lee</td><td>KOR</td><td>6.9984</td></tr>
</tbody>
</table>
</body></html>
"""


def test_parse_ascentstats_table_extracts_ranking_rows():
    parsed = _parse_ascentstats_table(_ASCENTSTATS_SAMPLE_HTML)
    assert len(parsed) == 3
    assert parsed[0] == RankedAthlete(
        rank=1, name="Sorato Anraku", country="JPN", rating=9.3306
    )
    assert parsed[2].rank == 3
    assert parsed[2].name == "Dohyun Lee"


def test_parse_ascentstats_table_returns_empty_on_garbage():
    assert _parse_ascentstats_table("<html><body>no table here</body></html>") == []


# ---------------------------------------------------------------------------
# Wikipedia (IFSC) HTML parser
# ---------------------------------------------------------------------------


_WIKIPEDIA_SAMPLE_HTML = """
<html><body>
<h2><span id="Men's_overall">Men's overall</span> standings</h2>
<table class="wikitable">
<tbody>
<tr><th>Rank</th><th>Name</th><th>Country</th><th>Points</th></tr>
<tr><td>1</td><td><a href="/wiki/Sorato_Anraku">Sorato Anraku</a></td><td>JPN</td><td>3365</td></tr>
<tr><td>2</td><td><a href="/wiki/Meichi_Narasaki">Meichi Narasaki</a></td><td>JPN</td><td>2860</td></tr>
</tbody>
</table>
</body></html>
"""


def test_parse_wikipedia_standings_finds_section_and_rows():
    parsed = _parse_wikipedia_standings(_WIKIPEDIA_SAMPLE_HTML, "M")
    assert len(parsed) == 2
    assert parsed[0].rank == 1
    assert parsed[0].name == "Sorato Anraku"
    assert parsed[0].rating == 3365.0


def test_parse_wikipedia_standings_returns_empty_when_section_missing():
    # Women's section requested, only Men's present.
    parsed = _parse_wikipedia_standings(_WIKIPEDIA_SAMPLE_HTML, "F")
    assert parsed == []


# ---------------------------------------------------------------------------
# Cache write/round-trip via refresh_* (mocked HTTP)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Tiny stand-in for httpx.Client used to exercise refresh_* without network."""

    def __init__(self, response_text: str, status: int = 200):
        self._text = response_text
        self._status = status

    def get(self, url, headers=None):  # noqa: D401 - mirror the httpx surface
        class _Resp:
            def __init__(self, text: str, status: int):
                self.status_code = status
                self.text = text

        return _Resp(self._text, self._status)

    def close(self) -> None:
        pass


def test_refresh_ascentstats_writes_cache_file(monkeypatch, tmp_path):
    fake = _FakeClient(_ASCENTSTATS_SAMPLE_HTML)
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fake)
    out = refresh_ascentstats(2026, "M", cache_dir=tmp_path)
    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["source"] == "ascentstats"
    assert payload["ranking"][0]["name"] == "Sorato Anraku"


def test_refresh_ifsc_official_writes_cache_file(monkeypatch, tmp_path):
    fake = _FakeClient(_WIKIPEDIA_SAMPLE_HTML)
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fake)
    out = refresh_ifsc_official(2024, "boulder", "M", cache_dir=tmp_path)
    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["source"] == "ifsc_official"
    assert payload["ranking"][0]["name"] == "Sorato Anraku"


def test_refresh_returns_none_on_empty_payload(monkeypatch, tmp_path):
    fake = _FakeClient("<html><body>no useful table</body></html>")
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fake)
    assert refresh_ascentstats(2026, "M", cache_dir=tmp_path) is None


def test_refresh_returns_none_on_http_error(monkeypatch, tmp_path):
    fake = _FakeClient("", status=503)
    monkeypatch.setattr("httpx.Client", lambda *a, **k: fake)
    assert refresh_ifsc_official(2024, "boulder", "M", cache_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# Fixture corpus sanity — every shipped JSON parses + has the right shape
# ---------------------------------------------------------------------------


def test_every_shipped_fixture_is_valid_json_with_required_keys():
    """The recorded fixtures must stay loadable as the cache schema evolves.

    If you add a new key, update load_snapshot too — this test will fail
    until both sides agree.
    """
    for source_dir in FIXTURE_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        for path in source_dir.glob("*.json"):
            payload = json.loads(path.read_text())
            for key in ("source", "season", "discipline", "gender", "ranking"):
                assert key in payload, f"{path}: missing {key}"
            assert payload["source"] == source_dir.name
            assert isinstance(payload["ranking"], list) and payload["ranking"]
            for row in payload["ranking"]:
                assert "rank" in row and "name" in row, (
                    f"{path}: ranking row missing rank/name"
                )


# ---------------------------------------------------------------------------
# Live-network smoke tests (skipped by default — opt-in via -m network)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_fetch_ascentstats_men_2026():
    ranking = fetch_ascentstats_ranking(2026, "M")
    assert ranking, "live AscentStats fetch returned empty"
    assert ranking[0].rank == 1


@pytest.mark.network
def test_live_fetch_ifsc_official_2024_boulder_men():
    ranking = fetch_ifsc_official_ranking(2024, "boulder", "M")
    assert ranking, "live IFSC fetch returned empty"
    assert ranking[0].rank == 1
