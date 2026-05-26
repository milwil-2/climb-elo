"""External-ranking scrapers + on-disk cache (Issue #44 + AscentStats addition).

This module provides ranking snapshots from two external systems used as
*baselines* in the backtest harness:

  - **IFSC official season-end ranking** (the system we're trying to beat).
    Sourced from end-of-season Wikipedia summary tables of the IFSC World
    Cup overall standings, because the IFSC's own rankings widget
    (``components.ifsc-climbing.org/rankings/``) is served from a host that
    requires an interactive browser session and the legacy
    ``ifsc.results.info`` API has no rankings endpoint. The Wikipedia tables
    are the same rank-order canonically published by the IFSC each season —
    we just consume them through a stable, scrapeable surface.
  - **AscentStats** (``ascentstats.com``) — an independent Bayesian dynamic
    Bradley-Terry MCMC implementation on IFSC bouldering 2008–2026. Its
    season pages have a stable URL pattern (``/ranking-pages/ranking-{year}-
    {gender}.html``) and parse cleanly with a small HTML-table reader.

Public surface
--------------

Two free functions per source — one ``fetch_*`` that returns the parsed
rankings dict, and one ``load_*_snapshot`` that consults a local cache and
falls back to the fetch path on a miss. Engines in
:mod:`climbing_elo.engine.baselines` call ``load_*_snapshot`` only.

Cache layout
------------

``data/external_rankings/{source}/{year}-{discipline}-{gender}.json``

JSON shape::

    {
        "source": "ifsc_official" | "ascentstats",
        "season": 2024,
        "discipline": "boulder",
        "gender": "M",
        "fetched_at": "2026-05-26",
        "ranking": [
            {"rank": 1, "name": "Sorato Anraku", "country": "JPN", "rating": 3365.0},
            ...
        ]
    }

The ``data/external_rankings/`` directory is **gitignored**.  Recorded
fixtures live under ``tests/fixtures/external_rankings/`` and ship in-repo so
CI can run without network access.

Athlete-id resolution
---------------------

External rankings publish *names*, not IFSC numeric IDs.  The engines map a
name to our internal ``Athlete.id`` via a case-insensitive
``Athlete.name`` lookup, with a small set of known-difference normalizations
(diacritics, surname casing).  Unmatched names are skipped — they simply
won't get a rank-derived μ, and the harness will treat the athlete as
default-rated for that round.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Source = Literal["ifsc_official", "ascentstats"]
DisciplineKey = Literal["boulder", "lead", "speed"]
GenderKey = Literal["M", "F"]


@dataclass(frozen=True)
class RankedAthlete:
    """One row from an external ranking table.

    ``rank`` is 1-indexed.  ``rating`` is the score the source publishes
    (IFSC: World Cup points; AscentStats: bit-grade rating).  We keep it as
    a float so downstream engines can use it as an alternative monotone
    transform for μ if they wish, but the *default* mapping is rank-based so
    the two sources stay comparable.
    """

    rank: int
    name: str
    country: str | None
    rating: float | None = None


# ---------------------------------------------------------------------------
# Cache locations
# ---------------------------------------------------------------------------

#: Project root resolved relative to this file (``src/climbing_elo/scraper/...``).
_PKG_ROOT = Path(__file__).resolve().parents[3]

#: Writable cache populated by live fetches.  Gitignored.
DEFAULT_CACHE_DIR = _PKG_ROOT / "data" / "external_rankings"

#: Recorded snapshots shipped with the repo so tests + offline runs work.
#: Tests + the engines fall back to this dir when ``DEFAULT_CACHE_DIR`` has
#: no matching entry.
FIXTURE_DIR = _PKG_ROOT / "tests" / "fixtures" / "external_rankings"


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

#: Hand-curated overrides for known name-shape mismatches between external
#: sources and our DB.  Keyed by ``normalize_name(external_name)`` →
#: ``normalize_name(canonical_db_name)``.  Add entries here when a backtest
#: log shows unmatched names that should match.
_NAME_OVERRIDES: dict[str, str] = {
    # Wikipedia uses "Oceana"; IFSC + ascentstats use "Oceania".
    "oceana mackenzie": "oceania mackenzie",
    "anastasia sanders": "annie sanders",  # nickname used by AscentStats
    "chon jong-won": "jongwon chon",
    "lee dohyun": "dohyun lee",
    "seo chae-hyun": "chaehyun seo",
    "kim chaeyeong": "chaeyoung kim",
    "kibeom kwon": "kibeom kwon",
}


def normalize_name(name: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace.

    Used as the lookup key for athlete-name → DB id matching.  Idempotent:
    ``normalize_name(normalize_name(x)) == normalize_name(x)``.
    """
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()
    n = re.sub(r"\s+", " ", n)
    return _NAME_OVERRIDES.get(n, n)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    source: Source,
    season: int,
    discipline: DisciplineKey,
    gender: GenderKey,
    *,
    cache_dir: Path,
) -> Path:
    return cache_dir / source / f"{season}-{discipline}-{gender}.json"


def _read_cache(path: Path) -> list[RankedAthlete] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read cache %s: %s", path, exc)
        return None
    return [
        RankedAthlete(
            rank=int(r["rank"]),
            name=str(r["name"]),
            country=r.get("country"),
            rating=(float(r["rating"]) if r.get("rating") is not None else None),
        )
        for r in payload.get("ranking", [])
    ]


def _write_cache(
    path: Path,
    source: Source,
    season: int,
    discipline: DisciplineKey,
    gender: GenderKey,
    ranking: list[RankedAthlete],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "season": season,
        "discipline": discipline,
        "gender": gender,
        "fetched_at": date.today().isoformat(),
        "ranking": [
            {
                "rank": r.rank,
                "name": r.name,
                "country": r.country,
                "rating": r.rating,
            }
            for r in ranking
        ],
    }
    # Stable JSON (sorted keys, trailing newline) so diffs are minimal when
    # rankings are refreshed.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_snapshot(
    source: Source,
    season: int,
    discipline: DisciplineKey,
    gender: GenderKey,
    *,
    cache_dir: Path | None = None,
    fixture_dir: Path | None = None,
) -> list[RankedAthlete]:
    """Load a ranking snapshot from cache, falling back to bundled fixtures.

    Never hits the network — engines that want a live refresh call the
    ``fetch_*`` functions directly and pass the result through
    ``_write_cache``.  This keeps the backtest harness deterministic and
    offline-runnable.

    Returns ``[]`` if neither the cache nor the fixtures have an entry,
    rather than raising — the calling engine treats an empty ranking as
    "no athletes have a prior, fall back to default μ".
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    fixture_dir = fixture_dir or FIXTURE_DIR
    for path in (
        _cache_path(source, season, discipline, gender, cache_dir=cache_dir),
        _cache_path(source, season, discipline, gender, cache_dir=fixture_dir),
    ):
        entries = _read_cache(path)
        if entries:
            return entries
    log.info(
        "No snapshot found for %s/%s/%s/%s — engine will fall back to defaults.",
        source,
        season,
        discipline,
        gender,
    )
    return []


# ---------------------------------------------------------------------------
# IFSC official: end-of-season World Cup overall ranking
# ---------------------------------------------------------------------------

#: Wikipedia URL template for season-end IFSC World Cup standings.  These
#: pages mirror the IFSC's own published season-end ranking and are the
#: scrape surface we use given that the IFSC widget host
#: (``components.ifsc-climbing.org``) requires an interactive browser
#: session and ``ifsc.results.info`` has no rankings endpoint.
WIKIPEDIA_IFSC_URLS = {
    "boulder": "https://en.wikipedia.org/wiki/Bouldering_at_the_{season}_IFSC_Climbing_World_Cup",
    "lead": "https://en.wikipedia.org/wiki/Lead_climbing_at_the_{season}_IFSC_Climbing_World_Cup",
    "speed": "https://en.wikipedia.org/wiki/Speed_climbing_at_the_{season}_IFSC_Climbing_World_Cup",
}


def fetch_ifsc_official_ranking(
    season: int,
    discipline: DisciplineKey,
    gender: GenderKey,
    *,
    client: httpx.Client | None = None,
) -> list[RankedAthlete]:
    """Live-fetch the IFSC season-end ranking for one discipline/gender.

    Returns ``[]`` on network failure rather than raising so callers can
    fall back to a cached/fixture snapshot without try/except scaffolding.
    Intended to be called by a refresh script, not by the engines on the
    hot path.
    """
    url = WIKIPEDIA_IFSC_URLS.get(discipline)
    if not url:
        raise ValueError(f"Unsupported discipline {discipline!r}")
    full_url = url.format(season=season)

    close_after = False
    if client is None:
        client = httpx.Client(timeout=20.0, follow_redirects=True)
        close_after = True
    try:
        resp = client.get(
            full_url,
            headers={"User-Agent": "ClimbingELO/0.1 (research)"},
        )
        if resp.status_code != 200:
            log.warning("Wikipedia returned %d for %s", resp.status_code, full_url)
            return []
        return _parse_wikipedia_standings(resp.text, gender)
    except httpx.HTTPError as exc:
        log.error("Failed to fetch %s: %s", full_url, exc)
        return []
    finally:
        if close_after:
            client.close()


def _parse_wikipedia_standings(html: str, gender: GenderKey) -> list[RankedAthlete]:
    """Best-effort regex-driven parse of a Wikipedia World Cup standings page.

    Wikipedia tables are stable in shape but messy in HTML; we look for the
    "Men's overall" / "Women's overall" section header and then extract
    ``(rank, name, points)`` triples from the following wikitable.

    This parser is intentionally permissive — if anything goes wrong we
    return ``[]`` and the caller falls back to fixtures. The fixtures are
    the source of truth for offline / CI runs; live fetches are a
    convenience for the data-refresh script.
    """
    section = "Men's overall" if gender == "M" else "Women's overall"
    # Find a header that mentions the section, then take the next ~50 KB
    # of HTML and try to extract rows.
    header_pat = re.compile(
        rf"<h[2-4][^>]*>.*?{re.escape(section)}.*?</h[2-4]>",
        re.IGNORECASE | re.DOTALL,
    )
    m = header_pat.search(html)
    if not m:
        return []
    chunk = html[m.end() : m.end() + 80_000]
    table_match = re.search(
        r'<table[^>]*class="[^"]*wikitable[^"]*".*?</table>', chunk, re.DOTALL
    )
    if not table_match:
        return []
    table = table_match.group(0)

    out: list[RankedAthlete] = []
    rank_seen = 0
    # Very loose row pattern: rank cell, then a name link, then country flag/abbr, then points.
    row_pat = re.compile(
        r"<tr>(?P<row>.*?)</tr>",
        re.DOTALL,
    )
    cell_pat = re.compile(r"<t[dh][^>]*>(?P<cell>.*?)</t[dh]>", re.DOTALL)
    tag_strip = re.compile(r"<[^>]+>")
    for row_m in row_pat.finditer(table):
        cells = [
            re.sub(r"\s+", " ", tag_strip.sub("", c.group("cell"))).strip()
            for c in cell_pat.finditer(row_m.group("row"))
        ]
        if len(cells) < 3:
            continue
        # Look for an integer rank in cell 0 and a parseable points value
        # in the last few cells. If neither parses, skip.
        try:
            rank = int(cells[0])
        except ValueError:
            continue
        name_cell = cells[1] if len(cells) > 1 else ""
        country_cell = cells[2] if len(cells) > 2 else ""
        # Some Wikipedia variants put country before name — detect by
        # checking if cell 1 looks like a 3-letter code.
        if re.fullmatch(r"[A-Z]{3}", country_cell) is None and re.fullmatch(
            r"[A-Z]{3}", name_cell
        ):
            name_cell, country_cell = country_cell, name_cell
        # Try to parse a points value from any of the remaining cells.
        points: float | None = None
        for c in reversed(cells):
            try:
                points = float(c.replace(",", ""))
                break
            except ValueError:
                continue
        if not name_cell:
            continue
        rank_seen += 1
        out.append(
            RankedAthlete(
                rank=rank,
                name=name_cell,
                country=country_cell or None,
                rating=points,
            )
        )
        if rank_seen >= 50:
            break
    return out


def refresh_ifsc_official(
    season: int,
    discipline: DisciplineKey,
    gender: GenderKey,
    *,
    cache_dir: Path | None = None,
) -> Path | None:
    """Live-fetch and persist one IFSC ranking snapshot.

    Returns the cache path on success, ``None`` on failure.  Called by the
    operator refresh script — not by the engines or the harness.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    ranking = fetch_ifsc_official_ranking(season, discipline, gender)
    if not ranking:
        return None
    path = _cache_path("ifsc_official", season, discipline, gender, cache_dir=cache_dir)
    _write_cache(path, "ifsc_official", season, discipline, gender, ranking)
    return path


# ---------------------------------------------------------------------------
# AscentStats: independent Bayesian Bradley-Terry, Boulder only
# ---------------------------------------------------------------------------

ASCENTSTATS_URL = (
    "https://ascentstats.com/ranking-pages/ranking-{season}-{gender_slug}.html"
)


def fetch_ascentstats_ranking(
    season: int,
    gender: GenderKey,
    *,
    client: httpx.Client | None = None,
) -> list[RankedAthlete]:
    """Live-fetch one AscentStats year-snapshot.  Boulder only (their scope)."""
    gender_slug = "men" if gender == "M" else "women"
    url = ASCENTSTATS_URL.format(season=season, gender_slug=gender_slug)

    close_after = False
    if client is None:
        client = httpx.Client(timeout=20.0, follow_redirects=True)
        close_after = True
    try:
        resp = client.get(
            url,
            headers={"User-Agent": "ClimbingELO/0.1 (research)"},
        )
        if resp.status_code != 200:
            log.warning("AscentStats returned %d for %s", resp.status_code, url)
            return []
        return _parse_ascentstats_table(resp.text)
    except httpx.HTTPError as exc:
        log.error("Failed to fetch %s: %s", url, exc)
        return []
    finally:
        if close_after:
            client.close()


def _parse_ascentstats_table(html: str) -> list[RankedAthlete]:
    """Parse the single ranking table on an AscentStats year page.

    The page renders a static ``<table>`` with columns
    ``Rank | Athlete | Country | Rating``.  Robust against extra columns:
    we identify rank by integer parse and pluck the rating from the last
    parseable float cell.
    """
    table_match = re.search(r"<table[^>]*>.*?</table>", html, re.DOTALL | re.IGNORECASE)
    if not table_match:
        return []
    table = table_match.group(0)
    row_pat = re.compile(r"<tr>(?P<row>.*?)</tr>", re.DOTALL)
    cell_pat = re.compile(r"<t[dh][^>]*>(?P<cell>.*?)</t[dh]>", re.DOTALL)
    tag_strip = re.compile(r"<[^>]+>")
    out: list[RankedAthlete] = []
    for row_m in row_pat.finditer(table):
        cells = [
            re.sub(r"\s+", " ", tag_strip.sub("", c.group("cell"))).strip()
            for c in cell_pat.finditer(row_m.group("row"))
        ]
        if len(cells) < 3:
            continue
        try:
            rank = int(cells[0])
        except ValueError:
            continue
        name = cells[1]
        country = cells[2] if len(cells) > 2 else None
        rating: float | None = None
        for c in reversed(cells):
            try:
                rating = float(c)
                break
            except ValueError:
                continue
        if not name:
            continue
        out.append(RankedAthlete(rank=rank, name=name, country=country, rating=rating))
    return out


def refresh_ascentstats(
    season: int,
    gender: GenderKey,
    *,
    cache_dir: Path | None = None,
) -> Path | None:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    ranking = fetch_ascentstats_ranking(season, gender)
    if not ranking:
        return None
    path = _cache_path("ascentstats", season, "boulder", gender, cache_dir=cache_dir)
    _write_cache(path, "ascentstats", season, "boulder", gender, ranking)
    return path


# ---------------------------------------------------------------------------
# Convenience: bulk refresh across a year range
# ---------------------------------------------------------------------------


def refresh_all(
    seasons: Iterable[int],
    *,
    cache_dir: Path | None = None,
) -> dict[str, list[Path]]:
    """Refresh IFSC + AscentStats snapshots for every season in ``seasons``.

    Returns a per-source list of cache paths that were successfully written.
    Intended to be invoked by a maintainer script, not by tests.
    """
    written: dict[str, list[Path]] = {"ifsc_official": [], "ascentstats": []}
    for season in seasons:
        for discipline in ("boulder", "lead", "speed"):  # type: ignore[assignment]
            for gender in ("M", "F"):  # type: ignore[assignment]
                p = refresh_ifsc_official(
                    season, discipline, gender, cache_dir=cache_dir
                )
                if p:
                    written["ifsc_official"].append(p)
        for gender in ("M", "F"):  # type: ignore[assignment]
            p = refresh_ascentstats(season, gender, cache_dir=cache_dir)
            if p:
                written["ascentstats"].append(p)
    return written
