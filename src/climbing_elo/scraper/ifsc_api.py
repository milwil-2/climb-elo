"""IFSC Results API client.

Scrapes competition results from the official IFSC/World Climbing results API
at ifsc.results.info. No authentication needed — just a referer header.

API structure:
  GET /api/v1/                                    → seasons list
  GET /api/v1/season_leagues/{league_id}           → events + d_cat IDs
  GET /api/v1/events/{event_id}/result/{d_cat_id}  → full results with rankings

Boulder score formats across years:
  - 2012-2020: "4t5 5b12" (tops, top_attempts, bonuses, bonus_attempts)
                In this era, "b" = bonus/zone; "t" = top.
  - 2021-2024: "4T5z 6 7" (tops, zones, total_top_attempts, total_zone_attempts)
  - 2025+: Floating-point total points (e.g. "124.9"), but ascent-level
            structured data (top, zone, top_tries, zone_tries) is still present.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)

log = logging.getLogger(__name__)

BASE_URL = "https://ifsc.results.info"
HEADERS = {
    "User-Agent": "ClimbingELO/0.1 (research project)",
    "Referer": "https://ifsc.results.info/",
    "Accept": "application/json",
}
REQUEST_DELAY = 0.5


@dataclass
class ScrapeReport:
    seasons_scraped: int = 0
    events_scraped: int = 0
    results_created: int = 0
    athletes_created: int = 0
    errors: list[str] = field(default_factory=list)




def _api_get(client: httpx.Client, path: str) -> dict | list | None:
    url = f"{BASE_URL}{path}" if path.startswith("/") else path
    try:
        resp = client.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        log.warning("HTTP %d for %s", resp.status_code, url)
        return None
    except Exception as e:
        log.error("Request failed for %s: %s", url, e)
        return None


def _classify_tier(event_name: str, league_name: str) -> EventTier:
    name = event_name.lower()
    league = league_name.lower()
    if "olympic" in name or "games" in league:
        return EventTier.OLYMPICS
    if "world championship" in name:
        return EventTier.WORLD_CHAMPIONSHIP
    if "continental" in name or "european" in league or "asia" in league or "pan america" in league:
        return EventTier.CONTINENTAL
    return EventTier.WORLD_CUP


def _parse_round_type(name: str) -> RoundType:
    name = name.lower()
    if "final" in name:
        return RoundType.FINAL
    if "semi" in name:
        return RoundType.SEMI
    return RoundType.QUALIFICATION


def _parse_lead_score(score_str: str | None) -> tuple[str, float | None]:
    if not score_str:
        return ("", None)
    raw = score_str.strip()
    if raw.upper() == "TOP":
        return (raw, 999.0)
    if raw.endswith("+"):
        try:
            return (raw, float(raw[:-1]) + 0.5)
        except ValueError:
            return (raw, None)
    try:
        return (raw, float(raw))
    except ValueError:
        return (raw, None)


def _parse_boulder_score(
    score_str: str | None,
    ascents: list[dict] | None = None,
) -> tuple[str, float | None]:
    """Parse a Boulder round score into (raw_str, normalized_float).

    Normalized score = tops * 1000 + zones * 100 - top_attempts * 10 - zone_attempts

    This produces an ordinal ranking value where:
      - More tops is always better
      - More zones is better (secondary)
      - Fewer attempts is better (tertiary)

    Handles three historical score formats:
      - Pre-2021: "4t5 5b12"  (tops t top_attempts  bonuses b bonus_attempts)
      - 2021-2024: "4T5z 6 7" (tops T zones z total_top_attempts total_zone_attempts)
      - 2025+: floating-point total like "124.9" — derive from per-ascent data
    """
    raw = (score_str or "").strip()
    if not raw or raw.upper() in ("DNF", "DNS", "-", ""):
        return (raw, None)

    # If ascent-level structured data is available, compute directly from it.
    # This handles 2025+ format and is the most accurate for any year.
    if ascents:
        try:
            tops = sum(1 for a in ascents if a.get("top"))
            zones = sum(1 for a in ascents if a.get("zone"))
            top_attempts = sum(
                int(a.get("top_tries") or 0) for a in ascents if a.get("top")
            )
            zone_attempts = sum(
                int(a.get("zone_tries") or 0) for a in ascents if a.get("zone")
            )
            normalized = tops * 1000 + zones * 100 - top_attempts * 10 - zone_attempts
            return (raw, float(normalized))
        except (TypeError, ValueError):
            log.debug("Malformed ascent data, falling back to score string parse")

    # Fall back to parsing the score string (older data with no ascent details).

    # 2021-2024: "4T5z 6 7"
    m = re.match(
        r"(\d+)[Tt](\d+)[Zz]\s+(\d+)\s+(\d+)",
        raw,
    )
    if m:
        tops, zones, top_att, zone_att = (int(x) for x in m.groups())
        return (raw, float(tops * 1000 + zones * 100 - top_att * 10 - zone_att))

    # Pre-2021: "4t5 5b12"  (t=tops, b=bonus/zone)
    m = re.match(
        r"(\d+)[Tt](\d+)\s+(\d+)[Bb](\d+)",
        raw,
    )
    if m:
        tops, top_att, zones, zone_att = (int(x) for x in m.groups())
        return (raw, float(tops * 1000 + zones * 100 - top_att * 10 - zone_att))

    # 2025+ numeric-only total points — rank is already encoded in the rank field;
    # try to convert directly to a float for use as normalized score.
    try:
        return (raw, float(raw))
    except ValueError:
        pass

    log.debug("Could not parse boulder score: %r", raw)
    return (raw, None)


def _get_or_create_athlete(
    session: Session,
    athlete_id: int,
    firstname: str,
    lastname: str,
    country: str,
    gender: Gender,
    cache: dict[int, Athlete],
) -> Athlete:
    if athlete_id in cache:
        return cache[athlete_id]

    existing = session.execute(
        select(Athlete).where(Athlete.name == f"{firstname} {lastname}", Athlete.gender == gender)
    ).scalar_one_or_none()

    if existing:
        cache[athlete_id] = existing
        return existing

    athlete = Athlete(
        name=f"{firstname} {lastname}",
        gender=gender,
        nationality=country[:3] if country else None,
    )
    session.add(athlete)
    session.flush()
    cache[athlete_id] = athlete
    return athlete


def get_seasons(client: httpx.Client) -> list[dict]:
    data = _api_get(client, "/api/v1/")
    if not data:
        return []
    return data.get("seasons", [])


def scrape_season(
    client: httpx.Client,
    session: Session,
    season_info: dict,
    league_url: str,
    league_name: str,
    discipline: str = "lead",
) -> ScrapeReport:
    """Scrape results for one season's World Cup league.

    Args:
        discipline: The discipline to scrape — "lead" or "boulder" (case-insensitive).
                    Matched against the d_cat discipline field from the API.
    """
    report = ScrapeReport()
    athlete_cache: dict[int, Athlete] = {}
    season_name = season_info.get("name", "?")
    disc_lower = discipline.lower()

    if disc_lower not in ("lead", "boulder", "speed"):
        raise ValueError(
            f"Unsupported discipline {discipline!r}; expected one of: lead, boulder, speed"
        )

    if disc_lower == "boulder":
        db_discipline = Discipline.BOULDER
    elif disc_lower == "speed":
        db_discipline = Discipline.SPEED
    else:
        db_discipline = Discipline.LEAD

    league_data = _api_get(client, league_url)
    if not league_data:
        report.errors.append(f"Failed to fetch league {league_url}")
        return report

    d_cats = league_data.get("d_cats", [])

    # Filter d_cats to only the requested discipline.
    # For "boulder" we explicitly exclude "boulder&lead" combined categories.
    if disc_lower == "boulder":
        target_dcats = {
            dc["id"]: ("M" if "Men" in dc["name"] else "F")
            for dc in d_cats
            if "boulder" in dc.get("discipline", "").lower()
            and "lead" not in dc.get("discipline", "").lower()
        }
    else:
        target_dcats = {
            dc["id"]: ("M" if "Men" in dc["name"] else "F")
            for dc in d_cats
            if disc_lower in dc.get("discipline", "").lower()
            and "boulder" not in dc.get("discipline", "").lower()
        }

    if not target_dcats:
        log.info("No %s categories in season %s", discipline, season_name)
        return report

    events = league_data.get("events", [])
    for ev in events:
        event_id = ev.get("event_id")
        event_name = ev.get("event", "Unknown")

        event_target_dcats = []
        for dc in ev.get("d_cats", []):
            if dc.get("id") in target_dcats and dc.get("status") == "finished":
                event_target_dcats.append(dc)

        if not event_target_dcats:
            continue

        start_date_str = ev.get("local_start_date", f"{season_name}-01-01")
        try:
            event_date = date.fromisoformat(start_date_str)
        except ValueError:
            event_date = date(int(season_name), 1, 1)

        tier = _classify_tier(event_name, league_name)

        existing_event = session.execute(
            select(Event).where(
                Event.name == event_name,
                Event.season == int(season_name),
                Event.discipline == db_discipline,
            )
        ).scalar_one_or_none()

        if existing_event:
            log.debug("Skipping existing event: %s (%s)", event_name, discipline)
            continue

        db_event = Event(
            name=event_name,
            tier=tier,
            country=None,
            season=int(season_name),
            start_date=event_date,
            discipline=db_discipline,
        )
        session.add(db_event)
        session.flush()

        rounds_seen: dict[str, Round] = {}

        for dc in event_target_dcats:
            dcat_id = dc["id"]
            gender = Gender.M if target_dcats[dcat_id] == "M" else Gender.F

            time.sleep(REQUEST_DELAY)
            result_data = _api_get(client, f"/api/v1/events/{event_id}/result/{dcat_id}")
            if not result_data:
                report.errors.append(f"Failed to fetch results for event {event_id} dcat {dcat_id}")
                continue

            for athlete_entry in result_data.get("ranking", []):
                ifsc_athlete_id = athlete_entry.get("athlete_id")
                firstname = athlete_entry.get("firstname", "")
                lastname = athlete_entry.get("lastname", "")
                country = athlete_entry.get("country", "")

                athlete = _get_or_create_athlete(
                    session, ifsc_athlete_id, firstname, lastname, country, gender, athlete_cache
                )

                for rnd_data in athlete_entry.get("rounds", []):
                    round_name = rnd_data.get("round_name", "Unknown")
                    round_type = _parse_round_type(round_name)
                    round_key = f"{db_event.id}_{round_type.value}_{gender.value}"

                    if round_key not in rounds_seen:
                        db_round = Round(
                            event_id=db_event.id,
                            round_type=round_type,
                            gender=gender,
                        )
                        session.add(db_round)
                        session.flush()
                        rounds_seen[round_key] = db_round

                    db_round = rounds_seen[round_key]
                    rank = rnd_data.get("rank")
                    score_raw = rnd_data.get("score", "")
                    ascents = rnd_data.get("ascents", [])

                    if db_discipline == Discipline.BOULDER:
                        raw_str, normalized = _parse_boulder_score(score_raw, ascents or None)
                    else:
                        # Lead score parsing (original logic)
                        if ascents and round_type != RoundType.QUALIFICATION:
                            last_ascent = ascents[-1] if ascents else {}
                            ascent_score = last_ascent.get("score", score_raw)
                            raw_str, normalized = _parse_lead_score(ascent_score)
                        else:
                            raw_str, normalized = _parse_lead_score(score_raw)

                    existing_result = session.execute(
                        select(Result).where(
                            Result.round_id == db_round.id,
                            Result.athlete_id == athlete.id,
                        )
                    ).scalar_one_or_none()

                    if existing_result:
                        continue

                    dns = rank is None
                    dnf = False

                    try:
                        rank_int = int(rank) if rank is not None else 999
                    except (TypeError, ValueError):
                        log.warning("Non-integer rank %r for athlete %s; skipping", rank, athlete.id)
                        continue

                    result = Result(
                        round_id=db_round.id,
                        athlete_id=athlete.id,
                        rank=rank_int,
                        raw_score=raw_str,
                        score_normalized=normalized,
                        dnf=dnf,
                        dns=dns,
                    )
                    session.add(result)
                    report.results_created += 1

        for rnd in rounds_seen.values():
            count = session.execute(
                select(Result).where(Result.round_id == rnd.id)
            ).all()
            rnd.athlete_count = len(count)

        session.commit()
        report.events_scraped += 1
        log.info("Scraped %s (%s) [%s] — %d results", event_name, season_name, discipline, report.results_created)

    report.seasons_scraped = 1
    report.athletes_created = len(athlete_cache)
    return report


def scrape_all_seasons(
    session: Session,
    min_year: int = 2006,
    max_year: int = 2026,
    discipline: str = "lead",
) -> ScrapeReport:
    """Scrape results for all seasons in the given year range.

    Args:
        discipline: "lead" or "boulder" (case-insensitive).
    """
    total_report = ScrapeReport()

    with httpx.Client() as client:
        seasons = get_seasons(client)
        if not seasons:
            total_report.errors.append("Failed to fetch seasons index")
            return total_report

        for season_info in seasons:
            year_str = season_info.get("name", "")
            try:
                year = int(year_str)
            except ValueError:
                continue

            if year < min_year or year > max_year:
                continue

            wc_league = None
            for lg in season_info.get("leagues", []):
                if lg.get("league_id") == 1:
                    wc_league = lg
                    break

            if not wc_league:
                log.info("No WC league for season %s, skipping", year_str)
                continue

            log.info("Scraping season %s [%s]...", year_str, discipline)
            time.sleep(REQUEST_DELAY)

            report = scrape_season(
                client,
                session,
                season_info,
                wc_league["url"],
                wc_league.get("name", ""),
                discipline=discipline,
            )

            total_report.seasons_scraped += report.seasons_scraped
            total_report.events_scraped += report.events_scraped
            total_report.results_created += report.results_created
            total_report.athletes_created += report.athletes_created
            total_report.errors.extend(report.errors)

    return total_report
