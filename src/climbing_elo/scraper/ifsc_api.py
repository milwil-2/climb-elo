"""IFSC Results API client.

Scrapes competition results from the official IFSC/World Climbing results API
at ifsc.results.info. No authentication needed — just a referer header.

API structure:
  GET /api/v1/                                    → seasons list
  GET /api/v1/season_leagues/{league_id}           → events + d_cat IDs
  GET /api/v1/events/{event_id}/result/{d_cat_id}  → full results with rankings
"""
from __future__ import annotations

import logging
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


def _parse_speed_score(score_str: str | None) -> tuple[str, float | None, bool, bool]:
    """Parse a speed climbing time string.

    Returns (raw_str, seconds_float | None, dnf, dns).
    DNS/DNF/FS (false start) are treated as did-not-finish; score is None.
    """
    if not score_str:
        return ("", None, False, True)
    raw = score_str.strip()
    upper = raw.upper()
    if upper in ("DNS",):
        return (raw, None, False, True)
    if upper in ("DNF", "FS", "FALSE START", "DQ"):
        return (raw, None, True, False)
    try:
        return (raw, float(raw), False, False)
    except ValueError:
        return (raw, None, False, False)


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
    discipline: Discipline = Discipline.LEAD,
) -> ScrapeReport:
    """Scrape all results for one season's World Cup league for the given discipline."""
    report = ScrapeReport()
    athlete_cache: dict[int, Athlete] = {}
    season_name = season_info.get("name", "?")

    # Determine the discipline keyword to filter d_cats
    if discipline == Discipline.SPEED:
        disc_keyword = "speed"
    else:
        disc_keyword = "lead"

    league_data = _api_get(client, league_url)
    if not league_data:
        report.errors.append(f"Failed to fetch league {league_url}")
        return report

    d_cats = league_data.get("d_cats", [])
    target_dcats = {
        dc["id"]: ("M" if "Men" in dc["name"] else "F")
        for dc in d_cats
        if disc_keyword in dc.get("discipline", "").lower()
    }

    if not target_dcats:
        log.info("No %s categories in season %s", disc_keyword.capitalize(), season_name)
        return report

    events = league_data.get("events", [])
    for ev in events:
        event_id = ev.get("event_id")
        event_name = ev.get("event", "Unknown")

        event_dcats = []
        for dc in ev.get("d_cats", []):
            if dc.get("id") in target_dcats and dc.get("status") == "finished":
                event_dcats.append(dc)

        if not event_dcats:
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
                Event.discipline == discipline,
            )
        ).scalar_one_or_none()

        if existing_event:
            log.debug("Skipping existing event: %s (%s)", event_name, discipline.value)
            continue

        db_event = Event(
            name=event_name,
            tier=tier,
            country=None,
            season=int(season_name),
            start_date=event_date,
            discipline=discipline,
        )
        session.add(db_event)
        session.flush()

        rounds_seen: dict[str, Round] = {}

        for dc in event_dcats:
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

                    if discipline == Discipline.SPEED:
                        raw_str, normalized, dnf, dns = _parse_speed_score(score_raw)
                        if rank is None and not dnf:
                            dns = True
                    else:
                        ascents = rnd_data.get("ascents", [])
                        if ascents and round_type != RoundType.QUALIFICATION:
                            last_ascent = ascents[-1] if ascents else {}
                            ascent_score = last_ascent.get("score", score_raw)
                            raw_str, normalized = _parse_lead_score(ascent_score)
                        else:
                            raw_str, normalized = _parse_lead_score(score_raw)
                        dns = rank is None
                        dnf = False

                    existing_result = session.execute(
                        select(Result).where(
                            Result.round_id == db_round.id,
                            Result.athlete_id == athlete.id,
                        )
                    ).scalar_one_or_none()

                    if existing_result:
                        continue

                    result = Result(
                        round_id=db_round.id,
                        athlete_id=athlete.id,
                        rank=rank if rank else 999,
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
        log.info("Scraped %s (%s) [%s] — %d results", event_name, season_name, discipline.value, report.results_created)

    report.seasons_scraped = 1
    report.athletes_created = len(athlete_cache)
    return report


def scrape_all_seasons(
    session: Session,
    min_year: int = 2006,
    max_year: int = 2026,
    discipline: Discipline = Discipline.LEAD,
) -> ScrapeReport:
    """Scrape results for all seasons in the given year range for the given discipline."""
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

            log.info("Scraping season %s [%s]...", year_str, discipline.value)
            time.sleep(REQUEST_DELAY)

            report = scrape_season(
                client, session, season_info, wc_league["url"], wc_league.get("name", ""),
                discipline=discipline,
            )

            total_report.seasons_scraped += report.seasons_scraped
            total_report.events_scraped += report.events_scraped
            total_report.results_created += report.results_created
            total_report.athletes_created += report.athletes_created
            total_report.errors.extend(report.errors)

    return total_report
