"""Load IFSC competition results from Kaggle CSV datasets into the database.

Supports multiple Kaggle dataset schemas — the loader auto-detects columns
and maps them to our models. Place CSV files in data/ and run
scripts/seed_from_kaggle.py.

Expected schemas (auto-detected by column names):

Schema A (mxmlnv/ifsc-competition-climbing):
  Columns vary but typically include:
  event, year, category, discipline, round, rank, name, country, score, ...

Schema B (brkurzawa/ifsc-sport-climbing-competition-results):
  Columns: Competition, Year, Gender, Discipline, Round, Rank, Name, Country, Score, ...

Schema C (gabrielenglert/ifsc-climbing-competition-data):
  Similar structure with slight column name variations.

The loader normalizes all schemas into our Athlete, Event, Round, Result models.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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


@dataclass
class LoadReport:
    events_created: int = 0
    athletes_created: int = 0
    results_created: int = 0
    rows_skipped: int = 0
    errors: list[str] = field(default_factory=list)


from dataclasses import field  # noqa: E402 (needed for LoadReport)


def _detect_columns(headers: list[str]) -> dict[str, str]:
    """Map normalized header names to our field names."""
    header_lower = [h.strip().lower().replace(" ", "_") for h in headers]
    mapping = {}

    patterns = {
        "event": ["event", "competition", "competition_name", "comp"],
        "year": ["year", "season"],
        "discipline": ["discipline", "disc"],
        "round": ["round", "round_name", "stage"],
        "rank": ["rank", "position", "place"],
        "name": ["name", "athlete", "athlete_name", "full_name", "firstname"],
        "country": ["country", "nationality", "nation", "country_code"],
        "gender": ["gender", "sex", "category_gender"],
        "score": ["score", "result", "points"],
        "category": ["category", "cat"],
        "date": ["date", "start_date", "event_date"],
    }

    for field_name, candidates in patterns.items():
        for candidate in candidates:
            if candidate in header_lower:
                mapping[field_name] = headers[header_lower.index(candidate)]
                break

    return mapping


def _parse_gender(raw: str) -> Gender | None:
    raw = raw.strip().upper()
    if raw in ("M", "MALE", "MEN"):
        return Gender.M
    if raw in ("F", "FEMALE", "WOMEN", "W"):
        return Gender.F
    return None


def _parse_round_type(raw: str) -> RoundType | None:
    raw = raw.strip().lower()
    if "final" in raw:
        return RoundType.FINAL
    if "semi" in raw:
        return RoundType.SEMI
    if "qualif" in raw or "qual" in raw:
        return RoundType.QUALIFICATION
    return None


def _classify_event_tier(event_name: str) -> EventTier:
    name_lower = event_name.lower()
    if "olympic" in name_lower:
        return EventTier.OLYMPICS
    if "world championship" in name_lower or "wch" in name_lower:
        return EventTier.WORLD_CHAMPIONSHIP
    if "continental" in name_lower or "european championship" in name_lower or "asian championship" in name_lower:
        return EventTier.CONTINENTAL
    return EventTier.WORLD_CUP


def _normalize_lead_score(raw: str | None) -> tuple[str, float | None]:
    """Parse Lead score strings like '34+', 'TOP', '28' into normalized floats.

    Returns (raw_string, normalized_float).
    """
    if not raw:
        return ("", None)

    raw = raw.strip()
    if not raw or raw == "-":
        return (raw, None)

    if raw.upper() == "TOP":
        return (raw, 999.0)

    match = re.match(r"^(\d+)\+?$", raw)
    if match:
        base = int(match.group(1))
        has_plus = raw.endswith("+")
        return (raw, base + 0.5 if has_plus else float(base))

    try:
        return (raw, float(raw))
    except ValueError:
        return (raw, None)


def _get_or_create_athlete(
    session: Session,
    name: str,
    gender: Gender,
    country: str | None,
    cache: dict[tuple[str, Gender], Athlete],
) -> Athlete:
    key = (name.strip(), gender)
    if key in cache:
        return cache[key]

    existing = session.execute(
        select(Athlete).where(Athlete.name == key[0], Athlete.gender == gender)
    ).scalar_one_or_none()

    if existing:
        cache[key] = existing
        return existing

    athlete = Athlete(name=key[0], gender=gender, nationality=country)
    session.add(athlete)
    session.flush()
    cache[key] = athlete
    return athlete


def _get_or_create_event(
    session: Session,
    name: str,
    year: int,
    tier: EventTier,
    cache: dict[str, Event],
) -> Event:
    key = f"{name}_{year}"
    if key in cache:
        return cache[key]

    existing = session.execute(
        select(Event).where(Event.name == name, Event.season == year)
    ).scalar_one_or_none()

    if existing:
        cache[key] = existing
        return existing

    event = Event(
        name=name,
        tier=tier,
        season=year,
        start_date=date(year, 1, 1),
        discipline=Discipline.LEAD,
    )
    session.add(event)
    session.flush()
    cache[key] = event
    return event


def _get_or_create_round(
    session: Session,
    event: Event,
    round_type: RoundType,
    gender: Gender,
    cache: dict[str, Round],
) -> Round:
    key = f"{event.id}_{round_type.value}_{gender.value}"
    if key in cache:
        return cache[key]

    existing = session.execute(
        select(Round).where(
            Round.event_id == event.id,
            Round.round_type == round_type,
            Round.gender == gender,
        )
    ).scalar_one_or_none()

    if existing:
        cache[key] = existing
        return existing

    rnd = Round(event_id=event.id, round_type=round_type, gender=gender)
    session.add(rnd)
    session.flush()
    cache[key] = rnd
    return rnd


def load_csv(session: Session, csv_path: Path) -> LoadReport:
    """Load a single CSV file of IFSC results into the database.

    Filters to Lead discipline only. Skips rows that can't be parsed.
    """
    report = LoadReport()
    athlete_cache: dict[tuple[str, Gender], Athlete] = {}
    event_cache: dict[str, Event] = {}
    round_cache: dict[str, Round] = {}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            log.error("CSV has no headers: %s", csv_path)
            return report

        col_map = _detect_columns(list(reader.fieldnames))
        log.info("Detected columns: %s", col_map)

        required = {"event", "year", "rank", "name"}
        missing = required - set(col_map.keys())
        if missing:
            log.error("Missing required columns %s in %s", missing, csv_path)
            return report

        for row_num, row in enumerate(reader, start=2):
            try:
                disc_raw = row.get(col_map.get("discipline", ""), "").strip().lower()
                if disc_raw and "lead" not in disc_raw:
                    continue

                gender_raw = row.get(col_map.get("gender", ""), "")
                gender = _parse_gender(gender_raw)
                if gender is None:
                    cat_raw = row.get(col_map.get("category", ""), "").lower()
                    if "women" in cat_raw or "female" in cat_raw:
                        gender = Gender.F
                    elif "men" in cat_raw or "male" in cat_raw:
                        gender = Gender.M
                    else:
                        report.rows_skipped += 1
                        continue

                round_raw = row.get(col_map.get("round", ""), "")
                round_type = _parse_round_type(round_raw)
                if round_type is None:
                    round_type = RoundType.FINAL

                rank_raw = row.get(col_map.get("rank", ""), "").strip()
                if not rank_raw or not rank_raw.isdigit():
                    report.rows_skipped += 1
                    continue
                rank = int(rank_raw)

                event_name = row[col_map["event"]].strip()
                year = int(row[col_map["year"]].strip())
                athlete_name = row[col_map["name"]].strip()
                country = row.get(col_map.get("country", ""), "").strip()[:3] or None

                score_raw = row.get(col_map.get("score", ""), "")
                raw_score, score_normalized = _normalize_lead_score(score_raw)

                tier = _classify_event_tier(event_name)
                event = _get_or_create_event(session, event_name, year, tier, event_cache)
                if event.id not in {e.id for e in event_cache.values() if e.season < year}:
                    report.events_created += 1

                athlete = _get_or_create_athlete(
                    session, athlete_name, gender, country, athlete_cache
                )
                rnd = _get_or_create_round(session, event, round_type, gender, round_cache)

                existing_result = session.execute(
                    select(Result).where(
                        Result.round_id == rnd.id,
                        Result.athlete_id == athlete.id,
                    )
                ).scalar_one_or_none()

                if existing_result:
                    continue

                result = Result(
                    round_id=rnd.id,
                    athlete_id=athlete.id,
                    rank=rank,
                    raw_score=raw_score,
                    score_normalized=score_normalized,
                    dnf=False,
                    dns=False,
                )
                session.add(result)
                report.results_created += 1

            except Exception as e:
                msg = f"Row {row_num}: {e}"
                log.warning(msg)
                report.errors.append(msg)
                report.rows_skipped += 1

        session.commit()

    rnd_count = len(round_cache)
    for rnd_obj in round_cache.values():
        rnd_obj.athlete_count = session.execute(
            select(Result).where(Result.round_id == rnd_obj.id)
        ).all().__len__()
    session.commit()

    report.athletes_created = len(athlete_cache)
    report.events_created = len(event_cache)

    log.info(
        "Loaded %s: %d events, %d athletes, %d results, %d skipped, %d errors",
        csv_path.name,
        report.events_created,
        report.athletes_created,
        report.results_created,
        report.rows_skipped,
        len(report.errors),
    )
    return report
