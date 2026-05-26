from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    PROVISIONAL_THRESHOLD,
    AthleteRating,
    AthleteResult,
    calculate_round_updates,
)
from climbing_elo.models import (
    Discipline,
    Event,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)

log = logging.getLogger(__name__)

ROUND_ORDER = {
    RoundType.QUALIFICATION: 0,
    RoundType.SEMI: 1,
    RoundType.FINAL: 2,
}


@dataclass
class BackfillReport:
    events_processed: int = 0
    rounds_processed: int = 0
    athletes_rated: set[int] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


def _load_current_ratings(session: Session, discipline: Discipline) -> dict[int, AthleteRating]:
    stmt = select(Rating).where(Rating.discipline == discipline)
    ratings = {}
    for row in session.execute(stmt).scalars():
        ratings[row.athlete_id] = AthleteRating(
            athlete_id=row.athlete_id,
            mu=row.mu,
            sigma=row.sigma,
            n_events=row.n_events,
            last_event_at=row.last_event_at,
            provisional=row.provisional,
        )
    return ratings


def _get_or_create_rating(
    session: Session,
    athlete_id: int,
    discipline: Discipline,
    ratings_cache: dict[int, AthleteRating],
) -> AthleteRating:
    if athlete_id in ratings_cache:
        return ratings_cache[athlete_id]

    rating = AthleteRating(athlete_id=athlete_id)
    ratings_cache[athlete_id] = rating

    db_rating = Rating(
        athlete_id=athlete_id,
        discipline=discipline,
        mu=DEFAULT_MU,
        sigma=DEFAULT_SIGMA,
        n_events=0,
        provisional=True,
    )
    session.add(db_rating)
    return rating


def run_backfill(
    session: Session,
    discipline: Discipline = Discipline.LEAD,
    from_date: date | None = None,
) -> BackfillReport:
    report = BackfillReport()

    stmt = (
        select(Event)
        .where(Event.discipline == discipline)
        .order_by(Event.start_date.asc())
    )
    if from_date:
        stmt = stmt.where(Event.start_date >= from_date)

    events = list(session.execute(stmt).scalars())
    ratings_cache = _load_current_ratings(session, discipline)

    for event in events:
        rounds = sorted(event.rounds, key=lambda r: ROUND_ORDER.get(r.round_type, 0))
        event_had_updates = False

        for rnd in rounds:
            results = list(
                session.execute(
                    select(Result).where(Result.round_id == rnd.id)
                ).scalars()
            )
            if not results:
                continue

            athlete_results = []
            for res in results:
                _get_or_create_rating(session, res.athlete_id, discipline, ratings_cache)
                athlete_results.append(AthleteResult(
                    athlete_id=res.athlete_id,
                    rank=res.rank or 999,
                    score_normalized=res.score_normalized,
                    dnf=res.dnf,
                    dns=res.dns,
                ))

            try:
                updates = calculate_round_updates(
                    athlete_results,
                    ratings_cache,
                    event.tier,
                    rnd.round_type,
                    event.start_date,
                )
            except Exception as e:
                msg = f"Error processing round {rnd.id} of event {event.id}: {e}"
                log.error(msg)
                report.errors.append(msg)
                continue

            for upd in updates:
                ar = ratings_cache[upd.athlete_id]
                ar.mu = upd.mu_after
                ar.sigma = upd.sigma_after

                db_rating = session.execute(
                    select(Rating).where(
                        Rating.athlete_id == upd.athlete_id,
                        Rating.discipline == discipline,
                    )
                ).scalar_one()
                db_rating.mu = upd.mu_after
                db_rating.sigma = upd.sigma_after

                pairs_json = [
                    {
                        "opponent_id": p.opponent_id,
                        "result": p.result,
                        "expected": p.expected,
                        "actual": p.actual,
                        "delta": p.delta,
                        "margin_multiplier": p.margin_multiplier,
                    }
                    for p in upd.contributing_pairs
                ]
                session.add(RatingHistory(
                    athlete_id=upd.athlete_id,
                    event_id=event.id,
                    round_id=rnd.id,
                    mu_before=upd.mu_before,
                    mu_after=upd.mu_after,
                    sigma_before=upd.sigma_before,
                    sigma_after=upd.sigma_after,
                    contributing_pairs=pairs_json,
                ))

                report.athletes_rated.add(upd.athlete_id)

            report.rounds_processed += 1
            event_had_updates = True

        if event_had_updates:
            seen_athletes: set[int] = set()
            for rnd in rounds:
                for res in rnd.results:
                    if res.dns or res.athlete_id in seen_athletes:
                        continue
                    seen_athletes.add(res.athlete_id)

            for aid in seen_athletes:
                ar = ratings_cache.get(aid)
                if ar:
                    ar.n_events += 1
                    ar.last_event_at = event.start_date
                    ar.provisional = ar.n_events < PROVISIONAL_THRESHOLD

                    db_rating = session.execute(
                        select(Rating).where(
                            Rating.athlete_id == aid,
                            Rating.discipline == discipline,
                        )
                    ).scalar_one()
                    db_rating.n_events = ar.n_events
                    db_rating.last_event_at = ar.last_event_at
                    db_rating.provisional = ar.provisional

            session.commit()
            report.events_processed += 1
            log.info(
                "Processed event %s (%s) — %d rounds",
                event.name, event.start_date, len(rounds),
            )

    return report
