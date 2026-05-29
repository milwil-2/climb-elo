from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import (
    DEFAULT_CONFIG,
    DEFAULT_MU,
    DEFAULT_SIGMA,
    AthleteRating,
    AthleteResult,
    EloConfig,
    calculate_round_updates,
    compute_tournament_participation_bonus,
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


def _is_postgres(session: Session) -> bool:
    """Return True when the session is backed by a PostgreSQL engine."""
    try:
        return session.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


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


def _load_current_ratings(
    session: Session, discipline: Discipline
) -> dict[int, AthleteRating]:
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


def force_reset_for_discipline(session: Session, discipline: Discipline) -> int:
    """Wipe computed ratings for a discipline so backfill can recompute from scratch.

    Deletes every ``RatingHistory`` row attached to an event of this discipline
    and resets every ``Rating`` row of this discipline back to engine defaults
    (mu=1500, sigma=350, n_events=0, last_event_at=NULL, provisional=True).
    The raw competition data (``Athlete``, ``Event``, ``Round``, ``Result``) is
    untouched — only the computed-rating layer is wiped.

    Required after engine changes that need *existing* rows to be recomputed
    (K-factor regrid, σ formula bumps, etc.) — the backfill is otherwise
    idempotent on `(athlete_id, round_id, kind)` and silently skips rounds
    that already have history rows, so engine changes never propagate to
    historical ratings without an explicit reset.

    Returns the number of ``RatingHistory`` rows deleted. The caller is
    responsible for committing the surrounding transaction.

    **Per-discipline scope**: a reset of LEAD does not touch BOULDER or
    SPEED ratings. The combined BOULDER_LEAD aggregate (``Discipline.BOULDER_LEAD``)
    is its own discipline row in ``ratings`` and is left alone here — it
    will be naturally refreshed the next time ``compute_combined_ratings.py``
    runs.
    """
    event_id_rows = session.execute(
        select(Event.id).where(Event.discipline == discipline)
    ).all()
    event_ids = [row[0] for row in event_id_rows]

    rows_deleted = 0
    if event_ids:
        result = session.execute(
            delete(RatingHistory).where(RatingHistory.event_id.in_(event_ids))
        )
        rows_deleted = result.rowcount or 0

    session.execute(
        update(Rating)
        .where(Rating.discipline == discipline)
        .values(
            mu=DEFAULT_MU,
            sigma=DEFAULT_SIGMA,
            n_events=0,
            last_event_at=None,
            provisional=True,
        )
    )
    return rows_deleted


def run_backfill(
    session: Session,
    discipline: Discipline = Discipline.LEAD,
    from_date: date | None = None,
    end_date: date | None = None,
    config: EloConfig = DEFAULT_CONFIG,
) -> BackfillReport:
    """Run ELO backfill for all events in the given date range.

    Args:
        session: SQLAlchemy session.
        discipline: Discipline to process.
        from_date: If set, only process events on or after this date.
        end_date: If set, only process events strictly before this date.
        config: ELO engine configuration. Defaults to ``DEFAULT_CONFIG``;
            pass a custom :class:`EloConfig` to run alternative K-factor /
            MOV / Glicko-2 parameters (e.g. for #80 regrid sweeps) without
            monkey-patching module globals.
    """
    report = BackfillReport()

    stmt = (
        select(Event)
        .where(Event.discipline == discipline)
        .order_by(Event.start_date.asc())
    )
    if from_date:
        stmt = stmt.where(Event.start_date >= from_date)
    if end_date:
        stmt = stmt.where(Event.start_date < end_date)

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

            # Idempotency guard: skip rounds that have already been rated.
            # On Postgres the ON CONFLICT DO NOTHING handles duplicates; on
            # SQLite (tests) we detect them here to avoid IntegrityError.
            # We only check ``kind='pair'`` rows — TPB rows (one per FINAL
            # round) are layered on after all rounds are processed; they
            # must not block a partially-rated event from being completed.
            already_rated = session.execute(
                select(RatingHistory)
                .where(
                    RatingHistory.round_id == rnd.id,
                    RatingHistory.kind == "pair",
                )
                .limit(1)
            ).scalar_one_or_none()
            if already_rated is not None:
                log.debug("Round %d already rated, skipping", rnd.id)
                continue

            athlete_results = []
            for res in results:
                _get_or_create_rating(
                    session, res.athlete_id, discipline, ratings_cache
                )
                athlete_results.append(
                    AthleteResult(
                        athlete_id=res.athlete_id,
                        rank=res.rank or 999,
                        score_normalized=res.score_normalized,
                        dnf=res.dnf,
                        dns=res.dns,
                    )
                )

            try:
                updates = calculate_round_updates(
                    athlete_results,
                    ratings_cache,
                    event.tier,
                    rnd.round_type,
                    event.start_date,
                    discipline=discipline,
                    config=config,
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
                # Issue #81 — carry the refit Glicko-2 volatility forward in
                # the in-memory cache so it evolves across the backfill run.
                # No DB column; a fresh backfill re-seeds deterministically.
                ar.volatility = upd.volatility_after
                # Issue #89 Fix 3 — set last_event_at to this event's date so
                # subsequent rounds of the SAME event see zero inactivity gap
                # and don't re-inflate σ. Without this, the round 2 / round 3
                # of a multi-round event computes φ_inflated from the prior
                # event's date, clamps to σ_ceiling, and wipes out the per-
                # round σ shrinkage that round 1 just earned. The DB column
                # is only updated once per event (see the seen_athletes block
                # below) — that's intentional, the cache update here is the
                # in-event guard.
                ar.last_event_at = event.start_date

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
                if _is_postgres(session):
                    from sqlalchemy.dialects.postgresql import insert as pg_insert

                    stmt = (
                        pg_insert(RatingHistory)
                        .values(
                            athlete_id=upd.athlete_id,
                            event_id=event.id,
                            round_id=rnd.id,
                            mu_before=upd.mu_before,
                            mu_after=upd.mu_after,
                            sigma_before=upd.sigma_before,
                            sigma_after=upd.sigma_after,
                            contributing_pairs=pairs_json,
                            kind="pair",
                        )
                        .on_conflict_do_nothing(
                            index_elements=["athlete_id", "round_id", "kind"]
                        )
                    )
                    session.execute(stmt)
                else:
                    session.add(
                        RatingHistory(
                            athlete_id=upd.athlete_id,
                            event_id=event.id,
                            round_id=rnd.id,
                            mu_before=upd.mu_before,
                            mu_after=upd.mu_after,
                            sigma_before=upd.sigma_before,
                            sigma_after=upd.sigma_after,
                            contributing_pairs=pairs_json,
                            kind="pair",
                        )
                    )

                report.athletes_rated.add(upd.athlete_id)

            report.rounds_processed += 1
            event_had_updates = True

        if event_had_updates:
            _apply_tpb_for_event(
                session, event, rounds, discipline, ratings_cache, config
            )

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
                    ar.provisional = ar.n_events < config.provisional_threshold

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
                event.name,
                event.start_date,
                len(rounds),
            )

    return report


def _apply_tpb_for_event(
    session: Session,
    event: Event,
    rounds: list[Round],
    discipline: Discipline,
    ratings_cache: dict[int, AthleteRating],
    config: EloConfig,
) -> None:
    """Apply the Tournament Participation Bonus to one event (Issue #90).

    Pulls the FINAL round's results, computes a zero-sum tier-weighted bonus,
    writes synthetic ``RatingHistory(kind='tpb')`` rows pointed at the final
    round, and bumps each athlete's μ. No-op if no final round exists or if
    a tpb row is already present (idempotent).
    """
    final_round = next((r for r in rounds if r.round_type == RoundType.FINAL), None)
    if final_round is None:
        return

    already_applied = session.execute(
        select(RatingHistory)
        .where(
            RatingHistory.round_id == final_round.id,
            RatingHistory.kind == "tpb",
        )
        .limit(1)
    ).scalar_one_or_none()
    if already_applied is not None:
        return

    final_results = list(
        session.execute(
            select(Result).where(Result.round_id == final_round.id)
        ).scalars()
    )
    if not final_results:
        return

    athlete_results = [
        AthleteResult(
            athlete_id=res.athlete_id,
            rank=res.rank or 999,
            score_normalized=res.score_normalized,
            dnf=res.dnf,
            dns=res.dns,
        )
        for res in final_results
    ]

    contributions = compute_tournament_participation_bonus(
        athlete_results, event.tier, config
    )
    if not contributions:
        return

    for contrib in contributions:
        ar = ratings_cache.get(contrib.athlete_id)
        if ar is None:
            continue
        mu_before = ar.mu
        mu_after = mu_before + contrib.delta
        ar.mu = mu_after

        db_rating = session.execute(
            select(Rating).where(
                Rating.athlete_id == contrib.athlete_id,
                Rating.discipline == discipline,
            )
        ).scalar_one()
        db_rating.mu = mu_after

        tpb_payload = {
            "rank": contrib.rank,
            "gross_bonus": contrib.gross_bonus,
            "debit": contrib.debit,
            "tier": event.tier.value,
        }
        if _is_postgres(session):
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = (
                pg_insert(RatingHistory)
                .values(
                    athlete_id=contrib.athlete_id,
                    event_id=event.id,
                    round_id=final_round.id,
                    mu_before=mu_before,
                    mu_after=mu_after,
                    sigma_before=ar.sigma,
                    sigma_after=ar.sigma,
                    contributing_pairs=tpb_payload,
                    kind="tpb",
                )
                .on_conflict_do_nothing(
                    index_elements=["athlete_id", "round_id", "kind"]
                )
            )
            session.execute(stmt)
        else:
            session.add(
                RatingHistory(
                    athlete_id=contrib.athlete_id,
                    event_id=event.id,
                    round_id=final_round.id,
                    mu_before=mu_before,
                    mu_after=mu_after,
                    sigma_before=ar.sigma,
                    sigma_after=ar.sigma,
                    contributing_pairs=tpb_payload,
                    kind="tpb",
                )
            )
