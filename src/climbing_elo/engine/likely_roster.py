"""Likely-competitor roster fallback for upcoming events (Issue #33).

When the IFSC API has not yet published a registered-athletes list for an
upcoming event (typically 7-14 days before it starts), this module derives a
"likely roster" from season attendance patterns.

An athlete is considered a likely competitor in discipline X, season Y if they
have competed in >= ``threshold`` (default 60%) of the season's finished World
Cup events in that discipline, filtered by gender.

Fallback: if fewer than ``min_events_for_threshold`` events have finished (early
in the season), we return the top ``cap`` athletes by current μ for the
discipline (filtered by gender and requiring >= 3 career events).
"""
from __future__ import annotations

from sqlalchemy import distinct, func, select

from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    Result,
    Round,
    RoundType,
)


def likely_competitors(
    session,
    discipline: Discipline,
    season: int,
    gender: Gender,
    threshold: float = 0.6,
    cap: int = 64,
    min_events_for_threshold: int = 3,
) -> list[int]:
    """Return athlete_ids likely to compete in the next event of this discipline+season.

    Parameters
    ----------
    session:
        A SQLAlchemy Session bound to the climbing-elo database.
    discipline:
        The competition discipline (e.g. ``Discipline.LEAD``).
    season:
        The calendar year of the season (e.g. 2026).
    gender:
        Filter athlete participation by gender (``Gender.M`` or ``Gender.F``).
    threshold:
        Fraction of season events an athlete must have attended to be considered
        a likely competitor (default 0.6 = 60%).
    cap:
        Maximum number of athletes to return.  Prevents runaway Monte Carlo cost
        (mirrors ``_MAX_ATHLETES_PER_PROJECTION_CARD`` in routes.py).
    min_events_for_threshold:
        Minimum number of finished World Cup events before the attendance
        threshold is applied.  If fewer events have occurred, we fall back to
        top-``cap`` athletes by μ.

    Returns
    -------
    list[int]
        Athlete IDs ordered by current μ descending, capped at ``cap``.
        Returns an empty list if no athletes can be found.
    """
    # ------------------------------------------------------------------
    # Step 1: count distinct finished World Cup events for this
    #         discipline + season.  "Finished" means the event has at
    #         least one Round with Results stored (i.e. backfill has run).
    # ------------------------------------------------------------------
    # We use a subquery that finds event_ids that have ≥1 result row via
    # Round, which is a safe proxy for "backfill has processed this event".
    events_with_results_subq = (
        select(distinct(Event.id))
        .join(Round, Round.event_id == Event.id)
        .join(Result, Result.round_id == Round.id)
        .where(
            Event.discipline == discipline,
            Event.season == season,
            Event.tier == EventTier.WORLD_CUP,
            Round.gender == gender,
        )
        .scalar_subquery()
    )

    total_events: int = session.execute(
        select(func.count()).select_from(
            select(distinct(Event.id))
            .join(Round, Round.event_id == Event.id)
            .join(Result, Result.round_id == Round.id)
            .where(
                Event.discipline == discipline,
                Event.season == season,
                Event.tier == EventTier.WORLD_CUP,
                Round.gender == gender,
            )
            .subquery()
        )
    ).scalar_one()

    # ------------------------------------------------------------------
    # Step 2: if not enough events, fall back to top-cap-by-mu
    # ------------------------------------------------------------------
    if total_events < min_events_for_threshold:
        return _top_by_mu(session, discipline, gender, cap)

    # ------------------------------------------------------------------
    # Step 3: count per-athlete participations (distinct events, exclude DNS)
    # ------------------------------------------------------------------
    # An athlete "participated" in an event if they have ≥1 non-DNS Result
    # in a Round of that event.
    attendance_subq = (
        select(
            Result.athlete_id,
            func.count(distinct(Event.id)).label("attended"),
        )
        .join(Round, Result.round_id == Round.id)
        .join(Event, Round.event_id == Event.id)
        .where(
            Event.discipline == discipline,
            Event.season == season,
            Event.tier == EventTier.WORLD_CUP,
            Round.gender == gender,
            Result.dns.is_(False),
        )
        .group_by(Result.athlete_id)
        .subquery()
    )

    # Fetch athletes who meet the threshold, joined with their current Rating
    # so we can sort by μ and enforce minimum career events.
    rows = session.execute(
        select(Athlete.id, Rating.mu, attendance_subq.c.attended)
        .join(attendance_subq, Athlete.id == attendance_subq.c.athlete_id)
        .join(
            Rating,
            (Rating.athlete_id == Athlete.id) & (Rating.discipline == discipline),
        )
        .where(
            Athlete.gender == gender,
            # attendance fraction >= threshold
            attendance_subq.c.attended >= threshold * total_events,
        )
        .order_by(Rating.mu.desc())
        .limit(cap)
    ).all()

    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _top_by_mu(
    session,
    discipline: Discipline,
    gender: Gender,
    cap: int,
    min_career_events: int = 3,
) -> list[int]:
    """Return top-``cap`` athlete IDs by current μ for a discipline+gender.

    Only athletes with ``n_events >= min_career_events`` are included to avoid
    surfacing unrated newcomers at the top of the list.
    """
    rows = session.execute(
        select(Athlete.id)
        .join(Rating, Rating.athlete_id == Athlete.id)
        .where(
            Athlete.gender == gender,
            Rating.discipline == discipline,
            Rating.n_events >= min_career_events,
        )
        .order_by(Rating.mu.desc())
        .limit(cap)
    ).all()
    return [row[0] for row in rows]
