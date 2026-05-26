"""Likely-competitor roster fallback for upcoming events (Issue #33, updated #62).

When the IFSC API has not yet published a registered-athletes list for an
upcoming event (typically 7-14 days before it starts), this module derives a
"likely roster" from current-season participation.

An athlete is considered a likely competitor in discipline X, season Y, gender G
if they have competed in at least one World Cup event in that discipline and
season (non-DNS result, gender-matched round).

Pre-season (no finished events yet): returns [] — no roster is fabricated.
"""

from __future__ import annotations

from sqlalchemy import distinct, select

from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    Result,
    Round,
)


def likely_competitors(
    session,
    discipline: Discipline,
    season: int,
    gender: Gender,
    cap: int = 64,
) -> list[int]:
    """Return athlete_ids likely to compete in the next event of this discipline+season.

    An athlete qualifies if they have at least one non-DNS Result in a World Cup
    event of the given discipline, season, and gender.  If no athletes meet this
    criterion (e.g. the season has not started yet), an empty list is returned.

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
    cap:
        Maximum number of athletes to return.  Prevents runaway Monte Carlo cost
        (mirrors ``_MAX_ATHLETES_PER_PROJECTION_CARD`` in routes.py).

    Returns
    -------
    list[int]
        Athlete IDs ordered by current μ descending, capped at ``cap``.
        Returns an empty list if no athletes have competed this season.
    """
    # Subquery: distinct athlete IDs with ≥1 non-DNS result in a WC event
    # for the requested discipline + season + gender.
    participated_subq = (
        select(distinct(Result.athlete_id).label("athlete_id"))
        .join(Round, Result.round_id == Round.id)
        .join(Event, Round.event_id == Event.id)
        .where(
            Event.discipline == discipline,
            Event.season == season,
            Event.tier == EventTier.WORLD_CUP,
            Round.gender == gender,
            Result.dns.is_(False),
        )
        .subquery()
    )

    rows = session.execute(
        select(Athlete.id)
        .join(participated_subq, Athlete.id == participated_subq.c.athlete_id)
        .join(
            Rating,
            (Rating.athlete_id == Athlete.id) & (Rating.discipline == discipline),
        )
        .where(Athlete.gender == gender)
        .order_by(Rating.mu.desc())
        .limit(cap)
    ).all()

    return [row[0] for row in rows]
