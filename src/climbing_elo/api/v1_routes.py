"""Public REST API v1 endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from climbing_elo.database import get_session_factory
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
)
from climbing_elo.api.schemas import (
    AthleteDetail,
    AthleteHistoryResponse,
    AthleteRating,
    DisciplineInfo,
    EventDetail,
    EventsResponse,
    EventSummary,
    HistoryPoint,
    LeaderboardEntry,
    LeaderboardResponse,
    RecentEvent,
    ResultRow,
    RoundDetail,
)

router = APIRouter(prefix="/api/v1", tags=["v1"])


def _session():
    factory = get_session_factory()
    return factory()


# ---------------------------------------------------------------------------
# Helpers — map user-facing strings to enum values
# ---------------------------------------------------------------------------

_DISCIPLINE_ALIASES: dict[str, Discipline] = {
    "lead": Discipline.LEAD,
    "l": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "b": Discipline.BOULDER,
    "speed": Discipline.SPEED,
    "s": Discipline.SPEED,
    "boulder_lead": Discipline.BOULDER_LEAD,
    "combined": Discipline.BOULDER_LEAD,
    "bl": Discipline.BOULDER_LEAD,
}

_GENDER_ALIASES: dict[str, Gender] = {
    "m": Gender.M,
    "f": Gender.F,
    # uppercase variants handled by lowercasing before lookup
}


def _resolve_discipline(raw: str) -> Discipline:
    disc = _DISCIPLINE_ALIASES.get(raw.lower())
    if disc is None:
        valid = "lead, boulder, speed, boulder_lead, combined"
        raise HTTPException(
            status_code=422,
            detail=f"Invalid discipline '{raw}'. Valid values: {valid}",
        )
    return disc


def _resolve_gender(raw: str) -> Gender:
    g = _GENDER_ALIASES.get(raw.lower().strip() if raw else "")
    if g is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid gender '{raw}'. Valid values: M, F",
        )
    return g


def _discipline_label(d: Discipline) -> str:
    return {
        Discipline.LEAD: "lead",
        Discipline.BOULDER: "boulder",
        Discipline.SPEED: "speed",
        Discipline.BOULDER_LEAD: "boulder_lead",
    }[d]


# ---------------------------------------------------------------------------
# GET /api/v1/disciplines
# ---------------------------------------------------------------------------

@router.get("/disciplines", response_model=list[DisciplineInfo], summary="List supported disciplines")
async def list_disciplines() -> list[DisciplineInfo]:
    """Return all supported disciplines and their API codes."""
    return [
        DisciplineInfo(code="lead", name="Lead", description="Lead climbing — athletes attempt a single route, scored by height reached."),
        DisciplineInfo(code="boulder", name="Boulder", description="Bouldering — short powerful problems, scored by tops and zones."),
        DisciplineInfo(code="speed", name="Speed", description="Speed climbing — head-to-head race on a standardised 15-m route."),
        DisciplineInfo(code="boulder_lead", name="Boulder & Lead (Combined)", description="Combined discipline replacing the former 'combined' format."),
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/leaderboard
# ---------------------------------------------------------------------------

@router.get("/leaderboard", response_model=LeaderboardResponse, summary="Get paginated leaderboard")
async def leaderboard(
    discipline: str = Query("lead", description="Discipline: lead, boulder, speed, boulder_lead / combined"),
    gender: str = Query("M", description="Gender: M or F"),
    limit: int = Query(50, ge=1, le=100, description="Number of results (1–100)"),
    offset: int = Query(0, ge=0, le=10000, description="Pagination offset (max 10000)"),
) -> LeaderboardResponse:
    """
    Return paginated ELO leaderboard for the requested discipline and gender.

    Athletes are ranked by descending μ (mean rating).
    """
    disc = _resolve_discipline(discipline)
    gen = _resolve_gender(gender)

    with _session() as session:
        base_stmt = (
            select(Rating, Athlete)
            .join(Athlete, Rating.athlete_id == Athlete.id)
            .where(Rating.discipline == disc, Athlete.gender == gen)
        )

        total: int = session.execute(
            select(func.count()).select_from(base_stmt.subquery())
        ).scalar_one()

        rows = session.execute(
            base_stmt.order_by(Rating.mu.desc()).limit(limit).offset(offset)
        ).all()

        items = [
            LeaderboardEntry(
                rank=offset + i + 1,
                athlete_id=athlete.id,
                name=athlete.name,
                nationality=athlete.nationality,
                gender=athlete.gender.value,
                mu=round(rating.mu, 2),
                sigma=round(rating.sigma, 2),
                n_events=rating.n_events,
                provisional=rating.provisional,
                last_event_at=rating.last_event_at,
            )
            for i, (rating, athlete) in enumerate(rows)
        ]

    return LeaderboardResponse(
        discipline=_discipline_label(disc),
        gender=gen.value,
        limit=limit,
        offset=offset,
        total=total,
        items=items,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/athletes/{athlete_id}
# ---------------------------------------------------------------------------

@router.get("/athletes/{athlete_id}", response_model=AthleteDetail, summary="Get athlete details")
async def athlete_detail(athlete_id: int) -> AthleteDetail:
    """
    Return an athlete's profile, all discipline ratings, and up to 20 most recent events.
    """
    with _session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")

        ratings_rows = session.execute(
            select(Rating).where(Rating.athlete_id == athlete_id)
        ).scalars().all()

        ratings = [
            AthleteRating(
                discipline=_discipline_label(r.discipline),
                mu=round(r.mu, 2),
                sigma=round(r.sigma, 2),
                n_events=r.n_events,
                provisional=r.provisional,
                last_event_at=r.last_event_at,
            )
            for r in ratings_rows
        ]

        # Recent events across all disciplines — last 20 by date
        history_rows = list(session.execute(
            select(RatingHistory, Event)
            .join(Event, RatingHistory.event_id == Event.id)
            .where(RatingHistory.athlete_id == athlete_id)
            .order_by(Event.start_date.desc())
            .limit(20)
        ).all())

        # De-duplicate per event (keep last round's history entry)
        seen_events: set[int] = set()
        recent_events: list[RecentEvent] = []
        for rh, event in history_rows:
            if event.id in seen_events:
                continue
            seen_events.add(event.id)
            delta = rh.mu_after - rh.mu_before
            recent_events.append(RecentEvent(
                event_id=event.id,
                event_name=event.name,
                season=event.season,
                discipline=_discipline_label(event.discipline),
                mu_before=round(rh.mu_before, 2),
                mu_after=round(rh.mu_after, 2),
                delta=round(delta, 2),
            ))

    return AthleteDetail(
        id=athlete.id,
        name=athlete.name,
        nationality=athlete.nationality,
        gender=athlete.gender.value,
        year_of_birth=athlete.year_of_birth,
        ratings=ratings,
        recent_events=recent_events,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/athletes/{athlete_id}/history
# ---------------------------------------------------------------------------

@router.get(
    "/athletes/{athlete_id}/history",
    response_model=AthleteHistoryResponse,
    summary="Get athlete rating history",
)
async def athlete_history(
    athlete_id: int,
    discipline: str = Query("lead", description="Discipline: lead, boulder, speed, boulder_lead / combined"),
) -> AthleteHistoryResponse:
    """
    Return the full rating-over-time history for one athlete and discipline.

    Each point is the state after the **final round** of a given event, making
    it suitable for rendering a time-series chart.
    """
    disc = _resolve_discipline(discipline)

    with _session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            raise HTTPException(status_code=404, detail=f"Athlete {athlete_id} not found")

        # Fetch all rating-history rows for this athlete × discipline, sorted by date
        rows = list(session.execute(
            select(RatingHistory, Event, Round)
            .join(Event, RatingHistory.event_id == Event.id)
            .join(Round, RatingHistory.round_id == Round.id)
            .where(
                RatingHistory.athlete_id == athlete_id,
                Event.discipline == disc,
            )
            .order_by(Event.start_date.asc(), Round.round_type.asc())
        ).all())

        # Keep only the last round per event so each event appears once
        event_last: dict[int, tuple] = {}
        for rh, event, rnd in rows:
            event_last[event.id] = (rh, event, rnd)

        points: list[HistoryPoint] = []
        for rh, event, _rnd in event_last.values():
            points.append(HistoryPoint(
                event_id=event.id,
                event_name=event.name,
                event_date=event.start_date,
                season=event.season,
                mu_after=round(rh.mu_after, 2),
                sigma_after=round(rh.sigma_after, 2),
                mu_before=round(rh.mu_before, 2),
                sigma_before=round(rh.sigma_before, 2),
                delta=round(rh.mu_after - rh.mu_before, 2),
            ))

    return AthleteHistoryResponse(
        athlete_id=athlete.id,
        athlete_name=athlete.name,
        discipline=_discipline_label(disc),
        points=points,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/events
# ---------------------------------------------------------------------------

@router.get("/events", response_model=EventsResponse, summary="List events")
async def list_events(
    discipline: Optional[str] = Query(None, description="Filter by discipline"),
    season: Optional[int] = Query(None, ge=2000, le=2100, description="Filter by season year"),
    limit: int = Query(50, ge=1, le=100, description="Number of results (1–100)"),
    offset: int = Query(0, ge=0, le=10000, description="Pagination offset (max 10000)"),
) -> EventsResponse:
    """
    Return a paginated list of competition events, optionally filtered by
    discipline and/or season.
    """
    disc: Optional[Discipline] = None
    if discipline is not None:
        disc = _resolve_discipline(discipline)

    with _session() as session:
        base_stmt = select(Event)
        if disc is not None:
            base_stmt = base_stmt.where(Event.discipline == disc)
        if season is not None:
            base_stmt = base_stmt.where(Event.season == season)

        total: int = session.execute(
            select(func.count()).select_from(base_stmt.subquery())
        ).scalar_one()

        events = session.execute(
            base_stmt.order_by(Event.start_date.desc()).limit(limit).offset(offset)
        ).scalars().all()

        items = [
            EventSummary(
                id=e.id,
                name=e.name,
                tier=e.tier.value,
                country=e.country,
                season=e.season,
                start_date=e.start_date,
                discipline=_discipline_label(e.discipline),
            )
            for e in events
        ]

    return EventsResponse(limit=limit, offset=offset, total=total, items=items)


# ---------------------------------------------------------------------------
# GET /api/v1/events/{event_id}
# ---------------------------------------------------------------------------

@router.get("/events/{event_id}", response_model=EventDetail, summary="Get event details")
async def event_detail(event_id: int) -> EventDetail:
    """
    Return full event details including all rounds and per-athlete results with
    pre/post ELO ratings.
    """
    with _session() as session:
        event = session.get(Event, event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

        rounds_out: list[RoundDetail] = []
        for rnd in sorted(event.rounds, key=lambda r: r.round_type.value):
            result_rows_raw = session.execute(
                select(Result, Athlete)
                .join(Athlete, Result.athlete_id == Athlete.id)
                .where(Result.round_id == rnd.id)
                .order_by(Result.rank.asc())
            ).all()

            athlete_ids_in_round = [ath.id for _, ath in result_rows_raw]
            rh_by_athlete: dict[int, RatingHistory] = {}
            if athlete_ids_in_round:
                for rh in session.execute(
                    select(RatingHistory).where(
                        RatingHistory.round_id == rnd.id,
                        RatingHistory.athlete_id.in_(athlete_ids_in_round),
                    )
                ).scalars():
                    rh_by_athlete[rh.athlete_id] = rh

            results_out: list[ResultRow] = []
            for res, ath in result_rows_raw:
                rh = rh_by_athlete.get(ath.id)
                delta = round(rh.mu_after - rh.mu_before, 2) if rh else None
                results_out.append(ResultRow(
                    athlete_id=ath.id,
                    athlete_name=ath.name,
                    nationality=ath.nationality,
                    rank=res.rank,
                    raw_score=res.raw_score,
                    dnf=res.dnf,
                    dns=res.dns,
                    mu_before=round(rh.mu_before, 2) if rh else None,
                    mu_after=round(rh.mu_after, 2) if rh else None,
                    delta=delta,
                ))

            rounds_out.append(RoundDetail(
                id=rnd.id,
                round_type=rnd.round_type.value,
                gender=rnd.gender.value,
                athlete_count=rnd.athlete_count,
                results=results_out,
            ))

    return EventDetail(
        id=event.id,
        name=event.name,
        tier=event.tier.value,
        country=event.country,
        season=event.season,
        start_date=event.start_date,
        discipline=_discipline_label(event.discipline),
        rounds=rounds_out,
    )
