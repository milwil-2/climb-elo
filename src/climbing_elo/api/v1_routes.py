"""Public REST API v1 endpoints."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError

from climbing_elo.api.limiter import limiter

from climbing_elo.cache import predictions_cache
from climbing_elo.database import get_session_factory
from climbing_elo.engine.activity import (
    INACTIVE_THRESHOLD_MONTHS,
    RETIRED_THRESHOLD_YEARS,
    sigma_now,
)
from climbing_elo.engine.likely_roster import likely_competitors
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_podium_probabilities,
)
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventForecast,
    EventForecastScore,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
)
from climbing_elo.api.schemas import (
    AthleteCombined,
    AthleteDetail,
    AthleteHistoryResponse,
    AthleteRating,
    AthleteSearchResult,
    CombinedLeaderboardEntry,
    CombinedLeaderboardResponse,
    DisciplineInfo,
    EventDetail,
    EventForecastResponse,
    EventForecastRow,
    EventForecastScoreRow,
    EventsResponse,
    EventSummary,
    GenderPrediction,
    HistoryPoint,
    LeaderboardEntry,
    LeaderboardResponse,
    ModelPerformanceEventEntry,
    ModelPerformanceResponse,
    PredictedAthlete,
    ProjectionEntry,
    ProjectionRequest,
    ProjectionResponse,
    RecentEvent,
    ResultRow,
    RoundDetail,
    UpcomingPredictionEntry,
    UpcomingPredictionsResponse,
)

#: Maximum athletes per projection request — matches form cap.
_MAX_ATHLETES_PER_PROJECTION = 64

#: Disciplines surfaced on the predictions endpoint (no BOULDER_LEAD for upcoming).
_PREDICTIONS_DISCIPLINES = [
    ("lead", "Lead", Discipline.LEAD),
    ("boulder", "Boulder", Discipline.BOULDER),
    ("speed", "Speed", Discipline.SPEED),
]

#: Cap on upcoming events per discipline to bound Monte Carlo work.
_MAX_UPCOMING_PER_DISCIPLINE = 50

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


@router.get(
    "/disciplines",
    response_model=list[DisciplineInfo],
    summary="List supported disciplines",
)
async def list_disciplines() -> list[DisciplineInfo]:
    """Return all supported disciplines and their API codes."""
    return [
        DisciplineInfo(
            code="lead",
            name="Lead",
            description="Lead climbing — athletes attempt a single route, scored by height reached.",
        ),
        DisciplineInfo(
            code="boulder",
            name="Boulder",
            description="Bouldering — short powerful problems, scored by tops and zones.",
        ),
        DisciplineInfo(
            code="speed",
            name="Speed",
            description="Speed climbing — head-to-head race on a standardised 15-m route.",
        ),
        DisciplineInfo(
            code="boulder_lead",
            name="Boulder & Lead (Combined)",
            description="Combined discipline replacing the former 'combined' format.",
        ),
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/leaderboard
# ---------------------------------------------------------------------------


@router.get(
    "/leaderboard",
    response_model=LeaderboardResponse,
    summary="Get paginated leaderboard",
)
async def leaderboard(
    discipline: str = Query(
        "lead", description="Discipline: lead, boulder, speed, boulder_lead / combined"
    ),
    gender: str = Query("M", description="Gender: M or F"),
    view: Literal["active", "all", "legacy"] = Query(
        "active",
        description=(
            "Activity filter: `active` (default — last 12 months), "
            "`all` (smart all-time list, excludes likely-retired athletes), "
            "or `legacy` (no filter — pre-#91 behaviour)."
        ),
    ),
    limit: int = Query(50, ge=1, le=100, description="Number of results (1–100)"),
    offset: int = Query(0, ge=0, le=10000, description="Pagination offset (max 10000)"),
) -> LeaderboardResponse:
    """
    Return paginated ELO leaderboard for the requested discipline and gender.

    Athletes are ranked by descending μ (mean rating).

    **view**: Default changed from `legacy` (all athletes) to `active` (last
    12-months competitors) on 2026-05-26 per #91. Pass `view=all` for the
    smart all-time list (hides athletes classified by
    ``engine.activity.is_likely_retired_simple`` — i.e. either manually
    flagged via ``Athlete.retired_at`` or >3 years since their last event),
    `view=legacy` for the pre-#91 behaviour.
    """
    disc = _resolve_discipline(discipline)
    gen = _resolve_gender(gender)

    today = date.today()
    base_stmt = (
        select(Rating, Athlete)
        .join(Athlete, Rating.athlete_id == Athlete.id)
        .where(Rating.discipline == disc, Athlete.gender == gen)
    )

    if view == "active":
        cutoff = today - timedelta(days=int(INACTIVE_THRESHOLD_MONTHS * 30.4375))
        base_stmt = base_stmt.where(Rating.last_event_at >= cutoff)
    elif view == "all":
        cutoff = today - timedelta(days=int(RETIRED_THRESHOLD_YEARS * 365.25))
        base_stmt = base_stmt.where(
            Athlete.retired_at.is_(None),
            or_(Rating.last_event_at.is_(None), Rating.last_event_at >= cutoff),
        )
    # else: "legacy" — no filter.

    with _session() as session:
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
                sigma=round(sigma_now(rating.sigma, rating.last_event_at), 2),
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
# GET /api/v1/athletes  — name search / typeahead
# ---------------------------------------------------------------------------


@router.get(
    "/athletes",
    response_model=list[AthleteSearchResult],
    summary="Search athletes by name",
)
async def search_athletes(
    q: str = Query(
        ...,
        min_length=1,
        max_length=80,
        description="Case-insensitive substring match on athlete name.",
    ),
    gender: Optional[str] = Query(None, description="Optional gender filter: M or F."),
    discipline: Optional[str] = Query(
        None,
        description=(
            "Optional discipline: lead, boulder, speed, boulder_lead / combined. "
            "When supplied, `mu` is the rating in this discipline and only "
            "athletes with such a rating are returned."
        ),
    ),
    limit: int = Query(20, ge=1, le=50, description="Max results (1–50)."),
) -> list[AthleteSearchResult]:
    """
    Search athletes by name (case-insensitive substring).

    Powers the typeahead pickers on the `/athletes` index and `/head-to-head`
    pages. Results are ordered by `mu` descending (athletes with no rating sort
    last). When `discipline` is supplied, `mu` is the rating in that discipline
    and only athletes that have such a rating are returned. With no discipline,
    `mu` is the athlete's highest rating across all disciplines (or `null`).

    **Rate limit**: the default 120/min applies. Typeahead callers must debounce
    (~250–300 ms) so per-keystroke fetches stay well under the cap.
    """
    gen: Optional[Gender] = None
    if gender is not None:
        gen = _resolve_gender(gender)

    disc: Optional[Discipline] = None
    if discipline is not None:
        disc = _resolve_discipline(discipline)

    pattern = f"%{q.strip()}%"

    with _session() as session:
        if disc is not None:
            # Join the requested-discipline rating; only matching athletes
            # (those that have a rating in this discipline) are returned.
            stmt = (
                select(Athlete, Rating.mu)
                .join(
                    Rating,
                    (Rating.athlete_id == Athlete.id) & (Rating.discipline == disc),
                )
                .where(Athlete.name.ilike(pattern))
            )
            if gen is not None:
                stmt = stmt.where(Athlete.gender == gen)
            stmt = stmt.order_by(Rating.mu.desc()).limit(limit)
            rows = session.execute(stmt).all()
            return [
                AthleteSearchResult(
                    id=ath.id,
                    name=ath.name,
                    nationality=ath.nationality,
                    gender=ath.gender.value,
                    mu=round(mu, 2) if mu is not None else None,
                )
                for ath, mu in rows
            ]

        # No discipline: mu is the athlete's highest rating across disciplines.
        # LEFT JOIN onto the per-athlete max(mu) so athletes without any rating
        # still match (mu = null) and sort last.
        max_mu_subq = (
            select(
                Rating.athlete_id.label("athlete_id"),
                func.max(Rating.mu).label("max_mu"),
            )
            .group_by(Rating.athlete_id)
            .subquery()
        )
        stmt = (
            select(Athlete, max_mu_subq.c.max_mu)
            .outerjoin(max_mu_subq, max_mu_subq.c.athlete_id == Athlete.id)
            .where(Athlete.name.ilike(pattern))
        )
        if gen is not None:
            stmt = stmt.where(Athlete.gender == gen)
        # NULL mu sorts last under desc() in both SQLite and Postgres' default
        # (NULLS LAST for DESC on SQLite; we coalesce to be portable).
        stmt = stmt.order_by(func.coalesce(max_mu_subq.c.max_mu, -1.0).desc()).limit(
            limit
        )
        rows = session.execute(stmt).all()
        return [
            AthleteSearchResult(
                id=ath.id,
                name=ath.name,
                nationality=ath.nationality,
                gender=ath.gender.value,
                mu=round(max_mu, 2) if max_mu is not None else None,
            )
            for ath, max_mu in rows
        ]


# ---------------------------------------------------------------------------
# GET /api/v1/athletes/{athlete_id}
# ---------------------------------------------------------------------------


@router.get(
    "/athletes/{athlete_id}",
    response_model=AthleteDetail,
    summary="Get athlete details",
)
async def athlete_detail(athlete_id: int) -> AthleteDetail:
    """
    Return an athlete's profile, all discipline ratings, and up to 20 most recent events.
    """
    with _session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            raise HTTPException(
                status_code=404, detail=f"Athlete {athlete_id} not found"
            )

        ratings_rows = (
            session.execute(select(Rating).where(Rating.athlete_id == athlete_id))
            .scalars()
            .all()
        )

        ratings = [
            AthleteRating(
                discipline=_discipline_label(r.discipline),
                mu=round(r.mu, 2),
                sigma=round(sigma_now(r.sigma, r.last_event_at), 2),
                n_events=r.n_events,
                provisional=r.provisional,
                last_event_at=r.last_event_at,
            )
            for r in ratings_rows
        ]

        # Recent events across all disciplines — last 20 by date
        history_rows = list(
            session.execute(
                select(RatingHistory, Event)
                .join(Event, RatingHistory.event_id == Event.id)
                .where(RatingHistory.athlete_id == athlete_id)
                .order_by(Event.start_date.desc())
                .limit(20)
            ).all()
        )

        # De-duplicate per event (keep last round's history entry)
        seen_events: set[int] = set()
        recent_events: list[RecentEvent] = []
        for rh, event in history_rows:
            if event.id in seen_events:
                continue
            seen_events.add(event.id)
            delta = rh.mu_after - rh.mu_before
            recent_events.append(
                RecentEvent(
                    event_id=event.id,
                    event_name=event.name,
                    season=event.season,
                    discipline=_discipline_label(event.discipline),
                    mu_before=round(rh.mu_before, 2),
                    mu_after=round(rh.mu_after, 2),
                    delta=round(delta, 2),
                )
            )

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
    discipline: str = Query(
        "lead", description="Discipline: lead, boulder, speed, boulder_lead / combined"
    ),
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
            raise HTTPException(
                status_code=404, detail=f"Athlete {athlete_id} not found"
            )

        # Fetch all rating-history rows for this athlete × discipline, sorted by date
        rows = list(
            session.execute(
                select(RatingHistory, Event, Round)
                .join(Event, RatingHistory.event_id == Event.id)
                .join(Round, RatingHistory.round_id == Round.id)
                .where(
                    RatingHistory.athlete_id == athlete_id,
                    Event.discipline == disc,
                )
                .order_by(Event.start_date.asc(), Round.round_type.asc())
            ).all()
        )

        # Keep only the last round per event so each event appears once
        event_last: dict[int, tuple] = {}
        for rh, event, rnd in rows:
            event_last[event.id] = (rh, event, rnd)

        points: list[HistoryPoint] = []
        for rh, event, _rnd in event_last.values():
            points.append(
                HistoryPoint(
                    event_id=event.id,
                    event_name=event.name,
                    event_date=event.start_date,
                    season=event.season,
                    mu_after=round(rh.mu_after, 2),
                    sigma_after=round(rh.sigma_after, 2),
                    mu_before=round(rh.mu_before, 2),
                    sigma_before=round(rh.sigma_before, 2),
                    delta=round(rh.mu_after - rh.mu_before, 2),
                )
            )

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
    season: Optional[int] = Query(
        None, ge=2000, le=2100, description="Filter by season year"
    ),
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

        events = (
            session.execute(
                base_stmt.order_by(Event.start_date.desc()).limit(limit).offset(offset)
            )
            .scalars()
            .all()
        )

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


@router.get(
    "/events/{event_id}", response_model=EventDetail, summary="Get event details"
)
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
                results_out.append(
                    ResultRow(
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
                    )
                )

            rounds_out.append(
                RoundDetail(
                    id=rnd.id,
                    round_type=rnd.round_type.value,
                    gender=rnd.gender.value,
                    athlete_count=rnd.athlete_count,
                    results=results_out,
                )
            )

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


# ---------------------------------------------------------------------------
# GET /api/v1/combined/leaderboard
# ---------------------------------------------------------------------------


@router.get(
    "/combined/leaderboard",
    response_model=CombinedLeaderboardResponse,
    summary="Get paginated combined (Boulder+Lead) leaderboard",
)
async def combined_leaderboard(
    gender: str = Query("M", description="Gender: M or F"),
    limit: int = Query(50, ge=1, le=100, description="Number of results (1–100)"),
    offset: int = Query(0, ge=0, le=10000, description="Pagination offset (max 10000)"),
) -> CombinedLeaderboardResponse:
    """
    Return paginated combined (BOULDER_LEAD) ELO leaderboard for the requested gender.

    Each entry includes the combined μ/σ plus the underlying boulder and lead
    breakdown so API consumers can see how the geometric mean is formed.

    Athletes are ranked by descending μ_combined.
    """
    gen = _resolve_gender(gender)

    with _session() as session:
        # Main query: join BOULDER_LEAD Rating → Athlete
        base_stmt = (
            select(Rating, Athlete)
            .join(Athlete, Rating.athlete_id == Athlete.id)
            .where(Rating.discipline == Discipline.BOULDER_LEAD, Athlete.gender == gen)
        )

        total: int = session.execute(
            select(func.count()).select_from(base_stmt.subquery())
        ).scalar_one()

        rows = session.execute(
            base_stmt.order_by(Rating.mu.desc()).limit(limit).offset(offset)
        ).all()

        items: list[CombinedLeaderboardEntry] = []
        for i, (combined_rating, athlete) in enumerate(rows):
            # Fetch individual boulder and lead ratings for the breakdown
            boulder_rating = session.execute(
                select(Rating).where(
                    Rating.athlete_id == athlete.id,
                    Rating.discipline == Discipline.BOULDER,
                )
            ).scalar_one_or_none()
            lead_rating = session.execute(
                select(Rating).where(
                    Rating.athlete_id == athlete.id,
                    Rating.discipline == Discipline.LEAD,
                )
            ).scalar_one_or_none()

            items.append(
                CombinedLeaderboardEntry(
                    rank=offset + i + 1,
                    athlete_id=athlete.id,
                    name=athlete.name,
                    nationality=athlete.nationality,
                    gender=athlete.gender.value,
                    mu=round(combined_rating.mu, 2),
                    sigma=round(
                        sigma_now(combined_rating.sigma, combined_rating.last_event_at),
                        2,
                    ),
                    n_events=combined_rating.n_events,
                    provisional=combined_rating.provisional,
                    last_event_at=combined_rating.last_event_at,
                    mu_boulder=round(boulder_rating.mu, 2) if boulder_rating else 0.0,
                    mu_lead=round(lead_rating.mu, 2) if lead_rating else 0.0,
                    sigma_boulder=round(
                        sigma_now(boulder_rating.sigma, boulder_rating.last_event_at), 2
                    )
                    if boulder_rating
                    else 0.0,
                    sigma_lead=round(
                        sigma_now(lead_rating.sigma, lead_rating.last_event_at), 2
                    )
                    if lead_rating
                    else 0.0,
                )
            )

    return CombinedLeaderboardResponse(
        gender=gen.value,
        limit=limit,
        offset=offset,
        total=total,
        items=items,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/athletes/{athlete_id}/combined
# ---------------------------------------------------------------------------


@router.get(
    "/athletes/{athlete_id}/combined",
    response_model=AthleteCombined,
    summary="Get athlete's combined (Boulder+Lead) rating",
)
async def athlete_combined(athlete_id: int) -> AthleteCombined:
    """
    Return one athlete's combined BOULDER_LEAD rating plus the underlying
    boulder and lead breakdown.

    Returns 404 if the athlete does not have a combined rating (e.g. they have
    not competed in both disciplines or ``compute_combined_ratings.py`` has not
    been run yet).
    """
    with _session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            raise HTTPException(
                status_code=404, detail=f"Athlete {athlete_id} not found"
            )

        combined_rating = session.execute(
            select(Rating).where(
                Rating.athlete_id == athlete_id,
                Rating.discipline == Discipline.BOULDER_LEAD,
            )
        ).scalar_one_or_none()

        if combined_rating is None:
            raise HTTPException(
                status_code=404,
                detail=f"Athlete {athlete_id} has no combined (Boulder+Lead) rating",
            )

        boulder_rating = session.execute(
            select(Rating).where(
                Rating.athlete_id == athlete_id,
                Rating.discipline == Discipline.BOULDER,
            )
        ).scalar_one_or_none()

        lead_rating = session.execute(
            select(Rating).where(
                Rating.athlete_id == athlete_id,
                Rating.discipline == Discipline.LEAD,
            )
        ).scalar_one_or_none()

    return AthleteCombined(
        athlete_id=athlete.id,
        name=athlete.name,
        nationality=athlete.nationality,
        gender=athlete.gender.value,
        mu_combined=round(combined_rating.mu, 2),
        sigma_combined=round(
            sigma_now(combined_rating.sigma, combined_rating.last_event_at), 2
        ),
        n_events_combined=combined_rating.n_events,
        provisional_combined=combined_rating.provisional,
        mu_boulder=round(boulder_rating.mu, 2) if boulder_rating else 0.0,
        mu_lead=round(lead_rating.mu, 2) if lead_rating else 0.0,
        sigma_boulder=round(
            sigma_now(boulder_rating.sigma, boulder_rating.last_event_at), 2
        )
        if boulder_rating
        else 0.0,
        sigma_lead=round(sigma_now(lead_rating.sigma, lead_rating.last_event_at), 2)
        if lead_rating
        else 0.0,
        last_event_at=combined_rating.last_event_at,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/projections
# ---------------------------------------------------------------------------


def _make_projection_cache_key(discipline: Discipline, athlete_ids: list[int]) -> str:
    """Stable cache key for a projection request fingerprint."""
    sorted_ids = sorted(athlete_ids)
    payload = json.dumps({"disc": discipline.value, "ids": sorted_ids}, sort_keys=True)
    return "api:projections:" + hashlib.sha256(payload.encode()).hexdigest()


@router.post(
    "/projections",
    response_model=ProjectionResponse,
    summary="Compute podium probability projections",
)
@limiter.limit("10/minute")
async def projections(
    request: Request, response: Response, body: ProjectionRequest
) -> ProjectionResponse:
    """
    Run Monte Carlo podium-probability projections for a custom set of athletes.

    Supply a discipline and a list of 2–64 athlete IDs. The engine draws 10,000
    simulated events from each athlete's rating distribution and returns win,
    podium, top-8, and expected-rank probabilities.

    **Caching**: results are cached in-memory for 1 hour keyed on the request
    fingerprint (discipline + sorted athlete IDs). Repeated identical requests
    are served from cache at negligible cost.

    **Rate limit**: 10 requests/min per IP (stricter than the 120/min default
    because each uncached call runs 10k Monte Carlo simulations).
    """
    disc = _resolve_discipline(body.discipline)

    # Validate no duplicates
    if len(body.athlete_ids) != len(set(body.athlete_ids)):
        raise HTTPException(
            status_code=422,
            detail="athlete_ids must not contain duplicates",
        )

    cache_key = _make_projection_cache_key(disc, body.athlete_ids)
    cached = predictions_cache.get(cache_key)

    with _session() as session:
        from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA

        proj_inputs: list[AthleteProjectionInput] = []
        for aid in body.athlete_ids:
            athlete = session.get(Athlete, aid)
            if not athlete:
                raise HTTPException(
                    status_code=404,
                    detail=f"Athlete {aid} not found",
                )
            rating = session.execute(
                select(Rating).where(
                    Rating.athlete_id == aid,
                    Rating.discipline == disc,
                )
            ).scalar_one_or_none()
            mu = rating.mu if rating else DEFAULT_MU
            sigma = (
                sigma_now(rating.sigma, rating.last_event_at)
                if rating
                else DEFAULT_SIGMA
            )
            proj_inputs.append(
                AthleteProjectionInput(
                    athlete_id=aid,
                    mu=mu,
                    sigma=sigma,
                    name=athlete.name,
                )
            )

        if cached is not None:
            probs = cached
        else:
            probs = compute_podium_probabilities(proj_inputs, n_simulations=10_000)
            predictions_cache.set(cache_key, probs)

        items = [
            ProjectionEntry(
                athlete_id=a.athlete_id,
                name=a.name,
                mu=round(a.mu, 2),
                sigma=round(a.sigma, 2),
                win=probs[a.athlete_id]["win"],
                podium=probs[a.athlete_id]["podium"],
                top_8=probs[a.athlete_id]["top_8"],
                expected_rank=probs[a.athlete_id]["expected_rank"],
            )
            for a in sorted(
                proj_inputs, key=lambda x: probs[x.athlete_id]["expected_rank"]
            )
        ]

    return ProjectionResponse(
        discipline=_discipline_label(disc),
        n_athletes=len(proj_inputs),
        n_simulations=10_000,
        items=items,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/predictions/upcoming
# ---------------------------------------------------------------------------


@router.get(
    "/predictions/upcoming",
    response_model=UpcomingPredictionsResponse,
    summary="List upcoming events with predicted top-3",
)
@limiter.limit("60/minute")
async def predictions_upcoming(
    request: Request,
    response: Response,
    discipline: Optional[str] = Query(
        None,
        description="Filter by discipline: lead, boulder, speed. Omit for all.",
    ),
    season: Optional[int] = Query(
        None,
        ge=2000,
        le=2100,
        description="Filter by season year. Defaults to the current year.",
    ),
) -> UpcomingPredictionsResponse:
    """
    Return upcoming events (start_date >= today) with predicted top-3 per gender.

    For events with registered athletes (stored via the scraper), projections use
    those athletes.  For events without a registered list, the endpoint falls back
    to a *likely-competitor roster* derived from season attendance patterns (same
    logic as the /predictions HTML page).

    **Roster source flag**: each entry in ``predictions`` carries a
    ``from_likely_roster: bool`` field.  When ``true``, the IFSC start-list for
    that event has not been published yet and the roster was estimated from the
    athletes who competed in ≥ 60 % of this season's finished World Cup events
    (early-season fallback: top-64 by current μ when fewer than 3 events have
    finished).  Consumers should surface this flag prominently so users know the
    athlete list is a forecast, not an official registration.  The HTML dashboard
    shows a "Predicted roster · registrations unavailable" badge in this case.

    **Filters**:
    - ``discipline``: lead, boulder, or speed (boulder_lead excluded — no upcoming
      combined events in the IFSC calendar).
    - ``season``: season year. Defaults to the current calendar year.

    Results are capped at 50 upcoming events per discipline and cached for 1 hour.
    """
    from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA

    today = date.today()
    effective_season = season if season is not None else today.year

    # Build the list of disciplines to query
    if discipline is not None:
        disc_enum = _resolve_discipline(discipline)
        # Restrict to the three "atomic" disciplines (no BOULDER_LEAD for upcoming)
        if disc_enum == Discipline.BOULDER_LEAD:
            raise HTTPException(
                status_code=422,
                detail="boulder_lead / combined is not available for upcoming predictions. "
                "Use lead or boulder individually.",
            )
        disciplines_to_query = [(_discipline_label(disc_enum), disc_enum)]
    else:
        disciplines_to_query = [
            (label, disc) for (label, _, disc) in _PREDICTIONS_DISCIPLINES
        ]

    all_entries: list[UpcomingPredictionEntry] = []

    with _session() as session:
        for disc_key, disc_enum in disciplines_to_query:
            stmt = (
                select(Event)
                .where(
                    Event.discipline == disc_enum,
                    Event.start_date >= today,
                    Event.season == effective_season,
                )
                .order_by(Event.start_date.asc())
                .limit(_MAX_UPCOMING_PER_DISCIPLINE)
            )
            upcoming_events = list(session.execute(stmt).scalars())

            for ev in upcoming_events:
                result_count = session.execute(
                    select(func.count(Result.id))
                    .join(Round, Result.round_id == Round.id)
                    .where(Round.event_id == ev.id)
                ).scalar_one()

                has_athletes = result_count > 0
                from_likely_roster = False
                gender_predictions: list[GenderPrediction] = []

                if has_athletes:
                    available_genders = sorted({rnd.gender for rnd in ev.rounds})
                    for gender_enum in available_genders:
                        seen: set[int] = set()
                        athlete_ids_in_event: list[int] = []
                        for rnd in ev.rounds:
                            if rnd.gender != gender_enum:
                                continue
                            for res in session.execute(
                                select(Result).where(
                                    Result.round_id == rnd.id,
                                    Result.dns.is_(False),
                                )
                            ).scalars():
                                if res.athlete_id not in seen:
                                    athlete_ids_in_event.append(res.athlete_id)
                                    seen.add(res.athlete_id)

                        if not athlete_ids_in_event:
                            continue

                        proj_inputs: list[AthleteProjectionInput] = []
                        for aid in athlete_ids_in_event:
                            athlete = session.get(Athlete, aid)
                            if not athlete:
                                continue
                            rating = session.execute(
                                select(Rating).where(
                                    Rating.athlete_id == aid,
                                    Rating.discipline == disc_enum,
                                )
                            ).scalar_one_or_none()
                            mu = rating.mu if rating else DEFAULT_MU
                            sigma = (
                                sigma_now(
                                    rating.sigma,
                                    rating.last_event_at,
                                    today=ev.start_date,
                                )
                                if rating
                                else DEFAULT_SIGMA
                            )
                            proj_inputs.append(
                                AthleteProjectionInput(
                                    athlete_id=aid,
                                    mu=mu,
                                    sigma=sigma,
                                    name=athlete.name,
                                )
                            )

                        if len(proj_inputs) > _MAX_ATHLETES_PER_PROJECTION:
                            proj_inputs = sorted(
                                proj_inputs, key=lambda a: a.mu, reverse=True
                            )[:_MAX_ATHLETES_PER_PROJECTION]

                        _fingerprint = ":".join(
                            f"{a.athlete_id},{a.mu:.2f},{a.sigma:.2f}"
                            for a in sorted(proj_inputs, key=lambda a: a.athlete_id)
                        )
                        _cache_key = (
                            f"api:projections:event:{ev.id}"
                            f":disc:{disc_enum.value}"
                            f":gender:{gender_enum.value}"
                            f":athletes:{_fingerprint}"
                        )
                        probs = predictions_cache.get(_cache_key)
                        if probs is None:
                            probs = compute_podium_probabilities(
                                proj_inputs, n_simulations=10_000
                            )
                            predictions_cache.set(_cache_key, probs)

                        ranked = sorted(
                            proj_inputs,
                            key=lambda a: probs[a.athlete_id]["expected_rank"],
                        )
                        top3 = [
                            PredictedAthlete(
                                athlete_id=a.athlete_id,
                                name=a.name,
                                win=probs[a.athlete_id]["win"],
                                podium=probs[a.athlete_id]["podium"],
                                expected_rank=probs[a.athlete_id]["expected_rank"],
                            )
                            for a in ranked[:3]
                        ]
                        gender_predictions.append(
                            GenderPrediction(
                                gender=gender_enum.value,
                                total_athletes=len(proj_inputs),
                                top_3=top3,
                            )
                        )

                else:
                    # Likely-competitor fallback
                    from climbing_elo.cache import likely_roster_cache

                    for gender_enum in [Gender.M, Gender.F]:
                        _roster_key = (
                            f"roster:{disc_enum.value}:{ev.season}:{gender_enum.value}"
                        )
                        athlete_ids_roster = likely_roster_cache.get(_roster_key)
                        if athlete_ids_roster is None:
                            athlete_ids_roster = likely_competitors(
                                session,
                                disc_enum,
                                ev.season,
                                gender_enum,
                            )
                            likely_roster_cache.set(_roster_key, athlete_ids_roster)

                        if not athlete_ids_roster:
                            continue

                        proj_inputs_fb: list[AthleteProjectionInput] = []
                        for aid in athlete_ids_roster[:_MAX_ATHLETES_PER_PROJECTION]:
                            athlete = session.get(Athlete, aid)
                            if not athlete:
                                continue
                            rating = session.execute(
                                select(Rating).where(
                                    Rating.athlete_id == aid,
                                    Rating.discipline == disc_enum,
                                )
                            ).scalar_one_or_none()
                            mu = rating.mu if rating else DEFAULT_MU
                            sigma = (
                                sigma_now(
                                    rating.sigma,
                                    rating.last_event_at,
                                    today=ev.start_date,
                                )
                                if rating
                                else DEFAULT_SIGMA
                            )
                            proj_inputs_fb.append(
                                AthleteProjectionInput(
                                    athlete_id=aid,
                                    mu=mu,
                                    sigma=sigma,
                                    name=athlete.name,
                                )
                            )

                        if len(proj_inputs_fb) < 2:
                            continue

                        _fingerprint_fb = ":".join(
                            f"{a.athlete_id},{a.mu:.2f},{a.sigma:.2f}"
                            for a in sorted(proj_inputs_fb, key=lambda a: a.athlete_id)
                        )
                        _cache_key_fb = (
                            f"api:projections:likely:{ev.id}"
                            f":disc:{disc_enum.value}"
                            f":gender:{gender_enum.value}"
                            f":athletes:{_fingerprint_fb}"
                        )
                        probs_fb = predictions_cache.get(_cache_key_fb)
                        if probs_fb is None:
                            probs_fb = compute_podium_probabilities(
                                proj_inputs_fb, n_simulations=10_000
                            )
                            predictions_cache.set(_cache_key_fb, probs_fb)

                        ranked_fb = sorted(
                            proj_inputs_fb,
                            key=lambda a: probs_fb[a.athlete_id]["expected_rank"],
                        )
                        top3_fb = [
                            PredictedAthlete(
                                athlete_id=a.athlete_id,
                                name=a.name,
                                win=probs_fb[a.athlete_id]["win"],
                                podium=probs_fb[a.athlete_id]["podium"],
                                expected_rank=probs_fb[a.athlete_id]["expected_rank"],
                            )
                            for a in ranked_fb[:3]
                        ]
                        gender_predictions.append(
                            GenderPrediction(
                                gender=gender_enum.value,
                                total_athletes=len(proj_inputs_fb),
                                top_3=top3_fb,
                            )
                        )

                    if gender_predictions:
                        from_likely_roster = True

                all_entries.append(
                    UpcomingPredictionEntry(
                        event_id=ev.id,
                        event_name=ev.name,
                        discipline=_discipline_label(ev.discipline),
                        season=ev.season,
                        start_date=ev.start_date,
                        tier=ev.tier.value,
                        country=ev.country,
                        has_registered_athletes=has_athletes,
                        from_likely_roster=from_likely_roster,
                        genders=gender_predictions,
                    )
                )

    # Normalise the discipline label in the response
    discipline_label: Optional[str] = None
    if discipline is not None:
        discipline_label = _discipline_label(_resolve_discipline(discipline))

    return UpcomingPredictionsResponse(
        discipline=discipline_label,
        season=effective_season,
        total=len(all_entries),
        items=all_entries,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/events/{event_id}/forecast
# ---------------------------------------------------------------------------


@router.get(
    "/events/{event_id}/forecast",
    response_model=EventForecastResponse,
    summary="Get frozen forecast for one event + gender",
)
async def event_forecast(
    event_id: int,
    gender: str = Query(..., description="Gender: M or F"),
    is_backfill: bool = Query(
        False,
        description=(
            "Set to ``true`` to return retro-replay rows (forecasts seeded by "
            "the historical backfill script). Defaults to live forecasts only."
        ),
    ),
) -> EventForecastResponse:
    """
    Return the frozen :class:`EventForecast` rows for one (event, gender,
    is_backfill) tuple plus the matching :class:`EventForecastScore` row if
    the event has been scored.

    Rows are ordered by ``prob_win`` descending (with ``expected_rank``
    ascending as a tiebreak — lower rank is better).

    ``score`` is ``null`` when post-event scoring has not yet run (e.g. the
    event is upcoming or in-progress). A missing forecast altogether returns
    ``404``.
    """
    gen = _resolve_gender(gender)

    with _session() as session:
        event = session.get(Event, event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

        try:
            rows = list(
                session.execute(
                    select(EventForecast, Athlete)
                    .join(Athlete, EventForecast.athlete_id == Athlete.id)
                    .where(
                        EventForecast.event_id == event_id,
                        EventForecast.gender == gen,
                        EventForecast.is_backfill == is_backfill,
                    )
                    .order_by(
                        EventForecast.prob_win.desc(),
                        EventForecast.expected_rank.asc(),
                    )
                ).all()
            )
        except SQLAlchemyError:
            # Tables not yet provisioned on this deploy — treat as 404.
            session.rollback()
            rows = []

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No forecast found for event {event_id}, gender {gen.value}, "
                    f"is_backfill={is_backfill}"
                ),
            )

        forecast_items = [
            EventForecastRow(
                athlete_id=athlete.id,
                name=athlete.name,
                mu_at_forecast=round(fc.mu_at_forecast, 2),
                sigma_at_forecast=round(fc.sigma_at_forecast, 2),
                prob_qualify=fc.prob_qualify,
                prob_reach_semi=fc.prob_reach_semi,
                prob_reach_final=fc.prob_reach_final,
                prob_podium=fc.prob_podium,
                prob_win=fc.prob_win,
                expected_rank=round(fc.expected_rank, 4),
                roster_source=fc.roster_source,
                engine_version=fc.engine_version,
                generated_at=fc.generated_at,
            )
            for fc, athlete in rows
        ]

        score_row = session.execute(
            select(EventForecastScore).where(
                EventForecastScore.event_id == event_id,
                EventForecastScore.gender == gen,
                EventForecastScore.is_backfill == is_backfill,
            )
        ).scalar_one_or_none()

        score_out: Optional[EventForecastScoreRow] = None
        if score_row is not None:
            score_out = EventForecastScoreRow(
                event_id=score_row.event_id,
                gender=score_row.gender.value,
                is_backfill=score_row.is_backfill,
                engine_version=score_row.engine_version,
                n_athletes=score_row.n_athletes,
                n_simulations=score_row.n_simulations,
                brier_semi=score_row.brier_semi,
                brier_final=score_row.brier_final,
                brier_podium=score_row.brier_podium,
                brier_win=score_row.brier_win,
                logloss_semi=score_row.logloss_semi,
                logloss_final=score_row.logloss_final,
                logloss_podium=score_row.logloss_podium,
                logloss_win=score_row.logloss_win,
                top3_intersection=score_row.top3_intersection,
                top8_intersection=score_row.top8_intersection,
                spearman_rank=score_row.spearman_rank,
                computed_at=score_row.computed_at,
            )

    return EventForecastResponse(forecast=forecast_items, score=score_out)


# ---------------------------------------------------------------------------
# GET /api/v1/model-performance
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@router.get(
    "/model-performance",
    response_model=ModelPerformanceResponse,
    summary="Aggregate model-performance metrics across scored events",
)
async def model_performance(
    season: Optional[int] = Query(
        None,
        ge=2000,
        le=2100,
        description=(
            "Season year to aggregate over. Defaults to the current calendar year. "
            "Public dashboards should keep this at the default so historical "
            "backfill rows from prior seasons stay hidden."
        ),
    ),
    discipline: Optional[str] = Query(
        None,
        description=(
            "Optional discipline filter: lead, boulder, speed. ``boulder_lead`` "
            "is rejected (combined events are not separately forecast). Omit to "
            "aggregate across all three atomic disciplines."
        ),
    ),
    gender: Optional[str] = Query(
        None, description="Optional gender filter: M or F. Omit for both."
    ),
    include_backfill: bool = Query(
        False,
        description=(
            "When ``false`` (default) only live forecast scores are returned. "
            "Set to ``true`` to include retro-replay rows."
        ),
    ),
) -> ModelPerformanceResponse:
    """
    Aggregate :class:`EventForecastScore` rows across the requested filters.

    Returns mean Brier / log-loss for the podium and win stages plus the
    top-3 intersection rate (``sum(top3_intersection) / (3 * n_events)``) and a
    per-event slim summary suitable for a calibration / hit-miss list view.

    An empty result set returns ``200`` with ``n_events_scored=0`` and an empty
    ``aggregates`` object — clients should treat that as "nothing scored yet"
    rather than an error.
    """
    effective_season = season if season is not None else datetime.now(timezone.utc).year

    disc_enum: Optional[Discipline] = None
    if discipline is not None:
        disc_enum = _resolve_discipline(discipline)
        if disc_enum == Discipline.BOULDER_LEAD:
            raise HTTPException(
                status_code=422,
                detail=(
                    "boulder_lead / combined is not available for model-performance "
                    "aggregation. Use lead, boulder, or speed individually, or omit."
                ),
            )

    gen_enum: Optional[Gender] = None
    if gender is not None:
        gen_enum = _resolve_gender(gender)

    discipline_label = _discipline_label(disc_enum) if disc_enum is not None else "all"

    with _session() as session:
        stmt = (
            select(EventForecastScore, Event)
            .join(Event, EventForecastScore.event_id == Event.id)
            .where(
                EventForecastScore.is_backfill == include_backfill,
                Event.season == effective_season,
            )
        )
        if disc_enum is not None:
            stmt = stmt.where(Event.discipline == disc_enum)
        else:
            # Exclude BOULDER_LEAD (combined events aren't separately
            # forecast) AND Speed (#132 — the projection layer's
            # finishing-order MC is a poor fit for Speed's bracket format,
            # so we don't surface noisy Speed numbers in the aggregate).
            # Speed forecast rows still exist in the DB; clients can pass
            # ?discipline=speed explicitly if they want to inspect them.
            stmt = stmt.where(
                Event.discipline.in_([Discipline.LEAD, Discipline.BOULDER])
            )
        if gen_enum is not None:
            stmt = stmt.where(EventForecastScore.gender == gen_enum)

        try:
            rows = list(session.execute(stmt.order_by(Event.start_date.asc())).all())
        except SQLAlchemyError:
            # event_forecast_scores not provisioned yet on this deploy.
            session.rollback()
            rows = []

        n_events = len(rows)

        if n_events == 0:
            return ModelPerformanceResponse(
                season=effective_season,
                discipline=discipline_label,
                n_events_scored=0,
                aggregates={},
                events=[],
            )

        brier_podium = [s.brier_podium for s, _ in rows]
        brier_win = [s.brier_win for s, _ in rows]
        logloss_podium = [s.logloss_podium for s, _ in rows]
        logloss_win = [s.logloss_win for s, _ in rows]
        top3_total = sum(s.top3_intersection for s, _ in rows)

        aggregates: dict[str, float] = {
            "brier_podium_mean": _mean(brier_podium),
            "brier_win_mean": _mean(brier_win),
            "logloss_podium_mean": _mean(logloss_podium),
            "logloss_win_mean": _mean(logloss_win),
            "top3_intersection_rate": top3_total / (3 * n_events),
        }

        events_out = [
            ModelPerformanceEventEntry(
                event_id=ev.id,
                name=ev.name,
                gender=score.gender.value,
                brier_podium=score.brier_podium,
                top3_intersection=score.top3_intersection,
                n_athletes=score.n_athletes,
            )
            for score, ev in rows
        ]

    return ModelPerformanceResponse(
        season=effective_season,
        discipline=discipline_label,
        n_events_scored=n_events,
        aggregates=aggregates,
        events=events_out,
    )
