from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
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

router = APIRouter()


def _get_session():
    factory = get_session_factory()
    return factory()


def _get_rankings(session, gender: Gender, discipline: Discipline, limit: int = 50):
    stmt = (
        select(Rating, Athlete)
        .join(Athlete, Rating.athlete_id == Athlete.id)
        .where(Rating.discipline == discipline, Athlete.gender == gender)
        .order_by(Rating.mu.desc())
        .limit(limit)
    )
    rows = session.execute(stmt).all()
    return [
        {
            "rank": i + 1,
            "id": athlete.id,
            "name": athlete.name,
            "nationality": athlete.nationality or "—",
            "mu": round(rating.mu, 1),
            "sigma": round(rating.sigma, 1),
            "n_events": rating.n_events,
            "provisional": rating.provisional,
        }
        for i, (rating, athlete) in enumerate(rows)
    ]


@router.get("/", response_class=HTMLResponse)
async def leaderboard(request: Request):
    templates = request.app.state.templates
    disciplines = [
        ("lead", "Lead", Discipline.LEAD),
        ("boulder", "Boulder", Discipline.BOULDER),
        ("speed", "Speed", Discipline.SPEED),
        ("combined", "Combined", Discipline.BOULDER_LEAD),
    ]
    with _get_session() as session:
        standings = {}
        for key, label, disc in disciplines:
            men = _get_rankings(session, Gender.M, disc)
            women = _get_rankings(session, Gender.F, disc)
            standings[key] = {
                "label": label,
                "has_data": len(men) > 0 or len(women) > 0,
                "men": men,
                "women": women,
            }
    return templates.TemplateResponse(request, "index.html", {"standings": standings})


@router.get("/athletes/{athlete_id}", response_class=HTMLResponse)
async def athlete_profile(request: Request, athlete_id: int):
    templates = request.app.state.templates
    with _get_session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            return HTMLResponse("Athlete not found", status_code=404)

        rating = session.execute(
            select(Rating).where(
                Rating.athlete_id == athlete_id,
                Rating.discipline == Discipline.LEAD,
            )
        ).scalar_one_or_none()

        history = list(
            session.execute(
                select(RatingHistory, Event)
                .join(Event, RatingHistory.event_id == Event.id)
                .where(RatingHistory.athlete_id == athlete_id)
                .order_by(Event.start_date.asc())
            ).all()
        )

        chart_labels = []
        chart_mu = []
        chart_sigma_upper = []
        chart_sigma_lower = []
        recent_events = []

        for rh, event in history:
            label = f"{event.name} ({event.season})"
            chart_labels.append(label)
            chart_mu.append(round(rh.mu_after, 1))
            chart_sigma_upper.append(round(rh.mu_after + rh.sigma_after, 1))
            chart_sigma_lower.append(round(rh.mu_after - rh.sigma_after, 1))

        for rh, event in reversed(history[-20:]):
            delta = rh.mu_after - rh.mu_before

            # Best (lowest) rank achieved by the athlete across all rounds in this event.
            # Exclude DNS (did not start) entries; include DNF so we still show a rank
            # if the athlete at least started.
            place_row = session.execute(
                select(func.min(Result.rank))
                .join(Round, Result.round_id == Round.id)
                .where(
                    Round.event_id == event.id,
                    Result.athlete_id == athlete_id,
                    Result.dns.is_(False),
                    Result.rank.is_not(None),
                )
            ).scalar_one_or_none()

            recent_events.append({
                "event_id": event.id,
                "event_name": event.name,
                "season": event.season,
                "mu_before": round(rh.mu_before, 1),
                "mu_after": round(rh.mu_after, 1),
                "delta": round(delta, 1),
                "delta_class": "positive" if delta > 0 else "negative" if delta < 0 else "",
                "place": place_row,
            })

    return templates.TemplateResponse(request, "athlete.html", {
        "athlete": {
            "id": athlete.id,
            "name": athlete.name,
            "nationality": athlete.nationality or "—",
            "gender": athlete.gender.value,
        },
        "rating": {
            "mu": round(rating.mu, 1) if rating else None,
            "sigma": round(rating.sigma, 1) if rating else None,
            "n_events": rating.n_events if rating else 0,
            "provisional": rating.provisional if rating else True,
        },
        "chart_labels": json.dumps(chart_labels),
        "chart_mu": json.dumps(chart_mu),
        "chart_sigma_upper": json.dumps(chart_sigma_upper),
        "chart_sigma_lower": json.dumps(chart_sigma_lower),
        "recent_events": recent_events,
    })


@router.get("/events", response_class=HTMLResponse)
async def event_list(request: Request):
    templates = request.app.state.templates
    with _get_session() as session:
        stmt = (
            select(Event)
            .where(Event.discipline == Discipline.LEAD)
            .order_by(Event.start_date.desc())
            .limit(200)
        )
        events = list(session.execute(stmt).scalars())
        event_data = [
            {
                "id": e.id,
                "name": e.name,
                "season": e.season,
                "tier": e.tier.value.replace("_", " ").title(),
                "date": str(e.start_date),
            }
            for e in events
        ]
    return templates.TemplateResponse(request, "events.html", {"events": event_data})


@router.get("/events/{event_id}", response_class=HTMLResponse)
async def event_detail(request: Request, event_id: int):
    templates = request.app.state.templates
    with _get_session() as session:
        event = session.get(Event, event_id)
        if not event:
            return HTMLResponse("Event not found", status_code=404)

        rounds_data = []
        for rnd in sorted(event.rounds, key=lambda r: r.round_type.value):
            results = list(
                session.execute(
                    select(Result, Athlete)
                    .join(Athlete, Result.athlete_id == Athlete.id)
                    .where(Result.round_id == rnd.id)
                    .order_by(Result.rank.asc())
                ).all()
            )

            result_rows = []
            for res, athlete in results:
                rh = session.execute(
                    select(RatingHistory).where(
                        RatingHistory.athlete_id == athlete.id,
                        RatingHistory.round_id == rnd.id,
                    )
                ).scalar_one_or_none()

                delta = (rh.mu_after - rh.mu_before) if rh else None
                result_rows.append({
                    "athlete_id": athlete.id,
                    "name": athlete.name,
                    "nationality": athlete.nationality or "—",
                    "rank": res.rank,
                    "raw_score": res.raw_score or "—",
                    "mu_before": round(rh.mu_before, 1) if rh else "—",
                    "mu_after": round(rh.mu_after, 1) if rh else "—",
                    "delta": round(delta, 1) if delta is not None else "—",
                    "delta_class": "positive" if delta and delta > 0 else "negative" if delta and delta < 0 else "",
                })

            rounds_data.append({
                "round_type": rnd.round_type.value.title(),
                "gender": rnd.gender.value,
                "results": result_rows,
            })

    return templates.TemplateResponse(request, "event.html", {
        "event": {
            "id": event.id,
            "name": event.name,
            "season": event.season,
            "tier": event.tier.value.replace("_", " ").title(),
        },
        "rounds": rounds_data,
    })


@router.get("/breakdown/{athlete_id}/{event_id}", response_class=HTMLResponse)
async def rating_breakdown(request: Request, athlete_id: int, event_id: int):
    templates = request.app.state.templates
    with _get_session() as session:
        athlete = session.get(Athlete, athlete_id)
        event = session.get(Event, event_id)
        if not athlete or not event:
            return HTMLResponse("Not found", status_code=404)

        histories = list(
            session.execute(
                select(RatingHistory, Round)
                .join(Round, RatingHistory.round_id == Round.id)
                .where(
                    RatingHistory.athlete_id == athlete_id,
                    RatingHistory.event_id == event_id,
                )
                .order_by(Round.round_type)
            ).all()
        )

        rounds_breakdown = []
        for rh, rnd in histories:
            pairs = rh.contributing_pairs or []
            resolved_pairs = []
            for p in pairs:
                opponent = session.get(Athlete, p["opponent_id"])
                resolved_pairs.append({
                    "opponent_name": opponent.name if opponent else f"ID {p['opponent_id']}",
                    "result": p["result"],
                    "expected": p["expected"],
                    "actual": p["actual"],
                    "delta": p["delta"],
                    "margin_multiplier": p["margin_multiplier"],
                })

            rounds_breakdown.append({
                "round_type": rnd.round_type.value.title(),
                "mu_before": round(rh.mu_before, 1),
                "mu_after": round(rh.mu_after, 1),
                "delta": round(rh.mu_after - rh.mu_before, 1),
                "pairs": resolved_pairs,
            })

    return templates.TemplateResponse(request, "breakdown.html", {
        "athlete": {"id": athlete.id, "name": athlete.name},
        "event": {"id": event.id, "name": event.name, "season": event.season},
        "rounds": rounds_breakdown,
    })
