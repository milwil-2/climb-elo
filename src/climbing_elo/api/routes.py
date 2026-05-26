from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select

from climbing_elo.database import get_session_factory
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_podium_probabilities,
    predict_winner,
)
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)

router = APIRouter()


def _get_session():
    factory = get_session_factory()
    return factory()


def _get_rankings(session, gender: Gender, discipline: Discipline, limit: int = 200):
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


def _get_nationalities(session, gender: Gender, discipline: Discipline) -> list[str]:
    stmt = (
        select(Athlete.nationality)
        .join(Rating, Rating.athlete_id == Athlete.id)
        .where(Rating.discipline == discipline, Athlete.gender == gender, Athlete.nationality.isnot(None))
        .distinct()
        .order_by(Athlete.nationality)
    )
    return [n for n in session.execute(stmt).scalars() if n]


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
            men_nats = _get_nationalities(session, Gender.M, disc)
            women_nats = _get_nationalities(session, Gender.F, disc)
            standings[key] = {
                "label": label,
                "has_data": len(men) > 0 or len(women) > 0,
                "men": men,
                "women": women,
                "men_nats": men_nats,
                "women_nats": women_nats,
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
async def event_list(request: Request, season: Optional[int] = Query(default=None)):
    templates = request.app.state.templates
    with _get_session() as session:
        # Fetch all available seasons for the dropdown
        all_seasons = list(
            session.execute(
                select(Event.season)
                .where(Event.discipline == Discipline.LEAD)
                .distinct()
                .order_by(Event.season.desc())
            ).scalars()
        )

        stmt = select(Event).where(Event.discipline == Discipline.LEAD)
        if season is not None:
            stmt = stmt.where(Event.season == season)
        stmt = stmt.order_by(Event.start_date.desc()).limit(500)

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
    return templates.TemplateResponse(request, "events.html", {
        "events": event_data,
        "all_seasons": all_seasons,
        "selected_season": season,
    })


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


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

_DISCIPLINE_DISPLAY = {
    Discipline.LEAD: "Lead",
    Discipline.BOULDER: "Boulder",
    Discipline.SPEED: "Speed",
    Discipline.BOULDER_LEAD: "Combined",
}

_DISCIPLINE_FROM_STR: dict[str, Discipline] = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
    "combined": Discipline.BOULDER_LEAD,
}


def _build_projection_rows(
    session,
    athletes: list[AthleteProjectionInput],
    probs: dict[int, dict[str, float]],
) -> list[dict]:
    """Merge simulation results with athlete metadata for template rendering."""
    rows = []
    for a in athletes:
        p = probs[a.athlete_id]
        rows.append({
            "athlete_id": a.athlete_id,
            "name": a.name,
            "mu": round(a.mu, 1),
            "sigma": round(a.sigma, 1),
            "win": f"{p['win'] * 100:.1f}%",
            "podium": f"{p['podium'] * 100:.1f}%",
            "top_8": f"{p['top_8'] * 100:.1f}%",
            "expected_rank": p["expected_rank"],
            # Raw floats for sorting
            "win_raw": p["win"],
            "podium_raw": p["podium"],
        })
    rows.sort(key=lambda r: r["expected_rank"])
    # Assign display rank
    for i, row in enumerate(rows):
        row["proj_rank"] = i + 1
    return rows


# ---------------------------------------------------------------------------
# GET /projections/new  — manual projection form
# POST /projections/new — run projection and render results inline
# ---------------------------------------------------------------------------

@router.get("/projections/new", response_class=HTMLResponse)
async def projections_new_form(request: Request):
    templates = request.app.state.templates
    with _get_session() as session:
        athletes = list(
            session.execute(
                select(Athlete).order_by(Athlete.name)
            ).scalars()
        )
        athlete_list = [
            {"id": a.id, "name": a.name, "gender": a.gender.value}
            for a in athletes
        ]
    return templates.TemplateResponse(request, "projections_new.html", {
        "athletes": athlete_list,
        "disciplines": [
            {"key": k, "label": v}
            for k, v in [
                ("lead", "Lead"),
                ("boulder", "Boulder"),
                ("speed", "Speed"),
                ("combined", "Combined"),
            ]
        ],
        "result": None,
        "error": None,
    })


@router.post("/projections/new", response_class=HTMLResponse)
async def projections_new_submit(
    request: Request,
    discipline: Annotated[str, Form()],
    athlete_ids: Annotated[list[int], Form()],
):
    templates = request.app.state.templates

    disc = _DISCIPLINE_FROM_STR.get(discipline)

    with _get_session() as session:
        athletes_all = list(
            session.execute(select(Athlete).order_by(Athlete.name)).scalars()
        )
        athlete_list = [
            {"id": a.id, "name": a.name, "gender": a.gender.value}
            for a in athletes_all
        ]
        disciplines_ctx = [
            {"key": k, "label": v}
            for k, v in [
                ("lead", "Lead"),
                ("boulder", "Boulder"),
                ("speed", "Speed"),
                ("combined", "Combined"),
            ]
        ]

        if disc is None:
            return templates.TemplateResponse(request, "projections_new.html", {
                "athletes": athlete_list,
                "disciplines": disciplines_ctx,
                "result": None,
                "error": "Invalid discipline selected.",
            })

        if len(athlete_ids) < 2:
            return templates.TemplateResponse(request, "projections_new.html", {
                "athletes": athlete_list,
                "disciplines": disciplines_ctx,
                "result": None,
                "error": "Please select at least 2 athletes.",
            })

        # Bound athlete count to prevent DoS via Monte Carlo blowup.
        # 10k sims x large N becomes both memory- and CPU-expensive per request.
        MAX_ATHLETES_PER_FORM = 128
        if len(athlete_ids) > MAX_ATHLETES_PER_FORM:
            return templates.TemplateResponse(request, "projections_new.html", {
                "athletes": athlete_list,
                "disciplines": disciplines_ctx,
                "result": None,
                "error": f"Please select at most {MAX_ATHLETES_PER_FORM} athletes.",
            })

        # Load ratings for selected athletes
        proj_inputs: list[AthleteProjectionInput] = []
        for aid in athlete_ids:
            athlete = session.get(Athlete, aid)
            if not athlete:
                continue
            rating = session.execute(
                select(Rating).where(
                    Rating.athlete_id == aid,
                    Rating.discipline == disc,
                )
            ).scalar_one_or_none()
            if rating is None:
                # Fall back to defaults so the athlete still appears
                from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA
                mu, sigma = DEFAULT_MU, DEFAULT_SIGMA
            else:
                mu, sigma = rating.mu, rating.sigma
            proj_inputs.append(AthleteProjectionInput(
                athlete_id=aid,
                mu=mu,
                sigma=sigma,
                name=athlete.name,
            ))

        probs = compute_podium_probabilities(proj_inputs, n_simulations=10_000)
        rows = _build_projection_rows(session, proj_inputs, probs)

        result_ctx = {
            "discipline": _DISCIPLINE_DISPLAY.get(disc, discipline),
            "rows": rows,
            "source": "manual",
        }

    return templates.TemplateResponse(request, "projections_new.html", {
        "athletes": athlete_list,
        "disciplines": disciplines_ctx,
        "result": result_ctx,
        "error": None,
        "selected_discipline": discipline,
        "selected_athlete_ids": athlete_ids,
    })


# ---------------------------------------------------------------------------
# GET /projections/{event_id}  — projection for a past/in-progress event
# ---------------------------------------------------------------------------

@router.get("/projections/{event_id}", response_class=HTMLResponse)
async def event_projections(request: Request, event_id: int, gender: str = "M"):
    templates = request.app.state.templates
    with _get_session() as session:
        event = session.get(Event, event_id)
        if not event:
            return HTMLResponse("Event not found", status_code=404)

        # Resolve gender filter
        try:
            gender_enum = Gender(gender.upper())
        except ValueError:
            gender_enum = Gender.M

        # Collect unique athletes from all rounds of this event (for the gender)
        # Use qualification round first, fall back to any round if needed.
        athlete_ids_ordered: list[int] = []
        seen: set[int] = set()

        round_priority = [RoundType.QUALIFICATION, RoundType.SEMI, RoundType.FINAL]
        rounds_by_type: dict[RoundType, Round] = {
            rnd.round_type: rnd
            for rnd in event.rounds
            if rnd.gender == gender_enum
        }

        for rt in round_priority:
            rnd = rounds_by_type.get(rt)
            if rnd is None:
                continue
            results = list(
                session.execute(
                    select(Result)
                    .where(Result.round_id == rnd.id)
                ).scalars()
            )
            for res in results:
                if not res.dns and res.athlete_id not in seen:
                    athlete_ids_ordered.append(res.athlete_id)
                    seen.add(res.athlete_id)

        if not athlete_ids_ordered:
            return templates.TemplateResponse(request, "projections.html", {
                "event": {
                    "id": event.id,
                    "name": event.name,
                    "season": event.season,
                    "tier": event.tier.value.replace("_", " ").title(),
                },
                "discipline": _DISCIPLINE_DISPLAY.get(event.discipline, event.discipline.value),
                "gender": gender_enum.value,
                "rows": [],
                "winner": None,
                "error": "No athlete results found for this event.",
            })

        # Build projection inputs
        proj_inputs: list[AthleteProjectionInput] = []
        for aid in athlete_ids_ordered:
            athlete = session.get(Athlete, aid)
            if not athlete:
                continue
            # Use the rating BEFORE the event if available, otherwise current rating
            # For simplicity we use the current rating (post-event) — for past events
            # this is fine as context; for live events there is no post rating yet.
            rating = session.execute(
                select(Rating).where(
                    Rating.athlete_id == aid,
                    Rating.discipline == event.discipline,
                )
            ).scalar_one_or_none()
            if rating is None:
                from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA
                mu, sigma = DEFAULT_MU, DEFAULT_SIGMA
            else:
                mu, sigma = rating.mu, rating.sigma
            proj_inputs.append(AthleteProjectionInput(
                athlete_id=aid,
                mu=mu,
                sigma=sigma,
                name=athlete.name,
            ))

        probs = compute_podium_probabilities(proj_inputs, n_simulations=10_000)
        rows = _build_projection_rows(session, proj_inputs, probs)
        winner_id = predict_winner(proj_inputs)
        winner_name = next(
            (r["name"] for r in rows if r["athlete_id"] == winner_id), None
        )

        # Available genders for this event
        available_genders = sorted({rnd.gender.value for rnd in event.rounds})

    return templates.TemplateResponse(request, "projections.html", {
        "event": {
            "id": event.id,
            "name": event.name,
            "season": event.season,
            "tier": event.tier.value.replace("_", " ").title(),
        },
        "discipline": _DISCIPLINE_DISPLAY.get(event.discipline, event.discipline.value),
        "gender": gender_enum.value,
        "available_genders": available_genders,
        "rows": rows,
        "winner": winner_name,
        "error": None,
    })


# ---------------------------------------------------------------------------
# GET /predictions  — upcoming event predictions hub
# ---------------------------------------------------------------------------

#: Upcoming event d_cat statuses to surface on the predictions page.
_UPCOMING_STATUSES: frozenset[str] = frozenset({"scheduled", "registration", "live"})

#: Disciplines shown on the predictions page, in display order.
_PREDICTIONS_DISCIPLINES = [
    ("lead", "Lead", Discipline.LEAD),
    ("boulder", "Boulder", Discipline.BOULDER),
    ("speed", "Speed", Discipline.SPEED),
]

#: Cap on events per discipline shown on the predictions page to bound Monte Carlo work.
_MAX_UPCOMING_PER_DISCIPLINE = 50


@router.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request):
    """List upcoming IFSC events with ELO-based outcome predictions.

    For each upcoming event (start_date >= today) that has at least one result
    stored (i.e. athletes registered via the scraper), we run Monte Carlo
    simulations and show the predicted top-3 per gender.  Events with no
    stored athlete data show a "Select athletes manually" call-to-action
    linking to ``/projections/new``.

    Events are capped at ``_MAX_UPCOMING_PER_DISCIPLINE`` per discipline to
    prevent excessive computation on page load.  All DB access uses the
    SQLAlchemy ORM — no raw SQL.
    """
    templates = request.app.state.templates
    today = date.today()

    grouped: list[dict] = []

    with _get_session() as session:
        for disc_key, disc_label, disc_enum in _PREDICTIONS_DISCIPLINES:
            # Fetch upcoming events for this discipline ordered by start date.
            stmt = (
                select(Event)
                .where(
                    Event.discipline == disc_enum,
                    Event.start_date >= today,
                )
                .order_by(Event.start_date.asc())
                .limit(_MAX_UPCOMING_PER_DISCIPLINE)
            )
            upcoming_events = list(session.execute(stmt).scalars())

            disc_events: list[dict] = []
            for ev in upcoming_events:
                # Determine whether athlete results are available for this event.
                # An event row exists but may have no rounds/results if it was
                # stored by scrape_upcoming_events (which doesn't fetch athlete lists).
                result_count = session.execute(
                    select(func.count(Result.id))
                    .join(Round, Result.round_id == Round.id)
                    .where(Round.event_id == ev.id)
                ).scalar_one()

                has_athletes = result_count > 0

                predictions_data: dict | None = None
                if has_athletes:
                    # Run projections for each gender that has data.
                    gender_predictions: list[dict] = []
                    available_genders = sorted({rnd.gender for rnd in ev.rounds})

                    for gender_enum in available_genders:
                        # Collect unique athletes from the event's rounds.
                        seen: set[int] = set()
                        athlete_ids: list[int] = []
                        for rnd in ev.rounds:
                            if rnd.gender != gender_enum:
                                continue
                            results_q = session.execute(
                                select(Result)
                                .where(
                                    Result.round_id == rnd.id,
                                    Result.dns.is_(False),
                                )
                            ).scalars()
                            for res in results_q:
                                if res.athlete_id not in seen:
                                    athlete_ids.append(res.athlete_id)
                                    seen.add(res.athlete_id)

                        if not athlete_ids:
                            continue

                        proj_inputs: list[AthleteProjectionInput] = []
                        for aid in athlete_ids:
                            athlete = session.get(Athlete, aid)
                            if not athlete:
                                continue
                            rating = session.execute(
                                select(Rating).where(
                                    Rating.athlete_id == aid,
                                    Rating.discipline == disc_enum,
                                )
                            ).scalar_one_or_none()
                            if rating is None:
                                from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA
                                mu, sigma = DEFAULT_MU, DEFAULT_SIGMA
                            else:
                                mu, sigma = rating.mu, rating.sigma
                            proj_inputs.append(AthleteProjectionInput(
                                athlete_id=aid,
                                mu=mu,
                                sigma=sigma,
                                name=athlete.name,
                            ))

                        probs = compute_podium_probabilities(proj_inputs, n_simulations=10_000)

                        # Build predicted top-3 sorted by expected_rank.
                        ranked = sorted(proj_inputs, key=lambda a: probs[a.athlete_id]["expected_rank"])
                        top3 = [
                            {
                                "athlete_id": a.athlete_id,
                                "name": a.name,
                                "win": f"{probs[a.athlete_id]['win'] * 100:.1f}%",
                                "podium": f"{probs[a.athlete_id]['podium'] * 100:.1f}%",
                                "expected_rank": probs[a.athlete_id]["expected_rank"],
                            }
                            for a in ranked[:3]
                        ]
                        gender_predictions.append({
                            "gender": gender_enum.value,
                            "top3": top3,
                            "total_athletes": len(proj_inputs),
                        })

                    predictions_data = {"genders": gender_predictions}

                disc_events.append({
                    "id": ev.id,
                    "name": ev.name,
                    "season": ev.season,
                    "tier": ev.tier.value.replace("_", " ").title(),
                    "date": str(ev.start_date),
                    "has_athletes": has_athletes,
                    "predictions": predictions_data,
                })

            grouped.append({
                "key": disc_key,
                "label": disc_label,
                "events": disc_events,
                "has_data": len(disc_events) > 0,
            })

    return templates.TemplateResponse(request, "predictions.html", {
        "grouped": grouped,
        "today": str(today),
    })
