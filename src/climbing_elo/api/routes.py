from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select

from climbing_elo.cache import likely_roster_cache, predictions_cache
from climbing_elo.engine.likely_roster import likely_competitors
from climbing_elo.database import get_session_factory
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    ProgressionResult,
    RoundConfig,
    compute_partial_event_probabilities,
    compute_podium_probabilities,
    default_event_format,
    predict_winner,
    simulate_event_progression,
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
        .where(
            Rating.discipline == discipline,
            Athlete.gender == gender,
            Athlete.nationality.isnot(None),
        )
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

            recent_events.append(
                {
                    "event_id": event.id,
                    "event_name": event.name,
                    "season": event.season,
                    "mu_before": round(rh.mu_before, 1),
                    "mu_after": round(rh.mu_after, 1),
                    "delta": round(delta, 1),
                    "delta_class": "positive"
                    if delta > 0
                    else "negative"
                    if delta < 0
                    else "",
                    "place": place_row,
                }
            )

    return templates.TemplateResponse(
        request,
        "athlete.html",
        {
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
        },
    )


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
    return templates.TemplateResponse(
        request,
        "events.html",
        {
            "events": event_data,
            "all_seasons": all_seasons,
            "selected_season": season,
        },
    )


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

            # Batch-fetch RatingHistory for all athletes in this round in ONE query
            # instead of N+1. Without this, a 50-athlete qualification round caused
            # the page to hang (caught by smoke_test.py).
            athlete_ids_in_round = [ath.id for _, ath in results]
            rh_by_athlete: dict[int, RatingHistory] = {}
            if athlete_ids_in_round:
                for rh in session.execute(
                    select(RatingHistory).where(
                        RatingHistory.round_id == rnd.id,
                        RatingHistory.athlete_id.in_(athlete_ids_in_round),
                    )
                ).scalars():
                    rh_by_athlete[rh.athlete_id] = rh

            result_rows = []
            for res, athlete in results:
                rh = rh_by_athlete.get(athlete.id)
                delta = (rh.mu_after - rh.mu_before) if rh else None
                result_rows.append(
                    {
                        "athlete_id": athlete.id,
                        "name": athlete.name,
                        "nationality": athlete.nationality or "—",
                        "rank": res.rank,
                        "raw_score": res.raw_score or "—",
                        "mu_before": round(rh.mu_before, 1) if rh else "—",
                        "mu_after": round(rh.mu_after, 1) if rh else "—",
                        "delta": round(delta, 1) if delta is not None else "—",
                        "delta_class": "positive"
                        if delta and delta > 0
                        else "negative"
                        if delta and delta < 0
                        else "",
                    }
                )

            rounds_data.append(
                {
                    "round_type": rnd.round_type.value.title(),
                    "gender": rnd.gender.value,
                    "results": result_rows,
                }
            )

    return templates.TemplateResponse(
        request,
        "event.html",
        {
            "event": {
                "id": event.id,
                "name": event.name,
                "season": event.season,
                "tier": event.tier.value.replace("_", " ").title(),
            },
            "rounds": rounds_data,
        },
    )


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
                resolved_pairs.append(
                    {
                        "opponent_name": opponent.name
                        if opponent
                        else f"ID {p['opponent_id']}",
                        "result": p["result"],
                        "expected": p["expected"],
                        "actual": p["actual"],
                        "delta": p["delta"],
                        "margin_multiplier": p["margin_multiplier"],
                    }
                )

            rounds_breakdown.append(
                {
                    "round_type": rnd.round_type.value.title(),
                    "mu_before": round(rh.mu_before, 1),
                    "mu_after": round(rh.mu_after, 1),
                    "delta": round(rh.mu_after - rh.mu_before, 1),
                    "pairs": resolved_pairs,
                }
            )

    return templates.TemplateResponse(
        request,
        "breakdown.html",
        {
            "athlete": {"id": athlete.id, "name": athlete.name},
            "event": {"id": event.id, "name": event.name, "season": event.season},
            "rounds": rounds_breakdown,
        },
    )


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
        rows.append(
            {
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
            }
        )
    rows.sort(key=lambda r: r["expected_rank"])
    # Assign display rank
    for i, row in enumerate(rows):
        row["proj_rank"] = i + 1
    return rows


def _build_progression_rows(
    progression: list[ProgressionResult],
    round_configs: list[RoundConfig],
) -> list[dict]:
    """Convert ProgressionResult objects to template-friendly dicts.

    Each row contains per-round advance probabilities (pre-formatted as
    percentage strings + raw floats for colour-coding) plus final podium/win.
    Rows are already sorted by descending mu (best first) by
    simulate_event_progression, so we just assign display ranks here.
    """
    rows = []
    for i, pr in enumerate(progression):
        round_probs = []
        for rc in round_configs:
            raw = pr.advance_probs.get(rc.round_type, 0.0)
            pct = raw * 100
            if pct >= 70:
                css = "prob-green"
            elif pct >= 30:
                css = "prob-yellow"
            else:
                css = "prob-red"
            round_probs.append(
                {
                    "round_type": rc.round_type,
                    "round_label": rc.round_type.title(),
                    "raw": raw,
                    "pct": f"{pct:.1f}%",
                    "css": css,
                }
            )

        # Final podium colour
        fp_raw = pr.final_podium_prob
        fp_pct = fp_raw * 100
        if fp_pct >= 70:
            fp_css = "prob-green"
        elif fp_pct >= 30:
            fp_css = "prob-yellow"
        else:
            fp_css = "prob-red"

        rows.append(
            {
                "proj_rank": i + 1,
                "athlete_id": pr.athlete_id,
                "name": pr.name,
                "mu": round(pr.mu, 1),
                "round_probs": round_probs,
                "final_podium": f"{fp_pct:.1f}%",
                "final_podium_raw": fp_raw,
                "final_podium_css": fp_css,
                "final_win": f"{pr.final_win_prob * 100:.1f}%",
                "final_win_raw": pr.final_win_prob,
            }
        )
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
            session.execute(select(Athlete).order_by(Athlete.name)).scalars()
        )
        athlete_list = [
            {"id": a.id, "name": a.name, "gender": a.gender.value} for a in athletes
        ]
    return templates.TemplateResponse(
        request,
        "projections_new.html",
        {
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
        },
    )


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
            {"id": a.id, "name": a.name, "gender": a.gender.value} for a in athletes_all
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
            return templates.TemplateResponse(
                request,
                "projections_new.html",
                {
                    "athletes": athlete_list,
                    "disciplines": disciplines_ctx,
                    "result": None,
                    "error": "Invalid discipline selected.",
                },
            )

        if len(athlete_ids) < 2:
            return templates.TemplateResponse(
                request,
                "projections_new.html",
                {
                    "athletes": athlete_list,
                    "disciplines": disciplines_ctx,
                    "result": None,
                    "error": "Please select at least 2 athletes.",
                },
            )

        # Bound athlete count to prevent DoS via Monte Carlo blowup.
        # 10k sims x large N becomes both memory- and CPU-expensive per request.
        MAX_ATHLETES_PER_FORM = 128
        if len(athlete_ids) > MAX_ATHLETES_PER_FORM:
            return templates.TemplateResponse(
                request,
                "projections_new.html",
                {
                    "athletes": athlete_list,
                    "disciplines": disciplines_ctx,
                    "result": None,
                    "error": f"Please select at most {MAX_ATHLETES_PER_FORM} athletes.",
                },
            )

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
            proj_inputs.append(
                AthleteProjectionInput(
                    athlete_id=aid,
                    mu=mu,
                    sigma=sigma,
                    name=athlete.name,
                )
            )

        probs = compute_podium_probabilities(proj_inputs, n_simulations=10_000)
        rows = _build_projection_rows(session, proj_inputs, probs)

        result_ctx = {
            "discipline": _DISCIPLINE_DISPLAY.get(disc, discipline),
            "rows": rows,
            "source": "manual",
        }

    return templates.TemplateResponse(
        request,
        "projections_new.html",
        {
            "athletes": athlete_list,
            "disciplines": disciplines_ctx,
            "result": result_ctx,
            "error": None,
            "selected_discipline": discipline,
            "selected_athlete_ids": athlete_ids,
        },
    )


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
            rnd.round_type: rnd for rnd in event.rounds if rnd.gender == gender_enum
        }

        for rt in round_priority:
            rnd = rounds_by_type.get(rt)
            if rnd is None:
                continue
            results = list(
                session.execute(
                    select(Result).where(Result.round_id == rnd.id)
                ).scalars()
            )
            for res in results:
                if not res.dns and res.athlete_id not in seen:
                    athlete_ids_ordered.append(res.athlete_id)
                    seen.add(res.athlete_id)

        if not athlete_ids_ordered:
            return templates.TemplateResponse(
                request,
                "projections.html",
                {
                    "event": {
                        "id": event.id,
                        "name": event.name,
                        "season": event.season,
                        "tier": event.tier.value.replace("_", " ").title(),
                    },
                    "discipline": _DISCIPLINE_DISPLAY.get(
                        event.discipline, event.discipline.value
                    ),
                    "gender": gender_enum.value,
                    "rows": [],
                    "winner": None,
                    "error": "No athlete results found for this event.",
                },
            )

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
            proj_inputs.append(
                AthleteProjectionInput(
                    athlete_id=aid,
                    mu=mu,
                    sigma=sigma,
                    name=athlete.name,
                )
            )

        # Available genders for this event
        available_genders = sorted({rnd.gender.value for rnd in event.rounds})

        # Determine whether the event has multiple rounds we can simulate progression for.
        # We use the actual rounds present in the DB; fall back to default_event_format
        # when fewer than 2 gender-specific rounds are found.
        gender_round_types = sorted(
            {rnd.round_type for rnd in event.rounds if rnd.gender == gender_enum},
            key=lambda rt: rt.value,
        )
        use_progression = len(gender_round_types) >= 2

        if use_progression:
            # Build RoundConfig list from actual DB rounds, ordered qualification → semi → final.
            _rt_order = {
                RoundType.QUALIFICATION: 0,
                RoundType.SEMI: 1,
                RoundType.FINAL: 2,
            }
            # Use default advance counts for each round type.
            _default_format = default_event_format(event.tier.value)
            _default_advance = {
                rc.round_type: rc.advance_count for rc in _default_format
            }
            _round_type_to_str = {
                RoundType.QUALIFICATION: "qualification",
                RoundType.SEMI: "semifinal",
                RoundType.FINAL: "final",
            }

            sorted_rt = sorted(gender_round_types, key=lambda rt: _rt_order.get(rt, 99))
            round_configs: list[RoundConfig] = []
            for rt in sorted_rt:
                rt_str = _round_type_to_str.get(rt, rt.value)
                advance = _default_advance.get(rt_str, 8)
                round_configs.append(
                    RoundConfig(round_type=rt_str, advance_count=advance)
                )

            progression_results = simulate_event_progression(
                proj_inputs, rounds=round_configs, n_simulations=10_000
            )
            winner_name = progression_results[0].name if progression_results else None

            # Build template rows from ProgressionResult objects.
            rows = _build_progression_rows(progression_results, round_configs)
        else:
            probs = compute_podium_probabilities(proj_inputs, n_simulations=10_000)
            rows = _build_projection_rows(session, proj_inputs, probs)
            winner_id = predict_winner(proj_inputs)
            winner_name = next(
                (r["name"] for r in rows if r["athlete_id"] == winner_id), None
            )
            progression_results = None
            round_configs = []

    return templates.TemplateResponse(
        request,
        "projections.html",
        {
            "event": {
                "id": event.id,
                "name": event.name,
                "season": event.season,
                "tier": event.tier.value.replace("_", " ").title(),
            },
            "discipline": _DISCIPLINE_DISPLAY.get(
                event.discipline, event.discipline.value
            ),
            "gender": gender_enum.value,
            "available_genders": available_genders,
            "rows": rows,
            "winner": winner_name,
            "use_progression": use_progression,
            "round_configs": [{"round_type": rc.round_type} for rc in round_configs]
            if round_configs
            else [],
            "error": None,
        },
    )


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
_MAX_ATHLETES_PER_PROJECTION_CARD = 64


# ---------------------------------------------------------------------------
# GET /head-to-head        — autocomplete form
# GET /head-to-head/{a}/{b} — result page
# ---------------------------------------------------------------------------


@router.get("/head-to-head", response_class=HTMLResponse)
async def head_to_head_form(request: Request):
    """Render the head-to-head athlete selection form."""
    templates = request.app.state.templates
    with _get_session() as session:
        athletes = list(
            session.execute(select(Athlete).order_by(Athlete.name)).scalars()
        )
        athlete_list = [
            {"id": a.id, "name": a.name, "gender": a.gender.value} for a in athletes
        ]
    return templates.TemplateResponse(
        request,
        "head_to_head_new.html",
        {
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
        },
    )


@router.get("/head-to-head/{a_id}/{b_id}", response_class=HTMLResponse)
async def head_to_head_result(
    request: Request,
    a_id: int,
    b_id: int,
    discipline: str = Query(default="lead"),
):
    """Return a head-to-head comparison page for two athletes."""
    from climbing_elo.engine.elo import expected_score as _expected_score

    templates = request.app.state.templates

    # --- validate inputs ---
    if a_id == b_id:
        return HTMLResponse(
            "Cannot compare an athlete against themselves.", status_code=400
        )

    disc = _DISCIPLINE_FROM_STR.get(discipline)
    if disc is None:
        return HTMLResponse(
            f"Invalid discipline '{discipline}'. Must be one of: lead, boulder, speed, combined.",
            status_code=400,
        )

    with _get_session() as session:
        athlete_a = session.get(Athlete, a_id)
        athlete_b = session.get(Athlete, b_id)

        if not athlete_a:
            return HTMLResponse(f"Athlete {a_id} not found.", status_code=404)
        if not athlete_b:
            return HTMLResponse(f"Athlete {b_id} not found.", status_code=404)

        rating_a = session.execute(
            select(Rating).where(
                Rating.athlete_id == a_id,
                Rating.discipline == disc,
            )
        ).scalar_one_or_none()
        rating_b = session.execute(
            select(Rating).where(
                Rating.athlete_id == b_id,
                Rating.discipline == disc,
            )
        ).scalar_one_or_none()

        if rating_a is None:
            return HTMLResponse(
                f"{athlete_a.name} has no {_DISCIPLINE_DISPLAY[disc]} rating.",
                status_code=404,
            )
        if rating_b is None:
            return HTMLResponse(
                f"{athlete_b.name} has no {_DISCIPLINE_DISPLAY[disc]} rating.",
                status_code=404,
            )

        # --- win probability (analytic) ---
        win_a = _expected_score(rating_a.mu, rating_b.mu)
        win_b = 1.0 - win_a

        # --- past meetings ---
        # Events in which both athletes have a Result (for this discipline)
        shared_subq = (
            select(Round.event_id)
            .join(Event, Round.event_id == Event.id)
            .join(Result, Result.round_id == Round.id)
            .where(
                Event.discipline == disc,
                Result.athlete_id.in_([a_id, b_id]),
            )
            .group_by(Round.event_id)
            .having(func.count(func.distinct(Result.athlete_id)) == 2)
            .subquery()
        )

        shared_events = list(
            session.execute(
                select(Event)
                .join(shared_subq, Event.id == shared_subq.c.event_id)
                .order_by(Event.start_date.desc())
            ).scalars()
        )
        past_meetings = len(shared_events)
        most_recent_shared_event = shared_events[0] if shared_events else None

        # --- rating history for chart ---
        def _load_history(athlete_id: int) -> tuple[list[str], list[float]]:
            # Cap at most-recent 200 events so the chart query is bounded.
            # Use desc + reverse to get the tail of the timeline, not the head.
            rows = list(
                session.execute(
                    select(RatingHistory, Event)
                    .join(Event, RatingHistory.event_id == Event.id)
                    .where(
                        RatingHistory.athlete_id == athlete_id,
                        Event.discipline == disc,
                    )
                    .order_by(Event.start_date.desc())
                    .limit(200)
                ).all()
            )
            rows.reverse()
            labels = [f"{ev.name} ({ev.season})" for _rh, ev in rows]
            mus = [round(rh.mu_after, 1) for rh, _ev in rows]
            return labels, mus

        labels_a, mus_a = _load_history(a_id)
        labels_b, mus_b = _load_history(b_id)

        # Merge label sets for the shared time axis
        all_labels = sorted(set(labels_a) | set(labels_b))

        # Map each athlete's history into the merged axis (None for gaps)
        def _align(
            labels: list[str], mus: list[float], all_lbl: list[str]
        ) -> list[float | None]:
            lmap = dict(zip(labels, mus))
            return [lmap.get(lbl) for lbl in all_lbl]

        aligned_a = _align(labels_a, mus_a, all_labels)
        aligned_b = _align(labels_b, mus_b, all_labels)

    return templates.TemplateResponse(
        request,
        "head_to_head.html",
        {
            "athlete_a": {
                "id": athlete_a.id,
                "name": athlete_a.name,
                "nationality": athlete_a.nationality or "—",
                "gender": athlete_a.gender.value,
                "mu": round(rating_a.mu, 1),
                "sigma": round(rating_a.sigma, 1),
                "n_events": rating_a.n_events,
            },
            "athlete_b": {
                "id": athlete_b.id,
                "name": athlete_b.name,
                "nationality": athlete_b.nationality or "—",
                "gender": athlete_b.gender.value,
                "mu": round(rating_b.mu, 1),
                "sigma": round(rating_b.sigma, 1),
                "n_events": rating_b.n_events,
            },
            "discipline_key": discipline,
            "discipline_label": _DISCIPLINE_DISPLAY[disc],
            "win_a": round(win_a * 100, 1),
            "win_b": round(win_b * 100, 1),
            "past_meetings": past_meetings,
            "most_recent_shared_event": (
                {
                    "id": most_recent_shared_event.id,
                    "name": most_recent_shared_event.name,
                    "season": most_recent_shared_event.season,
                }
                if most_recent_shared_event
                else None
            ),
            # Pass raw Python lists; template uses |tojson for safe HTML escaping.
            "chart_labels": all_labels,
            "chart_mu_a": aligned_a,
            "chart_mu_b": aligned_b,
            "disciplines": [
                {"key": k, "label": v}
                for k, v in [
                    ("lead", "Lead"),
                    ("boulder", "Boulder"),
                    ("speed", "Speed"),
                    ("combined", "Combined"),
                ]
            ],
        },
    )


# ---------------------------------------------------------------------------
# GET /live/{event_id}  — live event view (auto-updating leaderboard + projections)
# ---------------------------------------------------------------------------


@router.get("/live/{event_id}", response_class=HTMLResponse)
async def live_event_view(request: Request, event_id: int, gender: str = "M"):
    """Render the live event page for a given event_id.

    Loads event metadata, current leaderboard state (Result rows ordered by rank),
    and computes initial projections.  The page then subscribes to
    /live/{event_id}/stream via JavaScript EventSource for real-time updates.
    """
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

        available_genders = sorted({rnd.gender.value for rnd in event.rounds})

        # Build current leaderboard: all results ordered by (round_type, rank)
        # We want the most recent / highest round first (final > semi > qual).
        _rt_order = {
            RoundType.FINAL: 0,
            RoundType.SEMI: 1,
            RoundType.QUALIFICATION: 2,
        }
        sorted_rounds = sorted(
            [r for r in event.rounds if r.gender == gender_enum],
            key=lambda r: _rt_order.get(r.round_type, 99),
        )

        # Collect the current state per athlete — prefer the latest round result.
        # Maps athlete_id → {rank, score, round_type, name, athlete_id}
        athlete_best: dict[int, dict] = {}
        for rnd in sorted_rounds:
            results = list(
                session.execute(
                    select(Result, Athlete)
                    .join(Athlete, Result.athlete_id == Athlete.id)
                    .where(Result.round_id == rnd.id, Result.dns.is_(False))
                    .order_by(Result.rank.asc())
                ).all()
            )
            for res, athlete in results:
                if athlete.id not in athlete_best:
                    athlete_best[athlete.id] = {
                        "athlete_id": athlete.id,
                        "name": athlete.name,
                        "rank": res.rank,
                        "score": res.raw_score or "—",
                        "round_type": rnd.round_type.value,
                    }

        leaderboard_rows = sorted(
            athlete_best.values(),
            key=lambda r: r["rank"] if r["rank"] is not None else 9999,
        )

        # Build projection inputs from leaderboard.
        # Athletes with a finished rank are "completed"; the rest are "remaining".
        # For this initial render all athletes in DB are "completed" since we only
        # have stored results (not live mid-round state).  We use
        # compute_partial_event_probabilities for correctness.
        completed: list[tuple[AthleteProjectionInput, int]] = []
        remaining: list[AthleteProjectionInput] = []

        for row in leaderboard_rows:
            aid = row["athlete_id"]
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
            athlete_obj = session.get(Athlete, aid)
            inp = AthleteProjectionInput(
                athlete_id=aid,
                mu=mu,
                sigma=sigma,
                name=athlete_obj.name if athlete_obj else str(aid),
            )
            if row["rank"] is not None:
                completed.append((inp, row["rank"]))
            else:
                remaining.append(inp)

        projection_rows: list[dict] = []
        if completed or remaining:
            probs = compute_partial_event_probabilities(
                completed_athletes=completed,
                remaining_athletes=remaining,
                n_simulations=10_000,
            )
            all_inputs = [inp for inp, _ in completed] + remaining
            for inp in sorted(
                all_inputs, key=lambda a: probs[a.athlete_id]["expected_rank"]
            ):
                p = probs[inp.athlete_id]
                projection_rows.append(
                    {
                        "athlete_id": inp.athlete_id,
                        "name": inp.name,
                        "mu": round(inp.mu, 1),
                        "win": f"{p['win'] * 100:.1f}%",
                        "podium": f"{p['podium'] * 100:.1f}%",
                        "expected_rank": p["expected_rank"],
                        "win_raw": p["win"],
                        "podium_raw": p["podium"],
                        "is_completed": inp.athlete_id in {aid for inp, _ in completed},
                    }
                )
            for i, row in enumerate(projection_rows):
                row["proj_rank"] = i + 1

    return templates.TemplateResponse(
        request,
        "live.html",
        {
            "event": {
                "id": event.id,
                "name": event.name,
                "season": event.season,
                "tier": event.tier.value.replace("_", " ").title(),
                "discipline": _DISCIPLINE_DISPLAY.get(
                    event.discipline, event.discipline.value
                ),
            },
            "gender": gender_enum.value,
            "available_genders": available_genders,
            "leaderboard": leaderboard_rows,
            "projections": projection_rows,
            "stream_url": f"/live/{event_id}/stream",
            # Initial athlete data for JS to seed the leaderboard state
            # (serialised to JSON in the template via |tojson)
            "initial_athletes": {
                row["athlete_id"]: {
                    "name": row["name"],
                    "rank": row["rank"],
                    "score": row["score"],
                    "round_type": row["round_type"],
                }
                for row in leaderboard_rows
            },
        },
    )


@router.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request):
    """List upcoming World Climbing (formerly IFSC) events with ELO-based outcome predictions.

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

                # Flag set to True when we use the likely-competitor fallback
                # so the template can surface a disclaimer.
                from_likely_roster = False

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
                                select(Result).where(
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
                                from climbing_elo.engine.elo import (
                                    DEFAULT_MU,
                                    DEFAULT_SIGMA,
                                )

                                mu, sigma = DEFAULT_MU, DEFAULT_SIGMA
                            else:
                                mu, sigma = rating.mu, rating.sigma
                            proj_inputs.append(
                                AthleteProjectionInput(
                                    athlete_id=aid,
                                    mu=mu,
                                    sigma=sigma,
                                    name=athlete.name,
                                )
                            )

                        # Cap per-event athlete count for the landing page Monte Carlo,
                        # so a 200-athlete qualification field doesn't make the page hang.
                        # Sort by mu descending and take the top N — the predicted podium
                        # comes from this group anyway.
                        if len(proj_inputs) > _MAX_ATHLETES_PER_PROJECTION_CARD:
                            proj_inputs = sorted(
                                proj_inputs, key=lambda a: a.mu, reverse=True
                            )[:_MAX_ATHLETES_PER_PROJECTION_CARD]

                        # Cache key encodes event, discipline, gender, and the
                        # sorted athlete+rating fingerprint so stale ratings
                        # don't silently persist. TTL=1h handles staleness;
                        # callers can call predictions_cache.clear() after a
                        # scrape run for immediate freshness.
                        _athlete_fingerprint = ":".join(
                            f"{a.athlete_id},{a.mu:.2f},{a.sigma:.2f}"
                            for a in sorted(proj_inputs, key=lambda a: a.athlete_id)
                        )
                        _cache_key = (
                            f"projections:event:{ev.id}"
                            f":disc:{disc_enum.value}"
                            f":gender:{gender_enum.value}"
                            f":athletes:{_athlete_fingerprint}"
                        )

                        probs = predictions_cache.get(_cache_key)
                        if probs is None:
                            probs = compute_podium_probabilities(
                                proj_inputs, n_simulations=10_000
                            )
                            predictions_cache.set(_cache_key, probs)

                        # Build predicted top-3 sorted by expected_rank.
                        ranked = sorted(
                            proj_inputs,
                            key=lambda a: probs[a.athlete_id]["expected_rank"],
                        )
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
                        gender_predictions.append(
                            {
                                "gender": gender_enum.value,
                                "top3": top3,
                                "total_athletes": len(proj_inputs),
                            }
                        )

                    predictions_data = {"genders": gender_predictions}

                else:
                    # No registered athletes yet — fall back to likely-competitor
                    # roster derived from season attendance.
                    gender_predictions_fallback: list[dict] = []
                    for gender_enum in [Gender.M, Gender.F]:
                        # Check cache first.
                        _roster_cache_key = (
                            f"roster:{disc_enum.value}:{ev.season}:{gender_enum.value}"
                        )
                        athlete_ids = likely_roster_cache.get(_roster_cache_key)
                        if athlete_ids is None:
                            athlete_ids = likely_competitors(
                                session,
                                disc_enum,
                                ev.season,
                                gender_enum,
                            )
                            likely_roster_cache.set(_roster_cache_key, athlete_ids)

                        if not athlete_ids:
                            continue

                        # Cap and build projection inputs.
                        proj_inputs_fallback: list[AthleteProjectionInput] = []
                        for aid in athlete_ids[:_MAX_ATHLETES_PER_PROJECTION_CARD]:
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
                                from climbing_elo.engine.elo import (
                                    DEFAULT_MU,
                                    DEFAULT_SIGMA,
                                )

                                mu, sigma = DEFAULT_MU, DEFAULT_SIGMA
                            else:
                                mu, sigma = rating.mu, rating.sigma
                            proj_inputs_fallback.append(
                                AthleteProjectionInput(
                                    athlete_id=aid,
                                    mu=mu,
                                    sigma=sigma,
                                    name=athlete.name,
                                )
                            )

                        if len(proj_inputs_fallback) < 2:
                            continue

                        _athlete_fingerprint_fb = ":".join(
                            f"{a.athlete_id},{a.mu:.2f},{a.sigma:.2f}"
                            for a in sorted(
                                proj_inputs_fallback, key=lambda a: a.athlete_id
                            )
                        )
                        _cache_key_fb = (
                            f"projections:likely:{ev.id}"
                            f":disc:{disc_enum.value}"
                            f":gender:{gender_enum.value}"
                            f":athletes:{_athlete_fingerprint_fb}"
                        )
                        probs_fb = predictions_cache.get(_cache_key_fb)
                        if probs_fb is None:
                            probs_fb = compute_podium_probabilities(
                                proj_inputs_fallback, n_simulations=10_000
                            )
                            predictions_cache.set(_cache_key_fb, probs_fb)

                        ranked_fb = sorted(
                            proj_inputs_fallback,
                            key=lambda a: probs_fb[a.athlete_id]["expected_rank"],
                        )
                        top3_fb = [
                            {
                                "athlete_id": a.athlete_id,
                                "name": a.name,
                                "win": f"{probs_fb[a.athlete_id]['win'] * 100:.1f}%",
                                "podium": f"{probs_fb[a.athlete_id]['podium'] * 100:.1f}%",
                                "expected_rank": probs_fb[a.athlete_id][
                                    "expected_rank"
                                ],
                            }
                            for a in ranked_fb[:3]
                        ]
                        gender_predictions_fallback.append(
                            {
                                "gender": gender_enum.value,
                                "top3": top3_fb,
                                "total_athletes": len(proj_inputs_fallback),
                            }
                        )

                    if gender_predictions_fallback:
                        predictions_data = {"genders": gender_predictions_fallback}
                        from_likely_roster = True

                disc_events.append(
                    {
                        "id": ev.id,
                        "name": ev.name,
                        "season": ev.season,
                        "tier": ev.tier.value.replace("_", " ").title(),
                        "date": str(ev.start_date),
                        "has_athletes": has_athletes,
                        "predictions": predictions_data,
                        "from_likely_roster": from_likely_roster,
                    }
                )

            grouped.append(
                {
                    "key": disc_key,
                    "label": disc_label,
                    "events": disc_events,
                    "has_data": len(disc_events) > 0,
                }
            )

    return templates.TemplateResponse(
        request,
        "predictions.html",
        {
            "grouped": grouped,
            "today": str(today),
        },
    )
