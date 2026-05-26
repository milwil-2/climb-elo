"""v2 HTML routes — monochrome dashboard redesign (Issue #40).

Mounted at /v2/. All routes produce HTML via templates_v2/ Jinja templates.
The original dashboard at / (routes.py) remains untouched.
"""

from __future__ import annotations

import json
import math
from datetime import date

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select

from climbing_elo.database import get_session_factory
from climbing_elo.engine.elo import expected_score as _expected_score
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_podium_probabilities,
)
from climbing_elo.cache import predictions_cache
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

router = APIRouter(prefix="/v2")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR_NAME = "templates_v2"


def _templates(request: Request):
    """Return Jinja2Templates pointed at templates_v2/."""
    from fastapi.templating import Jinja2Templates
    from pathlib import Path

    d = Path(__file__).resolve().parent.parent / _TEMPLATES_DIR_NAME
    return Jinja2Templates(directory=str(d))


def _session():
    factory = get_session_factory()
    return factory()


# Discipline display mapping
_DISC_LABEL = {
    Discipline.LEAD: "Lead",
    Discipline.BOULDER: "Boulder",
    Discipline.SPEED: "Speed",
    Discipline.BOULDER_LEAD: "Boulder + Lead",
}

_DISC_KEY_TO_ENUM = {
    "B": Discipline.BOULDER,
    "L": Discipline.LEAD,
    "S": Discipline.SPEED,
    "BL": Discipline.BOULDER_LEAD,
}

_GENDER_LABEL = {
    Gender.M: "Men",
    Gender.F: "Women",
}


def _get_rankings_v2(session, gender: Gender, discipline: Discipline, limit: int = 200):
    """Return a ranked list of athletes for a given discipline/gender."""
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
            "year_of_birth": athlete.year_of_birth,
            "mu": round(rating.mu, 1),
            "sigma": round(rating.sigma, 1),
            "n_events": rating.n_events,
            "provisional": rating.provisional,
            "last_event_at": rating.last_event_at,
        }
        for i, (rating, athlete) in enumerate(rows)
    ]


def _get_90d_delta(session, athlete_id: int, discipline: Discipline) -> float:
    """Return the rating delta over the last ~90 days (3 most-recent events)."""
    rows = list(
        session.execute(
            select(RatingHistory, Event)
            .join(Event, RatingHistory.event_id == Event.id)
            .where(
                RatingHistory.athlete_id == athlete_id,
                Event.discipline == discipline,
            )
            .order_by(Event.start_date.desc())
            .limit(3)
        ).all()
    )
    if not rows:
        return 0.0
    # Δ = latest mu_after - earliest mu_before in this window
    latest_rh = rows[0][0]
    earliest_rh = rows[-1][0]
    return round(latest_rh.mu_after - earliest_rh.mu_before, 1)


def _ticker_context(session) -> dict:
    """Build the context dict for the sticky ticker.

    Returns:
        live_event: None (TODO: populate when DB has a live_event status field)
        ticker_items: list of {kind, text, delta?, tag?} dicts

    TODO: When the DB gains an Event.status field, replace the None here with
    a query for Event objects where status == 'live'.
    """
    live_event = None  # TODO: query for live events when Event.status field exists

    ticker_items = []

    # Top 10 recent rating movers (last 7 days worth of RatingHistory rows)
    try:
        recent_histories = list(
            session.execute(
                select(RatingHistory, Athlete, Event)
                .join(Athlete, RatingHistory.athlete_id == Athlete.id)
                .join(Event, RatingHistory.event_id == Event.id)
                .order_by(RatingHistory.id.desc())
                .limit(30)
            ).all()
        )

        # Pick athletes with the largest |delta|, de-dup by athlete
        seen_athletes: set[int] = set()
        movers = []
        for rh, athlete, event in recent_histories:
            if athlete.id in seen_athletes:
                continue
            seen_athletes.add(athlete.id)
            delta = round(rh.mu_after - rh.mu_before, 0)
            if delta != 0:
                movers.append((athlete.name, int(delta)))
            if len(movers) >= 10:
                break

        # Sort by abs delta descending
        movers.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, delta in movers[:10]:
            ticker_items.append(
                {
                    "kind": "delta",
                    "name": name,
                    "delta": delta,
                }
            )
    except Exception:
        pass

    # Next 3 upcoming events
    try:
        today = date.today()
        upcoming = list(
            session.execute(
                select(Event)
                .where(Event.start_date > today)
                .order_by(Event.start_date.asc())
                .limit(3)
            ).scalars()
        )
        for ev in upcoming:
            days = (ev.start_date - today).days
            ticker_items.append(
                {
                    "kind": "upcoming",
                    "name": ev.name,
                    "days": days,
                    "tag": f"In {days}d",
                }
            )
    except Exception:
        pass

    return {
        "live_event": live_event,
        "ticker_items": ticker_items,
    }


def _nav_context(active_page: str) -> dict:
    """Return navigation context (which page is active)."""
    return {"active_page": active_page}


# ---------------------------------------------------------------------------
# GET /v2/  — Landing page
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def v2_landing(request: Request):
    t = _templates(request)

    with _session() as session:
        # Top 8 athletes by mu in Boulder, Men (default)
        men_boulder = _get_rankings_v2(session, Gender.M, Discipline.BOULDER, limit=8)
        women_boulder = _get_rankings_v2(session, Gender.F, Discipline.BOULDER, limit=8)
        men_lead = _get_rankings_v2(session, Gender.M, Discipline.LEAD, limit=8)
        women_lead = _get_rankings_v2(session, Gender.F, Discipline.LEAD, limit=8)
        men_speed = _get_rankings_v2(session, Gender.M, Discipline.SPEED, limit=8)
        women_speed = _get_rankings_v2(session, Gender.F, Discipline.SPEED, limit=8)
        men_bl = _get_rankings_v2(session, Gender.M, Discipline.BOULDER_LEAD, limit=8)
        women_bl = _get_rankings_v2(session, Gender.F, Discipline.BOULDER_LEAD, limit=8)

        # App metrics
        total_athletes = session.execute(select(func.count(Athlete.id))).scalar_one()
        total_events = session.execute(select(func.count(Event.id))).scalar_one()
        total_ratings = session.execute(
            select(func.count(RatingHistory.id))
        ).scalar_one()

        ticker = _ticker_context(session)

    ctx = {
        "leaderboard": {
            "B": {"men": men_boulder, "women": women_boulder},
            "L": {"men": men_lead, "women": women_lead},
            "S": {"men": men_speed, "women": women_speed},
            "BL": {"men": men_bl, "women": women_bl},
        },
        "stats": {
            "total_athletes": total_athletes,
            "total_events": total_events,
            "total_ratings": total_ratings,
        },
        **ticker,
        **_nav_context("landing"),
    }

    return t.TemplateResponse(request, "landing.html", ctx)


# ---------------------------------------------------------------------------
# GET /v2/leaderboard
# ---------------------------------------------------------------------------


@router.get("/leaderboard", response_class=HTMLResponse)
async def v2_leaderboard(
    request: Request,
    disc: str = Query(default="B"),
    gender: str = Query(default="M"),
):
    t = _templates(request)

    disc_enum = _DISC_KEY_TO_ENUM.get(disc.upper(), Discipline.BOULDER)
    gender_enum = Gender.M if gender.upper() == "M" else Gender.F

    with _session() as session:
        rows = _get_rankings_v2(session, gender_enum, disc_enum, limit=20)
        ticker = _ticker_context(session)

        # Add 90d delta to each row
        for row in rows:
            row["delta_90d"] = _get_90d_delta(session, row["id"], disc_enum)

        # Summary stats
        all_rows = _get_rankings_v2(session, gender_enum, disc_enum, limit=200)
        top_mu = all_rows[0]["mu"] if all_rows else 0
        mid_idx = len(all_rows) // 2
        median_mu = all_rows[mid_idx]["mu"] if all_rows else 0
        total_count = len(all_rows)

    disc_label = _DISC_LABEL.get(disc_enum, disc)
    gender_label = _GENDER_LABEL.get(gender_enum, gender)

    ctx = {
        "rows": rows,
        "disc": disc.upper(),
        "disc_label": disc_label,
        "gender": gender_enum.value,
        "gender_label": gender_label,
        "top_mu": top_mu,
        "median_mu": median_mu,
        "total_count": total_count,
        **ticker,
        **_nav_context("leaderboard"),
    }
    return t.TemplateResponse(request, "leaderboard.html", ctx)


# ---------------------------------------------------------------------------
# GET /v2/athletes  — redirect to first athlete in default discipline
# ---------------------------------------------------------------------------


@router.get("/athletes", response_class=HTMLResponse)
async def v2_athletes_index(request: Request):
    with _session() as session:
        first = session.execute(
            select(Athlete).order_by(Athlete.id.asc()).limit(1)
        ).scalar_one_or_none()

    if first:
        return RedirectResponse(url=f"/v2/athletes/{first.id}", status_code=302)

    t = _templates(request)
    with _session() as session:
        ticker = _ticker_context(session)
    return t.TemplateResponse(
        request,
        "athletes.html",
        {
            "athlete": None,
            "sidebar_athletes": [],
            **ticker,
            **_nav_context("athletes"),
        },
    )


# ---------------------------------------------------------------------------
# GET /v2/athletes/{athlete_id}  — athlete profile
# ---------------------------------------------------------------------------


@router.get("/athletes/{athlete_id}", response_class=HTMLResponse)
async def v2_athlete_profile(request: Request, athlete_id: int):
    t = _templates(request)

    with _session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            return HTMLResponse("Athlete not found", status_code=404)

        # All ratings for this athlete
        ratings_rows = list(
            session.execute(
                select(Rating).where(Rating.athlete_id == athlete_id)
            ).scalars()
        )
        ratings_by_disc: dict[str, dict] = {}
        for r in ratings_rows:
            key = {
                Discipline.BOULDER: "B",
                Discipline.LEAD: "L",
                Discipline.SPEED: "S",
                Discipline.BOULDER_LEAD: "BL",
            }.get(r.discipline, r.discipline.value)
            ratings_by_disc[key] = {
                "mu": round(r.mu, 1),
                "sigma": round(r.sigma, 1),
                "n_events": r.n_events,
            }

        # Primary rating = best by mu (prefer Lead then Boulder then Speed)
        pref_order = ["L", "B", "S", "BL"]
        primary_disc_key = next((k for k in pref_order if k in ratings_by_disc), None)
        primary_rating = ratings_by_disc.get(
            primary_disc_key or "L", {"mu": None, "sigma": None}
        )
        primary_disc_label = _DISC_LABEL.get(
            _DISC_KEY_TO_ENUM.get(primary_disc_key or "L", Discipline.LEAD), "Lead"
        )

        # Rating history for chart (use primary discipline)
        primary_disc_enum = _DISC_KEY_TO_ENUM.get(
            primary_disc_key or "L", Discipline.LEAD
        )
        history_rows = list(
            session.execute(
                select(RatingHistory, Event)
                .join(Event, RatingHistory.event_id == Event.id)
                .where(
                    RatingHistory.athlete_id == athlete_id,
                    Event.discipline == primary_disc_enum,
                )
                .order_by(Event.start_date.asc())
            ).all()
        )

        # De-dup by event (keep last round per event)
        event_last: dict[int, tuple] = {}
        for rh, ev in history_rows:
            event_last[ev.id] = (rh, ev)

        chart_labels = []
        chart_mu = []
        for rh, ev in event_last.values():
            chart_labels.append(str(ev.start_date))
            chart_mu.append(round(rh.mu_after, 1))

        # Recent events (last 5 across all disciplines)
        recent_rh = list(
            session.execute(
                select(RatingHistory, Event)
                .join(Event, RatingHistory.event_id == Event.id)
                .where(RatingHistory.athlete_id == athlete_id)
                .order_by(Event.start_date.desc())
                .limit(20)
            ).all()
        )
        seen_ev: set[int] = set()
        recent_events = []
        for rh, ev in recent_rh:
            if ev.id in seen_ev:
                continue
            seen_ev.add(ev.id)
            # Best place in this event
            place_row = session.execute(
                select(func.min(Result.rank))
                .join(Round, Result.round_id == Round.id)
                .where(
                    Round.event_id == ev.id,
                    Result.athlete_id == athlete_id,
                    Result.dns.is_(False),
                    Result.rank.is_not(None),
                )
            ).scalar_one_or_none()

            delta = round(rh.mu_after - rh.mu_before, 1)
            disc_display = {
                Discipline.BOULDER: "Boulder",
                Discipline.LEAD: "Lead",
                Discipline.SPEED: "Speed",
                Discipline.BOULDER_LEAD: "B+L",
            }.get(ev.discipline, ev.discipline.value)

            recent_events.append(
                {
                    "event_id": ev.id,
                    "event_name": ev.name,
                    "date": str(ev.start_date),
                    "discipline": disc_display,
                    "place": place_row,
                    "delta": delta,
                    "delta_sign": "+" if delta > 0 else ("−" if delta < 0 else ""),
                    "delta_abs": abs(delta),
                }
            )
            if len(recent_events) >= 5:
                break

        # Sidebar: top 14 athletes in primary discipline, same gender
        sidebar_athletes = _get_rankings_v2(
            session, athlete.gender, primary_disc_enum, limit=14
        )

        ticker = _ticker_context(session)

    ctx = {
        "athlete": {
            "id": athlete.id,
            "name": athlete.name,
            "nationality": athlete.nationality or "—",
            "year_of_birth": athlete.year_of_birth,
            "gender": athlete.gender.value,
        },
        "primary_rating": primary_rating,
        "primary_disc_label": primary_disc_label,
        "primary_disc_key": primary_disc_key or "L",
        "ratings_by_disc": ratings_by_disc,
        "chart_labels": json.dumps(chart_labels),
        "chart_mu": json.dumps(chart_mu),
        "recent_events": recent_events,
        "sidebar_athletes": sidebar_athletes,
        "current_athlete_id": athlete_id,
        **ticker,
        **_nav_context("athletes"),
    }
    return t.TemplateResponse(request, "athletes.html", ctx)


# ---------------------------------------------------------------------------
# GET /v2/projections
# ---------------------------------------------------------------------------


@router.get("/projections", response_class=HTMLResponse)
async def v2_projections(request: Request):
    t = _templates(request)

    today = date.today()

    with _session() as session:
        # Get up to 4 upcoming events across disciplines
        upcoming_events = []
        for disc_enum in [Discipline.LEAD, Discipline.BOULDER, Discipline.SPEED]:
            evs = list(
                session.execute(
                    select(Event)
                    .where(
                        Event.discipline == disc_enum,
                        Event.start_date > today,
                    )
                    .order_by(Event.start_date.asc())
                    .limit(2)
                ).scalars()
            )
            for ev in evs:
                upcoming_events.append(ev)
                if len(upcoming_events) >= 4:
                    break
            if len(upcoming_events) >= 4:
                break

        proj_cards = []
        for ev in upcoming_events[:4]:
            days_until = (ev.start_date - today).days

            # Get top athletes for this discipline (both genders for display)
            top_athletes_m = _get_rankings_v2(
                session, Gender.M, ev.discipline, limit=20
            )
            top_athletes_f = _get_rankings_v2(
                session, Gender.F, ev.discipline, limit=20
            )

            # Run projections for men
            proj_rows_m = []
            if top_athletes_m:
                inputs_m = [
                    AthleteProjectionInput(
                        athlete_id=a["id"], mu=a["mu"], sigma=a["sigma"], name=a["name"]
                    )
                    for a in top_athletes_m[:10]
                ]
                _cache_key = f"v2:proj:{ev.id}:M:{ev.discipline.value}"
                probs_m = predictions_cache.get(_cache_key)
                if probs_m is None:
                    probs_m = compute_podium_probabilities(
                        inputs_m, n_simulations=10_000
                    )
                    predictions_cache.set(_cache_key, probs_m)

                sorted_m = sorted(
                    inputs_m, key=lambda a: probs_m[a.athlete_id]["expected_rank"]
                )
                max_p = max(
                    (probs_m[a.athlete_id]["podium"] for a in sorted_m[:6]),
                    default=0.001,
                )
                for i, a in enumerate(sorted_m[:6]):
                    p = probs_m[a.athlete_id]["podium"]
                    proj_rows_m.append(
                        {
                            "rank": i + 1,
                            "name": a.name,
                            "pct_win": f"{probs_m[a.athlete_id]['win'] * 100:.1f}",
                            "pct_podium": f"{p * 100:.1f}",
                            "pct_top8": f"{probs_m[a.athlete_id]['top_8'] * 100:.1f}",
                            "p_podium_raw": p,
                            "bar_pct": min(100.0, (p / max_p) * 100)
                            if max_p > 0
                            else 0.0,
                        }
                    )

            disc_label = _DISC_LABEL.get(ev.discipline, ev.discipline.value)
            proj_cards.append(
                {
                    "event_id": ev.id,
                    "event_name": ev.name,
                    "date": str(ev.start_date),
                    "discipline": disc_label,
                    "discipline_key": {
                        Discipline.LEAD: "L",
                        Discipline.BOULDER: "B",
                        Discipline.SPEED: "S",
                        Discipline.BOULDER_LEAD: "BL",
                    }.get(ev.discipline, "L"),
                    "days_until": days_until,
                    "athletes_m": len(top_athletes_m),
                    "athletes_f": len(top_athletes_f),
                    "proj_rows": proj_rows_m,
                }
            )

        ticker = _ticker_context(session)

    ctx = {
        "proj_cards": proj_cards,
        **ticker,
        **_nav_context("projections"),
    }
    return t.TemplateResponse(request, "projections.html", ctx)


# ---------------------------------------------------------------------------
# GET /v2/head-to-head  — athlete selection form
# ---------------------------------------------------------------------------


@router.get("/head-to-head", response_class=HTMLResponse)
async def v2_h2h_form(request: Request):
    t = _templates(request)

    with _session() as session:
        # Load top athletes by discipline for the selects
        men_boulder = _get_rankings_v2(session, Gender.M, Discipline.BOULDER, limit=20)
        women_boulder = _get_rankings_v2(
            session, Gender.F, Discipline.BOULDER, limit=20
        )
        men_lead = _get_rankings_v2(session, Gender.M, Discipline.LEAD, limit=20)
        women_lead = _get_rankings_v2(session, Gender.F, Discipline.LEAD, limit=20)

        ticker = _ticker_context(session)

    ctx = {
        "pools": {
            "B": {"M": men_boulder, "F": women_boulder},
            "L": {"M": men_lead, "F": women_lead},
        },
        "h2h_result": None,
        **ticker,
        **_nav_context("h2h"),
    }
    return t.TemplateResponse(request, "head_to_head.html", ctx)


# ---------------------------------------------------------------------------
# GET /v2/head-to-head/{a_id}/{b_id}  — H2H result
# ---------------------------------------------------------------------------


@router.get("/head-to-head/{a_id}/{b_id}", response_class=HTMLResponse)
async def v2_h2h_result(
    request: Request,
    a_id: int,
    b_id: int,
    discipline: str = Query(default="L"),
):
    t = _templates(request)

    if a_id == b_id:
        return HTMLResponse(
            "Cannot compare an athlete against themselves.", status_code=400
        )

    disc_enum = _DISC_KEY_TO_ENUM.get(discipline.upper(), Discipline.LEAD)

    with _session() as session:
        athlete_a = session.get(Athlete, a_id)
        athlete_b = session.get(Athlete, b_id)

        if not athlete_a:
            return HTMLResponse(f"Athlete {a_id} not found.", status_code=404)
        if not athlete_b:
            return HTMLResponse(f"Athlete {b_id} not found.", status_code=404)

        rating_a = session.execute(
            select(Rating).where(
                Rating.athlete_id == a_id,
                Rating.discipline == disc_enum,
            )
        ).scalar_one_or_none()
        rating_b = session.execute(
            select(Rating).where(
                Rating.athlete_id == b_id,
                Rating.discipline == disc_enum,
            )
        ).scalar_one_or_none()

        if rating_a is None or rating_b is None:
            return HTMLResponse(
                "One or both athletes do not have ratings for this discipline.",
                status_code=404,
            )

        # Win probability (analytic ELO expectation)
        win_a = _expected_score(rating_a.mu, rating_b.mu)
        win_b = 1.0 - win_a

        # Shared events count
        shared_subq = (
            select(Round.event_id)
            .join(Event, Round.event_id == Event.id)
            .join(Result, Result.round_id == Round.id)
            .where(
                Event.discipline == disc_enum,
                Result.athlete_id.in_([a_id, b_id]),
            )
            .group_by(Round.event_id)
            .having(func.count(func.distinct(Result.athlete_id)) == 2)
            .subquery()
        )
        past_meetings = session.execute(
            select(func.count()).select_from(shared_subq)
        ).scalar_one()

        # Rating history for chart
        def _load_history(athlete_id: int) -> tuple[list[str], list[float]]:
            rows = list(
                session.execute(
                    select(RatingHistory, Event)
                    .join(Event, RatingHistory.event_id == Event.id)
                    .where(
                        RatingHistory.athlete_id == athlete_id,
                        Event.discipline == disc_enum,
                    )
                    .order_by(Event.start_date.desc())
                    .limit(50)
                ).all()
            )
            rows.reverse()
            labels = [str(ev.start_date) for _, ev in rows]
            mus = [round(rh.mu_after, 1) for rh, _ in rows]
            return labels, mus

        labels_a, mus_a = _load_history(a_id)
        labels_b, mus_b = _load_history(b_id)

        # Merge on shared label axis
        all_labels = sorted(set(labels_a) | set(labels_b))
        lmap_a = dict(zip(labels_a, mus_a))
        lmap_b = dict(zip(labels_b, mus_b))
        aligned_a = [lmap_a.get(lbl) for lbl in all_labels]
        aligned_b = [lmap_b.get(lbl) for lbl in all_labels]

        # Ring geometry: R=70, C=2πR
        R = 70
        C = 2 * math.pi * R
        dash_offset_a = round(C * (1 - win_a), 2)

        mu_gap = round(rating_a.mu - rating_b.mu, 1)

        # Pools for the selects
        gender_a = athlete_a.gender
        gender_b = athlete_b.gender
        pool_gender = gender_a if gender_a == gender_b else Gender.M
        pool = _get_rankings_v2(session, pool_gender, disc_enum, limit=20)

        ticker = _ticker_context(session)

    disc_label = _DISC_LABEL.get(disc_enum, discipline)

    ctx = {
        "h2h_result": {
            "athlete_a": {
                "id": athlete_a.id,
                "name": athlete_a.name,
                "nationality": athlete_a.nationality or "—",
                "mu": round(rating_a.mu, 1),
                "sigma": round(rating_a.sigma, 1),
                "n_events": rating_a.n_events,
            },
            "athlete_b": {
                "id": athlete_b.id,
                "name": athlete_b.name,
                "nationality": athlete_b.nationality or "—",
                "mu": round(rating_b.mu, 1),
                "sigma": round(rating_b.sigma, 1),
                "n_events": rating_b.n_events,
            },
            "win_a": round(win_a * 100, 1),
            "win_b": round(win_b * 100, 1),
            "win_a_frac": round(win_a, 4),
            "ring_R": R,
            "ring_C": round(C, 2),
            "ring_dash_offset": dash_offset_a,
            "past_meetings": past_meetings,
            "mu_gap": mu_gap,
            "disc_label": disc_label,
            "disc_key": discipline.upper(),
            "chart_labels": json.dumps(all_labels),
            "chart_mu_a": json.dumps(aligned_a),
            "chart_mu_b": json.dumps(aligned_b),
        },
        "pool": pool,
        "a_id": a_id,
        "b_id": b_id,
        "selected_disc": discipline.upper(),
        **ticker,
        **_nav_context("h2h"),
    }
    return t.TemplateResponse(request, "head_to_head.html", ctx)


# ---------------------------------------------------------------------------
# GET /v2/api  — API reference page
# ---------------------------------------------------------------------------


@router.get("/api", response_class=HTMLResponse)
async def v2_api_page(request: Request):
    t = _templates(request)

    endpoints = [
        {
            "method": "GET",
            "path": "/api/v1/leaderboard",
            "desc": "Paginated ELO rankings. Query: discipline, gender, limit, offset.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/athletes/{id}",
            "desc": "Athlete profile with all discipline ratings + recent events.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/athletes/{id}/history",
            "desc": "Rating-over-time history for charts. Query: discipline.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/athletes/{id}/combined",
            "desc": "Combined (Boulder+Lead) rating. 404 if not available.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/events",
            "desc": "Paginated event list. Query: discipline, season, limit, offset.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/events/{id}",
            "desc": "Event details with rounds and per-athlete pre/post ELO.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/combined/leaderboard",
            "desc": "Paginated Boulder+Lead combined leaderboard. Query: gender, limit, offset.",
            "status": "200",
        },
        {
            "method": "POST",
            "path": "/api/v1/projections",
            "desc": "Monte Carlo podium probabilities. Cached 1h.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/api/v1/predictions/upcoming",
            "desc": "Upcoming events with predicted top-3 per gender.",
            "status": "200",
        },
        {
            "method": "GET",
            "path": "/live/{event_id}/stream",
            "desc": "Server-Sent Events stream for in-progress competitions.",
            "status": "200",
        },
    ]

    with _session() as session:
        ticker = _ticker_context(session)

    ctx = {
        "endpoints": endpoints,
        **ticker,
        **_nav_context("api"),
    }
    return t.TemplateResponse(request, "api.html", ctx)
