"""HTML routes — monochrome dashboard.

Mounted at /. All routes produce HTML via templates/ Jinja templates.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from typing import Optional

from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import func, inspect as sa_inspect, or_, select

from climbing_elo.cache import (
    html_page_cache,
    likely_roster_cache,
    predictions_cache,
    ratings_fingerprint,
)
from climbing_elo.database import get_session_factory
from climbing_elo.engine.activity import (
    INACTIVE_THRESHOLD_MONTHS,
    RETIRED_THRESHOLD_YEARS,
    sigma_now,
)
from climbing_elo.engine.elo import expected_score as _expected_score
from climbing_elo.engine.event_status import (
    LIVE_WINDOW_DAYS,
    EventStatus,
    bulk_event_status,
    event_status,
)
from climbing_elo.engine.likely_roster import likely_competitors
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_partial_event_probabilities,
    compute_podium_probabilities,
)
from climbing_elo.live.livestream import youtube_embed_url
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventForecast,
    EventForecastScore,
    EventTier,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TEMPLATES_DIR_NAME = "templates"


def _templates(request: Request):
    """Return Jinja2Templates pointed at templates/."""
    from fastapi.templating import Jinja2Templates

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

_DISC_ENUM_TO_KEY = {v: k for k, v in _DISC_KEY_TO_ENUM.items()}

# Accept both short codes ("L", "B", "S", "BL") and long names
# ("lead", "boulder", "speed", "boulder_lead") in query params.
_DISC_FULL_NAME_TO_ENUM = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
    "boulder_lead": Discipline.BOULDER_LEAD,
    "bl": Discipline.BOULDER_LEAD,
}


def _parse_discipline(value: str) -> Discipline | None:
    """Resolve a discipline query param to its enum, or return None if invalid."""
    if not value:
        return None
    short = _DISC_KEY_TO_ENUM.get(value.upper())
    if short is not None:
        return short
    return _DISC_FULL_NAME_TO_ENUM.get(value.lower())


_GENDER_LABEL = {
    Gender.M: "Men",
    Gender.F: "Women",
}


def _build_proj_inputs_batched(
    session,
    athlete_ids: list[int],
    disc_enum: Discipline,
) -> list[AthleteProjectionInput]:
    """Build AthleteProjectionInput list with a single batched join query.

    Replaces the per-athlete ``session.get(Athlete, aid)`` +
    ``select(Rating)`` N+1 loop with one join query covering all athlete_ids
    at once.  Athletes missing a rating receive the ELO defaults so they still
    appear in projections.
    """
    from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA

    if not athlete_ids:
        return []

    rows = session.execute(
        select(Athlete, Rating)
        .join(Rating, Rating.athlete_id == Athlete.id)
        .where(
            Athlete.id.in_(athlete_ids),
            Rating.discipline == disc_enum,
        )
    ).all()
    athletes_by_id: dict[int, tuple[Athlete, Rating]] = {a.id: (a, r) for a, r in rows}

    # Fallback: fetch athletes that have no rating for this discipline.
    ids_missing_rating = [aid for aid in athlete_ids if aid not in athletes_by_id]
    athletes_no_rating: dict[int, Athlete] = {}
    if ids_missing_rating:
        for ath in session.execute(
            select(Athlete).where(Athlete.id.in_(ids_missing_rating))
        ).scalars():
            athletes_no_rating[ath.id] = ath

    proj_inputs: list[AthleteProjectionInput] = []
    for aid in athlete_ids:
        if aid in athletes_by_id:
            ath, rating = athletes_by_id[aid]
            proj_inputs.append(
                AthleteProjectionInput(
                    athlete_id=aid,
                    mu=rating.mu,
                    sigma=sigma_now(rating.sigma, rating.last_event_at),
                    name=ath.name,
                )
            )
        elif aid in athletes_no_rating:
            ath = athletes_no_rating[aid]
            proj_inputs.append(
                AthleteProjectionInput(
                    athlete_id=aid,
                    mu=DEFAULT_MU,
                    sigma=DEFAULT_SIGMA,
                    name=ath.name,
                )
            )
        # else: unknown athlete_id — skip silently (same as before)
    return proj_inputs


#: Valid values for the leaderboard ``view`` query param (#91).
LEADERBOARD_VIEWS: tuple[str, ...] = ("active", "all", "legacy")


def _get_rankings_v2(
    session,
    gender: Gender,
    discipline: Discipline,
    limit: int = 200,
    view: str = "active",
    today: Optional[date] = None,
):
    """Return a ranked list of athletes for a given discipline/gender.

    The ``view`` parameter (Issue #91) controls activity filtering:

    - ``"active"`` (default): only athletes whose ``last_event_at`` falls
      within the last :data:`INACTIVE_THRESHOLD_MONTHS` months.  Hides
      ghost athletes whose σ inflated but whose μ never deflated.
    - ``"all"``: every athlete that the ``is_likely_retired`` heuristic
      does not flag.  In SQL this is ``retired_at IS NULL AND
      (last_event_at IS NULL OR last_event_at >= today - 3 years)``.
    - ``"legacy"``: no filter — the pre-#91 behaviour.  Kept for debug.

    ``today`` may be pinned for deterministic tests; otherwise uses
    :func:`date.today`.  Invalid ``view`` values fall back to ``"active"``
    rather than raising — the HTML route is forgiving by design.
    """
    if today is None:
        today = date.today()

    stmt = (
        select(Rating, Athlete)
        .join(Athlete, Rating.athlete_id == Athlete.id)
        .where(Rating.discipline == discipline, Athlete.gender == gender)
    )

    if view == "active":
        # 12-month window — INACTIVE_THRESHOLD_MONTHS * ~30 days. Using
        # ``timedelta(days=365)`` keeps the filter portable across SQLite
        # (no INTERVAL support) and Postgres.
        cutoff = today - timedelta(days=int(INACTIVE_THRESHOLD_MONTHS * 30.4375))
        stmt = stmt.where(Rating.last_event_at >= cutoff)
    elif view == "all":
        cutoff = today - timedelta(days=int(RETIRED_THRESHOLD_YEARS * 365.25))
        stmt = stmt.where(
            Athlete.retired_at.is_(None),
            or_(Rating.last_event_at.is_(None), Rating.last_event_at >= cutoff),
        )
    # "legacy" (or anything else) — no extra filter.

    stmt = stmt.order_by(Rating.mu.desc()).limit(limit)
    rows = session.execute(stmt).all()
    return [
        {
            "rank": i + 1,
            "id": athlete.id,
            "name": athlete.name,
            "nationality": athlete.nationality or "—",
            "year_of_birth": athlete.year_of_birth,
            # Computed age for table views (#106) — bare birth year looked like
            # a rating next to the μ column. Nullable: None renders as "—".
            "age": (date.today().year - athlete.year_of_birth)
            if athlete.year_of_birth
            else None,
            "mu": round(rating.mu, 1),
            "sigma": round(sigma_now(rating.sigma, rating.last_event_at), 1),
            "n_events": rating.n_events,
            "provisional": rating.provisional,
            "last_event_at": rating.last_event_at,
        }
        for i, (rating, athlete) in enumerate(rows)
    ]


def _get_90d_delta(session, athlete_id: int, discipline: Discipline) -> float:
    """Return the rating delta over the last ~90 days (3 most-recent events).

    Kept for any single-athlete caller; the leaderboard now uses the batched
    :func:`_get_90d_deltas` to avoid an N+1 across the page.
    """
    return _get_90d_deltas(session, [athlete_id], discipline).get(athlete_id, 0.0)


def _get_90d_deltas(
    session, athlete_ids: list[int], discipline: Discipline
) -> dict[int, float]:
    """Batched 90d rating delta for many athletes in a single query.

    For each athlete the delta is ``latest.mu_after - earliest.mu_before``
    across that athlete's 3 most-recent ``RatingHistory`` rows (ordered by
    ``Event.start_date`` descending) in the given discipline — identical
    semantics to the per-athlete :func:`_get_90d_delta`, but computed for every
    id in ``athlete_ids`` with one round-trip instead of one query per row.

    Returns ``{athlete_id: delta}``; athletes with no history in the discipline
    are omitted (callers should default to ``0.0``).
    """
    if not athlete_ids:
        return {}

    # Rank each athlete's history rows by recency (most-recent event first),
    # partitioned per athlete. ``RatingHistory.id`` is a deterministic
    # tiebreaker for rows sharing a start_date (multiple rounds / a tpb row in
    # one event); the original per-row query left ties unordered, so this is a
    # strict superset of its behaviour and matches on all non-tied data.
    rn = (
        func.row_number()
        .over(
            partition_by=RatingHistory.athlete_id,
            order_by=(Event.start_date.desc(), RatingHistory.id.desc()),
        )
        .label("rn")
    )
    ranked = (
        select(
            RatingHistory.athlete_id.label("athlete_id"),
            RatingHistory.mu_after.label("mu_after"),
            RatingHistory.mu_before.label("mu_before"),
            rn,
        )
        .join(Event, RatingHistory.event_id == Event.id)
        .where(
            RatingHistory.athlete_id.in_(athlete_ids),
            Event.discipline == discipline,
        )
        .subquery()
    )

    # Pull only the top-3 rows per athlete (<=3 * len(athlete_ids) rows) and
    # collapse in Python: rn==1 is the latest, max(rn) is the earliest.
    deltas: dict[int, float] = {}
    rows = session.execute(
        select(
            ranked.c.athlete_id,
            ranked.c.mu_after,
            ranked.c.mu_before,
            ranked.c.rn,
        )
        .where(ranked.c.rn <= 3)
        .order_by(ranked.c.athlete_id, ranked.c.rn)
    ).all()

    latest_after: dict[int, float] = {}
    earliest_before: dict[int, float] = {}
    for aid, mu_after, mu_before, rank in rows:
        if rank == 1:
            latest_after[aid] = mu_after
        # The last row seen per athlete (largest rn, ordered asc) is the
        # earliest; overwriting on each row leaves the max-rn value.
        earliest_before[aid] = mu_before

    for aid, latest in latest_after.items():
        deltas[aid] = round(latest - earliest_before[aid], 1)
    return deltas


def _ticker_context(session) -> dict:
    """Build the context dict for the sticky ticker.

    Returns:
        live_event: ``{"id": int, "name": str}`` for the most recently-started
            event whose ``event_status()`` is LIVE, or None when nothing is
            currently in progress.
        ticker_items: list of {kind, text, delta?, tag?} dicts
    """
    # Candidate window: an event can only be LIVE if its start_date sits in
    # [today - LIVE_WINDOW_DAYS, today]. That's typically 0-3 events on a
    # World Cup weekend, so we let bulk_event_status() collapse FINISHED
    # cases via a single EXISTS query (no N+1).
    today = date.today()
    live_event = None
    try:
        candidates = list(
            session.execute(
                select(Event)
                .where(
                    Event.start_date >= today - timedelta(days=LIVE_WINDOW_DAYS),
                    Event.start_date <= today,
                )
                .order_by(Event.start_date.desc(), Event.id.desc())
            ).scalars()
        )
        if candidates:
            statuses = bulk_event_status(candidates, today=today, session=session)
            for ev in candidates:
                if statuses.get(ev.id) == EventStatus.LIVE:
                    live_event = {"id": ev.id, "name": ev.name}
                    break
    except Exception:
        # Ticker is decorative — never let a query failure 500 the page.
        live_event = None

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
# GET /favicon.ico  — silence browser-tab requests (Issue #73)
# ---------------------------------------------------------------------------

_FAVICON_PATH = Path(__file__).resolve().parent.parent / "static" / "favicon.svg"


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the site favicon.

    Browsers request /favicon.ico on every page load regardless of whether a
    <link rel="icon"> tag is present.  Without this route the request falls
    through to the catch-all and returns 500.  Modern browsers accept SVG
    favicons served under the .ico path.
    """
    return FileResponse(_FAVICON_PATH, media_type="image/svg+xml")


# ---------------------------------------------------------------------------
# GET /  — Landing page
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def v2_landing(request: Request):
    t = _templates(request)

    with _session() as session:
        cache_key = f"html:landing:fp:{ratings_fingerprint(session)}"
        ctx = html_page_cache.get(cache_key)
        if ctx is None:
            # Top 8 athletes by mu per discipline / gender.
            men_boulder = _get_rankings_v2(
                session, Gender.M, Discipline.BOULDER, limit=8
            )
            women_boulder = _get_rankings_v2(
                session, Gender.F, Discipline.BOULDER, limit=8
            )
            men_lead = _get_rankings_v2(session, Gender.M, Discipline.LEAD, limit=8)
            women_lead = _get_rankings_v2(session, Gender.F, Discipline.LEAD, limit=8)
            men_speed = _get_rankings_v2(session, Gender.M, Discipline.SPEED, limit=8)
            women_speed = _get_rankings_v2(session, Gender.F, Discipline.SPEED, limit=8)
            men_bl = _get_rankings_v2(
                session, Gender.M, Discipline.BOULDER_LEAD, limit=8
            )
            women_bl = _get_rankings_v2(
                session, Gender.F, Discipline.BOULDER_LEAD, limit=8
            )

            # App metrics
            total_athletes = session.execute(
                select(func.count(Athlete.id))
            ).scalar_one()
            total_events = session.execute(select(func.count(Event.id))).scalar_one()
            total_ratings = session.execute(
                select(func.count(RatingHistory.id))
            ).scalar_one()

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
            }
            html_page_cache.set(cache_key, ctx)

        # Ticker is time-sensitive — compute fresh, outside the cached data.
        ticker = _ticker_context(session)

    ctx = {**ctx, **ticker, **_nav_context("landing")}

    return t.TemplateResponse(request, "landing.html", ctx)


# ---------------------------------------------------------------------------
# GET /leaderboard
# ---------------------------------------------------------------------------


@router.get("/leaderboard", response_class=HTMLResponse)
async def v2_leaderboard(
    request: Request,
    disc: str = Query(default="B"),
    gender: str = Query(default="M"),
    view: str = Query(default="active"),
):
    t = _templates(request)

    disc_enum = _DISC_KEY_TO_ENUM.get(disc.upper(), Discipline.BOULDER)
    gender_enum = Gender.M if gender.upper() == "M" else Gender.F
    # Default to "active" for any unrecognised value (#91).
    view_norm = view.lower() if view else "active"
    if view_norm not in LEADERBOARD_VIEWS:
        view_norm = "active"

    today = date.today()
    # Months-of-inactivity badge cutoff for the "all" view (#91).
    badge_active_cutoff = today - timedelta(
        days=int(INACTIVE_THRESHOLD_MONTHS * 30.4375)
    )

    with _session() as session:
        cache_key = (
            f"html:leaderboard:disc:{disc_enum.value}:gender:{gender_enum.value}"
            f":view:{view_norm}:fp:{ratings_fingerprint(session)}"
        )
        ctx = html_page_cache.get(cache_key)
        if ctx is None:
            # Fetch the summary set once (limit=200) and slice the top 20 for
            # display — replaces the previous two full _get_rankings_v2 calls
            # (one at limit=20, one at limit=200) with a single materialisation.
            all_rows = _get_rankings_v2(
                session, gender_enum, disc_enum, limit=200, view=view_norm, today=today
            )
            rows = all_rows[:20]

            # Batched 90d delta for the displayed rows — one query instead of a
            # per-row N+1 (was up to 20 separate round-trips).
            deltas = _get_90d_deltas(session, [r["id"] for r in rows], disc_enum)
            for row in rows:
                row["delta_90d"] = deltas.get(row["id"], 0.0)
                last = row.get("last_event_at")
                if last is None:
                    row["activity_label"] = None
                elif last >= badge_active_cutoff:
                    row["activity_label"] = "Active"
                else:
                    months = max(1, int((today - last).days / 30.4375))
                    row["activity_label"] = f"Inactive {months}mo"

            top_mu = all_rows[0]["mu"] if all_rows else 0
            mid_idx = len(all_rows) // 2
            median_mu = all_rows[mid_idx]["mu"] if all_rows else 0
            total_count = len(all_rows)

            ctx = {
                "rows": rows,
                "disc": disc.upper(),
                "disc_label": _DISC_LABEL.get(disc_enum, disc),
                "gender": gender_enum.value,
                "gender_label": _GENDER_LABEL.get(gender_enum, gender),
                "view": view_norm,
                "top_mu": top_mu,
                "median_mu": median_mu,
                "total_count": total_count,
            }
            html_page_cache.set(cache_key, ctx)

        # Ticker is time-sensitive (countdowns) and cheap — compute fresh,
        # outside the cached page data.
        ticker = _ticker_context(session)

    ctx = {**ctx, **ticker, **_nav_context("leaderboard")}
    return t.TemplateResponse(request, "leaderboard.html", ctx)


# ---------------------------------------------------------------------------
# GET /athletes  — searchable, browsable athlete index (#102)
# ---------------------------------------------------------------------------


@router.get("/athletes", response_class=HTMLResponse)
async def v2_athletes_index(
    request: Request,
    disc: str = Query(default="L"),
    gender: str = Query(default="M"),
):
    """Searchable / browsable athletes index.

    Renders an in-page filterable list (discipline + gender pills, client-side
    name/country filter, rating sort) backed by a single batched query, plus a
    debounced typeahead that hits ``GET /api/v1/athletes`` for a global search
    across the whole population. Every row links to ``/athletes/{id}``.
    """
    t = _templates(request)

    disc_enum = _DISC_KEY_TO_ENUM.get(disc.upper(), Discipline.LEAD)
    gender_enum = Gender.M if gender.upper() == "M" else Gender.F

    with _session() as session:
        cache_key = (
            f"html:athletes:disc:{disc_enum.value}:gender:{gender_enum.value}"
            f":fp:{ratings_fingerprint(session)}"
        )
        ctx = html_page_cache.get(cache_key)
        if ctx is None:
            # Active-view ranked rows for the selected discipline + gender. Cap
            # at 300 so the page stays light; the typeahead covers the long tail.
            rows = _get_rankings_v2(
                session,
                gender_enum,
                disc_enum,
                limit=300,
                view="active",
            )
            ctx = {
                "rows": rows,
                "disc": disc_enum.value,
                "disc_label": _DISC_LABEL.get(disc_enum, disc),
                "gender": gender_enum.value,
                "gender_label": _GENDER_LABEL.get(gender_enum, gender),
                "total_count": len(rows),
            }
            html_page_cache.set(cache_key, ctx)

        # Ticker is time-sensitive — compute fresh, outside the cached data.
        ticker = _ticker_context(session)

    ctx = {**ctx, **ticker, **_nav_context("athletes")}
    return t.TemplateResponse(request, "athletes_index.html", ctx)


# ---------------------------------------------------------------------------
# GET /athletes/{athlete_id}  — athlete profile
# ---------------------------------------------------------------------------


@router.get("/athletes/{athlete_id}", response_class=HTMLResponse)
async def v2_athlete_profile(request: Request, athlete_id: int):
    """Render the rich athlete profile page (Issue #86).

    Context sections (top to bottom in the template):
      1. Header card        — photo + name + nationality + year_of_birth
      2. Body metrics       — only fields that are non-NULL
      3. Current ratings    — one row per non-null discipline (μ, σ, rank, n)
      4. ELO graph          — Chart.js line per discipline + ±σ band + event markers
      5. Recent changes     — last 5 events with Δμ + top-3 opponents
      6. Combined breakdown — only rendered when athlete has a BOULDER_LEAD row
      7. Full event history — collapsible per-season tables
    """
    t = _templates(request)

    with _session() as session:
        athlete = session.get(Athlete, athlete_id)
        if not athlete:
            return HTMLResponse("Athlete not found", status_code=404)

        # ---------------------------------------------------------------
        # 1. All ratings for this athlete
        # ---------------------------------------------------------------
        ratings_rows = list(
            session.execute(
                select(Rating).where(Rating.athlete_id == athlete_id)
            ).scalars()
        )
        ratings_by_disc: dict[str, dict] = {}
        for r in ratings_rows:
            key = _DISC_ENUM_TO_KEY.get(r.discipline, r.discipline.value)

            # Compute the athlete's rank in this discipline among same-gender
            # athletes. Filtering by gender means we can compare apples-to-apples
            # against the leaderboard rendered elsewhere.
            rank_row = session.execute(
                select(func.count())
                .select_from(Rating)
                .join(Athlete, Rating.athlete_id == Athlete.id)
                .where(
                    Rating.discipline == r.discipline,
                    Athlete.gender == athlete.gender,
                    Rating.mu > r.mu,
                )
            ).scalar_one()
            ratings_by_disc[key] = {
                "mu": round(r.mu, 1),
                "sigma": round(sigma_now(r.sigma, r.last_event_at), 1),
                "n_events": r.n_events,
                "rank": int(rank_row) + 1,
                "provisional": r.provisional,
            }

        # Primary rating: prefer Lead > Boulder > Speed > BL (matches sidebar
        # convention so the page makes sense even for speed specialists).
        pref_order = ["L", "B", "S", "BL"]
        primary_disc_key = next((k for k in pref_order if k in ratings_by_disc), None)
        primary_rating = ratings_by_disc.get(
            primary_disc_key or "L",
            {"mu": None, "sigma": None, "n_events": 0, "rank": None},
        )
        primary_disc_enum = _DISC_KEY_TO_ENUM.get(
            primary_disc_key or "L", Discipline.LEAD
        )
        primary_disc_label = _DISC_LABEL.get(primary_disc_enum, "Lead")

        # ---------------------------------------------------------------
        # 2. Rating history per discipline — for the multi-series Chart.js
        # ---------------------------------------------------------------
        # Re-uses the same de-dup logic as /api/v1/athletes/{id}/history
        # (keep one point per event = the latest round). We store the mu and
        # sigma after the event so the template can render a ±σ band.
        rating_history_by_discipline: dict[str, list[dict]] = {}
        event_markers: list[dict] = []  # Olympics / WCh / WC finals

        if ratings_by_disc:
            disc_enums_with_rating = [
                _DISC_KEY_TO_ENUM[k] for k in ratings_by_disc if k in _DISC_KEY_TO_ENUM
            ]

            all_history = list(
                session.execute(
                    select(RatingHistory, Event, Round)
                    .join(Event, RatingHistory.event_id == Event.id)
                    .join(Round, RatingHistory.round_id == Round.id)
                    .where(
                        RatingHistory.athlete_id == athlete_id,
                        Event.discipline.in_(disc_enums_with_rating),
                    )
                    .order_by(Event.start_date.asc(), Round.round_type.asc())
                ).all()
            )

            # Group by (discipline, event_id) and keep the *last* round's
            # post-event state. Round.round_type sorts alphabetically:
            # final < qualification < semi — not the wall-clock order. But the
            # value we want is "the round that finalised the rating for this
            # event", which by construction is the last one inserted. We use
            # the latest by RatingHistory.id within an event.
            per_disc_event: dict[
                tuple[Discipline, int], tuple[RatingHistory, Event]
            ] = {}
            for rh, ev, _rnd in all_history:
                key = (ev.discipline, ev.id)
                prev = per_disc_event.get(key)
                if prev is None or rh.id > prev[0].id:
                    per_disc_event[key] = (rh, ev)

            for disc_enum in disc_enums_with_rating:
                disc_key = _DISC_ENUM_TO_KEY.get(disc_enum, disc_enum.value)
                points: list[dict] = []
                for (d_e, _eid), (rh, ev) in per_disc_event.items():
                    if d_e != disc_enum:
                        continue
                    points.append(
                        {
                            "date": str(ev.start_date),
                            "event_id": ev.id,
                            "event_name": ev.name,
                            "season": ev.season,
                            "mu": round(rh.mu_after, 1),
                            "sigma": round(rh.sigma_after, 1),
                            "tier": ev.tier.value,
                        }
                    )
                points.sort(key=lambda p: p["date"])
                rating_history_by_discipline[disc_key] = points

                # Major-event markers go in a separate dataset for the chart.
                for p in points:
                    if p["tier"] in (
                        EventTier.OLYMPICS.value,
                        EventTier.WORLD_CHAMPIONSHIP.value,
                    ):
                        event_markers.append(
                            {
                                "date": p["date"],
                                "mu": p["mu"],
                                "tier": p["tier"],
                                "disc": disc_key,
                                "event_name": p["event_name"],
                            }
                        )
                    # World Cup Finals are events whose name contains "final",
                    # but we don't have a per-event flag for that. Mark the
                    # season-closing World Cup event instead.
                # (No reliable per-event flag for "WC Final" — Olympics + WChs
                # are enough to anchor the timeline.)

        # ---------------------------------------------------------------
        # 3. Recent changes — last 5 events across all disciplines, with the
        #    top-3 opponents and pairwise Δμ contributions for context.
        # ---------------------------------------------------------------
        recent_rh = list(
            session.execute(
                select(RatingHistory, Event)
                .join(Event, RatingHistory.event_id == Event.id)
                .where(RatingHistory.athlete_id == athlete_id)
                .order_by(Event.start_date.desc())
                .limit(40)
            ).all()
        )
        seen_ev: set[int] = set()
        recent_events = []

        for rh, ev in recent_rh:
            if ev.id in seen_ev:
                continue
            seen_ev.add(ev.id)
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

            # Sum delta across all of this athlete's rounds in the event so
            # the listed number matches what "Recent ELO changes" implies.
            total_delta_row = session.execute(
                select(
                    func.sum(RatingHistory.mu_after - RatingHistory.mu_before)
                ).where(
                    RatingHistory.athlete_id == athlete_id,
                    RatingHistory.event_id == ev.id,
                )
            ).scalar_one_or_none()
            delta = round(float(total_delta_row or 0.0), 1)

            disc_display = _DISC_LABEL.get(ev.discipline, ev.discipline.value)

            # Top-3 opponents = pull pairs from the *last* (i.e. final) round's
            # contributing_pairs, sorted by abs(delta), then resolve names.
            # Only consider kind='pair' rows: TPB rows (#90) store
            # contributing_pairs as a dict ({"rank":…,"gross_bonus":…}), not a
            # list of pair-dicts, so iterating one would yield string keys and
            # crash p.get(). The isinstance guard is belt-and-suspenders.
            last_round_rh = max(
                (
                    h
                    for h, _e in recent_rh
                    if _e.id == ev.id
                    and h.kind == "pair"
                    and isinstance(h.contributing_pairs, list)
                    and h.contributing_pairs
                ),
                key=lambda h: h.id,
                default=None,
            )
            opponents: list[dict] = []
            if last_round_rh and last_round_rh.contributing_pairs:
                pairs = sorted(
                    last_round_rh.contributing_pairs,
                    key=lambda p: abs(p.get("delta", 0.0)),
                    reverse=True,
                )[:3]
                opponent_ids = [int(p["opponent_id"]) for p in pairs]
                opp_map: dict[int, str] = {}
                if opponent_ids:
                    for opp in session.execute(
                        select(Athlete).where(Athlete.id.in_(opponent_ids))
                    ).scalars():
                        opp_map[opp.id] = opp.name
                for p in pairs:
                    oid = int(p["opponent_id"])
                    pair_delta = round(float(p.get("delta", 0.0)), 1)
                    opponents.append(
                        {
                            "id": oid,
                            "name": opp_map.get(oid, f"ID {oid}"),
                            "delta": pair_delta,
                            "result": p.get("result"),
                        }
                    )

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
                    "opponents": opponents,
                }
            )
            if len(recent_events) >= 5:
                break

        # ---------------------------------------------------------------
        # 4. Combined (B+L) breakdown — only when we have a BL rating.
        # ---------------------------------------------------------------
        combined_breakdown: dict | None = None
        if (
            "BL" in ratings_by_disc
            and "B" in ratings_by_disc
            and "L" in ratings_by_disc
        ):
            from scripts.compute_combined_ratings import load_combined_weights

            weights = load_combined_weights()
            if weights.source == "learned":
                weights_used = (
                    f"learned (w_L={weights.w_lead:.3f}, w_B={weights.w_boulder:.3f})"
                )
            else:
                weights_used = "geometric_mean"
            combined_breakdown = {
                "boulder_mu": ratings_by_disc["B"]["mu"],
                "lead_mu": ratings_by_disc["L"]["mu"],
                "combined_mu": ratings_by_disc["BL"]["mu"],
                "weights_used": weights_used,
                "w_lead": round(weights.w_lead, 3),
                "w_boulder": round(weights.w_boulder, 3),
            }

        # ---------------------------------------------------------------
        # 5. Full event history — grouped by season → discipline.
        # ---------------------------------------------------------------
        all_events_rows = list(
            session.execute(
                select(RatingHistory, Event)
                .join(Event, RatingHistory.event_id == Event.id)
                .where(RatingHistory.athlete_id == athlete_id)
                .order_by(Event.start_date.desc(), Event.id.desc())
            ).all()
        )

        history_by_season: dict[int, list[dict]] = {}
        seen_event_for_history: set[int] = set()

        for rh, ev in all_events_rows:
            if ev.id in seen_event_for_history:
                continue
            seen_event_for_history.add(ev.id)
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
            total_delta_row = session.execute(
                select(
                    func.sum(RatingHistory.mu_after - RatingHistory.mu_before)
                ).where(
                    RatingHistory.athlete_id == athlete_id,
                    RatingHistory.event_id == ev.id,
                )
            ).scalar_one_or_none()
            ev_delta = round(float(total_delta_row or 0.0), 1)

            history_by_season.setdefault(ev.season, []).append(
                {
                    "event_id": ev.id,
                    "event_name": ev.name,
                    "date": str(ev.start_date),
                    "discipline": _DISC_LABEL.get(ev.discipline, ev.discipline.value),
                    "tier": ev.tier.value.replace("_", " ").title(),
                    "place": place_row,
                    "delta": ev_delta,
                    "delta_sign": "+"
                    if ev_delta > 0
                    else ("−" if ev_delta < 0 else ""),
                    "delta_abs": abs(ev_delta),
                }
            )

        history_seasons_sorted = sorted(history_by_season.keys(), reverse=True)
        full_history = [
            {"season": s, "events": history_by_season[s]}
            for s in history_seasons_sorted
        ]

        # ---------------------------------------------------------------
        # 6. Sidebar (unchanged) + ticker
        # ---------------------------------------------------------------
        sidebar_athletes = _get_rankings_v2(
            session, athlete.gender, primary_disc_enum, limit=14
        )

        ticker = _ticker_context(session)

        # Computed age from year_of_birth (#106). The column is nullable; when
        # it is NULL we leave ``age`` as None and the template renders nothing.
        # We only store the birth *year* (not the full date), so this is the
        # conventional current-year − birth-year estimate.
        athlete_age = None
        if athlete.year_of_birth:
            athlete_age = date.today().year - athlete.year_of_birth

        # Capture all athlete fields inside the session — accessing them after
        # session close would trigger a DetachedInstanceError.
        athlete_ctx = {
            "id": athlete.id,
            "name": athlete.name,
            "nationality": athlete.nationality or "—",
            "year_of_birth": athlete.year_of_birth,
            "age": athlete_age,
            "gender": athlete.gender.value,
            "photo_url": athlete.photo_url,
            "height_cm": athlete.height_cm,
            "weight_kg": athlete.weight_kg,
            "wingspan_cm": athlete.wingspan_cm,
        }

    # Order ratings for the template in our preferred display order.
    disciplines_ordered = [
        (k, _DISC_LABEL.get(_DISC_KEY_TO_ENUM[k], k), ratings_by_disc[k])
        for k in ("L", "B", "S", "BL")
        if k in ratings_by_disc
    ]

    ctx = {
        "athlete": athlete_ctx,
        "primary_rating": primary_rating,
        "primary_disc_label": primary_disc_label,
        "primary_disc_key": primary_disc_key or "L",
        "ratings_by_disc": ratings_by_disc,
        "disciplines_ordered": disciplines_ordered,
        "rating_history_by_discipline": rating_history_by_discipline,
        "rating_history_json": json.dumps(rating_history_by_discipline),
        "event_markers_json": json.dumps(event_markers),
        "recent_events": recent_events,
        "combined_breakdown": combined_breakdown,
        "full_history": full_history,
        "sidebar_athletes": sidebar_athletes,
        "current_athlete_id": athlete_id,
        **ticker,
        **_nav_context("athletes"),
    }
    return t.TemplateResponse(request, "athletes.html", ctx)


# ---------------------------------------------------------------------------
# GET /projections
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
# GET /head-to-head  — athlete selection form
# ---------------------------------------------------------------------------


@router.get("/head-to-head", response_class=HTMLResponse)
async def v2_h2h_form(request: Request):
    """Empty head-to-head picker (#98).

    The page loads with two debounced typeahead inputs (backed by
    ``GET /api/v1/athletes``), a discipline control, and a gender control that
    explicitly supports cross-gender (man vs woman) comparison. No athlete pool
    is pre-loaded — selection is driven entirely by the search inputs.
    """
    t = _templates(request)

    with _session() as session:
        ticker = _ticker_context(session)

    ctx = {
        "h2h_result": None,
        "selected_disc": "L",
        "selected_gender": "",
        **ticker,
        **_nav_context("h2h"),
    }
    return t.TemplateResponse(request, "head_to_head.html", ctx)


# ---------------------------------------------------------------------------
# GET /head-to-head/{a_id}/{b_id}  — H2H result
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

    disc_enum = _parse_discipline(discipline)
    if disc_enum is None:
        return HTMLResponse(f"Invalid discipline: {discipline}.", status_code=400)

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

        # Ring geometry: R=70, C=2πR. Two-sided ring (#99): A's arc starts at
        # 12 o'clock and sweeps clockwise for its share; B's arc begins where A
        # ends (rotated win_a·360°) and sweeps the remainder, so the two meet
        # exactly and the underdog's larger share renders proportionally.
        #   - dash_offset_a leaves (1 - win_a) of the circle hidden → A's arc.
        #   - dash_offset_b is the symmetric value for B (kept for parity).
        #   - seg_a / seg_b are the literal visible arc lengths.
        #   - rot_b rotates B's circle so it starts at A's end.
        R = 70
        C = 2 * math.pi * R
        dash_offset_a = round(C * (1 - win_a), 2)
        dash_offset_b = round(C * (1 - win_b), 2)
        seg_a = round(C * win_a, 2)
        seg_b = round(C * win_b, 2)
        rot_b = round(win_a * 360.0, 2)

        mu_gap = round(rating_a.mu - rating_b.mu, 1)

        # Cross-gender support (#98): the pair may mix genders. The win-prob
        # math (analytic ELO expectation) is gender-agnostic; we only surface
        # the fact so the template can label it.
        gender_a = athlete_a.gender
        gender_b = athlete_b.gender
        cross_gender = gender_a != gender_b

        ticker = _ticker_context(session)

    disc_label = _DISC_LABEL.get(disc_enum, discipline)

    ctx = {
        "h2h_result": {
            "athlete_a": {
                "id": athlete_a.id,
                "name": athlete_a.name,
                "nationality": athlete_a.nationality or "—",
                "gender": gender_a.value,
                "mu": round(rating_a.mu, 1),
                "sigma": round(sigma_now(rating_a.sigma, rating_a.last_event_at), 1),
                "n_events": rating_a.n_events,
            },
            "athlete_b": {
                "id": athlete_b.id,
                "name": athlete_b.name,
                "nationality": athlete_b.nationality or "—",
                "gender": gender_b.value,
                "mu": round(rating_b.mu, 1),
                "sigma": round(sigma_now(rating_b.sigma, rating_b.last_event_at), 1),
                "n_events": rating_b.n_events,
            },
            "win_a": round(win_a * 100, 1),
            "win_b": round(win_b * 100, 1),
            "win_a_frac": round(win_a, 4),
            "ring_R": R,
            "ring_C": round(C, 2),
            "ring_dash_offset": dash_offset_a,
            "ring_dash_offset_b": dash_offset_b,
            "ring_seg_a": seg_a,
            "ring_seg_b": seg_b,
            "ring_rot_b": rot_b,
            "past_meetings": past_meetings,
            "no_shared_events": past_meetings == 0,
            "cross_gender": cross_gender,
            "mu_gap": mu_gap,
            "disc_label": disc_label,
            "disc_key": _DISC_ENUM_TO_KEY.get(disc_enum, "L"),
            "chart_labels": json.dumps(all_labels),
            "chart_mu_a": json.dumps(aligned_a),
            "chart_mu_b": json.dumps(aligned_b),
        },
        "a_id": a_id,
        "b_id": b_id,
        "selected_disc": _DISC_ENUM_TO_KEY.get(disc_enum, "L"),
        "selected_gender": "",
        **ticker,
        **_nav_context("h2h"),
    }
    return t.TemplateResponse(request, "head_to_head.html", ctx)


# ---------------------------------------------------------------------------
# GET /api  — API reference page
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


# ---------------------------------------------------------------------------
# GET /events  — event list (newest first, optional discipline + season filter)
# ---------------------------------------------------------------------------

_DISC_LABEL_KEY_TO_ENUM = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
}


@router.get("/events", response_class=HTMLResponse)
async def v2_events(
    request: Request,
    discipline: str = Query(default="lead"),
    season: Optional[int] = Query(default=None),
):
    t = _templates(request)

    disc_key = discipline.lower()
    disc_enum = _DISC_LABEL_KEY_TO_ENUM.get(disc_key, Discipline.LEAD)

    with _session() as session:
        # All seasons available for this discipline (for the dropdown)
        all_seasons = list(
            session.execute(
                select(Event.season)
                .where(Event.discipline == disc_enum)
                .distinct()
                .order_by(Event.season.desc())
            ).scalars()
        )

        stmt = select(Event).where(Event.discipline == disc_enum)
        if season is not None:
            stmt = stmt.where(Event.season == season)
        stmt = stmt.order_by(Event.start_date.desc()).limit(500)

        events = list(session.execute(stmt).scalars())
        event_rows = [
            {
                "id": e.id,
                "name": e.name,
                "season": e.season,
                "tier": e.tier.value.replace("_", " ").title(),
                "date": str(e.start_date),
            }
            for e in events
        ]

        ticker = _ticker_context(session)

    ctx = {
        "events": event_rows,
        "all_seasons": all_seasons,
        "selected_season": season,
        "discipline_key": disc_key,
        "discipline_label": _DISC_LABEL.get(disc_enum, "Lead"),
        **ticker,
        **_nav_context("events"),
    }
    return t.TemplateResponse(request, "events.html", ctx)


# ---------------------------------------------------------------------------
# Forecast recap helpers (shared by /events/{id} and /predictions/{id})
# ---------------------------------------------------------------------------


def _forecast_tables_present(session) -> bool:
    """Return True if both forecast tables have been provisioned on this DB.

    First-deploy guard. `Base.metadata.create_all` is skipped against Postgres
    in `create_app()` (see `api/app.py`), so the forecast tables only exist on
    a prod DB once `scripts/init_forecast_tables.py` (run by the daily workflow
    or manually) has executed. Until then any query against them would crash
    every page that uses this helper — including the high-traffic event detail.

    Inspect-based detection — avoids issuing a query that would fail and poison
    the session's transaction state.
    """
    inspector = sa_inspect(session.get_bind())
    return inspector.has_table(EventForecast.__tablename__) and inspector.has_table(
        EventForecastScore.__tablename__
    )


def _build_forecast_recap(
    session,
    event: Event,
    *,
    is_backfill: bool = False,
    top_n: int = 3,
) -> list[dict]:
    """Build the per-gender "Predicted vs Actual" recap panels for an event.

    Returns one dict per gender that has a forecast row, ordered M then F.
    Each dict has the shape consumed by the recap panel template:

        {
            "gender": "M",
            "gender_label": "Men",
            "predicted_top3": [{"athlete_id", "name", "prob_podium"}],
            "actual_top3":    [{"athlete_id", "name", "rank"}],
            "rows": [{"athlete_id", "name", "predicted_rank", "actual_rank",
                      "marker"}],  # marker ∈ {"hit", "predicted_only",
                                    #          "actual_only"}
            "top3_intersection": int | None,   # from score row
            "brier_podium": float | None,
            "n_athletes": int | None,
            "has_score": bool,
            "is_backfill": bool,
        }

    Returns an empty list when no forecasts are stored for the event, or when
    the forecast tables don't exist yet (deploy hasn't run the table-creation
    migration). The recap panel is purely additive, so a missing table must
    not break the underlying event/predictions page.
    """
    if not _forecast_tables_present(session):
        return []

    panels: list[dict] = []

    for gender_enum in (Gender.M, Gender.F):
        forecast_rows = list(
            session.execute(
                select(EventForecast, Athlete)
                .join(Athlete, EventForecast.athlete_id == Athlete.id)
                .where(
                    EventForecast.event_id == event.id,
                    EventForecast.gender == gender_enum,
                    EventForecast.is_backfill == is_backfill,
                )
                .order_by(EventForecast.prob_podium.desc())
            ).all()
        )

        if not forecast_rows:
            continue

        # Predicted top-N — sort by prob_podium DESC.
        predicted_top = [
            {
                "athlete_id": athlete.id,
                "name": athlete.name,
                "prob_podium": fc.prob_podium,
                "prob_win": fc.prob_win,
            }
            for fc, athlete in forecast_rows[:top_n]
        ]

        # Actual top-N — final round if present, else semifinal, else qual.
        final_round = None
        for round_type in (RoundType.FINAL, RoundType.SEMI, RoundType.QUALIFICATION):
            cand = next(
                (
                    r
                    for r in event.rounds
                    if r.round_type == round_type and r.gender == gender_enum
                ),
                None,
            )
            if cand is not None:
                final_round = cand
                break

        actual_top: list[dict] = []
        if final_round is not None:
            results = list(
                session.execute(
                    select(Result, Athlete)
                    .join(Athlete, Result.athlete_id == Athlete.id)
                    .where(
                        Result.round_id == final_round.id,
                        Result.dns.is_(False),
                        Result.rank.is_not(None),
                    )
                    .order_by(Result.rank.asc())
                    .limit(top_n)
                ).all()
            )
            actual_top = [
                {
                    "athlete_id": athlete.id,
                    "name": athlete.name,
                    "rank": res.rank,
                }
                for res, athlete in results
            ]

        actual_ids = {row["athlete_id"] for row in actual_top}

        # Build a unified row list — one entry per athlete in either top-K.
        # Marker: hit if in both, predicted_only if just predicted, actual_only
        # if just actual.
        seen: set[int] = set()
        rows_out: list[dict] = []
        for row in predicted_top:
            aid = row["athlete_id"]
            seen.add(aid)
            in_actual = aid in actual_ids
            rows_out.append(
                {
                    "athlete_id": aid,
                    "name": row["name"],
                    "prob_podium": row["prob_podium"],
                    "actual_rank": next(
                        (a["rank"] for a in actual_top if a["athlete_id"] == aid),
                        None,
                    ),
                    "marker": "hit" if in_actual else "predicted_only",
                }
            )
        for row in actual_top:
            aid = row["athlete_id"]
            if aid in seen:
                continue
            rows_out.append(
                {
                    "athlete_id": aid,
                    "name": row["name"],
                    "prob_podium": None,
                    "actual_rank": row["rank"],
                    "marker": "actual_only",
                }
            )

        # Score row (may not exist if scoring hasn't run yet).
        score_row = session.execute(
            select(EventForecastScore).where(
                EventForecastScore.event_id == event.id,
                EventForecastScore.gender == gender_enum,
                EventForecastScore.is_backfill == is_backfill,
            )
        ).scalar_one_or_none()

        panels.append(
            {
                "gender": gender_enum.value,
                "gender_label": _GENDER_LABEL.get(gender_enum, gender_enum.value),
                "predicted_top3": predicted_top,
                "actual_top3": actual_top,
                "rows": rows_out,
                "has_score": score_row is not None,
                "top3_intersection": (
                    score_row.top3_intersection if score_row is not None else None
                ),
                "brier_podium": (
                    score_row.brier_podium if score_row is not None else None
                ),
                "n_athletes": (
                    score_row.n_athletes
                    if score_row is not None
                    else len(forecast_rows)
                ),
                "is_backfill": is_backfill,
            }
        )

    return panels


def _event_has_final_results(session, event: Event) -> bool:
    """Return True when the event has at least one Result row in a FINAL round."""
    final_round_ids = [r.id for r in event.rounds if r.round_type == RoundType.FINAL]
    if not final_round_ids:
        return False
    count = session.execute(
        select(func.count(Result.id)).where(
            Result.round_id.in_(final_round_ids),
            Result.dns.is_(False),
        )
    ).scalar_one()
    return count > 0


# ---------------------------------------------------------------------------
# GET /events/{event_id}  — event detail (round-by-round results, pre/post μ)
# ---------------------------------------------------------------------------


@router.get("/events/{event_id}", response_class=HTMLResponse)
async def v2_event_detail(request: Request, event_id: int):
    t = _templates(request)

    with _session() as session:
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

            # Batch-fetch RatingHistory for all athletes in this round in ONE
            # query instead of N+1. Without this, a 50-athlete qualification
            # round caused the page to hang.
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
                        "mu_before": round(rh.mu_before, 1) if rh else None,
                        "mu_after": round(rh.mu_after, 1) if rh else None,
                        "delta": round(delta, 1) if delta is not None else None,
                    }
                )

            rounds_data.append(
                {
                    "round_type": rnd.round_type.value.title(),
                    "gender": rnd.gender.value,
                    "gender_label": "Men" if rnd.gender.value == "M" else "Women",
                    "results": result_rows,
                }
            )

        disc_label = _DISC_LABEL.get(event.discipline, event.discipline.value)
        forecast_panels = _build_forecast_recap(session, event)
        ticker = _ticker_context(session)
        is_live = event_status(event, session=session) == EventStatus.LIVE

    ctx = {
        "event": {
            "id": event.id,
            "name": event.name,
            "season": event.season,
            "tier": event.tier.value.replace("_", " ").title(),
            "date": str(event.start_date),
            "discipline_label": disc_label,
            "is_live": is_live,
        },
        "rounds": rounds_data,
        "forecast_panels": forecast_panels,
        **ticker,
        **_nav_context("events"),
    }
    return t.TemplateResponse(request, "event_detail.html", ctx)


# ---------------------------------------------------------------------------
# GET /breakdown/{athlete_id}/{event_id}  — pairwise contributing-pairs
# ---------------------------------------------------------------------------


@router.get("/breakdown/{athlete_id}/{event_id}", response_class=HTMLResponse)
async def v2_breakdown(request: Request, athlete_id: int, event_id: int):
    t = _templates(request)

    with _session() as session:
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
                    RatingHistory.kind == "pair",
                )
                .order_by(Round.round_type)
            ).all()
        )

        rounds_breakdown = []
        for rh, rnd in histories:
            pairs = rh.contributing_pairs or []

            # Batch-fetch opponent athletes in a single IN query.
            opponent_ids = list({p["opponent_id"] for p in pairs})
            opponents: dict[int, Athlete] = {}
            if opponent_ids:
                for opp in session.execute(
                    select(Athlete).where(Athlete.id.in_(opponent_ids))
                ).scalars():
                    opponents[opp.id] = opp

            resolved_pairs = []
            for p in pairs:
                opp = opponents.get(p["opponent_id"])
                resolved_pairs.append(
                    {
                        "opponent_id": p["opponent_id"],
                        "opponent_name": opp.name if opp else f"ID {p['opponent_id']}",
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

        # Issue #90: load any Tournament Participation Bonus row for this
        # athlete + event and surface it as a separate breakdown section.
        tpb_row = session.execute(
            select(RatingHistory).where(
                RatingHistory.athlete_id == athlete_id,
                RatingHistory.event_id == event_id,
                RatingHistory.kind == "tpb",
            )
        ).scalar_one_or_none()
        tpb_section = None
        if tpb_row is not None:
            payload = tpb_row.contributing_pairs or {}
            tpb_section = {
                "mu_before": round(tpb_row.mu_before, 1),
                "mu_after": round(tpb_row.mu_after, 1),
                "delta": round(tpb_row.mu_after - tpb_row.mu_before, 1),
                "rank": payload.get("rank"),
                "gross_bonus": round(payload.get("gross_bonus", 0.0), 2),
                "debit": round(payload.get("debit", 0.0), 2),
                "tier": payload.get("tier", "").replace("_", " ").title(),
            }

        ticker = _ticker_context(session)

    ctx = {
        "athlete": {"id": athlete.id, "name": athlete.name},
        "event": {
            "id": event.id,
            "name": event.name,
            "season": event.season,
        },
        "rounds": rounds_breakdown,
        "tpb": tpb_section,
        **ticker,
        **_nav_context("events"),
    }
    return t.TemplateResponse(request, "breakdown.html", ctx)


# ---------------------------------------------------------------------------
# GET /live/{event_id}  — live event view with SSE consumer
# ---------------------------------------------------------------------------


@router.get("/live/{event_id}", response_class=HTMLResponse)
async def v2_live_event(request: Request, event_id: int, gender: str = "M"):
    t = _templates(request)

    with _session() as session:
        event = session.get(Event, event_id)
        if not event:
            return HTMLResponse("Event not found", status_code=404)

        if event_status(event, session=session) != EventStatus.LIVE:
            return HTMLResponse(
                f"This event isn't currently live. "
                f'<a href="/events/{event_id}">View results &rarr;</a>',
                status_code=404,
            )

        try:
            gender_enum = Gender(gender.upper())
        except ValueError:
            gender_enum = Gender.M

        available_genders = sorted({rnd.gender.value for rnd in event.rounds})

        # Problem B fix: if the requested gender has no rounds for this event,
        # fall back to the first available gender rather than showing an empty page.
        available_gender_enums = sorted({rnd.gender for rnd in event.rounds})
        if available_gender_enums and gender_enum not in available_gender_enums:
            gender_enum = available_gender_enums[0]

        # Build current leaderboard: prefer the highest round result per athlete.
        _rt_order = {
            RoundType.FINAL: 0,
            RoundType.SEMI: 1,
            RoundType.QUALIFICATION: 2,
        }
        sorted_rounds = sorted(
            [r for r in event.rounds if r.gender == gender_enum],
            key=lambda r: _rt_order.get(r.round_type, 99),
        )

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

        # Problem A fix: detect pre-event (no Result rows stored yet) and fall
        # back to the likely-competitor roster so projections are still useful.
        pre_event = len(leaderboard_rows) == 0
        from_likely_roster = False
        proj_athlete_ids: list[int] = []

        if pre_event:
            _roster_cache_key = (
                f"roster:{event.discipline.value}:{event.season}:{gender_enum.value}"
            )
            cached_roster = likely_roster_cache.get(_roster_cache_key)
            if cached_roster is None:
                cached_roster = likely_competitors(
                    session, event.discipline, event.season, gender_enum
                )
                likely_roster_cache.set(_roster_cache_key, cached_roster)
            proj_athlete_ids = list(
                cached_roster[:_V2_MAX_ATHLETES_PER_PROJECTION_CARD]
            )
            from_likely_roster = bool(proj_athlete_ids)

        # Build projection inputs from leaderboard. Athletes with a finished
        # rank are "completed"; the rest are "remaining". (For pre-event/all-
        # finished events, all rows go into "completed"; we use
        # compute_partial_event_probabilities either way for correctness.)
        completed: list[tuple[AthleteProjectionInput, int]] = []
        remaining: list[AthleteProjectionInput] = []

        if pre_event and proj_athlete_ids:
            # Pre-event: use likely roster as "remaining" athletes (no ranks yet).
            for inp in _build_proj_inputs_batched(
                session, proj_athlete_ids, event.discipline
            ):
                remaining.append(inp)
        else:
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
                    mu, sigma = rating.mu, sigma_now(rating.sigma, rating.last_event_at)
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
            completed_ids = {inp.athlete_id for inp, _ in completed}
            for inp in sorted(
                all_inputs, key=lambda a: probs[a.athlete_id]["expected_rank"]
            ):
                p = probs[inp.athlete_id]
                projection_rows.append(
                    {
                        "athlete_id": inp.athlete_id,
                        "name": inp.name,
                        "mu": round(inp.mu, 1),
                        "win": f"{p['win'] * 100:.1f}",
                        "podium": f"{p['podium'] * 100:.1f}",
                        "expected_rank": p["expected_rank"],
                        "win_raw": p["win"],
                        "podium_raw": p["podium"],
                        "is_completed": inp.athlete_id in completed_ids,
                    }
                )
            for i, row in enumerate(projection_rows):
                row["proj_rank"] = i + 1

        ticker = _ticker_context(session)

        # Issue #23: capture livestream_url inside the session so detached-
        # instance access can't bite us if a refresh happens above.
        raw_livestream_url = event.livestream_url

    initial_athletes = {
        row["athlete_id"]: {
            "name": row["name"],
            "rank": row["rank"],
            "score": row["score"],
            "round_type": row["round_type"],
        }
        for row in leaderboard_rows
    }

    # Issue #23: optional YouTube live-stream embed. Defense-in-depth:
    # only youtube.com / youtu.be URLs survive validation; anything else
    # collapses to None and the template renders no iframe.
    embed_url = youtube_embed_url(raw_livestream_url)

    ctx = {
        "event": {
            "id": event.id,
            "name": event.name,
            "season": event.season,
            "tier": event.tier.value.replace("_", " ").title(),
            "discipline_label": _DISC_LABEL.get(
                event.discipline, event.discipline.value
            ),
        },
        "gender": gender_enum.value,
        "gender_label": "Men" if gender_enum.value == "M" else "Women",
        "available_genders": available_genders,
        "leaderboard": leaderboard_rows,
        "projections": projection_rows,
        "pre_event": pre_event,
        "from_likely_roster": from_likely_roster,
        "stream_url": f"/live/{event_id}/stream",
        "livestream_embed_url": embed_url,
        "livestream_watch_url": raw_livestream_url if embed_url else None,
        "initial_athletes": initial_athletes,
        **ticker,
        **_nav_context("live"),
    }
    return t.TemplateResponse(request, "live.html", ctx)


@router.get("/live/{event_id}/projections.json")
async def live_projections_json(event_id: int, gender: str = "M"):
    """Return current projection data for a live event as JSON.

    The SSE consumer in templates/live.html calls this endpoint so the
    projection panel can refresh with a small JSON payload (~1-5 KB)
    instead of re-fetching the full HTML page (~50-100 KB).

    Response shape::

        {
          "rows": [
            {
              "athlete_id": 61,
              "name": "Janja Garnbret",
              "mu": 2891.4,
              "win": "84.2",
              "podium": "97.6",
              "expected_rank": 1.18,
              "is_completed": false,
              "proj_rank": 1
            },
            ...
          ]
        }
    """
    with _session() as session:
        event = session.get(Event, event_id)
        if not event:
            return JSONResponse({"error": "Event not found"}, status_code=404)

        try:
            gender_enum = Gender(gender.upper())
        except ValueError:
            gender_enum = Gender.M

        available_gender_enums = sorted({rnd.gender for rnd in event.rounds})
        if available_gender_enums and gender_enum not in available_gender_enums:
            gender_enum = available_gender_enums[0]

        _rt_order = {
            RoundType.FINAL: 0,
            RoundType.SEMI: 1,
            RoundType.QUALIFICATION: 2,
        }
        sorted_rounds = sorted(
            [r for r in event.rounds if r.gender == gender_enum],
            key=lambda r: _rt_order.get(r.round_type, 99),
        )

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
                    }

        leaderboard_rows = sorted(
            athlete_best.values(),
            key=lambda r: r["rank"] if r["rank"] is not None else 9999,
        )

        pre_event = len(leaderboard_rows) == 0
        proj_athlete_ids: list[int] = []

        if pre_event:
            _roster_cache_key = (
                f"roster:{event.discipline.value}:{event.season}:{gender_enum.value}"
            )
            cached_roster = likely_roster_cache.get(_roster_cache_key)
            if cached_roster is None:
                cached_roster = likely_competitors(
                    session, event.discipline, event.season, gender_enum
                )
                likely_roster_cache.set(_roster_cache_key, cached_roster)
            proj_athlete_ids = list(
                cached_roster[:_V2_MAX_ATHLETES_PER_PROJECTION_CARD]
            )

        completed: list[tuple[AthleteProjectionInput, int]] = []
        remaining: list[AthleteProjectionInput] = []

        if pre_event and proj_athlete_ids:
            for inp in _build_proj_inputs_batched(
                session, proj_athlete_ids, event.discipline
            ):
                remaining.append(inp)
        else:
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
                    mu, sigma = rating.mu, sigma_now(rating.sigma, rating.last_event_at)
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
            completed_ids = {inp.athlete_id for inp, _ in completed}
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
                        "win": f"{p['win'] * 100:.1f}",
                        "podium": f"{p['podium'] * 100:.1f}",
                        "expected_rank": round(p["expected_rank"], 2),
                        "is_completed": inp.athlete_id in completed_ids,
                    }
                )
            for i, row in enumerate(projection_rows):
                row["proj_rank"] = i + 1

    return JSONResponse({"rows": projection_rows})


# ---------------------------------------------------------------------------
# GET /predictions  — upcoming-event predictions hub
# ---------------------------------------------------------------------------

_V2_PREDICTIONS_DISCIPLINES = [
    ("lead", "Lead", Discipline.LEAD),
    ("boulder", "Boulder", Discipline.BOULDER),
    ("speed", "Speed", Discipline.SPEED),
]
_V2_MAX_UPCOMING_PER_DISCIPLINE = 50
_V2_MAX_ATHLETES_PER_PROJECTION_CARD = 64
# The /predictions HTML route only renders top-3 win/podium %, which is stable
# at far fewer Monte Carlo draws than the public REST API uses. Lowering the
# per-card sim count from 10k to 2k cuts cold-cache render time (#97) with no
# visible change to the displayed percentages. The REST API
# (POST /api/v1/projections) keeps the full 10k for numerical precision.
_V2_PAGE_SIM_COUNT = 2_000


@router.get("/predictions", response_class=HTMLResponse)
async def v2_predictions(request: Request):
    """List upcoming World Cup events with ELO-based outcome predictions.

    Uses cached Monte Carlo simulations + likely-roster fallback when the
    registered athlete list is not yet published.
    """
    t = _templates(request)
    today = date.today()
    grouped: list[dict] = []

    with _session() as session:
        for disc_key, disc_label, disc_enum in _V2_PREDICTIONS_DISCIPLINES:
            stmt = (
                select(Event)
                .where(
                    Event.discipline == disc_enum,
                    Event.start_date >= today,
                )
                .order_by(Event.start_date.asc())
                .limit(_V2_MAX_UPCOMING_PER_DISCIPLINE)
            )
            upcoming_events = list(session.execute(stmt).scalars())

            disc_events: list[dict] = []
            for ev in upcoming_events:
                result_count = session.execute(
                    select(func.count(Result.id))
                    .join(Round, Result.round_id == Round.id)
                    .where(Round.event_id == ev.id)
                ).scalar_one()

                has_athletes = result_count > 0
                from_likely_roster = False
                predictions_data: dict | None = None

                if has_athletes:
                    gender_predictions: list[dict] = []
                    available_genders = sorted({rnd.gender for rnd in ev.rounds})

                    for gender_enum in available_genders:
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

                        # One batched join instead of N+1 per athlete.
                        proj_inputs: list[AthleteProjectionInput] = (
                            _build_proj_inputs_batched(session, athlete_ids, disc_enum)
                        )

                        if len(proj_inputs) > _V2_MAX_ATHLETES_PER_PROJECTION_CARD:
                            proj_inputs = sorted(
                                proj_inputs, key=lambda a: a.mu, reverse=True
                            )[:_V2_MAX_ATHLETES_PER_PROJECTION_CARD]

                        _fp = ":".join(
                            f"{a.athlete_id},{a.mu:.2f},{a.sigma:.2f}"
                            for a in sorted(proj_inputs, key=lambda a: a.athlete_id)
                        )
                        _cache_key = (
                            f"projections:event:{ev.id}"
                            f":disc:{disc_enum.value}"
                            f":gender:{gender_enum.value}"
                            f":n:{_V2_PAGE_SIM_COUNT}"
                            f":athletes:{_fp}"
                        )
                        probs = predictions_cache.get(_cache_key)
                        if probs is None:
                            probs = compute_podium_probabilities(
                                proj_inputs, n_simulations=_V2_PAGE_SIM_COUNT
                            )
                            predictions_cache.set(_cache_key, probs)

                        ranked = sorted(
                            proj_inputs,
                            key=lambda a: probs[a.athlete_id]["expected_rank"],
                        )
                        top3 = [
                            {
                                "athlete_id": a.athlete_id,
                                "name": a.name,
                                "win": f"{probs[a.athlete_id]['win'] * 100:.1f}",
                                "podium": f"{probs[a.athlete_id]['podium'] * 100:.1f}",
                                "expected_rank": probs[a.athlete_id]["expected_rank"],
                            }
                            for a in ranked[:3]
                        ]
                        gender_predictions.append(
                            {
                                "gender": gender_enum.value,
                                "gender_label": (
                                    "Men" if gender_enum.value == "M" else "Women"
                                ),
                                "top3": top3,
                                "total_athletes": len(proj_inputs),
                            }
                        )

                    predictions_data = {"genders": gender_predictions}

                else:
                    # Likely-competitor fallback.
                    gender_predictions_fb: list[dict] = []
                    for gender_enum in [Gender.M, Gender.F]:
                        _roster_key = (
                            f"roster:{disc_enum.value}:{ev.season}:{gender_enum.value}"
                        )
                        roster_ids = likely_roster_cache.get(_roster_key)
                        if roster_ids is None:
                            roster_ids = likely_competitors(
                                session, disc_enum, ev.season, gender_enum
                            )
                            likely_roster_cache.set(_roster_key, roster_ids)

                        if not roster_ids:
                            continue

                        # One batched join instead of N+1 per athlete.
                        proj_inputs_fb: list[AthleteProjectionInput] = (
                            _build_proj_inputs_batched(
                                session,
                                roster_ids[:_V2_MAX_ATHLETES_PER_PROJECTION_CARD],
                                disc_enum,
                            )
                        )

                        if len(proj_inputs_fb) < 2:
                            continue

                        _fp_fb = ":".join(
                            f"{a.athlete_id},{a.mu:.2f},{a.sigma:.2f}"
                            for a in sorted(proj_inputs_fb, key=lambda a: a.athlete_id)
                        )
                        _cache_key_fb = (
                            f"projections:likely:{ev.id}"
                            f":disc:{disc_enum.value}"
                            f":gender:{gender_enum.value}"
                            f":n:{_V2_PAGE_SIM_COUNT}"
                            f":athletes:{_fp_fb}"
                        )
                        probs_fb = predictions_cache.get(_cache_key_fb)
                        if probs_fb is None:
                            probs_fb = compute_podium_probabilities(
                                proj_inputs_fb, n_simulations=_V2_PAGE_SIM_COUNT
                            )
                            predictions_cache.set(_cache_key_fb, probs_fb)

                        ranked_fb = sorted(
                            proj_inputs_fb,
                            key=lambda a: probs_fb[a.athlete_id]["expected_rank"],
                        )
                        top3_fb = [
                            {
                                "athlete_id": a.athlete_id,
                                "name": a.name,
                                "win": f"{probs_fb[a.athlete_id]['win'] * 100:.1f}",
                                "podium": (
                                    f"{probs_fb[a.athlete_id]['podium'] * 100:.1f}"
                                ),
                                "expected_rank": probs_fb[a.athlete_id][
                                    "expected_rank"
                                ],
                            }
                            for a in ranked_fb[:3]
                        ]
                        gender_predictions_fb.append(
                            {
                                "gender": gender_enum.value,
                                "gender_label": (
                                    "Men" if gender_enum.value == "M" else "Women"
                                ),
                                "top3": top3_fb,
                                "total_athletes": len(proj_inputs_fb),
                            }
                        )

                    if gender_predictions_fb:
                        predictions_data = {"genders": gender_predictions_fb}
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

        ticker = _ticker_context(session)

    ctx = {
        "grouped": grouped,
        "today": str(today),
        "mode": "hub",
        **ticker,
        **_nav_context("predictions"),
    }
    return t.TemplateResponse(request, "predictions.html", ctx)


# ---------------------------------------------------------------------------
# GET /predictions/{event_id}  — per-event recap (frozen forecast vs actual)
# or forward-projection mode when the event is still upcoming.
# ---------------------------------------------------------------------------


@router.get("/predictions/{event_id}", response_class=HTMLResponse)
async def v2_predictions_event(request: Request, event_id: int):
    """Per-event prediction view.

    When the event has at least one FINAL-round result AND a stored
    :class:`EventForecast` set, render in **recap mode** using the frozen
    forecast rows + score row — no fresh Monte Carlo is run. Otherwise fall
    back to **forward mode** (run a Monte Carlo over the current rating
    distribution, same shape as the ``/projections`` page).
    """
    t = _templates(request)

    with _session() as session:
        event = session.get(Event, event_id)
        if not event:
            return HTMLResponse("Event not found", status_code=404)

        finished = _event_has_final_results(session, event)
        # Forecast panels: a list per gender with predicted top-3 + actual.
        recap_panels = _build_forecast_recap(session, event)

        mode = "recap" if (finished and recap_panels) else "forward"

        forward_genders: list[dict] = []
        if mode == "forward":
            # Forward projection — mirror the predictions hub card logic for
            # one event. Pull the top-N current-rating athletes per gender
            # (using likely roster if no athletes are registered yet) and run
            # compute_podium_probabilities.
            for gender_enum in (Gender.M, Gender.F):
                # Try to source athletes from the stored qualification round
                # first — if the event already has registered athletes.
                athlete_ids: list[int] = []
                seen: set[int] = set()
                for rnd in event.rounds:
                    if rnd.gender != gender_enum:
                        continue
                    res_list = list(
                        session.execute(
                            select(Result).where(
                                Result.round_id == rnd.id,
                                Result.dns.is_(False),
                            )
                        ).scalars()
                    )
                    for res in res_list:
                        if res.athlete_id not in seen:
                            athlete_ids.append(res.athlete_id)
                            seen.add(res.athlete_id)

                if not athlete_ids:
                    # Likely-roster fallback.
                    roster_ids = likely_competitors(
                        session, event.discipline, event.season, gender_enum
                    )
                    athlete_ids = list(roster_ids)[:32]

                if not athlete_ids:
                    continue

                proj_inputs = _build_proj_inputs_batched(
                    session, athlete_ids, event.discipline
                )
                if len(proj_inputs) < 2:
                    continue

                probs = compute_podium_probabilities(proj_inputs, n_simulations=5_000)
                ranked = sorted(
                    proj_inputs,
                    key=lambda a: probs[a.athlete_id]["expected_rank"],
                )
                top_rows = [
                    {
                        "athlete_id": a.athlete_id,
                        "name": a.name,
                        "win": f"{probs[a.athlete_id]['win'] * 100:.1f}",
                        "podium": f"{probs[a.athlete_id]['podium'] * 100:.1f}",
                    }
                    for a in ranked[:8]
                ]
                forward_genders.append(
                    {
                        "gender": gender_enum.value,
                        "gender_label": _GENDER_LABEL.get(
                            gender_enum, gender_enum.value
                        ),
                        "top_rows": top_rows,
                        "total_athletes": len(proj_inputs),
                    }
                )

        disc_label = _DISC_LABEL.get(event.discipline, event.discipline.value)
        ticker = _ticker_context(session)

    ctx = {
        "mode": mode,
        "event": {
            "id": event.id,
            "name": event.name,
            "season": event.season,
            "tier": event.tier.value.replace("_", " ").title(),
            "date": str(event.start_date),
            "discipline_label": disc_label,
        },
        "recap_panels": recap_panels,
        "forward_genders": forward_genders,
        **ticker,
        **_nav_context("predictions"),
    }
    return t.TemplateResponse(request, "predictions.html", ctx)


# ---------------------------------------------------------------------------
# GET /model-performance  — rolling Brier / hit-rate dashboard.
# ---------------------------------------------------------------------------


# Aliases for the discipline pill filter (?discipline=lead|boulder|speed|all).
_MODEL_PERF_DISC_ALIASES: dict[str, Discipline] = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
}


def _build_query_string(
    *,
    season: int,
    discipline: str | None,
    gender: str | None,
    include_backfill: bool,
) -> str:
    """Render the /model-performance filter set as a query string (#139)."""
    parts = [f"season={season}"]
    if discipline:
        parts.append(f"discipline={discipline}")
    if gender:
        parts.append(f"gender={gender}")
    if include_backfill:
        parts.append("include_backfill=1")
    return "?" + "&".join(parts)


def _count_alt_lane_scores(
    session,
    *,
    season: int,
    discipline: Discipline | None,
    gender: Gender | None,
    include_backfill: bool,
) -> int:
    """Count EventForecastScore rows in the opposite is_backfill lane (#139).

    Used to power the empty-state branch on /model-performance: when the
    live lane is empty for a freshly-finished season, surface a one-click
    link to flip the toggle and reveal retro-replay rows (and vice versa).
    """
    if not _forecast_tables_present(session):
        return 0
    stmt = (
        select(func.count(EventForecastScore.id))
        .join(Event, EventForecastScore.event_id == Event.id)
        .where(
            EventForecastScore.is_backfill == (not include_backfill),
            Event.season == season,
        )
    )
    if discipline is not None:
        stmt = stmt.where(Event.discipline == discipline)
    else:
        stmt = stmt.where(Event.discipline.in_([Discipline.LEAD, Discipline.BOULDER]))
    if gender is not None:
        stmt = stmt.where(EventForecastScore.gender == gender)
    return session.scalar(stmt) or 0


def _aggregate_model_performance(
    session,
    *,
    season: int,
    discipline: Discipline | None,
    gender: Gender | None,
    include_backfill: bool,
) -> dict:
    """Aggregate :class:`EventForecastScore` rows for the model-performance page.

    Mirrors the v1 ``/model-performance`` endpoint's query shape so the same
    numbers surface in both places (single source of truth — they share this
    helper). Returns a dict with ``aggregates`` (means + intersection rate),
    ``events`` (per-event rows linkable to the event detail page), and ``n``.
    """
    stmt = (
        select(EventForecastScore, Event)
        .join(Event, EventForecastScore.event_id == Event.id)
        .where(
            EventForecastScore.is_backfill == include_backfill,
            Event.season == season,
        )
    )
    if discipline is not None:
        stmt = stmt.where(Event.discipline == discipline)
    else:
        # Speed excluded from the default aggregate — the projection layer's
        # finishing-order MC is a poor fit for Speed's bracket format. See
        # #132 for the bracket-native Speed forecast follow-up.
        stmt = stmt.where(Event.discipline.in_([Discipline.LEAD, Discipline.BOULDER]))
    if gender is not None:
        stmt = stmt.where(EventForecastScore.gender == gender)

    if not _forecast_tables_present(session):
        return {"aggregates": {}, "events": [], "n": 0}

    rows = list(session.execute(stmt.order_by(Event.start_date.desc())).all())
    n = len(rows)

    if n == 0:
        return {"aggregates": {}, "events": [], "n": 0}

    brier_podium = [s.brier_podium for s, _ in rows]
    brier_win = [s.brier_win for s, _ in rows]
    logloss_podium = [s.logloss_podium for s, _ in rows]
    logloss_win = [s.logloss_win for s, _ in rows]
    top3_total = sum(s.top3_intersection for s, _ in rows)

    aggregates = {
        "brier_podium_mean": sum(brier_podium) / n,
        "brier_win_mean": sum(brier_win) / n,
        "logloss_podium_mean": sum(logloss_podium) / n,
        "logloss_win_mean": sum(logloss_win) / n,
        "top3_intersection_rate": top3_total / (3 * n),
    }

    events = [
        {
            "event_id": ev.id,
            "name": ev.name,
            "date": str(ev.start_date),
            "discipline": _DISC_LABEL.get(ev.discipline, ev.discipline.value),
            "gender": score.gender.value,
            "gender_label": _GENDER_LABEL.get(score.gender, score.gender.value),
            "brier_podium": score.brier_podium,
            "brier_win": score.brier_win,
            "top3_intersection": score.top3_intersection,
            "n_athletes": score.n_athletes,
            "is_backfill": score.is_backfill,
        }
        for score, ev in rows
    ]

    return {"aggregates": aggregates, "events": events, "n": n}


@router.get("/model-performance", response_class=HTMLResponse)
async def v2_model_performance(
    request: Request,
    season: Optional[int] = Query(None, ge=2000, le=2100),
    discipline: Optional[str] = Query(None),
    gender: Optional[str] = Query(None),
    include_backfill: bool = Query(False),
):
    """Rolling model-performance dashboard.

    Defaults: ``season = current year``, ``include_backfill = False`` — per
    the plan the public surface shows only this season's live forecasts.
    Historical retro-replay rows are revealed only when ``include_backfill``
    is explicitly set.
    """
    t = _templates(request)

    today = date.today()
    effective_season = season if season is not None else today.year

    disc_enum: Discipline | None = None
    disc_label_active = "All"
    if discipline:
        candidate = _MODEL_PERF_DISC_ALIASES.get(discipline.lower())
        # Quietly fall back to "all" on unknown values rather than 422 — this
        # is the public HTML route, not the API.
        if candidate is not None:
            disc_enum = candidate
            disc_label_active = _DISC_LABEL.get(candidate, discipline.title())

    gen_enum: Gender | None = None
    gender_label_active = "All"
    if gender:
        if gender.upper() == "M":
            gen_enum = Gender.M
            gender_label_active = "Men"
        elif gender.upper() == "F":
            gen_enum = Gender.F
            gender_label_active = "Women"

    with _session() as session:
        data = _aggregate_model_performance(
            session,
            season=effective_season,
            discipline=disc_enum,
            gender=gen_enum,
            include_backfill=include_backfill,
        )
        # Empty-state UX (#139): when the active lane is empty, surface how
        # many rows would appear with the include_backfill toggle flipped so
        # the empty state can offer a one-click rescue.
        n_alt_lane = (
            _count_alt_lane_scores(
                session,
                season=effective_season,
                discipline=disc_enum,
                gender=gen_enum,
                include_backfill=include_backfill,
            )
            if data["n"] == 0
            else 0
        )
        ticker = _ticker_context(session)

    # Season dropdown — last 3 calendar years incl. current. If a user picks a
    # season outside that window via the query string, include it too so the
    # current selection stays in the list.
    season_choices: list[int] = [today.year, today.year - 1, today.year - 2]
    if effective_season not in season_choices:
        season_choices.append(effective_season)
        season_choices.sort(reverse=True)

    # Pill filter state for the discipline.
    # Speed intentionally omitted — the projection layer uses a finishing-order
    # MC that's a poor approximation for Speed's bracket format. See #132 for
    # the planned bracket-native Speed forecast model; re-add the pill once
    # that lands.
    disc_pills = [
        {"key": "", "label": "All", "active": disc_enum is None},
        {
            "key": "lead",
            "label": "Lead",
            "active": disc_enum == Discipline.LEAD,
        },
        {
            "key": "boulder",
            "label": "Boulder",
            "active": disc_enum == Discipline.BOULDER,
        },
    ]

    ctx = {
        "season": effective_season,
        "season_choices": season_choices,
        "discipline_label_active": disc_label_active,
        "gender_label_active": gender_label_active,
        "disc_pills": disc_pills,
        "active_gender": gen_enum.value if gen_enum else "",
        "include_backfill": include_backfill,
        "aggregates": data["aggregates"],
        "events": data["events"],
        "n_events": data["n"],
        "n_alt_lane": n_alt_lane,
        # Preserve current filters on the "flip toggle" empty-state link.
        "alt_lane_qs": _build_query_string(
            season=effective_season,
            discipline=discipline,
            gender=gender,
            include_backfill=not include_backfill,
        ),
        **ticker,
        **_nav_context("model_performance"),
    }
    return t.TemplateResponse(request, "model_performance.html", ctx)
