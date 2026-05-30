"""Persistent forecast snapshotting (joyful-swinging-map plan).

Freezes a per-event, per-athlete Monte Carlo prediction snapshot to the
``event_forecasts`` table. Two modes:

* **Live** (``is_backfill=False``) — used by the daily snapshot job for upcoming
  events. Roster is the set of athletes already on a qualification-round result
  if any exist, otherwise the :func:`engine.likely_roster.likely_competitors`
  fallback. Ratings are pulled from the current :class:`Rating` rows.
* **Backfill** (``is_backfill=True``, ``as_of_date`` required) — used by the
  retro-replay script to seed historical model performance. Roster is the
  athletes who actually competed in the event (a non-DNS Result is on file).
  Per-athlete μ/σ are reconstructed from :class:`RatingHistory` strictly before
  ``as_of_date`` (defaulting to μ=1500, σ=350 if no prior history exists).

The function never commits — callers control transaction boundaries.

Speed events use the same finishing-order Monte Carlo as Lead/Boulder for v1.
This is a known approximation (the bracket format is not faithfully modelled)
and is tracked under #56.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    engine_version_tag,
)
from climbing_elo.engine.likely_roster import likely_competitors
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    ProgressionResult,
    default_event_format,
    simulate_event_progression,
)
from climbing_elo.models import (
    Athlete,
    Event,
    EventForecast,
    Gender,
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


def _confirmed_roster(session: Session, event_id: int, gender: Gender) -> list[int]:
    """Athletes already on a qualification-round result for this event+gender."""
    rows = session.execute(
        select(Result.athlete_id)
        .join(Round, Result.round_id == Round.id)
        .where(
            Round.event_id == event_id,
            Round.gender == gender,
            Round.round_type == RoundType.QUALIFICATION,
            Result.dns.is_(False),
        )
        .distinct()
    ).all()
    return [row[0] for row in rows]


def _backfill_roster(session: Session, event_id: int, gender: Gender) -> list[int]:
    """Athletes with any non-DNS Result in this event+gender (any round)."""
    rows = session.execute(
        select(Result.athlete_id)
        .join(Round, Result.round_id == Round.id)
        .where(
            Round.event_id == event_id,
            Round.gender == gender,
            Result.dns.is_(False),
        )
        .distinct()
    ).all()
    return [row[0] for row in rows]


def _rating_as_of(
    session: Session,
    athlete_id: int,
    discipline,
    as_of_date: date,
) -> tuple[float, float]:
    """Reconstruct (μ, σ) for an athlete as of ``as_of_date - 1 day``.

    Returns the ``mu_after``/``sigma_after`` of the latest
    :class:`RatingHistory` row whose event's ``start_date`` is strictly before
    ``as_of_date`` and whose event's discipline matches ``discipline``.
    Falls back to the default seed (μ=1500, σ=350) when no prior row exists.
    """
    row = session.execute(
        select(RatingHistory.mu_after, RatingHistory.sigma_after)
        .join(Round, RatingHistory.round_id == Round.id)
        .join(Event, Round.event_id == Event.id)
        .where(
            RatingHistory.athlete_id == athlete_id,
            Event.discipline == discipline,
            Event.start_date < as_of_date,
        )
        .order_by(Event.start_date.desc(), RatingHistory.id.desc())
        .limit(1)
    ).first()
    if row is None:
        return DEFAULT_MU, DEFAULT_SIGMA
    return float(row[0]), float(row[1])


def _build_projection_inputs(
    session: Session,
    athlete_ids: Iterable[int],
    event: Event,
    *,
    is_backfill: bool,
    as_of_date: date | None,
) -> list[AthleteProjectionInput]:
    inputs: list[AthleteProjectionInput] = []
    athlete_ids = list(athlete_ids)
    if not athlete_ids:
        return inputs

    # Fetch names in one shot for nicer logging / debug parity with the live
    # projection card. Order-preserving lookup map.
    name_rows = session.execute(
        select(Athlete.id, Athlete.name).where(Athlete.id.in_(athlete_ids))
    ).all()
    name_by_id = {row[0]: row[1] for row in name_rows}

    if is_backfill:
        assert as_of_date is not None  # enforced by caller
        for aid in athlete_ids:
            mu, sigma = _rating_as_of(session, aid, event.discipline, as_of_date)
            inputs.append(
                AthleteProjectionInput(
                    athlete_id=aid,
                    mu=mu,
                    sigma=sigma,
                    name=name_by_id.get(aid, ""),
                )
            )
        return inputs

    # Live mode — pull current Rating rows for the event's discipline.
    rating_rows = session.execute(
        select(Rating.athlete_id, Rating.mu, Rating.sigma).where(
            Rating.athlete_id.in_(athlete_ids),
            Rating.discipline == event.discipline,
        )
    ).all()
    rating_by_id = {row[0]: (float(row[1]), float(row[2])) for row in rating_rows}
    for aid in athlete_ids:
        mu, sigma = rating_by_id.get(aid, (DEFAULT_MU, DEFAULT_SIGMA))
        inputs.append(
            AthleteProjectionInput(
                athlete_id=aid,
                mu=mu,
                sigma=sigma,
                name=name_by_id.get(aid, ""),
            )
        )
    return inputs


def _upsert_forecast_row(
    session: Session,
    *,
    event_id: int,
    gender: Gender,
    athlete_id: int,
    is_backfill: bool,
    values: dict,
) -> EventForecast:
    """Upsert a single forecast row keyed on the unique constraint.

    Uses ``INSERT ... ON CONFLICT DO UPDATE`` on Postgres. On SQLite (tests)
    we emulate via SELECT-then-update/insert because the test schema doesn't
    benefit from the dialect-specific RETURNING path. Either way, the row is
    flushed to the session but NOT committed — the caller controls the
    transaction.

    The unique key is ``(event_id, gender, athlete_id, is_backfill,
    engine_version)``. ``engine_version`` is read from ``values`` — re-running
    at the same engine version overwrites in place; re-running at a new
    engine version inserts a fresh row.
    """
    engine_version = values["engine_version"]

    if _is_postgres(session):
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(EventForecast).values(
            event_id=event_id,
            gender=gender,
            athlete_id=athlete_id,
            is_backfill=is_backfill,
            **values,
        )
        update_cols = {
            "prob_qualify": stmt.excluded.prob_qualify,
            "prob_reach_semi": stmt.excluded.prob_reach_semi,
            "prob_reach_final": stmt.excluded.prob_reach_final,
            "prob_podium": stmt.excluded.prob_podium,
            "prob_win": stmt.excluded.prob_win,
            "expected_rank": stmt.excluded.expected_rank,
            "mu_at_forecast": stmt.excluded.mu_at_forecast,
            "sigma_at_forecast": stmt.excluded.sigma_at_forecast,
            "n_simulations": stmt.excluded.n_simulations,
            "roster_source": stmt.excluded.roster_source,
            "generated_at": stmt.excluded.generated_at,
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                "event_id",
                "gender",
                "athlete_id",
                "is_backfill",
                "engine_version",
            ],
            set_=update_cols,
        )
        session.execute(stmt)
        # Read back so the caller has the ORM object.
        row = session.execute(
            select(EventForecast).where(
                EventForecast.event_id == event_id,
                EventForecast.gender == gender,
                EventForecast.athlete_id == athlete_id,
                EventForecast.is_backfill == is_backfill,
                EventForecast.engine_version == engine_version,
            )
        ).scalar_one()
        return row

    # SQLite emulation: lookup, mutate-or-insert.
    existing = session.execute(
        select(EventForecast).where(
            EventForecast.event_id == event_id,
            EventForecast.gender == gender,
            EventForecast.athlete_id == athlete_id,
            EventForecast.is_backfill == is_backfill,
            EventForecast.engine_version == engine_version,
        )
    ).scalar_one_or_none()
    if existing is not None:
        for key, val in values.items():
            setattr(existing, key, val)
        session.flush()
        return existing

    row = EventForecast(
        event_id=event_id,
        gender=gender,
        athlete_id=athlete_id,
        is_backfill=is_backfill,
        **values,
    )
    session.add(row)
    session.flush()
    return row


def snapshot_forecast(
    session: Session,
    event_id: int,
    gender: Gender,
    *,
    is_backfill: bool = False,
    as_of_date: date | None = None,
    n_simulations: int = 10_000,
    rng_seed: int | None = None,
) -> list[EventForecast]:
    """Snapshot a Monte Carlo forecast for a single (event, gender).

    Live mode (``is_backfill=False``):
        Roster = athletes already on a qualification-round Result. If none,
        falls back to :func:`engine.likely_roster.likely_competitors`.
        Ratings come from the current :class:`Rating` rows for the event's
        discipline.

    Backfill mode (``is_backfill=True``, ``as_of_date`` required):
        Roster = athletes who actually competed (any non-DNS Result on file).
        Ratings reconstructed from :class:`RatingHistory` strictly before
        ``as_of_date``; fallback μ=1500, σ=350 if no prior history.

    The sim is run with :func:`engine.projections.simulate_event_progression`
    using :func:`engine.projections.default_event_format` for the event's
    tier.

    One row is upserted per athlete into ``event_forecasts`` keyed on
    ``(event_id, gender, athlete_id, is_backfill)``. Rows are flushed but
    NOT committed — the caller controls transaction boundaries.

    Returns:
        The list of upserted :class:`EventForecast` ORM objects, ordered to
        match the projection sim output (descending μ).
    """
    if is_backfill and as_of_date is None:
        raise ValueError("as_of_date is required when is_backfill=True")

    event = session.get(Event, event_id)
    if event is None:
        raise ValueError(f"event {event_id} not found")

    # --- Roster resolution -------------------------------------------------
    if is_backfill:
        athlete_ids = _backfill_roster(session, event_id, gender)
        roster_source = "backfill"
    else:
        athlete_ids = _confirmed_roster(session, event_id, gender)
        if athlete_ids:
            roster_source = "confirmed"
        else:
            athlete_ids = likely_competitors(
                session,
                discipline=event.discipline,
                season=event.season,
                gender=gender,
            )
            roster_source = "likely"

    if not athlete_ids:
        log.info(
            "snapshot_forecast: no roster for event=%s gender=%s (mode=%s)",
            event_id,
            gender.value,
            "backfill" if is_backfill else "live",
        )
        return []

    # --- Sim inputs --------------------------------------------------------
    inputs = _build_projection_inputs(
        session,
        athlete_ids,
        event,
        is_backfill=is_backfill,
        as_of_date=as_of_date,
    )

    rounds = default_event_format(event.tier.value)

    # --- Monte Carlo -------------------------------------------------------
    results: list[ProgressionResult] = simulate_event_progression(
        inputs,
        rounds=rounds,
        n_simulations=n_simulations,
        rng_seed=rng_seed,
    )

    # --- Upsert -----------------------------------------------------------
    # prob_qualify is plumbed by us (not the sim): 1.0 for confirmed /
    # backfill rosters and also 1.0 for likely rosters in v1 — the likely-
    # roster selector already filters to high-confidence entrants, so we
    # don't have a calibrated probability to inject. Documented in the plan.
    qualify_prob_by_source = {
        "confirmed": 1.0,
        "backfill": 1.0,
        "likely": 1.0,
    }
    base_qualify = qualify_prob_by_source[roster_source]

    mu_sigma_by_id = {inp.athlete_id: (inp.mu, inp.sigma) for inp in inputs}
    version = engine_version_tag()
    generated_at = datetime.now(timezone.utc)

    upserted: list[EventForecast] = []
    for pr in results:
        mu, sigma = mu_sigma_by_id.get(pr.athlete_id, (DEFAULT_MU, DEFAULT_SIGMA))
        # ``expected_rank`` is the true Monte Carlo mean rank computed inside
        # ``simulate_event_progression`` (#122): athletes that reach the final
        # contribute their 1-indexed rank in that round, eliminated athletes
        # contribute a sentinel of ``n + 1``.  Pre-#122 this field stored the
        # monotone proxy ``1 + (n - 1) * (1 - prob_reach_final)``.
        values = {
            "prob_qualify": base_qualify,
            "prob_reach_semi": min(base_qualify, pr.prob_reach_semi),
            "prob_reach_final": min(base_qualify, pr.prob_reach_final),
            "prob_podium": min(base_qualify, pr.final_podium_prob),
            "prob_win": min(base_qualify, pr.final_win_prob),
            "expected_rank": float(pr.expected_rank),
            "mu_at_forecast": mu,
            "sigma_at_forecast": sigma,
            "n_simulations": n_simulations,
            "roster_source": roster_source,
            "engine_version": version,
            "generated_at": generated_at,
        }

        row = _upsert_forecast_row(
            session,
            event_id=event_id,
            gender=gender,
            athlete_id=pr.athlete_id,
            is_backfill=is_backfill,
            values=values,
        )
        upserted.append(row)

    return upserted
