"""Projection-time forward σ inflation to the event date (Issue #170).

Before #170, ``_build_proj_inputs_batched`` inflated σ only up to
``date.today()``, so two future events on the same roster produced
identical Monte Carlo draws.  The fix threads the target event's
``start_date`` through as ``sigma_now``'s reference date; farther-future
events get wider σ → flatter podium distributions.

These tests cover the two layers:

1. ``_build_proj_inputs_batched`` respects ``projection_date`` (unit).
2. End-to-end: same roster + two ``projection_date`` values produce
   distinct podium distributions, with the top athlete's win probability
   strictly lower for the farther-future event (integration).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from climbing_elo.api.routes import _build_proj_inputs_batched
from climbing_elo.engine.activity import sigma_now
from climbing_elo.engine.elo import GLICKO2_INACTIVITY_GRACE_DAYS
from climbing_elo.engine.projections import compute_podium_probabilities
from climbing_elo.models import Athlete, Discipline, Gender, Rating


@pytest.fixture
def rated_athletes(db_session: Session) -> tuple[list[int], date]:
    """Seed 8 athletes with LEAD ratings whose last_event_at is >> grace period ago.

    Returns (athlete_ids, reference_last_event_date).  The gap between
    last_event_at and any projection_date >> grace + 30d is what makes
    ``sigma_now`` visibly diverge across the two projection dates the
    tests use.
    """
    last_event = date(2026, 1, 1)
    mus = [1750.0, 1700.0, 1680.0, 1650.0, 1620.0, 1600.0, 1570.0, 1540.0]
    ids: list[int] = []
    for i, mu in enumerate(mus):
        ath = Athlete(name=f"A{i}", gender=Gender.M)
        db_session.add(ath)
        db_session.flush()
        db_session.add(
            Rating(
                athlete_id=ath.id,
                discipline=Discipline.LEAD,
                mu=mu,
                sigma=100.0,
                n_events=10,
                provisional=False,
                last_event_at=last_event,
            )
        )
        ids.append(ath.id)
    db_session.flush()
    return ids, last_event


def test_build_proj_inputs_batched_respects_projection_date(
    db_session: Session, rated_athletes: tuple[list[int], date]
) -> None:
    """A later ``projection_date`` yields a strictly larger σ per athlete.

    Both dates sit well past the Glicko-2 grace window (``last_event`` is
    the fixture's Jan-1-2026 anchor), so both sides inflate — the later
    date just inflates further.
    """
    ids, last_event = rated_athletes
    near = last_event + timedelta(days=GLICKO2_INACTIVITY_GRACE_DAYS + 60)
    far = last_event + timedelta(days=GLICKO2_INACTIVITY_GRACE_DAYS + 60 + 90)

    inputs_near = _build_proj_inputs_batched(
        db_session, ids, Discipline.LEAD, projection_date=near
    )
    inputs_far = _build_proj_inputs_batched(
        db_session, ids, Discipline.LEAD, projection_date=far
    )

    assert len(inputs_near) == len(inputs_far) == len(ids)
    for near_in, far_in in zip(inputs_near, inputs_far, strict=True):
        assert near_in.athlete_id == far_in.athlete_id
        assert far_in.sigma > near_in.sigma
        # Sanity: the stored σ is 100; the near projection must have
        # already inflated past that.
        assert near_in.sigma > 100.0
        # And the near σ must equal the sigma_now Wiener step directly —
        # the batched helper is a thin wrapper around the same call.
        assert near_in.sigma == pytest.approx(sigma_now(100.0, last_event, today=near))


def test_build_proj_inputs_batched_default_matches_today(
    db_session: Session, rated_athletes: tuple[list[int], date]
) -> None:
    """Omitting ``projection_date`` falls back to ``date.today()`` (pre-#170)."""
    ids, _ = rated_athletes
    default_out = _build_proj_inputs_batched(db_session, ids, Discipline.LEAD)
    today_out = _build_proj_inputs_batched(
        db_session, ids, Discipline.LEAD, projection_date=date.today()
    )
    for a, b in zip(default_out, today_out, strict=True):
        assert a.sigma == pytest.approx(b.sigma)


def test_farther_future_event_produces_flatter_podium(
    db_session: Session, rated_athletes: tuple[list[int], date]
) -> None:
    """The top athlete's win prob is *strictly lower* for a farther-future event.

    This is the observable symptom the #170 fix targets: two future
    events on the same roster used to produce identical projections
    because σ was frozen at ``date.today()``.  With projection_date
    threading, the later event's σ is wider → the top athlete's win
    probability compresses toward the population mean.
    """
    ids, last_event = rated_athletes
    near = last_event + timedelta(days=GLICKO2_INACTIVITY_GRACE_DAYS + 60)
    far = last_event + timedelta(days=GLICKO2_INACTIVITY_GRACE_DAYS + 60 + 180)

    inputs_near = _build_proj_inputs_batched(
        db_session, ids, Discipline.LEAD, projection_date=near
    )
    inputs_far = _build_proj_inputs_batched(
        db_session, ids, Discipline.LEAD, projection_date=far
    )

    probs_near = compute_podium_probabilities(
        inputs_near, n_simulations=20_000, rng_seed=42
    )
    probs_far = compute_podium_probabilities(
        inputs_far, n_simulations=20_000, rng_seed=42
    )

    top_id = ids[0]  # Highest-μ athlete in the fixture.
    assert probs_far[top_id]["win"] < probs_near[top_id]["win"], (
        "Farther-future event should compress top athlete's win probability "
        f"(got near={probs_near[top_id]['win']:.4f}, "
        f"far={probs_far[top_id]['win']:.4f})"
    )
