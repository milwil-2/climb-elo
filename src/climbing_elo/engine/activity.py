"""Activity classification + read-time σ inflation (Issue #91 — Gap 2 from #88; Issue #151 PR A).

Glicko-2's σ inflates during inactivity but μ does not decay, so 5-year-absent
athletes still show on the raw leaderboard at their last rating.  The dashboard
addresses this with **two leaderboard views**:

1. ``active`` (default) — competed within the last 12 months.
2. ``all`` — every athlete that the heuristic below does *not* classify as
   retired (i.e. either competed within the last 3 years, or has a manual
   ``retired_at`` override that is NULL).
3. ``legacy`` — debug-only; no filter at all.

This module exposes two **pure-function** helpers:

* :func:`is_likely_retired_simple` — the activity classifier used by the view
  filter (#91).
* :func:`sigma_now` — the display-time σ helper that applies the Glicko-2
  Wiener-process inactivity inflation to a stored σ + ``last_event_at``
  without writing to the DB (#151 PR A).  Inactive athletes used to display
  the σ from their last event indefinitely; ``sigma_now`` widens the band at
  read time so the UI reflects real uncertainty.

Callers (HTML routes, REST routes, tests) provide ``last_event_at`` and
``retired_at`` / ``stored_sigma`` directly; we avoid touching the SQLAlchemy
session here so the heuristics are trivial to swap and to test.

The threshold (3 years) is intentionally generous: it preserves on-break
athletes such as Janja Garnbret (2024 sabbatical) while filtering out
long-term retirees (Coxsey 2021, AMMA 2015).  A follow-up issue is filed for
the age-aware refinement — gated on ``year_of_birth`` coverage (#86).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from climbing_elo.engine.elo import glicko2_inflate_phi

#: For the ``active`` view: athletes whose last event is within this many
#: months are considered active.  Not used by the classifier directly; it is
#: applied as a SQL filter in ``_get_rankings_v2``.
INACTIVE_THRESHOLD_MONTHS = 12

#: For the ``is_likely_retired`` heuristic: a gap of at least this many years
#: since the last competed event flags the athlete as likely retired.
RETIRED_THRESHOLD_YEARS = 3.0


def is_likely_retired_simple(
    last_event_at: Optional[date],
    retired_at: Optional[date],
    today: Optional[date] = None,
    threshold_years: float = RETIRED_THRESHOLD_YEARS,
) -> bool:
    """Return ``True`` if the athlete should be hidden from the All-time view.

    Pure function — no DB access.  Caller supplies ``last_event_at`` (from the
    discipline's ``Rating`` row) and ``retired_at`` (from the ``Athlete`` row).

    Rules, in order:

    1. ``retired_at`` is set → ``True`` (manual override, regardless of date).
    2. ``last_event_at`` is ``None`` → ``False`` (never-competed athletes are
       a different problem; they're not "retired").
    3. ``(today - last_event_at) >= threshold_years`` → ``True``.
    4. Otherwise → ``False``.

    Args:
        last_event_at: Date of the athlete's most recent competed event in the
            relevant discipline, or ``None`` if they've never competed.
        retired_at: Manual override.  Any non-NULL value flags the athlete as
            retired unconditionally.
        today: Reference date (defaults to :func:`date.today`).  Tests should
            pin this to make the threshold deterministic.
        threshold_years: Gap, in years, that flips the heuristic.  Defaults to
            :data:`RETIRED_THRESHOLD_YEARS` (3.0).

    Returns:
        ``True`` if the athlete is classified as likely retired.
    """
    if retired_at is not None:
        return True
    if last_event_at is None:
        return False
    if today is None:
        today = date.today()
    years_inactive = (today - last_event_at).days / 365.25
    return years_inactive >= threshold_years


def sigma_now(
    stored_sigma: float,
    last_event_at: Optional[date],
    today: Optional[date] = None,
) -> float:
    """Return the σ to display today for an athlete whose last event was on ``last_event_at``.

    Issue #151 PR A: applies Glicko-2's Wiener-process inactivity inflation
    (per #89) at read time, so leaderboards, athlete profiles, and projection
    inputs reflect a σ that has grown since the athlete's last event —
    without writing the inflated value back to the DB (no re-backfill, no
    stored drift; ``Rating.sigma`` remains the "last-event σ" of truth).

    Delegates to :func:`climbing_elo.engine.elo.glicko2_inflate_phi`, which
    honours the grace period (``GLICKO2_INACTIVITY_GRACE_DAYS``) and the
    ceiling (``EloConfig.sigma_ceiling``).

    Pure function — no DB access.

    Args:
        stored_sigma: σ as written to ``Rating.sigma``.
        last_event_at: Date of the athlete's most recent competed event in
            the relevant discipline.  ``None`` ⇒ returns ``stored_sigma``
            unchanged (never-competed athletes are not "inactive").
        today: Reference date (defaults to :func:`date.today`).  Tests
            should pin this to make the inflation deterministic.

    Returns:
        Display-time σ.  Equal to ``stored_sigma`` for athletes within the
        grace window; strictly greater after the grace; clamped at the
        ceiling.
    """
    if today is None:
        today = date.today()
    return glicko2_inflate_phi(stored_sigma, last_event_at, today)
