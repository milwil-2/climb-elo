"""Tests for ``climbing_elo.engine.activity`` (Issues #91 + #151 PR A).

Two pure functions:

* ``is_likely_retired_simple`` (#91) — boolean classifier driving the
  ``active``/``all`` leaderboard view filter.
* ``sigma_now`` (#151 PR A) — display-time σ inflation that widens the
  confidence band for inactive athletes without writing to the DB.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from climbing_elo.engine.activity import (
    INACTIVE_THRESHOLD_MONTHS,
    RETIRED_THRESHOLD_YEARS,
    is_likely_retired_simple,
    sigma_now,
)
from climbing_elo.engine.elo import (
    GLICKO2_DAYS_PER_MONTH,
    GLICKO2_INACTIVITY_GRACE_DAYS,
    GLICKO2_SIGMA_INACTIVITY,
    SIGMA_CEILING,
    glicko2_inflate_phi,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_threshold_constants_are_sane():
    """Both thresholds are positive and the active window is a year-ish."""
    assert RETIRED_THRESHOLD_YEARS > 0
    assert INACTIVE_THRESHOLD_MONTHS > 0
    # The "active" window must be strictly tighter than the "retired" cliff,
    # otherwise the All-time view ⊃ Active view invariant breaks.
    assert INACTIVE_THRESHOLD_MONTHS / 12.0 < RETIRED_THRESHOLD_YEARS


# ---------------------------------------------------------------------------
# Manual retired_at override
# ---------------------------------------------------------------------------


def test_retired_at_set_returns_true_even_if_last_event_recent():
    """Rule 1: manual override always wins."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=10)  # 10 days ago, very recent
    retired = today - timedelta(days=400)
    assert is_likely_retired_simple(last_event, retired, today=today) is True


def test_retired_at_set_returns_true_even_if_no_last_event():
    """Manual override wins even when last_event_at is None."""
    assert (
        is_likely_retired_simple(None, date(2024, 1, 1), today=date(2026, 5, 26))
        is True
    )


def test_retired_at_overrides_heuristic_threshold():
    """Manual override takes precedence over the heuristic, regardless of date."""
    today = date(2026, 5, 26)
    # Athlete competed yesterday but flagged retired today by news scraper
    yesterday = today - timedelta(days=1)
    assert is_likely_retired_simple(yesterday, today, today=today) is True


# ---------------------------------------------------------------------------
# Never-competed case
# ---------------------------------------------------------------------------


def test_never_competed_returns_false():
    """Rule 2: an athlete who never competed isn't "retired" — they're a
    different problem entirely (a roster ghost). Don't filter them with this
    heuristic."""
    assert is_likely_retired_simple(None, None, today=date(2026, 5, 26)) is False


# ---------------------------------------------------------------------------
# Recent activity
# ---------------------------------------------------------------------------


def test_recent_event_returns_false():
    """Rule 3: an athlete with a last event within the threshold is not
    retired."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=180)  # 6 months ago
    assert is_likely_retired_simple(last_event, None, today=today) is False


def test_event_exactly_one_year_ago_returns_false():
    """One year inactive is well within the 3-year threshold."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=365)
    assert is_likely_retired_simple(last_event, None, today=today) is False


# ---------------------------------------------------------------------------
# Long-inactive: heuristic kicks in
# ---------------------------------------------------------------------------


def test_event_four_years_ago_returns_true():
    """Rule 4: long-inactive athletes are flagged."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=int(4 * 365.25))
    assert is_likely_retired_simple(last_event, None, today=today) is True


def test_sachi_amma_like_case_returns_true():
    """Sachi AMMA last competed in 2015 — definitely retired by 2026."""
    today = date(2026, 5, 26)
    last_event = date(2015, 9, 1)
    assert is_likely_retired_simple(last_event, None, today=today) is True


def test_coxsey_like_case_returns_true():
    """Shauna Coxsey last competed ~2021 — retired by 2026."""
    today = date(2026, 5, 26)
    last_event = date(2021, 6, 1)
    assert is_likely_retired_simple(last_event, None, today=today) is True


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_event_just_past_threshold_returns_true():
    """The threshold check is ``>=``. A gap one day past the integer-rounded
    3-year mark is unambiguously retired."""
    today = date(2026, 5, 26)
    # +2 days past the threshold absorbs the floor() in the days→years
    # conversion so the boundary is definitely crossed.
    days = int(RETIRED_THRESHOLD_YEARS * 365.25) + 2
    last_event = today - timedelta(days=days)
    assert is_likely_retired_simple(last_event, None, today=today) is True


def test_event_well_under_threshold_returns_false():
    """Comfortably inside the threshold → not retired."""
    today = date(2026, 5, 26)
    days = int(RETIRED_THRESHOLD_YEARS * 365.25) - 30  # comfortably under
    last_event = today - timedelta(days=days)
    assert is_likely_retired_simple(last_event, None, today=today) is False


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


def test_custom_threshold_years_respected():
    """Passing a custom threshold changes the cliff."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=int(1.5 * 365.25))
    # Default 3y threshold → not retired
    assert is_likely_retired_simple(last_event, None, today=today) is False
    # Tightened 1y threshold → retired
    assert (
        is_likely_retired_simple(last_event, None, today=today, threshold_years=1.0)
        is True
    )


# ---------------------------------------------------------------------------
# Default today
# ---------------------------------------------------------------------------


def test_default_today_uses_current_date(monkeypatch):
    """When ``today`` is not passed, ``date.today()`` is used."""
    # We can't easily monkeypatch ``date.today`` (immutable C type), but we
    # can assert that calling without today doesn't crash AND that a clearly
    # ancient event is still classified retired.
    very_old = date(2000, 1, 1)
    assert is_likely_retired_simple(very_old, None) is True


# ---------------------------------------------------------------------------
# Type contract — function must accept the documented argument shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "last_event,retired,today,expected",
    [
        (None, None, date(2026, 5, 26), False),
        (None, date(2025, 1, 1), date(2026, 5, 26), True),
        (date(2026, 5, 20), None, date(2026, 5, 26), False),
        (date(2020, 1, 1), None, date(2026, 5, 26), True),
        (date(2020, 1, 1), date(2025, 6, 1), date(2026, 5, 26), True),  # both set
    ],
)
def test_truth_table_parametrised(last_event, retired, today, expected):
    """Parametrised sweep over the full truth table."""
    assert is_likely_retired_simple(last_event, retired, today=today) is expected


# ---------------------------------------------------------------------------
# sigma_now (#151 PR A) — display-time σ inflation
# ---------------------------------------------------------------------------


def test_sigma_now_returns_stored_when_last_event_none():
    """Never-competed athletes (no last_event_at) get no inflation."""
    assert sigma_now(120.0, None, today=date(2026, 5, 26)) == 120.0


def test_sigma_now_same_day_returns_stored():
    """An event today shouldn't widen σ — the grace branch (and the
    ``current_date <= last_event_at`` early-return in the engine) covers it."""
    today = date(2026, 5, 26)
    assert sigma_now(120.0, today, today=today) == 120.0


def test_sigma_now_within_grace_returns_stored():
    """Within ``GLICKO2_INACTIVITY_GRACE_DAYS`` (default 30): no inflation.

    This is the invariant that lets the existing test fixtures
    (last_event_at = today - 30d) keep their literal σ assertions after PR A.
    """
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=GLICKO2_INACTIVITY_GRACE_DAYS)
    assert sigma_now(120.0, last_event, today=today) == pytest.approx(120.0)


def test_sigma_now_one_day_past_grace_inflates():
    """One day past the grace window → σ widens (strictly greater)."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=GLICKO2_INACTIVITY_GRACE_DAYS + 1)
    out = sigma_now(120.0, last_event, today=today)
    assert out > 120.0


def test_sigma_now_matches_wiener_formula():
    """For a known gap, the inflated value matches σ² = σ₀² + σ_inactivity² · months exactly.

    Picks a gap divisible by the day-per-month constant so months_inactive is
    a clean integer.
    """
    today = date(2026, 5, 26)
    months = 12.0
    days = int(round(GLICKO2_DAYS_PER_MONTH * months))
    last_event = today - timedelta(days=days)
    stored = 100.0
    out = sigma_now(stored, last_event, today=today)
    # Engine uses days / GLICKO2_DAYS_PER_MONTH, so reproduce that exactly.
    actual_months = days / GLICKO2_DAYS_PER_MONTH
    expected = math.sqrt(stored**2 + GLICKO2_SIGMA_INACTIVITY**2 * actual_months)
    assert out == pytest.approx(expected, abs=1e-6)


def test_sigma_now_capped_at_ceiling():
    """An athlete years inactive saturates at ``SIGMA_CEILING``."""
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=int(10 * 365.25))
    # Start near the ceiling already so the cap definitely binds.
    out = sigma_now(300.0, last_event, today=today)
    assert out == pytest.approx(SIGMA_CEILING)


def test_sigma_now_monotone_in_gap():
    """σ_now is non-decreasing in the gap (Wiener process is monotone)."""
    today = date(2026, 5, 26)
    sigmas = [
        sigma_now(120.0, today - timedelta(days=d), today=today)
        for d in (0, 30, 60, 180, 365, 730)
    ]
    for a, b in zip(sigmas, sigmas[1:], strict=False):
        assert a <= b + 1e-9


def test_sigma_now_delegates_to_engine():
    """Sanity: the wrapper agrees with ``glicko2_inflate_phi`` exactly.

    If this ever drifts, callers using either function would disagree on the
    same input — the wrapper exists only to provide a clean read-site name +
    default ``today``, not to alter the math.
    """
    today = date(2026, 5, 26)
    last_event = today - timedelta(days=400)
    assert sigma_now(150.0, last_event, today=today) == glicko2_inflate_phi(
        150.0, last_event, today
    )


def test_sigma_now_defaults_today_to_date_today():
    """Calling without ``today`` doesn't crash and uses ``date.today()``."""
    # A clearly-ancient last event must produce a clearly-inflated σ.
    out = sigma_now(120.0, date(2010, 1, 1))
    assert out > 200.0  # ~16 years inactive → near or at the ceiling
