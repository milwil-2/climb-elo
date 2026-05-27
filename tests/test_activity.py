"""Tests for ``climbing_elo.engine.activity`` (Issue #91 — Gap 2 from #88).

The classifier is a pure function:

    is_likely_retired_simple(last_event_at, retired_at, today=None,
                             threshold_years=3.0) -> bool

These tests exercise every branch of the truth table:

    | retired_at | last_event_at      | expected |
    |------------|--------------------|----------|
    | set        | any                | True     |
    | None       | None               | False    |
    | None       | recent (< thresh)  | False    |
    | None       | old (>= thresh)    | True     |

Plus a few edge cases around the threshold boundary and a custom override.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from climbing_elo.engine.activity import (
    INACTIVE_THRESHOLD_MONTHS,
    RETIRED_THRESHOLD_YEARS,
    is_likely_retired_simple,
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
