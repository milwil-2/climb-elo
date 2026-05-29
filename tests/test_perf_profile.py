"""Unit tests for the pure helpers in scripts/perf_profile.py (#97).

Deliberately network-free and server-free: these exercise only the
``percentile`` / ``summarize`` statistics functions on fixed inputs so CI
stays fast and deterministic. No timing-based assertions (those would be
flaky).
"""

from __future__ import annotations

import math

import pytest

from scripts.perf_profile import percentile, summarize


# ---------------------------------------------------------------------------
# percentile()
# ---------------------------------------------------------------------------


def test_percentile_single_sample():
    assert percentile([42.0], 50.0) == 42.0
    assert percentile([42.0], 0.0) == 42.0
    assert percentile([42.0], 100.0) == 42.0


def test_percentile_min_and_max_bounds():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(samples, 0.0) == 10.0
    assert percentile(samples, 100.0) == 50.0


def test_percentile_median_odd_count():
    # 5 evenly spaced points -> median is the middle element.
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50.0) == 30.0


def test_percentile_median_even_count_interpolates():
    # 4 points -> p50 lands between the 2nd and 3rd via linear interpolation.
    assert percentile([10.0, 20.0, 30.0, 40.0], 50.0) == 25.0


def test_percentile_interpolation_quartile():
    # Linear-interpolation method (numpy default): p25 of 1..5 is 2.0.
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 25.0) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 75.0) == 4.0


def test_percentile_is_order_independent():
    a = percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50.0)
    b = percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0)
    assert a == b == 3.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50.0)


@pytest.mark.parametrize("bad_pct", [-1.0, 100.1, 1000.0])
def test_percentile_out_of_range_raises(bad_pct):
    with pytest.raises(ValueError):
        percentile([1.0, 2.0, 3.0], bad_pct)


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


def test_summarize_fixed_list():
    stats = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p50"] == 30.0
    assert math.isclose(stats["mean"], 30.0)


def test_summarize_single_sample():
    stats = summarize([7.5])
    assert stats == {"min": 7.5, "p50": 7.5, "mean": 7.5, "max": 7.5}


def test_summarize_unsorted_input():
    stats = summarize([50.0, 10.0, 30.0, 20.0, 40.0])
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p50"] == 30.0


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])
