"""Tests for the G-Elo (Szczecinski 2022) bucketed-MOV variant (Issue #84).

Each test in this module guards one of the acceptance criteria from the
issue:

* the bucket lookup returns the expected multiplier per representative gap,
* zero-sum on μ when GELo MOV is plugged into ``calculate_round_updates``,
* tier monotonicity is preserved (higher tier ⇒ bigger K, independent of MOV),
* the variant registers itself in ``BACKTEST_VARIANTS`` at import time,
* the default ELO variant is unchanged by the addition of GELo (so existing
  Lead/Boulder/Speed tests produce identical numbers).
"""

from __future__ import annotations

import math
from datetime import date

from climbing_elo.engine.elo import (
    AthleteRating,
    AthleteResult,
    EloConfig,
    calculate_round_updates,
    compute_boulder_margin_multiplier,
    compute_margin_multiplier,
    compute_speed_margin_multiplier,
)
from climbing_elo.engine.evaluation import BACKTEST_VARIANTS
from climbing_elo.engine.gelo import (
    DEFAULT_GELO_BUCKETS,
    GELO_BOULDER_BUCKETS,
    GELO_LEAD_BUCKETS,
    GELO_SPEED_BUCKETS,
    GELoEngine,
    _select_bucket,
    compute_gelo_margin_multiplier,
)
from climbing_elo.models import Discipline, EventTier, RoundType


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_gelo_variant_registered():
    """GELo must register itself in BACKTEST_VARIANTS at import time."""
    assert "gelo" in BACKTEST_VARIANTS
    assert BACKTEST_VARIANTS["gelo"] is GELoEngine


def test_gelo_engine_name():
    """The engine's name() identifier matches the registry slot."""
    # Pass ``None`` for session — name() doesn't touch it.
    e = GELoEngine.__new__(GELoEngine)
    assert e.name() == "gelo"


# ---------------------------------------------------------------------------
# _select_bucket — bucket lookup
# ---------------------------------------------------------------------------


def test_select_bucket_first_bucket():
    """A gap below the first upper bound returns the first multiplier."""
    assert _select_bucket(0.0, GELO_LEAD_BUCKETS) == 1.00
    assert _select_bucket(0.5, GELO_LEAD_BUCKETS) == 1.00


def test_select_bucket_boundary_falls_into_next():
    """A gap equal to ``upper_bound_exclusive`` falls into the *next* bucket."""
    # Lead bucket 0 ends at 1.0 (exclusive).
    assert _select_bucket(1.0, GELO_LEAD_BUCKETS) == 1.15


def test_select_bucket_intermediate():
    """Mid-table buckets are reached for in-range gaps."""
    assert _select_bucket(2.5, GELO_LEAD_BUCKETS) == 1.15
    assert _select_bucket(5.0, GELO_LEAD_BUCKETS) == 1.30


def test_select_bucket_overflow_returns_last():
    """Any gap above the last finite bound falls into the inf bucket."""
    assert _select_bucket(1e9, GELO_LEAD_BUCKETS) == 1.50
    assert _select_bucket(1e9, GELO_BOULDER_BUCKETS) == 1.50
    assert _select_bucket(1e9, GELO_SPEED_BUCKETS) == 1.50


# ---------------------------------------------------------------------------
# compute_gelo_margin_multiplier — representative gaps per discipline
# ---------------------------------------------------------------------------


def test_gelo_lead_per_bucket_multiplier():
    """Lead: representative gaps from each bucket return the right multiplier
    (with rating_gap=0 so the gap-conditioning factor is 1)."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # Bucket 0 (gap < 1.0)
    assert (
        compute_gelo_margin_multiplier(
            10.0, 10.5, Discipline.LEAD, rating_gap=0.0, config=cfg
        )
        == 1.00
    )
    # Bucket 1 (1.0 ≤ gap < 3.0)
    assert (
        compute_gelo_margin_multiplier(
            12.0, 10.0, Discipline.LEAD, rating_gap=0.0, config=cfg
        )
        == 1.15
    )
    # Bucket 2 (3.0 ≤ gap < 10.0)
    assert (
        compute_gelo_margin_multiplier(
            15.0, 10.0, Discipline.LEAD, rating_gap=0.0, config=cfg
        )
        == 1.30
    )
    # Bucket 3 (gap ≥ 10.0)
    assert (
        compute_gelo_margin_multiplier(
            25.0, 10.0, Discipline.LEAD, rating_gap=0.0, config=cfg
        )
        == 1.50
    )


def test_gelo_boulder_per_bucket_multiplier():
    """Boulder: representative gaps from each bucket (rating_gap=0)."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # Bucket 0 (gap < 100): attempt-level differences.
    assert (
        compute_gelo_margin_multiplier(
            2050.0, 2000.0, Discipline.BOULDER, rating_gap=0.0, config=cfg
        )
        == 1.00
    )
    # Bucket 1 (100 ≤ gap < 500): roughly one zone.
    assert (
        compute_gelo_margin_multiplier(
            2200.0, 2000.0, Discipline.BOULDER, rating_gap=0.0, config=cfg
        )
        == 1.15
    )
    # Bucket 2 (500 ≤ gap < 1000): about one top apart.
    assert (
        compute_gelo_margin_multiplier(
            2700.0, 2000.0, Discipline.BOULDER, rating_gap=0.0, config=cfg
        )
        == 1.30
    )
    # Bucket 3 (gap ≥ 1000): two-tops blowout.
    assert (
        compute_gelo_margin_multiplier(
            4000.0, 2000.0, Discipline.BOULDER, rating_gap=0.0, config=cfg
        )
        == 1.50
    )


def test_gelo_speed_per_bucket_multiplier():
    """Speed: representative gaps from each bucket (rating_gap=0)."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # Bucket 0 (gap < 0.05s): photo finish.
    assert (
        compute_gelo_margin_multiplier(
            5.20, 5.22, Discipline.SPEED, rating_gap=0.0, config=cfg
        )
        == 1.00
    )
    # Bucket 1 (0.05 ≤ gap < 0.20).
    assert (
        compute_gelo_margin_multiplier(
            5.20, 5.30, Discipline.SPEED, rating_gap=0.0, config=cfg
        )
        == 1.10
    )
    # Bucket 2 (0.20 ≤ gap < 0.50).
    assert (
        compute_gelo_margin_multiplier(
            5.20, 5.50, Discipline.SPEED, rating_gap=0.0, config=cfg
        )
        == 1.25
    )
    # Bucket 3 (gap ≥ 0.50).
    assert (
        compute_gelo_margin_multiplier(
            5.00, 7.00, Discipline.SPEED, rating_gap=0.0, config=cfg
        )
        == 1.50
    )


def test_gelo_none_score_returns_neutral():
    """Missing scores on either side should yield a no-op multiplier."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    assert (
        compute_gelo_margin_multiplier(None, 10.0, Discipline.LEAD, config=cfg) == 1.0
    )
    assert (
        compute_gelo_margin_multiplier(10.0, None, Discipline.LEAD, config=cfg) == 1.0
    )


def test_gelo_unknown_discipline_returns_neutral():
    """Disciplines without a bucket table (e.g. BOULDER_LEAD aggregate)
    return a neutral 1.0 multiplier — the engine should never crash."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # BOULDER_LEAD is an aggregate rating, not a competition format —
    # no bucket table defined, so the helper returns 1.0.
    mult = compute_gelo_margin_multiplier(
        10.0, 5.0, Discipline.BOULDER_LEAD, rating_gap=0.0, config=cfg
    )
    assert mult == 1.0


def test_gelo_gap_conditioning_still_applied():
    """Favourite-side blowouts should still see the gap-conditioning damp
    on top of the bucket multiplier (538-style behaviour is preserved)."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # Same score gap, but favourite has a big rating advantage.
    base = compute_gelo_margin_multiplier(
        25.0, 10.0, Discipline.LEAD, rating_gap=0.0, config=cfg
    )
    damped = compute_gelo_margin_multiplier(
        25.0, 10.0, Discipline.LEAD, rating_gap=600.0, config=cfg
    )
    # Damping is asymmetric — favourite wins shrink, upsets keep full bonus.
    assert damped < base


# ---------------------------------------------------------------------------
# Dispatch through compute_*_margin_multiplier when gelo_buckets is set
# ---------------------------------------------------------------------------


def test_lead_helper_dispatches_to_gelo_when_buckets_set():
    """When EloConfig.gelo_buckets is populated the Lead helper returns the
    bucketed multiplier instead of the continuous one."""
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # 2-hold gap → Lead bucket 1 → 1.15 (×1.0 gap-conditioning).
    mult = compute_margin_multiplier(
        12.0, 10.0, max_gap=20.0, rating_gap=0.0, config=cfg
    )
    assert mult == 1.15


def test_boulder_helper_dispatches_to_gelo_when_buckets_set():
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # 700-point gap → Boulder bucket 2 → 1.30 (×1.0 gap-conditioning).
    mult = compute_boulder_margin_multiplier(2700.0, 2000.0, rating_gap=0.0, config=cfg)
    assert mult == 1.30


def test_speed_helper_dispatches_to_gelo_when_buckets_set():
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    # 0.30s gap → Speed bucket 2 → 1.25 (×1.0 gap-conditioning).
    mult = compute_speed_margin_multiplier(5.20, 5.50, rating_gap=0.0, config=cfg)
    assert mult == 1.25


# ---------------------------------------------------------------------------
# Default variant unchanged — guards against accidental coupling
# ---------------------------------------------------------------------------


def test_default_lead_multiplier_unchanged_without_buckets():
    """A vanilla EloConfig (no gelo_buckets) must produce the same Lead
    multiplier as before the GELo work landed. Locks the production path."""
    cfg = EloConfig()  # gelo_buckets defaults to None
    # 10-hold gap, no rating gap → continuous formula:
    #   base = min(1 + 10/20, 1.5) = 1.5
    #   gap-conditioning at Δμ=0 → 1.0
    mult = compute_margin_multiplier(
        20.0, 10.0, max_gap=20.0, rating_gap=0.0, config=cfg
    )
    assert math.isclose(mult, 1.5)


def test_default_boulder_multiplier_unchanged_without_buckets():
    cfg = EloConfig()
    # 1500-point gap, no rating gap → continuous formula:
    #   base = min(1 + 1500/1000, 1.5) = 1.5
    mult = compute_boulder_margin_multiplier(2500.0, 1000.0, rating_gap=0.0, config=cfg)
    assert math.isclose(mult, 1.5)


def test_default_speed_multiplier_unchanged_without_buckets():
    cfg = EloConfig()
    # 1.0s gap, no rating gap → continuous formula:
    #   base = min(1 + 1.0/2.0, 1.5) = 1.5
    mult = compute_speed_margin_multiplier(5.0, 6.0, rating_gap=0.0, config=cfg)
    assert math.isclose(mult, 1.5)


# ---------------------------------------------------------------------------
# calculate_round_updates with GELo MOV — invariants
# ---------------------------------------------------------------------------


def _eight_results_with_scores():
    """Eight-athlete final, increasing rank ↔ decreasing score."""
    results = []
    for i in range(1, 9):
        # rank i → score = (9 - i) * 5  (so rank 1 has the highest score).
        results.append(
            AthleteResult(athlete_id=i, rank=i, score_normalized=float((9 - i) * 5))
        )
    return results


def _eight_ratings():
    return {
        i: AthleteRating(
            athlete_id=i,
            mu=1500 + (5 - i) * 30,
            n_events=10,
            provisional=False,
        )
        for i in range(1, 9)
    }


def test_gelo_round_updates_zero_sum():
    """Rating changes must sum to zero across all athletes when MOV is
    bucketed (Szczecinski 2022) — bucketing only modifies the multiplier
    inside the pair update, so the pairwise zero-sum invariant survives."""
    results = _eight_results_with_scores()
    ratings = _eight_ratings()
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
        config=cfg,
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 1e-4


def test_gelo_tier_monotonicity_preserved():
    """Higher tier ⇒ bigger K ⇒ bigger absolute μ change for the winner —
    independent of which MOV variant is in use.

    Same field, same MOV multipliers (bucketed identically), the only
    difference between the two runs is the K-factor table. The
    World Championship Final K (=15) is larger than the World Cup Final K
    (=12), so a winner should move further in the WCh run than in the WC run.
    """
    cfg = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)

    results = _eight_results_with_scores()
    ratings_wc = _eight_ratings()
    ratings_wch = _eight_ratings()

    wc_updates = calculate_round_updates(
        results,
        ratings_wc,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
        config=cfg,
    )
    wch_updates = calculate_round_updates(
        results,
        ratings_wch,
        EventTier.WORLD_CHAMPIONSHIP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
        config=cfg,
    )

    wc_by_id = {u.athlete_id: u for u in wc_updates}
    wch_by_id = {u.athlete_id: u for u in wch_updates}

    # Winner (rank 1, athlete_id 1) should move more in WCh than in WC.
    wc_winner_delta = wc_by_id[1].mu_after - wc_by_id[1].mu_before
    wch_winner_delta = wch_by_id[1].mu_after - wch_by_id[1].mu_before
    assert abs(wch_winner_delta) > abs(wc_winner_delta)


def test_gelo_changes_pair_multiplier_vs_continuous():
    """End-to-end sanity check: a round scored with the continuous MOV and
    the same round scored with the GELo bucketed MOV produce different
    rating deltas. Establishes that the variant *actually does something*
    (regression guard against accidentally wiring the default config in)."""
    results = _eight_results_with_scores()
    ratings_continuous = _eight_ratings()
    ratings_gelo = _eight_ratings()

    cfg_default = EloConfig()  # continuous MOV
    cfg_gelo = EloConfig(gelo_buckets=DEFAULT_GELO_BUCKETS)

    continuous_updates = calculate_round_updates(
        results,
        ratings_continuous,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
        config=cfg_default,
    )
    gelo_updates = calculate_round_updates(
        results,
        ratings_gelo,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
        config=cfg_gelo,
    )

    cont_by_id = {u.athlete_id: u for u in continuous_updates}
    gelo_by_id = {u.athlete_id: u for u in gelo_updates}

    # The winner's delta must differ between the two variants on at least
    # one of the 8 athletes — if they were identical, GELo isn't being
    # applied at all.
    deltas_differ = any(
        abs(
            (cont_by_id[i].mu_after - cont_by_id[i].mu_before)
            - (gelo_by_id[i].mu_after - gelo_by_id[i].mu_before)
        )
        > 1e-6
        for i in range(1, 9)
    )
    assert deltas_differ
