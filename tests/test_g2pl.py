"""Invariant tests for the ``g2pl`` challenger engine.

Each numbered test maps to the "Invariant tests (ship with the
implementation)" list in ``docs/PLAN_CHALLENGER_G2PL.md``:

1. field-size independence (winner Δμ at n=10 vs n=80 within 10%; the
   incumbent engine holds this too since #174 restored its base-K divisor),
2. higher rank ⇒ Δμ monotone within a round (equal φ),
3. σ′ ∈ [floor, ceiling] after every update; φ shrinks on evidence,
4. beating a high-φ opponent moves you less than beating a low-φ one,
5. tie symmetry — equal-φ tied athletes get equal-and-opposite Δμ,
6. determinism — same inputs → same outputs.

Plus registration, config validation, Speed delegation, the
margin-adjusted-outcome algebra, and the drift monitor.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from climbing_elo.engine.elo import (
    AthleteRating,
    AthleteResult,
    _DEFAULT_K_FACTORS,
    calculate_round_updates,
)
from climbing_elo.engine.evaluation import BACKTEST_VARIANTS
from climbing_elo.engine.g2pl import (
    DEFAULT_IMPORTANCE_WEIGHTS,
    FALLBACK_IMPORTANCE_WEIGHT,
    G2PLConfig,
    G2PLEngine,
    calculate_g2pl_round_updates,
    get_importance_weight,
    margin_adjusted_outcome,
    mu_drift,
)
from climbing_elo.models import Discipline, EventTier, RoundType

ROUND_DATE = date(2024, 6, 1)


def _field(n: int, mu: float = 1500.0, sigma: float = 130.0):
    """A rank-ordered field of ``n`` identically rated athletes."""
    results = [
        AthleteResult(athlete_id=i, rank=i, score_normalized=float((n + 1 - i) * 5))
        for i in range(1, n + 1)
    ]
    ratings = {
        i: AthleteRating(
            athlete_id=i,
            mu=mu,
            sigma=sigma,
            n_events=10,
            provisional=False,
            last_event_at=ROUND_DATE,
        )
        for i in range(1, n + 1)
    }
    return results, ratings


def _g2pl_round(results, ratings, **kwargs):
    defaults = dict(
        event_tier=EventTier.WORLD_CUP,
        round_type=RoundType.FINAL,
        event_date=ROUND_DATE,
        discipline=Discipline.LEAD,
    )
    defaults.update(kwargs)
    return calculate_g2pl_round_updates(results, ratings, **defaults)


def _winner_delta(updates):
    by_id = {u.athlete_id: u for u in updates}
    return by_id[1].mu_after - by_id[1].mu_before


# ---------------------------------------------------------------------------
# Registration / config
# ---------------------------------------------------------------------------


def test_g2pl_variant_registered():
    assert "g2pl" in BACKTEST_VARIANTS
    assert BACKTEST_VARIANTS["g2pl"] is G2PLEngine


def test_g2pl_engine_name():
    e = G2PLEngine.__new__(G2PLEngine)
    assert e.name() == "g2pl"


def test_importance_weights_seeded_from_k_ratios():
    """Seed rule: w = K / max(K), so max cell is exactly 1.0 and every cell
    keeps its relative importance."""
    max_k = max(k for row in _DEFAULT_K_FACTORS.values() for k in row.values())
    assert (
        max(w for row in DEFAULT_IMPORTANCE_WEIGHTS.values() for w in row.values())
        == 1.0
    )
    for tier, row in _DEFAULT_K_FACTORS.items():
        for rt, k in row.items():
            assert math.isclose(DEFAULT_IMPORTANCE_WEIGHTS[tier][rt], k / max_k)


def test_importance_weight_fallback_for_unmapped_cell():
    cfg = G2PLConfig(importance_weights={})
    assert (
        get_importance_weight(EventTier.WORLD_CUP, RoundType.FINAL, cfg)
        == FALLBACK_IMPORTANCE_WEIGHT
    )


def test_invalid_mov_mode_rejected():
    with pytest.raises(ValueError, match="mov_mode"):
        G2PLConfig(mov_mode="bogus")


def test_speed_raises_not_implemented():
    results, ratings = _field(4)
    with pytest.raises(NotImplementedError):
        _g2pl_round(results, ratings, discipline=Discipline.SPEED)


def test_degenerate_field_returns_empty():
    results, ratings = _field(1)
    assert _g2pl_round(results, ratings) == []


# ---------------------------------------------------------------------------
# Invariant 1 — field-size independence
# ---------------------------------------------------------------------------


def _winner_delta_ratio_g2pl(config=None):
    small_results, small_ratings = _field(10)
    big_results, big_ratings = _field(80)
    kwargs = {"config": config} if config is not None else {}
    d_small = _winner_delta(_g2pl_round(small_results, small_ratings, **kwargs))
    d_big = _winner_delta(_g2pl_round(big_results, big_ratings, **kwargs))
    return d_big / d_small


def test_field_size_independence_g2pl():
    """Winner Δμ at n=80 vs n=10 (same μ/σ everywhere) within 10%."""
    ratio = _winner_delta_ratio_g2pl()
    assert 0.9 <= ratio <= 1.1, f"winner Δμ ratio n=80/n=10 = {ratio:.3f}"


def test_field_size_ablation_exponent_zero_reintroduces_bias():
    """The design doc's literal per-pair sum (exponent=0) must show the bias
    the default normalization removes — guards against the normalization
    silently becoming a no-op."""
    ratio = _winner_delta_ratio_g2pl(G2PLConfig(field_normalization_exponent=0.0))
    assert ratio > 1.1, f"expected field-size bias at exponent=0, got {ratio:.3f}"


def test_field_size_independence_incumbent():
    """The incumbent engine satisfies invariant #1 as of #174.

    It used to be an expected failure: between #51 and #174 the μ side was a
    flat K·(1−E) accumulation with no field normalization, so deltas scaled
    linearly with the entry list. #174 restored the base-K divisor
    (``EloConfig.k_field_normalization_exponent``), so the incumbent now holds
    the same invariant the g2pl challenger does.
    """
    small_results, small_ratings = _field(10)
    big_results, big_ratings = _field(80)
    d_small = _winner_delta(
        calculate_round_updates(
            small_results,
            small_ratings,
            EventTier.WORLD_CUP,
            RoundType.FINAL,
            ROUND_DATE,
            discipline=Discipline.LEAD,
        )
    )
    d_big = _winner_delta(
        calculate_round_updates(
            big_results,
            big_ratings,
            EventTier.WORLD_CUP,
            RoundType.FINAL,
            ROUND_DATE,
            discipline=Discipline.LEAD,
        )
    )
    ratio = d_big / d_small
    assert 0.9 <= ratio <= 1.1, f"winner Δμ ratio n=80/n=10 = {ratio:.3f}"


# ---------------------------------------------------------------------------
# Invariant 2 — rank monotonicity (equal φ)
# ---------------------------------------------------------------------------


def test_rank_monotone_deltas():
    """With equal μ/σ across the field, a better rank must earn a strictly
    larger μ delta."""
    results, ratings = _field(8)
    updates = _g2pl_round(results, ratings)
    deltas = [
        u.mu_after - u.mu_before
        for u in sorted(updates, key=lambda u: u.athlete_id)  # id == rank here
    ]
    assert all(a > b for a, b in zip(deltas, deltas[1:])), deltas


# ---------------------------------------------------------------------------
# Invariant 3 — σ bounds; φ shrinks on evidence
# ---------------------------------------------------------------------------


def test_sigma_bounds_and_shrinkage():
    results, ratings = _field(8, sigma=200.0)
    cfg = G2PLConfig()
    updates = _g2pl_round(results, ratings, config=cfg)
    for u in updates:
        assert cfg.elo.sigma_floor <= u.sigma_after <= cfg.elo.sigma_ceiling
        # A full 8-athlete round is real evidence: φ must shrink.
        assert u.sigma_after < u.sigma_before


# ---------------------------------------------------------------------------
# Invariant 4 — beating a high-φ opponent moves you less
# ---------------------------------------------------------------------------


def test_high_phi_opponent_moves_winner_less():
    def winner_delta_vs(opponent_sigma: float) -> float:
        results = [
            AthleteResult(athlete_id=1, rank=1, score_normalized=30.0),
            AthleteResult(athlete_id=2, rank=2, score_normalized=20.0),
        ]
        ratings = {
            1: AthleteRating(
                athlete_id=1,
                mu=1500.0,
                sigma=100.0,
                n_events=10,
                provisional=False,
                last_event_at=ROUND_DATE,
            ),
            2: AthleteRating(
                athlete_id=2,
                mu=1500.0,
                sigma=opponent_sigma,
                n_events=10,
                provisional=False,
                last_event_at=ROUND_DATE,
            ),
        }
        return _winner_delta(_g2pl_round(results, ratings))

    assert winner_delta_vs(330.0) < winner_delta_vs(60.0)


# ---------------------------------------------------------------------------
# Invariant 5 — tie symmetry
# ---------------------------------------------------------------------------


def test_tie_symmetry_equal_phi():
    """Two equal-φ athletes who tie produce equal-and-opposite μ deltas
    (the favourite loses exactly what the underdog gains)."""
    results = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=25.0),
        AthleteResult(athlete_id=2, rank=1, score_normalized=25.0),
    ]
    ratings = {
        1: AthleteRating(
            athlete_id=1,
            mu=1600.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=ROUND_DATE,
        ),
        2: AthleteRating(
            athlete_id=2,
            mu=1400.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=ROUND_DATE,
        ),
    }
    updates = {u.athlete_id: u for u in _g2pl_round(results, ratings)}
    d1 = updates[1].mu_after - updates[1].mu_before
    d2 = updates[2].mu_after - updates[2].mu_before
    assert d1 < 0 < d2
    assert math.isclose(d1, -d2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Invariant 6 — determinism
# ---------------------------------------------------------------------------


def test_deterministic_updates():
    results, ratings_a = _field(8)
    _, ratings_b = _field(8)
    ua = _g2pl_round(results, ratings_a)
    ub = _g2pl_round(results, ratings_b)
    assert [(u.athlete_id, u.mu_after, u.sigma_after) for u in ua] == [
        (u.athlete_id, u.mu_after, u.sigma_after) for u in ub
    ]


# ---------------------------------------------------------------------------
# margin_adjusted_outcome algebra + mov_mode wiring
# ---------------------------------------------------------------------------


def test_margin_outcome_antisymmetry_and_tie_fixed_point():
    for mult in (0.0, 0.6, 1.0, 1.7, 3.0):
        s_win = margin_adjusted_outcome(1.0, mult, margin_cap=1.7)
        s_loss = margin_adjusted_outcome(0.0, mult, margin_cap=1.7)
        assert math.isclose(s_win + s_loss, 1.0)
        assert 0.0 <= s_loss <= 0.5 <= s_win <= 1.0
        assert margin_adjusted_outcome(0.5, mult, margin_cap=1.7) == 0.5


def test_margin_outcome_caps_at_full_win():
    assert margin_adjusted_outcome(1.0, 5.0, margin_cap=1.7) == 1.0
    assert margin_adjusted_outcome(1.0, 0.0, margin_cap=1.7) == 0.5


def test_mov_margin_mode_changes_updates():
    """mov_mode='margin' must actually alter deltas vs 'off' (the #84 A/B
    has two genuinely different arms)."""
    results, ratings_off = _field(8)
    _, ratings_margin = _field(8)
    u_off = _g2pl_round(results, ratings_off, config=G2PLConfig(mov_mode="off"))
    u_margin = _g2pl_round(
        results, ratings_margin, config=G2PLConfig(mov_mode="margin")
    )
    off_by_id = {u.athlete_id: u.mu_after for u in u_off}
    margin_by_id = {u.athlete_id: u.mu_after for u in u_margin}
    assert any(abs(off_by_id[i] - margin_by_id[i]) > 1e-6 for i in off_by_id)


# ---------------------------------------------------------------------------
# Pair attribution + drift monitor
# ---------------------------------------------------------------------------


def test_contributing_pairs_sum_to_delta():
    """Per-pair delta attribution must reconstruct the athlete's full Δμ
    (the breakdown-page contract)."""
    results, ratings = _field(8)
    updates = _g2pl_round(results, ratings)
    for u in updates:
        pair_sum = sum(p.delta for p in u.contributing_pairs)
        assert math.isclose(pair_sum, u.mu_after - u.mu_before, abs_tol=0.1)


def test_mu_drift_empty_and_signal():
    assert mu_drift([]) == 0.0
    results, ratings = _field(8)
    updates = _g2pl_round(results, ratings)
    assert math.isclose(
        mu_drift(updates),
        sum(u.mu_after - u.mu_before for u in updates) / len(updates),
    )
