"""Tests for the ELO rating engine.

Reproduces the PRD Appendix C worked example (adapted to K/(N-1) normalization).
Post-#51 the engine uses Glicko-2 RD integration — see tests below for
g(φ) function values, inactivity inflation, high-φ cold-start trajectory.
"""

import math
from datetime import date

from climbing_elo.engine.elo import (
    BOULDER_MARGIN_MAX_GAP,
    DEFAULT_SIGMA,
    GLICKO2_SCALE,
    MARGIN_CAP,
    MOV_RATING_SCALE,
    MOV_SOFTENING,
    SIGMA_CEILING,
    SIGMA_FLOOR,
    AthleteRating,
    AthleteResult,
    _gap_conditioning_factor,
    _is_new_boulder_format,
    calculate_round_updates,
    compute_boulder_margin_multiplier,
    compute_margin_multiplier,
    compute_speed_margin_multiplier,
    expected_score,
    glicko2_expected_score,
    glicko2_g,
    glicko2_inflate_phi,
    normalize_boulder_score,
)
from climbing_elo.models import Discipline, EventTier, RoundType


def test_expected_score_equal_ratings():
    assert abs(expected_score(1500, 1500) - 0.5) < 0.001


def test_expected_score_higher_rated_favored():
    e = expected_score(1700, 1500)
    assert e > 0.5
    assert abs(e - 0.759) < 0.01


def test_expected_score_prd_example():
    """PRD Appendix C: E_B vs A = 0.429."""
    e_b = expected_score(1700, 1750)
    assert abs(e_b - 0.429) < 0.001


def test_expected_score_symmetric():
    e_a = expected_score(1700, 1750)
    e_b = expected_score(1750, 1700)
    assert abs(e_a + e_b - 1.0) < 0.0001


def test_margin_multiplier_no_scores():
    assert compute_margin_multiplier(None, None) == 1.0


def test_margin_multiplier_capped():
    # MARGIN_CAP is tuned to 1.5 (from empirical grid search in tune_kfactors.py).
    # Any score gap large enough to exceed the cap should return MARGIN_CAP.
    from climbing_elo.engine import elo as elo_module

    mult = compute_margin_multiplier(40.0, 0.0, max_gap=20.0)
    assert mult == elo_module.MARGIN_CAP


def test_margin_multiplier_partial():
    mult = compute_margin_multiplier(30.0, 20.0, max_gap=20.0)
    assert abs(mult - 1.5) < 0.01


def test_inactivity_inflation_no_previous_event():
    """Athletes with no recorded last event keep their σ unchanged."""
    assert glicko2_inflate_phi(100.0, None, date(2024, 1, 1)) == 100.0


def test_inactivity_inflation_grace_period():
    """Activity gaps inside the 30-day grace window do not inflate φ."""
    # 15 days inactive — inside grace.
    sigma_after = glicko2_inflate_phi(100.0, date(2024, 1, 1), date(2024, 1, 16))
    assert sigma_after == 100.0


def test_inactivity_inflation_increases_sigma():
    """6 months inactivity inflates σ via the Wiener-process formula."""
    sigma_after = glicko2_inflate_phi(100.0, date(2024, 1, 1), date(2024, 7, 1))
    assert sigma_after > 100.0
    assert sigma_after < SIGMA_CEILING


def test_inactivity_inflation_capped_at_ceiling():
    """A pre-inflated φ near the ceiling stays clamped at the ceiling."""
    # Start at the ceiling — any positive inflation must not exceed it.
    sigma_after = glicko2_inflate_phi(
        SIGMA_CEILING - 5.0, date(2000, 1, 1), date(2024, 1, 1)
    )
    assert sigma_after == SIGMA_CEILING


def test_inactivity_inflation_monotonic_with_gap_length():
    """Longer gaps must inflate σ at least as much as shorter ones."""
    s_short = glicko2_inflate_phi(80.0, date(2024, 1, 1), date(2024, 7, 1))
    s_long = glicko2_inflate_phi(80.0, date(2024, 1, 1), date(2025, 1, 1))
    assert s_long >= s_short


# ---------------------------------------------------------------------------
# Glicko-2 primitive tests
# ---------------------------------------------------------------------------


def test_glicko2_g_at_zero_phi():
    """g(0) = 1.0 — full confidence in the opponent's rating."""
    assert math.isclose(glicko2_g(0.0), 1.0, rel_tol=1e-9)


def test_glicko2_g_decreases_with_phi():
    """g is monotonically decreasing — higher φ → lower weight."""
    assert glicko2_g(0.0) > glicko2_g(1.0) > glicko2_g(2.0) > glicko2_g(5.0)


def test_glicko2_g_approaches_zero_for_large_phi():
    """g(φ) → 0 as φ grows; an opponent with infinite RD contributes nothing."""
    assert glicko2_g(100.0) < 0.05


def test_glicko2_g_default_phi_value():
    """Reference: φ = 350/173.7178 ≈ 2.015 → g(φ) ≈ 0.669.

    Closed form: ``g(2.015) = 1/sqrt(1 + 3·2.015²/π²) ≈ 0.669``.
    """
    phi = DEFAULT_SIGMA / GLICKO2_SCALE
    g_val = glicko2_g(phi)
    assert 0.66 < g_val < 0.68


def test_glicko2_expected_score_equal_ratings_high_confidence():
    """Equal μ, low φ → expected score 0.5."""
    e = glicko2_expected_score(1500.0, 1500.0, phi_b=50.0)
    assert math.isclose(e, 0.5, rel_tol=1e-9)


def test_glicko2_expected_score_higher_rated_favoured():
    """Higher-μ athlete is favoured."""
    e = glicko2_expected_score(1700.0, 1500.0, phi_b=100.0)
    assert e > 0.5


def test_glicko2_expected_score_high_phi_opponent_dampens_edge():
    """Against a very high-φ opponent, the favourite's edge shrinks toward 0.5."""
    e_low_phi = glicko2_expected_score(1700.0, 1500.0, phi_b=50.0)
    e_high_phi = glicko2_expected_score(1700.0, 1500.0, phi_b=350.0)
    assert e_low_phi > e_high_phi > 0.5


def test_calculate_round_updates_zero_sum():
    """Rating changes must sum to zero across all athletes."""
    results = [AthleteResult(athlete_id=i, rank=i) for i in range(1, 9)]
    ratings = {
        i: AthleteRating(
            athlete_id=i, mu=1500 + (5 - i) * 30, n_events=10, provisional=False
        )
        for i in range(1, 9)
    }
    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_calculate_round_updates_prd_example():
    """PRD Appendix C adapted: B upsets A in 8-athlete final."""
    # Result order: B, A, D, C, E, F, H, G
    results = [
        AthleteResult(athlete_id=2, rank=1),  # B
        AthleteResult(athlete_id=1, rank=2),  # A
        AthleteResult(athlete_id=4, rank=3),  # D
        AthleteResult(athlete_id=3, rank=4),  # C
        AthleteResult(athlete_id=5, rank=5),  # E
        AthleteResult(athlete_id=6, rank=6),  # F
        AthleteResult(athlete_id=8, rank=7),  # H
        AthleteResult(athlete_id=7, rank=8),  # G
    ]
    ratings = {
        1: AthleteRating(1, mu=1750, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1700, n_events=10, provisional=False),
        3: AthleteRating(3, mu=1680, n_events=10, provisional=False),
        4: AthleteRating(4, mu=1650, n_events=10, provisional=False),
        5: AthleteRating(5, mu=1620, n_events=10, provisional=False),
        6: AthleteRating(6, mu=1600, n_events=10, provisional=False),
        7: AthleteRating(7, mu=1570, n_events=10, provisional=False),
        8: AthleteRating(8, mu=1540, n_events=10, provisional=False),
    }

    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )

    by_id = {u.athlete_id: u for u in updates}

    # B (upset winner) should have the largest positive delta
    deltas = {u.athlete_id: u.mu_after - u.mu_before for u in updates}
    assert deltas[2] == max(deltas.values())

    # G (collapsed from expected-5th to last) should have the largest negative
    assert deltas[7] == min(deltas.values())

    # A (2nd place, expected 1st) should still gain slightly
    assert deltas[1] > 0

    # B's upset should narrow the gap with A (was 50, should shrink)
    gap_before = 1750 - 1700
    gap_after = by_id[1].mu_after - by_id[2].mu_after
    assert gap_after < gap_before

    # Zero-sum
    total = sum(deltas.values())
    assert abs(total) < 0.0001

    # Contributing pairs recorded
    assert len(by_id[2].contributing_pairs) == 7  # winner beat 7 opponents
    assert all(p.result == "won" for p in by_id[2].contributing_pairs)
    assert len(by_id[7].contributing_pairs) == 7  # loser lost to 7


def test_dns_excluded():
    """Athletes marked DNS should not affect or be affected by results."""
    results = [
        AthleteResult(athlete_id=1, rank=1),
        AthleteResult(athlete_id=2, rank=2),
        AthleteResult(athlete_id=3, rank=0, dns=True),
    ]
    ratings = {
        1: AthleteRating(1, mu=1500, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1500, n_events=10, provisional=False),
        3: AthleteRating(3, mu=1500, n_events=10, provisional=False),
    }
    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )
    ids_updated = {u.athlete_id for u in updates}
    assert 3 not in ids_updated
    assert len(updates) == 2


def test_ties_produce_no_delta():
    """Tied athletes should have zero pairwise delta between them."""
    results = [
        AthleteResult(athlete_id=1, rank=1),
        AthleteResult(athlete_id=2, rank=2),
        AthleteResult(athlete_id=3, rank=2),  # tied with 2
    ]
    ratings = {
        1: AthleteRating(1, mu=1500, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1500, n_events=10, provisional=False),
        3: AthleteRating(3, mu=1500, n_events=10, provisional=False),
    }
    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )
    by_id = {u.athlete_id: u for u in updates}
    delta_2 = by_id[2].mu_after - by_id[2].mu_before
    delta_3 = by_id[3].mu_after - by_id[3].mu_before
    assert abs(delta_2 - delta_3) < 0.0001


def test_high_phi_athlete_phi_shrinks_more():
    """High-φ (cold-start) athletes have their φ shrink more in absolute terms
    after a round than already-confident athletes — that's the whole point of
    Glicko-2 RD integration. Replaces the legacy ``test_provisional_higher_k``.
    """
    results = [
        AthleteResult(athlete_id=1, rank=1),
        AthleteResult(athlete_id=2, rank=2),
    ]
    # Established pair (low σ, well-known ratings).
    ratings_established = {
        1: AthleteRating(1, mu=1500, sigma=80.0, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1500, sigma=80.0, n_events=10, provisional=False),
    }
    # Cold-start pair (high σ — fresh in the rating system).
    ratings_cold = {
        1: AthleteRating(1, mu=1500, sigma=DEFAULT_SIGMA, n_events=1, provisional=True),
        2: AthleteRating(2, mu=1500, sigma=DEFAULT_SIGMA, n_events=1, provisional=True),
    }

    updates_est = calculate_round_updates(
        results,
        ratings_established,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
    )
    updates_cold = calculate_round_updates(
        results,
        ratings_cold,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
    )

    # Cold-start athlete's φ should shrink notably more than established's.
    phi_shrink_est = updates_est[0].sigma_before - updates_est[0].sigma_after
    phi_shrink_cold = updates_cold[0].sigma_before - updates_cold[0].sigma_after
    assert phi_shrink_cold > phi_shrink_est
    # Both shrinks should be positive (round consumes uncertainty).
    assert phi_shrink_est >= 0
    assert phi_shrink_cold > 0
    # Floor should be respected.
    assert updates_est[0].sigma_after >= SIGMA_FLOOR
    assert updates_cold[0].sigma_after >= SIGMA_FLOOR


def test_single_athlete_no_updates():
    results = [AthleteResult(athlete_id=1, rank=1)]
    ratings = {1: AthleteRating(1, mu=1500)}
    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )
    assert len(updates) == 0


# ---------------------------------------------------------------------------
# Boulder margin weighting tests
# ---------------------------------------------------------------------------


def test_boulder_margin_no_scores():
    """Boulder multiplier returns 1.0 when scores are absent."""
    assert compute_boulder_margin_multiplier(None, None) == 1.0
    assert compute_boulder_margin_multiplier(None, 4000.0) == 1.0
    assert compute_boulder_margin_multiplier(3000.0, None) == 1.0


def test_boulder_margin_same_score():
    """Identical scores produce a 1.0 multiplier."""
    score = 4 * 1000 + 4 * 100 - 6 * 10 - 6
    assert compute_boulder_margin_multiplier(float(score), float(score)) == 1.0


def test_boulder_margin_one_top_gap():
    """A one-top gap (≈1000 points) should saturate near the MARGIN_CAP."""
    score_a = 4 * 1000 + 4 * 100 - 4 * 10 - 4
    score_b = 3 * 1000 + 4 * 100 - 3 * 10 - 4
    mult = compute_boulder_margin_multiplier(float(score_a), float(score_b))
    assert mult == min(1.99, MARGIN_CAP)


def test_boulder_margin_capped_at_max():
    """Very large Boulder score gap is capped at MARGIN_CAP."""
    mult = compute_boulder_margin_multiplier(5000.0, 0.0)
    assert mult == MARGIN_CAP


def test_boulder_margin_uses_boulder_max_gap():
    """Boulder max_gap is much larger than Lead max_gap to match score scale."""
    assert BOULDER_MARGIN_MAX_GAP >= 500.0


def test_boulder_round_updates_zero_sum():
    """Boulder rating changes must still sum to zero."""
    results = [
        AthleteResult(
            athlete_id=1, rank=1, score_normalized=4 * 1000 + 4 * 100 - 4 * 10 - 4
        ),
        AthleteResult(
            athlete_id=2, rank=2, score_normalized=3 * 1000 + 4 * 100 - 5 * 10 - 4
        ),
        AthleteResult(
            athlete_id=3, rank=3, score_normalized=2 * 1000 + 3 * 100 - 8 * 10 - 6
        ),
        AthleteResult(
            athlete_id=4, rank=4, score_normalized=1 * 1000 + 2 * 100 - 3 * 10 - 2
        ),
    ]
    ratings = {
        i: AthleteRating(athlete_id=i, mu=1500, n_events=10, provisional=False)
        for i in range(1, 5)
    }
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.BOULDER,
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_boulder_vs_lead_margin_scale():
    """Boulder margin multiplier should be smaller than Lead for same raw gap."""
    lead_mult = compute_margin_multiplier(50.0, 0.0, max_gap=20.0)
    boulder_mult = compute_boulder_margin_multiplier(50.0, 0.0)
    assert boulder_mult < lead_mult


# ---------------------------------------------------------------------------
# Boulder format detection and normalization tests
# ---------------------------------------------------------------------------


def test_is_new_boulder_format_decimal():
    """Decimal strings are recognised as the new 2025+ format."""
    assert _is_new_boulder_format("34.5") is True
    assert _is_new_boulder_format("25.0") is True
    assert _is_new_boulder_format("10.1") is True
    assert _is_new_boulder_format("0.0") is True


def test_is_new_boulder_format_integer_string():
    """Plain integer strings are also new-format (parseable as float)."""
    assert _is_new_boulder_format("100") is True


def test_is_new_boulder_format_old_format():
    """Old ordinal strings are NOT the new format."""
    assert _is_new_boulder_format("1T2z 3 4") is False
    assert _is_new_boulder_format("2T2z 2 2") is False
    assert _is_new_boulder_format("0T1z 0 5") is False


def test_normalize_boulder_score_new_format():
    """New-format decimal scores pass through as floats."""
    assert normalize_boulder_score("34.5") == 34.5
    assert normalize_boulder_score("25.0") == 25.0
    assert normalize_boulder_score("10.1") == 10.1


def test_normalize_boulder_score_old_format_ntz():
    """Old 'NTMz A B' format is parsed into tops*1000+zones*100-top_att*10-zone_att."""
    # "1T2z 3 4" → 1*1000 + 2*100 - 3*10 - 4 = 1166
    assert normalize_boulder_score("1T2z 3 4") == 1166.0
    # "2T2z 2 2" → 2*1000 + 2*100 - 2*10 - 2 = 2178
    assert normalize_boulder_score("2T2z 2 2") == 2178.0
    # "0T1z 0 5" → 0*1000 + 1*100 - 0*10 - 5 = 95
    assert normalize_boulder_score("0T1z 0 5") == 95.0


def test_normalize_boulder_score_old_format_tb():
    """Old 'NT A MBB' alternative format is also parsed correctly."""
    # "2T3 4B5" → tops=2, top_att=3, zones=4, zone_att=5
    # → 2*1000 + 4*100 - 3*10 - 5 = 2365
    assert normalize_boulder_score("2T3 4B5") == 2365.0


def test_normalize_boulder_score_dnf_dns():
    """DNF / DNS / empty strings return None."""
    assert normalize_boulder_score("DNF") is None
    assert normalize_boulder_score("DNS") is None
    assert normalize_boulder_score("") is None
    assert normalize_boulder_score("-") is None


def test_normalize_boulder_score_unparseable():
    """Unparseable garbage returns None."""
    assert normalize_boulder_score("GARBAGE") is None


def test_boulder_old_format_zero_sum():
    """Zero-sum invariant holds when score_normalized comes from old-format Boulder raw scores."""
    # Parse old-format raw scores into normalized floats
    raw_scores = ["4T4z 4 4", "3T4z 5 4", "2T3z 8 6", "1T2z 3 2"]
    norm_scores = [normalize_boulder_score(s) for s in raw_scores]

    results = [
        AthleteResult(athlete_id=i + 1, rank=i + 1, score_normalized=norm_scores[i])
        for i in range(4)
    ]
    ratings = {
        i: AthleteRating(athlete_id=i, mu=1500, n_events=10, provisional=False)
        for i in range(1, 5)
    }
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.BOULDER,
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_boulder_new_format_zero_sum():
    """Zero-sum invariant holds when score_normalized comes from new-format (decimal) Boulder scores."""
    raw_scores = ["34.5", "25.0", "10.1", "5.0"]
    norm_scores = [normalize_boulder_score(s) for s in raw_scores]

    results = [
        AthleteResult(athlete_id=i + 1, rank=i + 1, score_normalized=norm_scores[i])
        for i in range(4)
    ]
    ratings = {
        i: AthleteRating(athlete_id=i, mu=1500, n_events=10, provisional=False)
        for i in range(1, 5)
    }
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.BOULDER,
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


# Speed discipline tests
# ---------------------------------------------------------------------------


def test_speed_margin_multiplier_no_scores():
    """No times → neutral margin."""
    assert compute_speed_margin_multiplier(None, None) == 1.0


def test_speed_margin_multiplier_capped():
    """Gap >= max_gap should produce MARGIN_CAP."""
    mult = compute_speed_margin_multiplier(6.0, 8.1)
    assert mult == MARGIN_CAP


def test_speed_margin_multiplier_half_gap():
    """1.0 s gap with max 2.0 s → 1.5x multiplier, may be capped by MARGIN_CAP."""
    mult = compute_speed_margin_multiplier(6.5, 7.5)
    assert abs(mult - min(1.5, MARGIN_CAP)) < 0.01


def test_speed_margin_multiplier_argument_order_invariant():
    """Multiplier should be the same regardless of argument order."""
    m1 = compute_speed_margin_multiplier(6.5, 7.5)
    m2 = compute_speed_margin_multiplier(7.5, 6.5)
    assert abs(m1 - m2) < 1e-9


def test_speed_round_updates_zero_sum():
    """ELO is zero-sum for Speed qualification (ranked by time)."""
    results = [
        AthleteResult(athlete_id=i, rank=i, score_normalized=6.0 + i * 0.1)
        for i in range(1, 9)
    ]
    ratings = {
        i: AthleteRating(athlete_id=i, mu=1500.0, n_events=10, provisional=False)
        for i in range(1, 9)
    }
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.QUALIFICATION,
        date(2024, 6, 1),
        discipline=Discipline.SPEED,
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_speed_false_start_treated_as_dnf():
    """An athlete with DNF=True (false start) should rank at bottom with no margin bonus."""
    results = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=6.5),
        AthleteResult(athlete_id=2, rank=2, score_normalized=None, dnf=True),
    ]
    ratings = {
        1: AthleteRating(athlete_id=1, mu=1500.0, n_events=10, provisional=False),
        2: AthleteRating(athlete_id=2, mu=1500.0, n_events=10, provisional=False),
    }
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.SPEED,
    )
    by_id = {u.athlete_id: u for u in updates}
    # Margin multiplier for the pair should be 1.0 (DNF path)
    pair_for_winner = by_id[1].contributing_pairs[0]
    assert pair_for_winner.margin_multiplier == 1.0
    # Zero-sum still holds
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_speed_winner_gains_more_with_large_margin():
    """Larger time gap should produce a bigger rating swing than a small gap."""

    def _run(winner_time, loser_time):
        results = [
            AthleteResult(athlete_id=1, rank=1, score_normalized=winner_time),
            AthleteResult(athlete_id=2, rank=2, score_normalized=loser_time),
        ]
        ratings = {
            1: AthleteRating(1, mu=1500.0, n_events=10, provisional=False),
            2: AthleteRating(2, mu=1500.0, n_events=10, provisional=False),
        }
        upd = calculate_round_updates(
            results,
            ratings,
            EventTier.WORLD_CUP,
            RoundType.FINAL,
            date(2024, 6, 1),
            discipline=Discipline.SPEED,
        )
        return next(u for u in upd if u.athlete_id == 1).mu_after - 1500.0

    delta_small = _run(6.5, 6.6)  # 0.1 s gap
    delta_large = _run(6.5, 8.5)  # 2.0 s gap (capped)
    assert delta_large > delta_small


def test_speed_dns_excluded():
    """Athletes marked DNS should be excluded from Speed rounds too."""
    results = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=6.5),
        AthleteResult(athlete_id=2, rank=2, score_normalized=7.0),
        AthleteResult(athlete_id=3, rank=0, dns=True),
    ]
    ratings = {
        i: AthleteRating(i, mu=1500.0, n_events=10, provisional=False)
        for i in range(1, 4)
    }
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.QUALIFICATION,
        date(2024, 6, 1),
        discipline=Discipline.SPEED,
    )
    ids_updated = {u.athlete_id for u in updates}
    assert 3 not in ids_updated
    assert len(updates) == 2


# ---------------------------------------------------------------------------
# Gap-conditioned MOV (Issue #53)
# ---------------------------------------------------------------------------


def test_gap_conditioning_factor_zero_gap_is_one():
    """At Δμ = 0 (peer matchup), no damping — factor must equal 1.0."""
    assert _gap_conditioning_factor(0.0) == 1.0


def test_gap_conditioning_factor_negative_gap_is_one():
    """Upsets (Δμ < 0, underdog wins) keep full MOV bonus — factor = 1.0."""
    assert _gap_conditioning_factor(-100.0) == 1.0
    assert _gap_conditioning_factor(-500.0) == 1.0
    assert _gap_conditioning_factor(-1000.0) == 1.0


def test_gap_conditioning_factor_positive_gap_damps():
    """Favourite wins (Δμ > 0) are damped — factor < 1.0 and monotonic."""
    f_100 = _gap_conditioning_factor(100.0)
    f_400 = _gap_conditioning_factor(400.0)
    f_800 = _gap_conditioning_factor(800.0)
    assert 0.0 < f_800 < f_400 < f_100 < 1.0


def test_gap_conditioning_factor_at_scale_matches_538_formula():
    """Direct check: at Δμ = MOV_RATING_SCALE, factor = SOFTENING / (1 + SOFTENING)."""
    f = _gap_conditioning_factor(MOV_RATING_SCALE)
    expected = MOV_SOFTENING / (1.0 + MOV_SOFTENING)
    assert math.isclose(f, expected, rel_tol=1e-9)


def test_margin_multiplier_default_rating_gap_back_compat():
    """With rating_gap=0 (the default), gap-conditioned formula = legacy formula."""
    mult = compute_margin_multiplier(30.0, 20.0, max_gap=20.0)
    # Legacy: min(1 + 10/20, 1.5) = 1.5
    assert abs(mult - 1.5) < 0.01


def test_margin_multiplier_peer_matchup_full_bonus():
    """Δμ = 0 leaves the multiplier unchanged from the legacy formula."""
    legacy = min(1.0 + 10.0 / 20.0, MARGIN_CAP)  # 1.5
    new = compute_margin_multiplier(30.0, 20.0, max_gap=20.0, rating_gap=0.0)
    assert math.isclose(new, legacy, rel_tol=1e-9)


def test_margin_multiplier_favourite_wins_damped():
    """Favourite wins → damped multiplier (smaller than the peer-matchup case)."""
    peer = compute_margin_multiplier(30.0, 20.0, max_gap=20.0, rating_gap=0.0)
    favourite = compute_margin_multiplier(30.0, 20.0, max_gap=20.0, rating_gap=400.0)
    assert favourite < peer
    # Same gap, scaling factor = 2.2 / (1 + 2.2) = 0.6875
    expected = peer * (MOV_SOFTENING / (1.0 + MOV_SOFTENING))
    assert math.isclose(favourite, expected, rel_tol=1e-9)


def test_margin_multiplier_upset_keeps_full_bonus():
    """Upset (Δμ < 0, underdog wins) is asymmetric — no damping."""
    peer = compute_margin_multiplier(30.0, 20.0, max_gap=20.0, rating_gap=0.0)
    upset = compute_margin_multiplier(30.0, 20.0, max_gap=20.0, rating_gap=-400.0)
    assert math.isclose(upset, peer, rel_tol=1e-9)


def test_margin_multiplier_cap_still_respected_after_gap_conditioning():
    """MARGIN_CAP is on the base; gap conditioning only shrinks further."""
    # Huge score gap and peer matchup → exactly MARGIN_CAP.
    cap_peer = compute_margin_multiplier(100.0, 0.0, max_gap=20.0, rating_gap=0.0)
    assert cap_peer == MARGIN_CAP
    # Huge score gap and large favourite-side Δμ → strictly below MARGIN_CAP.
    cap_fav = compute_margin_multiplier(100.0, 0.0, max_gap=20.0, rating_gap=600.0)
    assert cap_fav < MARGIN_CAP
    # And it is the base * damping (= MARGIN_CAP * factor).
    expected = MARGIN_CAP * _gap_conditioning_factor(600.0)
    assert math.isclose(cap_fav, expected, rel_tol=1e-9)


def test_boulder_margin_gap_conditioned():
    """Boulder margin participates in gap conditioning too."""
    peer = compute_boulder_margin_multiplier(4000.0, 3000.0, rating_gap=0.0)
    favourite = compute_boulder_margin_multiplier(4000.0, 3000.0, rating_gap=500.0)
    upset = compute_boulder_margin_multiplier(4000.0, 3000.0, rating_gap=-500.0)
    assert favourite < peer
    assert math.isclose(upset, peer, rel_tol=1e-9)


def test_speed_margin_gap_conditioned():
    """Speed margin participates in gap conditioning too."""
    peer = compute_speed_margin_multiplier(6.0, 7.0, rating_gap=0.0)
    favourite = compute_speed_margin_multiplier(6.0, 7.0, rating_gap=500.0)
    upset = compute_speed_margin_multiplier(6.0, 7.0, rating_gap=-500.0)
    assert favourite < peer
    assert math.isclose(upset, peer, rel_tol=1e-9)


def test_zero_sum_invariant_holds_with_gap_conditioning():
    """The new MOV must not break the zero-sum invariant — heterogeneous μ field."""
    # Mixed favourites + underdogs so gap-conditioning fires in both directions.
    results = [AthleteResult(athlete_id=i, rank=i) for i in range(1, 9)]
    ratings = {
        1: AthleteRating(1, mu=1800, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1750, n_events=10, provisional=False),
        3: AthleteRating(3, mu=1700, n_events=10, provisional=False),
        4: AthleteRating(4, mu=1650, n_events=10, provisional=False),
        5: AthleteRating(5, mu=1600, n_events=10, provisional=False),
        6: AthleteRating(6, mu=1550, n_events=10, provisional=False),
        7: AthleteRating(7, mu=1500, n_events=10, provisional=False),
        8: AthleteRating(8, mu=1450, n_events=10, provisional=False),
    }
    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_calculate_round_updates_elite_vs_junior_smaller_swing_than_peer_vs_peer():
    """Headline test for Issue #53: elite crushing a junior by margin X should move
    less than a peer crushing a peer by the same margin X.

    Set up two 2-athlete rounds with identical *score* gap but different rating
    gaps. The favourite's μ swing should be smaller in the mismatched case.
    """
    score_winner = 30.0
    score_loser = 20.0

    # Peer matchup (Δμ = 0).
    results_peer = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=score_winner),
        AthleteResult(athlete_id=2, rank=2, score_normalized=score_loser),
    ]
    ratings_peer = {
        1: AthleteRating(1, mu=1500.0, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1500.0, n_events=10, provisional=False),
    }
    upd_peer = calculate_round_updates(
        results_peer,
        ratings_peer,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
    )
    swing_peer = next(u for u in upd_peer if u.athlete_id == 1).mu_after - 1500.0

    # Elite vs junior (Δμ = 600).
    results_elite = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=score_winner),
        AthleteResult(athlete_id=2, rank=2, score_normalized=score_loser),
    ]
    ratings_elite = {
        1: AthleteRating(1, mu=2100.0, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1500.0, n_events=10, provisional=False),
    }
    upd_elite = calculate_round_updates(
        results_elite,
        ratings_elite,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
    )
    swing_elite = next(u for u in upd_elite if u.athlete_id == 1).mu_after - 2100.0

    # Note: the elite is *already favoured* and so the (1 − E) term is small,
    # which also shrinks the swing. The gap-conditioned MOV stacks on top of
    # that. Either way, elite-side swing must be smaller than peer-side.
    assert swing_elite < swing_peer


# ---------------------------------------------------------------------------
# Tournament Participation Bonus (Issue #90 — Gap 1 from #88)
# ---------------------------------------------------------------------------


def _make_finished_field(n: int) -> list[AthleteResult]:
    """Helper: n athletes finishing in rank order 1..n."""
    return [
        AthleteResult(athlete_id=100 + i, rank=i + 1, score_normalized=float(100 - i))
        for i in range(n)
    ]


def test_tpb_zero_sum_invariant():
    """Sum of TPB deltas across the field must be zero (to FP tolerance)."""
    from climbing_elo.engine.elo import compute_tournament_participation_bonus

    for tier in (
        EventTier.OLYMPICS,
        EventTier.WORLD_CHAMPIONSHIP,
        EventTier.WORLD_CUP,
        EventTier.CONTINENTAL,
    ):
        contribs = compute_tournament_participation_bonus(
            _make_finished_field(20), tier
        )
        assert len(contribs) == 20
        total = sum(c.delta for c in contribs)
        assert abs(total) < 1e-9, f"{tier}: sum={total}"


def test_tpb_winner_gets_positive_delta():
    """Rank-1 finisher gets the largest gross bonus and a positive net delta."""
    from climbing_elo.engine.elo import compute_tournament_participation_bonus

    contribs = compute_tournament_participation_bonus(
        _make_finished_field(20), EventTier.OLYMPICS
    )
    winner = next(c for c in contribs if c.rank == 1)
    last = next(c for c in contribs if c.rank == 20)
    assert winner.gross_bonus == 30.0
    assert winner.delta > 0
    assert last.gross_bonus == 0.0
    assert last.delta < 0  # debited but received no gross


def test_tpb_tier_monotonic_at_rank_1():
    """Rank-1 net delta strictly decreases across tiers: Oly > WCh > WC > Cont."""
    from climbing_elo.engine.elo import compute_tournament_participation_bonus

    field = _make_finished_field(20)
    tiers = [
        EventTier.OLYMPICS,
        EventTier.WORLD_CHAMPIONSHIP,
        EventTier.WORLD_CUP,
        EventTier.CONTINENTAL,
    ]
    rank1_deltas = [
        next(
            c for c in compute_tournament_participation_bonus(field, t) if c.rank == 1
        ).delta
        for t in tiers
    ]
    assert rank1_deltas == sorted(rank1_deltas, reverse=True)
    assert rank1_deltas[0] > rank1_deltas[-1]


def test_tpb_handles_field_smaller_than_top_k():
    """A 5-athlete final (smaller than Olympics top-8) still zero-sums."""
    from climbing_elo.engine.elo import compute_tournament_participation_bonus

    contribs = compute_tournament_participation_bonus(
        _make_finished_field(5), EventTier.OLYMPICS
    )
    assert len(contribs) == 5
    assert abs(sum(c.delta for c in contribs)) < 1e-9
    # Rank-5 in Olympics table receives 7.5 gross — still > debit (avg of
    # 30+22.5+15+11.25+7.5 = 86.25 / 5 = 17.25), so rank-5 has negative delta.
    rank5 = next(c for c in contribs if c.rank == 5)
    assert rank5.gross_bonus == 7.5
    assert rank5.delta < 0


def test_tpb_excludes_dns():
    """DNS athletes don't appear in TPB output, and the rest still zero-sum."""
    from climbing_elo.engine.elo import compute_tournament_participation_bonus

    results = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=100.0),
        AthleteResult(athlete_id=2, rank=2, score_normalized=80.0),
        AthleteResult(athlete_id=3, rank=999, score_normalized=None, dns=True),
    ]
    contribs = compute_tournament_participation_bonus(results, EventTier.WORLD_CUP)
    ids = {c.athlete_id for c in contribs}
    assert 3 not in ids
    assert len(contribs) == 2
    assert abs(sum(c.delta for c in contribs)) < 1e-9


def test_tpb_single_athlete_returns_empty():
    """A 1-athlete event has no field — TPB returns []."""
    from climbing_elo.engine.elo import compute_tournament_participation_bonus

    results = [AthleteResult(athlete_id=1, rank=1, score_normalized=100.0)]
    contribs = compute_tournament_participation_bonus(results, EventTier.OLYMPICS)
    assert contribs == []


def test_tpb_does_not_disturb_pair_updates():
    """Existing pair-update test must still pass — TPB is layered on separately."""
    # Sanity: calling calculate_round_updates does NOT trigger TPB. TPB only
    # fires in backfill, after all rounds are processed. This guards against
    # an accidental fold of TPB into the round update.
    results = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=100.0),
        AthleteResult(athlete_id=2, rank=2, score_normalized=80.0),
    ]
    ratings = {
        1: AthleteRating(1, mu=1500.0, n_events=10),
        2: AthleteRating(2, mu=1500.0, n_events=10),
    }
    updates = calculate_round_updates(
        results, ratings, EventTier.OLYMPICS, RoundType.FINAL, date(2024, 6, 1)
    )
    # μ deltas must remain zero-sum on the pair update alone.
    total = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total) < 1e-9
