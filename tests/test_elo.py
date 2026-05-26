"""Tests for the ELO rating engine.

Reproduces the PRD Appendix C worked example (adapted to K/(N-1) normalization).
"""
from datetime import date

from climbing_elo.engine.elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    BOULDER_MARGIN_MAX_GAP,
    AthleteRating,
    AthleteResult,
    apply_time_decay,
    calculate_round_updates,
    compute_boulder_margin_multiplier,
    compute_margin_multiplier,
    expected_score,
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


def test_time_decay_no_previous():
    result = apply_time_decay(100.0, None, date(2024, 1, 1))
    assert result == 100.0


def test_time_decay_increases_sigma():
    sigma_after = apply_time_decay(100.0, date(2022, 1, 1), date(2024, 1, 1))
    assert sigma_after > 100.0


def test_time_decay_capped():
    sigma_after = apply_time_decay(100.0, date(2010, 1, 1), date(2024, 1, 1))
    assert sigma_after == 350.0


def test_calculate_round_updates_zero_sum():
    """Rating changes must sum to zero across all athletes."""
    results = [
        AthleteResult(athlete_id=i, rank=i) for i in range(1, 9)
    ]
    ratings = {
        i: AthleteRating(athlete_id=i, mu=1500 + (5 - i) * 30, n_events=10, provisional=False)
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


def test_provisional_higher_k():
    """Provisional athletes should have larger rating swings."""
    results = [
        AthleteResult(athlete_id=1, rank=1),
        AthleteResult(athlete_id=2, rank=2),
    ]
    ratings_established = {
        1: AthleteRating(1, mu=1500, n_events=10, provisional=False),
        2: AthleteRating(2, mu=1500, n_events=10, provisional=False),
    }
    ratings_provisional = {
        1: AthleteRating(1, mu=1500, n_events=1, provisional=True),
        2: AthleteRating(2, mu=1500, n_events=10, provisional=False),
    }

    updates_est = calculate_round_updates(
        results, ratings_established, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )
    updates_prov = calculate_round_updates(
        results, ratings_provisional, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1)
    )

    delta_est = updates_est[0].mu_after - updates_est[0].mu_before
    delta_prov = updates_prov[0].mu_after - updates_prov[0].mu_before
    assert delta_prov > delta_est


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
    score = 4 * 1000 + 4 * 100 - 6 * 10 - 6  # 4T4z 6 6 → 3934.0
    assert compute_boulder_margin_multiplier(float(score), float(score)) == 1.0


def test_boulder_margin_one_top_gap():
    """A one-top gap (≈1000 points) should give ≈1.9× multiplier, capped at 2.0."""
    # Athlete A: 4T4z 4 4 → 4*1000 + 4*100 - 4*10 - 4 = 3956
    score_a = 4 * 1000 + 4 * 100 - 4 * 10 - 4
    # Athlete B: 3T4z 3 4 → 3*1000 + 4*100 - 3*10 - 4 = 2966
    score_b = 3 * 1000 + 4 * 100 - 3 * 10 - 4
    mult = compute_boulder_margin_multiplier(float(score_a), float(score_b))
    # Gap = 990, max_gap = 1000 → 1 + 990/1000 = 1.99
    assert mult > 1.5
    assert mult <= 2.0


def test_boulder_margin_capped_at_2():
    """Very large Boulder score gap is capped at MARGIN_CAP (2.0)."""
    # 5 tops vs 0 tops — huge gap
    mult = compute_boulder_margin_multiplier(5000.0, 0.0)
    assert mult == 2.0


def test_boulder_margin_uses_boulder_max_gap():
    """Boulder max_gap is much larger than Lead max_gap to match score scale."""
    assert BOULDER_MARGIN_MAX_GAP >= 500.0  # sanity: at least 500 vs Lead's 20


def test_boulder_round_updates_zero_sum():
    """Boulder rating changes must still sum to zero."""
    # Scores: tops * 1000 + zones * 100 - top_att * 10 - zone_att
    results = [
        AthleteResult(athlete_id=1, rank=1, score_normalized=4 * 1000 + 4 * 100 - 4 * 10 - 4),
        AthleteResult(athlete_id=2, rank=2, score_normalized=3 * 1000 + 4 * 100 - 5 * 10 - 4),
        AthleteResult(athlete_id=3, rank=3, score_normalized=2 * 1000 + 3 * 100 - 8 * 10 - 6),
        AthleteResult(athlete_id=4, rank=4, score_normalized=1 * 1000 + 2 * 100 - 3 * 10 - 2),
    ]
    ratings = {
        i: AthleteRating(athlete_id=i, mu=1500, n_events=10, provisional=False)
        for i in range(1, 5)
    }
    updates = calculate_round_updates(
        results, ratings, EventTier.WORLD_CUP, RoundType.FINAL, date(2024, 6, 1),
        discipline=Discipline.BOULDER,
    )
    total_delta = sum(u.mu_after - u.mu_before for u in updates)
    assert abs(total_delta) < 0.0001


def test_boulder_vs_lead_margin_scale():
    """Boulder margin multiplier should be smaller than Lead for same raw gap."""
    # A gap of 50 points
    # Lead: 1 + 50/20 = 3.5 → capped at 2.0
    lead_mult = compute_margin_multiplier(50.0, 0.0, max_gap=20.0)
    # Boulder: 1 + 50/1000 = 1.05
    boulder_mult = compute_boulder_margin_multiplier(50.0, 0.0)
    assert boulder_mult < lead_mult
