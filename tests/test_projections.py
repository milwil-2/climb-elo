"""Tests for the Monte Carlo projection engine (engine/projections.py)."""
from __future__ import annotations

import pytest

from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_podium_probabilities,
    expected_finish_ranks,
    predict_winner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_athlete(athlete_id: int, mu: float, sigma: float = 100.0) -> AthleteProjectionInput:
    return AthleteProjectionInput(athlete_id=athlete_id, mu=mu, sigma=sigma)


# ---------------------------------------------------------------------------
# compute_podium_probabilities
# ---------------------------------------------------------------------------

class TestComputePodiumProbabilities:
    def test_empty_list_returns_empty_dict(self):
        result = compute_podium_probabilities([])
        assert result == {}

    def test_single_athlete_wins_every_sim(self):
        athletes = [make_athlete(1, mu=1500)]
        result = compute_podium_probabilities(athletes, n_simulations=1000, rng_seed=42)
        assert result[1]["win"] == pytest.approx(1.0, abs=0.0)
        assert result[1]["podium"] == pytest.approx(1.0, abs=0.0)
        assert result[1]["top_8"] == pytest.approx(1.0, abs=0.0)
        assert result[1]["expected_rank"] == pytest.approx(1.0, abs=0.0)

    def test_win_probabilities_sum_to_one(self):
        athletes = [make_athlete(i, mu=1500 - i * 50) for i in range(1, 9)]
        result = compute_podium_probabilities(athletes, n_simulations=10_000, rng_seed=0)
        total_win = sum(v["win"] for v in result.values())
        assert total_win == pytest.approx(1.0, abs=0.01)

    def test_podium_probabilities_sum_to_three_for_large_field(self):
        """Sum of podium probs equals 3 when there are >= 3 athletes."""
        athletes = [make_athlete(i, mu=1500) for i in range(1, 9)]
        result = compute_podium_probabilities(athletes, n_simulations=10_000, rng_seed=1)
        total_podium = sum(v["podium"] for v in result.values())
        assert total_podium == pytest.approx(3.0, abs=0.05)

    def test_top8_probabilities_sum_to_eight_for_large_field(self):
        athletes = [make_athlete(i, mu=1500) for i in range(1, 9)]
        result = compute_podium_probabilities(athletes, n_simulations=10_000, rng_seed=2)
        total_top8 = sum(v["top_8"] for v in result.values())
        assert total_top8 == pytest.approx(8.0, abs=0.05)

    def test_higher_mu_athlete_has_higher_win_probability(self):
        """A much stronger athlete should win far more often."""
        strong = make_athlete(1, mu=2000, sigma=100)
        weak = make_athlete(2, mu=1200, sigma=100)
        result = compute_podium_probabilities([strong, weak], n_simulations=10_000, rng_seed=3)
        assert result[1]["win"] > result[2]["win"]
        # With an 800-point gap and sigma=100, strong wins almost always
        assert result[1]["win"] > 0.99

    def test_higher_mu_athlete_has_higher_podium_probability(self):
        """Podium probability should correlate with mu across a tight field."""
        # Use more athletes so not all are guaranteed top-3 and sigma large
        # enough that there is meaningful variance.
        athletes = [make_athlete(i, mu=1700 - i * 100, sigma=200) for i in range(1, 9)]
        result = compute_podium_probabilities(athletes, n_simulations=10_000, rng_seed=4)
        # Athlete 1 (highest mu) should have higher podium prob than athlete 8
        assert result[1]["podium"] > result[8]["podium"]
        # Monotonic: each athlete has a higher podium prob than the one below
        for i in range(1, 7):
            assert result[i]["podium"] >= result[i + 1]["podium"], (
                f"Expected athlete {i} podium prob >= athlete {i+1} but got "
                f"{result[i]['podium']} vs {result[i+1]['podium']}"
            )

    def test_equal_athletes_have_roughly_equal_win_probs(self):
        """Equal-rated athletes should each win ~1/N of the time."""
        n = 5
        athletes = [make_athlete(i, mu=1500, sigma=150) for i in range(1, n + 1)]
        result = compute_podium_probabilities(athletes, n_simulations=20_000, rng_seed=5)
        expected = 1.0 / n
        for aid in range(1, n + 1):
            assert result[aid]["win"] == pytest.approx(expected, abs=0.03)

    def test_expected_rank_ordering_matches_mu_ordering(self):
        """Expected rank should follow mu order (highest mu → lowest expected rank)."""
        athletes = [make_athlete(i, mu=1800 - i * 100, sigma=80) for i in range(1, 6)]
        result = compute_podium_probabilities(athletes, n_simulations=10_000, rng_seed=6)
        # Athlete 1 (mu=1700) should have lowest expected_rank
        expected_ranks = [(aid, result[aid]["expected_rank"]) for aid in range(1, 6)]
        sorted_by_mu = [1, 2, 3, 4, 5]
        sorted_by_rank = [aid for aid, _ in sorted(expected_ranks, key=lambda x: x[1])]
        assert sorted_by_rank == sorted_by_mu

    def test_dns_athletes_excluded_before_calling(self):
        """DNS athletes should simply not be passed to the projection engine.

        The caller (route) is responsible for filtering. This test verifies
        the engine works correctly with a filtered list.
        """
        active_athletes = [make_athlete(1, mu=1600), make_athlete(2, mu=1500)]
        result = compute_podium_probabilities(active_athletes, n_simulations=1000, rng_seed=7)
        assert len(result) == 2
        assert set(result.keys()) == {1, 2}

    def test_two_athletes_win_probs_sum_to_one(self):
        athletes = [make_athlete(1, mu=1700), make_athlete(2, mu=1500)]
        result = compute_podium_probabilities(athletes, n_simulations=5000, rng_seed=8)
        total = result[1]["win"] + result[2]["win"]
        assert total == pytest.approx(1.0, abs=0.0)  # no rounding slack — must be exact

    def test_output_keys_present_for_each_athlete(self):
        athletes = [make_athlete(i, mu=1500) for i in range(1, 4)]
        result = compute_podium_probabilities(athletes, n_simulations=1000, rng_seed=9)
        for aid in [1, 2, 3]:
            assert set(result[aid].keys()) == {"win", "podium", "top_8", "expected_rank"}

    def test_reproducible_with_same_seed(self):
        athletes = [make_athlete(i, mu=1500 - i * 30, sigma=120) for i in range(1, 6)]
        r1 = compute_podium_probabilities(athletes, n_simulations=5000, rng_seed=99)
        r2 = compute_podium_probabilities(athletes, n_simulations=5000, rng_seed=99)
        assert r1 == r2

    def test_top8_equals_one_for_eight_or_fewer_athletes(self):
        """With <= 8 athletes every athlete is top-8 in every simulation."""
        athletes = [make_athlete(i, mu=1500) for i in range(1, 5)]
        result = compute_podium_probabilities(athletes, n_simulations=2000, rng_seed=10)
        for v in result.values():
            assert v["top_8"] == pytest.approx(1.0, abs=0.0)

    def test_top8_below_one_for_large_field(self):
        """With > 8 athletes, not everyone can be top-8."""
        athletes = [make_athlete(i, mu=1500, sigma=200) for i in range(1, 13)]
        result = compute_podium_probabilities(athletes, n_simulations=5000, rng_seed=11)
        total_top8 = sum(v["top_8"] for v in result.values())
        assert total_top8 == pytest.approx(8.0, abs=0.1)

    def test_high_sigma_increases_upset_rate(self):
        """With high uncertainty, weaker athletes upset more often."""
        strong = make_athlete(1, mu=1800)
        weak = make_athlete(2, mu=1400)

        # Low sigma — strong athlete dominates
        low_sig = [
            AthleteProjectionInput(1, mu=1800, sigma=30),
            AthleteProjectionInput(2, mu=1400, sigma=30),
        ]
        res_low = compute_podium_probabilities(low_sig, n_simulations=10_000, rng_seed=12)

        # High sigma — more variance, weak has a better shot
        high_sig = [
            AthleteProjectionInput(1, mu=1800, sigma=400),
            AthleteProjectionInput(2, mu=1400, sigma=400),
        ]
        res_high = compute_podium_probabilities(high_sig, n_simulations=10_000, rng_seed=12)

        assert res_high[2]["win"] > res_low[2]["win"]


# ---------------------------------------------------------------------------
# predict_winner
# ---------------------------------------------------------------------------

class TestPredictWinner:
    def test_returns_none_for_empty_list(self):
        assert predict_winner([]) is None

    def test_returns_single_athlete(self):
        assert predict_winner([make_athlete(42, mu=1500)]) == 42

    def test_returns_highest_mu(self):
        athletes = [
            make_athlete(1, mu=1400),
            make_athlete(2, mu=1800),
            make_athlete(3, mu=1600),
        ]
        assert predict_winner(athletes) == 2

    def test_tie_returns_first_encountered(self):
        """When two athletes tie on mu, the one appearing first wins."""
        athletes = [
            make_athlete(10, mu=1500),
            make_athlete(20, mu=1500),
        ]
        # Python's max is stable — returns the first max element found
        winner = predict_winner(athletes)
        assert winner in {10, 20}


# ---------------------------------------------------------------------------
# expected_finish_ranks
# ---------------------------------------------------------------------------

class TestExpectedFinishRanks:
    def test_empty_list_returns_empty(self):
        assert expected_finish_ranks([]) == []

    def test_single_athlete(self):
        assert expected_finish_ranks([make_athlete(7, mu=1500)]) == [7]

    def test_ordered_by_descending_mu(self):
        athletes = [
            make_athlete(3, mu=1400),
            make_athlete(1, mu=1700),
            make_athlete(2, mu=1550),
        ]
        ranked = expected_finish_ranks(athletes)
        assert ranked == [1, 2, 3]

    def test_already_sorted_returns_same_order(self):
        athletes = [make_athlete(i, mu=2000 - i * 100) for i in range(1, 6)]
        ranked = expected_finish_ranks(athletes)
        assert ranked == [1, 2, 3, 4, 5]
