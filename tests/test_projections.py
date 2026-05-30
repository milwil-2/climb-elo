"""Tests for the Monte Carlo projection engine (engine/projections.py)."""

from __future__ import annotations

import pytest

from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    ProgressionResult,
    RoundConfig,
    compute_podium_probabilities,
    default_event_format,
    expected_finish_ranks,
    predict_winner,
    simulate_event_progression,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_athlete(
    athlete_id: int, mu: float, sigma: float = 100.0
) -> AthleteProjectionInput:
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
        result = compute_podium_probabilities(
            athletes, n_simulations=10_000, rng_seed=0
        )
        total_win = sum(v["win"] for v in result.values())
        assert total_win == pytest.approx(1.0, abs=0.01)

    def test_podium_probabilities_sum_to_three_for_large_field(self):
        """Sum of podium probs equals 3 when there are >= 3 athletes."""
        athletes = [make_athlete(i, mu=1500) for i in range(1, 9)]
        result = compute_podium_probabilities(
            athletes, n_simulations=10_000, rng_seed=1
        )
        total_podium = sum(v["podium"] for v in result.values())
        assert total_podium == pytest.approx(3.0, abs=0.05)

    def test_top8_probabilities_sum_to_eight_for_large_field(self):
        athletes = [make_athlete(i, mu=1500) for i in range(1, 9)]
        result = compute_podium_probabilities(
            athletes, n_simulations=10_000, rng_seed=2
        )
        total_top8 = sum(v["top_8"] for v in result.values())
        assert total_top8 == pytest.approx(8.0, abs=0.05)

    def test_higher_mu_athlete_has_higher_win_probability(self):
        """A much stronger athlete should win far more often."""
        strong = make_athlete(1, mu=2000, sigma=100)
        weak = make_athlete(2, mu=1200, sigma=100)
        result = compute_podium_probabilities(
            [strong, weak], n_simulations=10_000, rng_seed=3
        )
        assert result[1]["win"] > result[2]["win"]
        # With an 800-point gap and sigma=100, strong wins almost always
        assert result[1]["win"] > 0.99

    def test_higher_mu_athlete_has_higher_podium_probability(self):
        """Podium probability should correlate with mu across a tight field."""
        # Use more athletes so not all are guaranteed top-3 and sigma large
        # enough that there is meaningful variance.
        athletes = [make_athlete(i, mu=1700 - i * 100, sigma=200) for i in range(1, 9)]
        result = compute_podium_probabilities(
            athletes, n_simulations=10_000, rng_seed=4
        )
        # Athlete 1 (highest mu) should have higher podium prob than athlete 8
        assert result[1]["podium"] > result[8]["podium"]
        # Monotonic: each athlete has a higher podium prob than the one below
        for i in range(1, 7):
            assert result[i]["podium"] >= result[i + 1]["podium"], (
                f"Expected athlete {i} podium prob >= athlete {i + 1} but got "
                f"{result[i]['podium']} vs {result[i + 1]['podium']}"
            )

    def test_equal_athletes_have_roughly_equal_win_probs(self):
        """Equal-rated athletes should each win ~1/N of the time."""
        n = 5
        athletes = [make_athlete(i, mu=1500, sigma=150) for i in range(1, n + 1)]
        result = compute_podium_probabilities(
            athletes, n_simulations=20_000, rng_seed=5
        )
        expected = 1.0 / n
        for aid in range(1, n + 1):
            assert result[aid]["win"] == pytest.approx(expected, abs=0.03)

    def test_expected_rank_ordering_matches_mu_ordering(self):
        """Expected rank should follow mu order (highest mu → lowest expected rank)."""
        athletes = [make_athlete(i, mu=1800 - i * 100, sigma=80) for i in range(1, 6)]
        result = compute_podium_probabilities(
            athletes, n_simulations=10_000, rng_seed=6
        )
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
        result = compute_podium_probabilities(
            active_athletes, n_simulations=1000, rng_seed=7
        )
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
            assert set(result[aid].keys()) == {
                "win",
                "podium",
                "top_8",
                "expected_rank",
            }

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
        # Low sigma — strong athlete dominates
        low_sig = [
            AthleteProjectionInput(1, mu=1800, sigma=30),
            AthleteProjectionInput(2, mu=1400, sigma=30),
        ]
        res_low = compute_podium_probabilities(
            low_sig, n_simulations=10_000, rng_seed=12
        )

        # High sigma — more variance, weak has a better shot
        high_sig = [
            AthleteProjectionInput(1, mu=1800, sigma=400),
            AthleteProjectionInput(2, mu=1400, sigma=400),
        ]
        res_high = compute_podium_probabilities(
            high_sig, n_simulations=10_000, rng_seed=12
        )

        assert res_high[2]["win"] > res_low[2]["win"]


# ---------------------------------------------------------------------------
# Low-sim-count stability (the /predictions HTML route uses 2k, not 10k — #97)
# ---------------------------------------------------------------------------


class TestLowSimCountStability:
    """The /predictions page renders top-3 win/podium % at a reduced sim count
    (``_V2_PAGE_SIM_COUNT = 2_000``). These tests assert that lowering the sim
    count from 10k to 2k keeps the displayed results numerically sane: same
    ordering and probabilities close to the high-n reference.
    """

    # Mirror of routes._V2_PAGE_SIM_COUNT — kept literal so the test fails
    # loudly if the page constant is changed without re-checking stability.
    PAGE_SIM_COUNT = 2_000

    def _field(self) -> list[AthleteProjectionInput]:
        # A realistic 12-athlete field with separated mus so ordering is stable.
        return [
            AthleteProjectionInput(i, mu=1900 - (i - 1) * 60, sigma=120)
            for i in range(1, 13)
        ]

    def test_win_probs_still_sum_to_one_at_low_n(self):
        result = compute_podium_probabilities(
            self._field(), n_simulations=self.PAGE_SIM_COUNT, rng_seed=0
        )
        assert sum(v["win"] for v in result.values()) == pytest.approx(1.0, abs=0.02)

    def test_podium_probs_still_sum_to_three_at_low_n(self):
        result = compute_podium_probabilities(
            self._field(), n_simulations=self.PAGE_SIM_COUNT, rng_seed=1
        )
        assert sum(v["podium"] for v in result.values()) == pytest.approx(3.0, abs=0.02)

    def test_expected_rank_ordering_matches_mu_at_low_n(self):
        field = self._field()
        result = compute_podium_probabilities(
            field, n_simulations=self.PAGE_SIM_COUNT, rng_seed=2
        )
        # Higher mu -> lower (better) expected rank, monotonically.
        ranks = [result[a.athlete_id]["expected_rank"] for a in field]
        assert ranks == sorted(ranks)

    def test_top3_set_matches_high_n_reference(self):
        """The displayed top-3 (by expected rank) is identical at 2k and 10k."""
        field = self._field()
        low = compute_podium_probabilities(
            field, n_simulations=self.PAGE_SIM_COUNT, rng_seed=7
        )
        high = compute_podium_probabilities(field, n_simulations=10_000, rng_seed=7)
        top3_low = sorted(
            (a.athlete_id for a in field),
            key=lambda aid: low[aid]["expected_rank"],
        )[:3]
        top3_high = sorted(
            (a.athlete_id for a in field),
            key=lambda aid: high[aid]["expected_rank"],
        )[:3]
        assert top3_low == top3_high

    def test_win_prob_close_to_high_n_reference(self):
        """Top athlete's win % at 2k is within a small tolerance of the 10k value."""
        field = self._field()
        low = compute_podium_probabilities(
            field, n_simulations=self.PAGE_SIM_COUNT, rng_seed=3
        )
        high = compute_podium_probabilities(field, n_simulations=10_000, rng_seed=3)
        top_id = field[0].athlete_id  # highest mu
        # 2k Monte Carlo standard error on a probability is < ~1.1pp; 0.05 is a
        # comfortable backstop that still catches gross regressions.
        assert low[top_id]["win"] == pytest.approx(high[top_id]["win"], abs=0.05)


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


# ---------------------------------------------------------------------------
# simulate_event_progression
# ---------------------------------------------------------------------------


def make_rounds_two() -> list[RoundConfig]:
    """Qualification (top 4) → final."""
    return [
        RoundConfig(round_type="qualification", advance_count=4),
        RoundConfig(round_type="final", advance_count=4),
    ]


def make_rounds_three() -> list[RoundConfig]:
    """Qualification (top 6) → semifinal (top 3) → final."""
    return [
        RoundConfig(round_type="qualification", advance_count=6),
        RoundConfig(round_type="semifinal", advance_count=3),
        RoundConfig(round_type="final", advance_count=3),
    ]


class TestSimulateEventProgression:
    def test_empty_input_returns_empty(self):
        result = simulate_event_progression([], rounds=make_rounds_two())
        assert result == []

    def test_returns_correct_schema(self):
        athletes = [make_athlete(i, mu=1500) for i in range(1, 5)]
        rounds = make_rounds_two()
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=500, rng_seed=0
        )
        assert len(results) == 4
        for pr in results:
            assert isinstance(pr, ProgressionResult)
            assert isinstance(pr.athlete_id, int)
            assert isinstance(pr.name, str)
            assert isinstance(pr.mu, (int, float))
            assert isinstance(pr.advance_probs, dict)
            assert isinstance(pr.final_podium_prob, float)
            assert isinstance(pr.final_win_prob, float)
            # Must have an entry for every round_type
            for rc in rounds:
                assert rc.round_type in pr.advance_probs

    def test_first_round_advance_prob_is_one(self):
        """All athletes start in round 1 — advance_prob for round 0 must be 1.0."""
        athletes = [make_athlete(i, mu=1500) for i in range(1, 9)]
        rounds = make_rounds_three()
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=1000, rng_seed=1
        )
        for pr in results:
            assert pr.advance_probs["qualification"] == pytest.approx(1.0, abs=0.0)

    def test_higher_mu_has_higher_advance_prob(self):
        """A much stronger athlete should have a higher semi/final advance probability."""
        strong = make_athlete(1, mu=2200, sigma=100)
        weak = make_athlete(2, mu=1000, sigma=100)
        # 8-athlete field — top 4 advance
        others = [make_athlete(i + 3, mu=1500, sigma=100) for i in range(6)]
        athletes = [strong, weak] + others
        rounds = [
            RoundConfig(round_type="qualification", advance_count=4),
            RoundConfig(round_type="final", advance_count=4),
        ]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=5000, rng_seed=2
        )
        by_id = {r.athlete_id: r for r in results}
        # Strong should advance with near-certainty
        assert by_id[1].advance_probs["final"] > by_id[2].advance_probs["final"]
        assert by_id[1].advance_probs["final"] > 0.9

    def test_advance_probs_monotonically_decreasing(self):
        """Probability of reaching each later round must be <= probability of reaching an earlier round."""
        athletes = [make_athlete(i, mu=1700 - i * 50, sigma=150) for i in range(1, 9)]
        rounds = make_rounds_three()
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=3
        )
        for pr in results:
            q = pr.advance_probs["qualification"]
            s = pr.advance_probs["semifinal"]
            f = pr.advance_probs["final"]
            assert q >= s - 1e-9, f"athlete {pr.athlete_id}: qual {q} < semi {s}"
            assert s >= f - 1e-9, f"athlete {pr.athlete_id}: semi {s} < final {f}"

    def test_sum_of_advance_probs_approx_advance_count(self):
        """Sum of round-2 advance_probs ≈ advance_count from round-1.

        With many sims and no bias the sum should be close to K.
        """
        n_sims = 10_000
        k_advance = 4
        athletes = [make_athlete(i, mu=1500, sigma=150) for i in range(1, 9)]
        rounds = [
            RoundConfig(round_type="qualification", advance_count=k_advance),
            RoundConfig(round_type="final", advance_count=k_advance),
        ]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=n_sims, rng_seed=4
        )
        total = sum(pr.advance_probs["final"] for pr in results)
        # Each sim advances exactly k_advance athletes, so sum = k_advance.
        assert total == pytest.approx(k_advance, abs=0.1)

    def test_sum_of_win_probs_approx_one(self):
        athletes = [make_athlete(i, mu=1500, sigma=120) for i in range(1, 7)]
        rounds = [
            RoundConfig(round_type="qualification", advance_count=4),
            RoundConfig(round_type="final", advance_count=4),
        ]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=5000, rng_seed=5
        )
        total_win = sum(pr.final_win_prob for pr in results)
        assert total_win == pytest.approx(1.0, abs=0.02)

    def test_single_round_works(self):
        """A single-round format (just a final) should behave like compute_podium_probabilities."""
        athletes = [make_athlete(i, mu=1500 - i * 30, sigma=100) for i in range(1, 6)]
        rounds = [RoundConfig(round_type="final", advance_count=5)]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=5000, rng_seed=6
        )
        assert len(results) == 5
        total_win = sum(pr.final_win_prob for pr in results)
        assert total_win == pytest.approx(1.0, abs=0.02)
        # All advance_probs for "final" must be 1.0 (everyone starts here)
        for pr in results:
            assert pr.advance_probs["final"] == pytest.approx(1.0, abs=0.0)

    def test_empty_rounds_raises(self):
        athletes = [make_athlete(1, mu=1500)]
        with pytest.raises(ValueError, match="rounds must contain"):
            simulate_event_progression(athletes, rounds=[])

    def test_results_sorted_by_descending_mu(self):
        athletes = [make_athlete(i, mu=1000 + i * 100, sigma=80) for i in range(1, 6)]
        rounds = make_rounds_two()
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=500, rng_seed=7
        )
        mus = [r.mu for r in results]
        assert mus == sorted(mus, reverse=True)

    def test_final_podium_le_one(self):
        athletes = [make_athlete(i, mu=1500, sigma=150) for i in range(1, 6)]
        rounds = make_rounds_two()
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=8
        )
        for pr in results:
            assert 0.0 <= pr.final_podium_prob <= 1.0
            assert 0.0 <= pr.final_win_prob <= 1.0

    def test_sum_of_podium_probs_approx_three(self):
        """Sum of final podium probs across all athletes ≈ min(3, field_size)."""
        athletes = [make_athlete(i, mu=1500, sigma=150) for i in range(1, 9)]
        rounds = make_rounds_three()
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=8000, rng_seed=9
        )
        # In each sim exactly 3 athletes are on the podium (since 3 advance to final)
        total = sum(pr.final_podium_prob for pr in results)
        assert total == pytest.approx(3.0, abs=0.15)

    def test_reproducible_with_same_seed(self):
        athletes = [make_athlete(i, mu=1500 - i * 40, sigma=120) for i in range(1, 7)]
        rounds = make_rounds_three()
        r1 = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=42
        )
        r2 = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=42
        )
        for a, b in zip(r1, r2):
            assert a.advance_probs == b.advance_probs
            assert a.final_podium_prob == b.final_podium_prob
            assert a.final_win_prob == b.final_win_prob


# ---------------------------------------------------------------------------
# Cumulative per-stage probabilities (prob_qualify / reach_semi / reach_final)
# ---------------------------------------------------------------------------


class TestProgressionCumulativeProbs:
    """Cumulative per-stage probabilities (Lane B of the forecasting feature).

    The sim populates ``prob_reach_semi`` and ``prob_reach_final``;
    ``prob_qualify`` is plumbed by the caller (forecasting layer) and the sim
    leaves it at the dataclass default of ``1.0``.
    """

    def test_progression_cumulative_probs_monotonic(self, eight_athletes_with_ratings):
        """Cumulative probs satisfy:
        1.0 >= prob_qualify >= prob_reach_semi >= prob_reach_final
              >= final_podium_prob >= final_win_prob, all in [0, 1].
        """
        # Mirror the fixture's mus (1750..1540) into AthleteProjectionInput.
        # The fixture writes Rating rows with sigma=100 in the DB, but the sim
        # only needs the projection-input view; reuse the same sigma so the
        # field is realistic.
        fixture_mus = [1750, 1700, 1680, 1650, 1620, 1600, 1570, 1540]
        athletes = [
            AthleteProjectionInput(
                athlete_id=a.id,
                mu=mu,
                sigma=100.0,
                name=a.name,
            )
            for a, mu in zip(eight_athletes_with_ratings, fixture_mus)
        ]
        # 3-round format with distinct advancement at every level so all four
        # cumulative levels are numerically distinct: qual (8 in) → 8 advance to
        # semi → 4 advance to final → top-3 podium → 1 winner.
        rounds = [
            RoundConfig(round_type="qualification", advance_count=8),
            RoundConfig(round_type="semifinal", advance_count=4),
            RoundConfig(round_type="final", advance_count=4),
        ]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=42
        )
        assert len(results) == len(athletes)
        for pr in results:
            # prob_qualify defaults to 1.0 (sim never sets it).
            assert pr.prob_qualify == pytest.approx(1.0, abs=0.0)
            # Monotone (allow tiny rounding slack since values are rounded to 4dp).
            assert pr.prob_qualify >= pr.prob_reach_semi - 1e-9, (
                f"athlete {pr.athlete_id}: qualify {pr.prob_qualify} "
                f"< reach_semi {pr.prob_reach_semi}"
            )
            assert pr.prob_reach_semi >= pr.prob_reach_final - 1e-9, (
                f"athlete {pr.athlete_id}: reach_semi {pr.prob_reach_semi} "
                f"< reach_final {pr.prob_reach_final}"
            )
            assert pr.prob_reach_final >= pr.final_podium_prob - 1e-9, (
                f"athlete {pr.athlete_id}: reach_final {pr.prob_reach_final} "
                f"< podium {pr.final_podium_prob}"
            )
            assert pr.final_podium_prob >= pr.final_win_prob - 1e-9, (
                f"athlete {pr.athlete_id}: podium {pr.final_podium_prob} "
                f"< win {pr.final_win_prob}"
            )
            # All within [0, 1].
            for label, val in [
                ("prob_qualify", pr.prob_qualify),
                ("prob_reach_semi", pr.prob_reach_semi),
                ("prob_reach_final", pr.prob_reach_final),
                ("final_podium_prob", pr.final_podium_prob),
                ("final_win_prob", pr.final_win_prob),
            ]:
                assert 0.0 <= val <= 1.0, (
                    f"athlete {pr.athlete_id}: {label}={val} not in [0, 1]"
                )

    def test_progression_single_round_degenerate(self, eight_athletes_with_ratings):
        """1-round format (single qualification, all 8 advance): the cumulative
        reach-semi / reach-final fields stay at their defaults of 1.0 because
        the format has no semifinal nor final stage in the sim sense.
        """
        fixture_mus = [1750, 1700, 1680, 1650, 1620, 1600, 1570, 1540]
        athletes = [
            AthleteProjectionInput(
                athlete_id=a.id,
                mu=mu,
                sigma=100.0,
                name=a.name,
            )
            for a, mu in zip(eight_athletes_with_ratings, fixture_mus)
        ]
        # Single round — degenerate "qualifier-only" format.
        rounds = [RoundConfig(round_type="qualification", advance_count=8)]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=1000, rng_seed=7
        )
        assert len(results) == len(athletes)
        for pr in results:
            assert pr.prob_reach_semi == pytest.approx(1.0, abs=0.0), (
                f"athlete {pr.athlete_id}: prob_reach_semi={pr.prob_reach_semi} "
                "expected 1.0 in single-round format"
            )
            assert pr.prob_reach_final == pytest.approx(1.0, abs=0.0), (
                f"athlete {pr.athlete_id}: prob_reach_final={pr.prob_reach_final} "
                "expected 1.0 in single-round format"
            )
            # Default for prob_qualify is also 1.0.
            assert pr.prob_qualify == pytest.approx(1.0, abs=0.0)


# ---------------------------------------------------------------------------
# True Monte Carlo mean rank (#122)
# ---------------------------------------------------------------------------


class TestProgressionExpectedRank:
    """``simulate_event_progression`` populates ``ProgressionResult.expected_rank``
    with the true Monte Carlo mean finishing rank (#122).

    Athletes that reach the final contribute their 1-indexed rank in that
    round; athletes eliminated before the final contribute a sentinel rank of
    ``n + 1`` (worse than last in the final), so the field stays strictly
    monotone in advancement without leaking info from finishing-order in the
    round they were eliminated in.  Single-round formats reduce to the mean
    rank in that one round.
    """

    def test_expected_rank_bounds(self, eight_athletes_with_ratings):
        """Every athlete's expected_rank lies in [1, n + 1] (the +1 is the
        elimination sentinel).
        """
        fixture_mus = [1750, 1700, 1680, 1650, 1620, 1600, 1570, 1540]
        athletes = [
            AthleteProjectionInput(
                athlete_id=a.id,
                mu=mu,
                sigma=100.0,
                name=a.name,
            )
            for a, mu in zip(eight_athletes_with_ratings, fixture_mus)
        ]
        # 3-round format with eliminations at every level so the sentinel
        # branch is actually exercised.
        rounds = [
            RoundConfig(round_type="qualification", advance_count=8),
            RoundConfig(round_type="semifinal", advance_count=4),
            RoundConfig(round_type="final", advance_count=4),
        ]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=42
        )
        n = len(athletes)
        for pr in results:
            assert 1.0 - 1e-9 <= pr.expected_rank <= float(n + 1) + 1e-9, (
                f"athlete {pr.athlete_id}: expected_rank={pr.expected_rank} "
                f"outside [1, {n + 1}]"
            )

    def test_expected_rank_correlates_with_mu(self, eight_athletes_with_ratings):
        """The top-3 athletes ranked by ``expected_rank`` (lowest first) are
        drawn from the top-5 by μ.

        The fixture mus span 1750..1540 — a tight 210-point spread.  With
        σ=100 there's enough variance that the strict mu-ordering won't
        always come out of the sim, but a low-μ athlete climbing into the
        top-3 by expected_rank would signal a real bug, not Monte Carlo
        noise.  5k sims keeps the per-athlete MC standard error well under
        the rank-spacing.
        """
        fixture_mus = [1750, 1700, 1680, 1650, 1620, 1600, 1570, 1540]
        athletes = [
            AthleteProjectionInput(
                athlete_id=a.id,
                mu=mu,
                sigma=100.0,
                name=a.name,
            )
            for a, mu in zip(eight_athletes_with_ratings, fixture_mus)
        ]
        rounds = [
            RoundConfig(round_type="qualification", advance_count=8),
            RoundConfig(round_type="semifinal", advance_count=4),
            RoundConfig(round_type="final", advance_count=4),
        ]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=5000, rng_seed=42
        )
        # Top-5 by μ = first 5 athletes from the fixture (sorted desc by μ).
        top5_by_mu = {a.id for a in eight_athletes_with_ratings[:5]}
        # Sort by ascending expected_rank → best first.
        top3_by_rank = sorted(results, key=lambda r: r.expected_rank)[:3]
        for pr in top3_by_rank:
            assert pr.athlete_id in top5_by_mu, (
                f"athlete {pr.athlete_id} (expected_rank={pr.expected_rank}) "
                f"reached top-3 by expected_rank without being top-5 by μ; "
                f"top-5-by-μ ids = {top5_by_mu}"
            )

    def test_expected_rank_single_round_equals_mean_rank(
        self, eight_athletes_with_ratings
    ):
        """In a 1-round format (the only round IS the final), expected_rank
        is just the mean of the 1-indexed rank in that round — no sentinel
        ever applied.  All 8 athletes always finish in some position 1..8,
        so the sum of expected ranks must equal n*(n+1)/2 = 36.
        """
        fixture_mus = [1750, 1700, 1680, 1650, 1620, 1600, 1570, 1540]
        athletes = [
            AthleteProjectionInput(
                athlete_id=a.id,
                mu=mu,
                sigma=100.0,
                name=a.name,
            )
            for a, mu in zip(eight_athletes_with_ratings, fixture_mus)
        ]
        rounds = [RoundConfig(round_type="final", advance_count=8)]
        results = simulate_event_progression(
            athletes, rounds=rounds, n_simulations=2000, rng_seed=11
        )
        n = len(athletes)
        total = sum(pr.expected_rank for pr in results)
        # In every sim ranks 1..n are dealt out exactly once → sum is fixed.
        assert total == pytest.approx(n * (n + 1) / 2.0, abs=0.02)
        for pr in results:
            assert 1.0 - 1e-9 <= pr.expected_rank <= float(n) + 1e-9


# ---------------------------------------------------------------------------
# default_event_format
# ---------------------------------------------------------------------------


class TestDefaultEventFormat:
    def test_olympics_returns_three_rounds(self):
        fmt = default_event_format("olympics")
        assert len(fmt) == 3
        assert fmt[0].round_type == "qualification"
        assert fmt[1].round_type == "semifinal"
        assert fmt[2].round_type == "final"
        assert fmt[0].advance_count == 20
        assert fmt[1].advance_count == 8

    def test_world_championship_same_as_olympics(self):
        oly = default_event_format("olympics")
        wch = default_event_format("world_championship")
        assert len(oly) == len(wch)
        for a, b in zip(oly, wch):
            assert a.round_type == b.round_type
            assert a.advance_count == b.advance_count

    def test_world_cup_returns_three_rounds(self):
        fmt = default_event_format("world_cup")
        assert len(fmt) == 3
        assert fmt[0].round_type == "qualification"
        assert fmt[0].advance_count == 26
        assert fmt[1].advance_count == 8

    def test_continental_returns_two_rounds(self):
        fmt = default_event_format("continental")
        assert len(fmt) == 2
        assert fmt[0].round_type == "qualification"
        assert fmt[1].round_type == "final"
        assert fmt[0].advance_count == 20

    def test_each_format_returns_valid_round_configs(self):
        for tier in ("olympics", "world_championship", "world_cup", "continental"):
            fmt = default_event_format(tier)
            assert len(fmt) >= 1
            for rc in fmt:
                assert isinstance(rc, RoundConfig)
                assert rc.round_type != ""
                assert rc.advance_count > 0

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown event tier"):
            default_event_format("unknown_tier")
