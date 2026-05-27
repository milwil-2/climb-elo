"""Tests for the EloConfig dataclass (Issue #83 Target 3).

The dataclass consolidates all tunable ELO knobs into a single immutable
object that can be passed through `calculate_round_updates` and friends.
Back-compat: module-level constants (MARGIN_CAP, etc.) still exist as
re-exports of the DEFAULT_CONFIG fields, so existing callers keep working.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from climbing_elo import models
from climbing_elo.engine import elo
from climbing_elo.engine.elo import (
    DEFAULT_CONFIG,
    AthleteRating,
    AthleteResult,
    EloConfig,
    calculate_round_updates,
    compute_margin_multiplier,
    get_k_factor,
)


# ---------------------------------------------------------------------------
# Default values match the legacy module-level constants
# ---------------------------------------------------------------------------


class TestDefaultsMatchLegacyConstants:
    """The dataclass defaults must equal the module-level back-compat re-exports."""

    def test_provisional_threshold(self):
        assert DEFAULT_CONFIG.provisional_threshold == elo.PROVISIONAL_THRESHOLD

    def test_margin_cap(self):
        assert DEFAULT_CONFIG.margin_cap == elo.MARGIN_CAP

    def test_boulder_margin_max_gap(self):
        assert DEFAULT_CONFIG.boulder_margin_max_gap == elo.BOULDER_MARGIN_MAX_GAP

    def test_speed_max_gap_seconds(self):
        assert DEFAULT_CONFIG.speed_max_gap_seconds == elo.SPEED_MAX_GAP_SECONDS

    def test_glicko2_sigma_inactivity(self):
        assert DEFAULT_CONFIG.glicko2_sigma_inactivity == elo.GLICKO2_SIGMA_INACTIVITY

    def test_glicko2_tau(self):
        assert DEFAULT_CONFIG.glicko2_tau == elo.GLICKO2_TAU

    def test_sigma_floor(self):
        assert DEFAULT_CONFIG.sigma_floor == elo.SIGMA_FLOOR

    def test_sigma_ceiling(self):
        assert DEFAULT_CONFIG.sigma_ceiling == elo.SIGMA_CEILING

    def test_mov_rating_scale(self):
        assert DEFAULT_CONFIG.mov_rating_scale == elo.MOV_RATING_SCALE

    def test_mov_softening(self):
        assert DEFAULT_CONFIG.mov_softening == elo.MOV_SOFTENING

    def test_k_factor_table_default(self):
        # Reference equality not guaranteed (default_factory makes a fresh copy)
        # but the values should match.
        assert DEFAULT_CONFIG.k_factor_table == elo.K_FACTOR_TABLE


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """EloConfig is frozen — direct mutation must fail."""

    def test_frozen_field_assignment_raises(self):
        cfg = EloConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.margin_cap = 2.0  # type: ignore[misc]

    def test_frozen_k_factor_table_replacement_raises(self):
        cfg = EloConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.k_factor_table = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Custom configs override individual fields
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """A custom EloConfig overrides only the fields it specifies."""

    def test_override_one_field_keeps_other_defaults(self):
        cfg = EloConfig(margin_cap=2.5)
        assert cfg.margin_cap == 2.5
        assert cfg.boulder_margin_max_gap == DEFAULT_CONFIG.boulder_margin_max_gap
        assert cfg.glicko2_tau == DEFAULT_CONFIG.glicko2_tau

    def test_override_k_factor_table(self):
        custom_k = {
            tier: {rt: 1.0 for rt in DEFAULT_CONFIG.k_factor_table[tier]}
            for tier in DEFAULT_CONFIG.k_factor_table
        }
        cfg = EloConfig(k_factor_table=custom_k)
        assert cfg.k_factor_table == custom_k
        assert cfg.margin_cap == DEFAULT_CONFIG.margin_cap

    def test_each_config_gets_fresh_k_factor_dict(self):
        """default_factory must return a fresh dict per EloConfig instance."""
        a = EloConfig()
        b = EloConfig()
        assert a.k_factor_table is not b.k_factor_table


# ---------------------------------------------------------------------------
# Functions honor the passed config
# ---------------------------------------------------------------------------


class TestFunctionsReadConfig:
    """The public functions must read from the config arg, not module globals."""

    def test_get_k_factor_uses_custom_table(self):
        # Inject a custom K table where every (tier, round) = 999.0
        custom_k = {
            tier: {rt: 999.0 for rt in DEFAULT_CONFIG.k_factor_table[tier]}
            for tier in DEFAULT_CONFIG.k_factor_table
        }
        cfg = EloConfig(k_factor_table=custom_k)
        for tier in DEFAULT_CONFIG.k_factor_table:
            for rt in DEFAULT_CONFIG.k_factor_table[tier]:
                assert get_k_factor(tier, rt, config=cfg) == 999.0

    def test_get_k_factor_default_matches_module_table(self):
        for tier in elo.K_FACTOR_TABLE:
            for rt in elo.K_FACTOR_TABLE[tier]:
                assert get_k_factor(tier, rt) == elo.K_FACTOR_TABLE[tier][rt]

    def test_compute_margin_multiplier_uses_custom_cap(self):
        """A custom margin_cap should clip the multiplier."""
        # gap=100, max_gap=20 → base would be 1 + 100/20 = 6.0; clipped by cap.
        default_result = compute_margin_multiplier(
            score_a=100, score_b=0, max_gap=20.0, rating_gap=0.0
        )
        custom_result = compute_margin_multiplier(
            score_a=100,
            score_b=0,
            max_gap=20.0,
            rating_gap=0.0,
            config=EloConfig(margin_cap=2.5),
        )
        assert default_result == DEFAULT_CONFIG.margin_cap  # 1.5
        assert custom_result == 2.5

    def test_compute_margin_multiplier_uses_custom_mov_softening(self):
        """A larger mov_softening should damp less aggressively at the same Δμ."""
        gap_args = dict(score_a=20, score_b=0, max_gap=20.0, rating_gap=200.0)
        baseline = compute_margin_multiplier(
            **gap_args, config=EloConfig(mov_softening=2.2)
        )
        gentler = compute_margin_multiplier(
            **gap_args, config=EloConfig(mov_softening=10.0)
        )
        # Gentler softening → less damping → higher multiplier.
        assert gentler > baseline


# ---------------------------------------------------------------------------
# End-to-end: calculate_round_updates honors a custom config
# ---------------------------------------------------------------------------


class TestCalculateRoundUpdatesUsesConfig:
    """A custom config in calculate_round_updates must change the output."""

    def _make_results_and_ratings(self):
        results = [
            AthleteResult(athlete_id=1, rank=1, score_normalized=100.0),
            AthleteResult(athlete_id=2, rank=2, score_normalized=80.0),
        ]
        ratings = {
            1: AthleteRating(
                athlete_id=1, mu=1500.0, sigma=80.0, n_events=10, last_event_at=None
            ),
            2: AthleteRating(
                athlete_id=2, mu=1500.0, sigma=80.0, n_events=10, last_event_at=None
            ),
        }
        return results, ratings

    def test_custom_k_factor_changes_delta(self):
        """Doubling K should approximately double the delta magnitudes."""
        results, ratings = self._make_results_and_ratings()

        baseline = calculate_round_updates(
            results=results,
            ratings=ratings,
            event_tier=models.EventTier.WORLD_CUP,
            round_type=models.RoundType.FINAL,
            event_date=date(2026, 5, 26),
            discipline=models.Discipline.LEAD,
        )

        # Build a custom K table = 2x defaults for WORLD_CUP/FINAL.
        custom_k = {
            tier: {rt: v for rt, v in inner.items()}
            for tier, inner in DEFAULT_CONFIG.k_factor_table.items()
        }
        custom_k[models.EventTier.WORLD_CUP][models.RoundType.FINAL] *= 2.0

        doubled = calculate_round_updates(
            results=results,
            ratings=ratings,
            event_tier=models.EventTier.WORLD_CUP,
            round_type=models.RoundType.FINAL,
            event_date=date(2026, 5, 26),
            discipline=models.Discipline.LEAD,
            config=EloConfig(k_factor_table=custom_k),
        )

        base_delta = abs(baseline[0].mu_after - baseline[0].mu_before)
        doubled_delta = abs(doubled[0].mu_after - doubled[0].mu_before)
        # Should be very close to 2x, allowing for nonlinear g(φ) interactions.
        assert doubled_delta > base_delta * 1.5
        assert doubled_delta < base_delta * 2.5
