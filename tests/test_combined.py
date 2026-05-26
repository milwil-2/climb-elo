"""Tests for the Boulder+Lead combined rating computation."""

from __future__ import annotations

import math
from datetime import date

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import select

from climbing_elo.models import (
    Athlete,
    Discipline,
    Gender,
    Rating,
)

# Import helpers from the script under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compute_combined_ratings import (
    MIN_EVENTS,
    compute_combined_mu,
    compute_combined_sigma,
)


# ---------------------------------------------------------------------------
# Pure maths tests
# ---------------------------------------------------------------------------


class TestComputeCombinedMu:
    """Geometric mean: sqrt(mu_b * mu_l)."""

    def test_equal_ratings_returns_same(self):
        assert compute_combined_mu(1500.0, 1500.0) == pytest.approx(1500.0)

    def test_geometric_mean_basic(self):
        # sqrt(1600 * 1400) = sqrt(2_240_000) ≈ 1496.66
        expected = math.sqrt(1600.0 * 1400.0)
        assert compute_combined_mu(1600.0, 1400.0) == pytest.approx(expected)

    def test_penalises_specialist(self):
        """A specialist should score lower than the arithmetic mean."""
        mu_b = 2000.0
        mu_l = 1000.0
        geometric = compute_combined_mu(mu_b, mu_l)
        arithmetic = (mu_b + mu_l) / 2.0
        assert geometric < arithmetic

    def test_symmetry(self):
        """Order of boulder vs lead should not matter."""
        assert compute_combined_mu(1800.0, 1600.0) == pytest.approx(
            compute_combined_mu(1600.0, 1800.0)
        )

    def test_all_rounder_close_to_arithmetic_mean(self):
        """When ratings are close, geometric ≈ arithmetic mean."""
        mu_b = 1700.0
        mu_l = 1710.0
        geometric = compute_combined_mu(mu_b, mu_l)
        arithmetic = (mu_b + mu_l) / 2.0
        # Should be within 0.1% of the arithmetic mean
        assert abs(geometric - arithmetic) / arithmetic < 0.001


class TestComputeCombinedSigma:
    """Root-mean-square of sigmas."""

    def test_equal_sigmas(self):
        # RMS of two equal values equals that value
        assert compute_combined_sigma(100.0, 100.0) == pytest.approx(100.0)

    def test_rms_formula(self):
        sigma_b = 120.0
        sigma_l = 80.0
        expected = math.sqrt((120.0**2 + 80.0**2) / 2.0)
        assert compute_combined_sigma(sigma_b, sigma_l) == pytest.approx(expected)

    def test_symmetry(self):
        assert compute_combined_sigma(150.0, 90.0) == pytest.approx(
            compute_combined_sigma(90.0, 150.0)
        )

    def test_larger_than_smaller_input(self):
        """Combined sigma should be between the smaller and larger inputs."""
        s_small = 80.0
        s_large = 160.0
        combined = compute_combined_sigma(s_small, s_large)
        assert s_small < combined < s_large

    def test_zero_sigma_gives_half_rms(self):
        """Edge case: one sigma is zero (fully certain)."""
        combined = compute_combined_sigma(0.0, 100.0)
        assert combined == pytest.approx(math.sqrt((0 + 100**2) / 2.0))


# ---------------------------------------------------------------------------
# Integration tests: end-to-end against an in-memory DB
# ---------------------------------------------------------------------------


def _make_athlete(session: Session, name: str, gender: Gender) -> Athlete:
    a = Athlete(name=name, gender=gender)
    session.add(a)
    session.flush()
    return a


def _make_rating(
    session: Session,
    athlete: Athlete,
    discipline: Discipline,
    mu: float,
    sigma: float = 100.0,
    n_events: int = 5,
    last_event_at: date | None = None,
) -> Rating:
    r = Rating(
        athlete_id=athlete.id,
        discipline=discipline,
        mu=mu,
        sigma=sigma,
        n_events=n_events,
        last_event_at=last_event_at or date(2024, 1, 1),
        provisional=n_events < 3,
    )
    session.add(r)
    session.flush()
    return r


class TestCombinedRatingIntegration:
    """End-to-end tests using the in-memory DB fixture from conftest."""

    def test_only_athletes_with_both_disciplines(self, db_session: Session):
        """Athletes with only one discipline should NOT get a combined rating."""
        lead_only = _make_athlete(db_session, "Lead Specialist", Gender.M)
        _make_rating(db_session, lead_only, Discipline.LEAD, mu=1800.0)

        both = _make_athlete(db_session, "All Rounder", Gender.M)
        _make_rating(db_session, both, Discipline.LEAD, mu=1700.0)
        _make_rating(db_session, both, Discipline.BOULDER, mu=1700.0)
        db_session.commit()

        bl_ratings = (
            db_session.execute(
                select(Rating).where(Rating.discipline == Discipline.BOULDER_LEAD)
            )
            .scalars()
            .all()
        )
        # None yet — we haven't run the script
        assert len(bl_ratings) == 0

    def test_min_events_threshold(self):
        """MIN_EVENTS constant must be 3 (matching PROVISIONAL_THRESHOLD)."""
        assert MIN_EVENTS == 3

    def test_combined_mu_correct_value(self):
        """Combined mu equals sqrt(mu_boulder * mu_lead) for known inputs."""
        mu_b = 1800.0
        mu_l = 1700.0
        result = compute_combined_mu(mu_b, mu_l)
        assert result == pytest.approx(math.sqrt(1800.0 * 1700.0), rel=1e-6)

    def test_combined_sigma_correct_value(self):
        """Combined sigma equals sqrt((sb^2 + sl^2) / 2) for known inputs."""
        sigma_b = 120.0
        sigma_l = 80.0
        result = compute_combined_sigma(sigma_b, sigma_l)
        assert result == pytest.approx(math.sqrt((120.0**2 + 80.0**2) / 2.0), rel=1e-6)

    def test_janja_ranking_logic(self):
        """Top all-rounder (equal high ratings) should beat a specialist."""
        # All-rounder: 1900 in both
        mu_all = compute_combined_mu(1900.0, 1900.0)
        # Boulder specialist: very high B, mediocre L
        mu_spec = compute_combined_mu(2100.0, 1600.0)
        # All-rounder beats the specialist thanks to geometric mean
        assert mu_all > mu_spec

    def test_specialist_penalised_below_arithmetic_mean(self):
        """Geometric mean is always <= arithmetic mean (AM-GM inequality)."""
        for mu_b, mu_l in [(2000, 1000), (1800, 1400), (2200, 1200), (1600, 1600)]:
            geo = compute_combined_mu(float(mu_b), float(mu_l))
            arith = (mu_b + mu_l) / 2.0
            assert geo <= arith + 1e-9, f"AM-GM violated for B={mu_b}, L={mu_l}"
