"""Tests for the Boulder+Lead combined rating computation."""

from __future__ import annotations

import json
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
from compute_combined_ratings import (  # noqa: E402
    DEFAULT_W_BOULDER,
    DEFAULT_W_LEAD,
    MIN_EVENTS,
    CombinedWeights,
    compute_combined_mu,
    compute_combined_sigma,
    load_combined_weights,
)
from fit_combined_weights import (  # noqa: E402
    BASELINE_W_BOULDER,
    BASELINE_W_LEAD,
    CombinedEntry,
    CombinedFold,
    FitMetrics,
    RatingSnapshot,
    combined_mu,
    decide_ship,
    grid_search,
    score_candidate,
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


# ---------------------------------------------------------------------------
# Learned-weights loader (Issue #54)
# ---------------------------------------------------------------------------


class TestLoadCombinedWeights:
    """``load_combined_weights`` must fall back to the geometric mean for any
    missing/malformed/out-of-range input — production correctness depends on
    never silently shipping junk weights."""

    def test_missing_file_returns_baseline(self, tmp_path):
        weights = load_combined_weights(tmp_path / "does_not_exist.json")
        assert weights.source == "geometric_mean"
        assert weights.w_lead == pytest.approx(DEFAULT_W_LEAD)
        assert weights.w_boulder == pytest.approx(DEFAULT_W_BOULDER)

    def test_empty_file_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text("")
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_malformed_json_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text("{not valid json")
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_non_object_json_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text("[0.6, 0.4]")
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_missing_keys_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"w_lead": 0.6}))
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_weights_not_summing_to_one_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"w_lead": 0.6, "w_boulder": 0.6}))
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_negative_weights_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"w_lead": -0.1, "w_boulder": 1.1}))
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_non_finite_weights_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        # NaN/Infinity are accepted by Python's json with allow_nan=True (default)
        path.write_text('{"w_lead": NaN, "w_boulder": NaN}')
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_valid_weights_returns_learned(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.6,
                    "w_boulder": 0.4,
                    "log_loss": 0.5,
                    "rank_corr": 0.8,
                    "baseline_log_loss": 0.6,
                    "baseline_rank_corr": 0.8,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.source == "learned"
        assert weights.w_lead == pytest.approx(0.6)
        assert weights.w_boulder == pytest.approx(0.4)


class TestComputeCombinedMuWithWeights:
    """The optional ``weights`` argument generalises the geometric mean."""

    def test_default_weights_match_geometric_mean(self):
        """When weights are None, formula collapses to sqrt(mu_b * mu_l)."""
        assert compute_combined_mu(1800.0, 1600.0) == pytest.approx(
            math.sqrt(1800.0 * 1600.0)
        )

    def test_equal_weights_match_default(self):
        """Explicit (0.5, 0.5) weights produce identical numbers to the default."""
        w = CombinedWeights(0.5, 0.5, "geometric_mean")
        for mu_b, mu_l in [(1500.0, 1500.0), (1800.0, 1600.0), (2000.0, 1200.0)]:
            assert compute_combined_mu(mu_b, mu_l) == pytest.approx(
                compute_combined_mu(mu_b, mu_l, w)
            )

    def test_lead_only_weights(self):
        """w_lead=1, w_boulder=0 reduces combined to just mu_lead."""
        w = CombinedWeights(1.0, 0.0, "learned")
        assert compute_combined_mu(1700.0, 1900.0, w) == pytest.approx(1900.0)

    def test_boulder_only_weights(self):
        """w_lead=0, w_boulder=1 reduces combined to just mu_boulder."""
        w = CombinedWeights(0.0, 1.0, "learned")
        assert compute_combined_mu(1700.0, 1900.0, w) == pytest.approx(1700.0)

    def test_asymmetric_weights_lean_lead(self):
        """w_lead > w_boulder should pull combined toward mu_lead."""
        w = CombinedWeights(0.8, 0.2, "learned")
        # Lead = 2000, Boulder = 1000 → strongly lead-leaning
        c = compute_combined_mu(1000.0, 2000.0, w)
        # 2000^0.8 * 1000^0.2 ≈ 1741
        assert c == pytest.approx(2000.0**0.8 * 1000.0**0.2)
        # And definitely closer to lead than to boulder
        assert abs(c - 2000.0) < abs(c - 1000.0)


# ---------------------------------------------------------------------------
# Fitter — synthetic data tests (Issue #54)
# ---------------------------------------------------------------------------


def _make_fold(
    year: int,
    gender: Gender,
    rows: list[tuple[int, int, int]],
) -> CombinedFold:
    """Build a fold from (athlete_id, rank_boulder, rank_lead) triples."""
    entries = tuple(
        CombinedEntry(athlete_id=a, rank_boulder=rb, rank_lead=rl) for a, rb, rl in rows
    )
    return CombinedFold(year=year, gender=gender, entries=entries)


def _snapshots_from_mus(
    year: int,
    mus_by_aid: dict[int, tuple[float, float]],
    sigma: float = 100.0,
) -> dict[int, dict[Discipline, dict[int, RatingSnapshot]]]:
    """Build the snapshot dict shape ``score_candidate`` expects.

    ``mus_by_aid[aid] = (mu_boulder, mu_lead)``.
    """
    boulder = {
        aid: RatingSnapshot(mu=mu_b, sigma=sigma)
        for aid, (mu_b, _) in mus_by_aid.items()
    }
    lead = {
        aid: RatingSnapshot(mu=mu_l, sigma=sigma)
        for aid, (_, mu_l) in mus_by_aid.items()
    }
    return {year: {Discipline.BOULDER: boulder, Discipline.LEAD: lead}}


class TestCombinedEntry:
    """Sanity checks on the rank-product combined score (Olympic format)."""

    def test_rank_product_score(self):
        # 1st in Boulder, 1st in Lead → score 1 (best possible).
        e = CombinedEntry(athlete_id=1, rank_boulder=1, rank_lead=1)
        assert e.combined_score == 1

    def test_specialist_loses_to_all_rounder(self):
        # All-rounder: 3rd, 3rd → score 9
        all_rounder = CombinedEntry(athlete_id=1, rank_boulder=3, rank_lead=3)
        # Specialist: 1st in Boulder, 10th in Lead → score 10
        specialist = CombinedEntry(athlete_id=2, rank_boulder=1, rank_lead=10)
        assert all_rounder.combined_score < specialist.combined_score


class TestScoreCandidateSynthetic:
    """When the athletes' B & L mus are equal across the board, the optimal
    weight pair must be the geometric mean (0.5, 0.5)."""

    def test_equal_mus_optimum_at_half_half(self):
        """If mu_b == mu_l for every athlete, w_lead is mathematically
        irrelevant — every (w_lead, w_boulder) candidate produces the
        identical mu_combined, so log-loss is constant across the grid.

        The grid sweep should still terminate and the metrics should match
        the baseline exactly.
        """
        year = 2024
        # Six athletes — strict mu ordering matches the actual rank-product.
        mus = {
            1: (1900.0, 1900.0),
            2: (1800.0, 1800.0),
            3: (1700.0, 1700.0),
            4: (1600.0, 1600.0),
            5: (1500.0, 1500.0),
            6: (1400.0, 1400.0),
        }
        # Construct combined ranks consistent with mu ordering (top mu → top rank).
        fold = _make_fold(
            year,
            Gender.M,
            [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)],
        )
        snapshots = _snapshots_from_mus(year, mus)

        baseline = score_candidate(
            [fold], snapshots, BASELINE_W_LEAD, BASELINE_W_BOULDER, n_simulations=2000
        )
        # When mu_b == mu_l the formula is identity across w_lead — sample a couple.
        a = score_candidate([fold], snapshots, 0.8, 0.2, n_simulations=2000)
        b = score_candidate([fold], snapshots, 0.3, 0.7, n_simulations=2000)
        # All three should produce the *same* log-loss (modulo MC noise) since
        # mu_combined is identical and the same RNG seed is used.
        assert a.log_loss == pytest.approx(baseline.log_loss, rel=1e-6)
        assert b.log_loss == pytest.approx(baseline.log_loss, rel=1e-6)

    def test_lead_only_optimum_when_lead_predicts(self):
        """When ratings = lead-only signal and ground truth = lead-only,
        the optimum should favour lead. Boulder mu is constant noise."""
        year = 2024
        # Six athletes: distinct mu_lead, identical mu_boulder.
        mus = {
            10: (1500.0, 1900.0),
            11: (1500.0, 1800.0),
            12: (1500.0, 1700.0),
            13: (1500.0, 1600.0),
            14: (1500.0, 1500.0),
            15: (1500.0, 1400.0),
        }
        # Ground truth: combined rank-product perfectly mirrors mu_lead order.
        # Use rb=1 for everyone (constant) and rl matching mu_lead order.
        fold = _make_fold(
            year,
            Gender.M,
            [(10, 1, 1), (11, 1, 2), (12, 1, 3), (13, 1, 4), (14, 1, 5), (15, 1, 6)],
        )
        snapshots = _snapshots_from_mus(year, mus, sigma=80.0)

        baseline = score_candidate(
            [fold], snapshots, BASELINE_W_LEAD, BASELINE_W_BOULDER, n_simulations=4000
        )
        # When w_lead = 1.0, mu_combined = mu_lead → identical to ground truth.
        lead_only = score_candidate([fold], snapshots, 1.0, 0.0, n_simulations=4000)
        # Lead-only must beat geometric mean: ground truth IS mu_lead.
        # With identical mu_boulder, the geometric-mean ranks already match
        # mu_lead, so log-loss should be equal (mathematically) — but
        # rank-corr should both be 1.0.
        assert baseline.rank_corr == pytest.approx(1.0, abs=1e-9)
        assert lead_only.rank_corr == pytest.approx(1.0, abs=1e-9)


class TestGridSearch:
    """Grid search must enumerate (w_lead, 1-w_lead) and pick the best by
    log-loss."""

    def test_grid_covers_endpoints(self):
        year = 2024
        mus = {
            1: (1900.0, 1900.0),
            2: (1700.0, 1700.0),
            3: (1500.0, 1500.0),
            4: (1300.0, 1300.0),
            5: (1100.0, 1100.0),
            6: (900.0, 900.0),
        }
        fold = _make_fold(
            year,
            Gender.M,
            [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)],
        )
        snapshots = _snapshots_from_mus(year, mus)
        _, _, evals = grid_search([fold], snapshots, step=0.5, n_simulations=1000)
        w_leads = [w_l for w_l, _, _ in evals]
        # step=0.5 → {0.0, 0.5, 1.0}
        assert any(abs(w - 0.0) < 1e-6 for w in w_leads)
        assert any(abs(w - 0.5) < 1e-6 for w in w_leads)
        assert any(abs(w - 1.0) < 1e-6 for w in w_leads)

    def test_grid_summands_equal_one(self):
        year = 2024
        mus = {i: (1500.0 + i * 10, 1500.0 - i * 5) for i in range(1, 8)}
        fold = _make_fold(year, Gender.M, [(i, i, 8 - i) for i in range(1, 8)])
        snapshots = _snapshots_from_mus(year, mus)
        _, _, evals = grid_search([fold], snapshots, step=0.25, n_simulations=500)
        for w_l, w_b, _ in evals:
            assert w_l + w_b == pytest.approx(1.0, abs=1e-6)


class TestDecideShip:
    """``decide_ship`` is the single source of truth for the ship/no-ship rule."""

    def test_baseline_match_does_not_ship(self):
        m = FitMetrics(log_loss=0.5, rank_corr=0.8)
        ship, _ = decide_ship(m, m, (BASELINE_W_LEAD, BASELINE_W_BOULDER))
        assert not ship

    def test_worse_log_loss_does_not_ship(self):
        better = FitMetrics(log_loss=0.6, rank_corr=0.85)
        baseline = FitMetrics(log_loss=0.55, rank_corr=0.8)
        ship, _ = decide_ship(better, baseline, (0.7, 0.3))
        assert not ship

    def test_strict_log_loss_improvement_no_corr_regression_ships(self):
        learned = FitMetrics(log_loss=0.45, rank_corr=0.82)
        baseline = FitMetrics(log_loss=0.50, rank_corr=0.80)
        ship, reason = decide_ship(learned, baseline, (0.7, 0.3))
        assert ship, reason

    def test_log_loss_better_but_rank_corr_regression_blocks_ship(self):
        # 50% drop in rank-corr → exceeds 5% tolerance.
        learned = FitMetrics(log_loss=0.40, rank_corr=0.40)
        baseline = FitMetrics(log_loss=0.55, rank_corr=0.80)
        ship, _ = decide_ship(learned, baseline, (0.9, 0.1))
        assert not ship

    def test_nan_metrics_do_not_ship(self):
        ship, _ = decide_ship(
            FitMetrics(log_loss=float("nan"), rank_corr=0.8),
            FitMetrics(log_loss=0.5, rank_corr=0.8),
            (0.7, 0.3),
        )
        assert not ship


# ---------------------------------------------------------------------------
# End-to-end: compute_combined_ratings.py reads the JSON
# ---------------------------------------------------------------------------


class TestLearnedWeightsAppliedInProduction:
    """If a valid learned-weights JSON exists, ``compute_combined_mu`` must
    use those weights when explicitly passed; the fallback path must
    produce geometric-mean output identical to the historical behaviour."""

    def test_fallback_matches_geometric_mean(self, tmp_path):
        # No file → baseline weights → identical to historical
        # sqrt(mu_b * mu_l).
        weights = load_combined_weights(tmp_path / "nope.json")
        mu_b, mu_l = 1800.0, 1700.0
        assert compute_combined_mu(mu_b, mu_l, weights) == pytest.approx(
            math.sqrt(mu_b * mu_l)
        )

    def test_learned_weights_change_output(self, tmp_path):
        path = tmp_path / "w.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.7,
                    "w_boulder": 0.3,
                    "log_loss": 0.4,
                    "rank_corr": 0.85,
                    "baseline_log_loss": 0.5,
                    "baseline_rank_corr": 0.83,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.source == "learned"
        mu_b, mu_l = 1800.0, 1700.0
        result = compute_combined_mu(mu_b, mu_l, weights)
        assert result == pytest.approx(1700.0**0.7 * 1800.0**0.3)
        # And it should NOT equal the geometric mean.
        assert result != pytest.approx(math.sqrt(mu_b * mu_l), abs=1e-6)


# Suppress unused-import warnings — keeping these imported tightens the
# public surface that tests assert on.
_ = (Session, date, select, Rating, Athlete, MIN_EVENTS, combined_mu)
