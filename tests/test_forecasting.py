"""Tests for the forecast snapshot engine (``engine/forecasting.py``)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session

from climbing_elo.engine.forecasting import snapshot_forecast
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventForecast,
    EventTier,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)


def _add_qualification_results(
    db_session: Session, event: Event, athletes: list[Athlete]
) -> Round:
    """Seed a qualification round with one Result per athlete (non-DNS)."""
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.QUALIFICATION,
        athlete_count=len(athletes),
        gender=Gender.M,
    )
    db_session.add(rnd)
    db_session.flush()
    for i, athlete in enumerate(athletes):
        db_session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athlete.id,
                rank=i + 1,
                raw_score=str(40 - i),
                score_normalized=float(40 - i),
                dns=False,
                dnf=False,
            )
        )
    db_session.flush()
    return rnd


class TestSnapshotLive:
    def test_writes_one_row_per_athlete(
        self,
        db_session: Session,
        sample_event: Event,
        eight_athletes_with_ratings: list[Athlete],
    ):
        _add_qualification_results(
            db_session, sample_event, eight_athletes_with_ratings
        )
        rows = snapshot_forecast(
            db_session,
            event_id=sample_event.id,
            gender=Gender.M,
            n_simulations=200,
            rng_seed=42,
        )
        db_session.commit()

        assert len(rows) == 8
        persisted = (
            db_session.query(EventForecast)
            .filter_by(event_id=sample_event.id, gender=Gender.M)
            .all()
        )
        assert len(persisted) == 8
        # Roster source for confirmed-qual case.
        assert {r.roster_source for r in persisted} == {"confirmed"}
        # engine_version is non-empty.
        assert all(r.engine_version for r in persisted)
        # Snapshot mu/sigma echo the Rating rows (mu=1750..1540, sigma=100).
        expected_mus = {1750.0, 1700.0, 1680.0, 1650.0, 1620.0, 1600.0, 1570.0, 1540.0}
        assert {r.mu_at_forecast for r in persisted} == expected_mus
        assert all(r.sigma_at_forecast == 100.0 for r in persisted)

    def test_is_idempotent(
        self,
        db_session: Session,
        sample_event: Event,
        eight_athletes_with_ratings: list[Athlete],
    ):
        _add_qualification_results(
            db_session, sample_event, eight_athletes_with_ratings
        )
        snapshot_forecast(
            db_session,
            event_id=sample_event.id,
            gender=Gender.M,
            n_simulations=100,
            rng_seed=1,
        )
        db_session.commit()
        snapshot_forecast(
            db_session,
            event_id=sample_event.id,
            gender=Gender.M,
            n_simulations=100,
            rng_seed=2,
        )
        db_session.commit()

        persisted = (
            db_session.query(EventForecast)
            .filter_by(event_id=sample_event.id, gender=Gender.M)
            .all()
        )
        # Exactly N rows — the unique constraint upsert path overwrote rather
        # than duplicated.
        assert len(persisted) == 8

    def test_monotonicity(
        self,
        db_session: Session,
        sample_event: Event,
        eight_athletes_with_ratings: list[Athlete],
    ):
        _add_qualification_results(
            db_session, sample_event, eight_athletes_with_ratings
        )
        rows = snapshot_forecast(
            db_session,
            event_id=sample_event.id,
            gender=Gender.M,
            n_simulations=500,
            rng_seed=7,
        )
        db_session.commit()

        # By construction inside snapshot_forecast we cap each downstream
        # stage at prob_qualify; sim outputs are themselves nested correctly.
        for row in rows:
            assert 0.0 <= row.prob_win <= row.prob_podium <= row.prob_reach_final
            assert row.prob_reach_final <= row.prob_reach_semi
            assert row.prob_reach_semi <= row.prob_qualify <= 1.0


class TestSnapshotBackfill:
    def test_reconstructs_ratings_from_history(
        self,
        db_session: Session,
        eight_athletes: list[Athlete],
    ):
        # Build a historical event (the source of the RatingHistory row).
        prior_event = Event(
            name="Prior Event",
            tier=EventTier.WORLD_CUP,
            season=2023,
            start_date=date(2023, 6, 1),
            discipline=Discipline.LEAD,
        )
        db_session.add(prior_event)
        db_session.flush()
        prior_round = Round(
            event_id=prior_event.id,
            round_type=RoundType.FINAL,
            athlete_count=1,
            gender=Gender.M,
        )
        db_session.add(prior_round)
        db_session.flush()

        target_athlete = eight_athletes[0]

        # Seed a RatingHistory row anchoring mu_after=1800 for one athlete.
        db_session.add(
            RatingHistory(
                athlete_id=target_athlete.id,
                event_id=prior_event.id,
                round_id=prior_round.id,
                mu_before=1700.0,
                mu_after=1800.0,
                sigma_before=120.0,
                sigma_after=110.0,
                kind="pair",
            )
        )
        # The other athletes get no history — they should fall back to the
        # defaults (μ=1500, σ=350).
        db_session.flush()

        # Target event: backfill snapshot uses competitors with actual results.
        target_event = Event(
            name="Target Event",
            tier=EventTier.WORLD_CUP,
            season=2024,
            start_date=date(2024, 6, 1),
            discipline=Discipline.LEAD,
        )
        db_session.add(target_event)
        db_session.flush()
        target_round = Round(
            event_id=target_event.id,
            round_type=RoundType.QUALIFICATION,
            athlete_count=len(eight_athletes),
            gender=Gender.M,
        )
        db_session.add(target_round)
        db_session.flush()
        for i, athlete in enumerate(eight_athletes):
            db_session.add(
                Result(
                    round_id=target_round.id,
                    athlete_id=athlete.id,
                    rank=i + 1,
                    score_normalized=float(40 - i),
                    dns=False,
                )
            )
        db_session.flush()

        # as_of_date strictly after prior_event.start_date.
        as_of = date(2024, 1, 1)
        rows = snapshot_forecast(
            db_session,
            event_id=target_event.id,
            gender=Gender.M,
            is_backfill=True,
            as_of_date=as_of,
            n_simulations=200,
            rng_seed=11,
        )
        db_session.commit()

        assert len(rows) == 8
        target_row = next(r for r in rows if r.athlete_id == target_athlete.id)
        assert target_row.mu_at_forecast == 1800.0
        assert target_row.sigma_at_forecast == 110.0
        assert target_row.roster_source == "backfill"
        # Other athletes use the default seed.
        for other in eight_athletes[1:]:
            row = next(r for r in rows if r.athlete_id == other.id)
            assert row.mu_at_forecast == 1500.0
            assert row.sigma_at_forecast == 350.0


class TestSnapshotLikelyRoster:
    def test_falls_back_when_no_qualification_round(
        self,
        db_session: Session,
        sample_event: Event,
        eight_athletes_with_ratings: list[Athlete],
    ):
        # Sample_event has no qualification round at all. Set up a prior
        # World Cup event in the same season+discipline so
        # likely_competitors() can derive a roster.
        prior_event = Event(
            name="Earlier WC",
            tier=EventTier.WORLD_CUP,
            season=sample_event.season,
            start_date=date(2024, 4, 1),
            discipline=Discipline.LEAD,
        )
        db_session.add(prior_event)
        db_session.flush()
        prior_round = Round(
            event_id=prior_event.id,
            round_type=RoundType.QUALIFICATION,
            athlete_count=len(eight_athletes_with_ratings),
            gender=Gender.M,
        )
        db_session.add(prior_round)
        db_session.flush()
        # All eight athletes competed in the prior event ⇒ qualify as likely.
        for athlete in eight_athletes_with_ratings:
            db_session.add(
                Result(
                    round_id=prior_round.id,
                    athlete_id=athlete.id,
                    rank=1,
                    dns=False,
                )
            )
        db_session.flush()

        rows = snapshot_forecast(
            db_session,
            event_id=sample_event.id,
            gender=Gender.M,
            n_simulations=150,
            rng_seed=3,
        )
        db_session.commit()

        assert len(rows) == 8
        assert {r.roster_source for r in rows} == {"likely"}

    def test_returns_empty_when_no_roster_resolvable(
        self,
        db_session: Session,
        sample_event: Event,
    ):
        # No qualification round, no prior events ⇒ no roster.
        rows = snapshot_forecast(
            db_session,
            event_id=sample_event.id,
            gender=Gender.M,
            n_simulations=10,
            rng_seed=0,
        )
        assert rows == []


class TestSnapshotValidation:
    def test_backfill_requires_as_of_date(
        self,
        db_session: Session,
        sample_event: Event,
    ):
        with pytest.raises(ValueError, match="as_of_date"):
            snapshot_forecast(
                db_session,
                event_id=sample_event.id,
                gender=Gender.M,
                is_backfill=True,
            )

    def test_unknown_event_id_raises(self, db_session: Session):
        with pytest.raises(ValueError, match="not found"):
            snapshot_forecast(
                db_session,
                event_id=999_999,
                gender=Gender.M,
            )


def test_engine_version_is_stamped(
    db_session: Session,
    sample_event: Event,
    eight_athletes_with_ratings: list[Athlete],
):
    _add_qualification_results(db_session, sample_event, eight_athletes_with_ratings)
    rows = snapshot_forecast(
        db_session,
        event_id=sample_event.id,
        gender=Gender.M,
        n_simulations=50,
        rng_seed=0,
    )
    db_session.commit()
    # All rows in one call share the same engine version.
    assert len({r.engine_version for r in rows}) == 1
    # Format: ``<12-hex-config>-<short-sha-or-unknown>``.
    version = rows[0].engine_version
    parts = version.split("-")
    assert len(parts) >= 2
    assert len(parts[0]) == 12
    # Touch unused fixture to satisfy linters that import Rating without use.
    assert db_session.query(Rating).count() == 8
