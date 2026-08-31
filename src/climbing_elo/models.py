from __future__ import annotations

import enum
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Tz-aware UTC ``now()`` for SQLAlchemy ``default=`` factories.

    ``datetime.utcnow`` is deprecated in 3.12 and returns a naive datetime.
    Forecast snapshot timestamps need to round-trip to Postgres as UTC.
    """
    return datetime.now(timezone.utc)


def _enum_values(cls):
    """Use enum member values (not names) for Postgres native enum storage."""
    return [e.value for e in cls]


class Base(DeclarativeBase):
    pass


class Gender(str, enum.Enum):
    M = "M"
    F = "F"


class EventTier(str, enum.Enum):
    OLYMPICS = "olympics"
    WORLD_CHAMPIONSHIP = "world_championship"
    WORLD_CUP = "world_cup"
    CONTINENTAL = "continental"


class RoundType(str, enum.Enum):
    QUALIFICATION = "qualification"
    SEMI = "semi"
    FINAL = "final"


class Discipline(str, enum.Enum):
    LEAD = "L"
    BOULDER = "B"
    SPEED = "S"
    BOULDER_LEAD = "BL"


class Athlete(Base):
    __tablename__ = "athletes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    year_of_birth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    nationality: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    gender: Mapped[Gender] = mapped_column(
        Enum(Gender, values_callable=_enum_values), nullable=False
    )
    # Issue #86 — rich climber profile page.
    # Photos are hot-linked from ifsc.results.info (Option A). Body metrics are
    # populated only when IFSC publishes them; most rows remain NULL.
    # See ``scripts/scrape_athlete_profiles.py`` for the (manual) refresh job.
    photo_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    height_cm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    wingspan_cm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Issue #91 — dual-view leaderboard. Manual override; when non-NULL,
    # ``engine.activity.is_likely_retired_simple`` returns True regardless
    # of ``last_event_at``. NULL by default; populated case-by-case.
    retired_at: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    results: Mapped[list[Result]] = relationship(back_populates="athlete")
    ratings: Mapped[list[Rating]] = relationship(back_populates="athlete")
    rating_history: Mapped[list[RatingHistory]] = relationship(back_populates="athlete")

    __table_args__ = (
        UniqueConstraint("name", "gender", name="uq_athlete_name_gender"),
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    tier: Mapped[EventTier] = mapped_column(
        Enum(EventTier, values_callable=_enum_values), nullable=False
    )
    country: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    discipline: Mapped[Discipline] = mapped_column(
        Enum(Discipline, values_callable=_enum_values), nullable=False
    )
    # Optional YouTube live-stream URL (youtube.com or youtu.be). Populated
    # manually per upcoming event until an automated source is found.
    # Issue #23.
    livestream_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    rounds: Mapped[list[Round]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )
    rating_history: Mapped[list[RatingHistory]] = relationship(back_populates="event")

    __table_args__ = (
        UniqueConstraint(
            "name", "season", "discipline", name="uq_event_name_season_discipline"
        ),
    )


class Round(Base):
    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False)
    round_type: Mapped[RoundType] = mapped_column(
        Enum(RoundType, values_callable=_enum_values), nullable=False
    )
    athlete_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gender: Mapped[Gender] = mapped_column(
        Enum(Gender, values_callable=_enum_values), nullable=False
    )

    event: Mapped[Event] = relationship(back_populates="rounds")
    results: Mapped[list[Result]] = relationship(
        back_populates="round", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "event_id", "round_type", "gender", name="uq_round_event_type_gender"
        ),
    )


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id"), nullable=False)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), nullable=False)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    raw_score: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    score_normalized: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dnf: Mapped[bool] = mapped_column(Boolean, default=False)
    dns: Mapped[bool] = mapped_column(Boolean, default=False)

    round: Mapped[Round] = relationship(back_populates="results")
    athlete: Mapped[Athlete] = relationship(back_populates="results")

    __table_args__ = (
        UniqueConstraint("round_id", "athlete_id", name="uq_result_round_athlete"),
    )


class Rating(Base):
    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), nullable=False)
    discipline: Mapped[Discipline] = mapped_column(
        Enum(Discipline, values_callable=_enum_values),
        nullable=False,
        default=Discipline.LEAD,
    )
    mu: Mapped[float] = mapped_column(Float, nullable=False, default=1500.0)
    sigma: Mapped[float] = mapped_column(Float, nullable=False, default=350.0)
    n_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_event_at: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    provisional: Mapped[bool] = mapped_column(Boolean, default=True)

    athlete: Mapped[Athlete] = relationship(back_populates="ratings")

    __table_args__ = (
        UniqueConstraint(
            "athlete_id", "discipline", name="uq_rating_athlete_discipline"
        ),
    )


class RatingHistory(Base):
    __tablename__ = "rating_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    athlete_id: Mapped[int] = mapped_column(ForeignKey("athletes.id"), nullable=False)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id"), nullable=False)
    mu_before: Mapped[float] = mapped_column(Float, nullable=False)
    mu_after: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_before: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_after: Mapped[float] = mapped_column(Float, nullable=False)
    # Deferred: this JSON blob averages ~1.1KB and is 93% of the row's bytes,
    # but only the /breakdown page and the athlete-profile opponents preview
    # read it. Deferring keeps it out of every other RatingHistory load (the
    # ticker, history charts, event pages) — the dominant Supabase egress
    # driver before 2026-08. Query sites that need it use undefer(); Core-level
    # selects (snapshots/exports) are unaffected by ORM deferral.
    contributing_pairs: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, deferred=True
    )
    # Issue #90 — Tournament Participation Bonus (Gap 1 from #88).
    # Discriminator: 'pair' = standard pairwise round update (legacy
    # behaviour). 'tpb' = synthetic, event-level tier-weighted bonus whose
    # round_id points at the event's FINAL round. The unique constraint
    # includes ``kind`` so a pair row and a tpb row can coexist for the same
    # (athlete, final round) without colliding.
    kind: Mapped[str] = mapped_column(String, nullable=False, default="pair")

    athlete: Mapped[Athlete] = relationship(back_populates="rating_history")
    event: Mapped[Event] = relationship(back_populates="rating_history")

    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "round_id",
            "kind",
            name="uq_rating_history_athlete_round_kind",
        ),
        CheckConstraint("kind IN ('pair', 'tpb')", name="rating_history_kind_check"),
    )


# ---------------------------------------------------------------------------
# Persistent forecasts (joyful-swinging-map plan)
# ---------------------------------------------------------------------------
#
# Two additive tables that freeze a per-event, per-athlete Monte Carlo
# prediction snapshot (``EventForecast``) and the post-event accuracy scoring
# (``EventForecastScore``) against actual results. Mirrors the
# ``uq_rating_history_athlete_round_kind`` idempotency pattern above:
# upserts are keyed on the unique constraint so daily re-snapshots in the
# 7-day pre-event window overwrite cleanly, and a separate ``is_backfill``
# row co-exists for retro-replay rows.
#
# ``engine_version`` is a stable identifier produced by
# :func:`climbing_elo.engine.elo.engine_version_tag` — sha256-12 of the
# ``EloConfig`` field tuple plus the short git SHA. The version is part of
# the unique constraint, so a re-snapshot at a new engine version inserts a
# fresh row alongside the prior-version one rather than overwriting it.


class EventForecast(Base):
    """Frozen Monte Carlo forecast for one (event, gender, athlete, is_backfill).

    Daily snapshot job upserts on the unique constraint so the most-recent
    pre-start row is the canonical "locked" forecast for scoring. A separate
    row with ``is_backfill=True`` may co-exist for retro-replay rows.
    """

    __tablename__ = "event_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id"), nullable=False, index=True
    )
    gender: Mapped[Gender] = mapped_column(
        Enum(Gender, values_callable=_enum_values), nullable=False
    )
    athlete_id: Mapped[int] = mapped_column(
        ForeignKey("athletes.id"), nullable=False, index=True
    )

    # Cumulative stage probabilities (all in [0, 1]).
    # Monotone non-increasing for a single athlete:
    # prob_qualify >= prob_reach_semi >= prob_reach_final >= prob_podium >= prob_win
    prob_qualify: Mapped[float] = mapped_column(Float, nullable=False)
    prob_reach_semi: Mapped[float] = mapped_column(Float, nullable=False)
    prob_reach_final: Mapped[float] = mapped_column(Float, nullable=False)
    prob_podium: Mapped[float] = mapped_column(Float, nullable=False)
    prob_win: Mapped[float] = mapped_column(Float, nullable=False)
    expected_rank: Mapped[float] = mapped_column(Float, nullable=False)

    # Snapshot of the rating used as the sim input — kept for reproducibility
    # so we can re-run a forecast offline without consulting RatingHistory.
    mu_at_forecast: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_at_forecast: Mapped[float] = mapped_column(Float, nullable=False)
    n_simulations: Mapped[int] = mapped_column(Integer, nullable=False)

    # 'confirmed' = scraped registration list; 'likely' = engine.likely_roster
    # fallback; 'backfill' = retro-replay using actual competitors.
    roster_source: Mapped[str] = mapped_column(String, nullable=False)

    is_backfill: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Stable short identifier of the engine version that produced this row.
    # See ``climbing_elo.engine.elo.engine_version_tag``. Indexed because the
    # ``/model-performance`` aggregation pages filter on it.
    engine_version: Mapped[str] = mapped_column(String, nullable=False, index=True)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    event: Mapped[Event] = relationship()
    athlete: Mapped[Athlete] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "gender",
            "athlete_id",
            "is_backfill",
            "engine_version",
            name="uq_event_forecast_event_gender_athlete_backfill_version",
        ),
    )


class EventForecastScore(Base):
    """Post-event scoring of a frozen forecast against actual results.

    One row per (event, gender, is_backfill). Aggregates Brier / log-loss
    across the per-stage cumulative probabilities, plus top-K intersection
    and Spearman rank correlation of predicted vs actual finishing order.
    """

    __tablename__ = "event_forecast_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id"), nullable=False, index=True
    )
    gender: Mapped[Gender] = mapped_column(
        Enum(Gender, values_callable=_enum_values), nullable=False
    )
    is_backfill: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    engine_version: Mapped[str] = mapped_column(String, nullable=False)

    n_athletes: Mapped[int] = mapped_column(Integer, nullable=False)
    n_simulations: Mapped[int] = mapped_column(Integer, nullable=False)

    # Mean ``(p - y)^2`` across athletes for each cumulative stage.
    brier_semi: Mapped[float] = mapped_column(Float, nullable=False)
    brier_final: Mapped[float] = mapped_column(Float, nullable=False)
    brier_podium: Mapped[float] = mapped_column(Float, nullable=False)
    brier_win: Mapped[float] = mapped_column(Float, nullable=False)

    # Mean cross-entropy across athletes for each cumulative stage; clipped
    # at ε=1e-9 inside ``score_forecast`` to avoid ``log(0)``.
    logloss_semi: Mapped[float] = mapped_column(Float, nullable=False)
    logloss_final: Mapped[float] = mapped_column(Float, nullable=False)
    logloss_podium: Mapped[float] = mapped_column(Float, nullable=False)
    logloss_win: Mapped[float] = mapped_column(Float, nullable=False)

    # Size of predicted top-K ∩ actual top-K. 0–3 and 0–8 respectively.
    top3_intersection: Mapped[int] = mapped_column(Integer, nullable=False)
    top8_intersection: Mapped[int] = mapped_column(Integer, nullable=False)

    # Spearman rank correlation of predicted vs actual finishing rank.
    # Nullable because it is undefined for a 1-athlete field or for
    # all-tied predictions (zero variance).
    spearman_rank: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    event: Mapped[Event] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "gender",
            "is_backfill",
            "engine_version",
            name="uq_event_forecast_score_event_gender_backfill_version",
        ),
    )
