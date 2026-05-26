from __future__ import annotations

import enum
from datetime import date
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    contributing_pairs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    athlete: Mapped[Athlete] = relationship(back_populates="rating_history")
    event: Mapped[Event] = relationship(back_populates="rating_history")

    __table_args__ = (
        UniqueConstraint(
            "athlete_id", "round_id", name="uq_rating_history_athlete_round"
        ),
    )
