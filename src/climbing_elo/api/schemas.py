"""Pydantic response schemas for the v1 REST API."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict

from climbing_elo.models import Discipline, Gender


# ---------------------------------------------------------------------------
# Shared / primitive schemas
# ---------------------------------------------------------------------------

class DisciplineInfo(BaseModel):
    code: str
    name: str
    description: str


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    athlete_id: int
    name: str
    nationality: Optional[str]
    gender: str
    mu: float
    sigma: float
    n_events: int
    provisional: bool
    last_event_at: Optional[date]


class LeaderboardResponse(BaseModel):
    discipline: str
    gender: str
    limit: int
    offset: int
    total: int
    items: list[LeaderboardEntry]


# ---------------------------------------------------------------------------
# Athlete
# ---------------------------------------------------------------------------

class AthleteRating(BaseModel):
    discipline: str
    mu: float
    sigma: float
    n_events: int
    provisional: bool
    last_event_at: Optional[date]


class RecentEvent(BaseModel):
    event_id: int
    event_name: str
    season: int
    discipline: str
    mu_before: float
    mu_after: float
    delta: float


class AthleteDetail(BaseModel):
    id: int
    name: str
    nationality: Optional[str]
    gender: str
    year_of_birth: Optional[int]
    ratings: list[AthleteRating]
    recent_events: list[RecentEvent]


# ---------------------------------------------------------------------------
# Athlete rating history (for charts)
# ---------------------------------------------------------------------------

class HistoryPoint(BaseModel):
    event_id: int
    event_name: str
    event_date: date
    season: int
    mu_after: float
    sigma_after: float
    mu_before: float
    sigma_before: float
    delta: float


class AthleteHistoryResponse(BaseModel):
    athlete_id: int
    athlete_name: str
    discipline: str
    points: list[HistoryPoint]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventSummary(BaseModel):
    id: int
    name: str
    tier: str
    country: Optional[str]
    season: int
    start_date: date
    discipline: str


class EventsResponse(BaseModel):
    limit: int
    offset: int
    total: int
    items: list[EventSummary]


class ResultRow(BaseModel):
    athlete_id: int
    athlete_name: str
    nationality: Optional[str]
    rank: Optional[int]
    raw_score: Optional[str]
    dnf: bool
    dns: bool
    mu_before: Optional[float]
    mu_after: Optional[float]
    delta: Optional[float]


class RoundDetail(BaseModel):
    id: int
    round_type: str
    gender: str
    athlete_count: int
    results: list[ResultRow]


class EventDetail(BaseModel):
    id: int
    name: str
    tier: str
    country: Optional[str]
    season: int
    start_date: date
    discipline: str
    rounds: list[RoundDetail]
