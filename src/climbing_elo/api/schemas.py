"""Pydantic response schemas for the v1 REST API."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class AthleteSearchResult(BaseModel):
    """One match from GET /api/v1/athletes (name search / typeahead).

    ``mu`` is the athlete's rating in the requested discipline, or their highest
    rating across all disciplines when no discipline is supplied. ``None`` when
    the athlete has no rating at all (or none for the requested discipline).
    """

    id: int
    name: str
    nationality: Optional[str]
    gender: str
    mu: Optional[float]


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


# ---------------------------------------------------------------------------
# Combined (Boulder+Lead) ratings
# ---------------------------------------------------------------------------


class CombinedLeaderboardEntry(LeaderboardEntry):
    """Leaderboard entry for the BOULDER_LEAD combined discipline.

    Extends LeaderboardEntry with the per-discipline breakdown so API consumers
    can see the individual boulder and lead ratings alongside the combined score.
    """

    mu_boulder: float
    mu_lead: float
    sigma_boulder: float
    sigma_lead: float


class CombinedLeaderboardResponse(BaseModel):
    gender: str
    limit: int
    offset: int
    total: int
    items: list[CombinedLeaderboardEntry]


class AthleteCombined(BaseModel):
    """Single-athlete combined rating with per-discipline breakdown."""

    athlete_id: int
    name: str
    nationality: Optional[str]
    gender: str
    mu_combined: float
    sigma_combined: float
    n_events_combined: int
    provisional_combined: bool
    mu_boulder: float
    mu_lead: float
    sigma_boulder: float
    sigma_lead: float
    last_event_at: Optional[date]


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


class ProjectionRequest(BaseModel):
    """Request body for POST /api/v1/projections."""

    discipline: str = Field(
        ...,
        description="Discipline code: lead, boulder, speed, boulder_lead / combined",
    )
    athlete_ids: List[int] = Field(
        ...,
        min_length=2,
        max_length=64,
        description="List of athlete IDs to project (2–64, no duplicates)",
    )


class ProjectionEntry(BaseModel):
    """Per-athlete projection result."""

    athlete_id: int
    name: str
    mu: float
    sigma: float
    win: float
    podium: float
    top_8: float
    expected_rank: float


class ProjectionResponse(BaseModel):
    """Response for POST /api/v1/projections."""

    discipline: str
    n_athletes: int
    n_simulations: int
    items: list[ProjectionEntry]


# ---------------------------------------------------------------------------
# Upcoming predictions
# ---------------------------------------------------------------------------


class PredictedAthlete(BaseModel):
    """One athlete in a predicted top-3."""

    athlete_id: int
    name: str
    win: float
    podium: float
    expected_rank: float


class GenderPrediction(BaseModel):
    """Predictions for one gender within an upcoming event."""

    gender: str
    total_athletes: int
    top_3: list[PredictedAthlete]


class UpcomingPredictionEntry(BaseModel):
    """One upcoming event with predicted top-3 per gender."""

    event_id: int
    event_name: str
    discipline: str
    season: int
    start_date: date
    tier: str
    country: Optional[str]
    has_registered_athletes: bool
    from_likely_roster: bool
    genders: list[GenderPrediction]


class UpcomingPredictionsResponse(BaseModel):
    """Response for GET /api/v1/predictions/upcoming."""

    discipline: Optional[str]
    season: Optional[int]
    total: int
    items: list[UpcomingPredictionEntry]
