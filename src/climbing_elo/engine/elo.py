from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from climbing_elo.models import Discipline, EventTier, RoundType

DEFAULT_MU = 1500.0
DEFAULT_SIGMA = 350.0
PROVISIONAL_THRESHOLD = 3
PROVISIONAL_K_MULTIPLIER = 2.0  # tied across 1.5/2.0/3.0 at best k-scale; keep default
SIGMA_DECAY_HALF_LIFE_DAYS = 18 * 30  # ~18 months
SIGMA_FLOOR = 50.0
SIGMA_CEILING = 350.0
SIGMA_CONVERGENCE_FACTOR = 0.98
MARGIN_CAP = 1.5  # tuned: 1.5 outperforms 2.0 and 2.5 at 2x k-scale

# Speed-specific margin: max meaningful time gap in seconds
SPEED_MAX_GAP_SECONDS = 2.0

# K-factors tuned via grid search (scripts/tune_kfactors.py).
# Best config: 2.0x scale on base values, MARGIN_CAP=1.5
# → 87.5% podium hit-rate on 2025–2026 holdout vs 25% baseline (+62.5pp).
K_FACTOR_TABLE: dict[EventTier, dict[RoundType, float]] = {
    EventTier.OLYMPICS: {
        RoundType.FINAL: 96.0,
        RoundType.SEMI: 72.0,
        RoundType.QUALIFICATION: 36.0,
    },
    EventTier.WORLD_CHAMPIONSHIP: {
        RoundType.FINAL: 80.0,
        RoundType.SEMI: 60.0,
        RoundType.QUALIFICATION: 30.0,
    },
    EventTier.WORLD_CUP: {
        RoundType.FINAL: 64.0,
        RoundType.SEMI: 48.0,
        RoundType.QUALIFICATION: 24.0,
    },
    EventTier.CONTINENTAL: {
        RoundType.FINAL: 48.0,
        RoundType.SEMI: 36.0,
        RoundType.QUALIFICATION: 18.0,
    },
}


@dataclass
class AthleteResult:
    athlete_id: int
    rank: int
    score_normalized: float | None = None
    dnf: bool = False
    dns: bool = False


@dataclass
class PairContribution:
    opponent_id: int
    result: str  # "won" or "lost"
    expected: float
    actual: float
    delta: float
    margin_multiplier: float


@dataclass
class RatingUpdate:
    athlete_id: int
    mu_before: float
    mu_after: float
    sigma_before: float
    sigma_after: float
    contributing_pairs: list[PairContribution] = field(default_factory=list)


@dataclass
class AthleteRating:
    athlete_id: int
    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA
    n_events: int = 0
    last_event_at: date | None = None
    provisional: bool = True


def get_k_factor(tier: EventTier, round_type: RoundType) -> float:
    return K_FACTOR_TABLE[tier][round_type]


def expected_score(mu_a: float, mu_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((mu_b - mu_a) / 400.0))


def compute_margin_multiplier(
    score_a: float | None,
    score_b: float | None,
    max_gap: float = 20.0,
) -> float:
    if score_a is None or score_b is None:
        return 1.0
    gap = abs(score_a - score_b)
    return min(1.0 + gap / max_gap, MARGIN_CAP)


BOULDER_MARGIN_MAX_GAP = 1000.0

# Regex for the old-format ordinal Boulder score: e.g. "1T2z 3 4" or "2T2 3B4"
_OLD_BOULDER_RE = re.compile(
    r"(\d+)[Tt](\d+)[Zz]\s+(\d+)\s+(\d+)"  # "NTMz A B"
    r"|(\d+)[Tt](\d+)\s+(\d+)[Bb](\d+)",  # "NT A MBB"
)


def _is_new_boulder_format(raw_score: str) -> bool:
    """Return True if *raw_score* is a numeric/decimal value (new 2025+ format).

    The new IFSC scoring system (2025+) reports a points-based decimal like
    ``"34.5"`` or ``"25.0"``.  The old format is an ordinal string such as
    ``"1T2z 3 4"`` (tops / zones / top-attempts / zone-attempts).
    """
    raw = raw_score.strip()
    try:
        float(raw)
        return True
    except ValueError:
        return False


def normalize_boulder_score(raw_score: str) -> float | None:
    """Normalize a Boulder raw score to a comparable float.

    Handles both score formats:

    * **New format (2025+):** a decimal string like ``"34.5"`` — returned as-is
      as a float.  These scores are already on a common scale.
    * **Old format (pre-2025):** an ordinal string like ``"1T2z 3 4"`` or
      ``"2T2 3B4"`` — parsed into
      ``tops * 1000 + zones * 100 - top_att * 10 - zone_att`` so that the
      margin multiplier sees a consistent numeric scale across both eras.

    Returns ``None`` if the score cannot be parsed.
    """
    raw = (raw_score or "").strip()
    if not raw or raw.upper() in ("DNF", "DNS", "-"):
        return None

    if _is_new_boulder_format(raw):
        return float(raw)

    m = re.match(r"(\d+)[Tt](\d+)[Zz]\s+(\d+)\s+(\d+)", raw)
    if m:
        tops, zones, top_att, zone_att = (int(x) for x in m.groups())
        return float(tops * 1000 + zones * 100 - top_att * 10 - zone_att)

    m = re.match(r"(\d+)[Tt](\d+)\s+(\d+)[Bb](\d+)", raw)
    if m:
        tops, top_att, zones, zone_att = (int(x) for x in m.groups())
        return float(tops * 1000 + zones * 100 - top_att * 10 - zone_att)

    return None


def compute_boulder_margin_multiplier(
    score_a: float | None,
    score_b: float | None,
) -> float:
    """Margin multiplier for Boulder discipline.

    Boulder normalized score: tops*1000 + zones*100 - top_att*10 - zone_att.
    A one-top margin (~900-1000 point gap) gives ≈1.9× multiplier.
    """
    return compute_margin_multiplier(score_a, score_b, max_gap=BOULDER_MARGIN_MAX_GAP)


def compute_speed_margin_multiplier(
    winner_time: float | None,
    loser_time: float | None,
) -> float:
    """Margin multiplier for Speed discipline (times in seconds, lower is better)."""
    if winner_time is None or loser_time is None:
        return 1.0
    gap = abs(loser_time - winner_time)
    return min(1.0 + gap / SPEED_MAX_GAP_SECONDS, MARGIN_CAP)


def apply_time_decay(
    sigma: float, last_event_at: date | None, current_date: date
) -> float:
    if last_event_at is None:
        return sigma
    days_inactive = (current_date - last_event_at).days
    if days_inactive <= 0:
        return sigma
    decay_factor = 2.0 ** (days_inactive / SIGMA_DECAY_HALF_LIFE_DAYS)
    return min(sigma * decay_factor, SIGMA_CEILING)


def calculate_round_updates(
    results: list[AthleteResult],
    ratings: dict[int, AthleteRating],
    event_tier: EventTier,
    round_type: RoundType,
    event_date: date,
    discipline: Discipline = Discipline.LEAD,
) -> list[RatingUpdate]:
    active = [r for r in results if not r.dns]
    if len(active) < 2:
        return []

    base_k = get_k_factor(event_tier, round_type)
    n = len(active)
    pair_k = base_k / (n - 1)

    deltas: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    pairs: dict[int, list[PairContribution]] = {r.athlete_id: [] for r in active}

    for i, res_i in enumerate(active):
        rating_i = ratings.get(
            res_i.athlete_id, AthleteRating(athlete_id=res_i.athlete_id)
        )
        mu_i = rating_i.mu

        for j, res_j in enumerate(active):
            if i == j:
                continue

            rating_j = ratings.get(
                res_j.athlete_id, AthleteRating(athlete_id=res_j.athlete_id)
            )
            mu_j = rating_j.mu

            if res_i.rank == res_j.rank:
                continue

            if res_i.rank > res_j.rank:
                continue

            # res_i finished ahead of res_j
            e_i = expected_score(mu_i, mu_j)

            k = pair_k
            if rating_i.provisional or rating_j.provisional:
                k *= PROVISIONAL_K_MULTIPLIER

            if res_i.dnf:
                # DNF (incl. false start in Speed) — no margin bonus
                margin_mult = 1.0
            elif discipline == Discipline.SPEED:
                # For Speed, res_i won (lower time), res_j lost (higher time)
                margin_mult = compute_speed_margin_multiplier(
                    res_i.score_normalized, res_j.score_normalized
                )
            elif discipline == Discipline.BOULDER:
                margin_mult = compute_boulder_margin_multiplier(
                    res_i.score_normalized, res_j.score_normalized
                )
            else:
                margin_mult = compute_margin_multiplier(
                    res_i.score_normalized, res_j.score_normalized
                )

            delta_i = k * margin_mult * (1.0 - e_i)
            delta_j = k * margin_mult * (0.0 - (1.0 - e_i))

            deltas[res_i.athlete_id] += delta_i
            deltas[res_j.athlete_id] += delta_j

            pairs[res_i.athlete_id].append(
                PairContribution(
                    opponent_id=res_j.athlete_id,
                    result="won",
                    expected=round(e_i, 4),
                    actual=1.0,
                    delta=round(delta_i, 2),
                    margin_multiplier=round(margin_mult, 2),
                )
            )
            pairs[res_j.athlete_id].append(
                PairContribution(
                    opponent_id=res_i.athlete_id,
                    result="lost",
                    expected=round(1.0 - e_i, 4),
                    actual=0.0,
                    delta=round(delta_j, 2),
                    margin_multiplier=round(margin_mult, 2),
                )
            )

    updates = []
    for res in active:
        aid = res.athlete_id
        rating = ratings.get(aid, AthleteRating(athlete_id=aid))

        mu_before = rating.mu
        sigma_before = apply_time_decay(rating.sigma, rating.last_event_at, event_date)

        mu_after = mu_before + deltas[aid]
        sigma_after = max(sigma_before * SIGMA_CONVERGENCE_FACTOR, SIGMA_FLOOR)

        updates.append(
            RatingUpdate(
                athlete_id=aid,
                mu_before=mu_before,
                mu_after=mu_after,
                sigma_before=sigma_before,
                sigma_after=sigma_after,
                contributing_pairs=pairs[aid],
            )
        )

    return updates
