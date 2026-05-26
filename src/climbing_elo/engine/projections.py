"""Monte Carlo projections for upcoming/in-progress climbing events.

Given a set of athletes and their current ELO ratings (mu, sigma), this module
simulates thousands of hypothetical performances to estimate win and podium
probabilities.

Performance model: each athlete's performance in a single event is drawn from
N(mu, sigma). Higher score = better finish. Ties are broken randomly (extremely
rare with continuous draws).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_ATHLETES_PER_PROJECTION = 256
MAX_SIMULATIONS = 50_000
SIGMA_FLOOR = 1e-6


@dataclass
class AthleteProjectionInput:
    athlete_id: int
    mu: float
    sigma: float
    name: str = ""


def compute_podium_probabilities(
    athletes: list[AthleteProjectionInput],
    n_simulations: int = 10_000,
    rng_seed: int | None = None,
) -> dict[int, dict[str, float]]:
    """Monte Carlo simulation of finish probabilities.

    For each simulation:
      - Draw performance from N(mu, sigma) for each athlete.
      - Rank athletes by descending performance score.
      - Tally top-1, top-3, top-8, and cumulative rank.

    Args:
        athletes: List of athlete inputs with mu/sigma.
        n_simulations: Number of Monte Carlo iterations (default 10,000).
        rng_seed: Optional seed for reproducibility.

    Returns:
        Dict mapping athlete_id to:
            ``win``           — fraction of sims finishing 1st
            ``podium``        — fraction of sims finishing top-3
            ``top_8``         — fraction of sims finishing top-8
            ``expected_rank`` — mean finishing rank across all sims
    """
    if not athletes:
        return {}

    if len(athletes) > MAX_ATHLETES_PER_PROJECTION:
        raise ValueError(
            f"too many athletes ({len(athletes)}); max is {MAX_ATHLETES_PER_PROJECTION}"
        )
    n_simulations = max(1, min(int(n_simulations), MAX_SIMULATIONS))

    rng = np.random.default_rng(rng_seed)
    n = len(athletes)

    mus = np.array([a.mu for a in athletes], dtype=np.float64)
    sigmas = np.array([a.sigma for a in athletes], dtype=np.float64)
    # Guard against zero/negative sigma reaching rng.normal (would raise).
    sigmas = np.clip(sigmas, SIGMA_FLOOR, None)

    # Shape: (n_simulations, n_athletes)
    # Each row is one simulated event; higher score = better performance.
    performances = rng.normal(mus, sigmas, size=(n_simulations, n))

    # Rank within each simulation: rank 1 = highest performance.
    # argsort descending → indices of athletes from best to worst per sim.
    # ranks[sim, athlete_idx] = finishing rank (1-based)
    order = np.argsort(-performances, axis=1)  # (n_simulations, n)
    ranks = np.empty_like(order)
    # Scatter ranks: for each sim, assign rank 1..n in sorted order
    sim_indices = np.arange(n_simulations)[:, np.newaxis]
    ranks[sim_indices, order] = np.arange(1, n + 1)[np.newaxis, :]

    win_counts = (ranks == 1).sum(axis=0)      # shape (n,)
    podium_counts = (ranks <= 3).sum(axis=0)   # shape (n,)
    top8_counts = (ranks <= 8).sum(axis=0)     # shape (n,)
    mean_rank = ranks.mean(axis=0)             # shape (n,)

    results: dict[int, dict[str, float]] = {}
    for i, athlete in enumerate(athletes):
        results[athlete.athlete_id] = {
            "win": round(float(win_counts[i]) / n_simulations, 4),
            "podium": round(float(podium_counts[i]) / n_simulations, 4),
            "top_8": round(float(top8_counts[i]) / n_simulations, 4),
            "expected_rank": round(float(mean_rank[i]), 2),
        }

    return results


# ---------------------------------------------------------------------------
# Round-by-round progression simulation
# ---------------------------------------------------------------------------

@dataclass
class RoundConfig:
    """Configuration for a single event round.

    Attributes:
        round_type: Identifier string — "qualification", "semifinal", or "final".
        advance_count: How many athletes advance from this round to the next.
            For the last round this value is ignored (everyone remaining competes).
    """
    round_type: str  # "qualification", "semifinal", "final"
    advance_count: int  # how many advance to next round


@dataclass
class ProgressionResult:
    """Per-athlete output from :func:`simulate_event_progression`.

    Attributes:
        athlete_id: Numeric athlete identifier.
        name: Human-readable name.
        mu: The athlete's ELO mean rating.
        advance_probs: Fraction of simulations in which the athlete reached each
            round, keyed by ``round_type``.  The first round is always 1.0 (all
            athletes start there).
        final_podium_prob: Fraction of simulations where the athlete finished in
            the top 3 of the **final** round (win + 2nd + 3rd combined).
        final_win_prob: Fraction of simulations where the athlete finished 1st
            in the final round.
    """
    athlete_id: int
    name: str
    mu: float
    advance_probs: dict[str, float] = field(default_factory=dict)
    final_podium_prob: float = 0.0
    final_win_prob: float = 0.0


def simulate_event_progression(
    athletes: list[AthleteProjectionInput],
    rounds: list[RoundConfig],
    n_simulations: int = 10_000,
    rng_seed: int | None = None,
) -> list[ProgressionResult]:
    """Run Monte Carlo trials of a multi-round event format.

    Each trial works as follows:

    1. All athletes enter the first round.  A performance score is drawn from
       N(mu, sigma) for each athlete.
    2. The top ``advance_count`` athletes (by performance) advance to the next round.
    3. In subsequent rounds fresh performance scores are drawn for the advancing subset.
    4. After the final round, podium (top-3) and win (1st) tallies are recorded.

    Advancement probabilities are the fraction of simulations in which each
    athlete reached that round.  The first round probability is always 1.0.
    Final-round podium/win probabilities are the fraction of simulations in
    which the athlete earned that outcome **given that they reached the final**.

    Args:
        athletes: Athletes to simulate.  Must not exceed MAX_ATHLETES_PER_PROJECTION.
        rounds: Ordered list of RoundConfig objects from first to last.
            Must contain at least one entry.  For a single-round event, supply
            a list with one RoundConfig — ``advance_count`` is ignored in that case.
        n_simulations: Number of independent Monte Carlo trials.  Clamped to
            [1, MAX_SIMULATIONS].
        rng_seed: Optional seed for reproducibility.

    Returns:
        List of :class:`ProgressionResult` objects, one per input athlete, sorted
        by descending mu (highest-rated first).

    Raises:
        ValueError: If ``athletes`` is longer than MAX_ATHLETES_PER_PROJECTION or
            ``rounds`` is empty.
    """
    if not athletes:
        return []

    if not rounds:
        raise ValueError("rounds must contain at least one RoundConfig")

    if len(athletes) > MAX_ATHLETES_PER_PROJECTION:
        raise ValueError(
            f"too many athletes ({len(athletes)}); max is {MAX_ATHLETES_PER_PROJECTION}"
        )

    n_simulations = max(1, min(int(n_simulations), MAX_SIMULATIONS))
    rng = np.random.default_rng(rng_seed)

    n = len(athletes)
    mus = np.array([a.mu for a in athletes], dtype=np.float64)
    sigmas = np.clip(
        np.array([a.sigma for a in athletes], dtype=np.float64),
        SIGMA_FLOOR,
        None,
    )

    # advance_counts[i] → how many survive round i.  The last round has no
    # elimination, so we keep the field size for tallying only.
    advance_counts: list[int] = []
    for i, rc in enumerate(rounds):
        if i < len(rounds) - 1:
            # Clamp so we never try to advance more than actually entered.
            advance_counts.append(rc.advance_count)
        else:
            # Final round — everyone who reached it competes.
            advance_counts.append(n)  # placeholder, not used for filtering

    # Per-athlete tallies across all simulations.
    # reached[athlete_idx, round_idx] = number of sims in which athlete reached round
    reached = np.zeros((n, len(rounds)), dtype=np.int64)
    # Podium/win counts are only tallied for the final round.
    final_podium = np.zeros(n, dtype=np.int64)
    final_win = np.zeros(n, dtype=np.int64)

    # All athletes start in round 0.
    reached[:, 0] = n_simulations

    for sim in range(n_simulations):
        # active_indices: indices (into the full athletes array) of those currently competing
        active = np.arange(n, dtype=np.int64)

        for round_idx, rc in enumerate(rounds):
            n_active = len(active)
            if n_active == 0:
                break

            # Draw performances for the active athletes.
            perf = rng.normal(mus[active], sigmas[active])

            if round_idx < len(rounds) - 1:
                # Not the last round: take the top-K performers.
                k = min(rc.advance_count, n_active)
                # argsort descending — take first k positions
                sorted_local = np.argsort(-perf)[:k]
                active = active[sorted_local]
                # Record that these athletes reached the next round.
                if round_idx + 1 < len(rounds):
                    reached[active, round_idx + 1] += 1
            else:
                # Final round: tally podium outcomes.
                sorted_local = np.argsort(-perf)  # best to worst
                # Top-1
                final_win[active[sorted_local[0]]] += 1
                # Top-3
                podium_k = min(3, n_active)
                for pos in range(podium_k):
                    final_podium[active[sorted_local[pos]]] += 1

    # Build results.
    results: list[ProgressionResult] = []
    for i, athlete in enumerate(athletes):
        adv: dict[str, float] = {}
        for round_idx, rc in enumerate(rounds):
            adv[rc.round_type] = round(float(reached[i, round_idx]) / n_simulations, 4)

        results.append(ProgressionResult(
            athlete_id=athlete.athlete_id,
            name=athlete.name,
            mu=athlete.mu,
            advance_probs=adv,
            final_podium_prob=round(float(final_podium[i]) / n_simulations, 4),
            final_win_prob=round(float(final_win[i]) / n_simulations, 4),
        ))

    # Sort by descending mu (highest-rated first).
    results.sort(key=lambda r: r.mu, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Default event formats
# ---------------------------------------------------------------------------

# Import here to avoid circular imports with the models package — projections.py
# deliberately has no dependency on SQLAlchemy models.  We use a local import
# so callers that don't need default_event_format don't pay the import cost.

def default_event_format(tier: str) -> list[RoundConfig]:
    """Return the default round progression for an event tier.

    The tier argument should be the **string value** of an EventTier enum
    (e.g. ``"olympics"``, ``"world_cup"``).  This avoids a hard dependency on
    the models module from within the engine layer.

    Overrides should be applied by the caller when event-specific metadata is
    available (e.g. the actual round structure stored in the DB).

    Args:
        tier: EventTier string value — one of ``"olympics"``,
            ``"world_championship"``, ``"world_cup"``, ``"continental"``.

    Returns:
        Ordered list of :class:`RoundConfig` from first to last round.

    Raises:
        ValueError: If ``tier`` is not a recognised EventTier string value.
    """
    tier_lower = tier.lower()
    if tier_lower in ("olympics", "world_championship"):
        return [
            RoundConfig(round_type="qualification", advance_count=20),
            RoundConfig(round_type="semifinal", advance_count=8),
            RoundConfig(round_type="final", advance_count=8),
        ]
    elif tier_lower == "world_cup":
        return [
            RoundConfig(round_type="qualification", advance_count=26),
            RoundConfig(round_type="semifinal", advance_count=8),
            RoundConfig(round_type="final", advance_count=8),
        ]
    elif tier_lower == "continental":
        return [
            RoundConfig(round_type="qualification", advance_count=20),
            RoundConfig(round_type="final", advance_count=20),
        ]
    else:
        raise ValueError(f"Unknown event tier: {tier!r}")


def compute_partial_event_probabilities(
    completed_athletes: list[tuple[AthleteProjectionInput, int]],
    remaining_athletes: list[AthleteProjectionInput],
    n_simulations: int = 10_000,
    rng_seed: int | None = None,
) -> dict[int, dict[str, float]]:
    """Like compute_podium_probabilities but locks completed athletes at their finished ranks.

    Completed athletes already have a deterministic rank (e.g. rank 1 = 1.0 win
    probability if they are actually ranked 1st).  Only the remaining athletes
    are simulated via Monte Carlo and assigned to the available rank slots above
    the lowest completed rank.

    This is a v1 approximation: completed athletes have fixed ranks regardless
    of any simulation; remaining athletes are sorted by simulated performance and
    assigned sequentially to the remaining rank slots.

    Args:
        completed_athletes: List of (AthleteProjectionInput, finished_rank) tuples.
            Finished ranks must be positive integers; duplicates are allowed but
            unusual (e.g. ties in qualifying).
        remaining_athletes: Athletes who have not yet competed.  Their probabilities
            are estimated via Monte Carlo within the unfilled rank slots.
        n_simulations: Monte Carlo iterations.
        rng_seed: Optional seed for reproducibility.

    Returns:
        Dict mapping athlete_id to the same schema as compute_podium_probabilities:
            ``win``           — fraction finishing 1st
            ``podium``        — fraction finishing top-3
            ``top_8``         — fraction finishing top-8
            ``expected_rank`` — mean finishing rank
    """
    if not completed_athletes and not remaining_athletes:
        return {}

    results: dict[int, dict[str, float]] = {}

    # Completed athletes: their rank is known with certainty.
    for athlete, rank in completed_athletes:
        results[athlete.athlete_id] = {
            "win": 1.0 if rank == 1 else 0.0,
            "podium": 1.0 if rank <= 3 else 0.0,
            "top_8": 1.0 if rank <= 8 else 0.0,
            "expected_rank": float(rank),
        }

    if not remaining_athletes:
        return results

    # Determine which rank slots are still open.
    taken_ranks = {rank for _, rank in completed_athletes}
    n_remaining = len(remaining_athletes)
    # Assign the lowest available rank slots to the remaining athletes.
    # The remaining athletes compete for all integer rank slots not yet taken,
    # starting from 1 up to (total).
    available_slots: list[int] = []
    candidate = 1
    while len(available_slots) < n_remaining:
        if candidate not in taken_ranks:
            available_slots.append(candidate)
        candidate += 1

    # Monte Carlo: simulate performance of remaining athletes and assign ranks.
    n_simulations = max(1, min(int(n_simulations), MAX_SIMULATIONS))
    rng = np.random.default_rng(rng_seed)
    n = len(remaining_athletes)

    mus = np.array([a.mu for a in remaining_athletes], dtype=np.float64)
    sigmas = np.clip(
        np.array([a.sigma for a in remaining_athletes], dtype=np.float64),
        SIGMA_FLOOR,
        None,
    )

    # (n_simulations, n) — higher score = better
    performances = rng.normal(mus, sigmas, size=(n_simulations, n))
    # For each sim, sort remaining athletes by descending performance and
    # assign them to the available_slots in order.
    slot_arr = np.array(available_slots, dtype=np.int64)  # shape (n_remaining,)
    # order[sim, k] = index of the k-th best remaining athlete in sim
    order = np.argsort(-performances, axis=1)  # (n_simulations, n)
    # ranks[sim, athlete_idx] = assigned rank slot
    ranks = np.empty_like(order)
    sim_indices = np.arange(n_simulations)[:, np.newaxis]
    ranks[sim_indices, order] = slot_arr[np.newaxis, :]

    win_counts = (ranks == 1).sum(axis=0)
    podium_counts = (ranks <= 3).sum(axis=0)
    top8_counts = (ranks <= 8).sum(axis=0)
    mean_rank = ranks.mean(axis=0)

    for i, athlete in enumerate(remaining_athletes):
        results[athlete.athlete_id] = {
            "win": round(float(win_counts[i]) / n_simulations, 4),
            "podium": round(float(podium_counts[i]) / n_simulations, 4),
            "top_8": round(float(top8_counts[i]) / n_simulations, 4),
            "expected_rank": round(float(mean_rank[i]), 2),
        }

    return results


def predict_winner(athletes: list[AthleteProjectionInput]) -> int | None:
    """Return athlete_id of the athlete with the highest mu rating.

    Returns None for an empty list.
    """
    if not athletes:
        return None
    return max(athletes, key=lambda a: a.mu).athlete_id


def expected_finish_ranks(
    athletes: list[AthleteProjectionInput],
) -> list[int]:
    """Return athlete_ids ordered by expected finishing position (best first).

    Athletes are ordered by descending mu (the deterministic best-guess ranking).
    """
    sorted_athletes = sorted(athletes, key=lambda a: a.mu, reverse=True)
    return [a.athlete_id for a in sorted_athletes]
