"""Monte Carlo projections for upcoming/in-progress climbing events.

Given a set of athletes and their current ELO ratings (mu, sigma), this module
simulates thousands of hypothetical performances to estimate win and podium
probabilities.

Performance model: each athlete's performance in a single event is drawn from
N(mu, sigma). Higher score = better finish. Ties are broken randomly (extremely
rare with continuous draws).
"""
from __future__ import annotations

from dataclasses import dataclass

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
