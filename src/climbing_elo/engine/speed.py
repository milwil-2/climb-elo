"""Bracket-native Speed climbing rating updates (Issue #56).

Background
----------

Speed climbing is structurally a *single-elimination bracket*, not a free-for-all
points race like Lead or Boulder. After qualification (a time trial), athletes
enter a knock-out bracket (1/16 → 1/8 → 1/4 → semi → small final + big final).
Each round consists of N/2 head-to-head matchups — an athlete only races a
specific opponent, not every other competitor in the field.

Pre-#56, the engine treated Speed identically to Lead/Boulder: a Plackett-Luce
pairwise decomposition over the *full* finishing order, generating O(N²) pair
updates per round. This wastes signal: an athlete who never raced rank-5 should
not have their μ moved by that imagined matchup.

This module implements the **bracket-native** alternative (R6 in
``docs/RATING_SYSTEM_RESEARCH.md``, GitHub issue #56):

1. **Adjacent-pair matchups only.** With only ranked finish data available
   (Option A — no schema change), we approximate the bracket by pairing
   athletes adjacent in the final ordering: rank 1 vs rank 2, rank 3 vs rank 4,
   and so on. These pairs map directly onto a real bracket's final-stage
   matchups (gold-medal race, bronze-medal race, 5th-place race, …) and are the
   closest approximation we can recover from a flat rank list.

   Option B — extending the schema with explicit bracket position columns and
   recording the *actual* matchups during scrape — is deferred to a follow-up
   issue. The IFSC results API does expose bracket structure, so this is
   feasible; it just costs a migration + scraper rewrite that exceeded the
   scope of #56.

2. **Davidson (1970) tie handling** — Bradley-Terry extended with a tie
   parameter ν:

   ::

       P(i beats j) = v_i / (v_i + v_j + ν·sqrt(v_i·v_j))
       P(tie)       = ν·sqrt(v_i·v_j) / (v_i + v_j + ν·sqrt(v_i·v_j))

   Where ``v_i = 10**(μ_i/400)`` is the standard Elo strength. For Speed, a
   pair of times within ε (default 0.01 s — finer than the IFSC's
   millisecond-resolution timing system) is treated as a *tie*, contributing a
   0.5/0.5 outcome score (Glicko-2 ``s_j = 0.5``) instead of 1/0.

   See Davidson 1970, *JASA* 65(329):317-328, DOI 10.1080/01621459.1970.10481082.

Zero-sum invariant
------------------

Each adjacent pair contributes equal-and-opposite μ deltas; summing across the
round yields zero to floating-point tolerance. Pairs are independent (no
shared athletes), so the invariant holds trivially. φ updates are per-athlete
and not zero-sum (same as the Lead/Boulder path — Glicko-2 explicitly allows
the round to consume uncertainty across the field).

Why not also touch Lead/Boulder?
--------------------------------

R6's scope is explicitly the Speed branch only. Lead and Boulder are *ranked
finishes* with no bracket structure — the existing P-L decomposition is
correct for those disciplines. This module is invoked only when
``discipline == Discipline.SPEED``; the Lead/Boulder code path is untouched.
"""

from __future__ import annotations

import math
from datetime import date

from climbing_elo.engine.elo import (
    GLICKO2_SCALE,
    AthleteRating,
    AthleteResult,
    EloConfig,
    PairContribution,
    RatingUpdate,
    compute_speed_margin_multiplier,
    get_k_factor,
    glicko2_g,
    glicko2_inflate_phi,
)
from climbing_elo.models import EventTier, RoundType

# ---------------------------------------------------------------------------
# Davidson (1970) tie model — pure helpers
# ---------------------------------------------------------------------------


def davidson_expected_scores(
    mu_a: float,
    mu_b: float,
    nu: float,
) -> tuple[float, float, float]:
    """Davidson 1970 outcome probabilities for the pair (a, b).

    Returns ``(p_a_wins, p_b_wins, p_tie)`` summing to 1.0.

    Using standard Elo strength ``v = 10**(μ/400)``::

        P(a wins) = v_a / (v_a + v_b + nu·sqrt(v_a·v_b))
        P(b wins) = v_b / (v_a + v_b + nu·sqrt(v_a·v_b))
        P(tie)    = nu·sqrt(v_a·v_b) / (v_a + v_b + nu·sqrt(v_a·v_b))

    At ``nu == 0`` this collapses to vanilla Bradley-Terry (no ties). For ν > 0
    a non-zero tie probability appears, peaking when μ_a == μ_b.

    For numerical stability we work with the *log-strength* gap rather than the
    raw exponentials — ``v_a / (v_a + v_b)`` = ``1 / (1 + 10**((μ_b−μ_a)/400))``.
    """
    # Δ = μ_a − μ_b in Elo points.
    delta = mu_a - mu_b
    # Standard logistic on the log-strength gap (denominator is normalised by
    # v_a + v_b implicitly via the logistic). We need three quantities:
    #   p_a_unnorm = v_a / (v_a + v_b)
    #   p_b_unnorm = v_b / (v_a + v_b)
    #   tie_term   = nu·sqrt(v_a·v_b) / (v_a + v_b)
    # Then the full denominator (v_a + v_b + nu·sqrt(v_a·v_b)) / (v_a + v_b)
    # equals 1 + tie_term, so the final probabilities divide by 1 + tie_term.
    p_a_unnorm = 1.0 / (1.0 + 10.0 ** (-delta / 400.0))
    p_b_unnorm = 1.0 - p_a_unnorm
    # sqrt(v_a·v_b) / (v_a + v_b) = 1 / (10**(delta/800) + 10**(-delta/800))
    # = sech-like form. Equivalent: sqrt(p_a · p_b).
    tie_term = nu * math.sqrt(p_a_unnorm * p_b_unnorm)
    denom = 1.0 + tie_term
    return p_a_unnorm / denom, p_b_unnorm / denom, tie_term / denom


def davidson_expected_score(
    mu_a: float,
    mu_b: float,
    nu: float,
) -> float:
    """Expected score for athlete a under the Davidson tie model.

    ``E_a = P(a wins) + 0.5 · P(tie)`` — the standard expected-score
    convention used in Elo/Glicko updates. Falls back to the vanilla logistic
    when ν = 0.

    With ν = 0: returns ``1 / (1 + 10**((μ_b − μ_a)/400))`` (legacy Elo
    expected score, unchanged from the pre-#56 formula).
    """
    p_a, _, p_tie = davidson_expected_scores(mu_a, mu_b, nu)
    return p_a + 0.5 * p_tie


# ---------------------------------------------------------------------------
# Pair update primitive — pure function on (μ, σ, times)
# ---------------------------------------------------------------------------


def compute_speed_pair_update(
    time_winner: float | None,
    time_loser: float | None,
    mu_winner: float,
    mu_loser: float,
    sigma_winner: float,
    sigma_loser: float,
    k_base: float,
    *,
    is_dnf: bool = False,
    config: EloConfig,
) -> tuple[float, float, float, float, float]:
    """Compute symmetric μ deltas + Glicko-2 variance contributions for one Speed pair.

    Pure function — no side effects, no DB access. Used by both the public
    :func:`calculate_speed_round_updates` entry point and the unit tests.

    Parameters
    ----------
    time_winner, time_loser
        Race times in seconds; ``None`` for DNFs / missing data. If either is
        ``None`` the MOV multiplier degrades to 1.0.
    mu_winner, mu_loser
        Display-scale μ for the two athletes (per the round's *finishing
        order*; ``winner`` finished ahead of ``loser``).
    sigma_winner, sigma_loser
        Display-scale RD (φ in Glicko-2 units before dividing by GLICKO2_SCALE).
    k_base
        Tier × round K factor from :func:`get_k_factor`.
    is_dnf
        If ``True``, treat the loser as DNF (no margin bonus, MOV mult = 1.0).
        Mirrors the existing Lead/Boulder DNF behaviour.
    config
        Engine config — read for ``speed_tie_epsilon_seconds``, the Davidson
        ν parameter, the margin cap, etc.

    Returns
    -------
    Tuple ``(delta_winner, delta_loser, v_inv_winner, v_inv_loser, margin_mult)``:

    * ``delta_winner``: μ delta to add to the winner (positive on a clean win,
      smaller positive on a tie, possibly slightly negative on a "win" by the
      lower-rated favourite due to gap-conditioning … actually no, we use
      ``actual − expected`` so a clean win where ``actual = 1.0`` and
      ``expected < 1.0`` is always ≥ 0).
    * ``delta_loser``: ``-delta_winner`` (zero-sum).
    * ``v_inv_winner``, ``v_inv_loser``: Glicko-2 variance contributions for
      the per-athlete φ shrinkage step.
    * ``margin_mult``: the margin multiplier actually applied (for breakdown
      recording).
    """
    # Treat the times as a possible tie.
    is_tie = False
    if (
        not is_dnf
        and time_winner is not None
        and time_loser is not None
        and abs(time_winner - time_loser) < config.speed_tie_epsilon_seconds
    ):
        is_tie = True

    # Glicko-2 g(φ) weighting — internal-scale φ.
    phi_winner_internal = sigma_winner / GLICKO2_SCALE
    phi_loser_internal = sigma_loser / GLICKO2_SCALE
    g_phi_loser = glicko2_g(phi_loser_internal)
    g_phi_winner = glicko2_g(phi_winner_internal)

    # Davidson expected score on the *winner side* — folds tie probability in.
    e_winner = davidson_expected_score(mu_winner, mu_loser, config.speed_davidson_nu)
    e_loser = 1.0 - e_winner

    # Actual outcome — 0.5 for ties, else 1.0 (winner side). The loser side is
    # always ``1 - actual_winner`` so we only need the winner-side value below.
    actual_winner = 0.5 if is_tie else 1.0

    # Margin multiplier — neutral on DNF/ties, gap-conditioned otherwise.
    rating_gap = mu_winner - mu_loser  # winner == finished-ahead athlete
    if is_dnf or is_tie:
        margin_mult = 1.0
    else:
        margin_mult = compute_speed_margin_multiplier(
            time_winner,
            time_loser,
            rating_gap=rating_gap,
            config=config,
        )

    # K_eff weights this pair by the opponent's certainty (g(φ_opp)) and by
    # MOV. Same shape as the Lead/Boulder path — we take the min so updates
    # are symmetric (zero-sum on μ).
    k_eff_winner = k_base * g_phi_loser * margin_mult
    k_eff_loser = k_base * g_phi_winner * margin_mult
    k_pair = min(k_eff_winner, k_eff_loser)

    # Pair contribution to μ.
    delta_winner = k_pair * (actual_winner - e_winner)
    delta_loser = -delta_winner

    # Glicko-2 variance contributions. Same formula as the Lead/Boulder path.
    v_inv_winner = g_phi_loser * g_phi_loser * e_winner * (1.0 - e_winner)
    v_inv_loser = g_phi_winner * g_phi_winner * e_loser * (1.0 - e_loser)

    return delta_winner, delta_loser, v_inv_winner, v_inv_loser, margin_mult


# ---------------------------------------------------------------------------
# Round-level entry point
# ---------------------------------------------------------------------------


def calculate_speed_round_updates(
    results: list[AthleteResult],
    ratings: dict[int, AthleteRating],
    event_tier: EventTier,
    round_type: RoundType,
    event_date: date,
    config: EloConfig,
) -> list[RatingUpdate]:
    """Bracket-native Speed round update (R6 / Issue #56).

    Mirrors the public contract of
    :func:`climbing_elo.engine.elo.calculate_round_updates` but processes only
    adjacent-rank pairs (Option A bracket approximation) and uses Davidson tie
    handling.

    Algorithm
    ---------

    1. Inflate each athlete's φ for inactivity (identical to the Lead/Boulder
       path).
    2. Sort active (non-DNS) athletes by ascending rank.
    3. Walk the sorted list pairwise — (rank 1, rank 2), (rank 3, rank 4), … —
       and for each pair call :func:`compute_speed_pair_update`. The winner is
       the lower-ranked athlete in each pair; ties on time are handled by
       Davidson and contribute 0.5/0.5 outcomes.
    4. Per-athlete φ shrinkage via the same simplified closed-form as the
       Lead/Boulder path.

    Returns the same :class:`RatingUpdate` shape as the main engine so the
    backfill layer doesn't need to know about the dispatch. Athletes with no
    adjacent-pair matchup (e.g. an odd field where the last athlete is
    unpaired) still get a :class:`RatingUpdate` row with the inflated σ but
    zero μ delta and no contributing pairs.
    """
    active = [r for r in results if not r.dns]
    if len(active) < 2:
        return []

    # Sort by rank ascending so that the kth pair is (rank 2k-1, rank 2k).
    active_sorted = sorted(active, key=lambda r: r.rank)

    base_k = get_k_factor(event_tier, round_type, config)

    # 1) Inflate σ for inactivity.
    sigma_inflated: dict[int, float] = {}
    for res in active_sorted:
        rating = ratings.get(res.athlete_id, AthleteRating(athlete_id=res.athlete_id))
        sigma_inflated[res.athlete_id] = glicko2_inflate_phi(
            rating.sigma, rating.last_event_at, event_date, config
        )

    deltas: dict[int, float] = {r.athlete_id: 0.0 for r in active_sorted}
    v_inv_sum: dict[int, float] = {r.athlete_id: 0.0 for r in active_sorted}
    pairs: dict[int, list[PairContribution]] = {r.athlete_id: [] for r in active_sorted}

    # 2 + 3) Walk adjacent pairs and apply the per-pair update.
    for idx in range(0, len(active_sorted) - 1, 2):
        winner_res = active_sorted[idx]
        loser_res = active_sorted[idx + 1]

        # If the two athletes share the same rank (a tie at the rank level),
        # respect it as a tie regardless of time — but we still let
        # compute_speed_pair_update treat it via the ε-tie path, which will
        # fire when times are close (or both None). The pure rank-tie case
        # produces no margin bonus and 0.5/0.5 — same as our existing tie
        # handling in the Lead/Boulder path.
        rank_tie = winner_res.rank == loser_res.rank

        rating_winner = ratings.get(
            winner_res.athlete_id,
            AthleteRating(athlete_id=winner_res.athlete_id),
        )
        rating_loser = ratings.get(
            loser_res.athlete_id,
            AthleteRating(athlete_id=loser_res.athlete_id),
        )

        # DNF logic: if the *loser* DNF'd (false start, fall) the winner
        # gets no margin bonus — match the existing speed_false_start test.
        is_dnf = winner_res.dnf or loser_res.dnf

        d_w, d_l, v_w, v_l, margin_mult = compute_speed_pair_update(
            time_winner=winner_res.score_normalized,
            time_loser=loser_res.score_normalized,
            mu_winner=rating_winner.mu,
            mu_loser=rating_loser.mu,
            sigma_winner=sigma_inflated[winner_res.athlete_id],
            sigma_loser=sigma_inflated[loser_res.athlete_id],
            k_base=base_k,
            is_dnf=is_dnf,
            config=config,
        )

        # Rank-tie override: if both finished with the same rank, force
        # actual=0.5 by zeroing out the asymmetric delta the pair_update
        # produced. (compute_speed_pair_update already treats time-ε-ties via
        # Davidson; this catch is for explicit rank ties without close times.)
        if rank_tie and not is_dnf:
            d_w = 0.0
            d_l = 0.0
            margin_mult = 1.0

        deltas[winner_res.athlete_id] += d_w
        deltas[loser_res.athlete_id] += d_l
        v_inv_sum[winner_res.athlete_id] += v_w
        v_inv_sum[loser_res.athlete_id] += v_l

        # Determine the result string for the breakdown view. "tied" surfaces
        # in the UI as a distinct outcome from win/loss.
        is_tie_for_record = rank_tie or (
            not is_dnf
            and winner_res.score_normalized is not None
            and loser_res.score_normalized is not None
            and abs(winner_res.score_normalized - loser_res.score_normalized)
            < config.speed_tie_epsilon_seconds
        )

        # Use the Davidson expected scores for the breakdown view so the
        # "expected" column reflects the model's true prediction (including
        # the implicit tie mass).
        e_w_breakdown = davidson_expected_score(
            rating_winner.mu, rating_loser.mu, config.speed_davidson_nu
        )

        if is_tie_for_record:
            result_w, result_l = "tied", "tied"
            actual_w, actual_l = 0.5, 0.5
        else:
            result_w, result_l = "won", "lost"
            actual_w, actual_l = 1.0, 0.0

        pairs[winner_res.athlete_id].append(
            PairContribution(
                opponent_id=loser_res.athlete_id,
                result=result_w,
                expected=round(e_w_breakdown, 4),
                actual=actual_w,
                delta=round(d_w, 2),
                margin_multiplier=round(margin_mult, 2),
            )
        )
        pairs[loser_res.athlete_id].append(
            PairContribution(
                opponent_id=winner_res.athlete_id,
                result=result_l,
                expected=round(1.0 - e_w_breakdown, 4),
                actual=actual_l,
                delta=round(d_l, 2),
                margin_multiplier=round(margin_mult, 2),
            )
        )

    # 4) Per-athlete φ shrinkage. Identical formula to the Lead/Boulder path —
    # the *only* structural difference is which pairs contribute to v_inv.
    updates: list[RatingUpdate] = []
    for res in active_sorted:
        aid = res.athlete_id
        rating = ratings.get(aid, AthleteRating(athlete_id=aid))

        mu_before = rating.mu
        sigma_before = sigma_inflated[aid]
        phi_internal = sigma_before / GLICKO2_SCALE

        v_inv = v_inv_sum.get(aid, 0.0)
        inv_phi_sq_new = 1.0 / (phi_internal * phi_internal) + v_inv
        phi_new = 1.0 / math.sqrt(inv_phi_sq_new)
        sigma_after_display = phi_new * GLICKO2_SCALE
        sigma_after = max(
            config.sigma_floor, min(config.sigma_ceiling, sigma_after_display)
        )

        mu_after = mu_before + deltas[aid]

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
