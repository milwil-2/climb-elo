"""ELO rating engine with Glicko-2 rating-deviation (RD) integration (Issue #51).

Background
----------

Originally a constant-K ELO engine. Issue #51 wires Glicko-2's RD (φ) into the
update so that:

* Beating a high-RD opponent moves you less (g(φ_opp) weighting).
* High-RD athletes (cold start, post-sabbatical) move further per round (large
  φ² scaling in the closed-form Glicko-2 step).
* Inactivity inflates φ via the Wiener-process formula
  ``φ_new = sqrt(φ_old² + σ_inactivity² · months_inactive)``.

Three design decisions (recorded in issue #51, comment from 2026-05-26):

1. **Inactivity inflation** uses calendar-time semantics — months since the
   athlete's last event — matching Glicko-2's Wiener-process model. Reuses the
   existing ``Rating.last_event_at`` Date column. (Decision over event-count
   semantics because event cadence is irregular and per-athlete event-skip
   enumeration would be expensive on every backfill step.)

2. **Margin-of-victory stays separate** from Glicko-2's outcome score
   ``s_j ∈ {0, 0.5, 1}``. The existing margin multiplier folds into the
   effective K instead: ``K_eff = K_base(tier,round) · g(φ_opp) · margin_mult``.
   This preserves issue #53 (MOV audit) as an independent change.

3. **Projection σ** reuses Glicko-2 φ (the value stored in the ``Rating.sigma``
   column) as the projection draw σ in :mod:`engine.projections`. One source of
   truth. Trade-off: φ is rating uncertainty, not performance variance — but
   the practical effect (wider draws for less-certain athletes) is
   directionally correct. Tracked for refinement in a follow-up issue.

Simplifications (deferred to follow-up issues filed against #51)
----------------------------------------------------------------

* **Volatility update**: we run a *simplified closed-form* φ update
  (``1/φ'² = 1/φ_inflated² + v_inv_sum``) without iterating Glickman's full
  Step 5 volatility-σ refit. The volatility is held fixed at the system
  constant ``GLICKO2_DEFAULT_VOLATILITY`` for inflation purposes only. This
  is the standard Glicko-1.5-style approximation; the full Glicko-2
  iteration buys an additional 1-2% calibration in long-running implementations
  and can be ported later.
* **K-factor table** values are halved as a conservative starting point —
  variable effective-K averages around constant K, so halving the base keeps
  per-round magnitudes within the previously-tuned operating range. A proper
  re-grid sweep is a follow-up.

Zero-sum invariant
------------------

μ updates remain pairwise-symmetric and therefore zero-sum across a round
(within floating-point tolerance). φ updates are *per-athlete* (each athlete's
new φ depends on the variance accumulated against all opponents seen in the
round) and are NOT zero-sum — Glicko-2 explicitly allows the round to
*consume* uncertainty across the field.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date

from climbing_elo.models import Discipline, EventTier, RoundType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MU = 1500.0
DEFAULT_SIGMA = 350.0  # display-scale RD for fresh athletes (high uncertainty)
PROVISIONAL_THRESHOLD = (
    3  # kept for UI badge only — Glicko-2 handles cold start via high φ
)

# Glicko-2 system constants (Glickman 2013).
GLICKO2_SCALE = 173.7178  # display-scale ↔ internal-scale conversion
GLICKO2_TAU = 0.5  # τ — recommended range [0.3, 1.2]; 0.5 is a moderate default
GLICKO2_DEFAULT_VOLATILITY = 0.06  # σ in Glicko-2 internal units (held fixed)

# Inactivity inflation: how fast φ grows during a competitive sabbatical.
# Tuned so that an athlete at φ=0.5 (RD≈87, well-established) who skips
# 12 months has their φ inflate to ~0.85 (RD≈148). That re-opens the
# rating to evidence at roughly the same magnitude as the legacy 18-month
# half-life decay did — but it now actually *acts* on the update.
GLICKO2_SIGMA_INACTIVITY = 5.0
GLICKO2_INACTIVITY_GRACE_DAYS = 30  # no inflation for activity gaps < 30 days
GLICKO2_DAYS_PER_MONTH = 30.0

SIGMA_FLOOR = 50.0  # display-scale RD floor (≈ φ=0.29; established athlete)
SIGMA_CEILING = 350.0  # display-scale RD ceiling (≈ φ=2.01; cold start)
PHI_FLOOR = SIGMA_FLOOR / GLICKO2_SCALE
PHI_CEILING = SIGMA_CEILING / GLICKO2_SCALE

MARGIN_CAP = 1.5  # MOV cap, retained from prior tuning

# ---------------------------------------------------------------------------
# Gap-conditioned MOV (Issue #53, 538-style)
# ---------------------------------------------------------------------------
#
# The base MOV multiplier ``min(1 + gap/max_gap, MARGIN_CAP)`` is unconditioned
# on the rating gap — an elite crushing a junior earns the same MOV bonus as
# an elite crushing a peer. 538's NFL/NBA experience and Kovalchik (2020) on
# ATP tennis both show this drives autocorrelation drift at the top.
#
# Fix: damp the MOV bonus when the *favourite* wins by Δμ:
#   multiplier = base · MOV_SOFTENING / (max(Δμ, 0)/MOV_RATING_SCALE + MOV_SOFTENING)
#
# Asymmetric on purpose — an upset (Δμ < 0, underdog wins) keeps the full MOV
# bonus, since a big-margin upset is genuinely high-information.
#
# Constants:
# * ``MOV_RATING_SCALE = 400.0`` — chosen for the current production μ regime
#   (top ≈ 2250, mean ≈ 1500, so Δμ in pairwise contests spans 0–750). At
#   Δμ=400 the damping factor is 2.2/3.2 ≈ 0.69 (31% reduction); at Δμ=800
#   it's 2.2/4.2 ≈ 0.52 (48% reduction). Matches 538's intent of "big bonuses
#   for peer matchups, shrink for mismatches".  Grid-search refinement is a
#   follow-up (#80 K-regrid pairs naturally with this).
# * ``MOV_SOFTENING = 2.2`` — 538 default. Controls how fast the damping
#   kicks in.  Lower → harsher damping; higher → milder.
MOV_RATING_SCALE = 400.0
MOV_SOFTENING = 2.2


def _gap_conditioning_factor(rating_gap: float) -> float:
    """538-style asymmetric damping factor for the MOV multiplier.

    ``factor = MOV_SOFTENING / (max(rating_gap, 0)/MOV_RATING_SCALE + MOV_SOFTENING)``

    Returns 1.0 when ``rating_gap <= 0`` (upset side — full MOV bonus retained)
    and shrinks toward 0 as the favourite's rating advantage grows.
    """
    positive_gap = max(rating_gap, 0.0)
    return MOV_SOFTENING / (positive_gap / MOV_RATING_SCALE + MOV_SOFTENING)


# Speed-specific margin: max meaningful time gap in seconds.
SPEED_MAX_GAP_SECONDS = 2.0

# K-factor table. Halved from prior production values as a conservative
# starting point — under variable effective-K each round will *average* close
# to these numbers but vary by opponent-φ. A proper regrid sweep is filed as
# a #51 follow-up issue.
K_FACTOR_TABLE: dict[EventTier, dict[RoundType, float]] = {
    EventTier.OLYMPICS: {
        RoundType.FINAL: 48.0,
        RoundType.SEMI: 36.0,
        RoundType.QUALIFICATION: 18.0,
    },
    EventTier.WORLD_CHAMPIONSHIP: {
        RoundType.FINAL: 40.0,
        RoundType.SEMI: 30.0,
        RoundType.QUALIFICATION: 15.0,
    },
    EventTier.WORLD_CUP: {
        RoundType.FINAL: 32.0,
        RoundType.SEMI: 24.0,
        RoundType.QUALIFICATION: 12.0,
    },
    EventTier.CONTINENTAL: {
        RoundType.FINAL: 24.0,
        RoundType.SEMI: 18.0,
        RoundType.QUALIFICATION: 9.0,
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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
    sigma: float = DEFAULT_SIGMA  # display-scale RD
    n_events: int = 0
    last_event_at: date | None = None
    provisional: bool = True


# ---------------------------------------------------------------------------
# Glicko-2 primitives
# ---------------------------------------------------------------------------


def glicko2_g(phi: float) -> float:
    """Glicko-2 weighting function (Glickman 2013, §2 step 3).

    ``g(φ) = 1 / sqrt(1 + 3φ² / π²)``

    Approaches 1.0 as φ → 0 (very confident opponent) and 0.0 as φ → ∞ (very
    uncertain opponent). A round against an opponent with huge φ contributes
    little to your rating change — we don't yet trust the comparison.
    """
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def glicko2_expected_score(mu_a: float, mu_b: float, phi_b: float) -> float:
    """Glicko-2 expected score on the display scale.

    Equivalent to the standard Glicko-2 ``E`` formula but operating on
    display-scale μ (Elo points) and display-scale φ (RD). Internally we
    convert to Glickman's internal units (divide μ-gap by GLICKO2_SCALE,
    divide φ by GLICKO2_SCALE) before applying the logistic.

    ``E = 1 / (1 + exp(-g(φ_b_internal) · (μ_a_internal - μ_b_internal)))``
    """
    phi_b_internal = phi_b / GLICKO2_SCALE
    mu_gap_internal = (mu_a - mu_b) / GLICKO2_SCALE
    return 1.0 / (1.0 + math.exp(-glicko2_g(phi_b_internal) * mu_gap_internal))


def glicko2_inflate_phi(
    sigma_display: float,
    last_event_at: date | None,
    current_date: date,
) -> float:
    """Inflate φ for inactivity per Glicko-2's Wiener-process model.

    Calendar-time semantics — months since last event (decision 1 in module
    docstring). Returns the new display-scale RD, clamped at SIGMA_CEILING.

    Formula (internal units, then converted back):
        φ_new² = φ_old² + σ_inactivity² · months_inactive

    where ``σ_inactivity = GLICKO2_SIGMA_INACTIVITY`` is on the internal
    scale (≈ 0.029 RD-units per √month). With the chosen value of 5.0 (display
    scale), a 12-month gap inflates a φ=0.5 athlete (RD≈87) to φ≈0.85 (RD≈148);
    a fresh athlete at φ=2.014 (RD=350) stays clamped at the ceiling.
    """
    if last_event_at is None or current_date <= last_event_at:
        return sigma_display
    days_inactive = (current_date - last_event_at).days
    if days_inactive <= GLICKO2_INACTIVITY_GRACE_DAYS:
        return sigma_display
    months_inactive = days_inactive / GLICKO2_DAYS_PER_MONTH
    sigma_new = math.sqrt(
        sigma_display * sigma_display
        + GLICKO2_SIGMA_INACTIVITY * GLICKO2_SIGMA_INACTIVITY * months_inactive
    )
    return min(sigma_new, SIGMA_CEILING)


# ---------------------------------------------------------------------------
# Public expected-score wrapper (legacy callers / display)
# ---------------------------------------------------------------------------


def expected_score(mu_a: float, mu_b: float) -> float:
    """Legacy 400-scale expected score — used by display code only.

    The actual Glicko-2 update uses :func:`glicko2_expected_score` which is
    φ-weighted. This wrapper exists so that ``/breakdown/{a}/{e}`` and other
    UI pages keep showing the familiar 400-scale logistic.
    """
    return 1.0 / (1.0 + 10.0 ** ((mu_b - mu_a) / 400.0))


# ---------------------------------------------------------------------------
# K-factor + margin helpers (unchanged)
# ---------------------------------------------------------------------------


def get_k_factor(tier: EventTier, round_type: RoundType) -> float:
    return K_FACTOR_TABLE[tier][round_type]


def compute_margin_multiplier(
    score_a: float | None,
    score_b: float | None,
    max_gap: float = 20.0,
    rating_gap: float = 0.0,
) -> float:
    """Lead-style margin multiplier with optional 538-style gap conditioning.

    ``rating_gap`` is ``μ_winner − μ_loser`` (pre-update). Defaults to 0.0 for
    backward compatibility; the gap-conditioned version reduces exactly to the
    legacy formula when ``rating_gap == 0``. When the favourite (rating_gap > 0)
    wins, the bonus is damped per :func:`_gap_conditioning_factor`. Upsets
    (rating_gap < 0) keep the full bonus.

    Note: the MARGIN_CAP ceiling is enforced on the *base* multiplier; the
    conditioning factor can only shrink it further, never above the cap.
    """
    if score_a is None or score_b is None:
        return 1.0
    gap = abs(score_a - score_b)
    base = min(1.0 + gap / max_gap, MARGIN_CAP)
    return base * _gap_conditioning_factor(rating_gap)


BOULDER_MARGIN_MAX_GAP = 1000.0

# Regex for the old-format ordinal Boulder score: e.g. "1T2z 3 4" or "2T2 3B4"
_OLD_BOULDER_RE = re.compile(
    r"(\d+)[Tt](\d+)[Zz]\s+(\d+)\s+(\d+)"  # "NTMz A B"
    r"|(\d+)[Tt](\d+)\s+(\d+)[Bb](\d+)",  # "NT A MBB"
)


def _is_new_boulder_format(raw_score: str) -> bool:
    """Return True if *raw_score* is a numeric/decimal value (new 2025+ format)."""
    raw = raw_score.strip()
    try:
        float(raw)
        return True
    except ValueError:
        return False


def normalize_boulder_score(raw_score: str) -> float | None:
    """Normalize a Boulder raw score to a comparable float.

    Handles both formats:

    * **New format (2025+):** a decimal string like ``"34.5"`` — returned as-is.
    * **Old format (pre-2025):** an ordinal string like ``"1T2z 3 4"`` or
      ``"2T2 3B4"`` — parsed into
      ``tops * 1000 + zones * 100 - top_att * 10 - zone_att``.

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
    rating_gap: float = 0.0,
) -> float:
    """Margin multiplier for Boulder discipline (538-style gap-conditioned)."""
    return compute_margin_multiplier(
        score_a, score_b, max_gap=BOULDER_MARGIN_MAX_GAP, rating_gap=rating_gap
    )


def compute_speed_margin_multiplier(
    winner_time: float | None,
    loser_time: float | None,
    rating_gap: float = 0.0,
) -> float:
    """Margin multiplier for Speed discipline (times in seconds, lower is better).

    Gap-conditioned in the same fashion as Lead/Boulder — favourite wins get
    damped, upsets keep the full bonus. See :func:`compute_margin_multiplier`.
    """
    if winner_time is None or loser_time is None:
        return 1.0
    gap = abs(loser_time - winner_time)
    base = min(1.0 + gap / SPEED_MAX_GAP_SECONDS, MARGIN_CAP)
    return base * _gap_conditioning_factor(rating_gap)


# ---------------------------------------------------------------------------
# Round update — Glicko-2 path
# ---------------------------------------------------------------------------


def calculate_round_updates(
    results: list[AthleteResult],
    ratings: dict[int, AthleteRating],
    event_tier: EventTier,
    round_type: RoundType,
    event_date: date,
    discipline: Discipline = Discipline.LEAD,
) -> list[RatingUpdate]:
    """Compute per-athlete μ and φ updates for one round.

    Algorithm
    ---------

    1. Inflate each athlete's φ for inactivity (Wiener-process model on
       calendar-time).
    2. For each ordered pair (i ahead of j):
       a. Compute g(φ_j_internal), E = glicko2_expected_score(μ_i, μ_j, φ_j),
          and margin multiplier per discipline.
       b. K_eff = K_base(tier,round) · g(φ_j_internal) · margin_mult
          (decision 2: MOV stays separate from s_j, folds into K).
       c. delta_pair = K_eff · (1.0 − E)
          μ_i += +delta_pair ; μ_j += -delta_pair  (zero-sum on μ).
       d. Accumulate Glicko-2 variance contribution for *both* sides into the
          per-athlete v_inv accumulator:
              g_i_internal = g(φ_j_internal)
              v_inv_i += g_i² · E · (1−E)
              g_j_internal = g(φ_i_internal)
              v_inv_j += g_j² · E_j · (1−E_j)   with E_j = 1−E
    3. Per-athlete φ update (simplified closed-form):
          1/φ_new² = 1/φ_inflated² + v_inv_sum
       Clamped to [PHI_FLOOR, PHI_CEILING].

    Notes
    -----

    * **Zero-sum on μ** holds pair-by-pair (each pair contributes equal/opposite
      deltas). Across the round the sum is therefore zero to floating-point
      precision.
    * **φ update is not zero-sum** — that's by design; Glicko-2 lets each
      round *consume* uncertainty across the field.
    * **DNS** athletes are excluded entirely (no φ inflation, no update).
    * **PROVISIONAL_K_MULTIPLIER is retired** — Glicko-2 handles cold start
      natively via large initial φ → larger μ update via φ² scaling at the
      end. The ``provisional`` flag is still set on the Rating row for UI use.
    """
    active = [r for r in results if not r.dns]
    if len(active) < 2:
        return []

    base_k = get_k_factor(event_tier, round_type)

    # 1) Inflate φ for inactivity, store the inflated display-scale RDs.
    sigma_inflated: dict[int, float] = {}
    for res in active:
        rating = ratings.get(res.athlete_id, AthleteRating(athlete_id=res.athlete_id))
        sigma_inflated[res.athlete_id] = glicko2_inflate_phi(
            rating.sigma, rating.last_event_at, event_date
        )

    deltas: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    v_inv_sum: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    pairs: dict[int, list[PairContribution]] = {r.athlete_id: [] for r in active}

    for i, res_i in enumerate(active):
        rating_i = ratings.get(
            res_i.athlete_id, AthleteRating(athlete_id=res_i.athlete_id)
        )
        mu_i = rating_i.mu
        sigma_i = sigma_inflated[res_i.athlete_id]
        phi_i_internal = sigma_i / GLICKO2_SCALE

        for j, res_j in enumerate(active):
            if i == j:
                continue
            # Process each unordered pair once — skip the loser side.
            if res_i.rank >= res_j.rank:
                continue
            # (At this point: res_i finished ahead of res_j; ties are skipped.)

            rating_j = ratings.get(
                res_j.athlete_id, AthleteRating(athlete_id=res_j.athlete_id)
            )
            mu_j = rating_j.mu
            sigma_j = sigma_inflated[res_j.athlete_id]
            phi_j_internal = sigma_j / GLICKO2_SCALE

            # g(φ) for both sides — used both for K weighting and v_inv.
            g_phi_j = glicko2_g(phi_j_internal)
            g_phi_i = glicko2_g(phi_i_internal)

            # E from i's perspective (i is favoured to win if mu_i > mu_j).
            e_i = glicko2_expected_score(mu_i, mu_j, sigma_j)
            e_j = 1.0 - e_i  # symmetric

            # Margin multiplier per discipline. The rating gap (μ_winner −
            # μ_loser, using *pre-update* μs) feeds the 538-style gap
            # conditioning: favourite wins (Δμ > 0) get a damped bonus, upsets
            # (Δμ < 0) keep the full bonus. See ``_gap_conditioning_factor``.
            rating_gap = mu_i - mu_j  # res_i is the winner (rank < res_j's rank)
            if res_i.dnf:
                margin_mult = 1.0
            elif discipline == Discipline.SPEED:
                margin_mult = compute_speed_margin_multiplier(
                    res_i.score_normalized,
                    res_j.score_normalized,
                    rating_gap=rating_gap,
                )
            elif discipline == Discipline.BOULDER:
                margin_mult = compute_boulder_margin_multiplier(
                    res_i.score_normalized,
                    res_j.score_normalized,
                    rating_gap=rating_gap,
                )
            else:
                margin_mult = compute_margin_multiplier(
                    res_i.score_normalized,
                    res_j.score_normalized,
                    rating_gap=rating_gap,
                )

            # K_eff weights this pair by the opponent's certainty (g(φ_opp))
            # and by MOV. Decision 2: MOV stays separate from s_j; the score
            # outcome stays in {0, 0.5, 1} per Glicko-2 spec.
            k_eff_for_i = base_k * g_phi_j * margin_mult
            k_eff_for_j = base_k * g_phi_i * margin_mult

            # Pair contribution to μ (symmetric so the round remains zero-sum).
            # We use the *minimum* of the two effective K factors for symmetric
            # application — this preserves zero-sum exactly. Using a single
            # K per pair (rather than per-side) is the standard approach in
            # most production Glicko-2 implementations that need zero-sum.
            k_pair = min(k_eff_for_i, k_eff_for_j)
            delta_pair = k_pair * (1.0 - e_i)
            deltas[res_i.athlete_id] += delta_pair
            deltas[res_j.athlete_id] -= delta_pair

            # Glicko-2 variance contributions. Each side accumulates against
            # the opponent's g(φ); this drives the per-athlete φ shrinkage.
            v_inv_sum[res_i.athlete_id] += g_phi_j * g_phi_j * e_i * (1.0 - e_i)
            v_inv_sum[res_j.athlete_id] += g_phi_i * g_phi_i * e_j * (1.0 - e_j)

            pairs[res_i.athlete_id].append(
                PairContribution(
                    opponent_id=res_j.athlete_id,
                    result="won",
                    expected=round(e_i, 4),
                    actual=1.0,
                    delta=round(delta_pair, 2),
                    margin_multiplier=round(margin_mult, 2),
                )
            )
            pairs[res_j.athlete_id].append(
                PairContribution(
                    opponent_id=res_i.athlete_id,
                    result="lost",
                    expected=round(e_j, 4),
                    actual=0.0,
                    delta=round(-delta_pair, 2),
                    margin_multiplier=round(margin_mult, 2),
                )
            )

    # 3) Per-athlete φ update + assemble RatingUpdate.
    updates = []
    for res in active:
        aid = res.athlete_id
        rating = ratings.get(aid, AthleteRating(athlete_id=aid))

        mu_before = rating.mu
        sigma_before = sigma_inflated[aid]
        phi_internal = sigma_before / GLICKO2_SCALE

        # Simplified closed-form Glicko-2 φ update:
        #   1/φ_new² = 1/φ_inflated² + Σ g(φ_opp)² · E · (1-E)
        # (full volatility iteration is a follow-up — see module docstring.)
        v_inv = v_inv_sum.get(aid, 0.0)
        inv_phi_sq_new = 1.0 / (phi_internal * phi_internal) + v_inv
        phi_new = 1.0 / math.sqrt(inv_phi_sq_new)
        sigma_after_display = phi_new * GLICKO2_SCALE
        # Clamp to display-scale floor/ceiling.
        sigma_after = max(SIGMA_FLOOR, min(SIGMA_CEILING, sigma_after_display))

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
