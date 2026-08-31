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

Volatility update (Issue #81)
-----------------------------

The per-round φ update runs the **full Glicko-2 Step 5 volatility refit** via
the Illinois root-find (:func:`glicko2_update_volatility`), replacing the
earlier simplified closed-form (``1/φ'² = 1/φ_inflated² + v_inv``). Each round:

1. Refit σ' from the round's (v, Δ) evidence (Step 5, Illinois algorithm).
2. Inflate φ by the refit volatility — ``φ* = sqrt(φ² + σ'²)`` (Step 6).
3. Shrink with the variance evidence — ``1/φ'² = 1/φ*² + 1/v`` (Step 7).

Volatility evolves in-memory across a backfill run (seeded at
``GLICKO2_DEFAULT_VOLATILITY``); it has no DB column, so a fresh backfill
re-seeds deterministically. Calendar-time inactivity inflation
(:func:`glicko2_inflate_phi`) still runs first, before the round update.

Simplifications (deferred to follow-up issues filed against #51)
----------------------------------------------------------------

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

import copy
import dataclasses
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date

from climbing_elo.models import Discipline, EventTier, RoundType

# ---------------------------------------------------------------------------
# Constants — initial / scalar defaults
# ---------------------------------------------------------------------------
#
# These are the immutable defaults that seed :class:`EloConfig`. Top-level
# constants like ``MARGIN_CAP``, ``MOV_RATING_SCALE``, ``K_FACTOR_TABLE`` are
# kept further down as **back-compat re-exports** of ``DEFAULT_CONFIG`` fields;
# new code should accept ``config: EloConfig`` and read from it instead of
# importing the bare constants. The re-exports are pinned at import time and
# are NOT a live view onto the active config — mutating them no longer changes
# engine behaviour (see Issue #83 Target 3).

DEFAULT_MU = 1500.0
DEFAULT_SIGMA = 350.0  # display-scale RD for fresh athletes (high uncertainty)

# Glicko-2 system constants (Glickman 2013).
GLICKO2_SCALE = 173.7178  # display-scale ↔ internal-scale conversion
GLICKO2_DEFAULT_VOLATILITY = 0.06  # σ in Glicko-2 internal units (initial value)
GLICKO2_INACTIVITY_GRACE_DAYS = 30  # no inflation for activity gaps < 30 days
GLICKO2_DAYS_PER_MONTH = 30.0
GLICKO2_VOLATILITY_EPSILON = 1e-6  # Illinois root-find convergence tolerance
GLICKO2_VOLATILITY_MAX_ITER = 100  # Illinois iteration hard cap (safety)

# Default K-factor table. Halved from prior production values as a
# conservative starting point — under variable effective-K each round will
# *average* close to these numbers but vary by opponent-φ. A proper regrid
# sweep is filed as a #51 follow-up issue (#80).
#
# Default Tournament Participation Bonus table (Issue #90 — Gap 1 from #88).
#
# Per-tier ordered list of μ-credit values for the top-K finishers (rank 1
# first). Top-K finishers receive the per-rank bonus; the *total* bonus is
# then debited uniformly across ALL participants (top-K included) so the
# event-level sum is zero. See ``compute_tournament_participation_bonus``.
#
# Starting values per the #90 issue body. Olympics decay linearly to 0 at
# rank 8; lower tiers compress the curve and stop earlier.
_DEFAULT_TPB_TABLE: dict[EventTier, list[float]] = {
    EventTier.OLYMPICS: [30.0, 22.5, 15.0, 11.25, 7.5, 5.0, 2.5, 0.0],
    EventTier.WORLD_CHAMPIONSHIP: [20.0, 15.0, 10.0, 7.5, 5.0, 3.33, 1.67, 0.0],
    EventTier.WORLD_CUP: [12.0, 9.0, 6.0, 4.0, 2.0, 0.0],
    EventTier.CONTINENTAL: [5.0, 3.33, 1.67, 0.0],
}


def _default_tpb_table() -> dict[EventTier, list[float]]:
    """Deep copy of the default TPB table for :class:`EloConfig`."""
    return copy.deepcopy(_DEFAULT_TPB_TABLE)


# K values updated 2026-05-27 per the regrid sweep in docs/K_REGRID_REPORT.md.
# WC / WCh / Continental cells with data in the source DB were re-tuned via
# coordinate descent to keep μ-p95 in the elite band [1900, 2200]; Olympics
# and SEMI cells lacked data in the sweep and retain their pre-regrid values.
# Re-run scripts/regrid_k_factors.py after any change to the effective-K math
# (σ_inactivity, margin cap, MOV gap-conditioning) and apply the recommended
# dict here.
_DEFAULT_K_FACTORS: dict[EventTier, dict[RoundType, float]] = {
    EventTier.OLYMPICS: {
        RoundType.FINAL: 48.0,  # unchanged — no Olympics data in source DB
        RoundType.SEMI: 36.0,  # unchanged — no data
        RoundType.QUALIFICATION: 18.0,  # unchanged — no data
    },
    EventTier.WORLD_CHAMPIONSHIP: {
        RoundType.FINAL: 15.0,  # was 40.0
        RoundType.SEMI: 30.0,  # unchanged — no semi-round data in source DB
        RoundType.QUALIFICATION: 3.75,  # was 15.0
    },
    EventTier.WORLD_CUP: {
        RoundType.FINAL: 12.0,  # was 32.0
        RoundType.SEMI: 24.0,  # unchanged — no semi-round data in source DB
        RoundType.QUALIFICATION: 4.5,  # was 12.0
    },
    EventTier.CONTINENTAL: {
        RoundType.FINAL: 6.0,  # was 24.0
        RoundType.SEMI: 18.0,  # unchanged — no semi-round data in source DB
        RoundType.QUALIFICATION: 4.5,  # was 9.0
    },
}


def _default_k_factors() -> dict[EventTier, dict[RoundType, float]]:
    """Return a deep copy of the default K-factor table.

    Used as the ``default_factory`` for :class:`EloConfig` so each config
    gets its own nested-dict instance and callers can mutate without
    accidentally aliasing the module-level default.
    """
    return copy.deepcopy(_DEFAULT_K_FACTORS)


# ---------------------------------------------------------------------------
# EloConfig — single source of truth for tunable engine knobs (Issue #83 T3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EloConfig:
    """Immutable bag of all tunable knobs read by the ELO engine.

    Pass a custom instance to :func:`calculate_round_updates` (and friends)
    to run with alternative parameters — e.g. for K-factor regrid sweeps
    (#80) — without monkey-patching module globals.

    Fields
    ------

    Cold-start
        * ``provisional_threshold`` — n_events below which an athlete is
          flagged ``provisional`` for UI badge purposes only (Glicko-2
          handles the cold-start update magnitude via high φ).

    Margin-of-victory
        * ``margin_cap`` — ceiling on the base MOV multiplier
          ``min(1 + gap/max_gap, margin_cap)``.
        * ``boulder_margin_max_gap`` — denominator for Boulder MOV.
        * ``speed_max_gap_seconds`` — denominator for Speed MOV (lower-is-
          better times).

    Glicko-2 (Issue #51)
        * ``glicko2_sigma_inactivity`` — Wiener-process σ that drives φ
          inflation during competitive sabbaticals (display scale).
        * ``glicko2_tau`` — system constant τ, recommended [0.3, 1.2];
          0.5 is a moderate default. Constrains how fast the per-round
          volatility refit (Issue #81 Illinois iteration) can move σ.

    σ clamping
        * ``sigma_floor`` — display-scale RD floor (≈ φ=0.29; established
          athlete).
        * ``sigma_ceiling`` — display-scale RD ceiling (≈ φ=2.01; cold
          start).

    σ field-size normalization (Issue #95)
        * ``sigma_field_normalization_exponent`` — damps the accumulated
          Glicko-2 variance evidence (``v_inv``) by ``max(n−1, 1) ** exponent``
          where ``n`` is the round's field size. A Plackett-Luce-decomposed
          ranking yields (n−1) pairwise "games" per athlete, but a single
          multi-athlete ranking is **not** (n−1) independent Glicko-2 games —
          left undamped it collapses σ straight to the floor after one event
          (mirroring the μ over-counting that ``pair_k = base_k/(n−1)`` already
          fixes). ``1.0`` (default) = full normalization (one round ≈ one game
          of evidence); ``0.0`` = the old over-counting behaviour (escape hatch
          for ablation / regression tests).

    MOV gap-conditioning (Issue #53, 538-style)
        * ``mov_rating_scale`` — Δμ at which the damping factor becomes
          ``softening / (1 + softening)``.
        * ``mov_softening`` — controls how fast damping kicks in; lower
          values damp harsher.

    K-factor table
        * ``k_factor_table`` — nested dict
          ``{EventTier: {RoundType: float}}``. Defaults to a fresh deep
          copy of ``_DEFAULT_K_FACTORS``.
    """

    # Cold-start
    provisional_threshold: int = 3

    # Margin-of-victory
    # margin_cap bumped 1.5 → 1.7 on 2026-05-28 per the #85 MOV grid sweep
    # (docs/MOV_REGRID_REPORT.md, data/grid_search/mov/sweep_2026-05-28.json).
    margin_cap: float = 1.7  # was 1.5
    boulder_margin_max_gap: float = 1000.0
    speed_max_gap_seconds: float = 2.0

    # Speed bracket-native model (Issue #56)
    # ``speed_tie_epsilon_seconds`` — gap below which two Speed times count as
    # a tie under the Davidson model. Default 0.01 s is finer than the IFSC's
    # millisecond-resolution timing system; in practice ties at this level only
    # fire on rare exact-match data.
    # ``speed_davidson_nu`` — Davidson (1970) tie parameter. ν = 0 collapses to
    # vanilla Bradley-Terry (no tie mass); ν > 0 allocates probability to the
    # tie outcome that peaks at μ_a == μ_b. A small default of 0.1 keeps the
    # tie mass low for typical Δμ but lets ε-close times produce a non-trivial
    # update.
    speed_tie_epsilon_seconds: float = 0.01
    speed_davidson_nu: float = 0.1

    # Glicko-2
    # σ_inactivity bumped 5.0 → 25.0 on 2026-05-27 per the #89 investigation:
    # at 5.0 the Wiener-process inflation was too weak to produce visible σ
    # growth even over multi-year sabbaticals (12-month gap from σ=200 only
    # reaches σ≈207.4, indistinguishable from noise). 25.0 produces a modest
    # but plausible signal (12-month gap σ=200 → σ≈217.9). A proper grid
    # sweep over [10, 15, 25, 50] against the backtest harness is filed as
    # follow-up to this PR; 25.0 is a defensible starting point per the
    # investigation doc.
    glicko2_sigma_inactivity: float = 25.0
    glicko2_tau: float = 0.5

    # σ clamping (display scale)
    sigma_floor: float = 50.0
    sigma_ceiling: float = 350.0

    # σ field-size normalization (Issue #95). Divide accumulated v_inv by
    # max(n-1, 1) ** exponent so one multi-athlete round contributes ≈ one
    # Glicko-2 game of evidence rather than (n-1). 1.0 = full normalization,
    # 0.0 = legacy over-counting (collapses σ to the floor after one event).
    sigma_field_normalization_exponent: float = 1.0

    # MOV gap-conditioning (Issue #53). Values locked 2026-05-28 by the #85
    # grid sweep: (rating_scale, softening, margin_cap) = (200, 2.2, 1.7) was
    # the in-band winner — highest top-3 hit (0.8676 vs 0.8382 baseline) AND
    # lowest podium log-loss (0.2516 vs 0.2733) with μ-p95=2045 in the
    # [1900, 2200] elite band. See docs/MOV_REGRID_REPORT.md and
    # data/grid_search/mov/sweep_2026-05-28.json.
    mov_rating_scale: float = 200.0  # was 400.0
    mov_softening: float = 2.2  # unchanged

    # K-factor table
    k_factor_table: dict[EventTier, dict[RoundType, float]] = field(
        default_factory=_default_k_factors
    )

    # Tournament Participation Bonus (Issue #90 — Gap 1 from #88)
    # Per-tier ordered list of μ-credit values for top-K finishers (rank 1
    # first). See ``compute_tournament_participation_bonus`` for the
    # zero-sum debit math.
    tpb_table: dict[EventTier, list[float]] = field(default_factory=_default_tpb_table)

    # G-Elo bucketed MOV (Issue #84 — Szczecinski 2022 benchmark variant).
    # When ``None`` (the default), the engine uses the production continuous
    # MOV formula ``min(1 + gap/max_gap, margin_cap)``. When set to a mapping
    # of ``Discipline → [(upper_bound_exclusive, multiplier), ...]``, the
    # MOV helpers dispatch to :func:`climbing_elo.engine.gelo.compute_gelo_margin_multiplier`
    # instead. This is an opt-in benchmark knob — the default config keeps
    # production behaviour byte-identical. See ``engine/gelo.py`` for the
    # default bucket tables and rationale.
    gelo_buckets: (
        dict[Discipline, "list[tuple[float, float]] | tuple[tuple[float, float], ...]"]
        | None
    ) = None


DEFAULT_CONFIG = EloConfig()


# ---------------------------------------------------------------------------
# Engine version stamping (joyful-swinging-map plan)
# ---------------------------------------------------------------------------
#
# Frozen-forecast rows (``EventForecast`` / ``EventForecastScore``) carry a
# stable identifier of the engine that produced them. The tag is the SHA-256
# (truncated to 12 hex chars) of the canonical JSON of the ``EloConfig``
# field tuple, concatenated with the short git SHA — so a knob change OR a
# code change yields a fresh version string. The git SHA is read once at
# module import via ``subprocess`` and cached; a non-git environment (e.g.
# a Vercel build that strips ``.git``) falls back to ``"unknown"`` cleanly.


def _read_git_sha() -> str:
    """Return short git SHA of HEAD, or ``"unknown"`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


# Cached at import time per the plan: forecast snapshotting may run many
# times in a single process and we don't want to re-fork a git subprocess
# for every call. A stale SHA across long-lived workers is fine — engine
# behaviour is fully captured by the config-hash half of the tag.
_GIT_SHA: str = _read_git_sha()


def engine_version_tag(config: EloConfig | None = None) -> str:
    """Stable short identifier for the engine version that produced a forecast.

    Returns ``f"{sha256_12(config)}-{short_git_sha}"`` — a 12-char hash of
    the canonical-JSON-serialized config field tuple, joined to the
    module-cached short git SHA (or ``"unknown"`` outside a git checkout).

    Parameters
    ----------
    config:
        The :class:`EloConfig` whose hash is computed. Defaults to
        :data:`DEFAULT_CONFIG`.

    Notes
    -----
    Deterministic and side-effect-free. Used by the forecast snapshot job to
    stamp ``EventForecast.engine_version`` and ``EventForecastScore.engine_version``
    so changes to K factors, σ_inactivity, MOV, TPB are all reflected. Frozen
    rows are never re-snapshotted across version bumps; the new version applies
    only to events that don't yet have a row for that version.
    """
    cfg = config or DEFAULT_CONFIG
    payload = json.dumps(
        dataclasses.asdict(cfg),
        sort_keys=True,
        default=str,
    )
    config_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{config_hash}-{_GIT_SHA}"


# ---------------------------------------------------------------------------
# Back-compat re-exports (Issue #83 Target 3)
# ---------------------------------------------------------------------------
#
# These exist so callers doing ``from climbing_elo.engine.elo import
# MARGIN_CAP`` keep working. New code should accept ``config: EloConfig`` and
# read fields from there. These constants are **pinned at import time** — they
# are not a live view onto an active config. Mutating them does NOT change
# engine behaviour any more (post-#83 Target 3). See ``EloConfig`` docstring.

PROVISIONAL_THRESHOLD = DEFAULT_CONFIG.provisional_threshold
MARGIN_CAP = DEFAULT_CONFIG.margin_cap
BOULDER_MARGIN_MAX_GAP = DEFAULT_CONFIG.boulder_margin_max_gap
SPEED_MAX_GAP_SECONDS = DEFAULT_CONFIG.speed_max_gap_seconds
GLICKO2_SIGMA_INACTIVITY = DEFAULT_CONFIG.glicko2_sigma_inactivity
GLICKO2_TAU = DEFAULT_CONFIG.glicko2_tau
SIGMA_FLOOR = DEFAULT_CONFIG.sigma_floor
SIGMA_CEILING = DEFAULT_CONFIG.sigma_ceiling
MOV_RATING_SCALE = DEFAULT_CONFIG.mov_rating_scale
MOV_SOFTENING = DEFAULT_CONFIG.mov_softening
K_FACTOR_TABLE = DEFAULT_CONFIG.k_factor_table
TPB_TABLE = DEFAULT_CONFIG.tpb_table

PHI_FLOOR = SIGMA_FLOOR / GLICKO2_SCALE
PHI_CEILING = SIGMA_CEILING / GLICKO2_SCALE


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


def _gap_conditioning_factor(
    rating_gap: float, config: EloConfig = DEFAULT_CONFIG
) -> float:
    """538-style asymmetric damping factor for the MOV multiplier.

    ``factor = softening / (max(rating_gap, 0)/rating_scale + softening)``

    Returns 1.0 when ``rating_gap <= 0`` (upset side — full MOV bonus retained)
    and shrinks toward 0 as the favourite's rating advantage grows. Reads
    ``mov_softening`` and ``mov_rating_scale`` from *config*.
    """
    positive_gap = max(rating_gap, 0.0)
    return config.mov_softening / (
        positive_gap / config.mov_rating_scale + config.mov_softening
    )


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
    # Updated Glicko-2 internal-scale volatility (Issue #81). The backfill
    # writes this back to the in-memory AthleteRating cache so it evolves
    # across the run; it has no DB column.
    volatility_after: float = GLICKO2_DEFAULT_VOLATILITY


@dataclass
class AthleteRating:
    athlete_id: int
    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA  # display-scale RD
    n_events: int = 0
    last_event_at: date | None = None
    provisional: bool = True
    # Glicko-2 internal-scale volatility σ (Issue #81). Persisted in-memory
    # across a backfill run; not stored in the DB (no schema change), so each
    # fresh backfill re-seeds at GLICKO2_DEFAULT_VOLATILITY. Evolved per round
    # by the full Illinois volatility iteration.
    volatility: float = GLICKO2_DEFAULT_VOLATILITY


@dataclass
class TPBContribution:
    """Per-athlete μ delta from the Tournament Participation Bonus (Issue #90)."""

    athlete_id: int
    rank: int
    gross_bonus: float  # tier-table credit for top-K, 0.0 for the rest
    debit: float  # share of total bonus pulled back, applied to every athlete
    delta: float  # gross_bonus - debit; sums to zero across the event


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
    config: EloConfig = DEFAULT_CONFIG,
) -> float:
    """Inflate σ (display-scale RD) for inactivity per Glicko-2's Wiener-process model.

    Calendar-time semantics — months since last event (decision 1 in module
    docstring). Returns the new display-scale RD, clamped at
    ``config.sigma_ceiling``.

    Formula (display scale; we keep the units consistent across the calculation
    rather than round-tripping through Glickman's internal scale):
        σ_new² = σ_old² + σ_inactivity² · months_inactive

    With the default ``σ_inactivity = 25.0`` (post-#89 bump from 5.0), a
    12-month sabbatical from σ=200 produces σ_new ≈ 217.9 — a modest but
    visible "this athlete has been away" signal. From σ=100 the same gap
    produces σ_new ≈ 134.6. A fresh athlete at σ=350 stays clamped at the
    ceiling regardless. See ``docs/INVESTIGATION_SIGMA_CEILING.md`` for the
    σ_inactivity sweep rationale.
    """
    if last_event_at is None or current_date <= last_event_at:
        return sigma_display
    days_inactive = (current_date - last_event_at).days
    if days_inactive <= GLICKO2_INACTIVITY_GRACE_DAYS:
        return sigma_display
    months_inactive = days_inactive / GLICKO2_DAYS_PER_MONTH
    sigma_inactivity = config.glicko2_sigma_inactivity
    sigma_new = math.sqrt(
        sigma_display * sigma_display
        + sigma_inactivity * sigma_inactivity * months_inactive
    )
    return min(sigma_new, config.sigma_ceiling)


def glicko2_update_volatility(
    sigma: float,
    phi: float,
    v: float,
    delta: float,
    tau: float,
    epsilon: float = GLICKO2_VOLATILITY_EPSILON,
    max_iter: int = GLICKO2_VOLATILITY_MAX_ITER,
) -> float:
    """Full Glicko-2 Step 5 volatility update via the Illinois algorithm (#81).

    Solves for the new volatility ``σ'`` such that ``f(x) = 0`` where (Glickman
    2013, *Example of the Glicko-2 system*, Step 5)::

        f(x) = e^x (Δ² − φ² − v − e^x) / (2 (φ² + v + e^x)²)  −  (x − ln σ²) / τ²

    using the Illinois variant of regula falsi (the root-finder Glickman
    recommends). All arguments are on the Glicko-2 **internal** scale (φ, σ
    converted from display RD by dividing by ``GLICKO2_SCALE``; μ-derived Δ and
    v likewise internal-scale).

    Args:
        sigma: current volatility σ (internal scale, ~0.06 for a fresh player).
        phi: pre-update rating deviation φ (internal scale).
        v: estimated variance of the rating from game outcomes (internal scale).
        delta: estimated rating improvement Δ = v · Σ g(φ_j)(s_j − E_j).
        tau: system constant τ constraining volatility change over time.
        epsilon: convergence tolerance on the bracket width.
        max_iter: safety cap on Illinois iterations.

    Returns:
        The updated volatility ``σ'`` (internal scale, always > 0).
    """
    a = math.log(sigma * sigma)
    delta_sq = delta * delta
    phi_sq = phi * phi
    tau_sq = tau * tau

    def f(x: float) -> float:
        ex = math.exp(x)
        denom = phi_sq + v + ex
        return (ex * (delta_sq - phi_sq - v - ex)) / (2.0 * denom * denom) - (
            x - a
        ) / tau_sq

    # Initial bracket [A, B] per Glickman Step 5.2.
    big_a = a
    if delta_sq > phi_sq + v:
        big_b = math.log(delta_sq - phi_sq - v)
    else:
        # Expand downward by k·τ until f turns negative.
        k = 1
        while f(a - k * tau) < 0.0 and k < max_iter:
            k += 1
        big_b = a - k * tau

    f_a = f(big_a)
    f_b = f(big_b)

    # Illinois iteration (modified regula falsi).
    iters = 0
    while abs(big_b - big_a) > epsilon and iters < max_iter:
        c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(c)
        if f_c * f_b <= 0.0:
            big_a = big_b
            f_a = f_b
        else:
            f_a = f_a / 2.0  # Illinois weighting halves the stale endpoint
        big_b = c
        f_b = f_c
        iters += 1

    return math.exp(big_a / 2.0)


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


def get_k_factor(
    tier: EventTier,
    round_type: RoundType,
    config: EloConfig = DEFAULT_CONFIG,
) -> float:
    """Look up the base K-factor for a (tier, round) pair.

    Reads ``config.k_factor_table``; pass a custom :class:`EloConfig` to swap
    in an alternative K-factor schedule (e.g. for #80 regrid sweeps) without
    monkey-patching module globals.
    """
    return config.k_factor_table[tier][round_type]


def compute_margin_multiplier(
    score_a: float | None,
    score_b: float | None,
    max_gap: float = 20.0,
    rating_gap: float = 0.0,
    config: EloConfig = DEFAULT_CONFIG,
) -> float:
    """Lead-style margin multiplier with optional 538-style gap conditioning.

    ``rating_gap`` is ``μ_winner − μ_loser`` (pre-update). Defaults to 0.0 for
    backward compatibility; the gap-conditioned version reduces exactly to the
    legacy formula when ``rating_gap == 0``. When the favourite (rating_gap > 0)
    wins, the bonus is damped per :func:`_gap_conditioning_factor`. Upsets
    (rating_gap < 0) keep the full bonus.

    Note: the ``config.margin_cap`` ceiling is enforced on the *base*
    multiplier; the conditioning factor can only shrink it further, never
    above the cap.

    When ``config.gelo_buckets`` is set, dispatch to the G-Elo (Szczecinski
    2022) bucketed MOV variant — see :mod:`climbing_elo.engine.gelo`. The
    default config has ``gelo_buckets=None`` so production behaviour is
    unchanged.
    """
    if score_a is None or score_b is None:
        return 1.0
    # Defence-in-depth (#199): a non-finite score/gap must never propagate into
    # K. Even though the scrapers now reject NaN/Inf at ingest, fall back to a
    # neutral multiplier here so no corrupt value can ever reach a rating delta.
    # compute_boulder_margin_multiplier delegates here, so this covers it too.
    if not (
        math.isfinite(score_a) and math.isfinite(score_b) and math.isfinite(rating_gap)
    ):
        return 1.0
    if config.gelo_buckets is not None:
        from climbing_elo.engine.gelo import compute_gelo_margin_multiplier

        return compute_gelo_margin_multiplier(
            score_a, score_b, Discipline.LEAD, rating_gap=rating_gap, config=config
        )
    gap = abs(score_a - score_b)
    base = min(1.0 + gap / max_gap, config.margin_cap)
    return base * _gap_conditioning_factor(rating_gap, config)


# Regex for the old-format ordinal Boulder score: e.g. "1T2z 3 4" or "2T2 3B4".
# Attempt counts are optional in the lowercase pre-2018 feed (e.g. "0t 4b10");
# kept in sync with normalize_boulder_score (#115). Currently unused.
_OLD_BOULDER_RE = re.compile(
    r"(\d+)[Tt](\d+)[Zz]\s+(\d+)\s+(\d+)"  # "NTMz A B"
    r"|(\d+)[Tt](\d*)\s+(\d+)[Bb](\d*)",  # "Nt[att] M b[att]"
)


def normalize_boulder_score(raw_score: str) -> float | None:
    """Normalize a Boulder raw score to the canonical ordinal scale.

    The whole DB uses the ordinal scale ``tops * 1000 + zones * 100 -
    top_att * 10 - zone_att`` (#117). Two feed formats can be recovered from
    the raw string alone:

    * Post-2018 ``"NTMz A B"`` — e.g. ``"1T2z 3 4"``.
    * Pre-2018 lowercase ``"Nt[att] Mb[att]"`` — attempt counts optional
      (``"0t 4b10"``, ``"0t 0b"`` etc., see #115).

    The **2025+ decimal feed** (``"124.9"``) cannot be normalised from the raw
    string alone — the ordinal value is reconstructed at scrape time from the
    structured ``ascents`` payload (``scraper.ifsc_api._parse_boulder_score``).
    Decimal-only raws therefore return ``None`` here: preserving the ``124.9``
    value would silently mix a 0-125 scale into a 0-6000 corpus and inflate MOV
    saturation (#117 / #84).

    Returns ``None`` for DNF/DNS/empty/decimal-only inputs.
    """
    raw = (raw_score or "").strip()
    if not raw or raw.upper() in ("DNF", "DNS", "-"):
        return None

    m = re.match(r"(\d+)[Tt](\d+)[Zz]\s+(\d+)\s+(\d+)", raw)
    if m:
        tops, zones, top_att, zone_att = (int(x) for x in m.groups())
        return float(tops * 1000 + zones * 100 - top_att * 10 - zone_att)

    # ``Nt[att] M b[att]`` form. The attempt counts are OPTIONAL: the pre-2018
    # lowercase feed omits them for 0-top athletes (e.g. ``"0t 4b10"`` = 0 tops,
    # 4 zones, 10 zone-attempts) and sometimes for 0-zone too (``"0t 0b"``).
    # Without ``\d*``/default-0 those 1,523 real results normalised to ``None``
    # and silently vanished from the boulder field pre-2018 (Issue #115).
    m = re.match(r"(\d+)[Tt](\d*)\s+(\d+)[Bb](\d*)", raw)
    if m:
        tops = int(m.group(1))
        top_att = int(m.group(2) or 0)
        zones = int(m.group(3))
        zone_att = int(m.group(4) or 0)
        return float(tops * 1000 + zones * 100 - top_att * 10 - zone_att)

    return None


def compute_boulder_margin_multiplier(
    score_a: float | None,
    score_b: float | None,
    rating_gap: float = 0.0,
    config: EloConfig = DEFAULT_CONFIG,
) -> float:
    """Margin multiplier for Boulder discipline (538-style gap-conditioned).

    Reads ``boulder_margin_max_gap`` from *config*. When ``config.gelo_buckets``
    is set, dispatches to the Szczecinski bucketed MOV (see
    :mod:`climbing_elo.engine.gelo`).
    """
    if score_a is None or score_b is None:
        return 1.0
    if config.gelo_buckets is not None:
        from climbing_elo.engine.gelo import compute_gelo_margin_multiplier

        return compute_gelo_margin_multiplier(
            score_a,
            score_b,
            Discipline.BOULDER,
            rating_gap=rating_gap,
            config=config,
        )
    return compute_margin_multiplier(
        score_a,
        score_b,
        max_gap=config.boulder_margin_max_gap,
        rating_gap=rating_gap,
        config=config,
    )


def compute_speed_margin_multiplier(
    winner_time: float | None,
    loser_time: float | None,
    rating_gap: float = 0.0,
    config: EloConfig = DEFAULT_CONFIG,
) -> float:
    """Margin multiplier for Speed discipline (times in seconds, lower is better).

    Gap-conditioned in the same fashion as Lead/Boulder — favourite wins get
    damped, upsets keep the full bonus. See :func:`compute_margin_multiplier`.
    Reads ``speed_max_gap_seconds`` and ``margin_cap`` from *config*.

    When ``config.gelo_buckets`` is set, dispatches to the Szczecinski
    bucketed MOV (see :mod:`climbing_elo.engine.gelo`).
    """
    if winner_time is None or loser_time is None:
        return 1.0
    # Defence-in-depth (#199): reject non-finite inputs (Speed does not delegate
    # to compute_margin_multiplier) so no corrupt value reaches a rating delta.
    if not (
        math.isfinite(winner_time)
        and math.isfinite(loser_time)
        and math.isfinite(rating_gap)
    ):
        return 1.0
    if config.gelo_buckets is not None:
        from climbing_elo.engine.gelo import compute_gelo_margin_multiplier

        return compute_gelo_margin_multiplier(
            winner_time,
            loser_time,
            Discipline.SPEED,
            rating_gap=rating_gap,
            config=config,
        )
    gap = abs(loser_time - winner_time)
    base = min(1.0 + gap / config.speed_max_gap_seconds, config.margin_cap)
    return base * _gap_conditioning_factor(rating_gap, config)


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
    config: EloConfig = DEFAULT_CONFIG,
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
    3. Per-athlete φ update (full Glicko-2 Step 5-7, Issue #81):
          σ' = Illinois-refit volatility from (v, Δ)
          φ* = sqrt(φ_inflated² + σ'²)
          1/φ_new² = 1/φ*² + 1/v   (= 1/φ*² + v_inv after field-normalization)
       Clamped to [sigma_floor, sigma_ceiling] on the display scale.

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

    Speed dispatch (Issue #56)
    --------------------------

    When ``discipline == Discipline.SPEED`` this function delegates to
    :func:`climbing_elo.engine.speed.calculate_speed_round_updates`, which
    processes only *adjacent-rank* head-to-head matchups (the closest
    approximation we can recover to the actual elimination bracket without a
    schema change) and uses Davidson 1970 tie handling on near-ε times. The
    Lead/Boulder path below is untouched.
    """
    if discipline == Discipline.SPEED:
        # Local import to avoid a circular dependency at module load time
        # (engine.speed imports primitives from engine.elo).
        from climbing_elo.engine.speed import calculate_speed_round_updates

        return calculate_speed_round_updates(
            results,
            ratings,
            event_tier,
            round_type,
            event_date,
            config,
        )

    active = [r for r in results if not r.dns]
    if len(active) < 2:
        return []

    base_k = get_k_factor(event_tier, round_type, config)

    # 1) Inflate φ for inactivity, store the inflated display-scale RDs.
    sigma_inflated: dict[int, float] = {}
    for res in active:
        rating = ratings.get(res.athlete_id, AthleteRating(athlete_id=res.athlete_id))
        sigma_inflated[res.athlete_id] = glicko2_inflate_phi(
            rating.sigma, rating.last_event_at, event_date, config
        )

    deltas: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    v_inv_sum: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    # Glicko-2 Step 4 accumulator: Σ_j g(φ_j) · (s_j − E_j) per athlete, on the
    # internal scale (#81 full volatility iteration). Feeds Δ = v · this_sum.
    delta_terms_sum: dict[int, float] = {r.athlete_id: 0.0 for r in active}
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
                    config=config,
                )
            elif discipline == Discipline.BOULDER:
                margin_mult = compute_boulder_margin_multiplier(
                    res_i.score_normalized,
                    res_j.score_normalized,
                    rating_gap=rating_gap,
                    config=config,
                )
            else:
                margin_mult = compute_margin_multiplier(
                    res_i.score_normalized,
                    res_j.score_normalized,
                    rating_gap=rating_gap,
                    config=config,
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

            # Glicko-2 Step 4 outcome-weighted residual: g(φ_opp)·(s − E).
            # Winner i: s=1; loser j: s=0. Drives the Δ estimate that the full
            # volatility iteration (#81) consumes.
            delta_terms_sum[res_i.athlete_id] += g_phi_j * (1.0 - e_i)
            delta_terms_sum[res_j.athlete_id] += g_phi_i * (0.0 - e_j)

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
        volatility = rating.volatility or GLICKO2_DEFAULT_VOLATILITY

        # Field-size normalization (Issue #95): the v_inv / delta-term
        # accumulators sum one term per pairwise comparison, i.e. (n−1)
        # Glicko-2 "games" of evidence in an n-athlete round. But a single
        # Plackett-Luce-decomposed ranking is NOT (n−1) independent games —
        # left undamped it collapses σ to the floor after one event. Divide the
        # accumulated evidence by max(n−1, 1) ** exponent so one round
        # contributes ≈ one game (exponent=1.0, the default). This mirrors the
        # μ-side ``pair_k = base_k/(n−1)`` normalization. exponent=0.0 restores
        # the legacy over-counting behaviour (escape hatch / ablation knob).
        v_inv = v_inv_sum.get(aid, 0.0)
        delta_terms = delta_terms_sum.get(aid, 0.0)
        n_field = len(active)
        if n_field > 1 and config.sigma_field_normalization_exponent:
            norm = float(n_field - 1) ** config.sigma_field_normalization_exponent
            v_inv /= norm
            delta_terms /= norm

        if v_inv > 0.0:
            # Full Glicko-2 update (Issue #81): refit the volatility σ' via the
            # Illinois root-find (Step 5), inflate φ by the new volatility
            # (Step 6: φ* = sqrt(φ² + σ'²)), then shrink with the round's
            # variance evidence (Step 7: 1/φ'² = 1/φ*² + 1/v).
            v = 1.0 / v_inv
            delta = v * delta_terms
            volatility_after = glicko2_update_volatility(
                volatility,
                phi_internal,
                v,
                delta,
                config.glicko2_tau,
            )
            phi_star_sq = (
                phi_internal * phi_internal + volatility_after * volatility_after
            )
            inv_phi_sq_new = 1.0 / phi_star_sq + v_inv
            phi_new = 1.0 / math.sqrt(inv_phi_sq_new)
        else:
            # Zero-game rating period for this athlete (e.g. all ties): no
            # variance evidence, so only the volatility inflation acts on φ and
            # the volatility itself is unchanged.
            volatility_after = volatility
            phi_new = math.sqrt(
                phi_internal * phi_internal + volatility_after * volatility_after
            )

        sigma_after_display = phi_new * GLICKO2_SCALE
        # Clamp to display-scale floor/ceiling (read from config so callers
        # can ablate σ decay by setting floor == ceiling == DEFAULT_SIGMA).
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
                volatility_after=volatility_after,
            )
        )

    return updates


# ---------------------------------------------------------------------------
# Tournament Participation Bonus (Issue #90 — Gap 1 from #88)
# ---------------------------------------------------------------------------
#
# A tier-weighted, zero-sum μ credit applied *per event* on top of the
# pairwise round updates. The pairwise math is unchanged — TPB is layered on
# afterward by the backfill, written as a separate ``RatingHistory`` row with
# ``kind='tpb'`` whose ``round_id`` points at the event's FINAL round.
#
# Why a separate layer (not folded into K)?
#   1. Keeps the pairwise zero-sum invariant clean — pair tests still pass.
#   2. Lets the breakdown page render TPB as its own line: easier to explain
#      and easier to ablate.
#   3. Backtests can A/B turn TPB on/off without re-running the pair update.
#
# Zero-sum mechanism
#   gross_bonus[r] = tpb_table[tier][r-1]   (0.0 if r > len(table))
#   total_bonus    = Σ gross_bonus
#   debit          = total_bonus / N         (N = number of participants)
#   delta          = gross_bonus - debit     (sums to zero across the field)
#
# DNS athletes are excluded from the field count entirely. They neither
# receive the bonus nor share in the debit.


def compute_tournament_participation_bonus(
    results: list[AthleteResult],
    event_tier: EventTier,
    config: EloConfig = DEFAULT_CONFIG,
) -> list[TPBContribution]:
    """Per-athlete μ deltas for the tier-weighted Tournament Participation Bonus.

    Top-K finishers (per ``config.tpb_table[event_tier]``) receive a gross μ
    credit; the total credit is then debited uniformly across every
    participant so the sum of deltas is exactly zero.

    Args:
        results: Per-athlete final standings for the event (use the rank from
            the FINAL round). DNS athletes are silently excluded.
        event_tier: One of :class:`EventTier`. Determines the tier curve.
        config: Engine config providing ``tpb_table``.

    Returns:
        One :class:`TPBContribution` per non-DNS athlete, ordered by rank.
        Returns an empty list if fewer than 2 athletes finished — there's no
        "field" for a 1-athlete event, so the bonus is meaningless.
    """
    active = [r for r in results if not r.dns]
    if len(active) < 2:
        return []

    table = config.tpb_table.get(event_tier, [])
    n_field = len(active)

    gross: dict[int, float] = {}
    for res in active:
        idx = res.rank - 1
        gross[res.athlete_id] = table[idx] if 0 <= idx < len(table) else 0.0

    total_bonus = sum(gross.values())
    debit = total_bonus / n_field if n_field > 0 else 0.0

    contributions = [
        TPBContribution(
            athlete_id=res.athlete_id,
            rank=res.rank,
            gross_bonus=gross[res.athlete_id],
            debit=debit,
            delta=gross[res.athlete_id] - debit,
        )
        for res in active
    ]
    return sorted(contributions, key=lambda c: c.rank)
