"""``g2pl`` — canonical Glicko-2 over Plackett-Luce pairs (challenger engine).

Design doc: ``docs/PLAN_CHALLENGER_G2PL.md``. Origin: the Phase-1/2 multi-agent
audit (issues #174–#190).

Why this module exists
----------------------

The incumbent (:mod:`climbing_elo.engine.elo`) is a *hybrid*: Glicko-2's φ
machinery grafted onto a constant-K pairwise ELO core. Three separate
step-size mechanisms stack up there — the K table, the ``g(φ_opponent)``
weight, and the MOV multiplier — and the μ update itself is still flat
``K·(1 − E)`` accumulation. When #51 removed the ``pair_k = base_k/(n − 1)``
normalization it did so on the grounds that "Glicko-2's v is already the right
normalizer", but the canonical variance-scaled μ update that provides that
normalization was never implemented (#174). The measured consequence is that
per-round μ deltas scale with entry-list size (2.8×–5.2× for identical
relative performance) — a per-cell K regrid can rescale the mean but cannot
remove within-cell attendance bias.

``g2pl`` finishes #51 properly:

* One update, Glickman's canonical step 7 — ``μ' = μ + φ'² · Δ̃``.
* One step-size knob, the **importance weight** ``w(tier, round_type)``,
  seeded from the incumbent's ``_DEFAULT_K_FACTORS`` cell ratios so relative
  tier/round importance carries over.
* No TPB (#175 must be fixed and re-measured before it can be trusted).
* One inactivity mechanism — the existing :func:`glicko2_inflate_phi`.

Everything Glicko-2 (``g(φ)``, expected score, the Illinois volatility
iteration, the scale constants, the σ clamps) is **imported** from
:mod:`climbing_elo.engine.elo`; none of it is reimplemented here.

Scope
-----

LEAD and BOULDER only. Speed keeps the incumbent bracket-native model
(#184/#56) — :func:`calculate_g2pl_round_updates` raises for
``Discipline.SPEED`` and :class:`G2PLEngine` delegates that discipline back to
the incumbent so a mixed-discipline backtest still runs.

Known trade-off (stated, not hidden)
------------------------------------

Canonical Glicko-2 is **not zero-sum across a round**: each athlete's μ step
is scaled by their own φ'², so a field of mixed-certainty athletes does not
cancel exactly. The incumbent's exact zero-sum invariant is therefore replaced
by a monitored *population-drift* metric — :func:`mu_drift` — which the
evaluation path can watch per season (design-doc alarm threshold: ±5 μ/season).

Field-size normalization (documented deviation from the design doc)
-------------------------------------------------------------------

The design doc writes the accumulators as bare sums over opponents::

    v_i⁻¹ = Σ_j w_r · g(φ_j)² · E_ij(1 − E_ij)
    Δ̃_i  = Σ_j w_r · g(φ_j) · (s_ij − E_ij)

Taken literally, that treats an n-athlete round as (n − 1) *independent*
Glicko-2 games, and the canonical update is only field-size independent in the
asymptotic regime where the accumulated evidence dominates the prior precision
``1/φ*²``. At realistic climbing parameters it does not: a numeric probe of the
winner's Δμ at n=10 vs n=80 (equal σ, mean-matched opponent field) gives ratios
of **1.35×–8.4×** depending on ``(w, σ)`` — i.e. invariant test 1 would fail.

So the accumulators are divided by ``(n − 1) ** field_normalization_exponent``
(default ``1.0``), which is exactly the statement *"one round is one weighted
observation of a ranking, not (n − 1) independent games"* — the same reasoning
that already justifies ``EloConfig.sigma_field_normalization_exponent`` (#95)
on the incumbent's σ path, applied here to the single shared μ/σ path. With
the default exponent the same probe lands at **1.4%–3.6%** spread across
n ∈ {10, 20, 40, 80}. Setting ``field_normalization_exponent=0.0`` restores
the design doc's literal formula for ablation.

Note this only rescales ``w`` by ``1/(n − 1)``; the relative importance
ordering of the ``(tier, round_type)`` cells is untouched.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import (
    DEFAULT_CONFIG,
    DEFAULT_MU,
    DEFAULT_SIGMA,
    GLICKO2_DEFAULT_VOLATILITY,
    GLICKO2_SCALE,
    AthleteRating,
    AthleteResult,
    EloConfig,
    PairContribution,
    RatingUpdate,
    _DEFAULT_K_FACTORS,
    calculate_round_updates,
    compute_boulder_margin_multiplier,
    compute_margin_multiplier,
    glicko2_expected_score,
    glicko2_g,
    glicko2_inflate_phi,
    glicko2_update_volatility,
)
from climbing_elo.engine.evaluation import (
    RatingForecast,
    register_variant,
)
from climbing_elo.models import (
    Discipline,
    Event,
    EventTier,
    Result,
    RoundType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Importance weights — the replacement for the K table
# ---------------------------------------------------------------------------
#
# Seed rule from the design doc: ``w(tier, round) = K(tier, round) / max K``.
# The largest incumbent cell (Olympics FINAL = 48.0) maps to 1.0 and every
# other cell keeps its *relative* importance. Derived programmatically so a
# future K change (or a joint w sweep replacing this seed) has exactly one
# place to look.


def _seed_importance_weights() -> dict[EventTier, dict[RoundType, float]]:
    """Importance-weight table seeded from ``_DEFAULT_K_FACTORS`` ratios."""
    max_k = max(k for row in _DEFAULT_K_FACTORS.values() for k in row.values())
    return {
        tier: {rt: k / max_k for rt, k in row.items()}
        for tier, row in _DEFAULT_K_FACTORS.items()
    }


#: ``{EventTier: {RoundType: w}}`` — normalised so ``max(w) == 1.0``.
DEFAULT_IMPORTANCE_WEIGHTS: dict[EventTier, dict[RoundType, float]] = (
    _seed_importance_weights()
)

#: Fallback weight for a ``(tier, round_type)`` cell missing from the table.
FALLBACK_IMPORTANCE_WEIGHT = 0.25

#: Valid values for :attr:`G2PLConfig.mov_mode`.
MOV_MODES = ("off", "margin")

#: Disciplines ``g2pl`` implements. Speed stays on the incumbent (#184/#56).
SUPPORTED_DISCIPLINES = (Discipline.LEAD, Discipline.BOULDER)


def _default_importance_weights() -> dict[EventTier, dict[RoundType, float]]:
    return {tier: dict(row) for tier, row in DEFAULT_IMPORTANCE_WEIGHTS.items()}


@dataclass(frozen=True)
class G2PLConfig:
    """Tunable knobs for the ``g2pl`` challenger.

    Fields
    ------
    elo:
        The incumbent :class:`~climbing_elo.engine.elo.EloConfig`, used purely
        as the source of the *shared* Glicko-2 constants — ``glicko2_tau``,
        ``glicko2_sigma_inactivity``, ``sigma_floor`` / ``sigma_ceiling`` and
        the MOV shape parameters (``margin_cap``, ``mov_rating_scale``,
        ``mov_softening``, ``boulder_margin_max_gap``). Its ``k_factor_table``
        is deliberately **not** consulted: ``g2pl`` replaces K with
        :attr:`importance_weights`.
    importance_weights:
        ``w(tier, round_type)`` — the single step-size table. Defaults to
        :data:`DEFAULT_IMPORTANCE_WEIGHTS`.
    mov_mode:
        ``"off"`` (v1 default) — pure outcome ``s ∈ {0, ½, 1}``.
        ``"margin"`` — margin-adjusted outcome ``s* = ½ + (s − ½)·m(gap)``
        with ``m`` the existing gap-conditioned multiplier mapped into
        ``[0, 1]``; ``s*_ij + s*_ji = 1`` holds by construction. This pairing
        *is* the #84 A/B, run on one axis instead of three.
    field_normalization_exponent:
        Divides both accumulators by ``(n − 1) ** exponent``. ``1.0`` (default)
        = one round is one weighted observation; ``0.0`` = the design doc's
        literal per-pair sum (ablation only — fails the field-size invariant).
        See the module docstring.
    """

    elo: EloConfig = DEFAULT_CONFIG
    importance_weights: dict[EventTier, dict[RoundType, float]] = field(
        default_factory=_default_importance_weights
    )
    mov_mode: str = "off"
    field_normalization_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.mov_mode not in MOV_MODES:
            raise ValueError(
                f"mov_mode must be one of {MOV_MODES!r}, got {self.mov_mode!r}"
            )


DEFAULT_G2PL_CONFIG = G2PLConfig()


def get_importance_weight(
    tier: EventTier,
    round_type: RoundType,
    config: G2PLConfig = DEFAULT_G2PL_CONFIG,
) -> float:
    """Look up ``w(tier, round_type)``, falling back for unmapped cells."""
    return config.importance_weights.get(tier, {}).get(
        round_type, FALLBACK_IMPORTANCE_WEIGHT
    )


# ---------------------------------------------------------------------------
# Margin-adjusted outcome (mov_mode="margin")
# ---------------------------------------------------------------------------


def margin_adjusted_outcome(s: float, multiplier: float, margin_cap: float) -> float:
    """Map an outcome ``s`` and a MOV ``multiplier`` to ``s*`` in ``[0, 1]``.

    ``s* = ½ + (s − ½) · m`` with ``m = clamp(multiplier / margin_cap, 0, 1)``.

    Two properties matter:

    * **Anti-symmetry is preserved** — the pair shares one ``m``, so
      ``s*_ij + s*_ji = 1 + (s_ij + s_ji − 1)·m = 1``.
    * **A tie stays a tie** — ``s = ½`` is a fixed point for every ``m``.

    Dividing by ``margin_cap`` (rather than rescaling ``[1, cap] → [0, 1]``)
    is deliberate: a zero-margin win must still count as a win, only a
    *damped* one. A maximal-margin, peer-versus-peer win reaches ``s* = 1``;
    the #53 gap-conditioning damp shrinks ``m`` for favourite-side blowouts,
    which now shows up as partial credit on the outcome instead of as a third
    multiplier on the step size.
    """
    m = multiplier / margin_cap if margin_cap > 0 else 1.0
    m = min(1.0, max(0.0, m))
    return 0.5 + (s - 0.5) * m


def _pair_margin_multiplier(
    winner: AthleteResult,
    loser: AthleteResult,
    rating_gap: float,
    discipline: Discipline,
    config: G2PLConfig,
) -> float:
    """MOV multiplier for one ordered pair, or 1.0 when MOV is off."""
    if config.mov_mode == "off":
        return 1.0
    if winner.dnf:
        # Matches the incumbent: a DNF "win" carries no margin information.
        return 1.0
    if discipline == Discipline.BOULDER:
        return compute_boulder_margin_multiplier(
            winner.score_normalized,
            loser.score_normalized,
            rating_gap=rating_gap,
            config=config.elo,
        )
    return compute_margin_multiplier(
        winner.score_normalized,
        loser.score_normalized,
        rating_gap=rating_gap,
        config=config.elo,
    )


# ---------------------------------------------------------------------------
# Round update
# ---------------------------------------------------------------------------


def calculate_g2pl_round_updates(
    results: list[AthleteResult],
    ratings: dict[int, AthleteRating],
    event_tier: EventTier,
    round_type: RoundType,
    event_date: date,
    discipline: Discipline = Discipline.LEAD,
    config: G2PLConfig = DEFAULT_G2PL_CONFIG,
    collect_pairs: bool = True,
) -> list[RatingUpdate]:
    """Canonical Glicko-2 update over the Plackett-Luce pairwise decomposition.

    Algorithm (design doc §Specification)
    -------------------------------------

    1. **Inactivity** — :func:`glicko2_inflate_phi` on the athlete's stored σ
       at round time. One mechanism, unchanged from the incumbent.
    2. Work on the Glicko-2 internal scale (``GLICKO2_SCALE``).
    3. For athlete *i* against every other non-DNS athlete *j*::

           E_ij   = 1/(1 + exp(−g(φ_j)(μ_i − μ_j)))
           v_i⁻¹  = Σ_j w · g(φ_j)² · E_ij(1 − E_ij)
           Δ̃_i    = Σ_j w · g(φ_j) · (s_ij − E_ij)

       ``s_ij`` is 1 / ½ / 0 for finished-ahead / tied / finished-behind —
       the same decomposition ``engine/backfill.py`` feeds the incumbent,
       except that ties are scored ½ instead of being dropped. Both
       accumulators are then divided by ``(n − 1) ** exponent`` (see the
       module docstring).
    4. **Volatility** — the existing Illinois iteration with ``Δ = v · Δ̃``.
    5. ``φ* = sqrt(φ² + σ'²)`` then ``φ'⁻² = φ*⁻² + v⁻¹``.
    6. **μ update** — ``μ' = μ + φ'² · Δ̃`` (Glickman step 7). This is the
       #174 fix: the step size is variance-scaled rather than flat-K.
    7. Back to display scale, σ clamped to ``[sigma_floor, sigma_ceiling]``.

    Differences from :func:`~climbing_elo.engine.elo.calculate_round_updates`

    * No ``k_pair = min(k_eff_i, k_eff_j)`` — there is no per-pair K.
    * MOV is either absent (``mov_mode="off"``) or folded into the *outcome*
      (``"margin"``), never into the step size.
    * Not zero-sum on μ — see :func:`mu_drift`.

    Args:
        results: The round's standings. DNS rows are dropped entirely.
        ratings: Mutable ``{athlete_id: AthleteRating}`` cache. Read-only here;
            the caller applies the returned updates.
        event_tier: Tier of the parent event (selects ``w``).
        round_type: Round type (selects ``w``).
        event_date: Used for the inactivity inflation.
        discipline: LEAD or BOULDER. SPEED raises ``NotImplementedError``.
        config: :class:`G2PLConfig`.
        collect_pairs: Populate ``RatingUpdate.contributing_pairs``. The
            backtest path passes ``False`` (nothing reads them, and an
            80-athlete qualification round would build ~6 400 objects).

    Returns:
        One :class:`~climbing_elo.engine.elo.RatingUpdate` per non-DNS athlete.
    """
    if discipline == Discipline.SPEED:
        raise NotImplementedError(
            "g2pl covers LEAD and BOULDER only; Speed stays on the incumbent "
            "bracket-native model (see engine/speed.py, issues #184/#56)."
        )

    active = [r for r in results if not r.dns]
    n_field = len(active)
    if n_field < 2:
        return []

    weight = get_importance_weight(event_tier, round_type, config)
    elo_cfg = config.elo

    # --- 1) Inactivity inflation (display scale). ---------------------------
    sigma_inflated: dict[int, float] = {}
    mu_by_id: dict[int, float] = {}
    for res in active:
        rating = ratings.get(res.athlete_id, AthleteRating(athlete_id=res.athlete_id))
        sigma_inflated[res.athlete_id] = glicko2_inflate_phi(
            rating.sigma, rating.last_event_at, event_date, elo_cfg
        )
        mu_by_id[res.athlete_id] = rating.mu

    v_inv_sum: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    delta_tilde_sum: dict[int, float] = {r.athlete_id: 0.0 for r in active}
    # Per-athlete list of (opponent_id, result, expected, actual, term, mov)
    # kept un-scaled; the μ attribution is only known once φ' is computed.
    pair_terms: dict[int, list[tuple[int, str, float, float, float, float]]] = {
        r.athlete_id: [] for r in active
    }

    # --- 2/3) Pairwise accumulation over every unordered pair. --------------
    for idx_i in range(n_field):
        res_i = active[idx_i]
        aid_i = res_i.athlete_id
        mu_i = mu_by_id[aid_i]
        sigma_i = sigma_inflated[aid_i]
        g_i = glicko2_g(sigma_i / GLICKO2_SCALE)

        for idx_j in range(idx_i + 1, n_field):
            res_j = active[idx_j]
            aid_j = res_j.athlete_id
            mu_j = mu_by_id[aid_j]
            sigma_j = sigma_inflated[aid_j]
            g_j = glicko2_g(sigma_j / GLICKO2_SCALE)

            # Each side's expectation uses the *opponent's* g(φ) — so
            # E_ij + E_ji == 1 only when φ_i == φ_j. That asymmetry is
            # canonical Glicko-2, not an approximation to be smoothed over.
            e_i = glicko2_expected_score(mu_i, mu_j, sigma_j)
            e_j = glicko2_expected_score(mu_j, mu_i, sigma_i)

            if res_i.rank < res_j.rank:
                s_i, s_j = 1.0, 0.0
                winner, loser = res_i, res_j
                rating_gap = mu_i - mu_j
            elif res_i.rank > res_j.rank:
                s_i, s_j = 0.0, 1.0
                winner, loser = res_j, res_i
                rating_gap = mu_j - mu_i
            else:
                s_i = s_j = 0.5
                winner = loser = res_i
                rating_gap = 0.0

            mov = 1.0
            if config.mov_mode != "off" and s_i != s_j:
                mov = _pair_margin_multiplier(
                    winner, loser, rating_gap, discipline, config
                )
                s_i = margin_adjusted_outcome(s_i, mov, elo_cfg.margin_cap)
                s_j = margin_adjusted_outcome(s_j, mov, elo_cfg.margin_cap)

            v_inv_sum[aid_i] += weight * g_j * g_j * e_i * (1.0 - e_i)
            v_inv_sum[aid_j] += weight * g_i * g_i * e_j * (1.0 - e_j)

            term_i = weight * g_j * (s_i - e_i)
            term_j = weight * g_i * (s_j - e_j)
            delta_tilde_sum[aid_i] += term_i
            delta_tilde_sum[aid_j] += term_j

            if collect_pairs:
                label_i = "won" if s_i > s_j else ("lost" if s_i < s_j else "tied")
                label_j = "won" if s_j > s_i else ("lost" if s_j < s_i else "tied")
                pair_terms[aid_i].append(
                    (aid_j, label_i, round(e_i, 4), round(s_i, 4), term_i, mov)
                )
                pair_terms[aid_j].append(
                    (aid_i, label_j, round(e_j, 4), round(s_j, 4), term_j, mov)
                )

    # --- 4-7) Per-athlete canonical update. ---------------------------------
    norm = 1.0
    if config.field_normalization_exponent:
        norm = float(n_field - 1) ** config.field_normalization_exponent

    updates: list[RatingUpdate] = []
    for res in active:
        aid = res.athlete_id
        rating = ratings.get(aid, AthleteRating(athlete_id=aid))
        mu_before = rating.mu
        sigma_before = sigma_inflated[aid]
        phi = sigma_before / GLICKO2_SCALE
        volatility = rating.volatility or GLICKO2_DEFAULT_VOLATILITY

        v_inv = v_inv_sum[aid] / norm
        delta_tilde = delta_tilde_sum[aid] / norm

        if v_inv > 0.0:
            v = 1.0 / v_inv
            volatility_after = glicko2_update_volatility(
                volatility,
                phi,
                v,
                v * delta_tilde,
                elo_cfg.glicko2_tau,
            )
            phi_star_sq = phi * phi + volatility_after * volatility_after
            phi_new_sq = 1.0 / (1.0 / phi_star_sq + v_inv)
        else:
            # No usable evidence this round (degenerate E): only the
            # volatility inflation acts on φ, and μ is unchanged.
            volatility_after = volatility
            phi_new_sq = phi * phi + volatility_after * volatility_after

        # Glickman step 7 on the internal scale, converted back for storage.
        mu_after = mu_before + GLICKO2_SCALE * phi_new_sq * delta_tilde

        sigma_after = max(
            elo_cfg.sigma_floor,
            min(elo_cfg.sigma_ceiling, math.sqrt(phi_new_sq) * GLICKO2_SCALE),
        )

        contributions: list[PairContribution] = []
        if collect_pairs:
            # Attribute the round's μ move back to the pairs that produced it:
            # Δμ = SCALE · φ'² · Σ_j term_j, so each pair's share is exactly
            # SCALE · φ'² · term_j / norm. The shares sum to Δμ by construction.
            scale = GLICKO2_SCALE * phi_new_sq / norm
            contributions = [
                PairContribution(
                    opponent_id=opp,
                    result=label,
                    expected=expected,
                    actual=actual,
                    delta=round(term * scale, 2),
                    margin_multiplier=round(mov, 2),
                )
                for opp, label, expected, actual, term, mov in pair_terms[aid]
            ]

        updates.append(
            RatingUpdate(
                athlete_id=aid,
                mu_before=mu_before,
                mu_after=mu_after,
                sigma_before=sigma_before,
                sigma_after=sigma_after,
                contributing_pairs=contributions,
                volatility_after=volatility_after,
            )
        )

    return updates


def mu_drift(updates: Iterable[RatingUpdate]) -> float:
    """Mean μ change across a set of updates — the population-drift monitor.

    Canonical Glicko-2 trades the incumbent's exact zero-sum invariant for a
    variance-scaled step, so a round's μ deltas no longer cancel. The design
    doc's replacement guard is this drift statistic: mean μ change per season
    must stay ≈ 0 (alarm at ±5 μ/season).
    """
    deltas = [u.mu_after - u.mu_before for u in updates]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


# ---------------------------------------------------------------------------
# G2PLEngine — backtest variant
# ---------------------------------------------------------------------------

_ROUND_ORDER = {
    RoundType.QUALIFICATION: 0,
    RoundType.SEMI: 1,
    RoundType.FINAL: 2,
}


class G2PLEngine:
    """:class:`~climbing_elo.engine.evaluation.RatingEngine` for ``g2pl``.

    The harness has already backfilled the working DB with the *incumbent*
    engine, so — exactly like :class:`~climbing_elo.engine.gelo.GELoEngine` and
    the #38 baselines — this engine ignores those ``Rating`` rows and re-derives
    its own ratings into an in-memory dict on first ``predict()`` for a
    discipline.

    Leakage guard: the working DB copy holds the *whole* source dataset, not
    just the training window, so the snapshot must be date-filtered. The
    harness passes ``cutoff_date`` (via
    :func:`~climbing_elo.engine.evaluation.runner._build_engine`, which
    inspects the factory signature) and only events strictly before it are
    replayed — mirroring ``run_backfill(..., end_date=split.train_end_date)``.

    Args:
        session: Session on the harness's training-end DB copy.
        cutoff_date: Train on events strictly before this date. ``None``
            replays everything (direct/manual use only).
        config: :class:`G2PLConfig` override — the hook for the importance
            weight sweep and the ``mov_mode`` A/B.
    """

    def __init__(
        self,
        session: Session,
        cutoff_date: date | None = None,
        config: G2PLConfig | None = None,
    ):
        self._session = session
        self._cutoff_date = cutoff_date
        self.config = config if config is not None else DEFAULT_G2PL_CONFIG
        self._snapshots: dict[Discipline, dict[int, AthleteRating]] = {}
        #: discipline → {season: mean μ drift} — the population-drift monitor.
        self.drift_by_season: dict[Discipline, dict[int, float]] = {}

    def name(self) -> str:
        return "g2pl"

    def _build_snapshot(self, discipline: Discipline) -> dict[int, AthleteRating]:
        """Replay every training event for ``discipline`` through ``g2pl``."""
        use_incumbent = discipline not in SUPPORTED_DISCIPLINES
        if use_incumbent:
            # Speed / aggregate disciplines fall back to the incumbent so a
            # mixed-discipline backtest still produces numbers (design doc:
            # "Speed stays on the incumbent").
            log.info(
                "g2pl does not cover %s — delegating to the incumbent engine.",
                discipline.value,
            )

        ratings: dict[int, AthleteRating] = {}
        drift: dict[int, list[float]] = {}

        stmt = (
            select(Event)
            .where(Event.discipline == discipline)
            .order_by(Event.start_date.asc())
        )
        if self._cutoff_date is not None:
            stmt = stmt.where(Event.start_date < self._cutoff_date)

        for event in self._session.execute(stmt).scalars():
            rounds = sorted(
                event.rounds, key=lambda r: _ROUND_ORDER.get(r.round_type, 0)
            )
            event_had_updates = False

            for rnd in rounds:
                results = list(
                    self._session.execute(
                        select(Result).where(Result.round_id == rnd.id)
                    ).scalars()
                )
                if not results:
                    continue

                athlete_results: list[AthleteResult] = []
                for res in results:
                    ratings.setdefault(
                        res.athlete_id, AthleteRating(athlete_id=res.athlete_id)
                    )
                    athlete_results.append(
                        AthleteResult(
                            athlete_id=res.athlete_id,
                            rank=res.rank or 999,
                            score_normalized=res.score_normalized,
                            dnf=res.dnf,
                            dns=res.dns,
                        )
                    )

                if use_incumbent:
                    updates = calculate_round_updates(
                        athlete_results,
                        ratings,
                        event.tier,
                        rnd.round_type,
                        event.start_date,
                        discipline=discipline,
                        config=self.config.elo,
                    )
                else:
                    updates = calculate_g2pl_round_updates(
                        athlete_results,
                        ratings,
                        event.tier,
                        rnd.round_type,
                        event.start_date,
                        discipline=discipline,
                        config=self.config,
                        collect_pairs=False,
                    )

                if updates:
                    drift.setdefault(event.season, []).append(mu_drift(updates))

                for upd in updates:
                    ar = ratings[upd.athlete_id]
                    ar.mu = upd.mu_after
                    ar.sigma = upd.sigma_after
                    ar.volatility = upd.volatility_after
                    # #89 Fix 3 parity: later rounds of the SAME event must
                    # not re-inflate σ for an inactivity gap already consumed.
                    ar.last_event_at = event.start_date
                event_had_updates = bool(updates) or event_had_updates

            if event_had_updates:
                seen = {
                    r.athlete_id for rnd in rounds for r in rnd.results if not r.dns
                }
                for aid in seen:
                    ar = ratings.get(aid)
                    if ar is None:
                        continue
                    ar.n_events += 1
                    ar.last_event_at = event.start_date
                    ar.provisional = ar.n_events < self.config.elo.provisional_threshold

        per_season = {
            season: sum(vals) / len(vals) for season, vals in drift.items() if vals
        }
        self.drift_by_season[discipline] = per_season
        worst = max((abs(d) for d in per_season.values()), default=0.0)
        if worst > 5.0:
            log.warning(
                "g2pl population drift exceeds ±5 μ/season for %s (worst %.2f).",
                discipline.value,
                worst,
            )

        return ratings

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]:
        if discipline not in self._snapshots:
            self._snapshots[discipline] = self._build_snapshot(discipline)
        snap = self._snapshots[discipline]

        out: dict[int, RatingForecast] = {}
        for aid in athletes_in_round:
            ar = snap.get(aid)
            if ar is None:
                out[aid] = RatingForecast(
                    athlete_id=aid, mu=DEFAULT_MU, sigma=DEFAULT_SIGMA, n_events=0
                )
            else:
                out[aid] = RatingForecast(
                    athlete_id=aid, mu=ar.mu, sigma=ar.sigma, n_events=ar.n_events
                )
        return out


# ---------------------------------------------------------------------------
# Registration — fires at import time
# ---------------------------------------------------------------------------

register_variant("g2pl", G2PLEngine)
