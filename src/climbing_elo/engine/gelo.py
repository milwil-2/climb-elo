"""G-Elo (Szczecinski 2022) bucketed margin-of-victory variant (Issue #84).

Background
----------

Reference: Szczecinski, L. (2022). *G-Elo: An Extension of the Elo Algorithm
for Outcome Predictions in Multi-Way Tournaments.* The paper proposes a
**bucketed** margin-of-victory multiplier — discrete bins of score margin
each carrying a constant multiplier — in place of the continuous
``min(1 + gap/max_gap, MARGIN_CAP)`` widely used in 538-style implementations.

Why benchmark it
^^^^^^^^^^^^^^^^

The production engine currently uses a continuous MOV (`engine/elo.py`):
``base = min(1 + gap/max_gap, margin_cap)`` damped by the 538-style
gap-conditioning factor.  Bucketed MOV has two theoretical advantages:

1. **Outlier robustness** — a continuous formula treats a ``1 → 99``
   point-gap as 99× more informative than a ``1 → 2`` gap. Bucketing caps
   each tier at a fixed multiplier, so a single extreme blowout cannot
   dominate the round.
2. **Discrete interpretability** — the operator can reason about MOV in
   units the sport uses ("crushed by ≥10 holds", "lost by 0.5s"), which
   makes tuning more legible than tweaking a continuous-formula constant.

Whether *G-Elo actually wins on the climbing fixture set* is an open
empirical question.  This module ships the variant; the user runs the
benchmark.

Scope (what this module contains)
---------------------------------

* :data:`GELO_LEAD_BUCKETS`, :data:`GELO_BOULDER_BUCKETS`,
  :data:`GELO_SPEED_BUCKETS` — the default per-discipline bucket tables
  (chosen to span the empirical score-gap distribution we see in the
  ``tests/fixtures/external_rankings/`` corpus).
* :func:`compute_gelo_margin_multiplier` — bucketed MOV.  Signature
  mirrors the production ``compute_margin_multiplier`` family but takes
  a ``discipline`` to pick the right bucket table.
* :class:`GELoEngine` — :class:`~climbing_elo.engine.evaluation.RatingEngine`
  implementation.  Re-runs backfill against the training-DB with a
  custom :class:`~climbing_elo.engine.elo.EloConfig` whose
  ``gelo_buckets`` field is populated, so each pair update uses the
  bucketed multiplier in place of the continuous one.

Out of scope (file follow-ups)
------------------------------

* Tuning the bucket edges via grid search — only relevant if GELo wins
  the head-to-head on the production fixtures.
* Replacing the production engine with GELo — this is purely a benchmark
  variant.  ``"current"`` stays the default.
* Per-tier bucket tables — Szczecinski's formulation is league-wide; if
  per-tier buckets ever look promising they would be filed as a separate
  issue.

Zero-sum invariant
------------------

GELo touches only the *margin multiplier*; the pairwise μ update is
otherwise identical to the production engine.  Pair updates remain
symmetric so the round-level zero-sum invariant on μ is preserved.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    AthleteRating,
    AthleteResult,
    EloConfig,
    _gap_conditioning_factor,
    calculate_round_updates,
)
from climbing_elo.engine.evaluation import (
    RatingForecast,
    register_variant,
)
from climbing_elo.models import (
    Discipline,
    Event,
    Result,
    RoundType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucket tables
# ---------------------------------------------------------------------------
#
# A bucket is ``(upper_bound_exclusive, multiplier)`` — the first bucket
# whose ``upper_bound_exclusive`` strictly exceeds the absolute score gap is
# selected. The final bucket uses ``float('inf')`` as its upper bound so
# every gap falls into exactly one bucket.
#
# Bucket edges chosen to span the empirical climbing distribution:
#
# * **Lead** uses the production ``score_normalized`` scale — integer
#   holds, with ``"34+"`` → 34.5 and ``"TOP"`` → 999.0. A gap of <1 means
#   "same hold or fractional difference"; ≥10 means "demolished by 10+
#   holds or a TOP-vs-mid-route blowout".
# * **Boulder** uses the production pre-2025 normalised score
#   (``tops * 1000 + zones * 100 - top_att * 10 - zone_att``) — one full
#   top is worth ~1000 normalised points; one full zone ~100. The bucket
#   edges {100, 500, 1000, ∞} therefore map roughly to {"same number of
#   tops, attempt difference", "one top closer", "one extra top", "two
#   extra tops or more"}. New 2025+ decimal scores live in the same
#   numeric range (typically 0–100) so the bucket edges remain reasonable.
# * **Speed** uses time-in-seconds, lower-is-better. Bucket edges
#   {0.05, 0.2, 0.5, ∞} correspond to {photo-finish, normal margin, clear
#   margin, blowout / DNF disparity}.
#
# Multipliers grow with the bucket — a bigger margin earns more — and
# are capped well below the continuous-MOV ``margin_cap=1.5`` ceiling
# would allow in extreme cases, on purpose. Bucketing's theoretical
# robustness benefit comes from this cap.

#: ``(upper_bound_exclusive, multiplier)`` tuples in ascending order.
GELoBucketTable = Sequence[tuple[float, float]]


GELO_LEAD_BUCKETS: GELoBucketTable = (
    (1.0, 1.00),  # same-hold or fractional gap (e.g. "34+ vs 34")
    (3.0, 1.15),  # 1–2 holds
    (10.0, 1.30),  # 3–9 holds
    (float("inf"), 1.50),  # 10+ holds (or TOP-vs-low blowout)
)


GELO_BOULDER_BUCKETS: GELoBucketTable = (
    (100.0, 1.00),  # within one full zone (sub-attempt differences)
    (500.0, 1.15),  # within ~one zone gap
    (1000.0, 1.30),  # roughly one top difference
    (float("inf"), 1.50),  # two or more tops apart
)


GELO_SPEED_BUCKETS: GELoBucketTable = (
    (0.05, 1.00),  # photo finish
    (0.20, 1.10),  # normal race margin
    (0.50, 1.25),  # clear margin
    (float("inf"), 1.50),  # blowout / DNF-style gap
)


#: Default mapping of discipline → bucket table. Used as the default for
#: :attr:`EloConfig.gelo_buckets` when callers opt into the GELo variant
#: without supplying their own table.
DEFAULT_GELO_BUCKETS: dict[Discipline, GELoBucketTable] = {
    Discipline.LEAD: GELO_LEAD_BUCKETS,
    Discipline.BOULDER: GELO_BOULDER_BUCKETS,
    Discipline.SPEED: GELO_SPEED_BUCKETS,
}


# ---------------------------------------------------------------------------
# Bucketed multiplier
# ---------------------------------------------------------------------------


def _select_bucket(gap: float, buckets: GELoBucketTable) -> float:
    """Return the multiplier for the first bucket strictly containing ``gap``.

    ``buckets`` must be sorted by ``upper_bound_exclusive`` ascending and
    end with an ``inf`` sentinel.  The first bucket whose upper bound
    *strictly exceeds* ``gap`` is selected — so a gap of exactly the
    upper bound falls into the *next* bucket.

    Examples
    --------

    With ``GELO_LEAD_BUCKETS = [(1.0, 1.00), (3.0, 1.15), (10.0, 1.30),
    (inf, 1.50)]``:

    * ``gap=0.0`` → 1.00
    * ``gap=0.5`` → 1.00
    * ``gap=1.0`` → 1.15  (first bucket with bound > 1.0)
    * ``gap=2.99`` → 1.15
    * ``gap=3.0``  → 1.30
    * ``gap=999``  → 1.50
    """
    for upper, mult in buckets:
        if gap < upper:
            return mult
    # Unreachable when the last bucket is ``(inf, ...)`` — defensive
    # fall-through returns the final multiplier.
    return buckets[-1][1]


def compute_gelo_margin_multiplier(
    score_a: float | None,
    score_b: float | None,
    discipline: Discipline,
    rating_gap: float = 0.0,
    config: EloConfig | None = None,
) -> float:
    """Bucketed margin-of-victory multiplier (Szczecinski 2022).

    Parameters
    ----------
    score_a, score_b:
        Normalised scores for the winner / loser of the pair. ``None`` on
        either side means we can't read a margin — return 1.0 (no bonus).
    discipline:
        Which bucket table to use. Falls back to a 1.0 multiplier when the
        discipline is unsupported (e.g. ``Discipline.BOULDER_LEAD``, which
        is an aggregate rating, not a competition format).
    rating_gap:
        ``μ_winner − μ_loser`` (pre-update). Threaded through the same
        538-style gap-conditioning damping used by the continuous MOV —
        we keep gap-conditioning *on top of* the bucketed base so that
        favourite-side blowouts are still damped relative to upsets.
        Decision rationale: bucketing replaces the *unconditioned* MOV
        formula; the gap-conditioning damp is an independent (#53) layer
        we explicitly want to preserve so the two variants are comparable
        on a single axis.
    config:
        Engine config. When ``config.gelo_buckets`` is set, that mapping
        wins over the module-level default; otherwise
        :data:`DEFAULT_GELO_BUCKETS` is consulted.  Pass ``None`` to use
        all defaults.
    """
    if score_a is None or score_b is None:
        return 1.0

    if config is not None and config.gelo_buckets is not None:
        buckets = config.gelo_buckets.get(discipline)
    else:
        buckets = DEFAULT_GELO_BUCKETS.get(discipline)

    if buckets is None:
        # Discipline outside the defined bucket set (e.g. BOULDER_LEAD
        # aggregate) — no opinion, return neutral multiplier.
        return 1.0

    gap = abs(score_a - score_b)
    base = _select_bucket(gap, buckets)

    if config is None:
        # No gap-conditioning available — return the raw bucket multiplier.
        return base
    return base * _gap_conditioning_factor(rating_gap, config)


# ---------------------------------------------------------------------------
# GELoEngine — backtest variant
# ---------------------------------------------------------------------------
#
# Follows the same pattern as ``StrippedEloEngine`` in ``engine/baselines.py``:
# the harness has already run a production backfill against the working DB,
# so we cannot reuse the Rating rows it produced. Instead we re-derive
# ratings from scratch into an in-memory dict using a custom :class:`EloConfig`
# whose ``gelo_buckets`` field is populated.  ``predict()`` returns from
# that private snapshot.

_ROUND_ORDER = {
    RoundType.QUALIFICATION: 0,
    RoundType.SEMI: 1,
    RoundType.FINAL: 2,
}


class GELoEngine:
    """Glicko-2 engine with bucketed Szczecinski-style MOV multiplier.

    Construction triggers a fresh in-memory backfill against the harness
    training DB using :class:`EloConfig` with ``gelo_buckets`` populated.
    The production MOV path is bypassed at the pair level — every pair
    update reads its multiplier from the appropriate per-discipline
    bucket table.

    Args:
        session: SQLAlchemy session pointing at the training-end DB.
        buckets: Optional override for the per-discipline bucket tables.
            Defaults to :data:`DEFAULT_GELO_BUCKETS`. Useful for follow-up
            grid-search work that tunes the bucket edges.
    """

    def __init__(
        self,
        session: Session,
        buckets: dict[Discipline, GELoBucketTable] | None = None,
    ):
        self._session = session
        self._buckets: dict[Discipline, GELoBucketTable] = (
            buckets if buckets is not None else dict(DEFAULT_GELO_BUCKETS)
        )
        # discipline → {athlete_id → AthleteRating}
        self._snapshots: dict[Discipline, dict[int, AthleteRating]] = {}

    def name(self) -> str:
        return "gelo"

    def _build_snapshot(self, discipline: Discipline) -> dict[int, AthleteRating]:
        """Re-run backfill for ``discipline`` using bucketed-MOV.

        Mirrors ``StrippedEloEngine._build_snapshot`` (see ``baselines.py``)
        but plugs ``gelo_buckets`` into the :class:`EloConfig` instead of
        ablating MOV / σ knobs.
        """
        # Build a config that swaps in the bucketed MOV while keeping all
        # other production tunables (K_FACTOR_TABLE, sigma_floor/ceiling,
        # margin_cap, gap-conditioning) at their default values. This is
        # the cleanest A/B test: we change MOV and nothing else.
        elo_config = EloConfig(gelo_buckets=self._buckets)

        ratings: dict[int, AthleteRating] = {}

        events = list(
            self._session.execute(
                select(Event)
                .where(Event.discipline == discipline)
                .order_by(Event.start_date.asc())
            ).scalars()
        )

        for event in events:
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
                    if res.athlete_id not in ratings:
                        ratings[res.athlete_id] = AthleteRating(
                            athlete_id=res.athlete_id,
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
                updates = calculate_round_updates(
                    athlete_results,
                    ratings,
                    event.tier,
                    rnd.round_type,
                    event.start_date,
                    discipline=discipline,
                    config=elo_config,
                )
                for upd in updates:
                    ar = ratings[upd.athlete_id]
                    ar.mu = upd.mu_after
                    ar.sigma = upd.sigma_after
                event_had_updates = bool(updates) or event_had_updates

            if event_had_updates:
                distinct = {
                    r.athlete_id for rnd in rounds for r in rnd.results if not r.dns
                }
                for aid in distinct:
                    if aid in ratings:
                        ratings[aid].n_events += 1
                        ratings[aid].last_event_at = event.start_date
                        if ratings[aid].n_events >= elo_config.provisional_threshold:
                            ratings[aid].provisional = False

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
            if aid in snap:
                ar = snap[aid]
                out[aid] = RatingForecast(
                    athlete_id=aid,
                    mu=ar.mu,
                    sigma=ar.sigma,
                    n_events=ar.n_events,
                )
            else:
                out[aid] = RatingForecast(
                    athlete_id=aid,
                    mu=DEFAULT_MU,
                    sigma=DEFAULT_SIGMA,
                    n_events=0,
                )
        return out


# ---------------------------------------------------------------------------
# Registration — fires at import time
# ---------------------------------------------------------------------------

register_variant("gelo", GELoEngine)
