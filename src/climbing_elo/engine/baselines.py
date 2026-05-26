"""Baseline rating engines for the backtest harness (Issue #38).

These four baselines exist for one reason: to establish the *floor* against
which the production engine must clear the bar. If the current ELO engine
can't beat persistence on log-loss, our bells-and-whistles aren't earning
their keep.

Variants
--------

``random``
    Uniformly random athlete μ per event. Sigma is held at the default
    (so the Monte Carlo projector spreads probabilities across the field).
    Uses a deterministic seed derived from ``BacktestDataset.rng_seed`` so
    two runs produce byte-identical reports.

``persistence``
    Predict the athlete's most-recent finish in this discipline before the
    training cutoff. Naive but a famously strong baseline in many sports.
    Lower historical rank → higher predicted μ.

``ifsc_official``
    Use the official IFSC season-end ranking as the prediction. Marked as
    a TODO stub here — implementing the scrape against
    ``components.ifsc-climbing.org/rankings/`` is a meaningful sub-project
    (auth, pagination, caching). The stub returns no-prediction (default
    μ/σ for everyone) so the harness still runs end-to-end. Flagged
    explicitly so the report shows "this baseline isn't real yet".

``stripped_elo``
    Current engine but with each piece of machinery turned off:
      - ``MARGIN_CAP=1.0`` (no margin-of-victory bonus)
      - No 2× provisional K multiplier (cold-start athletes update as fast
        as veterans)
      - σ frozen at ``DEFAULT_SIGMA=350`` (no time decay, no convergence)
    Isolates "what does each feature buy us?" — if stripped beats current
    on any metric, that feature is the wrong abstraction.

All four engines register themselves at import time, so importing this
module from ``engine.evaluation`` is enough to make them appear under
``--variant`` in the CLI.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine import elo as elo_mod
from climbing_elo.engine.elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    AthleteRating,
    AthleteResult,
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
    Round,
    RoundType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ROUND_ORDER = {
    RoundType.QUALIFICATION: 0,
    RoundType.SEMI: 1,
    RoundType.FINAL: 2,
}


def _seed_from_session(session: Session, fallback: int = 0) -> int:
    """Derive a stable seed from the working DB so two runs against the same
    snapshot produce identical random orderings.

    We use the (athlete_id, discipline, mu) tuple count as a cheap fingerprint
    — the harness re-runs backfill on the same training cutoff, so this is
    stable across re-runs. The ``fallback`` is mixed in so the same engine
    instance produces different orderings for different disciplines.
    """
    from climbing_elo.models import Rating

    fp = 0
    for r in session.execute(select(Rating)).scalars():
        # Hash mix — order-independent (sum) so it's stable regardless of
        # SQLite row ordering.
        fp = (fp + r.athlete_id * 1000003 + int(r.mu * 100)) % (2**31 - 1)
    return (fp + fallback) % (2**31 - 1)


# ---------------------------------------------------------------------------
# RandomEngine
# ---------------------------------------------------------------------------


class RandomEngine:
    """Uniformly random athlete ordering per event.

    Each ``predict()`` call returns a fresh random μ per athlete. Sigma is
    held at ``DEFAULT_SIGMA`` so the Monte Carlo projector still produces
    sensible podium probabilities (a random ordering with σ=0 would lock the
    ranking in stone, which is not the spirit of "random").

    The RNG is seeded once per engine instance using a fingerprint of the
    DB snapshot, so two runs against the same training data produce
    identical reports.
    """

    # Range chosen to match the empirical μ range produced by the current
    # engine (~1200–1900). Wider = more confident predictions, narrower =
    # more "everybody is roughly equal".
    MU_LOW = 1200.0
    MU_HIGH = 1900.0

    def __init__(self, session: Session):
        self._session = session
        # The harness re-creates the engine for every split, so we capture
        # the DB fingerprint at construction time.
        self._seed = _seed_from_session(session)
        self._rng = random.Random(self._seed)

    def name(self) -> str:
        return "random"

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]:
        out: dict[int, RatingForecast] = {}
        # Sort for a stable iteration order (sets / generator inputs are
        # otherwise non-deterministic across Python invocations).
        for aid in sorted(athletes_in_round):
            mu = self._rng.uniform(self.MU_LOW, self.MU_HIGH)
            out[aid] = RatingForecast(
                athlete_id=aid,
                mu=mu,
                sigma=DEFAULT_SIGMA,
                n_events=0,
            )
        return out


# ---------------------------------------------------------------------------
# PersistenceEngine
# ---------------------------------------------------------------------------


class PersistenceEngine:
    """Predict the athlete's most-recent finish in this discipline.

    Lower historical rank ⇒ higher predicted μ. We map rank to μ by

        μ = DEFAULT_MU + (MU_PER_RANK_STEP * (MEDIAN_RANK - rank))

    so rank 1 sits well above default, rank 30 sits below, and we never
    venture outside the rough ±200 band the production engine occupies.

    Unrated athletes (no prior result in this discipline) fall back to
    ``DEFAULT_MU/DEFAULT_SIGMA`` with ``n_events=0`` — the harness then
    classifies them into the "cold-start" tenure bucket so calibration is
    measured separately.
    """

    MU_PER_RANK_STEP = 12.0  # ~10pt per rank — produces a 300pt total spread
    MEDIAN_RANK = 15.0  # σ-zero ranking that maps to μ=DEFAULT_MU

    def __init__(self, session: Session):
        self._session = session
        # discipline → {athlete_id: (most_recent_rank, n_events_seen)}
        self._cache: dict[Discipline, dict[int, tuple[int, int]]] = {}

    def name(self) -> str:
        return "persistence"

    def _build_snapshot(self, discipline: Discipline) -> dict[int, tuple[int, int]]:
        """Load latest-finish-per-athlete for ``discipline``.

        Returns a dict keyed by athlete_id, value is
        ``(rank_in_most_recent_event, n_events_seen)``.
        """
        snap: dict[int, tuple[int, int]] = {}
        # Pull all (athlete_id, rank, event.start_date) tuples in the
        # discipline ordered by date. We sweep in chronological order so
        # the last write per athlete is the most recent.
        rows = self._session.execute(
            select(Result.athlete_id, Result.rank, Event.start_date)
            .join(Round, Round.id == Result.round_id)
            .join(Event, Event.id == Round.event_id)
            .where(
                Event.discipline == discipline,
                Result.rank.is_not(None),
                ~Result.dns,
            )
            .order_by(Event.start_date.asc(), Result.rank.asc())
        ).all()

        n_events_per_athlete: dict[int, set[date]] = {}
        for aid, rank, when in rows:
            n_events_per_athlete.setdefault(aid, set()).add(when)
            snap[aid] = (rank, 0)  # rank overwritten — last write wins

        for aid, days in n_events_per_athlete.items():
            current_rank, _ = snap[aid]
            snap[aid] = (current_rank, len(days))
        return snap

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]:
        if discipline not in self._cache:
            self._cache[discipline] = self._build_snapshot(discipline)
        snap = self._cache[discipline]

        out: dict[int, RatingForecast] = {}
        for aid in athletes_in_round:
            if aid in snap:
                rank, n_events = snap[aid]
                mu = DEFAULT_MU + self.MU_PER_RANK_STEP * (self.MEDIAN_RANK - rank)
                out[aid] = RatingForecast(
                    athlete_id=aid,
                    mu=mu,
                    sigma=DEFAULT_SIGMA,
                    n_events=n_events,
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
# IFSCOfficialEngine
# ---------------------------------------------------------------------------


class IFSCOfficialEngine:
    """Use the official IFSC season-end ranking as the prediction.

    NOT YET IMPLEMENTED — see issue #38.

    Implementing this properly requires scraping
    ``components.ifsc-climbing.org/rankings/`` (or extending
    :mod:`climbing_elo.scraper.ifsc_api` with a rankings endpoint), caching
    the results locally, and mapping IFSC rank to predicted μ. That's a
    meaningful sub-project (auth handshake, pagination, cache invalidation)
    and we deliberately defer it.

    The stub returns ``DEFAULT_MU/DEFAULT_SIGMA`` for every athlete so the
    harness still produces a complete report — the report will show this
    baseline as equivalent to "no information", which is the truthful
    outcome until the scrape lands.

    TODO(#38-followup): scrape and cache IFSC season-end rankings, populate
    a per-discipline season-end snapshot keyed on (athlete_id, season,
    discipline), then translate rank → μ via the same formula as
    :class:`PersistenceEngine`.
    """

    def __init__(self, session: Session):
        self._session = session
        log.warning(
            "ifsc_official is a stub — scraping IFSC rankings is deferred "
            "per issue #38 (see docstring TODO)."
        )

    def name(self) -> str:
        return "ifsc_official"

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]:
        return {
            aid: RatingForecast(
                athlete_id=aid,
                mu=DEFAULT_MU,
                sigma=DEFAULT_SIGMA,
                n_events=0,
            )
            for aid in athletes_in_round
        }


# ---------------------------------------------------------------------------
# StrippedEloEngine
# ---------------------------------------------------------------------------


@dataclass
class _StrippedConfig:
    """Stripped ELO knobs — each one is a feature we want to ablate.

    The defaults are the *stripped* values; the production defaults live in
    :mod:`engine.elo`.
    """

    margin_cap: float = 1.0  # vs. 1.5 in production: no MOV bonus
    provisional_k_multiplier: float = 1.0  # vs. 2.0: no cold-start boost
    sigma_floor: float = DEFAULT_SIGMA  # σ frozen at default
    sigma_ceiling: float = DEFAULT_SIGMA  # no decay
    sigma_convergence_factor: float = 1.0  # no convergence


class StrippedEloEngine:
    """Current engine with each non-trivial feature ablated.

    On construction we re-run a stripped backfill against the same training
    DB the harness prepared, but with:

      - ``MARGIN_CAP=1.0``       — pairwise updates ignore score margin
      - provisional K = 1.0      — no 2× boost for cold-start athletes
      - σ frozen at 350           — no decay-toward-ceiling on inactivity,
                                    no 0.98× convergence per event

    Implementation notes
    --------------------

    The harness has already run *production* ``run_backfill`` before
    constructing the engine. We can't undo that mutation in-place (the
    session has fresh Rating rows), so we re-derive ratings from scratch in
    a private in-memory dict instead of writing back to the DB. ``predict``
    returns from that private snapshot.

    Why not monkey-patch the module constants and re-run ``run_backfill``?
    Because ``run_backfill`` reads from / writes to the session's Rating
    table, and we already have one set of post-backfill rows there. Stomping
    on them would leak production state into a "stripped" run and confuse
    any downstream observers (logging, eventual breakdowns, etc.).
    """

    def __init__(self, session: Session, config: _StrippedConfig | None = None):
        self._session = session
        self._config = config or _StrippedConfig()
        # discipline → {athlete_id → AthleteRating}
        self._snapshots: dict[Discipline, dict[int, AthleteRating]] = {}

    def name(self) -> str:
        return "stripped_elo"

    def _build_snapshot(self, discipline: Discipline) -> dict[int, AthleteRating]:
        """Re-run backfill for ``discipline`` with stripped parameters."""
        cfg = self._config
        ratings: dict[int, AthleteRating] = {}

        events = list(
            self._session.execute(
                select(Event)
                .where(Event.discipline == discipline)
                .order_by(Event.start_date.asc())
            ).scalars()
        )

        # Monkey-patch the module constants for the duration of this rebuild.
        # We restore them on exit so other engines (e.g. ``current``) running
        # in the same process aren't affected.  This is the cleanest place
        # to inject the stripped knobs without forking ``calculate_round_updates``.
        orig_margin_cap = elo_mod.MARGIN_CAP
        orig_prov_k = elo_mod.PROVISIONAL_K_MULTIPLIER
        orig_sigma_floor = elo_mod.SIGMA_FLOOR
        orig_sigma_ceiling = elo_mod.SIGMA_CEILING
        orig_sigma_conv = elo_mod.SIGMA_CONVERGENCE_FACTOR
        try:
            elo_mod.MARGIN_CAP = cfg.margin_cap
            elo_mod.PROVISIONAL_K_MULTIPLIER = cfg.provisional_k_multiplier
            elo_mod.SIGMA_FLOOR = cfg.sigma_floor
            elo_mod.SIGMA_CEILING = cfg.sigma_ceiling
            elo_mod.SIGMA_CONVERGENCE_FACTOR = cfg.sigma_convergence_factor

            for event in events:
                rounds = sorted(
                    event.rounds, key=lambda r: _ROUND_ORDER.get(r.round_type, 0)
                )
                seen_athletes: set[int] = set()
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
                                # σ frozen at floor — stripped of decay
                                sigma=cfg.sigma_floor,
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
                    )
                    for upd in updates:
                        ar = ratings[upd.athlete_id]
                        ar.mu = upd.mu_after
                        # Stripped: σ stays at the floor.
                        ar.sigma = cfg.sigma_floor
                        if not res.dns:
                            seen_athletes.add(upd.athlete_id)
                    event_had_updates = bool(updates) or event_had_updates

                if event_had_updates:
                    # Increment n_events once per event (matches production).
                    distinct = {
                        r.athlete_id for rnd in rounds for r in rnd.results if not r.dns
                    }
                    for aid in distinct:
                        if aid in ratings:
                            ratings[aid].n_events += 1
                            ratings[aid].last_event_at = event.start_date
                            # Stripped: provisional doesn't matter (K mult = 1).
                            ratings[aid].provisional = False
        finally:
            elo_mod.MARGIN_CAP = orig_margin_cap
            elo_mod.PROVISIONAL_K_MULTIPLIER = orig_prov_k
            elo_mod.SIGMA_FLOOR = orig_sigma_floor
            elo_mod.SIGMA_CEILING = orig_sigma_ceiling
            elo_mod.SIGMA_CONVERGENCE_FACTOR = orig_sigma_conv

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


# Note: we register at module-import time so that importing this module
# from ``engine/evaluation.py`` (with an F401-suppressed import) is
# sufficient to expose all four variants under the ``--variant`` flag.
register_variant("random", RandomEngine)
register_variant("persistence", PersistenceEngine)
register_variant("ifsc_official", IFSCOfficialEngine)
register_variant("stripped_elo", StrippedEloEngine)
