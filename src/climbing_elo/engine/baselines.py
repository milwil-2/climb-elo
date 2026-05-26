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
    Use the official IFSC season-end World Cup ranking as the prediction.
    Rankings are sourced via :mod:`climbing_elo.scraper.external_rankings`
    (see that module for why we consume Wikipedia tables rather than the
    IFSC widget host). Athletes that appear in the most recent season-end
    ranking before the training cutoff get a rank-derived μ (lower rank ⇒
    higher μ); unranked athletes default to ``DEFAULT_MU``.

``ascentstats``
    Use the AscentStats community Bayesian dynamic Bradley-Terry rating
    as the prediction. Boulder only — the upstream source only publishes
    boulder ratings. Same rank-based μ translation as ``ifsc_official``
    for cross-source comparability. Tracked in the research doc under R5
    (external validation) and added alongside #44 to give us a second
    "system we're trying to beat" beyond the official ranking.

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
    Athlete,
    Discipline,
    Event,
    Result,
    Round,
    RoundType,
)
from climbing_elo.scraper.external_rankings import (
    DisciplineKey,
    RankedAthlete,
    Source,
    load_snapshot,
    normalize_name,
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
# Shared rank-based engine (drives IFSCOfficial and AscentStats)
# ---------------------------------------------------------------------------


# Map the SQLAlchemy enum to the string keys our snapshot files use. Speed
# is intentionally absent from AscentStats (boulder-only source); the
# engines filter that out below.
_DISCIPLINE_KEY: dict[Discipline, DisciplineKey] = {
    Discipline.BOULDER: "boulder",
    Discipline.LEAD: "lead",
    Discipline.SPEED: "speed",
}


class _RankSnapshotEngine:
    """Common machinery for snapshot-fed baselines.

    Both :class:`IFSCOfficialEngine` and :class:`AscentStatsEngine` consume
    a per-season-end ranking and translate athlete rank → predicted μ. The
    only differences between the two engines are:

      - ``_SOURCE`` (which snapshot directory to read)
      - which disciplines are supported

    Subclasses set those two attrs and inherit ``predict()``.

    Snapshot selection
    ------------------

    The harness asks for predictions at a moment in time but does not pass
    the training cutoff to the engine. We work around this by snapshotting
    the *most recent season for which a fixture exists*. In production,
    operators refresh fixtures yearly; for the in-CI test corpus this is
    deterministic.

    Athlete-id resolution
    ---------------------

    Snapshots are name-keyed; the DB is id-keyed. On first call we build a
    per-engine ``normalize_name(name) → athlete_id`` index spanning *all*
    athletes (genders combined, since rankings are gender-bucketed and we
    have no gender filter on the predict surface). If two athletes share a
    normalized name (rare — handled by the override table) the first one
    wins.
    """

    _SOURCE: Source = "ifsc_official"
    _SUPPORTED_DISCIPLINES: tuple[Discipline, ...] = (
        Discipline.BOULDER,
        Discipline.LEAD,
        Discipline.SPEED,
    )

    # Same shape as PersistenceEngine — keeps the three rank-based baselines
    # directly comparable. A 15th-place athlete maps to default μ, top
    # finishers shift up, deep finishers shift down.
    MU_PER_RANK_STEP = 12.0
    MEDIAN_RANK = 15.0

    #: Seasons to try when looking up the most recent snapshot. We descend
    #: from the current calendar year so newer fixtures are preferred.
    _SEASON_PROBE_WINDOW = 6

    def __init__(self, session: Session):
        self._session = session
        # discipline → {athlete_id: (rank, n_events_proxy)}
        self._cache: dict[Discipline, dict[int, tuple[int, int]]] = {}
        self._name_index: dict[str, int] | None = None

    def name(self) -> str:  # pragma: no cover — overridden by subclasses
        raise NotImplementedError

    # ---------------- internal helpers ----------------

    def _build_name_index(self) -> dict[str, int]:
        """Cache normalised-name → athlete_id for the current session.

        Iterates :class:`Athlete` once. ~5k rows in production, ~10 ms.
        """
        index: dict[str, int] = {}
        for row in self._session.execute(select(Athlete.id, Athlete.name)).all():
            key = normalize_name(row.name)
            # Don't clobber — first match wins. The override table in the
            # scraper handles the known ambiguous cases.
            index.setdefault(key, row.id)
        return index

    def _latest_snapshot(
        self,
        discipline: Discipline,
    ) -> list[RankedAthlete]:
        """Return the most recent non-empty snapshot for both genders combined.

        Walks years descending from the current calendar year. For each
        candidate year we concatenate the M and F rankings — the harness
        does not give us a gender filter, and athletes only ever appear in
        one gender's ranking, so this is safe.
        """
        discipline_key = _DISCIPLINE_KEY.get(discipline)
        if discipline_key is None:
            return []

        from datetime import date as _date

        current_year = _date.today().year
        for season in range(current_year, current_year - self._SEASON_PROBE_WINDOW, -1):
            combined: list[RankedAthlete] = []
            for gender in ("M", "F"):
                combined.extend(
                    load_snapshot(self._SOURCE, season, discipline_key, gender)  # type: ignore[arg-type]
                )
            if combined:
                return combined
        return []

    def _build_cache(self, discipline: Discipline) -> dict[int, tuple[int, int]]:
        if discipline not in self._SUPPORTED_DISCIPLINES:
            return {}
        if self._name_index is None:
            self._name_index = self._build_name_index()

        snapshot = self._latest_snapshot(discipline)
        if not snapshot:
            log.info(
                "No %s snapshot available for discipline=%s — engine will "
                "return default μ for all athletes.",
                self._SOURCE,
                discipline.value,
            )
            return {}

        cache: dict[int, tuple[int, int]] = {}
        unmatched = 0
        for entry in snapshot:
            key = normalize_name(entry.name)
            athlete_id = self._name_index.get(key)
            if athlete_id is None:
                unmatched += 1
                continue
            # Earlier (better) rank wins if the name is duplicated across
            # genders or appears twice in a concatenated snapshot.
            existing = cache.get(athlete_id)
            if existing is None or entry.rank < existing[0]:
                # We expose the rank as the n_events_proxy for harness
                # tenure-stratification so cold-start athletes (no
                # ranking) are visibly distinct from veterans (top-N).
                cache[athlete_id] = (entry.rank, max(1, 31 - entry.rank))
        if unmatched:
            log.info(
                "%s: %d ranked athletes not matched to DB (likely names "
                "absent from our scraped corpus).",
                self._SOURCE,
                unmatched,
            )
        return cache

    # ---------------- public RatingEngine surface ----------------

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]:
        if discipline not in self._cache:
            self._cache[discipline] = self._build_cache(discipline)
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


class IFSCOfficialEngine(_RankSnapshotEngine):
    """Predict using the official IFSC season-end World Cup ranking.

    Implementation notes
    --------------------

    Snapshots are sourced from ``tests/fixtures/external_rankings/ifsc_official/``
    (recorded) and ``data/external_rankings/ifsc_official/`` (live, gitignored).
    See :mod:`climbing_elo.scraper.external_rankings` for the scrape path
    and the rationale for using Wikipedia summary tables rather than the
    IFSC's own rankings widget.

    Engine state is built lazily per-discipline on first ``predict()`` call,
    then cached for the lifetime of the engine instance (the harness
    creates one engine per backtest split).
    """

    _SOURCE: Source = "ifsc_official"
    _SUPPORTED_DISCIPLINES = (Discipline.BOULDER, Discipline.LEAD, Discipline.SPEED)

    def name(self) -> str:
        return "ifsc_official"


# ---------------------------------------------------------------------------
# AscentStatsEngine
# ---------------------------------------------------------------------------


class AscentStatsEngine(_RankSnapshotEngine):
    """Predict using AscentStats' Bayesian Bradley-Terry rating (Boulder only).

    AscentStats publishes annual top-N boulder rankings on
    ``ascentstats.com``. We scrape and cache them via
    :mod:`climbing_elo.scraper.external_rankings`.

    For Lead/Speed we return defaults — AscentStats does not publish those
    disciplines. The engine still produces a complete forecast (default μ
    for everyone), so the harness can run the full Lead/Speed pipeline
    against this variant; the resulting report will simply show
    "ascentstats" sitting at the default-prediction floor for those
    disciplines, which is the truthful outcome.
    """

    _SOURCE: Source = "ascentstats"
    _SUPPORTED_DISCIPLINES = (Discipline.BOULDER,)

    def name(self) -> str:
        return "ascentstats"


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
register_variant("ascentstats", AscentStatsEngine)
register_variant("stripped_elo", StrippedEloEngine)
