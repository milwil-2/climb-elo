"""Backtest orchestration — engine variants, OOS modes, dataset, and the runner loop.

State safety (CRITICAL):
    The legacy ``scripts/run_backtest.py`` issued
    ``session.execute(delete(RatingHistory))`` / ``delete(Rating)`` against the
    production DB — corrupting the live ratings table. **This module must never
    touch the production DB.** All backfills run against a *copy* of the source
    DB (file-based, located in a temp directory). The copy is removed when the
    runner exits. The production session/factory is read-only from this
    module's perspective.

Variant pluggability:
    The :class:`RatingEngine` protocol lets downstream issues (#38 — baseline
    engines, #39 — out-of-sample modes) register alternative rating
    implementations against the same harness. Variants register via
    :func:`register_variant` and are looked up by name through the
    ``BACKTEST_VARIANTS`` registry. The default variant ``"current"`` wraps the
    existing production ELO engine in :class:`CurrentEloEngine`.

Out-of-sample modes:
    The :class:`OOSMode` protocol describes the contract for OOS strategies
    (single-cutoff holdout, walk-forward, leave-one-out, …). #39 plugs new
    modes in by registering them in ``OOS_MODES``. This module ships the
    default ``"holdout"`` mode (last N seasons held out — matches the legacy
    behaviour).
"""

from __future__ import annotations

import inspect
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.database import get_session_factory, init_db
from climbing_elo.engine.backfill import force_reset_for_discipline, run_backfill
from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA
from climbing_elo.models import (
    Discipline,
    Event,
    Rating,
)

from .metrics import (
    RoundPrediction,
    _aggregate_metrics,
    _field_size_bucket,
    _stratify,
    _stratify_athlete_rounds,
    score_split_events,
)
from .reports import BacktestReport, render_markdown

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public protocols / dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RatingForecast:
    """Per-athlete probabilistic forecast issued by a :class:`RatingEngine`.

    ``mu`` and ``sigma`` are the canonical N(μ, σ) parameters consumed by the
    Monte Carlo projector. ``n_events`` lets downstream metrics stratify by
    athlete tenure (cold-start diagnostics).
    """

    athlete_id: int
    mu: float
    sigma: float
    n_events: int = 0


@runtime_checkable
class RatingEngine(Protocol):
    """Variant-pluggable rating engine.

    A backtest variant is *any* object that can be asked, before evaluating a
    held-out round, "given these athletes and this discipline, what is your
    probabilistic forecast?"

    Implementations live in this module (the default current-engine wrapper)
    and in downstream issues — #38 adds baseline engines (random, persistence,
    n_events, stripped-down Elo). The protocol intentionally does not mention
    *training*: the harness handles dataset preparation and asks the engine
    for forecasts at predict-time only. How the engine got its state — DB
    rows, cached object, fitted model — is its own business.

    Methods
    -------
    name() -> str
        Stable identifier used in report headers and CLI flags.
    predict(athletes_in_round, discipline) -> dict[int, RatingForecast]
        Return a forecast for each athlete competing in the round. The keys
        of the returned dict must be a superset of ``athletes_in_round``;
        missing athletes are treated as default-rated by the harness.
    """

    def name(self) -> str: ...  # pragma: no cover

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Default variant: the current production engine
# ---------------------------------------------------------------------------


class CurrentEloEngine:
    """Default variant — reads ratings from the (training-DB) ``Rating`` table.

    Usage pattern: the harness runs backfill on training events into an
    isolated DB copy, then constructs this engine with a session pointing at
    the *training-end* state. Subsequent ``predict()`` calls return the frozen
    μ/σ at training-cutoff.
    """

    def __init__(self, session: Session):
        self._session = session
        # Cache the snapshot once — backfill state is frozen for this engine.
        self._snapshot: dict[Discipline, dict[int, RatingForecast]] = {}

    def name(self) -> str:
        return "current"

    def _snapshot_for(self, discipline: Discipline) -> dict[int, RatingForecast]:
        if discipline in self._snapshot:
            return self._snapshot[discipline]

        snap: dict[int, RatingForecast] = {}
        for r in self._session.execute(
            select(Rating).where(Rating.discipline == discipline)
        ).scalars():
            snap[r.athlete_id] = RatingForecast(
                athlete_id=r.athlete_id,
                mu=r.mu,
                sigma=r.sigma,
                n_events=r.n_events,
            )
        self._snapshot[discipline] = snap
        return snap

    def predict(
        self,
        athletes_in_round: Iterable[int],
        discipline: Discipline,
    ) -> dict[int, RatingForecast]:
        snap = self._snapshot_for(discipline)
        out: dict[int, RatingForecast] = {}
        for aid in athletes_in_round:
            if aid in snap:
                out[aid] = snap[aid]
            else:
                out[aid] = RatingForecast(
                    athlete_id=aid,
                    mu=DEFAULT_MU,
                    sigma=DEFAULT_SIGMA,
                    n_events=0,
                )
        return out


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


# Factory signature: given a session pointing at the trained DB, return a
# concrete engine.  Downstream baselines (Issue #38) register here.
EngineFactory = Callable[[Session], RatingEngine]

BACKTEST_VARIANTS: dict[str, EngineFactory] = {
    "current": CurrentEloEngine,
}


def register_variant(name: str, factory: EngineFactory) -> None:
    """Register an additional engine variant.

    Downstream issues call this at import-time (e.g. in
    ``engine/baselines.py`` for #38).  The ``"current"`` slot is the default
    and should not be overwritten by downstream variants.
    """
    BACKTEST_VARIANTS[name] = factory


# ---------------------------------------------------------------------------
# Out-of-sample mode contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainEvalSplit:
    """One concrete train/eval split produced by an OOS mode.

    ``label`` identifies the split in the report (e.g. ``"holdout-2024"`` or
    ``"walk-forward-fold-3"``).  ``train_end_date`` is passed to
    :func:`run_backfill` as ``end_date=`` — only events strictly before this
    date are used for training.  ``eval_event_ids`` enumerates the rounds we
    score predictions against.
    """

    label: str
    train_end_date: date
    eval_event_ids: tuple[int, ...]


@runtime_checkable
class OOSMode(Protocol):
    """Strategy for splitting the dataset into train/eval folds.

    Issue #39 will add walk-forward and leave-one-out implementations against
    this contract.
    """

    def name(self) -> str: ...  # pragma: no cover

    def splits(
        self,
        session: Session,
        discipline: Discipline,
    ) -> list[TrainEvalSplit]: ...  # pragma: no cover


@dataclass
class HoldoutMode:
    """Default OOS mode: hold out the most recent ``n_seasons`` of events.

    Reproduces the legacy ``scripts/run_backtest.py`` behaviour exactly,
    modulo metric scope.
    """

    n_seasons: int = 2

    def name(self) -> str:
        return f"holdout-{self.n_seasons}s"

    def splits(
        self,
        session: Session,
        discipline: Discipline,
    ) -> list[TrainEvalSplit]:
        from sqlalchemy import func

        max_season = session.execute(
            select(func.max(Event.season)).where(Event.discipline == discipline)
        ).scalar()
        if max_season is None:
            return []

        cutoff_season = max_season - self.n_seasons + 1
        cutoff_date = date(cutoff_season, 1, 1)

        holdout_event_ids = tuple(
            session.execute(
                select(Event.id)
                .where(
                    Event.discipline == discipline,
                    Event.start_date >= cutoff_date,
                )
                .order_by(Event.start_date.asc())
            ).scalars()
        )
        return [
            TrainEvalSplit(
                label=f"holdout-{cutoff_season}-{max_season}",
                train_end_date=cutoff_date,
                eval_event_ids=holdout_event_ids,
            )
        ]


OOS_MODES: dict[str, Callable[..., OOSMode]] = {
    "holdout": HoldoutMode,
}


def register_oos_mode(name: str, factory: Callable[..., OOSMode]) -> None:
    """Register an additional OOS mode (Issue #39 plug-point)."""
    OOS_MODES[name] = factory


# ---------------------------------------------------------------------------
# Backtest dataset + report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestDataset:
    """Inputs that fully determine a backtest run.

    Kept tiny on purpose — anything that affects the metric output must be a
    field here so the report header captures it and reproducibility holds.
    """

    disciplines: tuple[Discipline, ...]
    n_simulations: int = 10_000
    rng_seed: int = 42
    # Source SQLite DB to copy from. ``None`` means the runner is being driven
    # via ``in_memory_session=`` (tests) and no file copy is needed. Production
    # invocations (``scripts/run_backtest.py``) must pass an explicit path —
    # the harness only works against a SQLite source it can ``shutil.copy``.
    source_db_path: Path | None = None


def _build_engine(
    factory: EngineFactory,
    session: Session,
    cutoff_date: date,
) -> "RatingEngine":
    """Construct an engine, passing ``cutoff_date`` when the factory supports it.

    Snapshot-based engines (:class:`~climbing_elo.engine.baselines.IFSCOfficialEngine`
    and :class:`~climbing_elo.engine.baselines.AscentStatsEngine`) accept a
    ``cutoff_date`` keyword argument so they can restrict snapshot selection to
    seasons that pre-date the training window, preventing data leakage.  Other
    engines (e.g. ``current``, ``persistence``) only accept a ``session``
    positional argument; we fall back gracefully for those.
    """
    sig = inspect.signature(factory)
    if "cutoff_date" in sig.parameters:
        return factory(session, cutoff_date=cutoff_date)  # type: ignore[call-arg]
    return factory(session)


class BacktestRunner:
    """Coordinator — wires together dataset, engine variant, OOS mode, metrics.

    Lifecycle:

    1. ``__init__`` — copy the source DB to a private temp file (state safety).
    2. ``run()`` — for each split:
         a. Restore the DB copy to its pristine state (cheap: shutil.copy).
         b. Run backfill on training events only (``end_date=split.train_end``).
         c. Construct the engine variant against the training-end DB.
         d. Score every event in ``split.eval_event_ids`` round-by-round.
       Then aggregate across splits, render JSON + markdown.
    3. ``close()`` (also via context manager) — remove the temp directory.

    The harness never opens a write session against the source DB. The temp
    DB copy is the only mutated artefact.
    """

    def __init__(
        self,
        dataset: BacktestDataset,
        variant: str = "current",
        oos_mode: OOSMode | None = None,
        output_dir: Path | None = None,
        in_memory_session: Session | None = None,
    ):
        """Construct a runner.

        Args:
            dataset: Backtest inputs (disciplines, MC count, seed, source DB).
            variant: Engine variant name (registered in ``BACKTEST_VARIANTS``).
            oos_mode: Out-of-sample mode (default: :class:`HoldoutMode`).
            output_dir: If set, JSON + markdown are written here on
                :meth:`run`. ``None`` disables file output (tests).
            in_memory_session: Test-only escape hatch. When provided, the
                runner uses this session directly instead of copying the
                source DB. The session is mutated by backfill, so callers are
                expected to roll back any other state they care about.
                Production callers should leave this ``None``.
        """
        if variant not in BACKTEST_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {sorted(BACKTEST_VARIANTS)}"
            )
        self.dataset = dataset
        self.variant_name = variant
        self.engine_factory = BACKTEST_VARIANTS[variant]
        self.oos_mode: OOSMode = oos_mode if oos_mode is not None else HoldoutMode()
        self.output_dir = output_dir
        self._in_memory_session = in_memory_session

        if in_memory_session is not None:
            # No DB-file copying needed; tests own the lifecycle.
            self._tmpdir = None
            self._pristine_copy = None
            self._working_copy = None
            return

        # --- State safety: copy the source DB to a private temp file. ---
        if dataset.source_db_path is None:
            raise ValueError(
                "BacktestDataset.source_db_path must be set when running the "
                "harness without in_memory_session. The backtest copies the "
                "source DB into a temp file and only supports SQLite sources."
            )
        self._tmpdir = Path(tempfile.mkdtemp(prefix="climbing_elo_backtest_"))
        self._pristine_copy = self._tmpdir / "pristine.db"
        self._working_copy = self._tmpdir / "working.db"
        if dataset.source_db_path.exists():
            shutil.copy(dataset.source_db_path, self._pristine_copy)
        else:
            # Source missing — initialise an empty schema'd DB so downstream
            # code doesn't crash on an empty file.
            init_db(self._pristine_copy)

    # -- Context manager support ----------------------------------------------

    def __enter__(self) -> "BacktestRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._tmpdir is not None and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- Public entry point ---------------------------------------------------

    def run(self) -> BacktestReport:
        """Run the full backtest matrix over the configured dataset.

        Output is always deterministic given the inputs: ``rng_seed`` flows
        into the Monte Carlo projector, sorted-key JSON serialisation
        eliminates dict-ordering nondeterminism, and the splits are processed
        in a stable order (disciplines as supplied, splits in the order the
        OOS mode returned them).

        The ``generated_at`` timestamp is the *only* non-deterministic part of
        the report; reproducibility tests should ignore that field.
        """
        report = BacktestReport(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            variant=self.variant_name,
            oos_mode=self.oos_mode.name(),
            rng_seed=self.dataset.rng_seed,
            n_simulations=self.dataset.n_simulations,
            disciplines=[d.value for d in self.dataset.disciplines],
        )

        all_predictions: list[RoundPrediction] = []
        # Per-discipline cache of predictions for auxiliary artifacts
        # (e.g. the leave-one-athlete-out convergence trace).
        per_discipline_predictions: dict[Discipline, list[RoundPrediction]] = {}

        for discipline in self.dataset.disciplines:
            splits = self._discover_splits(discipline)
            for split in splits:
                split_predictions = self._run_split(discipline, split)
                report.splits.append(
                    {
                        "discipline": discipline.value,
                        "label": split.label,
                        "train_end_date": split.train_end_date.isoformat(),
                        "n_eval_events": len(split.eval_event_ids),
                        "metrics": _aggregate_metrics(split_predictions),
                    }
                )
                all_predictions.extend(split_predictions)
                per_discipline_predictions.setdefault(discipline, []).extend(
                    split_predictions
                )

        report.aggregate = self._aggregate_with_strata(all_predictions)

        if self.output_dir is not None:
            self._write_outputs(report)
            self._maybe_write_auxiliary(per_discipline_predictions)

        return report

    def _discover_splits(self, discipline: Discipline) -> list[TrainEvalSplit]:
        """Ask the OOS mode for splits, using a read-only view of the source."""
        if self._in_memory_session is not None:
            return self.oos_mode.splits(self._in_memory_session, discipline)
        assert self._pristine_copy is not None and self._working_copy is not None
        shutil.copy(self._pristine_copy, self._working_copy)
        factory = get_session_factory(self._working_copy)
        with factory() as session:
            return self.oos_mode.splits(session, discipline)

    # -- Internals -----------------------------------------------------------

    def _run_split(
        self,
        discipline: Discipline,
        split: TrainEvalSplit,
    ) -> list[RoundPrediction]:
        """Train on events before cutoff, score every eval event.

        Two modes:

        - **DB-copy** (production): restore working DB from pristine, open a
          new session, backfill on training events, then score.
        - **In-memory** (tests): use the supplied session directly. The
          session is mutated by backfill — tests that need cross-call
          isolation must snapshot/restore themselves.
        """
        if self._in_memory_session is not None:
            session = self._in_memory_session
            # #192: the backfill is idempotent, so a pre-backfilled source
            # would otherwise keep its end-state ratings — future leakage.
            force_reset_for_discipline(session, discipline)
            run_backfill(session, discipline, end_date=split.train_end_date)
            session.commit()
            engine = _build_engine(self.engine_factory, session, split.train_end_date)
            return score_split_events(
                session,
                engine,
                split,
                discipline,
                n_simulations=self.dataset.n_simulations,
                rng_seed=self.dataset.rng_seed,
            )

        assert self._pristine_copy is not None and self._working_copy is not None
        shutil.copy(self._pristine_copy, self._working_copy)
        factory = get_session_factory(self._working_copy)
        with factory() as session:
            # #192: see the in-memory branch — reset computed ratings so the
            # training-end state is recomputed from raw results, not carried
            # over from a pre-backfilled source DB.
            force_reset_for_discipline(session, discipline)
            run_backfill(session, discipline, end_date=split.train_end_date)
            session.commit()
            engine = _build_engine(self.engine_factory, session, split.train_end_date)
            return score_split_events(
                session,
                engine,
                split,
                discipline,
                n_simulations=self.dataset.n_simulations,
                rng_seed=self.dataset.rng_seed,
            )

    def _aggregate_with_strata(
        self, predictions: list[RoundPrediction]
    ) -> dict[str, Any]:
        agg = _aggregate_metrics(predictions)
        agg["stratifications"] = {
            "by_tier": _stratify(predictions, lambda rp: rp.tier),
            "by_round": _stratify(predictions, lambda rp: rp.round_type),
            "by_discipline": _stratify(predictions, lambda rp: rp.discipline),
            "by_season": _stratify(predictions, lambda rp: str(rp.season)),
            "by_field_size": _stratify(
                predictions, lambda rp: _field_size_bucket(rp.field_size)
            ),
            "by_tenure": _stratify_athlete_rounds(
                predictions, lambda ath: ath["tenure_bucket"]
            ),
        }
        return agg

    def _write_outputs(self, report: BacktestReport) -> None:
        out = self.output_dir
        assert out is not None
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(report.to_json() + "\n")
        (out / "report.md").write_text(render_markdown(report) + "\n")

    def _maybe_write_auxiliary(
        self,
        per_discipline_predictions: dict[Discipline, list[RoundPrediction]],
    ) -> None:
        """Emit auxiliary artifacts that depend on the active OOS mode.

        Currently handles the leave-one-athlete-out convergence trace
        (Issue #39) — written when the OOS mode advertises an
        ``emit_convergence_trace`` flag. Other modes are silent here.
        """
        if not getattr(self.oos_mode, "emit_convergence_trace", False):
            return
        # Local import to avoid a circular dependency at module load time.
        from climbing_elo.engine.oos_modes import (
            LeaveOneAthleteOutMode,
            build_convergence_trace,
            write_convergence_trace,
        )

        if not isinstance(self.oos_mode, LeaveOneAthleteOutMode):
            return

        # The athlete is single-discipline by construction (the mode took a
        # single athlete_id). Pick the first discipline that actually
        # produced predictions for this athlete.
        for discipline, preds in per_discipline_predictions.items():
            if not preds:
                continue
            session_cm = self._auxiliary_session()
            with session_cm as session:
                trace = build_convergence_trace(
                    session, self.oos_mode, discipline, preds
                )
            assert self.output_dir is not None
            if trace["trace"]:
                write_convergence_trace(self.output_dir, trace)
                # Only one discipline can match a single athlete cleanly —
                # bail after the first non-empty trace.
                return

    def _auxiliary_session(self):
        """Return a context manager that yields a Session for read-only work.

        Re-uses the working DB copy (it's already in training-end state) for
        production runs, or the supplied in-memory session for tests.
        """
        if self._in_memory_session is not None:

            class _Passthrough:
                def __init__(self, sess: Session):
                    self._sess = sess

                def __enter__(self) -> Session:
                    return self._sess

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

            return _Passthrough(self._in_memory_session)

        assert self._working_copy is not None
        factory = get_session_factory(self._working_copy)
        return factory()
