"""Backtest evaluation harness — probabilistic metrics, stratifications, variant-pluggable.

This module is the foundation for R0 (the comprehensive backtesting work described
in ``docs/RATING_SYSTEM_RESEARCH.md`` §5 R0). It replaces the legacy single-metric
``scripts/run_backtest.py``.

Design notes
------------

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

Probabilistic source:
    All probabilistic metrics consume the output of
    :func:`compute_podium_probabilities` from ``engine.projections``. That
    function is the canonical Monte Carlo source — we never reinvent
    probability generation here.

Output:
    A single JSON report + a human-readable markdown summary, written to
    ``data/backtests/<UTC-timestamp>/`` (or a caller-supplied directory).
    Reproducibility is enforced: rng seed is part of the report header, and
    two runs with the same inputs produce byte-identical JSON.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.database import DEFAULT_DB_PATH, get_session_factory, init_db
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA
from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_podium_probabilities,
)
from climbing_elo.models import (
    Discipline,
    Event,
    Rating,
    Result,
    Round,
)

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
# Metric primitives
# ---------------------------------------------------------------------------

EPSILON = 1e-12  # log-loss probability clamp


def _clip_prob(p: float) -> float:
    return min(max(p, EPSILON), 1.0 - EPSILON)


def _log_loss(y_true: list[int], y_prob: list[float]) -> float:
    """Binary log-loss with [eps, 1-eps] clipping.

    ``y_true`` is a 0/1 list, ``y_prob`` the predicted probabilities.
    Returns NaN for empty input so callers can detect missing strata.
    """
    if not y_true:
        return float("nan")
    s = 0.0
    for y, p in zip(y_true, y_prob):
        p = _clip_prob(p)
        s -= math.log(p) if y == 1 else math.log(1.0 - p)
    return s / len(y_true)


def _brier(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return float("nan")
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def _calibration_buckets(
    y_true: list[int],
    y_prob: list[float],
    n_buckets: int = 10,
) -> list[dict[str, float]]:
    """Reliability-diagram buckets.

    Returns ``n_buckets`` dicts with ``lo``, ``hi``, ``count``, ``predicted_mean``
    (mean of predicted probabilities in the bucket) and ``empirical_rate``
    (fraction of actual positives).  Empty buckets emit ``count=0`` with NaN
    rates — kept so the bucket grid is stable across runs.
    """
    bounds = [i / n_buckets for i in range(n_buckets + 1)]
    out: list[dict[str, float]] = []
    for b in range(n_buckets):
        lo, hi = bounds[b], bounds[b + 1]
        bucket_y, bucket_p = [], []
        for y, p in zip(y_true, y_prob):
            # Inclusive on the upper edge for the last bucket so p=1.0 lands.
            in_bucket = (p >= lo) and (p < hi if b < n_buckets - 1 else p <= hi)
            if in_bucket:
                bucket_y.append(y)
                bucket_p.append(p)
        count = len(bucket_y)
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "predicted_mean": (sum(bucket_p) / count if count else float("nan")),
                "empirical_rate": (sum(bucket_y) / count if count else float("nan")),
            }
        )
    return out


def _spearman(predicted: list[float], actual: list[float]) -> float:
    """Spearman rank correlation between two equal-length lists.

    NaN on n<2 or zero variance.  Average ranks are used for ties (consistent
    with scipy's default).
    """
    n = len(predicted)
    if n < 2 or len(actual) != n:
        return float("nan")

    def _rank(xs: list[float]) -> list[float]:
        # Sort by value, then assign average rank for ties.
        order = sorted(range(n), key=lambda i: xs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    r1 = _rank(predicted)
    r2 = _rank(actual)
    m1 = sum(r1) / n
    m2 = sum(r2) / n
    num = sum((r1[i] - m1) * (r2[i] - m2) for i in range(n))
    den1 = math.sqrt(sum((r - m1) ** 2 for r in r1))
    den2 = math.sqrt(sum((r - m2) ** 2 for r in r2))
    if den1 == 0 or den2 == 0:
        return float("nan")
    return num / (den1 * den2)


# ---------------------------------------------------------------------------
# Per-round prediction record
# ---------------------------------------------------------------------------


@dataclass
class RoundPrediction:
    """One scored round — the atomic unit fed to the metric aggregator."""

    event_id: int
    event_name: str
    season: int
    tier: str
    discipline: str
    round_type: str
    field_size: int
    # athlete_id → predicted probabilities + actual outcome
    athletes: list[dict[str, Any]] = field(default_factory=list)
    # Predicted vs actual rank ordering (for Spearman).
    predicted_mu: list[float] = field(default_factory=list)
    actual_rank: list[int] = field(default_factory=list)
    # Predicted top-K identifiers (for hit-rate).
    predicted_top1: list[int] = field(default_factory=list)
    predicted_top3: list[int] = field(default_factory=list)
    predicted_top8: list[int] = field(default_factory=list)
    actual_top1: list[int] = field(default_factory=list)
    actual_top3: list[int] = field(default_factory=list)
    actual_top8: list[int] = field(default_factory=list)


# Tenure buckets (per the R0 spec §5).
TENURE_BUCKETS: list[tuple[str, int, int]] = [
    ("1-3", 1, 3),
    ("4-10", 4, 10),
    ("11-30", 11, 30),
    ("30+", 31, 10_000),
]


def _tenure_bucket(n_events: int) -> str:
    for label, lo, hi in TENURE_BUCKETS:
        if lo <= n_events <= hi:
            return label
    return "0"  # athlete has zero prior events (cold-start)


# Field-size buckets — chosen to cluster around the typical IFSC formats
# (final = 8, semi = 20–26, qual = 40–80).
FIELD_SIZE_BUCKETS: list[tuple[str, int, int]] = [
    ("<=8", 0, 8),
    ("9-20", 9, 20),
    ("21-40", 21, 40),
    ("41+", 41, 10_000),
]


def _field_size_bucket(n: int) -> str:
    for label, lo, hi in FIELD_SIZE_BUCKETS:
        if lo <= n <= hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _aggregate_metrics(predictions: list[RoundPrediction]) -> dict[str, Any]:
    """Compute the metric matrix across a list of scored rounds."""
    win_true, win_p = [], []
    podium_true, podium_p = [], []
    top8_true, top8_p = [], []
    # Spearman per round, then averaged.
    spearmans: list[float] = []
    top1_hits = top3_hits = top8_hits = 0
    n_rounds = len(predictions)

    for rp in predictions:
        for ath in rp.athletes:
            win_true.append(ath["actual_win"])
            win_p.append(ath["p_win"])
            podium_true.append(ath["actual_podium"])
            podium_p.append(ath["p_podium"])
            top8_true.append(ath["actual_top8"])
            top8_p.append(ath["p_top8"])
        rho = _spearman(rp.predicted_mu, [-r for r in rp.actual_rank])
        if not math.isnan(rho):
            spearmans.append(rho)

        if rp.actual_top1 and rp.predicted_top1[:1] == rp.actual_top1[:1]:
            top1_hits += 1
        if rp.actual_top3 and set(rp.predicted_top3) & set(rp.actual_top3):
            top3_hits += 1
        if rp.actual_top8 and set(rp.predicted_top8) & set(rp.actual_top8):
            top8_hits += 1

    return {
        "n_rounds": n_rounds,
        "n_athlete_rounds": len(win_true),
        "log_loss_win": _log_loss(win_true, win_p),
        "log_loss_podium": _log_loss(podium_true, podium_p),
        "log_loss_top8": _log_loss(top8_true, top8_p),
        "brier_win": _brier(win_true, win_p),
        "brier_podium": _brier(podium_true, podium_p),
        "brier_top8": _brier(top8_true, top8_p),
        "mean_spearman": (statistics.fmean(spearmans) if spearmans else float("nan")),
        "hit_rate_top1": top1_hits / n_rounds if n_rounds else float("nan"),
        "hit_rate_top3": top3_hits / n_rounds if n_rounds else float("nan"),
        "hit_rate_top8": top8_hits / n_rounds if n_rounds else float("nan"),
        "calibration_win": _calibration_buckets(win_true, win_p),
        "calibration_podium": _calibration_buckets(podium_true, podium_p),
        "calibration_top8": _calibration_buckets(top8_true, top8_p),
    }


def _stratify(
    predictions: list[RoundPrediction],
    key: Callable[[RoundPrediction], str],
) -> dict[str, dict[str, Any]]:
    """Group predictions by ``key(prediction)`` and compute per-group metrics."""
    groups: dict[str, list[RoundPrediction]] = {}
    for rp in predictions:
        groups.setdefault(key(rp), []).append(rp)
    return {
        # Sort keys alphabetically for reproducible JSON output.
        k: _aggregate_metrics(groups[k])
        for k in sorted(groups.keys())
    }


def _stratify_athlete_rounds(
    predictions: list[RoundPrediction],
    key_for_athlete: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    """Group athlete-level predictions for stratifications keyed off the
    athlete (e.g. tenure buckets — the bucket depends on the athlete's
    n_events, not on the round)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rp in predictions:
        for ath in rp.athletes:
            groups.setdefault(key_for_athlete(ath), []).append(ath)

    out: dict[str, dict[str, Any]] = {}
    for k in sorted(groups.keys()):
        ath_list = groups[k]
        win_true = [a["actual_win"] for a in ath_list]
        win_p = [a["p_win"] for a in ath_list]
        pod_true = [a["actual_podium"] for a in ath_list]
        pod_p = [a["p_podium"] for a in ath_list]
        top8_true = [a["actual_top8"] for a in ath_list]
        top8_p = [a["p_top8"] for a in ath_list]
        out[k] = {
            "n_athlete_rounds": len(ath_list),
            "log_loss_win": _log_loss(win_true, win_p),
            "log_loss_podium": _log_loss(pod_true, pod_p),
            "log_loss_top8": _log_loss(top8_true, top8_p),
            "brier_win": _brier(win_true, win_p),
            "brier_podium": _brier(pod_true, pod_p),
            "brier_top8": _brier(top8_true, top8_p),
            "calibration_podium": _calibration_buckets(pod_true, pod_p),
        }
    return out


# ---------------------------------------------------------------------------
# Backtest dataset + runner
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
    # Source DB to copy from. Defaults to the production DB.
    source_db_path: Path = DEFAULT_DB_PATH


@dataclass
class BacktestReport:
    """Top-level report — serialised to JSON + markdown."""

    generated_at: str
    variant: str
    oos_mode: str
    rng_seed: int
    n_simulations: int
    disciplines: list[str]
    splits: list[dict[str, Any]] = field(default_factory=list)
    # Aggregate (across all splits + disciplines).
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise with sorted keys + fixed float repr for byte-stability."""
        return json.dumps(asdict(self), sort_keys=True, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Custom JSON encoder hook.

    NaNs are rendered as the literal string ``"NaN"`` so the JSON loads in
    any standards-compliant parser (Python's allow-nan is non-portable).
    Stable across runs.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Unserialisable {type(obj).__name__}")


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

        report.aggregate = self._aggregate_with_strata(all_predictions)

        if self.output_dir is not None:
            self._write_outputs(report)

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
            run_backfill(session, discipline, end_date=split.train_end_date)
            session.commit()
            engine = self.engine_factory(session)
            return self._score_split_events(session, engine, split, discipline)

        assert self._pristine_copy is not None and self._working_copy is not None
        shutil.copy(self._pristine_copy, self._working_copy)
        factory = get_session_factory(self._working_copy)
        with factory() as session:
            run_backfill(session, discipline, end_date=split.train_end_date)
            session.commit()
            engine = self.engine_factory(session)
            return self._score_split_events(session, engine, split, discipline)

    def _score_split_events(
        self,
        session: Session,
        engine: RatingEngine,
        split: TrainEvalSplit,
        discipline: Discipline,
    ) -> list[RoundPrediction]:
        predictions: list[RoundPrediction] = []
        for eid in split.eval_event_ids:
            event = session.get(Event, eid)
            if event is None or event.discipline != discipline:
                continue
            # Sort rounds for stable output ordering.
            rounds_sorted = sorted(event.rounds, key=lambda r: r.round_type.value)
            for rnd in rounds_sorted:
                rp = self._score_round(session, engine, event, rnd, discipline)
                if rp is not None:
                    predictions.append(rp)
        return predictions

    def _score_round(
        self,
        session: Session,
        engine: RatingEngine,
        event: Event,
        rnd: Round,
        discipline: Discipline,
    ) -> RoundPrediction | None:
        results = list(
            session.execute(
                select(Result)
                .where(Result.round_id == rnd.id, ~Result.dns)
                .order_by(Result.rank.asc().nulls_last())
            ).scalars()
        )
        # Filter results without a finishing rank — we can't score them.
        results = [r for r in results if r.rank is not None]
        if len(results) < 3:
            return None

        forecasts = engine.predict(
            (r.athlete_id for r in results),
            discipline,
        )

        inputs = [
            AthleteProjectionInput(
                athlete_id=r.athlete_id,
                mu=forecasts[r.athlete_id].mu,
                sigma=forecasts[r.athlete_id].sigma,
            )
            for r in results
        ]
        # Per-athlete probability matrix from the canonical MC source.
        # The rng_seed is parameterised so two runs produce identical numbers.
        probs = compute_podium_probabilities(
            inputs,
            n_simulations=self.dataset.n_simulations,
            rng_seed=self.dataset.rng_seed,
        )

        # Actual outcome.
        results_sorted = sorted(results, key=lambda r: r.rank)
        actual_top1 = {results_sorted[0].athlete_id}
        actual_top3 = {r.athlete_id for r in results_sorted[:3]}
        actual_top8 = {
            r.athlete_id for r in results_sorted[: min(8, len(results_sorted))]
        }

        rp = RoundPrediction(
            event_id=event.id,
            event_name=event.name,
            season=event.season,
            tier=event.tier.value,
            discipline=discipline.value,
            round_type=rnd.round_type.value,
            field_size=len(results),
        )
        for r in results:
            f = forecasts[r.athlete_id]
            p = probs[r.athlete_id]
            rp.athletes.append(
                {
                    "athlete_id": r.athlete_id,
                    "mu": f.mu,
                    "sigma": f.sigma,
                    "n_events": f.n_events,
                    "tenure_bucket": _tenure_bucket(f.n_events),
                    "p_win": p["win"],
                    "p_podium": p["podium"],
                    "p_top8": p["top_8"],
                    "actual_rank": r.rank,
                    "actual_win": int(r.athlete_id in actual_top1),
                    "actual_podium": int(r.athlete_id in actual_top3),
                    "actual_top8": int(r.athlete_id in actual_top8),
                }
            )

        # Sort athletes by descending mu — predicted ordering.
        by_mu = sorted(results, key=lambda r: forecasts[r.athlete_id].mu, reverse=True)
        rp.predicted_top1 = [by_mu[0].athlete_id]
        rp.predicted_top3 = [r.athlete_id for r in by_mu[:3]]
        rp.predicted_top8 = [r.athlete_id for r in by_mu[: min(8, len(by_mu))]]
        rp.actual_top1 = [results_sorted[0].athlete_id]
        rp.actual_top3 = [r.athlete_id for r in results_sorted[:3]]
        rp.actual_top8 = [
            r.athlete_id for r in results_sorted[: min(8, len(results_sorted))]
        ]
        rp.predicted_mu = [forecasts[r.athlete_id].mu for r in results]
        rp.actual_rank = [r.rank for r in results]
        return rp

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


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def render_markdown(report: BacktestReport) -> str:
    """Human-readable markdown rendering of a :class:`BacktestReport`."""
    lines: list[str] = []
    lines.append("# Backtest report")
    lines.append("")
    lines.append(f"- Generated: {report.generated_at}")
    lines.append(f"- Variant: `{report.variant}`")
    lines.append(f"- OOS mode: `{report.oos_mode}`")
    lines.append(f"- Disciplines: {', '.join(report.disciplines)}")
    lines.append(f"- RNG seed: {report.rng_seed}")
    lines.append(f"- MC simulations per round: {report.n_simulations}")
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    agg = report.aggregate
    lines.append(f"- Rounds scored: {agg.get('n_rounds', 0)}")
    lines.append(f"- Athlete-rounds: {agg.get('n_athlete_rounds', 0)}")
    lines.append("")
    lines.append("| Metric | Win | Podium | Top-8 |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| Log-loss | "
        f"{_fmt(agg.get('log_loss_win'))} | "
        f"{_fmt(agg.get('log_loss_podium'))} | "
        f"{_fmt(agg.get('log_loss_top8'))} |"
    )
    lines.append(
        "| Brier | "
        f"{_fmt(agg.get('brier_win'))} | "
        f"{_fmt(agg.get('brier_podium'))} | "
        f"{_fmt(agg.get('brier_top8'))} |"
    )
    lines.append("")
    lines.append(f"- Mean Spearman ρ: {_fmt(agg.get('mean_spearman'))}")
    lines.append(f"- Top-1 hit rate: {_fmt(agg.get('hit_rate_top1'))}")
    lines.append(f"- Top-3 hit rate: {_fmt(agg.get('hit_rate_top3'))}")
    lines.append(f"- Top-8 hit rate: {_fmt(agg.get('hit_rate_top8'))}")
    lines.append("")

    lines.append("## Splits")
    lines.append("")
    for s in report.splits:
        m = s["metrics"]
        lines.append(
            f"- **{s['discipline']} / {s['label']}** "
            f"(train < {s['train_end_date']}, "
            f"n_events={s['n_eval_events']}): "
            f"log-loss podium={_fmt(m.get('log_loss_podium'))}, "
            f"top-3 hit={_fmt(m.get('hit_rate_top3'))}"
        )
    lines.append("")

    strata = agg.get("stratifications", {})

    def _table(title: str, key: str, group_label: str) -> None:
        s = strata.get(key, {})
        if not s:
            return
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"| {group_label} | n | LL win | LL pod | LL top-8 | Brier pod |")
        lines.append("|---|---|---|---|---|---|")
        for k in sorted(s.keys()):
            row = s[k]
            lines.append(
                f"| {k} | {row.get('n_athlete_rounds', row.get('n_rounds', 0))} | "
                f"{_fmt(row.get('log_loss_win'))} | "
                f"{_fmt(row.get('log_loss_podium'))} | "
                f"{_fmt(row.get('log_loss_top8'))} | "
                f"{_fmt(row.get('brier_podium'))} |"
            )
        lines.append("")

    lines.append("## Stratifications")
    lines.append("")
    _table("By tier", "by_tier", "Tier")
    _table("By round", "by_round", "Round")
    _table("By discipline", "by_discipline", "Discipline")
    _table("By season", "by_season", "Season")
    _table("By field size", "by_field_size", "Field size")
    _table("By tenure", "by_tenure", "Tenure (n_events)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def make_default_output_dir(root: Path | None = None) -> Path:
    """Return ``data/backtests/<utc-timestamp>/`` (creating parents)."""
    root = root or (DEFAULT_DB_PATH.parent / "backtests")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out
