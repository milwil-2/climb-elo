"""Scoring primitives — probabilistic metrics, stratifications, and the per-round
prediction record consumed by the backtest runner.

The metric primitives (:func:`_log_loss`, :func:`_brier`, :func:`_calibration_buckets`,
:func:`_spearman`) and aggregation helpers (:func:`_aggregate_metrics`,
:func:`_stratify`, :func:`_stratify_athlete_rounds`) are pure functions with no
DB dependency.

The per-round scoring helpers (:func:`score_round`, :func:`score_split_events`)
DO touch the DB: given a SQLAlchemy session, a rating engine, and an event /
round, they build a :class:`RoundPrediction` from the actual results + the
engine's forecast. They live here (rather than on :class:`BacktestRunner`) so
the runner module stays focused on lifecycle / orchestration concerns.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine.projections import (
    AthleteProjectionInput,
    compute_podium_probabilities,
)
from climbing_elo.models import Discipline, Event, Result, Round

if TYPE_CHECKING:
    from .runner import RatingEngine, TrainEvalSplit

EPSILON = 1e-12  # log-loss probability clamp


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


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
# Per-round / per-split scoring (DB-backed)
# ---------------------------------------------------------------------------


def score_round(
    session: Session,
    engine: "RatingEngine",
    event: Event,
    rnd: Round,
    discipline: Discipline,
    *,
    n_simulations: int,
    rng_seed: int,
) -> RoundPrediction | None:
    """Score one round — build a :class:`RoundPrediction` from DB results.

    Returns ``None`` if the round has fewer than 3 finishers (no useful signal
    for podium / Spearman metrics).
    """
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
        n_simulations=n_simulations,
        rng_seed=rng_seed,
    )

    # Actual outcome.
    results_sorted = sorted(results, key=lambda r: r.rank)
    actual_top1 = {results_sorted[0].athlete_id}
    actual_top3 = {r.athlete_id for r in results_sorted[:3]}
    actual_top8 = {r.athlete_id for r in results_sorted[: min(8, len(results_sorted))]}

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


def score_split_events(
    session: Session,
    engine: "RatingEngine",
    split: "TrainEvalSplit",
    discipline: Discipline,
    *,
    n_simulations: int,
    rng_seed: int,
) -> list[RoundPrediction]:
    """Score every (event, round) in the split's eval set."""
    predictions: list[RoundPrediction] = []
    for eid in split.eval_event_ids:
        event = session.get(Event, eid)
        if event is None or event.discipline != discipline:
            continue
        # Sort rounds for stable output ordering.
        rounds_sorted = sorted(event.rounds, key=lambda r: r.round_type.value)
        for rnd in rounds_sorted:
            rp = score_round(
                session,
                engine,
                event,
                rnd,
                discipline,
                n_simulations=n_simulations,
                rng_seed=rng_seed,
            )
            if rp is not None:
                predictions.append(rp)
    return predictions
